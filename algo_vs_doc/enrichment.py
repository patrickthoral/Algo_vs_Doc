"""Build the enriched per-stay dataset from CDM-lite tables.

Proprietary schema mappings are resolved before this module. Reviewable non-proprietary source
values are mapped to the shared vocabulary before enrichment. ``STAY_COLUMNS`` defines the shared
input contract.
"""

from __future__ import annotations

import pandas as pd
from pandas.api import types as ptypes

from algo_vs_doc import discharge_locations, operational_pressure
from algo_vs_doc.scoring import cdm, cohort_outcomes, life_support, pipeline
from algo_vs_doc.scoring.concepts import (
    CONDITION_CONCEPTS,
    DEVICE_CONCEPTS,
    VASOPRESSOR_DRUG_CONCEPTS,
)

#: Required CDM-lite tables.
CDM_TABLES = (
    "icu_stays",
    "measurement",
    "drug_exposure",
    "device_exposure",
    "condition_occurrence",
    "diagnosis_occurrence",
    "prediction",
)

#: Stay durations represented as numeric hours at the CDM-lite boundary.
STAY_DURATION_HOURS_COLUMNS = (
    "icu_los",
    "time_to_death",
    "time_to_icu_readmission",
)

#: Required hospital-agnostic columns supplied by private ETL.
STAY_COLUMNS = (
    # identity, cohort membership, and timing
    "icu_stay_id",
    "patient_id",
    "is_study_cohort",
    "icu_admitted_at",
    "icu_discharged_at",
    "icu_los",
    "icu_los_lt_12h",
    # who the patient was and under whose care
    "sex",
    "age_at_admission",
    "specialty_id",
    # optional hospital-local workload classification; missing where not recorded
    "patient_category_id",
    "patient_category_name",
    # hospital-local, non-proprietary receiving-location values for public mapping
    "discharge_location_id",
    "discharge_location_name",
    # team eligibility is observed only when an assessment workflow exists
    "ready_for_discharge",
    # origin of the first ICU readmission; either location or specialty may be unavailable
    "readmission_origin_location_id",
    "readmission_origin_location_name",
    "readmission_origin_specialty_id",
    "readmission_origin_specialty_name",
    "readmission_origin_inferred",
    # why they were admitted - non-proprietary specialty values pass through for later mapping
    "is_medical",
    "emergency_surgery",
    "elective_surgery",
    # treatment limitations, which bear on whether readmission was ever an option
    "do_not_readmit",
    "palliative_care",
    # where the stay ended, and what happened next
    "discharge_destination",
    "died_at",
    "died_on_icu",
    "time_to_death",
    "has_death_7days",
    "has_icu_readmission",
    "icu_readmission_time",
    "time_to_icu_readmission",
    "has_icu_readmission_7days",
    "outcome",
)

#: Boolean stay attributes. Missing values mean False; event timestamps remain nullable.
STAY_FLAG_COLUMNS = (
    "is_study_cohort",
    "icu_los_lt_12h",
    "is_medical",
    "emergency_surgery",
    "elective_surgery",
    "do_not_readmit",
    "palliative_care",
    "died_on_icu",
    "has_death_7days",
    "has_icu_readmission",
    "has_icu_readmission_7days",
    "outcome",
)

#: Nullable booleans where missing means unknown rather than False.
NULLABLE_STAY_FLAG_COLUMNS = (
    "ready_for_discharge",
    "readmission_origin_inferred",
)

# Canonical fields added by the public value-mapping stage before enrichment.
MAPPED_STAY_COLUMNS = (
    "specialty",
    "mapped_discharge_destination",
    "discharge_specialty",
    "discharge_unit_type",
    "readmission_origin_specialty",
    "readmission_origin_unit_type",
)


def load_cdm(project_root: str, hospital: str) -> dict[str, pd.DataFrame]:
    """Read all required CDM-lite tables for one hospital."""
    from algo_vs_doc.interim import (
        interim_path,
    )  # local: keeps this module import-light

    tables = {}
    for name in CDM_TABLES:
        path = interim_path(project_root, hospital, f"{name}.parquet")
        try:
            tables[name] = pd.read_parquet(path)
        except FileNotFoundError as error:
            raise FileNotFoundError(
                f"{path} is missing - run the private ETL's "
                f"'Transform to Common Data Model {hospital}' notebook first"
            ) from error
    return tables


def _as_flag(values: pd.Series) -> pd.Series:
    """Coerce a flag to bool, treating missing values as False without dtype downcasting."""
    return values.astype("boolean").fillna(False).astype(bool)


def check_stay_columns(icu_stays: pd.DataFrame) -> None:
    """Raise if ``icu_stays`` does not meet the shared stay-column contract."""
    missing = [column for column in STAY_COLUMNS if column not in icu_stays.columns]
    if missing:
        raise KeyError(
            f"icu_stays is missing {missing} - the ETL must supply every column in "
            f"enrichment.STAY_COLUMNS, since the shared analysis cannot derive them from a "
            f"private schema"
        )
    invalid_durations = [
        column
        for column in STAY_DURATION_HOURS_COLUMNS
        if not ptypes.is_numeric_dtype(icu_stays[column].dtype)
    ]
    if invalid_durations:
        raise TypeError(
            f"icu_stays duration columns must contain numeric hours: {invalid_durations}"
        )


def enrich(
    tables: dict[str, pd.DataFrame],
    *,
    hospital: str,
    condition_concepts: frozenset[str] = CONDITION_CONCEPTS,
    device_concepts: frozenset[str] = DEVICE_CONCEPTS,
    vasopressor_drug_concepts: frozenset[str] = VASOPRESSOR_DRUG_CONCEPTS,
) -> pd.DataFrame:
    """Merge severity, condition, life-support, and prediction data onto ICU stays.

    All ICU stays are retained so enrolment exclusions and group comparisons use the same source
    cohort.
    """
    icu_stays = tables["icu_stays"]
    check_stay_columns(icu_stays)

    mapped_columns = [column for column in MAPPED_STAY_COLUMNS if column in icu_stays]
    stays = icu_stays[[*STAY_COLUMNS, *mapped_columns]].copy()
    stays[list(STAY_FLAG_COLUMNS)] = stays[list(STAY_FLAG_COLUMNS)].apply(_as_flag)
    stays[list(NULLABLE_STAY_FLAG_COLUMNS)] = stays[
        list(NULLABLE_STAY_FLAG_COLUMNS)
    ].apply(lambda values: values.astype("boolean"))
    expected_outcome = stays["has_death_7days"].fillna(False) | stays[
        "has_icu_readmission_7days"
    ].fillna(False)
    study_rows = stays["is_study_cohort"].fillna(False)
    if (
        not stays.loc[study_rows, "outcome"]
        .fillna(False)
        .eq(expected_outcome.loc[study_rows])
        .all()
    ):
        raise ValueError(
            "outcome must equal ICU readmission or death within 7 days; "
            "HDU readmission is not part of the primary endpoint"
        )
    stays = cohort_outcomes.exclude_elective_readmissions(stays)
    discharge_locations.validate_destinations(stays["discharge_destination"])
    stays["post_discharge_status"] = discharge_locations.derive_post_discharge_status(
        stays
    )
    if "discharge_unit_type" in stays:
        stays["discharge_care_level"] = discharge_locations.derive_discharge_care_level(
            stays["discharge_unit_type"]
        )
    severity = pipeline.severity_summary(
        icu_stays,
        tables["measurement"],
        tables["drug_exposure"],
    )
    conditions = cdm.condition_flags(
        tables["condition_occurrence"],
        stays["icu_stay_id"],
        concepts=condition_concepts,
    )
    diagnosis_stays = stays.merge(
        conditions, on="icu_stay_id", how="left", validate="one_to_one"
    )
    diagnoses = cdm.diagnosis_windows(
        tables["diagnosis_occurrence"],
        diagnosis_stays,
        condition_concepts=condition_concepts,
    )
    support = life_support.presence_flags(
        tables["device_exposure"], stays["icu_stay_id"], concepts=device_concepts
    )
    vasopressor_support = life_support.vasopressor_inotrope_flags(
        tables["drug_exposure"],
        stays["icu_stay_id"],
        drug_concepts=vasopressor_drug_concepts,
    )
    predictions = cdm.predictions_wide(tables["prediction"])

    enriched = stays
    for addition in (
        severity,
        conditions,
        diagnoses,
        support,
        vasopressor_support,
        predictions,
    ):
        enriched = enriched.merge(addition, on="icu_stay_id", how="left")

    enriched = operational_pressure.add_operational_pressure(enriched)
    enriched["hospital"] = hospital
    return enriched
