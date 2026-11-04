"""Generate synthetic CDM-lite tables for tests and notebook execution.

The seeded data matches the required table structure but is not clinically meaningful and contains
no patient records.

Run directly with::

    python tests/fixtures/synthetic_cdm.py /tmp/demo
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

from algo_vs_doc.scoring.concepts import (
    CONDITION_CONCEPTS,
    DEVICE_CONCEPTS,
    O2_DEVICE_VALUES,
    SEDATIVE_DRUG_CONCEPT,
    VASOPRESSOR_DRUG_CONCEPTS,
)

#: plausible ranges per numeric measurement concept - wide enough that stays land on either side
#: of the scoring thresholds, so a component that stopped scoring shows up as a flat column.
NUMERIC_RANGES = {
    "platelets": (15, 350),
    "creatinine": (40, 400),
    "bilirubin": (3, 250),
    "map": (45, 105),
    "heart_rate": (45, 135),
    "respiratory_rate": (7, 32),
    "temperature": (34.5, 39.5),
    "spo2": (85, 100),
    "systolic_blood_pressure": (75, 175),
    "art_po2": (6, 16),
    "fio2_fraction": (0.21, 1.0),
    "o2_flow_rate": (0, 15),
    "peep": (5, 15),
    "urine_output": (0, 250),
    "gcs_eye": (1, 4),
    "gcs_motor": (1, 6),
    "gcs_verbal": (1, 5),
}

#: how each hospital serializes a physician's selected factors, matching what its own ETL writes.
#: LUMC's question was split across two form sections whose option ids overlap, so its tokens
#: name the section; VUmc asked one question, plus a free-text answer LUMC's form never offered.
FACTOR_TOKENS = {
    "LUMC": ["doc_factors:5||doc_factors_2:15", "doc_factors:10", "doc_factors_2:13"],
    "VUmc": ["doc_factors:30", "doc_factors:30||doc_factors:38", "doc_factors:41"],
}
FREE_TEXT = {"LUMC": [None], "VUmc": [None, "COVID", "pijnstilling", None]}
SPECIALTY_IDS = {
    "LUMC": [2, 4, 5, 9],
    "VUmc": [30300, 31300, 32000, 33085],
}
SEX_VALUES = {"LUMC": ["male", "female"], "VUmc": ["M", "V"]}


def _stays(rng, hospital: str, n: int) -> pd.DataFrame:
    admitted = pd.Timestamp("2023-01-01") + pd.to_timedelta(
        rng.integers(0, 400, n), "D"
    )
    length = pd.to_timedelta(rng.integers(4, 240, n), "h")
    discharged = admitted + length
    discharge_destination = rng.choice(
        ["ward", "home", "external_transfer", "died_on_icu"],
        n,
        p=[0.75, 0.08, 0.09, 0.08],
    )
    died_on_icu = discharge_destination == "died_on_icu"
    ward_location_ids = (
        rng.choice([6, 5, 24], n)
        if hospital == "LUMC"
        else rng.choice([201001056, 201001057, 201001061], n)
    )
    ward_location_names = {
        "LUMC": {
            6: "C09P - Hartziekten",
            5: "C09P - Hartbewaking",
            24: "J11Q - Neuro MC (VPL2)",
        },
        "VUmc": {
            201001056: "VUMC CARDIOLOGIE",
            201001057: "VUMC 5D HARTBEWAKING EN EHH",
            201001061: "VUMC 7D MEDIUM CARE",
        },
    }[hospital]
    discharge_location_id = np.where(
        discharge_destination == "ward",
        ward_location_ids,
        np.nan,
    )
    discharge_location_name = np.where(
        discharge_destination == "ward",
        pd.Series(ward_location_ids).map(ward_location_names),
        None,
    )
    # Readmission linkage is exercised in focused unit tests. Notebook fixtures use death
    # events so every recorded readmission would otherwise require a generated target stay.
    has_icu_readmission = np.zeros(n, dtype=bool)
    has_readmission = has_icu_readmission
    has_death_7days = rng.random(n) < 0.1
    has_icu_readmission_7days = np.zeros(n, dtype=bool)
    readiness = pd.array(
        rng.choice([True, False, None], n, p=[0.65, 0.1, 0.25]),
        dtype="boolean",
    )
    patient_category = (
        rng.choice([1, 2, 3], n) if hospital == "LUMC" else np.full(n, np.nan)
    )
    return pd.DataFrame(
        {
            "icu_stay_id": [f"{hospital}-{i:04d}" for i in range(n)],
            # fewer patients than stays, so readmissions exist and identifier mapping has
            # something to collapse
            "patient_id": rng.integers(1, int(n * 0.8) + 1, n),
            "is_study_cohort": np.ones(n, dtype=bool),
            "icu_admitted_at": admitted,
            "icu_discharged_at": discharged,
            "icu_los": length / pd.Timedelta("1h"),
            "icu_los_lt_12h": length < pd.Timedelta("12h"),
            "sex": rng.choice(SEX_VALUES[hospital], n),
            "age_at_admission": rng.integers(18, 95, n),
            "specialty_id": rng.choice(SPECIALTY_IDS[hospital], n),
            "patient_category_id": patient_category,
            "patient_category_name": np.where(
                hospital == "LUMC",
                pd.Series(patient_category).map(
                    {1: "Category 1", 2: "Category 2", 3: "Category 3"}
                ),
                None,
            ),
            "discharge_location_id": discharge_location_id,
            "discharge_location_name": discharge_location_name,
            "ready_for_discharge": readiness,
            "readmission_origin_location_id": np.where(
                has_readmission & (hospital == "VUmc"), 201001056, np.nan
            ),
            "readmission_origin_location_name": np.where(
                has_readmission & (hospital == "VUmc"),
                "VUMC CARDIOLOGIE",
                None,
            ),
            "readmission_origin_specialty_id": np.where(
                has_readmission,
                rng.choice(SPECIALTY_IDS[hospital], n),
                np.nan,
            ),
            "readmission_origin_specialty_name": None,
            "readmission_origin_inferred": pd.array(
                np.where(has_readmission, hospital == "LUMC", None),
                dtype="boolean",
            ),
            "is_medical": rng.random(n) < 0.55,
            "emergency_surgery": rng.random(n) < 0.2,
            "elective_surgery": rng.random(n) < 0.25,
            "do_not_readmit": rng.random(n) < 0.07,
            "palliative_care": rng.random(n) < 0.04,
            "discharge_destination": discharge_destination,
            "died_at": pd.Series(pd.NaT, index=range(n)).where(
                ~died_on_icu, discharged
            ),
            "died_on_icu": died_on_icu,
            "time_to_death": np.nan,
            "has_death_7days": has_death_7days,
            "has_icu_readmission": has_icu_readmission,
            "icu_readmission_time": pd.NaT,
            "time_to_icu_readmission": np.nan,
            "has_icu_readmission_7days": has_icu_readmission_7days,
            "outcome": has_death_7days | has_icu_readmission_7days,
        }
    )


def _measurements(rng, stays: pd.DataFrame, per_concept: int = 4) -> pd.DataFrame:
    rows = []
    for stay in stays.itertuples():
        hours = np.linspace(0, max(stay.icu_los, 1), per_concept)
        for concept, (low, high) in NUMERIC_RANGES.items():
            for hour in hours:
                rows.append(
                    {
                        "icu_stay_id": stay.icu_stay_id,
                        "measurement_concept_name": concept,
                        "measured_at": stay.icu_admitted_at
                        + pd.Timedelta(hours=float(hour)),
                        "value_as_number": float(rng.uniform(low, high)),
                        "value_as_string": None,
                    }
                )
        # the one categorical concept: pivoted from value_as_string, not value_as_number
        for hour in hours:
            rows.append(
                {
                    "icu_stay_id": stay.icu_stay_id,
                    "measurement_concept_name": "o2_device",
                    "measured_at": stay.icu_admitted_at
                    + pd.Timedelta(hours=float(hour)),
                    "value_as_number": None,
                    "value_as_string": str(rng.choice(sorted(O2_DEVICE_VALUES))),
                }
            )
    return pd.DataFrame(rows)


def _drug_exposure(rng, stays: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for stay in stays.itertuples():
        for drug in sorted(VASOPRESSOR_DRUG_CONCEPTS) + [SEDATIVE_DRUG_CONCEPT]:
            if rng.random() > 0.3:
                continue
            start = stay.icu_admitted_at + pd.Timedelta(hours=float(rng.uniform(0, 12)))
            rows.append(
                {
                    "icu_stay_id": stay.icu_stay_id,
                    "drug_concept_name": drug,
                    "drug_exposure_start_datetime": start,
                    "drug_exposure_end_datetime": start
                    + pd.Timedelta(hours=float(rng.uniform(1, 30))),
                    # spans the SOFA cardiovascular dose thresholds
                    "dose_value": float(rng.uniform(0.01, 0.6)),
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "icu_stay_id",
            "drug_concept_name",
            "drug_exposure_start_datetime",
            "drug_exposure_end_datetime",
            "dose_value",
        ],
    )


def _device_exposure(rng, stays: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for stay in stays.itertuples():
        for concept in sorted(DEVICE_CONCEPTS):
            if rng.random() > 0.25:
                continue
            start = stay.icu_admitted_at + pd.Timedelta(hours=float(rng.uniform(0, 6)))
            rows.append(
                {
                    "icu_stay_id": stay.icu_stay_id,
                    "device_concept_name": concept,
                    "device_exposure_start_datetime": start,
                    "device_exposure_end_datetime": start
                    + pd.Timedelta(hours=float(rng.uniform(2, 48))),
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "icu_stay_id",
            "device_concept_name",
            "device_exposure_start_datetime",
            "device_exposure_end_datetime",
        ],
    )


def _condition_occurrence(rng, stays: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "icu_stay_id": stay.icu_stay_id,
            "condition_concept_name": concept,
            "condition_start_datetime": pd.NaT,
        }
        for stay in stays.itertuples()
        for concept in sorted(CONDITION_CONCEPTS)
        if rng.random() < 0.15
    ]
    return pd.DataFrame(
        rows,
        columns=["icu_stay_id", "condition_concept_name", "condition_start_datetime"],
    )


def _diagnosis_occurrence(rng, stays: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for stay in stays.itertuples():
        if rng.random() < 0.3:
            rows.append(
                {
                    "patient_id": stay.patient_id,
                    "diagnosis_source_code": f"synthetic-{rng.integers(1, 5)}",
                    "diagnosis_source_label": "Synthetic diagnosis",
                    "diagnosed_at": stay.icu_admitted_at
                    + pd.Timedelta(hours=float(rng.uniform(0, max(stay.icu_los, 1)))),
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "patient_id",
            "diagnosis_source_code",
            "diagnosis_source_label",
            "diagnosed_at",
        ],
    )


def _prediction(rng, stays: pd.DataFrame, hospital: str) -> pd.DataFrame:
    rows = []
    for stay in stays.itertuples():
        # a quarter of stays have no physician prediction: that is what the enrolment funnel
        # excludes on, and what the missing-prediction comparison needs to have something to
        # compare, so it must be represented rather than generated away
        if rng.random() > 0.25:
            prediction_lead_hours = float(rng.uniform(0, min(20, stay.icu_los)))
            rows.append(
                {
                    "icu_stay_id": stay.icu_stay_id,
                    "predictor": "physician",
                    "prediction_timing": "at_doc_prediction",
                    "predicted_at": stay.icu_discharged_at
                    - pd.Timedelta(hours=prediction_lead_hours),
                    "prediction_value": float(rng.uniform(0, 1)),
                    "prediction_category": str(rng.choice(["low", "medium", "high"])),
                    "confidence": str(rng.choice(["low", "medium", "high"])),
                    "factors": str(rng.choice(FACTOR_TOKENS[hospital])),
                    "other_factors": rng.choice(FREE_TEXT[hospital]),
                    "features": None,
                }
            )
        for timing, when in (
            ("at_doc_prediction", stay.icu_discharged_at - pd.Timedelta(hours=6)),
            (
                "at_discharge",
                stay.icu_discharged_at
                - pd.Timedelta(minutes=float(rng.uniform(0, 90))),
            ),
        ):
            rows.append(
                {
                    "icu_stay_id": stay.icu_stay_id,
                    "predictor": "model",
                    "prediction_timing": timing,
                    "predicted_at": when,
                    "prediction_value": float(rng.uniform(0, 1)),
                    "prediction_category": None,
                    "confidence": None,
                    "factors": None,
                    "other_factors": None,
                    # the stored SHAP dictionary, as a literal - read with ast.literal_eval
                    "features": str(
                        {
                            "age": float(rng.uniform(-0.3, 0.3)),
                            "lactate__last__overall": float(rng.uniform(-0.3, 0.3)),
                            "bilirubin_total__mean__last_24h": float(
                                rng.uniform(-0.3, 0.3)
                            ),
                        }
                    ),
                }
            )
    return pd.DataFrame(rows)


def build(
    hospital: str, *, n_stays: int = 60, seed: int = 0
) -> dict[str, pd.DataFrame]:
    """Build one hospital's intermediate tables. Same seed and arguments give the same tables."""
    rng = np.random.default_rng(seed)
    stays = _stays(rng, hospital, n_stays)
    return {
        "icu_stays": stays,
        "measurement": _measurements(rng, stays),
        "drug_exposure": _drug_exposure(rng, stays),
        "device_exposure": _device_exposure(rng, stays),
        "condition_occurrence": _condition_occurrence(rng, stays),
        "diagnosis_occurrence": _diagnosis_occurrence(rng, stays),
        "prediction": _prediction(rng, stays, hospital),
    }


def write(
    project_root: str, *, hospitals=("VUmc", "LUMC"), n_stays: int = 60, seed: int = 0
) -> None:
    """Populate ``<project_root>/data/interim/<hospital>/`` for each hospital.

    Each hospital gets its own seed offset, so the two are not copies of one another - a bug
    that only shows up when the sites differ would otherwise be invisible.
    """
    from algo_vs_doc.interim import interim_path

    for offset, hospital in enumerate(hospitals):
        for name, table in build(hospital, n_stays=n_stays, seed=seed + offset).items():
            path = interim_path(project_root, hospital, f"{name}.parquet")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            table.to_parquet(path, index=False)
    os.makedirs(os.path.join(project_root, "data"), exist_ok=True)


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "synthetic"
    write(target)
    print(
        f"Wrote synthetic intermediate tables to {os.path.join(target, 'data', 'interim')}"
    )
