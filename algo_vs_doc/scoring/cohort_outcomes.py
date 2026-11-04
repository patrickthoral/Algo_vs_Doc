"""Classify death or ICU readmission within a follow-up horizon."""

from __future__ import annotations

import pandas as pd


def exclude_elective_readmissions(icu_stays: pd.DataFrame) -> pd.DataFrame:
    """Exclude linked elective ICU admissions from the readmission outcome.

    Observed readmissions whose target stay is unavailable remain outcomes and
    retain unknown elective-surgery status.
    """
    required = {
        "icu_stay_id",
        "patient_id",
        "is_study_cohort",
        "icu_admitted_at",
        "icu_readmission_time",
        "time_to_icu_readmission",
        "elective_surgery",
        "has_death_7days",
        "has_icu_readmission_7days",
        "outcome",
    }
    missing = sorted(required - set(icu_stays.columns))
    if missing:
        raise KeyError(f"icu_stays is missing {missing}")

    result = icu_stays.copy()
    result["icu_admitted_at"] = pd.to_datetime(result["icu_admitted_at"])
    result["icu_readmission_time"] = pd.to_datetime(result["icu_readmission_time"])
    targets = result[["patient_id", "icu_admitted_at", "elective_surgery"]].copy()
    if targets.duplicated(["patient_id", "icu_admitted_at"]).any():
        raise ValueError("ICU stays must be unique by patient and admission time")
    targets = targets.rename(
        columns={
            "icu_admitted_at": "icu_readmission_time",
            "elective_surgery": "readmission_elective_surgery",
        }
    )
    linked = result[["patient_id", "icu_readmission_time"]].merge(
        targets,
        on=["patient_id", "icu_readmission_time"],
        how="left",
        validate="many_to_one",
        sort=False,
    )
    linked.index = result.index
    needs_link = result["is_study_cohort"].astype("boolean").fillna(False) & result[
        "has_icu_readmission_7days"
    ].astype("boolean").fillna(False)
    linked_elective = linked["readmission_elective_surgery"].astype("boolean")
    result["readmission_stay_linked"] = (~needs_link) | linked_elective.notna()
    result["readmission_elective_surgery"] = linked_elective
    elective = linked_elective.fillna(False)
    result["has_elective_icu_readmission_7days"] = needs_link & elective
    result["has_icu_readmission_7days"] = needs_link & ~elective
    result["outcome"] = (
        result["has_death_7days"].astype("boolean").fillna(False)
        | result["has_icu_readmission_7days"]
    )
    return result


def time_to_next_admission(
    patient_key: pd.Series, icu_admitted_at: pd.Series, icu_discharged_at: pd.Series
) -> pd.DataFrame:
    """Return time from each discharge to the next same-type admission.

    Inputs contain one admission type and share an index. The result preserves that index.
    """
    df = pd.DataFrame(
        {
            "patient_key": patient_key,
            "icu_admitted_at": icu_admitted_at,
            "icu_discharged_at": icu_discharged_at,
        }
    )
    ordered = df.sort_values(["patient_key", "icu_admitted_at"])
    ordered["next_admission_time"] = ordered.groupby("patient_key")[
        "icu_admitted_at"
    ].shift(-1)
    ordered["has_next_admission"] = ordered["next_admission_time"].notna()
    ordered["time_to_next_admission"] = (
        ordered["next_admission_time"] - ordered["icu_discharged_at"]
    )
    return ordered[
        ["has_next_admission", "next_admission_time", "time_to_next_admission"]
    ].reindex(patient_key.index)


def classify_outcome(
    time_to_death: pd.Series,
    time_to_icu_readmission: pd.Series,
    *,
    horizon: pd.Timedelta = pd.Timedelta(days=7),
) -> pd.Series:
    """Flag death or ICU readmission within ``horizon``.

    Inputs are timedeltas from discharge; missing events do not contribute.
    """
    has_death = time_to_death <= horizon
    has_icu_readmission = time_to_icu_readmission <= horizon
    return has_death | has_icu_readmission


#: Outcome columns in reporting order.
OUTCOME_COLUMNS = [
    "time_to_death",
    "has_death_7days",
    "has_icu_readmission_7days",
    "outcome",
]


def summarize(
    icu_stays: pd.DataFrame,
    *,
    horizon: pd.Timedelta = pd.Timedelta(days=7),
) -> pd.DataFrame:
    """Return the composite outcome per stay in a follow-up-complete cohort."""
    result = icu_stays[["icu_stay_id"]].copy()
    time_to_death = icu_stays["died_at"] - icu_stays["icu_discharged_at"]
    result["time_to_death"] = time_to_death
    result["has_death_7days"] = time_to_death <= horizon
    result["has_icu_readmission_7days"] = (
        icu_stays["time_to_icu_readmission"] <= horizon
    )
    result["outcome"] = classify_outcome(
        time_to_death,
        icu_stays["time_to_icu_readmission"],
        horizon=horizon,
    )
    return result.reset_index(drop=True)
