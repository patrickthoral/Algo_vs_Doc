"""Validate and derive analysis columns from enriched flat cohort files.

Source-code mapping and questionnaire reconstruction occur before the CSV boundary.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from algo_vs_doc import cohorts as cohort_io
from algo_vs_doc import local_time
from algo_vs_doc import vocabulary
from algo_vs_doc.scoring.concepts import PREDICTION_LEVELS

FACTOR_SEPARATOR = "||"

CONFIDENCE_POOLING = {
    "low": "low/medium",
    "medium": "low/medium",
    "high": "high",
}

_REQUIRED_COLUMNS = (
    "hospital",
    "icu_discharged_at",
    "doc_predicted_at",
    "algo_predicted_at_discharge",
    "algo_predicted_at_doc_prediction",
    "doc_prediction",
    "doc_prediction_cat",
    "confidence",
    "doc_factors",
    "doc_other_factors",
    "discharge_care_level",
)

_PROBABILITY_COLUMNS = (
    "doc_prediction",
    "algo_prediction_at_discharge",
    "algo_prediction_at_doc_prediction",
)


def _require_columns(frame: pd.DataFrame) -> None:
    missing = [column for column in _REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise KeyError(
            f"analysis dataset is missing {missing} - rerun the Create Analysis Dataset notebook"
        )


def _validate_hospital(frame: pd.DataFrame, hospital: str) -> None:
    if hospital not in cohort_io.HOSPITALS:
        raise KeyError(
            f"unknown hospital {hospital!r}; expected one of {cohort_io.HOSPITALS}"
        )

    labels = set(frame["hospital"].dropna().unique())
    if frame["hospital"].isna().any() or labels != {hospital}:
        raise ValueError(
            f"analysis dataset hospital labels are {sorted(labels)!r}, expected only {hospital!r}"
        )


def _validate_levels(values: pd.Series, *, column: str) -> None:
    vocabulary.raise_unmapped(
        vocabulary.unmapped_values(values, PREDICTION_LEVELS),
        what=column,
        known=PREDICTION_LEVELS,
    )


def _validate_probabilities(frame: pd.DataFrame) -> None:
    for column in _PROBABILITY_COLUMNS:
        if column not in frame.columns:
            raise KeyError(
                f"analysis dataset is missing {column!r} - rerun the Create Analysis Dataset notebook"
            )
        values = pd.to_numeric(frame[column], errors="coerce")
        invalid = frame[column].notna() & (values.isna() | ~values.between(0, 1))
        if invalid.any():
            bad_values = sorted(frame.loc[invalid, column].unique(), key=repr)
            raise ValueError(
                f"{column} must contain proportions from 0 through 1; found {bad_values}"
            )


def _lead_time_hours(discharge: pd.Series, predicted_at: pd.Series) -> pd.Series:
    """Return signed hours from prediction to discharge."""
    return (
        pd.to_datetime(discharge) - pd.to_datetime(predicted_at)
    ).dt.total_seconds() / 3600


def _add_local_calendar_features(
    frame: pd.DataFrame, *, timestamp: str, prefix: str
) -> None:
    """Add local hour, weekday, and weekend columns to ``frame`` in place."""
    local = local_time.amsterdam(frame[timestamp])
    frame[f"{prefix}_local_hour"] = local.dt.hour.astype("Int64")
    frame[f"{prefix}_weekday"] = local.dt.weekday.astype("Int64")
    frame[f"{prefix}_is_weekend"] = local.dt.weekday.ge(5).astype("boolean")
    frame.loc[local.isna(), f"{prefix}_is_weekend"] = pd.NA


def normalize(frame: pd.DataFrame, hospital: str) -> pd.DataFrame:
    """Validate one enriched cohort and add columns used only by the analysis."""
    _require_columns(frame)
    _validate_hospital(frame, hospital)
    _validate_levels(frame["doc_prediction_cat"], column="doc_prediction_cat")
    _validate_levels(frame["confidence"], column="confidence")
    _validate_probabilities(frame)

    result = frame.copy()
    result["confidence_level"] = result["confidence"]
    result["confidence"] = result["confidence_level"].map(CONFIDENCE_POOLING)
    result["doc_lead_time_hours"] = _lead_time_hours(
        result["icu_discharged_at"], result["doc_predicted_at"]
    )
    result["algo_at_doc_prediction_lead_time_hours"] = _lead_time_hours(
        result["icu_discharged_at"], result["algo_predicted_at_doc_prediction"]
    )
    _add_local_calendar_features(
        result, timestamp="icu_discharged_at", prefix="discharge"
    )
    discharge_local = local_time.amsterdam(result["icu_discharged_at"])
    discharge_minutes = discharge_local.dt.hour * 60 + discharge_local.dt.minute
    result["discharge_after_18h"] = discharge_minutes.gt(18 * 60).astype("boolean")
    result.loc[discharge_local.isna(), "discharge_after_18h"] = pd.NA
    _add_local_calendar_features(
        result, timestamp="doc_predicted_at", prefix="prediction"
    )

    if hospital == "LUMC":
        has_prediction = result["algo_predicted_at_discharge"].notna()
        result["algo_at_discharge_lead_time_hours"] = pd.Series(
            np.where(has_prediction, 0.0, np.nan), index=result.index
        )
    else:
        result["algo_at_discharge_lead_time_hours"] = _lead_time_hours(
            result["icu_discharged_at"], result["algo_predicted_at_discharge"]
        )

    return result


def load_cohorts(
    data_dir: str,
    *,
    hospitals: Sequence[str] = cohort_io.HOSPITALS,
    include_combined: bool = True,
    **read_csv_kwargs,
) -> dict[str, pd.DataFrame]:
    """Load the flat hospital CSV files and prepare the requested analysis cohorts."""
    loaded = {
        hospital: normalize(
            cohort_io.load(data_dir, hospital, **read_csv_kwargs),
            hospital,
        )
        for hospital in hospitals
    }
    if include_combined and len(loaded) > 1:
        loaded[cohort_io.COMBINED] = cohort_io.combine(loaded.values())
    return loaded
