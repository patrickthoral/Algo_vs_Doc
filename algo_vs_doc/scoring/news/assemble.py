"""Score NEWS components and combine their latest pre-discharge values."""

from __future__ import annotations

from typing import Mapping

import pandas as pd

from algo_vs_doc.scoring import common
from algo_vs_doc.scoring.news import components

#: NEWS component score column to measurement concept.
#: AVPU is derived from processed GCS totals because it is not recorded directly.
CONCEPT_BY_COMPONENT = {
    "news_respiratory": "respiratory_rate",
    "news_oxygen_saturation": "spo2",
    "news_supplemental_oxygen": "o2_flow_rate",
    "news_temperature": "temperature",
    "news_bp": "systolic_blood_pressure",
    "news_heart_rate": "heart_rate",
}

#: Scorers for extracted NEWS component values.
_SCORERS = {
    "news_respiratory": components.score_respiratory_rate,
    "news_oxygen_saturation": components.score_spo2,
    "news_supplemental_oxygen": lambda v: components.score_supplemental_oxygen(v > 0),
    "news_temperature": components.score_temperature,
    "news_bp": components.score_blood_pressure,
    "news_heart_rate": components.score_heart_rate,
}

#: all seven component score columns, in the conventional reporting order.
COMPONENT_COLUMNS = [*CONCEPT_BY_COMPONENT, "news_avpu"]


def score_components(
    measurement: pd.DataFrame,
    *,
    concept_by_component: Mapping[str, str] = CONCEPT_BY_COMPONENT,
) -> dict[str, pd.DataFrame]:
    """Score six measured NEWS components on their independent timelines.

    AVPU is derived separately from processed GCS totals.
    """
    scored = {}
    for component, concept in concept_by_component.items():
        rows = measurement.loc[measurement["measurement_concept_name"] == concept]
        if rows.empty:
            continue
        frame = rows[["icu_stay_id", "measured_at"]].copy()
        frame[component] = _SCORERS[component](rows["value_as_number"])
        scored[component] = frame.reset_index(drop=True)
    return scored


def score_avpu_from_gcs(sofa_neuro: pd.DataFrame) -> pd.DataFrame:
    """Derive NEWS AVPU from processed GCS totals.

    Rows without a scoreable GCS total are excluded.
    """
    rows = sofa_neuro.dropna(subset=["gcs_total"])
    frame = rows[["icu_stay_id", "measured_at"]].copy()
    frame["news_avpu"] = components.score_avpu(rows["gcs_total"])
    return frame.reset_index(drop=True)


def at_discharge(
    scored_components: Mapping[str, pd.DataFrame],
    icu_stays: pd.DataFrame,
) -> pd.DataFrame:
    """Sum independently anchored NEWS components at discharge.

    Missing pre-discharge components contribute zero.
    """
    result = icu_stays[["icu_stay_id"]].drop_duplicates()
    for component, frame in scored_components.items():
        result = result.merge(
            common.nearest_before_discharge(frame, icu_stays, [component]),
            on="icu_stay_id",
            how="left",
        )

    present = [c for c in COMPONENT_COLUMNS if c in result.columns]
    result[present] = result[present].apply(pd.to_numeric, errors="coerce").fillna(0)
    result["news_discharge"] = result[present].sum(axis=1)
    return result.reset_index(drop=True)
