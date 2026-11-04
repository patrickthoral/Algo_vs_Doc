"""Shared stay-keyed operations for SOFA and NEWS components."""

from __future__ import annotations

from typing import Sequence

import pandas as pd


def add_day_index(
    measurements: pd.DataFrame,
    icu_stays: pd.DataFrame,
    *,
    clip_to_stay: bool = True,
) -> pd.DataFrame:
    """Add elapsed time and zero-based ICU day index to measurements.

    When ``clip_to_stay`` is true, measurements outside the ICU interval are removed.
    """
    merged = measurements.merge(
        icu_stays[["icu_stay_id", "icu_admitted_at", "icu_discharged_at"]],
        on="icu_stay_id",
        how="inner",
    )
    if clip_to_stay:
        merged = merged[
            (merged["measured_at"] >= merged["icu_admitted_at"])
            & (merged["measured_at"] <= merged["icu_discharged_at"])
        ]
    merged["time_since_admission"] = merged["measured_at"] - merged["icu_admitted_at"]
    merged["day_index"] = merged["time_since_admission"].dt.days
    return merged


def nearest_before_discharge(
    measurements: pd.DataFrame,
    icu_stays: pd.DataFrame,
    value_cols: Sequence[str],
    *,
    icu_stay_id_col: str = "icu_stay_id",
) -> pd.DataFrame:
    """Return each stay's latest values at or before ICU discharge.

    Null timestamps are excluded. ``icu_stay_id_col`` supports alternate stay identifiers.
    """
    df = measurements.dropna(subset=["measured_at"])
    return pd.merge_asof(
        icu_stays[[icu_stay_id_col, "icu_discharged_at"]].sort_values(
            "icu_discharged_at"
        ),
        df[[icu_stay_id_col, "measured_at", *value_cols]].sort_values("measured_at"),
        left_on="icu_discharged_at",
        right_on="measured_at",
        by=icu_stay_id_col,
        direction="backward",
    )[[icu_stay_id_col, *value_cols]]


def merge_intervals(
    icu_stay_id: pd.Series,
    started_at: pd.Series,
    ended_at: pd.Series,
    *,
    max_gap: pd.Timedelta = pd.Timedelta(0),
) -> pd.DataFrame:
    """Merge touching, overlapping, or near-adjacent intervals per stay.

    ``max_gap`` controls the permitted separation. Returns ``icu_stay_id, started_at, ended_at``.
    """
    df = pd.DataFrame(
        {"icu_stay_id": icu_stay_id, "started_at": started_at, "ended_at": ended_at}
    ).sort_values(["icu_stay_id", "started_at"])
    prev_ended_at = df.groupby("icu_stay_id")["ended_at"].shift()
    starts_new_block = prev_ended_at.isna() | (
        (df["started_at"] - prev_ended_at) > max_gap
    )
    block_id = starts_new_block.groupby(df["icu_stay_id"]).cumsum()
    return (
        df.assign(block_id=block_id)
        .groupby(["icu_stay_id", "block_id"], as_index=False)
        .agg(started_at=("started_at", "min"), ended_at=("ended_at", "max"))
    )[["icu_stay_id", "started_at", "ended_at"]]


def reduce_daily_max(
    scored: pd.DataFrame,
    *,
    score_col: str,
    icu_stay_id_col: str = "icu_stay_id",
    day_col: str = "day_index",
) -> pd.DataFrame:
    """Return the maximum score per stay and day."""
    return scored.groupby([icu_stay_id_col, day_col], as_index=False)[score_col].max()
