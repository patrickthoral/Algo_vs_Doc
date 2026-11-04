"""Reconstruct and summarize vasopressor or inotrope support intervals."""

from __future__ import annotations

import pandas as pd

from algo_vs_doc.scoring import common


def reconstruct_intervals_from_events(
    icu_stay_id: pd.Series,
    event_time: pd.Series,
    *,
    max_gap: pd.Timedelta,
) -> pd.DataFrame:
    """Merge discrete administration events separated by at most ``max_gap``."""
    return common.merge_intervals(icu_stay_id, event_time, event_time, max_gap=max_gap)


def reconstruct_intervals_from_mar_events(
    icu_stay_id: pd.Series, taken_at: pd.Series, order_end_at: pd.Series
) -> pd.DataFrame:
    """Build MAR intervals from each event to the next event in its group.

    The final event ends at its order end. Missing final end times remain null. Inputs and output
    share an index; callers can include a drug identifier in ``icu_stay_id`` to prevent cross-drug
    chaining.
    """
    df = pd.DataFrame(
        {
            "icu_stay_id": icu_stay_id,
            "started_at": taken_at,
            "order_end_at": order_end_at,
        }
    )
    ordered = df.sort_values(["icu_stay_id", "started_at"])
    ordered["ended_at"] = ordered.groupby("icu_stay_id")["started_at"].shift(-1)
    ordered["ended_at"] = ordered["ended_at"].fillna(ordered["order_end_at"])
    return ordered[["icu_stay_id", "started_at", "ended_at"]].reindex(taken_at.index)


def total_duration(
    intervals: pd.DataFrame, *, icu_stay_id_col: str = "icu_stay_id"
) -> pd.Series:
    """Sum support interval durations per stay, excluding gaps."""
    durations = intervals["ended_at"] - intervals["started_at"]
    return durations.groupby(intervals[icu_stay_id_col]).sum()
