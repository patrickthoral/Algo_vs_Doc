"""Derive ICU census and transfer-volume proxies from the study-period stay table."""

from __future__ import annotations

import numpy as np
import pandas as pd

from algo_vs_doc import local_time


def other_active_stays(
    admitted_at: pd.Series,
    discharged_at: pd.Series,
    evaluated_at: pd.Series,
) -> pd.Series:
    """Count other active ICU stays at each aligned evaluation timestamp.

    Stay intervals are half-open: ``admitted_at <= time < discharged_at``. An
    ongoing stay has no upper bound. The aligned row's own stay is subtracted
    when it is active at the evaluation time.
    """
    admitted = pd.to_datetime(admitted_at)
    discharged = pd.to_datetime(discharged_at)
    evaluated = pd.to_datetime(evaluated_at)

    valid_starts = admitted.dropna().sort_values().to_numpy()
    valid_ends = discharged.dropna().sort_values().to_numpy()
    result = pd.Series(pd.NA, index=evaluated.index, dtype="Int64")
    known = evaluated.notna()
    if not known.any():
        return result

    times = evaluated.loc[known].to_numpy()
    started = np.searchsorted(valid_starts, times, side="right")
    ended = np.searchsorted(valid_ends, times, side="right")
    active = started - ended
    own_active = admitted.loc[known].le(evaluated.loc[known]) & (
        discharged.loc[known].isna() | evaluated.loc[known].lt(discharged.loc[known])
    )
    result.loc[known] = active - own_active.astype(int).to_numpy()
    return result


def other_same_day_events(
    occurred_at: pd.Series,
    is_event: pd.Series,
    evaluated_at: pd.Series,
) -> pd.Series:
    """Count other flagged events on each aligned evaluation date."""
    occurred = local_time.amsterdam(occurred_at)
    evaluated = local_time.amsterdam(evaluated_at)
    event = is_event.fillna(False).astype(bool)
    event_day = occurred.dt.normalize()
    evaluation_day = evaluated.dt.normalize()
    counts = event_day.loc[event & event_day.notna()].value_counts()

    result = evaluation_day.map(counts).fillna(0).astype("Int64")
    result.loc[evaluation_day.isna()] = pd.NA
    own_event = event & event_day.eq(evaluation_day)
    result.loc[own_event] = result.loc[own_event] - 1
    return result


def same_day_events(
    occurred_at: pd.Series,
    is_event: pd.Series,
    evaluated_at: pd.Series,
) -> pd.Series:
    """Count all flagged events on each aligned Amsterdam-local calendar date."""
    occurred = local_time.amsterdam(occurred_at)
    evaluated = local_time.amsterdam(evaluated_at)
    event_day = occurred.dt.normalize()
    evaluation_day = evaluated.dt.normalize()
    event = is_event.fillna(False).astype(bool)
    counts = event_day.loc[event & event_day.notna()].value_counts()
    result = evaluation_day.map(counts).fillna(0).astype("Int64")
    result.loc[evaluation_day.isna()] = pd.NA
    return result


def add_operational_pressure(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with concurrent-census and same-day-transfer proxies."""
    required = {
        "icu_admitted_at",
        "icu_discharged_at",
        "discharge_destination",
        "elective_surgery",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(
            f"operational pressure requires {sorted(required)}; missing {missing}"
        )

    result = frame.copy()
    result["n_census_at_discharge"] = other_active_stays(
        result["icu_admitted_at"],
        result["icu_discharged_at"],
        result["icu_discharged_at"],
    )
    result["n_external_transfers_same_day"] = other_same_day_events(
        result["icu_discharged_at"],
        result["discharge_destination"].eq("external_transfer"),
        result["icu_discharged_at"],
    )
    result["n_elective_surgery_admissions_same_day"] = same_day_events(
        result["icu_admitted_at"],
        result["elective_surgery"],
        result["icu_discharged_at"],
    )
    return result
