"""Descriptive tables for the composite outcome and readmission pathways."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from algo_vs_doc import discharge_locations


def event_counts(frame: pd.DataFrame) -> pd.DataFrame:
    """Count mutually exclusive seven-day event types in reporting order."""
    column = "post_discharge_status"
    if column not in frame:
        raise KeyError(f"event counts require {column!r}")

    counts = (
        frame[column]
        .value_counts()
        .reindex(
            discharge_locations.POST_DISCHARGE_STATUSES,
            fill_value=0,
        )
    )
    denominator = int(counts.sum())
    percentage = (
        100 * counts / denominator
        if denominator
        else pd.Series(0.0, index=counts.index)
    )
    return pd.DataFrame(
        {
            "event_type": counts.index,
            "n": counts.to_numpy(dtype="int64"),
            "percentage": percentage.to_numpy(dtype="float64"),
        }
    )


def event_counts_by_cohort(
    cohorts: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    """Stack event counts for named hospital and pooled cohorts."""
    return pd.concat(
        [event_counts(frame).assign(cohort=name) for name, frame in cohorts.items()],
        ignore_index=True,
    )[["cohort", "event_type", "n", "percentage"]]


def readmission_pathways(frame: pd.DataFrame) -> pd.DataFrame:
    """Count discharge-specialty to readmission-origin-specialty pathways."""
    required = {
        "post_discharge_status",
        "discharge_specialty",
        "readmission_origin_specialty",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(
            f"readmission pathways require {sorted(required)}; missing {missing}"
        )

    readmission_statuses = {
        "icu_readmission",
        "readmission_and_death",
    }
    pathways = frame.loc[
        frame["post_discharge_status"].isin(readmission_statuses),
        ["discharge_specialty", "readmission_origin_specialty"],
    ].fillna("unknown")
    return (
        pathways.value_counts(sort=False)
        .rename("n")
        .reset_index()
        .sort_values(
            ["n", "discharge_specialty", "readmission_origin_specialty"],
            ascending=[False, True, True],
        )
        .reset_index(drop=True)
    )
