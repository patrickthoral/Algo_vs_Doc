"""Combine independent SOFA component timelines into daily and per-stay summaries."""

from __future__ import annotations

from typing import Mapping, Sequence

import pandas as pd

from algo_vs_doc.scoring import common

#: Six SOFA component columns in reporting order.
COMPONENT_COLUMNS = [
    "sofa_resp",
    "sofa_neuro",
    "sofa_cardvasc",
    "sofa_liver",
    "sofa_coagulation",
    "sofa_kidneys",
]


def daily_scores(
    components: Mapping[str, pd.DataFrame],
    *,
    component_columns: Sequence[str] = COMPONENT_COLUMNS,
    carry_forward_days: int = 1,
) -> pd.DataFrame:
    """Return daily worst component scores and their SOFA total.

    Missing days are inserted. Component values carry forward for ``carry_forward_days`` and
    remaining missing values become zero.
    """
    usable = [
        frame[["icu_stay_id", "day_index", column]]
        for column, frame in components.items()
        if column in frame.columns and not frame.empty
    ]
    if not usable:
        return pd.DataFrame(columns=["icu_stay_id", "day_index", "sofa_total"])

    scored = pd.concat(usable, ignore_index=True)
    scored = scored.dropna(subset=["day_index"])
    scored["day_index"] = scored["day_index"].astype(int)
    daily = scored.groupby(["icu_stay_id", "day_index"], as_index=False).max()

    daily = _fill_gap_days(daily)
    present = [c for c in component_columns if c in daily.columns]
    daily = daily.sort_values(["icu_stay_id", "day_index"])
    daily[present] = daily.groupby("icu_stay_id")[present].ffill(
        limit=carry_forward_days
    )
    daily[present] = daily[present].fillna(0)
    daily["sofa_total"] = daily[present].sum(axis=1)
    return daily.reset_index(drop=True)


def _fill_gap_days(daily: pd.DataFrame) -> pd.DataFrame:
    """Give every stay a consecutive ``day_index`` run from its first scored day to its last."""
    if daily.empty:
        return daily
    spans = daily.groupby("icu_stay_id")["day_index"].agg(["min", "max"]).reset_index()
    # Empty ``apply`` changes result type, so the function returns above.
    spans["day_index"] = spans.apply(
        lambda r: list(range(r["min"], r["max"] + 1)), axis=1
    )
    complete = spans.explode("day_index").drop(columns=["min", "max"])
    complete["day_index"] = complete["day_index"].astype(int)
    return complete.merge(daily, on=["icu_stay_id", "day_index"], how="left")


def at_admission(daily: pd.DataFrame) -> pd.DataFrame:
    """The total on day 0 (the admission day), per stay: ``icu_stay_id, sofa_admission``."""
    return (
        daily.loc[daily["day_index"] == 0, ["icu_stay_id", "sofa_total"]]
        .rename(columns={"sofa_total": "sofa_admission"})
        .reset_index(drop=True)
    )


def maximum(daily: pd.DataFrame) -> pd.DataFrame:
    """The worst daily total across the whole stay: ``icu_stay_id, sofa_max``."""
    return (
        daily.groupby("icu_stay_id", as_index=False)["sofa_total"]
        .max()
        .rename(columns={"sofa_total": "sofa_max"})
    )


def at_discharge(
    components: Mapping[str, pd.DataFrame],
    icu_stays: pd.DataFrame,
    *,
    component_columns: Sequence[str] = COMPONENT_COLUMNS,
) -> pd.DataFrame:
    """Build a discharge total from each component's latest pre-discharge value.

    Components without a pre-discharge value contribute zero.
    """
    result = icu_stays[["icu_stay_id"]].drop_duplicates()
    for column, frame in components.items():
        if column not in frame.columns:
            continue
        if "measured_at" in frame.columns:
            nearest = common.nearest_before_discharge(frame, icu_stays, [column])
        else:
            nearest = frame[["icu_stay_id", column]]
        result = result.merge(nearest, on="icu_stay_id", how="left")

    present = [c for c in component_columns if c in result.columns]
    result[present] = result[present].apply(pd.to_numeric, errors="coerce").fillna(0)
    result["sofa_discharge"] = result[present].sum(axis=1)
    return result.reset_index(drop=True)
