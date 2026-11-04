"""SOFA Kidneys component: creatinine (umol/L) and 24h urine output (mL)."""

from __future__ import annotations

import pandas as pd

from algo_vs_doc.scoring import common


def extrapolate_daily_urine_output(
    urine_output_sum: pd.Series, coverage: pd.Series
) -> pd.Series:
    """Extrapolate a partial day's summed urine output to a full 24h.

    ``coverage`` is the timespan actually covered by the summed readings for that day
    (e.g. shorter than 24h on the admission day or the discharge day) - a
    ``pd.Timedelta``-typed Series, one entry per (stay, day).
    """
    return urine_output_sum * (pd.Timedelta(days=1) / coverage)


def score(creatinine: pd.Series, urine_output_daily: pd.Series) -> pd.Series:
    """Score the SOFA Kidneys component from creatinine (umol/L) and extrapolated 24h urine output (mL).

    0; creatinine >110 -> 1; >170 -> 2; >300 or urine_output_daily <500 -> 3;
    >440 or urine_output_daily <200 -> 4.

    Both a creatinine and a urine-output value are expected to combine into one score;
    whether a day without one of the two gets scored at all (e.g. via an outer vs. inner
    join of the two source signals before calling this) is a decision for the caller.
    """
    creatinine = pd.to_numeric(creatinine, errors="coerce")
    sofa_kidneys = pd.Series(0, index=creatinine.index)
    sofa_kidneys = sofa_kidneys.mask(creatinine > 110, 1)
    sofa_kidneys = sofa_kidneys.mask(creatinine > 170, 2)
    sofa_kidneys = sofa_kidneys.mask((creatinine > 300) | (urine_output_daily < 500), 3)
    sofa_kidneys = sofa_kidneys.mask((creatinine > 440) | (urine_output_daily < 200), 4)
    return sofa_kidneys


def score_readings(
    icu_stays: pd.DataFrame, creatinine: pd.DataFrame, urine_output: pd.DataFrame
) -> pd.DataFrame:
    """Score SOFA Kidneys per creatinine reading, each combined with its own day's extrapolated 24h urine output.

    ``icu_stays`` needs ``icu_stay_id, icu_admitted_at, icu_discharged_at``. ``creatinine`` and
    ``urine_output`` each need ``icu_stay_id, measured_at, value_as_number`` (e.g. the
    ``"creatinine"``/``"urine_output"`` rows of a ``measurement`` table) - ``urine_output``
    must be per-reading, not already daily-summed; this sums and extrapolates it internally
    via ``extrapolate_daily_urine_output``, using each day's actual covered timespan (from
    first to last reading that day, not always a full 24h - shorter on the admission and
    discharge days in particular).

    Only a day with *both* a creatinine reading and at least one urine-output reading gets
    scored (inner join) - a day with just one of the two contributes no row, since
    ``score`` needs both to produce a meaningful result. Returns one row per contributing
    creatinine reading: ``icu_stay_id, measured_at, sofa_kidneys``.
    """
    creatinine_with_day = common.add_day_index(
        creatinine.rename(columns={"value_as_number": "creatinine"}), icu_stays
    )
    urine_with_day = common.add_day_index(
        urine_output.rename(columns={"value_as_number": "urine_output"}), icu_stays
    )

    daily_urine = urine_with_day.groupby(
        ["icu_stay_id", "day_index"], as_index=False
    ).agg(
        urine_output_sum=("urine_output", "sum"),
        last_measured_at=("time_since_admission", "max"),
    )
    # the timespan actually covered by this day's readings - not always a full 24h (e.g.
    # shorter on the admission/discharge day) - matches both hospitals' own "sofa_day_time"
    # computation: the day's last reading's time-since-admission, minus whole elapsed days.
    daily_urine["coverage"] = daily_urine["last_measured_at"] - pd.to_timedelta(
        daily_urine["day_index"], unit="D"
    )
    daily_urine["urine_output_daily"] = extrapolate_daily_urine_output(
        daily_urine["urine_output_sum"], daily_urine["coverage"]
    )

    merged = creatinine_with_day.merge(
        daily_urine[["icu_stay_id", "day_index", "urine_output_daily"]],
        on=["icu_stay_id", "day_index"],
        how="inner",
    )
    merged["sofa_kidneys"] = score(merged["creatinine"], merged["urine_output_daily"])
    return merged[["icu_stay_id", "measured_at", "sofa_kidneys"]].reset_index(drop=True)
