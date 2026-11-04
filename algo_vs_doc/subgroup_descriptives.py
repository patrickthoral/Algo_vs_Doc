"""Descriptive outcome tables for clinically relevant cohort subgroups."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


def admission_type(frame: pd.DataFrame) -> pd.Series:
    """Derive mutually exclusive medical and surgical admission types."""
    columns = ("is_medical", "emergency_surgery", "elective_surgery")
    missing = [column for column in columns if column not in frame]
    if missing:
        raise KeyError(f"admission type is missing {missing}")

    flags = frame.loc[:, columns].fillna(False).astype(bool)
    values = np.select(
        [
            flags["is_medical"],
            flags["emergency_surgery"],
            flags["elective_surgery"],
        ],
        ["medical", "emergency_surgery", "elective_surgery"],
        default="other_or_unknown",
    )
    return pd.Series(values, index=frame.index, name="admission_type")


def length_of_stay_group(values: pd.Series) -> pd.Series:
    """Group ICU length of stay using explicit clinically readable hour boundaries."""
    numeric = pd.to_numeric(values, errors="coerce")
    return pd.cut(
        numeric,
        bins=[-np.inf, 24, 72, 168, np.inf],
        right=False,
        labels=["<24 hours", "24-<72 hours", "72-<168 hours", ">=168 hours"],
    ).rename("icu_los_group")


def outcome_counts(frame: pd.DataFrame, group: pd.Series | str) -> pd.DataFrame:
    """Count stays and composite outcomes within one subgroup variable."""
    if "outcome" not in frame:
        raise KeyError("subgroup outcome counts require 'outcome'")
    groups = frame[group] if isinstance(group, str) else group
    name = group if isinstance(group, str) else group.name
    if not name:
        raise ValueError("subgroup series must have a name")

    working = pd.DataFrame(
        {
            "group": groups.astype("object").fillna("unknown"),
            "outcome": frame["outcome"],
        }
    )
    result = working.groupby("group", sort=False, observed=True)["outcome"].agg(
        n="size", events_n=lambda values: int(values.astype(bool).sum())
    )
    result["event_rate"] = result["events_n"] / result["n"]
    return result.reset_index().rename(columns={"group": "level"}).assign(variable=name)


def tables_by_cohort(cohorts: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Stack specialty, admission-type, and ICU-length-of-stay outcome tables."""
    rows = []
    for cohort, frame in cohorts.items():
        variables = [
            frame["specialty"].rename("admitting_specialty"),
            frame["discharge_care_level"],
            admission_type(frame),
            length_of_stay_group(frame["icu_los"]),
        ]
        if (
            "patient_category_name" in frame
            and frame["patient_category_name"].notna().any()
        ):
            variables.append(frame["patient_category_name"].rename("patient_category"))
        rows.extend(
            outcome_counts(frame, variable).assign(cohort=cohort)
            for variable in variables
        )
    return pd.concat(rows, ignore_index=True)[
        ["cohort", "variable", "level", "n", "events_n", "event_rate"]
    ]
