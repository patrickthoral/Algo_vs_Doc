"""Compare ICU-stay characteristics between groups.

The enrolment analysis uses this to assess whether prediction availability is associated with
patient or stay characteristics.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

#: how a stay's prediction availability is labelled, in reporting order.
PREDICTION_AVAILABILITY = ("both", "physician only", "model only", "neither")
PREDICTION_PAIR_AVAILABILITY = ("available", "unavailable")


def holm_adjust(p_values: pd.Series) -> pd.Series:
    """Return Holm-Bonferroni adjusted p-values while preserving missing values."""
    numeric = pd.to_numeric(p_values, errors="coerce")
    adjusted = pd.Series(np.nan, index=p_values.index, dtype=float)
    valid = numeric.notna()
    if valid.any():
        adjusted.loc[valid] = multipletests(numeric.loc[valid], method="holm")[1]
    return adjusted


def prediction_availability(
    data: pd.DataFrame,
    *,
    doc_column: str = "doc_predicted_at",
    algo_column: str = "algo_predicted_at_doc_prediction",
) -> pd.Series:
    """Classify stays by physician and model prediction availability."""
    has_doc = data[doc_column].notna()
    has_algo = data[algo_column].notna()
    labels = np.select(
        [has_doc & has_algo, has_doc & ~has_algo, ~has_doc & has_algo],
        ["both", "physician only", "model only"],
        default="neither",
    )
    return pd.Series(
        pd.Categorical(labels, categories=PREDICTION_AVAILABILITY, ordered=True),
        index=data.index,
        name="prediction_availability",
    )


def prediction_pair_availability(
    data: pd.DataFrame,
    *,
    doc_column: str = "doc_predicted_at",
    algo_column: str = "algo_predicted_at_doc_prediction",
) -> pd.Series:
    """Classify a paired prediction as available only when both predictions exist."""
    available = data[doc_column].notna() & data[algo_column].notna()
    labels = np.where(available, "available", "unavailable")
    return pd.Series(
        pd.Categorical(labels, categories=PREDICTION_PAIR_AVAILABILITY, ordered=True),
        index=data.index,
        name="prediction_pair_availability",
    )


def prediction_source_availability(
    data: pd.DataFrame, *, column: str, source: str
) -> pd.Series:
    """Classify one prediction source independently as available or unavailable."""
    labels = np.where(data[column].notna(), "available", "unavailable")
    return pd.Series(
        pd.Categorical(labels, categories=PREDICTION_PAIR_AVAILABILITY, ordered=True),
        index=data.index,
        name=f"{source}_prediction_availability",
    )


def _summarize_numeric(values: pd.Series) -> str:
    """Return median [IQR] for non-null numeric values."""
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return "-"
    q1, q3 = values.quantile([0.25, 0.75])
    return f"{values.median():.1f} [{q1:.1f}-{q3:.1f}]"


def _summarize_binary(values: pd.Series) -> str:
    """n (%) of the True/present cases, over the rows where the value is known."""
    known = values.dropna()
    if known.empty:
        return "-"
    positives = known.astype(bool).sum()
    return f"{positives} ({100 * positives / len(known):.1f}%)"


def _numeric_test(groups: Sequence[pd.Series]) -> tuple[str, float]:
    """Run a Kruskal-Wallis test on non-empty numeric groups."""
    usable = [pd.to_numeric(g, errors="coerce").dropna() for g in groups]
    usable = [g for g in usable if len(g) > 0]
    if len(usable) < 2 or all(len(g) < 2 for g in usable):
        return "Kruskal-Wallis", float("nan")
    try:
        return "Kruskal-Wallis", stats.kruskal(*usable).pvalue
    except ValueError:  # e.g. every value identical
        return "Kruskal-Wallis", float("nan")


def _categorical_test(
    data: pd.DataFrame, variable: str, group_column: str
) -> tuple[str, float]:
    """Chi-square on the contingency table, or NaN when it is too sparse to be meaningful."""
    table = pd.crosstab(data[variable].astype("object"), data[group_column])
    if table.shape[0] < 2 or table.shape[1] < 2 or table.to_numpy().sum() == 0:
        return "chi-square", float("nan")
    try:
        return "chi-square", stats.chi2_contingency(table).pvalue
    except ValueError:
        return "chi-square", float("nan")


def compare_groups(
    data: pd.DataFrame,
    group: pd.Series | str,
    variables: Sequence[str],
    *,
    binary: Sequence[str] = (),
    descriptive_only: Sequence[str] = (),
    labels: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Summarize variables by group and test group differences.

    Numeric variables use median [IQR]; ``binary`` variables use n (%). ``labels`` controls display
    names.

    Returns one row per variable with group summaries, test name, and p-value. Empty groups use
    ``-``; untestable comparisons use NaN.

    Variables in ``descriptive_only`` are summarized without a row-level test. This is intended
    for shared day-level exposures whose observations are not independent.
    """
    frame = data.copy()
    group_column = group if isinstance(group, str) else "_group"
    if not isinstance(group, str):
        frame[group_column] = group.values

    present = [g for g in frame[group_column].dropna().unique()]
    if hasattr(frame[group_column], "cat"):
        present = [g for g in frame[group_column].cat.categories if g in set(present)]
    counts = frame[group_column].value_counts()
    headers = {g: f"{g} (n={counts.get(g, 0)})" for g in present}

    rows = []
    for variable in variables:
        if variable not in frame.columns:
            continue
        is_binary = variable in binary
        row = {"variable": (labels or {}).get(variable, variable)}
        per_group = []
        for g in present:
            values = frame.loc[frame[group_column] == g, variable]
            row[headers[g]] = (
                _summarize_binary(values) if is_binary else _summarize_numeric(values)
            )
            per_group.append(values)
        if variable in descriptive_only:
            test, p = "descriptive only", float("nan")
        elif is_binary:
            test, p = _categorical_test(
                frame.dropna(subset=[variable]), variable, group_column
            )
        else:
            test, p = _numeric_test(per_group)
        row["test"] = test
        row["p"] = p
        rows.append(row)

    result = pd.DataFrame(rows, columns=["variable", *headers.values(), "test", "p"])
    result["p_holm"] = holm_adjust(result["p"])
    return result
