"""Summarize mapped life-support exposure by stay and category."""

from __future__ import annotations

from typing import Mapping, Sequence

import pandas as pd

from algo_vs_doc.scoring.concepts import (
    DEVICE_CONCEPTS,
    VASOPRESSOR_DRUG_CONCEPTS,
    VASOPRESSOR_INOTROPE_FLAG,
)

#: Roll-up flags derived from device concepts.
COMBINED_CONCEPTS = {
    "ventilation": ("invasive_ventilation", "non_invasive_ventilation")
}


def compute_durations(
    events: pd.DataFrame, *, icu_stay_id_col: str = "icu_stay_id"
) -> pd.DataFrame:
    """Return first-to-last event duration per stay and support category.

    Missing categories remain null.
    """
    summary = (
        events.groupby([icu_stay_id_col, "category"])["measured_at"]
        .agg(first_measured_at="min", last_measured_at="max")
        .reset_index()
    )
    summary["duration"] = summary["last_measured_at"] - summary["first_measured_at"]
    return summary.pivot_table(
        index=icu_stay_id_col, columns="category", values="duration", aggfunc="first"
    ).reset_index()


def combine_duration_sources(
    *sources: pd.DataFrame, icu_stay_id_col: str = "icu_stay_id"
) -> pd.DataFrame:
    """Combine duration frames using the longest value per stay and category."""
    combined = sources[0]
    for source in sources[1:]:
        combined = combined.merge(
            source, on=icu_stay_id_col, how="outer", suffixes=("_a", "_b")
        )
        shared_categories = {
            col[:-2]
            for col in combined.columns
            if col.endswith("_a") or col.endswith("_b")
        }
        for category in shared_categories:
            combined[category] = combined[[f"{category}_a", f"{category}_b"]].max(
                axis=1
            )
            combined = combined.drop(columns=[f"{category}_a", f"{category}_b"])
    return combined


def presence_flags(
    device_exposure: pd.DataFrame,
    icu_stay_id: pd.Series,
    *,
    concepts: frozenset[str] = DEVICE_CONCEPTS,
    combined: Mapping[str, Sequence[str]] = COMBINED_CONCEPTS,
) -> pd.DataFrame:
    """Return device-presence flags for every requested stay.

    Missing exposure rows map to false. ``combined`` defines roll-up flags such as ventilation.
    """
    present = (
        device_exposure.assign(_present=True)
        .pivot_table(
            index="icu_stay_id",
            columns="device_concept_name",
            values="_present",
            aggfunc="any",
        )
        .astype("boolean")
        .fillna(False)
        .astype(bool)
    )
    present = present.reindex(
        index=icu_stay_id, columns=sorted(concepts), fill_value=False
    )
    for name, parts in combined.items():
        available = [p for p in parts if p in present.columns]
        if available:
            present[name] = present[available].any(axis=1)
    return present.reset_index().rename(columns={"index": "icu_stay_id"})


def vasopressor_inotrope_flags(
    drug_exposure: pd.DataFrame,
    icu_stay_id: pd.Series,
    *,
    drug_concepts: frozenset[str] = VASOPRESSOR_DRUG_CONCEPTS,
) -> pd.DataFrame:
    """Return a vasopressor/inotrope exposure flag for every requested stay."""
    present = (
        drug_exposure.loc[
            drug_exposure["drug_concept_name"].isin(drug_concepts), ["icu_stay_id"]
        ]
        .drop_duplicates()
        .assign(**{VASOPRESSOR_INOTROPE_FLAG: True})
    )
    result = pd.DataFrame({"icu_stay_id": icu_stay_id}).merge(
        present, on="icu_stay_id", how="left"
    )
    result[VASOPRESSOR_INOTROPE_FLAG] = (
        result[VASOPRESSOR_INOTROPE_FLAG].astype("boolean").fillna(False).astype(bool)
    )
    return result
