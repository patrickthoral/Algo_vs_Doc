"""Map non-proprietary hospital specialty values to shared reporting categories."""

from __future__ import annotations

import pandas as pd

from algo_vs_doc import vocabulary

SPECIALTY_LABELS = {
    "cardiothoracic_surgery": "Cardiothoracic Surgery",
    "general_surgery": "General Surgery",
    "neurosurgery": "Neurosurgery",
    "otolaryngology": "Otolaryngology",
    "other_surgery": "Other Surgery",
    "internal_medicine": "Internal Medicine",
    "gastroenterology": "Gastroenterology",
    "cardiology": "Cardiology",
    "neurology": "Neurology",
    "pulmonary_medicine": "Pulmonary Medicine",
    "other_medical": "Other Medical",
    "unspecified": "Unspecified",
}

# These identifiers are non-proprietary values exported by each hospital's private schema layer.
# Source tables, columns, joins, and extraction logic remain private.
SOURCE_SPECIALTY_CATEGORIES = {
    "LUMC": {
        1: "cardiothoracic_surgery",
        2: "general_surgery",
        3: "neurosurgery",
        4: "neurology",
        5: "internal_medicine",
        6: "internal_medicine",
        7: "internal_medicine",
        9: "cardiology",
        10: "pulmonary_medicine",
        11: "gastroenterology",
        12: "otolaryngology",
        14: "other_surgery",
        15: "other_surgery",
        16: "other_medical",
    },
    "VUmc": {
        30200: "otolaryngology",
        30300: "general_surgery",
        30303: "general_surgery",
        30351: "general_surgery",
        30354: "general_surgery",
        30361: "cardiothoracic_surgery",
        30685: "other_surgery",
        30800: "neurosurgery",
        31300: "internal_medicine",
        31353: "internal_medicine",
        31354: "internal_medicine",
        31355: "internal_medicine",
        31358: "internal_medicine",
        31800: "gastroenterology",
        32000: "cardiology",
        32200: "pulmonary_medicine",
        32800: "cardiothoracic_surgery",
        32900: "other_medical",
        33050: "neurology",
        33085: "neurology",
        110000: "other_surgery",
        841700: "unspecified",
    },
}

SURGICAL_SPECIALTIES = (
    "cardiothoracic_surgery",
    "general_surgery",
    "neurosurgery",
    "otolaryngology",
    "other_surgery",
)

MEDICAL_SPECIALTIES = (
    "internal_medicine",
    "gastroenterology",
    "cardiology",
    "neurology",
    "pulmonary_medicine",
    "other_medical",
)

BASELINE_GROUPS = {
    "Surgical": SURGICAL_SPECIALTIES,
    "Medical": MEDICAL_SPECIALTIES,
    "Unspecified": ("unspecified",),
}


def categorize(values: pd.Series, *, hospital: str) -> pd.Series:
    """Map one hospital's source specialty identifiers to canonical categories."""
    try:
        mapping = SOURCE_SPECIALTY_CATEGORIES[hospital]
    except KeyError:
        raise KeyError(
            f"no specialty mapping for hospital {hospital!r} - "
            f"known: {sorted(SOURCE_SPECIALTY_CATEGORIES)}"
        ) from None

    normalized = values.astype("object").copy()
    numeric = pd.to_numeric(values, errors="coerce")
    integer_like = numeric.notna() & numeric.mod(1).eq(0)
    normalized.loc[integer_like] = numeric.loc[integer_like].astype("int64")

    result = normalized.map(mapping)
    result.loc[values.isna()] = "unspecified"
    vocabulary.raise_unmapped(
        vocabulary.unmapped_values(normalized, mapping),
        what=f"{hospital} specialty value(s)",
        known=mapping,
    )
    return result


def categorize_optional(values: pd.Series, *, hospital: str) -> pd.Series:
    """Map known specialty identifiers while preserving missing source values."""
    result = categorize(values, hospital=hospital)
    return result.mask(values.isna())


def validate(values: pd.Series) -> None:
    """Raise when values fall outside the public specialty vocabulary."""
    vocabulary.raise_unmapped(
        vocabulary.unmapped_values(values, SPECIALTY_LABELS),
        what="specialty category or categories",
        known=SPECIALTY_LABELS,
    )


def labels(values: pd.Series) -> pd.Series:
    """Return display labels for canonical specialty values."""
    validate(values)
    return values.map(SPECIALTY_LABELS)


def baseline_counts(frame: pd.DataFrame, *, column: str = "specialty") -> pd.Series:
    """Return ordered specialty counts for the baseline-characteristics table."""
    validate(frame[column])
    counts = frame[column].value_counts()
    rows: dict[str, int] = {}
    for group, categories in BASELINE_GROUPS.items():
        rows[group] = int(counts.reindex(categories, fill_value=0).sum())
        for category in categories:
            rows[SPECIALTY_LABELS[category]] = int(counts.get(category, 0))
    return pd.Series(rows, dtype="int64")
