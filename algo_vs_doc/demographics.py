"""Map non-proprietary demographic source values to shared categories."""

from __future__ import annotations

import pandas as pd

from algo_vs_doc import vocabulary

SEX_CATEGORIES = {
    "LUMC": {
        "female": "female",
        "male": "male",
    },
    "VUmc": {
        "V": "female",
        "M": "male",
    },
}


def categorize_sex(values: pd.Series, *, hospital: str) -> pd.Series:
    """Map one hospital's source sex values to canonical categories."""
    try:
        mapping = SEX_CATEGORIES[hospital]
    except KeyError:
        raise KeyError(
            f"no sex mapping for hospital {hospital!r} - "
            f"known: {sorted(SEX_CATEGORIES)}"
        ) from None

    vocabulary.raise_unmapped(
        vocabulary.unmapped_values(values, mapping),
        what=f"{hospital} sex value(s)",
        known=mapping,
    )
    return values.map(mapping)
