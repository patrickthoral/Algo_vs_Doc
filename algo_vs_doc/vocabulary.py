"""Strict vocabulary mappings.

Unknown non-null values raise because they indicate an incomplete mapping. Null values remain null.
"""

from __future__ import annotations

from typing import Collection, Mapping

import pandas as pd


def _plain(value):
    """numpy scalars render as ``np.int64(4)``; report the value the source actually holds."""
    return value.item() if hasattr(value, "item") else value


def unmapped_values(values: pd.Series, known: Collection) -> list:
    """The distinct non-null entries of ``values`` that ``known`` does not cover, as plain values."""
    known = set(known)
    return sorted(
        {_plain(v) for v in pd.Series(values).dropna().unique() if v not in known},
        key=repr,
    )


#: Maximum known values included in an error message.
_MAX_KNOWN_SHOWN = 12


def _describe_known(known: Collection) -> str:
    known = sorted(known, key=repr)
    if len(known) <= _MAX_KNOWN_SHOWN:
        return f"{known}"
    shown = known[:_MAX_KNOWN_SHOWN]
    return f"{shown} ... ({len(known)} values in total)"


def raise_unmapped(unknown: Collection, *, what: str, known: Collection) -> None:
    """Raise for unknown values, listing all unknown and a bounded known set."""
    unknown = list(unknown)
    if not unknown:
        return
    raise KeyError(
        f"{what}: {unknown} not covered by the mapping {_describe_known(known)} - "
        "extend the mapping rather than dropping these values"
    )


def map_strict(values: pd.Series, mapping: Mapping, *, what: str) -> pd.Series:
    """Map known values and raise for unknown non-null values."""
    raise_unmapped(unmapped_values(values, mapping), what=what, known=mapping)
    return pd.Series(values).map(dict(mapping))
