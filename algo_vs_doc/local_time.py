"""Interpret analysis timestamps in the study hospitals' local timezone."""

from __future__ import annotations

import pandas as pd

TIMEZONE = "Europe/Amsterdam"


def amsterdam(values: pd.Series) -> pd.Series:
    """Interpret naive wall times in Amsterdam and convert aware times to Amsterdam."""
    parsed = pd.to_datetime(values)
    if parsed.dt.tz is None:
        return parsed.dt.tz_localize(
            TIMEZONE,
            ambiguous="NaT",
            nonexistent="shift_forward",
        )
    return parsed.dt.tz_convert(TIMEZONE)
