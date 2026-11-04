"""SOFA Liver component: bilirubin (umol/L)."""

from __future__ import annotations

import pandas as pd


def score(bilirubin: pd.Series) -> pd.Series:
    """Score the SOFA Liver component from a bilirubin value in umol/L.

    0; >=20 -> 1; >32 -> 2; >101 -> 3; >204 -> 4. Categories are applied in ascending order so
    each later ``.mask()`` correctly overwrites with the more severe category.
    """
    bilirubin = pd.to_numeric(bilirubin, errors="coerce")
    sofa_liver = pd.Series(0, index=bilirubin.index)
    sofa_liver = sofa_liver.mask(bilirubin >= 20, 1)
    sofa_liver = sofa_liver.mask(bilirubin > 32, 2)
    sofa_liver = sofa_liver.mask(bilirubin > 101, 3)
    sofa_liver = sofa_liver.mask(bilirubin > 204, 4)
    return sofa_liver
