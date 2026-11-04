"""SOFA Coagulation component: platelets (x10^9/L)."""

from __future__ import annotations

import pandas as pd


def score(platelets: pd.Series) -> pd.Series:
    """Score the SOFA Coagulation component from a platelet count in x10^9/L.

    0; <150 -> 1; <100 -> 2; <50 -> 3; <20 -> 4.
    """
    platelets = pd.to_numeric(platelets, errors="coerce")
    sofa_coagulation = pd.Series(0, index=platelets.index)
    sofa_coagulation = sofa_coagulation.mask(platelets < 150, 1)
    sofa_coagulation = sofa_coagulation.mask(platelets < 100, 2)
    sofa_coagulation = sofa_coagulation.mask(platelets < 50, 3)
    sofa_coagulation = sofa_coagulation.mask(platelets < 20, 4)
    return sofa_coagulation
