"""Load named hospital cohorts from the flat analysis CSV files.

Each Create Analysis Dataset run writes one identifier-reduced CSV. This module loads those files, verifies their
hospital labels, and provides the individual and pooled cohorts used by downstream notebooks.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping, Sequence

import pandas as pd

HOSPITALS = ("VUmc", "LUMC")
COMBINED = "Combined"
COHORT_ORDER = (*HOSPITALS, COMBINED)
_HOSPITALS_BY_CASEFOLD = {hospital.casefold(): hospital for hospital in HOSPITALS}


def canonical_hospital(hospital: str) -> str:
    """Return the canonical hospital label, accepting capitalization variants."""
    try:
        return _HOSPITALS_BY_CASEFOLD[hospital.casefold()]
    except (AttributeError, KeyError) as error:
        raise ValueError(
            f"hospital must be one of {HOSPITALS}, ignoring case; got {hospital!r}"
        ) from error


def canonical_cohort(cohort: str) -> str:
    """Return a canonical hospital or Combined label, ignoring case."""
    labels = COHORT_ORDER
    try:
        return {label.casefold(): label for label in labels}[cohort.casefold()]
    except (AttributeError, KeyError) as error:
        raise ValueError(
            f"cohort must be one of {labels}, ignoring case; got {cohort!r}"
        ) from error


def dataset_path(data_dir: str, hospital: str) -> str:
    """Return `data_dir/<hospital>/algo_doc_dataset.csv` using the canonical label."""
    hospital = canonical_hospital(hospital)
    return os.path.join(data_dir, hospital, "algo_doc_dataset.csv")


def load(data_dir: str, hospital: str, **read_csv_kwargs) -> pd.DataFrame:
    """Load one hospital dataset and verify its canonical hospital label."""
    hospital = canonical_hospital(hospital)
    path = dataset_path(data_dir, hospital)
    frame = pd.read_csv(path, **read_csv_kwargs)
    if "hospital" not in frame.columns:
        raise KeyError(
            f"{path} has no 'hospital' column - rerun the Create Analysis Dataset notebook"
        )

    labels = set(frame["hospital"].dropna().unique())
    if frame["hospital"].isna().any() or labels != {hospital}:
        raise ValueError(
            f"{path} is labelled {sorted(labels)} but was loaded as {hospital!r} - "
            "the wrong file is at that path; its hospital label may be incomplete"
        )
    return frame


def combine(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Pool hospital datasets while retaining the union of their columns."""
    return pd.concat(list(frames), ignore_index=True, sort=False)


def load_all(
    data_dir: str,
    *,
    hospitals: Sequence[str] = HOSPITALS,
    include_combined: bool = True,
    **read_csv_kwargs,
) -> dict[str, pd.DataFrame]:
    """Load each requested hospital and optionally add the pooled cohort."""
    canonical = [canonical_hospital(hospital) for hospital in hospitals]
    loaded = {
        hospital: load(data_dir, hospital, **read_csv_kwargs) for hospital in canonical
    }
    if include_combined and len(loaded) > 1:
        loaded[COMBINED] = combine(loaded.values())
    return loaded


def split_by_hospital(cohort: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split a cohort into hospital frames in first-appearance order."""
    return {
        hospital: cohort[cohort["hospital"] == hospital]
        for hospital in cohort["hospital"].unique()
    }


def apply_to_cohorts(
    cohorts: Mapping[str, pd.DataFrame],
    function,
    *args,
    **kwargs,
) -> dict:
    """Apply one function to each cohort and return results under the same names."""
    return {name: function(frame, *args, **kwargs) for name, frame in cohorts.items()}
