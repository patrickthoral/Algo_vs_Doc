"""Paths for local interim tables exchanged between private ETL and shared scoring."""

from __future__ import annotations

import os

from algo_vs_doc.cohorts import canonical_hospital


def interim_path(project_root: str, hospital: str, filename: str) -> str:
    """Return ``<project_root>/data/interim/<hospital>/<filename>``.

    Capitalization variants are normalized. This function does not create directories.
    """
    hospital = canonical_hospital(hospital)
    return os.path.join(project_root, "data", "interim", hospital, filename)
