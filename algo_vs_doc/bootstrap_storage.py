"""Optional CSV persistence for bootstrap replicate statistics."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

_output_directory: Path | None = None
_random_seed: int | None = None
_n_bootstraps: int | None = None


def configure(output_directory, *, random_seed: int, n_bootstraps: int) -> None:
    """Enable deterministic CSV persistence for subsequent bootstrap calls."""
    global _output_directory, _random_seed, _n_bootstraps
    _output_directory = Path(output_directory)
    _output_directory.mkdir(parents=True, exist_ok=True)
    _random_seed = int(random_seed)
    _n_bootstraps = int(n_bootstraps)
    pd.DataFrame(
        [{"random_seed": _random_seed, "requested_bootstraps": _n_bootstraps}]
    ).to_csv(_output_directory / "run_metadata.csv", index=False)


def persist(label: str | None, analysis: str, records) -> None:
    """Write one deterministic CSV when persistence has been configured."""
    if _output_directory is None or label is None:
        return
    frame = pd.DataFrame(records).copy()
    if frame.empty:
        return
    if "replicate" not in frame:
        frame.insert(0, "replicate", range(len(frame)))
    frame.insert(0, "analysis", analysis)
    frame.insert(0, "label", label)
    frame["random_seed"] = _random_seed
    safe_label = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
    safe_analysis = re.sub(r"[^a-z0-9]+", "_", analysis.lower()).strip("_")
    frame.to_csv(_output_directory / f"{safe_label}__{safe_analysis}.csv", index=False)
