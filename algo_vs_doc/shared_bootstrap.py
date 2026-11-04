"""Run-scoped patient resampling and combination-prediction reuse.

Samples retain the incoming row and first-seen patient ordering. Different
eligibility sets or orders have distinct plans. Only integer patient draws and
prediction arrays are cached, never whole resampled dataframes. Call begin_run
after loading notebook parameters to prevent reuse across analysis runs.
"""

from hashlib import sha256

import numpy as np
import pandas as pd


class BootstrapRun:
    """Own shared sampling plans and fitted predictions for one analysis run."""

    def __init__(self):
        self.plans = {}
        self.predictions = {}
        self.sampling_runs = 0
        self.model_fits = 0

    def positions(self, size, count, seed, clusters=None):
        """Yield the same stored patient draws to every metric on this cohort."""
        if count < 0:
            raise ValueError("bootstrap count cannot be negative")
        values = np.arange(size) if clusters is None else np.asarray(clusters)
        if len(values) != size or pd.isna(values).any():
            raise ValueError(
                "bootstrap cluster identifiers must be observed and aligned"
            )
        codes, unique = pd.factorize(values, sort=False)
        key = (size, count, seed, sha256(pd.util.hash_array(values).tobytes()).digest())
        if key not in self.plans:
            rng = np.random.default_rng(seed)
            draws = np.empty((count, len(unique)), dtype=np.int32)
            for replicate in range(count):
                draws[replicate] = rng.choice(
                    len(unique), size=len(unique), replace=True
                )
            positions = [np.flatnonzero(codes == code) for code in range(len(unique))]
            self.plans[key] = (draws, positions)
            self.sampling_runs += 1
        draws, positions = self.plans[key]
        for draw in draws:
            yield (
                np.concatenate([positions[code] for code in draw])
                if len(draw)
                else np.array([], dtype=int)
            )

    def combination(self, data, columns, settings, fit):
        """Reuse a fit only when all model inputs, grouping and settings match."""
        frame = data.loc[:, columns].reset_index(drop=True)
        digest = sha256(
            pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes()
        ).digest()
        key = (tuple(map(str, frame.dtypes)), tuple(settings), len(frame), digest)
        if key not in self.predictions:
            self.predictions[key] = np.asarray(fit(), dtype=float).copy()
            self.model_fits += 1
        return self.predictions[key].copy()


_run = BootstrapRun()


def begin_run():
    """Reset caches explicitly at the start of a notebook analysis."""
    global _run
    _run = BootstrapRun()
    return _run


def current_run():
    """Return the shared coordinator for this process."""
    return _run
