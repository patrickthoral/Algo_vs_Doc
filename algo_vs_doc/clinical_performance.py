"""Clinically interpretable operating points for probabilistic predictions."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from math import ceil

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_curve,
)
from tqdm.auto import tqdm

from algo_vs_doc import bootstrap_storage, shared_bootstrap

INTERVAL_METRICS = (
    "sensitivity",
    "specificity",
    "positive_predictive_value",
    "negative_predictive_value",
    "f1",
)


def _paired(
    frame: pd.DataFrame,
    prediction: str,
    outcome: str,
    additional_columns: Iterable[str] = (),
) -> pd.DataFrame:
    """Return complete prediction/outcome pairs after validating probabilities."""
    columns = [prediction, outcome, *additional_columns]
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise KeyError(f"performance analysis is missing {missing}")
    paired = frame[columns].dropna().copy()
    paired[prediction] = pd.to_numeric(paired[prediction], errors="coerce")
    invalid = paired[prediction].isna() | ~paired[prediction].between(0, 1)
    if invalid.any():
        raise ValueError(f"{prediction} must contain probabilities from 0 through 1")
    paired[outcome] = paired[outcome].astype(bool)
    return paired


def _classification_metrics(observed: np.ndarray, flagged: np.ndarray) -> dict:
    tn, fp, fn, tp = confusion_matrix(observed, flagged, labels=[False, True]).ravel()
    return {
        "true_positive": int(tp),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_negative": int(tn),
        "sensitivity": tp / (tp + fn) if tp + fn else np.nan,
        "specificity": tn / (tn + fp) if tn + fp else np.nan,
        "positive_predictive_value": tp / (tp + fp) if tp + fp else np.nan,
        "negative_predictive_value": tn / (tn + fn) if tn + fn else np.nan,
        "f1": f1_score(observed, flagged, zero_division=0),
    }


def _cluster_samples(
    paired: pd.DataFrame,
    cluster_column: str,
    n_bootstraps: int,
    progress_desc: str | None = None,
    random_seed: int = 42,
) -> Iterable[pd.DataFrame]:
    samples = shared_bootstrap.current_run().positions(
        len(paired),
        n_bootstraps,
        random_seed,
        paired[cluster_column],
    )
    for positions in tqdm(
        samples,
        total=n_bootstraps,
        desc=progress_desc,
        disable=progress_desc is None,
        unit="bootstrap",
        leave=False,
    ):
        yield paired.iloc[positions].reset_index(drop=True)


def _add_percentile_intervals(result: dict, bootstrap_metrics: list[dict]) -> None:
    for metric in INTERVAL_METRICS:
        values = np.asarray([row[metric] for row in bootstrap_metrics], dtype=float)
        values = values[np.isfinite(values)]
        result[f"{metric}_ci_lower"] = (
            np.quantile(values, 0.025) if len(values) else np.nan
        )
        result[f"{metric}_ci_upper"] = (
            np.quantile(values, 0.975) if len(values) else np.nan
        )
        result[f"{metric}_valid_bootstraps"] = len(values)


def threshold_metrics(
    frame: pd.DataFrame,
    *,
    prediction: str,
    outcome: str,
    threshold: float,
    cluster_column: str | None = None,
    n_bootstraps: int = 0,
    progress_desc: str | None = None,
    random_seed: int = 42,
) -> dict:
    """Return confusion-matrix metrics for a prespecified probability threshold."""
    additional = [cluster_column] if cluster_column else []
    paired = _paired(frame, prediction, outcome, additional)
    observed = paired[outcome].to_numpy()
    flagged = paired[prediction].ge(threshold).to_numpy()
    result = {
        "threshold": float(threshold),
        "n": len(paired),
        "flagged_n": int(flagged.sum()),
        "alert_rate": float(flagged.mean()) if len(flagged) else np.nan,
        **_classification_metrics(observed, flagged),
    }
    if n_bootstraps:
        if not cluster_column:
            raise ValueError("cluster_column is required for bootstrap intervals")
        bootstrap_metrics = []
        for sample in _cluster_samples(
            paired, cluster_column, n_bootstraps, progress_desc, random_seed
        ):
            bootstrap_metrics.append(
                _classification_metrics(
                    sample[outcome].to_numpy(),
                    sample[prediction].ge(threshold).to_numpy(),
                )
            )
        _add_percentile_intervals(result, bootstrap_metrics)
        bootstrap_storage.persist(progress_desc, "threshold_metrics", bootstrap_metrics)
    return result


def youden_threshold(frame: pd.DataFrame, *, prediction: str, outcome: str) -> float:
    """Choose the empirical threshold maximizing sensitivity plus specificity minus one."""
    paired = _paired(frame, prediction, outcome)
    if paired[outcome].nunique() < 2:
        raise ValueError("Youden threshold requires both outcome classes")
    false_positive_rate, true_positive_rate, thresholds = roc_curve(
        paired[outcome], paired[prediction]
    )
    finite = np.isfinite(thresholds)
    index = np.argmax((true_positive_rate - false_positive_rate)[finite])
    return float(thresholds[finite][index])


def alert_rate_table(
    frame: pd.DataFrame,
    *,
    prediction: str,
    outcome: str,
    alert_rates: Iterable[float] = (0.05, 0.10, 0.15),
    cluster_column: str | None = None,
    n_bootstraps: int = 0,
    progress_desc: str | None = None,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Report events captured when flagging the highest-risk fixed fraction of stays.

    Ranking uses stable row order to make the requested number flagged deterministic when scores tie.
    The table therefore evaluates an operational alert budget rather than a reusable score threshold.
    """
    additional = [cluster_column] if cluster_column else []
    bootstrap_paired = _paired(frame, prediction, outcome, additional)
    paired = bootstrap_paired.sort_values(prediction, ascending=False, kind="stable")
    rates = list(alert_rates)
    if any(not 0 < rate <= 1 for rate in rates):
        raise ValueError("alert rates must be greater than 0 and at most 1")
    if n_bootstraps and not cluster_column:
        raise ValueError("cluster_column is required for bootstrap intervals")

    bootstrap_by_rate = {rate: [] for rate in rates}
    if n_bootstraps:
        for sample in _cluster_samples(
            bootstrap_paired, cluster_column, n_bootstraps, progress_desc, random_seed
        ):
            ranked = sample.sort_values(prediction, ascending=False, kind="stable")
            for rate in rates:
                sample_flagged_n = min(ceil(rate * len(ranked)), len(ranked))
                bootstrap_by_rate[rate].append(
                    _classification_metrics(
                        ranked[outcome].to_numpy(),
                        np.arange(len(ranked)) < sample_flagged_n,
                    )
                )

    events = int(paired[outcome].sum())
    rows = []
    for rate in rates:
        flagged_n = min(ceil(rate * len(paired)), len(paired))
        flagged = np.arange(len(paired)) < flagged_n
        metrics = _classification_metrics(paired[outcome].to_numpy(), flagged)
        threshold = (
            float(paired.iloc[flagged_n - 1][prediction]) if flagged_n else np.nan
        )
        row = {
            "target_alert_rate": rate,
            "threshold": threshold,
            "n": len(paired),
            "flagged_n": flagged_n,
            "observed_alert_rate": flagged_n / len(paired) if len(paired) else np.nan,
            "events_n": events,
            "events_captured_n": metrics["true_positive"],
            **metrics,
        }
        if n_bootstraps:
            _add_percentile_intervals(row, bootstrap_by_rate[rate])
        rows.append(row)
    bootstrap_storage.persist(
        progress_desc,
        "alert_rate_metrics",
        [
            {"replicate": replicate, "target_alert_rate": rate, **metrics}
            for rate, samples in bootstrap_by_rate.items()
            for replicate, metrics in enumerate(samples)
        ],
    )
    return pd.DataFrame(rows)


def risk_category_events(
    frame: pd.DataFrame,
    *,
    category_columns: Mapping[str, str],
    outcome: str,
) -> pd.DataFrame:
    """Count observed events within each named predictor's risk categories."""
    rows = []
    for source, column in category_columns.items():
        paired = frame[[column, outcome]].dropna()
        for category in ("low", "medium", "high"):
            selected = paired[column].eq(category)
            n = int(selected.sum())
            events = int(paired.loc[selected, outcome].astype(bool).sum())
            rows.append(
                {
                    "prediction_source": source,
                    "risk_category": category,
                    "n": n,
                    "events_n": events,
                    "event_rate": events / n if n else np.nan,
                }
            )
    return pd.DataFrame(rows)


def average_precision(frame: pd.DataFrame, *, prediction: str, outcome: str) -> float:
    """Return average precision for complete prediction/outcome pairs."""
    paired = _paired(frame, prediction, outcome)
    if paired.empty:
        return np.nan
    return float(average_precision_score(paired[outcome], paired[prediction]))


def average_precision_metrics(
    frame: pd.DataFrame,
    *,
    prediction: str,
    outcome: str,
    cluster_column: str | None = None,
    n_bootstraps: int = 0,
    progress_desc: str | None = None,
    random_seed: int = 42,
) -> dict:
    """Return average precision with an optional patient-clustered percentile interval."""
    additional = [cluster_column] if cluster_column else []
    paired = _paired(frame, prediction, outcome, additional)
    result = {
        "n": len(paired),
        "events_n": int(paired[outcome].sum()),
        "average_precision": (
            float(average_precision_score(paired[outcome], paired[prediction]))
            if paired[outcome].nunique() == 2
            else np.nan
        ),
    }
    if not n_bootstraps:
        return result
    if not cluster_column:
        raise ValueError("cluster_column is required for bootstrap intervals")

    estimates = []
    for sample in _cluster_samples(
        paired, cluster_column, n_bootstraps, progress_desc, random_seed
    ):
        if sample[outcome].nunique() < 2:
            continue
        estimates.append(average_precision_score(sample[outcome], sample[prediction]))
    result["average_precision_ci_lower"] = (
        np.quantile(estimates, 0.025) if estimates else np.nan
    )
    result["average_precision_ci_upper"] = (
        np.quantile(estimates, 0.975) if estimates else np.nan
    )
    result["average_precision_valid_bootstraps"] = len(estimates)
    bootstrap_storage.persist(
        progress_desc,
        "average_precision",
        [{"value": value} for value in estimates],
    )
    return result
