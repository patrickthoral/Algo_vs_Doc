"""Calibration estimates and patient-bootstrap interval summaries."""

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import (
    PerfectSeparationError,
    PerfectSeparationWarning,
)


METRICS = (
    "calibration_in_the_large",
    "calibration_slope_estimate",
    "observed_expected_ratio",
    "brier_score",
)


def estimates(y, probabilities):
    """Estimate calibration; use clipping only for the log-odds regressions.

    Undefined or separated regression fits return NaN. O/E and Brier score
    retain the original probabilities, including predictions of zero or one.
    """
    y, p = np.asarray(y, dtype=float), np.asarray(probabilities, dtype=float)
    if len(y) == 0 or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("calibration requires nonempty finite observations")
    if not np.isin(y, [0, 1]).all() or ((p < 0) | (p > 1)).any():
        raise ValueError(
            "calibration requires binary outcomes and probabilities in [0, 1]"
        )
    result = dict.fromkeys(METRICS, np.nan)
    result["observed_expected_ratio"] = y.sum() / p.sum() if p.sum() else np.nan
    result["brier_score"] = np.mean((y - p) ** 2)
    if len(np.unique(y)) < 2:
        return result
    clipped = np.clip(p, 1e-6, 1 - 1e-6)
    logit = np.log(clipped / (1 - clipped))
    fits = [("calibration_in_the_large", np.ones((len(y), 1)), logit, 0)]
    if np.ptp(logit) > 0:
        fits.append(("calibration_slope_estimate", sm.add_constant(logit), None, 1))
    for name, design, offset, index in fits:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", PerfectSeparationWarning)
                warnings.simplefilter("error", RuntimeWarning)
                fit = sm.GLM(
                    y, design, offset=offset, family=sm.families.Binomial()
                ).fit()
            if fit.converged and np.isfinite(fit.params[index]):
                result[name] = float(fit.params[index])
        except (
            PerfectSeparationError,
            PerfectSeparationWarning,
            RuntimeWarning,
            np.linalg.LinAlgError,
            ValueError,
        ):
            pass
    return result


def summarize(observed, replicates, requested):
    """Report percentile intervals and failed counts separately for each measure."""
    result = dict(observed)
    for metric in METRICS:
        values = np.array([record[metric] for record in replicates], dtype=float)
        values = values[np.isfinite(values)]
        bounds = (
            np.quantile(values, [0.025, 0.975])
            if len(values) >= 2
            else [np.nan, np.nan]
        )
        if not np.isfinite(observed[metric]):
            bounds = [np.nan, np.nan]
        result[f"{metric}_ci_lower"], result[f"{metric}_ci_upper"] = bounds
        result[f"{metric}_valid_bootstraps"] = len(values)
        result[f"{metric}_failed_bootstraps"] = requested - len(values)
    result["calibration_requested_bootstraps"] = requested
    result["calibration_intercept"] = _display(result, "calibration_in_the_large")
    result["calibration_slope"] = _display(result, "calibration_slope_estimate")
    return result


def _display(result, metric):
    return (
        f"{result[metric]:.3g} "
        f"[{result[metric + '_ci_lower']:.3g}, {result[metric + '_ci_upper']:.3g}]"
    )


def bootstrap(y, probabilities, *, clusters=None, n_bootstraps=1000, random_seed=42):
    """Bootstrap fixed predictions by patient, retaining repeated discharges."""
    from algo_vs_doc import shared_bootstrap

    y, p = np.asarray(y), np.asarray(probabilities)
    observed = estimates(y, p)
    draws = []
    for positions in shared_bootstrap.current_run().positions(
        len(y), n_bootstraps, random_seed, clusters
    ):
        draws.append(estimates(y[positions], p[positions]))
    return summarize(observed, draws, n_bootstraps)


def deciles(y, probabilities):
    """Return risk-quantile summaries, retaining ties and dropping duplicate cut points."""
    frame = pd.DataFrame({"observed": y, "predicted": probabilities}).dropna()
    if frame.predicted.nunique() <= 1:
        frame["group"] = 0
    else:
        frame["group"] = pd.qcut(frame.predicted, 10, duplicates="drop")
    return (
        frame.groupby("group", observed=True)
        .agg(
            predicted=("predicted", "mean"),
            observed=("observed", "mean"),
            n=("observed", "size"),
        )
        .reset_index(drop=True)
    )


def curve_intervals(
    frame,
    *,
    outcome,
    prediction,
    cluster_column,
    n_bootstraps=1000,
    random_seed=42,
    refit=None,
):
    """Patient-bootstrap pointwise LOWESS and fixed-risk-decile intervals.

    LOWESS uses frac=0.5 and no robust iterations for binary outcomes. Predictions
    are evaluated on the original prediction range in every resample, allowing
    boundary extrapolation within individual resamples. Ties retain identical
    decile membership. A refit callback returns new probabilities for a resample
    when predictions were estimated in the evaluation cohort. Curve intervals
    require at least two finite estimates; decile intervals retain the 80%
    valid-replicate requirement. Valid counts are exported. These are pointwise, not
    simultaneous bands. Empirical percentile intervals can collapse to zero in
    event-free groups; they do not establish absence of population risk.
    """
    from algo_vs_doc import shared_bootstrap

    if n_bootstraps < 2:
        raise ValueError("curve intervals require at least two bootstrap samples")
    frame = frame.reset_index(drop=True)
    y = frame[outcome].to_numpy(dtype=float)
    p = frame[prediction].to_numpy(dtype=float)
    estimates(y, p)
    grid = np.linspace(p.min(), p.max(), 101)
    if np.unique(p).size == 1:
        edges = np.array([-np.inf, np.inf])
    else:
        _, edges = pd.qcut(p, 10, retbins=True, duplicates="drop")
        edges[0], edges[-1] = -np.inf, np.inf
    group_count = len(edges) - 1

    def group_summary(outcomes, probabilities):
        ids = pd.cut(probabilities, edges, labels=False, include_lowest=True)
        return [
            np.mean(outcomes[ids == group]) if np.any(ids == group) else np.nan
            for group in range(group_count)
        ]

    observed_groups = np.asarray(group_summary(y, p))
    group_x = np.asarray(group_summary(p, p))
    curves = np.full((n_bootstraps, len(grid)), np.nan)
    groups = np.full((n_bootstraps, group_count), np.nan)
    from tqdm.auto import tqdm

    samples = shared_bootstrap.current_run().positions(
        len(frame), n_bootstraps, random_seed, frame[cluster_column]
    )
    for iteration, indices in enumerate(
        tqdm(
            samples,
            total=n_bootstraps,
            desc="Calibration curve intervals",
            unit="bootstrap",
        )
    ):
        sample = frame.iloc[indices].reset_index(drop=True)
        sampled_p = np.asarray(
            refit(sample) if refit else sample[prediction], dtype=float
        )
        sampled_y = sample[outcome].to_numpy(dtype=float)
        if not np.isfinite(sampled_p).all():
            continue
        curves[iteration] = _smooth(sampled_y, sampled_p, grid)
        groups[iteration] = group_summary(sampled_y, sampled_p)

    def bounds(values, minimum):
        valid = np.isfinite(values).sum(axis=0)
        limits = np.full((2, values.shape[1]), np.nan)
        for index in np.flatnonzero(valid >= minimum):
            limits[:, index] = np.quantile(
                values[np.isfinite(values[:, index]), index], [0.025, 0.975]
            )
        return limits, valid

    curve_bounds, curve_valid = bounds(curves, 2)
    group_bounds, group_valid = bounds(groups, max(2, np.ceil(0.8 * n_bootstraps)))
    return (
        pd.DataFrame(
            dict(
                predicted=grid,
                observed=_smooth(y, p, grid),
                lower=curve_bounds[0],
                upper=curve_bounds[1],
                valid_bootstraps=curve_valid,
            )
        ),
        pd.DataFrame(
            dict(
                predicted=group_x,
                observed=observed_groups,
                lower=group_bounds[0],
                upper=group_bounds[1],
                valid_bootstraps=group_valid,
            )
        ),
    )


def _smooth(y, p, grid):
    if np.unique(p).size < 2:
        return np.where(np.isclose(grid, p[0]), np.mean(y), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        fitted = sm.nonparametric.lowess(y, p, frac=0.5, it=0, xvals=grid)
    return np.where(np.isfinite(fitted), np.clip(fitted, 0, 1), np.nan)


def plot_tables(
    predictions,
    outcome,
    prediction_columns,
    *,
    cluster_column,
    n_bootstraps,
    random_seed,
    combination_specs=None,
):
    """Compute calibration curves, intervals and counts for subsequent rendering.

    Bootstrap count and seed must come from the analysis notebook parameters.
    Combined predictors are refitted within patient resamples.
    """
    from algo_vs_doc import analysis

    combination_specs = combination_specs or {}
    curves, decile_tables, histograms = [], [], []
    for label, column in prediction_columns.items():
        specification = combination_specs.get(column)
        required = [outcome, column, cluster_column]
        if specification:
            required.extend(
                [specification["prediction1"], specification["prediction2"]]
            )
        paired = predictions.dropna(subset=list(dict.fromkeys(required))).copy()
        refit = None
        if specification:

            def refit(sample, specification=specification, column=column):
                return analysis.combine_predictions(
                    sample,
                    outcome=outcome,
                    combined_prediction=column,
                    group_column=cluster_column,
                    random_seed=random_seed,
                    **specification,
                )[column]

            paired[column] = refit(paired)
        curve, groups = curve_intervals(
            paired,
            outcome=outcome,
            prediction=column,
            cluster_column=cluster_column,
            n_bootstraps=n_bootstraps,
            random_seed=random_seed,
            refit=refit,
        )
        metadata = dict(
            prediction_source=label,
            prediction_column=column,
            requested_bootstraps=n_bootstraps,
            random_seed=random_seed,
        )
        curves.append(curve.assign(**metadata))
        decile_tables.append(groups.assign(**metadata))
        y, p = paired[outcome].to_numpy(), paired[column].to_numpy()
        bins = np.linspace(0, 1, 31)
        events, _ = np.histogram(p[y == 1], bins=bins)
        nonevents, _ = np.histogram(p[y == 0], bins=bins)
        histograms.append(
            pd.DataFrame(
                dict(
                    bin_lower=bins[:-1],
                    bin_upper=bins[1:],
                    events=events,
                    nonevents=nonevents,
                )
            ).assign(**metadata)
        )
    return tuple(
        pd.concat(tables, ignore_index=True)
        for tables in (curves, decile_tables, histograms)
    )
