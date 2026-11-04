import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc,
    average_precision_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold
from itertools import combinations
from tqdm.auto import tqdm

from algo_vs_doc import (
    bootstrap_storage,
    calibration,
    clinical_performance,
    risk_factors,
    shared_bootstrap,
)


def _two_sided_bootstrap_p(values) -> float:
    """Return a finite-simulation two-sided bootstrap p-value."""
    sampled = np.asarray(values)
    lower_tail = int(np.count_nonzero(sampled <= 0))
    upper_tail = int(np.count_nonzero(sampled >= 0))
    return min(1.0, 2 * (min(lower_tail, upper_tail) + 1) / (len(sampled) + 1))


def combine_predictions(
    data,
    prediction1,
    prediction2,
    outcome,
    combined_prediction,
    *,
    group_column="patient_id",
    n_splits=5,
    prediction1_is_ordinal=False,
    random_seed=42,
):
    """Return grouped out-of-fold predictions, reusing identical fits within a run."""
    frame = data.copy()
    if group_column not in frame:
        frame[group_column] = frame.index
    columns = [prediction1, prediction2, outcome, group_column]
    values = shared_bootstrap.current_run().combination(
        frame,
        columns,
        (n_splits, prediction1_is_ordinal, random_seed),
        lambda: _fit_combined_predictions(
            frame,
            prediction1,
            prediction2,
            outcome,
            combined_prediction,
            group_column=group_column,
            n_splits=n_splits,
            prediction1_is_ordinal=prediction1_is_ordinal,
            random_seed=random_seed,
        )[combined_prediction],
    )
    result = data.copy()
    result[combined_prediction] = values
    return result


def _fit_combined_predictions(
    data,
    prediction1,
    prediction2,
    outcome,
    combined_prediction,
    *,
    group_column="patient_id",
    n_splits=5,
    prediction1_is_ordinal=False,
    random_seed=42,
):
    """Combine two predictions using patient-grouped out-of-fold logistic regression.

    An ordinal first prediction is represented by indicator variables, with its
    lowest observed level as the reference. This avoids assuming equal spacing
    between ordered categories.
    """
    result = data.copy()
    result[combined_prediction] = np.nan
    mask = result[[prediction1, prediction2, outcome]].notna().all(axis=1)
    paired = result.loc[mask, [prediction1, prediction2, outcome]].copy()
    if paired.empty or paired[outcome].nunique() < 2:
        return result

    prediction2_probability = paired[prediction2].clip(0.000001, 0.999999)
    prediction2_logit = np.log(
        prediction2_probability / (1 - prediction2_probability)
    ).rename("log2")
    if prediction1_is_ordinal:
        ordered = pd.Categorical(
            paired[prediction1], categories=["low", "medium", "high"], ordered=True
        )
        category_features = pd.get_dummies(
            ordered, prefix="physician", drop_first=True, dtype=float
        ).set_axis(paired.index)
        features = pd.concat([category_features, prediction2_logit], axis=1)
    else:
        prediction1_probability = paired[prediction1].clip(0.000001, 0.999999)
        prediction1_logit = np.log(
            prediction1_probability / (1 - prediction1_probability)
        ).rename("log1")
        features = pd.concat([prediction1_logit, prediction2_logit], axis=1)
    observed = paired[outcome].astype(bool)
    groups = (
        result.loc[mask, group_column]
        if group_column in result
        else pd.Series(result.index[mask], index=paired.index)
    )
    outcome_groups = pd.DataFrame(
        {"outcome": observed.to_numpy(), "group": groups.to_numpy()}
    )
    maximum_splits = min(
        n_splits,
        int(outcome_groups.groupby("outcome")["group"].nunique().min()),
        int(groups.nunique()),
    )
    if maximum_splits < 2:
        return result

    splitter = StratifiedGroupKFold(
        n_splits=maximum_splits, shuffle=True, random_state=random_seed
    )
    out_of_fold = pd.Series(np.nan, index=paired.index, dtype=float)
    for train_positions, test_positions in splitter.split(
        features, observed, groups=groups
    ):
        training_outcomes = observed.iloc[train_positions]
        if training_outcomes.nunique() < 2:
            out_of_fold.iloc[test_positions] = float(training_outcomes.mean())
            continue
        model = LogisticRegression(random_state=random_seed).fit(
            features.iloc[train_positions], training_outcomes
        )
        out_of_fold.iloc[test_positions] = model.predict_proba(
            features.iloc[test_positions]
        )[:, 1]
    result.loc[paired.index, combined_prediction] = out_of_fold
    return result


def nested_bootstrap_combined_auc(
    data,
    *,
    prediction1,
    prediction2,
    outcome,
    combined_prediction,
    comparison_predictions,
    group_column="patient_id",
    n_splits=5,
    n_bootstraps=1000,
    curve="roc",
    alert_rates=(0.05, 0.10, 0.15),
    prediction1_is_ordinal=False,
    progress_desc=None,
    random_seed=42,
):
    """Refit grouped cross-validation within each patient-bootstrap replicate."""
    required = {
        prediction1,
        prediction2,
        outcome,
        group_column,
        *comparison_predictions.values(),
    }
    complete = data.dropna(subset=list(required)).copy()
    if complete[outcome].nunique() < 2:
        raise ValueError("nested combination bootstrap requires both outcome classes")

    observed = combine_predictions(
        complete,
        prediction1,
        prediction2,
        outcome,
        combined_prediction,
        group_column=group_column,
        n_splits=n_splits,
        prediction1_is_ordinal=prediction1_is_ordinal,
        random_seed=random_seed,
    )
    observed_scores = {
        metric: calculate_score(
            metric, observed[outcome], observed[combined_prediction]
        )
        for metric in ("roc", "prc")
    }
    observed_calibration = calibration.estimates(
        observed[outcome], observed[combined_prediction]
    )
    calibration_samples = []
    observed_score = observed_scores[curve]
    observed_alerts = clinical_performance.alert_rate_table(
        observed,
        prediction=combined_prediction,
        outcome=outcome,
        alert_rates=alert_rates,
    )
    observed_differences = {
        label: observed_score
        - calculate_score(curve, observed[outcome], observed[column])
        for label, column in comparison_predictions.items()
    }

    score_samples = {"roc": [], "prc": []}
    alert_samples = {rate: [] for rate in alert_rates}
    differences = {label: [] for label in comparison_predictions}
    clusters = complete[group_column].reset_index(drop=True)
    complete = complete.reset_index(drop=True)
    for positions in tqdm(
        shared_bootstrap.current_run().positions(
            len(complete), n_bootstraps, random_seed, clusters
        ),
        total=n_bootstraps,
        desc=progress_desc,
        disable=progress_desc is None,
    ):
        sample = complete.iloc[positions].reset_index(drop=True)
        if sample[outcome].nunique() < 2:
            continue
        fitted = combine_predictions(
            sample,
            prediction1,
            prediction2,
            outcome,
            combined_prediction,
            group_column=group_column,
            n_splits=n_splits,
            prediction1_is_ordinal=prediction1_is_ordinal,
            random_seed=random_seed,
        )
        if fitted[combined_prediction].isna().any():
            continue
        fitted_scores = {
            metric: calculate_score(
                metric, fitted[outcome], fitted[combined_prediction]
            )
            for metric in ("roc", "prc")
        }
        for metric, value in fitted_scores.items():
            score_samples[metric].append(value)
        calibration_samples.append(
            calibration.estimates(fitted[outcome], fitted[combined_prediction])
        )
        combined_score = fitted_scores[curve]
        fitted_alerts = clinical_performance.alert_rate_table(
            fitted,
            prediction=combined_prediction,
            outcome=outcome,
            alert_rates=alert_rates,
        )
        for row in fitted_alerts.to_dict("records"):
            alert_samples[row["target_alert_rate"]].append(row)
        for label, column in comparison_predictions.items():
            agent_2_score = calculate_score(curve, fitted[outcome], fitted[column])
            differences[label].append(combined_score - agent_2_score)

    if not score_samples[curve]:
        raise ValueError("nested combination bootstrap produced no valid replicates")
    performance = {}
    for metric, values in score_samples.items():
        sampled = np.asarray(values)
        name = f"au{metric}"
        performance[name] = observed_scores[metric]
        performance[f"{name}_confidence_interval_lower"] = np.quantile(sampled, 0.025)
        performance[f"{name}_confidence_interval_upper"] = np.quantile(sampled, 0.975)
        performance[f"{name}_valid_bootstraps"] = len(sampled)
    performance["valid_bootstraps"] = performance[f"au{curve}_valid_bootstraps"]

    performance.update(
        calibration.summarize(observed_calibration, calibration_samples, n_bootstraps)
    )
    comparison_rows = []
    for label, values in differences.items():
        sampled = np.asarray(values)
        p_value = _two_sided_bootstrap_p(sampled)
        comparison_rows.append(
            {
                "agent_2": label,
                "difference_agent_1_minus_agent_2": observed_differences[label],
                "confidence_interval_lower": np.quantile(sampled, 0.025),
                "confidence_interval_upper": np.quantile(sampled, 0.975),
                "p": p_value,
                "valid_bootstraps": len(sampled),
            }
        )
    for index, row in observed_alerts.iterrows():
        samples = alert_samples[row["target_alert_rate"]]
        for metric in clinical_performance.INTERVAL_METRICS:
            values = np.asarray([sample[metric] for sample in samples], dtype=float)
            values = values[np.isfinite(values)]
            observed_alerts.loc[index, f"{metric}_ci_lower"] = (
                np.quantile(values, 0.025) if len(values) else np.nan
            )
            observed_alerts.loc[index, f"{metric}_ci_upper"] = (
                np.quantile(values, 0.975) if len(values) else np.nan
            )
            observed_alerts.loc[index, f"{metric}_valid_bootstraps"] = len(values)
    bootstrap_rows = [
        {"agent_2": label, "bootstrap_difference_agent_1_minus_agent_2": value}
        for label, values in differences.items()
        for value in values
    ]
    nested_records = [
        {"replicate": replicate, "metric": f"au{metric}", "value": value}
        for metric, values in score_samples.items()
        for replicate, value in enumerate(values)
    ]
    for rate, samples in alert_samples.items():
        nested_records.extend(
            {
                "replicate": replicate,
                "metric": metric,
                "target_alert_rate": rate,
                "value": sample[metric],
            }
            for replicate, sample in enumerate(samples)
            for metric in clinical_performance.INTERVAL_METRICS
        )
    nested_records.extend(
        {
            "replicate": replicate,
            "metric": "auc_difference",
            "agent_2": label,
            "value": value,
        }
        for label, values in differences.items()
        for replicate, value in enumerate(values)
    )
    bootstrap_storage.persist(progress_desc, "nested_combined", nested_records)
    return (
        performance,
        pd.DataFrame(comparison_rows),
        observed_alerts,
        pd.DataFrame(bootstrap_rows),
    )


def _bootstrap_positions(size, rng, clusters=None):
    """Sample row positions, retaining every row from each sampled cluster."""
    if clusters is None:
        return rng.randint(0, size, size)
    cluster_values = np.asarray(clusters)
    if pd.isna(cluster_values).any():
        raise ValueError("bootstrap cluster identifiers cannot be missing")
    unique_clusters = pd.unique(cluster_values)
    sampled_clusters = rng.choice(
        unique_clusters, size=len(unique_clusters), replace=True
    )
    return np.concatenate(
        [np.flatnonzero(cluster_values == cluster) for cluster in sampled_clusters]
    )


def bootstrap_discrimination(
    n,
    y,
    prob_y,
    *,
    clusters=None,
    progress_desc=None,
    random_seed=42,
):
    """Bootstrap AUROC and average-precision AUPRC using the same resamples."""
    y = pd.Series(y).reset_index(drop=True)
    prob_y = pd.Series(prob_y).reset_index(drop=True)
    if clusters is not None:
        clusters = pd.Series(clusters).reset_index(drop=True)
    observed = {metric: calculate_score(metric, y, prob_y) for metric in ("roc", "prc")}
    samples = {"roc": [], "prc": []}
    for indices in tqdm(
        shared_bootstrap.current_run().positions(len(prob_y), n, random_seed, clusters),
        total=n,
        desc=progress_desc,
        disable=progress_desc is None,
    ):
        if y.iloc[indices].nunique() < 2:
            continue
        for metric in samples:
            samples[metric].append(
                calculate_score(metric, y.iloc[indices], prob_y.iloc[indices])
            )

    result = {}
    for metric, values in samples.items():
        if not values:
            raise ValueError("bootstrap produced no valid replicates")
        name = f"au{metric}"
        result[name] = observed[metric]
        result[f"{name}_confidence_interval_lower"] = np.quantile(values, 0.025)
        result[f"{name}_confidence_interval_upper"] = np.quantile(values, 0.975)
        result[f"{name}_valid_bootstraps"] = len(values)
        bootstrap_storage.persist(
            progress_desc,
            name,
            [{"value": value} for value in values],
        )
    return result


def bootstrap_auc(
    n,
    y,
    prob_y,
    curve="roc",
    clusters=None,
    progress_desc=None,
    random_seed=42,
):
    """
    Bootstrap the predictions by sampling with replacement to calculate
    the 95% confidence interval of either ROC AUC (curve="roc") or
    PRC AUC (curve="prc")

    Returns:
    - confidence_lower, confidence_upper
    """

    y = pd.Series(y).reset_index(drop=True)
    prob_y = pd.Series(prob_y).reset_index(drop=True)
    observed = calculate_score(curve, y, prob_y)

    # init bootstrapping
    if clusters is not None:
        clusters = pd.Series(clusters).reset_index(drop=True)
    n_bootstraps = n
    bootstrapped_scores = []
    for indices in tqdm(
        shared_bootstrap.current_run().positions(
            len(prob_y), n_bootstraps, random_seed, clusters
        ),
        total=n_bootstraps,
        desc=progress_desc,
        disable=progress_desc is None,
    ):
        if len(np.unique(y[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue
        score = calculate_score(curve, y[indices], prob_y[indices])
        bootstrapped_scores.append(score)
    sorted_scores = np.array(bootstrapped_scores)
    bootstrap_storage.persist(
        progress_desc,
        f"au{curve}",
        [{"value": value} for value in bootstrapped_scores],
    )
    sorted_scores.sort()
    # Computing the lower and upper bound of the 95% confidence interval
    confidence_lower = sorted_scores[int(0.025 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.975 * len(sorted_scores))]
    return observed, confidence_lower, confidence_upper


def calculate_score(curve, y_true, y_pred):
    if curve == "roc":
        false_positive_rate, true_positive_rate, _ = roc_curve(y_true, y_pred)
        return auc(false_positive_rate, true_positive_rate)
    if curve == "prc":
        return average_precision_score(y_true, y_pred)
    raise ValueError("Invalid curve/metric specified. Must be either 'roc' or 'prc'")


def bootstrap_AUC_difference(
    n_bootstraps,
    y_true,
    y_pred_1,
    y_pred_2,
    curve="roc",
    clusters=None,
    progress_desc=None,
    random_seed=42,
):
    """Calculate prediction 1 minus prediction 2 using paired bootstrapping.

    :param n_bootstraps: number of resamples (bootstraps)
    :param y_true: dataframe of outcome labels
    :param y_pred_1: dataframe of predicted probabilities by model 1
    :param y_pred_2: dataframe of predicted probabilities of model 2
    :param curve: one of ["roc", "prc"]
    """
    y_true = pd.Series(y_true).reset_index(drop=True)
    y_pred_1 = pd.Series(y_pred_1).reset_index(drop=True)
    y_pred_2 = pd.Series(y_pred_2).reset_index(drop=True)
    if clusters is not None:
        clusters = pd.Series(clusters).reset_index(drop=True)

    # calculate observed difference (i.e. without all data before resampling)
    model1_score = calculate_score(curve, y_true, y_pred_1)
    model2_score = calculate_score(curve, y_true, y_pred_2)

    observed_difference = model1_score - model2_score

    bootstrapped_differences = []
    for indices in tqdm(
        shared_bootstrap.current_run().positions(
            len(y_pred_1), n_bootstraps, random_seed, clusters
        ),
        total=n_bootstraps,
        desc=progress_desc,
        disable=progress_desc is None,
    ):
        if len(np.unique(y_true[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be calculated: reject the sample
            continue

        model1_score = calculate_score(curve, y_true[indices], y_pred_1[indices])
        model2_score = calculate_score(curve, y_true[indices], y_pred_2[indices])
        bootstrapped_differences.append(model1_score - model2_score)

    bootstrapped_differences = np.array(bootstrapped_differences)
    bootstrap_storage.persist(
        progress_desc,
        f"au{curve}_difference",
        [{"value": value} for value in bootstrapped_differences],
    )
    bootstrapped_differences.sort()

    # Computing the lower and upper bound of the 95% confidence interval
    confidence_lower = bootstrapped_differences[
        int(0.025 * len(bootstrapped_differences))
    ]
    confidence_upper = bootstrapped_differences[
        int(0.975 * len(bootstrapped_differences))
    ]

    # p value
    # one-sided testing:
    # p_value = min(np.mean(bootstrapped_differences < 0), np.mean(bootstrapped_differences > 0))
    # two-sided testing:
    p_value = _two_sided_bootstrap_p(bootstrapped_differences)

    return (
        observed_difference,
        confidence_lower,
        confidence_upper,
        p_value,
        bootstrapped_differences,
        curve,
    )


def auc_differences_table(
    predictions,
    outcome,
    prediction_columns,
    curve="roc",
    n_bootstraps=1000,
    cluster_column=None,
    adjust_holm=True,
    progress_desc=None,
    random_seed=42,
):
    """Compare ordered prediction pairs as the first score minus the second."""

    rows = list(prediction_columns)[0:-1]
    columns = list(prediction_columns)[1:]

    table = pd.DataFrame(index=rows, columns=columns)

    data = {}
    family = []

    comparisons = list(combinations(prediction_columns, 2))
    comparison_iterations = tqdm(
        comparisons,
        desc=progress_desc,
        unit="comparison",
        disable=progress_desc is None,
    )
    for comparison in comparison_iterations:
        c1 = comparison[0]
        c2 = comparison[1]
        valid_predictions = ~(
            (pd.isnull(predictions[prediction_columns[c1]]))
            | (pd.isnull(predictions[prediction_columns[c2]]))
        )
        y_pred_1 = predictions.loc[valid_predictions, prediction_columns[c1]]

        y_pred_2 = predictions.loc[valid_predictions, prediction_columns[c2]]

        y_true = predictions.loc[valid_predictions, outcome]
        clusters = (
            predictions.loc[valid_predictions, cluster_column]
            if cluster_column is not None
            else None
        )

        print(f"Bootstrapping ({n_bootstraps} samples): {c1} vs. {c2}")
        (
            observed_difference,
            confidence_lower,
            confidence_upper,
            p_value,
            bootstrapped_differences,
            curve,
        ) = bootstrap_AUC_difference(
            n_bootstraps,
            y_true,
            y_pred_1,
            y_pred_2,
            curve=curve,
            clusters=clusters,
            progress_desc=(f"{progress_desc}: {c1} vs {c2}" if progress_desc else None),
            random_seed=random_seed,
        )

        # create table and store data
        table.loc[c1, c2] = (
            f"{observed_difference:#.2g} [{confidence_lower:#.2g}, {confidence_upper:#.2g}]\np = {p_value:#.2g}"
        )
        if c1 not in data.keys():
            # create empty dictionary
            data[c1] = {}

        data[c1][c2] = [
            observed_difference,
            confidence_lower,
            confidence_upper,
            p_value,
            bootstrapped_differences,
            curve,
        ]

        family.append((c1, c2, p_value))

    valid = [(c1, c2, p) for c1, c2, p in family if np.isfinite(p)]
    if valid and adjust_holm:
        adjusted = multipletests([p for _, _, p in valid], method="holm")[1]
        for (c1, c2, _), adjusted_p in zip(valid, adjusted):
            table.loc[c1, c2] += f"\nHolm-adjusted p = {adjusted_p:#.2g}"
            data[c1][c2].append(float(adjusted_p))

    return table, data


def calibration_loss(y_true, y_prob, bin_size=100):
    paired = pd.DataFrame(
        {"y_true": pd.Series(y_true).to_numpy(), "y_prob": pd.Series(y_prob).to_numpy()}
    ).sort_values("y_prob")
    y_prob = paired["y_prob"].reset_index(drop=True)
    y_true = paired["y_true"].reset_index(drop=True)
    if len(y_prob) <= bin_size:
        return np.nan
    loss = 0.0
    for i in np.arange(0, len(y_prob) - bin_size):
        avg_prob = y_true.iloc[i : i + bin_size].sum() / bin_size
        mean_pred = y_prob.iloc[i : i + bin_size].sum() / bin_size
        loss += np.abs(mean_pred - avg_prob)
    loss /= len(y_prob) - bin_size
    return loss


def performance_table(
    predictions,
    outcome,
    prediction_columns,
    curve="roc",
    n_bootstraps=1000,
    cluster_column=None,
    ordinal_predictions=(),
    progress_desc=None,
    random_seed=42,
):
    """
    Creates a table containing both discriminatory performance using the AUC metric (ROC or PRC) and
    calibration performance using calibration intercept, slope, loss, and Brier score.
    Uses bootstrapping to estimate 95% confidence intervals.

    Returns:
    - table containing performance metrics
    """

    table = pd.DataFrame()
    prediction_iterations = tqdm(
        prediction_columns,
        desc=progress_desc,
        unit="prediction",
        disable=progress_desc is None,
    )
    for c in prediction_iterations:
        is_ordinal = c in ordinal_predictions
        p = predictions.loc[
            ~pd.isnull(predictions[prediction_columns[c]]), prediction_columns[c]
        ].copy()
        if not is_ordinal:
            p.loc[p == 1] = 0.9999
            p.loc[p == 0] = 0.0001
        y = predictions.loc[~pd.isnull(predictions[prediction_columns[c]]), outcome]
        clusters = (
            predictions.loc[
                ~pd.isnull(predictions[prediction_columns[c]]), cluster_column
            ]
            if cluster_column is not None
            else None
        )
        # discrimination
        discrimination = bootstrap_discrimination(
            n_bootstraps,
            y,
            p,
            clusters=clusters,
            progress_desc=(f"{progress_desc}: {c}" if progress_desc else None),
            random_seed=random_seed,
        )
        if is_ordinal:
            calibration_results = {}
            calibration_intercept = calibration_slope = calibration_loss_value = (
                brier_score
            ) = np.nan
        else:
            original_p = predictions.loc[p.index, prediction_columns[c]]
            calibration_results = calibration.bootstrap(
                y,
                original_p,
                clusters=clusters,
                n_bootstraps=n_bootstraps,
                random_seed=random_seed,
            )
            calibration_intercept = calibration_results["calibration_intercept"]
            calibration_slope = calibration_results["calibration_slope"]
            calibration_loss_value = f"{calibration_loss(y, p):#.2g}"
            brier_score = calibration_results["brier_score"]
        # form table
        table = pd.concat(
            [
                table,
                pd.DataFrame(
                    [
                        {
                            "n": len(
                                predictions.loc[
                                    ~pd.isnull(predictions[prediction_columns[c]])
                                ]
                            ),
                            **discrimination,
                            "valid_bootstraps": discrimination[
                                f"au{curve}_valid_bootstraps"
                            ],
                            "auroc_with_confidence_interval": (
                                f"{discrimination['auroc']:#.2g} "
                                f"[{discrimination['auroc_confidence_interval_lower']:#.2g}, "
                                f"{discrimination['auroc_confidence_interval_upper']:#.2g}]"
                            ),
                            "auprc_with_confidence_interval": (
                                f"{discrimination['auprc']:#.2g} "
                                f"[{discrimination['auprc_confidence_interval_lower']:#.2g}, "
                                f"{discrimination['auprc_confidence_interval_upper']:#.2g}]"
                            ),
                            "calibration_intercept": calibration_intercept,
                            "calibration_slope": calibration_slope,
                            "calibration_loss": calibration_loss_value,
                            "brier_score": brier_score,
                            **calibration_results,
                        }
                    ]
                ),
            ]
        )
    table.index = prediction_columns.keys()
    return table


def create_max_time_difference_df(
    predictions_all_time_points,
    outcome,
    prediction_columns,
    curve="roc",
    n_bootstraps=1000,
    cluster_column=None,
    progress_desc=None,
    random_seed=42,
):
    AUC_list = (
        [f"AU{curve.upper()} {c}" for c in list(prediction_columns.keys())]
        + [f"LB {c}" for c in list(prediction_columns.keys())]
        + [f"UB {c}" for c in list(prediction_columns.keys())]
    )
    time_dif_data = pd.DataFrame(
        dict({"max_time_difference": range(4, 25)}, **{AUC: np.nan for AUC in AUC_list})
    )

    overall_progress = tqdm(
        total=len(time_dif_data) * len(prediction_columns),
        desc=progress_desc,
        unit="analysis",
        disable=progress_desc is None,
    )
    for t in time_dif_data.max_time_difference:
        predictions = predictions_all_time_points.copy()

        # remove predictions that are outside window of allowed (max) time difference
        predictions.loc[
            predictions["doc_lead_time_hours"] > t,
            ["doc_prediction", "doc_prediction_cat_score"],
        ] = np.nan
        predictions.loc[
            predictions["algo_at_doc_prediction_lead_time_hours"] > t,
            ["algo_prediction_at_doc_prediction"],
        ] = np.nan

        # bootstrap AUC for each prediction method (e.g. physician, model, ...)
        for c in prediction_columns:
            p = predictions.loc[
                ~pd.isnull(predictions[prediction_columns[c]]), prediction_columns[c]
            ]
            y = predictions.loc[~pd.isnull(predictions[prediction_columns[c]]), outcome]
            clusters = (
                predictions.loc[
                    ~pd.isnull(predictions[prediction_columns[c]]), cluster_column
                ]
                if cluster_column is not None
                else None
            )
            if p.empty or y.nunique() < 2:
                overall_progress.update()
                continue
            observed, confidence_lower, confidence_upper = bootstrap_auc(
                n_bootstraps,
                y,
                p,
                curve=curve,
                clusters=clusters,
                progress_desc=(
                    f"{progress_desc}: {c}, {t} h" if progress_desc else None
                ),
                random_seed=random_seed,
            )
            time_dif_data.loc[
                time_dif_data.max_time_difference == t,
                [f"AU{curve.upper()} {c}", f"LB {c}", f"UB {c}"],
            ] = [observed, confidence_lower, confidence_upper]
            overall_progress.update()
    overall_progress.close()
    return time_dif_data


def shap_to_dataframe(shap_dictionary, pat_deid, icu_discharged_at):
    shap_data = pd.DataFrame.from_records(
        {
            "feature_name": list(eval(shap_dictionary).keys()),
            "shap_value": list(eval(shap_dictionary).values()),
        }
    )
    shap_data["patient_id"] = pat_deid
    shap_data["icu_discharged_at"] = icu_discharged_at
    return shap_data


def construct_shap_data_frame(
    shap_variable, patient_id_variable, icu_discharged_at_variable
):
    shap_data = pd.DataFrame()
    for shap, patient_id, icu_discharged_at in zip(
        shap_variable, patient_id_variable, icu_discharged_at_variable
    ):
        if not pd.isnull(shap):
            shap_data = pd.concat(
                [shap_data, shap_to_dataframe(shap, patient_id, icu_discharged_at)]
            )
    return shap_data


def construct_doc_factors_data_frame(predictions):
    """Expand selected and free-text physician factors to one row per factor."""
    keys = predictions[["patient_id", "icu_discharged_at"]]
    selected = risk_factors.explode_factor_tokens(predictions).merge(
        keys, left_on="source_index", right_index=True, how="left"
    )
    selected["factor_text"] = pd.NA

    free_text = predictions.loc[
        predictions["doc_other_factors"].notna(),
        ["patient_id", "icu_discharged_at", "hospital", "doc_other_factors"],
    ].rename(columns={"doc_other_factors": "factor_text"})
    free_text[["source_index", "factor_field", "factor_id"]] = pd.NA

    columns = [
        "source_index",
        "patient_id",
        "icu_discharged_at",
        "hospital",
        "factor_field",
        "factor_id",
        "factor_text",
    ]
    return pd.concat([selected[columns], free_text[columns]], ignore_index=True)


def get_new_algo_features(shap_data, limit=10, absolute_values=True):
    shap_data = make_new_algo_name_feature(shap_data)
    shap_data["abs_shap_value"] = abs(shap_data.shap_value)
    if absolute_values:
        shap_data = shap_data.sort_values(
            ["patient_id", "icu_discharged_at", "abs_shap_value"], ascending=False
        )
    else:
        shap_data = shap_data.sort_values(
            ["patient_id", "icu_discharged_at", "shap_value"], ascending=False
        )
    shap_data = shap_data.drop_duplicates(
        subset=["patient_id", "icu_discharged_at", "new_feature_name"]
    )
    shap_data = shap_data.groupby(["patient_id", "icu_discharged_at"]).head(limit)
    return shap_data


def get_new_doc_features(doc_data):
    """Map physician factors to shared categories and deduplicate per stay."""
    selected = doc_data[doc_data["factor_id"].notna()].copy()
    selected = risk_factors.categorize_physician_factors(selected)

    free_text = doc_data[doc_data["factor_text"].notna()].copy()
    free_text["new_factors"] = risk_factors.categorize_free_text(
        free_text["factor_text"]
    )

    return pd.concat([selected, free_text], ignore_index=True).drop_duplicates(
        subset=["patient_id", "icu_discharged_at", "new_factors"]
    )


def add_factor_labels(doc_data):
    """Add readable labels to selected and free-text physician factors."""
    result = doc_data.copy()
    selected = result["factor_id"].notna()
    result.loc[selected, "factor_label"] = risk_factors.factor_labels(
        result.loc[selected]
    )
    result.loc[~selected, "factor_label"] = result.loc[~selected, "factor_text"]
    return result


def make_new_algo_name_feature(shap_data):
    """
    Groups the individual features into a group of features for comparing the model features
    against the physicians. Combines features from the predictions by the VUmc and LUMC model.
    """
    return risk_factors.categorize_model_features(shap_data)

    feature_name_mapping = {
        # General
        "age": "General",
        "length_of_stay_hours": "General",
        "time_since_start": "General",
        "sex__male": "General",
        "sex__female": "General",
        "body_mass_index__mean__last_24h": "General",
        # Respiratory
        "cough_stimulus__mode__last_24h__cough_reflex_normal": "Respiratory",
        "o2_flow__mean__last_24h": "Respiratory",
        "o2_flow__change_since_first__last_24h": "Respiratory",
        "respiratory_rate_measured__mean__last_24h__padded": "Respiratory",
        "respiratory_rate_measured__last__overall": "Respiratory",
        "pco2_arterial__mean__last_24h": "Respiratory",
        "tracheobronchial_toilet_quantity__total_dose__last_24h": "Respiratory",
        "tracheobronchial_toilet_quantity__total_value__last_24h": "Respiratory",
        "o2_saturation__mean__last_24h": "Respiratory",
        "o2_saturation__change_since_first__last_24h": "Respiratory",
        "tidal_volume_per_kg__mean__last_24h__padded": "Respiratory",
        "pao2_over_fio2__last__overall": "Respiratory",
        "fio2__mean__last_24h": "Respiratory",
        "peep__maximum__last_24h": "Respiratory",
        # Circulatory
        "creatine_kinase_mb_mass__last__overall": "Circulatory",
        "heart_rate__mean__last_24h__padded": "Circulatory",
        "heart_rate__mean__last_24h": "Circulatory",
        "arterial_blood_pressure_diastolic__mean__last_24h": "Circulatory",
        "arterial_blood_pressure_systolic__mean__last_24h": "Circulatory",
        "arterial_blood_pressure_mean__change_since_first__last_24h": "Circulatory",
        "lactate__last__overall": "Circulatory",
        "troponin_t__maximum__last_24h": "Circulatory",
        # Neurological
        "glasgow_coma_scale_total__maximum__last_24h": "Neurological",
        "pupil_diameter_max__mean__last_24h": "Neurological",
        "glasgow_coma_scale_total__minimum__last_24h": "Neurological",
        "richmond_agitation_sedation_scale_score__maximum__last_24h": "Neurological",
        # Renal and homeostasis
        "ph_arterial__last__overall": "Renal and homeostasis",
        "glucose__minimum__last_24h": "Renal and homeostasis",
        "sodium__mean__last_24h": "Renal and homeostasis",
        "albumin__mean__last_24h__padded": "Renal and homeostasis",
        "ureum_over_creatinine__mean__last_24h": "Renal and homeostasis",
        "ureum__mean__last_24h": "Renal and homeostasis",
        "fluid_out_urine__total_dose__last_24h": "Renal and homeostasis",
        "creatinine__last__overall": "Renal and homeostasis",
        "phosphate__mean__last_24h": "Renal and homeostasis",
        "chloride__mean__last_24h": "Renal and homeostasis",
        "base_excess__mean__last_24h": "Renal and homeostasis",
        "potassium__mean__last_24h": "Renal and homeostasis",
        "albumin__mean__last_24h": "Renal and homeostasis",
        "fluid_out_urine__total_value__last_24h": "Renal and homeostasis",
        "magnesium__mean__last_24h": "Renal and homeostasis",
        # Hematology
        "thrombocytes__mean__last_24h": "Hematology",
        "hemoglobin__mean__last_24h__padded": "Hematology",
        "prothrombin_time_inr__mean__last_24h__padded": "Hematology",
        "hemoglobin__mean__last_24h": "Hematology",
        "prothrombin_time_inr__mean__last_24h": "Hematology",
        "activated_partial_thromboplastin_time__mean__last_24h": "Hematology",
        # Liver
        "bilirubin_total__mean__last_24h": "Liver",
        "lactate_dehydrogenase__mean__last_24h": "Infection",
        "gamma_glutamyl_transferase__mean__last_24h": "Liver",
        "alanine_transaminase__mean__last_24h": "Liver",
        "aspartate_transaminase__mean__last_24h": "Liver",
        "alkaline_phosphatase__mean__last_24h": "Liver",
        # Infection
        "temperature_skin__mean__last_24h": "Infection",
        "temperature_internal__maximum__last_24h": "Infection",
        "leukocytes__mean__last_24h__padded": "Infection",
        "c_reactive_protein__mean__last_24h": "Infection",
        "temperature__minimum__last_24h": "Infection",
        "leukocytes__mean__last_24h": "Infection",
    }
    shap_data["new_feature_name"] = shap_data.feature_name.apply(
        lambda x: feature_name_mapping[x]
    )
    return shap_data


def make_new_doc_name_feature(doc_data):
    """
    Groups the individual risk factors into a group of risk factors for comparing the physicians risk factors against
    the model features. Combines risk factors documented in  VUmc and LUMC systems.
    """
    mask = ["Hematologie" in factor for factor in doc_data.factors]
    doc_data.loc[mask, "factors"] = "Hematologie pat"
    feature_name_mapping = {
        # VUmc
        # General
        "reden van opname": "General",
        "algemene (ziekte) indruk": "General",
        "leeftijd": "General",
        "eerdere heropname": "General",
        "gewicht-BMI": "General",
        "geslacht": "General",
        "Frailty": "General",
        "Maligniteit": "General",
        # Logistic
        "(verpleegkundige) zorgzwaarte": "Logistic",
        "beleid": "Logistic",
        "Niet-IC beleid ": "Logistic",
        "schizofreen": "Logistic",
        "persoonlijkheids problematiek/borderline": "Logistic",
        "depressief en verward": "Logistic",
        "Psychiatrische voorgeschiedenis": "Logistic",
        "Psychiatrisch zoektebeeld": "Logistic",
        "Psychiatrie": "Logistic",
        "woede-uitbarsting": "Logistic",
        "Isolatiemaatregelen": "Logistic",
        # Respiratory
        "oxygenatie/PaO2": "Respiratory",
        "Pulmonaal belaste voorgeschiedenis": "Respiratory",
        "ventilatie/PaCO2": "Respiratory",
        "st na pneumectomie": "Respiratory",
        "COVID": "Respiratory",
        # Circulatory
        "Bloeddruk": "Circulatory",
        "(recent) hartfalen": "Circulatory",
        "Cardiaal belaste voorgeschiedenis": "Circulatory",
        "(recente) cardiale ischemie": "Circulatory",
        "(recent) vasopressie/inotropie": "Circulatory",
        # Neurological
        "delier": "Neurological",
        "verlaagd bewustzijn/lage EMV score": "Neurological",
        "bewustzijn": "Neurological",
        # Renal and homeostasis
        "nierfalen (bestaand of risico verslechtering)": "Renal and homeostasis",
        # Hematology
        "(recidiverende) bloedingen": "Hematology",
        "Hematologie pat": "Hematology",
        # Infection
        "recidiverende infecties": "Infection",
        # Medication
        "pijnstilling": "Medication",
        # Other
        "pancreatitis, maagperforatie": "Abdominal",
        "nog onvoldoende pijncontrole": "Medication",
        # LUMC
        # General
        "Comorbiditeit: Kwetsbaarheid": "General",
        "Comorbiditeit: Overig": "General",
        "Demografisch (leeftijd, geslacht, etniciteit)": "General",
        "Comorbiditeit: Maligniteit": "General",
        "Opname kenmerk: reden van opname": "General",
        "Opname kenmerk: duur van opname of heropname": "General",
        # Logistic
        "Logistiek: zorgzwaarte voor afdeling": "Logistic",
        # Respiratory
        "Respiratie: bv. ventiliatie, zuurstofbehoefte, beademingsduur": "Respiratory",
        "Sputumproductie met uitzuigbehoefte": "Respiratory",
        "Comorbiditeit: Pulmonaal": "Respiratory",
        # Circulatory
        "Circulatie: bv. bloeddruk, ritme, ischemie, hartfalen, vochtbalans, vasopressie/inotropie": "Circulatory",
        "Comorbiditeit: Cardiaal": "Circulatory",
        # Neurological
        "Neurologie: bv. sub maximale EMV, IC verkregen zwakte, risico op verslechtering": "Neurological",
        # Renal and homeostasis
        "Nierfunctie: bv. nierinsufficientie, zuur/base- of electrolytstoornissen": "Renal and homeostasis",
        # Hematology
        "Hematologie: bv. bloedingsrisico": "Hematology",
        # Abdominal
        "Abdominaal: bv. leverinsufficientie": "Abdominal",
        # Infection
        "Infectie: bv. infectie matig onder controle, risico op infectie, isolatiemaatregelen": "Infection",
    }
    doc_data["new_factors"] = doc_data.factors.apply(lambda x: feature_name_mapping[x])
    return doc_data


def get_coverage(doc_feature_list, algo_feature_list):
    D = set(doc_feature_list)
    A = set(algo_feature_list)
    return 100 * (len(D) - len(D - A)) / len(D)


def create_shap_coverage_data(shap_data, doc_factors_data_new):
    cap_data = pd.DataFrame(
        dict(
            {"max_model_features": range(1, 11)},
            **{
                abs_values: np.nan
                for abs_values in ["absolute_values", "no_absolute_values"]
            },
        )
    )

    for limit in range(1, 11):
        for absolute_values in [True, False]:
            shap_data_new = get_new_algo_features(
                shap_data, limit=limit, absolute_values=absolute_values
            )
            # Remove factors with no overlap between algo and doc
            mask_algo = (
                (shap_data_new.new_feature_name == "Logistic")
                | (shap_data_new.new_feature_name == "Medication")
                | (shap_data_new.new_feature_name == "Other")
                | (shap_data_new.new_feature_name == "Liver")
            )
            shap_data_new = shap_data_new.loc[~mask_algo, :]
            mask_doc = (
                (doc_factors_data_new.new_factors == "Logistic")
                | (doc_factors_data_new.new_factors == "Medication")
                | (doc_factors_data_new.new_factors == "Other")
                | (doc_factors_data_new.new_factors == "Liver")
            )
            doc_factors_data_new = doc_factors_data_new.loc[~mask_doc, :]
            # Make data frame for comparison overlap factors algo and doc
            doc_factors_grouped = (
                doc_factors_data_new.groupby(["patient_id", "icu_discharged_at"])
                .new_factors.apply(list)
                .reset_index()
            )
            algo_factors_grouped = (
                shap_data_new.groupby(["patient_id", "icu_discharged_at"])
                .new_feature_name.apply(list)
                .reset_index()
            )
            feature_coverage_data = algo_factors_grouped.merge(
                doc_factors_grouped,
                how="outer",
                on=["patient_id", "icu_discharged_at"],
            )
            mask = ~pd.isnull(feature_coverage_data.new_feature_name) & ~pd.isnull(
                feature_coverage_data.new_factors
            )
            feature_coverage_data = feature_coverage_data[mask]
            feature_coverage_data["coverage"] = feature_coverage_data.apply(
                lambda x: get_coverage(x.new_factors, x.new_feature_name), axis=1
            )
            if absolute_values:
                cap_data.loc[
                    cap_data.max_model_features == limit, "absolute_values"
                ] = feature_coverage_data.coverage.mean()
            else:
                cap_data.loc[
                    cap_data.max_model_features == limit, "no_absolute_values"
                ] = feature_coverage_data.coverage.mean()
    return cap_data
