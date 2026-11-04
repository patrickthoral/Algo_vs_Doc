"""Paired exploratory contrasts with jointly refitted combination models."""

from itertools import combinations

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from algo_vs_doc import analysis, comparison, shared_bootstrap


PREDICTIONS = {
    "Physician (Percentage)": "doc_prediction",
    "Physician (Category)": "doc_prediction_cat_score",
    "Model at Physician Prediction": "algo_prediction_at_doc_prediction",
    "Combined (Percentage)": "algo_and_doc_prediction",
    "Combined (Category)": "algo_and_doc_cat_prediction",
}
MISSING_ROC = {(1, 3), (0, 4), (3, 4)}


def supplementary_contrasts(data, *, n_bootstraps, random_seed=42, n_splits=5):
    """Calculate three additional AUROC and all ten average-precision AUPRC contrasts.

    All comparisons use complete observations on the five agents' input fields.
    Both combinations are refitted using grouped out-of-fold prediction within
    each shared patient-cluster resample. Existing primary contrasts are not
    replaced. Exploratory p-values are unadjusted and explicitly labeled.
    """
    inputs = [
        "doc_prediction",
        "doc_prediction_cat",
        "doc_prediction_cat_score",
        "algo_prediction_at_doc_prediction",
        "outcome",
        "patient_cluster_id",
    ]
    complete = data.dropna(subset=inputs).reset_index(drop=True)
    if n_bootstraps < 1 or complete.outcome.nunique() != 2:
        raise ValueError("positive bootstrap count and both outcome classes required")

    def scores(frame):
        fitted = analysis.combine_predictions(
            frame,
            "doc_prediction",
            "algo_prediction_at_doc_prediction",
            "outcome",
            "algo_and_doc_prediction",
            group_column="patient_cluster_id",
            n_splits=n_splits,
            random_seed=random_seed,
        )
        fitted = analysis.combine_predictions(
            fitted,
            "doc_prediction_cat",
            "algo_prediction_at_doc_prediction",
            "outcome",
            "algo_and_doc_cat_prediction",
            group_column="patient_cluster_id",
            n_splits=n_splits,
            random_seed=random_seed,
            prediction1_is_ordinal=True,
        )
        if fitted[list(PREDICTIONS.values())].isna().any().any():
            return None
        return {
            curve: [
                analysis.calculate_score(curve, fitted.outcome, fitted[column])
                for column in PREDICTIONS.values()
            ]
            for curve in ("roc", "prc")
        }

    observed = scores(complete)
    if observed is None:
        raise ValueError("insufficient patient groups for combined predictions")
    specifications = [
        (curve, a, b)
        for curve in ("roc", "prc")
        for a, b in combinations(range(5), 2)
        if curve == "prc" or (a, b) in MISSING_ROC
    ]
    draws = {spec: [] for spec in specifications}
    samples = shared_bootstrap.current_run().positions(
        len(complete), n_bootstraps, random_seed, complete.patient_cluster_id
    )
    for replicate, positions in enumerate(
        tqdm(
            samples,
            total=n_bootstraps,
            desc="Exploratory paired ROC/PRC",
            unit="bootstrap",
        )
    ):
        sample = complete.iloc[positions].reset_index(drop=True)
        if sample.outcome.nunique() != 2:
            continue
        values = scores(sample)
        if values is None:
            continue
        for curve, a, b in specifications:
            draws[curve, a, b].append((replicate, values[curve][b] - values[curve][a]))
    labels = list(PREDICTIONS)
    rows, samples = [], []
    for contrast_id, (curve, a, b) in enumerate(specifications):
        records = draws[curve, a, b]
        if not records:
            raise ValueError("no valid paired exploratory bootstrap replicates")
        values = np.array([value for _, value in records])
        common = dict(
            contrast_id=contrast_id,
            metric=f"au{curve}",
            agent_1=labels[b],
            agent_2=labels[a],
        )
        rows.append(
            dict(
                **common,
                difference_agent_1_minus_agent_2=observed[curve][b]
                - observed[curve][a],
                confidence_interval_lower=np.quantile(values, 0.025),
                confidence_interval_upper=np.quantile(values, 0.975),
                p=analysis._two_sided_bootstrap_p(values),
                p_holm=np.nan,
                family="exploratory",
                valid_bootstraps=len(values),
                n=len(complete),
                events_n=int(complete.outcome.sum()),
            )
        )
        samples.extend(
            dict(
                **common,
                replicate=replicate,
                bootstrap_difference_agent_1_minus_agent_2=value,
            )
            for replicate, value in records
        )
    result = pd.DataFrame(rows)
    primary_prc = result.metric.eq("auprc") & result.agent_1.eq(
        "Model at Physician Prediction"
    )
    result.loc[primary_prc, "family"] = "secondary"
    result.loc[primary_prc, "p_holm"] = comparison.holm_adjust(
        result.loc[primary_prc, "p"]
    )
    return result, pd.DataFrame(samples)
