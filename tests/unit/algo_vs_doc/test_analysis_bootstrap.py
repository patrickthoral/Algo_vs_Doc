import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from algo_vs_doc import analysis


def test_prc_score_uses_average_precision_for_tied_risk_categories():
    outcomes = pd.Series([True, False, True, False, False, True])
    predictions = pd.Series([0.0, 0.0, 0.5, 0.5, 1.0, 1.0])

    result = analysis.calculate_score("prc", outcomes, predictions)

    assert result == average_precision_score(outcomes, predictions)


def test_cluster_bootstrap_samples_complete_patient_clusters():
    clusters = pd.Series(["a", "a", "b", "b", "c"])

    positions = analysis._bootstrap_positions(
        len(clusters), np.random.RandomState(42), clusters
    )

    counts = pd.Series(clusters.iloc[positions].to_numpy()).value_counts()
    for cluster, count in counts.items():
        expected_cluster_size = int(clusters.eq(cluster).sum())
        assert count % expected_cluster_size == 0


def test_bootstrap_auc_accepts_repeated_patient_clusters_without_mutating_inputs():
    outcomes = pd.Series(
        [False, True, False, True, False, True] * 20, index=range(120, 240)
    )
    predictions = pd.Series(np.linspace(0.01, 0.99, 120), index=range(240, 360))
    clusters = pd.Series(np.repeat(np.arange(60), 2), index=range(360, 480))
    original_indexes = (
        outcomes.index.copy(),
        predictions.index.copy(),
        clusters.index.copy(),
    )

    observed, lower, upper = analysis.bootstrap_auc(
        100, outcomes, predictions, clusters=clusters
    )

    assert np.isfinite([observed, lower, upper]).all()
    assert lower <= upper
    assert outcomes.index.equals(original_indexes[0])
    assert predictions.index.equals(original_indexes[1])
    assert clusters.index.equals(original_indexes[2])


def test_bootstrap_auc_is_reproducible_for_an_explicit_seed():
    outcomes = pd.Series([False, True] * 20)
    predictions = pd.Series(np.linspace(0.01, 0.99, 40))

    first = analysis.bootstrap_auc(50, outcomes, predictions, random_seed=123)
    second = analysis.bootstrap_auc(50, outcomes, predictions, random_seed=123)

    assert first == second


def test_bootstrap_auc_difference_returns_prediction_1_minus_prediction_2():
    outcomes = pd.Series([False, True] * 20)
    better = pd.Series([0.1, 0.9] * 20)
    worse = pd.Series([0.9, 0.1] * 20)

    difference = analysis.bootstrap_AUC_difference(
        50, outcomes, better, worse, random_seed=123
    )

    assert difference[0] == 1
    assert np.array_equal(difference[4], np.ones(50))


def test_bootstrap_discrimination_reuses_samples_for_auroc_and_auprc():
    outcomes = pd.Series([False, True] * 20)
    predictions = pd.Series(np.linspace(0.01, 0.99, 40))

    result = analysis.bootstrap_discrimination(
        50, outcomes, predictions, random_seed=123
    )

    assert 0 <= result["auprc"] <= 1
    assert 0 <= result["auroc"] <= 1
    assert result["auprc_valid_bootstraps"] == 50
    assert result["auroc_valid_bootstraps"] == 50


def test_two_sided_bootstrap_p_has_finite_simulation_lower_bound():
    result = analysis._two_sided_bootstrap_p(np.ones(10_000))

    assert result == 2 / 10_001
    assert result > 0
