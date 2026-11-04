import numpy as np
import pandas as pd
import pytest

from algo_vs_doc import (
    analysis,
    calibration,
    clinical_performance,
    exploratory_contrasts,
    shared_bootstrap,
)


@pytest.fixture(autouse=True)
def isolated_run():
    shared_bootstrap.begin_run()
    yield
    shared_bootstrap.begin_run()


def test_shared_positions_match_default_rng_draws():
    clusters = pd.Series(["b", "a", "b", "c", "d", "a"])
    run = shared_bootstrap.current_run()
    original = np.random.default_rng(42)
    expected = [
        analysis._bootstrap_positions(len(clusters), original, clusters)
        for _ in range(20)
    ]
    for _ in range(2):
        actual = list(run.positions(len(clusters), 20, 42, clusters))
        for left, right in zip(expected, actual):
            np.testing.assert_array_equal(left, right)
    assert run.sampling_runs == 1
    list(run.positions(5, 20, 42, clusters.iloc[:5]))
    assert run.sampling_runs == 2
    list(run.positions(6, 20, 42, clusters.iloc[::-1]))
    assert run.sampling_runs == 3


def fixture_frame():
    rng = np.random.default_rng(9)
    return pd.DataFrame(
        {
            "patient_cluster_id": np.repeat(np.arange(40), 2),
            "outcome": np.tile([False, False, True, True], 20),
            "doc_prediction": rng.uniform(0.05, 0.8, 80),
            "doc_prediction_cat": np.tile(["low", "medium", "high", "medium"], 20),
            "doc_prediction_cat_score": np.tile([0, 1, 2, 1], 20),
            "algo_prediction_at_doc_prediction": rng.uniform(0.05, 0.8, 80),
        }
    )


def test_metrics_share_one_resampling_plan_and_preserve_discrimination():
    frame = fixture_frame()
    n = 10
    result = analysis.bootstrap_discrimination(
        n,
        frame.outcome,
        frame.doc_prediction,
        clusters=frame.patient_cluster_id,
    )
    rng = np.random.default_rng(42)
    expected = []
    for _ in range(n):
        positions = analysis._bootstrap_positions(
            len(frame), rng, frame.patient_cluster_id
        )
        expected.append(
            analysis.calculate_score(
                "roc",
                frame.outcome.iloc[positions],
                frame.doc_prediction.iloc[positions],
            )
        )
    assert result["auroc_confidence_interval_lower"] == np.quantile(expected, 0.025)
    calibration.bootstrap(
        frame.outcome,
        frame.doc_prediction,
        clusters=frame.patient_cluster_id,
        n_bootstraps=n,
    )
    clinical_performance.alert_rate_table(
        frame,
        prediction="doc_prediction",
        outcome="outcome",
        cluster_column="patient_cluster_id",
        n_bootstraps=n,
    )
    clinical_performance.average_precision_metrics(
        frame,
        prediction="doc_prediction",
        outcome="outcome",
        cluster_column="patient_cluster_id",
        n_bootstraps=n,
    )
    assert shared_bootstrap.current_run().sampling_runs == 1


def test_nested_exploratory_and_curve_workflows_reuse_combination_fits():
    frame = fixture_frame()
    for ordinal, first, output in (
        (False, "doc_prediction", "algo_and_doc_prediction"),
        (True, "doc_prediction_cat", "algo_and_doc_cat_prediction"),
    ):
        analysis.nested_bootstrap_combined_auc(
            frame,
            prediction1=first,
            prediction2="algo_prediction_at_doc_prediction",
            outcome="outcome",
            combined_prediction=output,
            comparison_predictions={"Model": "algo_prediction_at_doc_prediction"},
            group_column="patient_cluster_id",
            prediction1_is_ordinal=ordinal,
            n_bootstraps=3,
        )
    run = shared_bootstrap.current_run()
    assert run.model_fits == 8
    exploratory_contrasts.supplementary_contrasts(frame, n_bootstraps=3)
    assert run.model_fits == 8
    frame["combined"] = analysis.combine_predictions(
        frame,
        "doc_prediction",
        "algo_prediction_at_doc_prediction",
        "outcome",
        "combined",
        group_column="patient_cluster_id",
    )["combined"]
    calibration.plot_tables(
        frame,
        "outcome",
        {"Combined": "combined"},
        cluster_column="patient_cluster_id",
        n_bootstraps=3,
        random_seed=42,
        combination_specs={
            "combined": {
                "prediction1": "doc_prediction",
                "prediction2": "algo_prediction_at_doc_prediction",
            }
        },
    )
    assert run.sampling_runs == 1
    assert run.model_fits == 8
    changed = frame.copy()
    changed.loc[0, "doc_prediction"] += 0.01
    analysis.combine_predictions(
        changed,
        "doc_prediction",
        "algo_prediction_at_doc_prediction",
        "outcome",
        "combined",
        group_column="patient_cluster_id",
    )
    assert run.model_fits == 9
