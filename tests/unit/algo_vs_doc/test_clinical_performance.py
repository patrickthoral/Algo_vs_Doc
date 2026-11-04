import pandas as pd
import pytest

from algo_vs_doc import clinical_performance


def _frame():
    return pd.DataFrame(
        {
            "risk": [0.9, 0.8, 0.4, 0.1],
            "outcome": [True, False, True, False],
            "physician_category": ["high", "high", "medium", "low"],
            "model_category": ["high", "medium", "medium", "low"],
            "patient": ["a", "b", "c", "d"],
        }
    )


def test_alert_rate_table_reports_the_operational_budget_and_events_captured():
    result = clinical_performance.alert_rate_table(
        _frame(), prediction="risk", outcome="outcome", alert_rates=[0.5]
    ).iloc[0]

    assert result["flagged_n"] == 2
    assert result["threshold"] == 0.8
    assert result["events_captured_n"] == 1
    assert result["sensitivity"] == 0.5
    assert result["positive_predictive_value"] == 0.5
    assert result["specificity"] == 0.5
    assert result["negative_predictive_value"] == 0.5
    assert result["f1"] == 0.5


def test_alert_rate_table_adds_patient_cluster_percentile_intervals():
    result = clinical_performance.alert_rate_table(
        _frame(),
        prediction="risk",
        outcome="outcome",
        alert_rates=[0.5],
        cluster_column="patient",
        n_bootstraps=100,
    ).iloc[0]

    assert 0 <= result["sensitivity_ci_lower"] <= result["sensitivity_ci_upper"] <= 1
    assert result["sensitivity_valid_bootstraps"] > 0


def test_threshold_metrics_reports_all_confusion_matrix_cells():
    result = clinical_performance.threshold_metrics(
        _frame(), prediction="risk", outcome="outcome", threshold=0.5
    )

    assert (result["true_positive"], result["false_positive"]) == (1, 1)
    assert (result["false_negative"], result["true_negative"]) == (1, 1)
    assert result["f1"] == 0.5


def test_youden_threshold_chooses_an_observed_operating_point():
    assert (
        clinical_performance.youden_threshold(
            _frame(), prediction="risk", outcome="outcome"
        )
        in _frame()["risk"].tolist()
    )


def test_average_precision_metrics_adds_patient_cluster_interval():
    result = clinical_performance.average_precision_metrics(
        _frame(),
        prediction="risk",
        outcome="outcome",
        cluster_column="patient",
        n_bootstraps=100,
    )

    assert result["n"] == 4
    assert result["events_n"] == 2
    assert 0 <= result["average_precision_ci_lower"] <= 1
    assert 0 <= result["average_precision_ci_upper"] <= 1
    assert result["average_precision_valid_bootstraps"] > 0


def test_risk_category_events_includes_zero_count_categories():
    result = clinical_performance.risk_category_events(
        _frame(),
        category_columns={"physician": "physician_category"},
        outcome="outcome",
    ).set_index("risk_category")

    assert result.loc["medium", "events_n"] == 1
    assert result.loc["low", "event_rate"] == 0


def test_invalid_alert_rate_is_rejected():
    with pytest.raises(ValueError, match="alert rates"):
        clinical_performance.alert_rate_table(
            _frame(), prediction="risk", outcome="outcome", alert_rates=[0]
        )
