import pandas as pd
import pytest

from algo_vs_doc import analysis


def test_combine_predictions_returns_patient_grouped_out_of_fold_probabilities():
    frame = pd.DataFrame(
        {
            "patient_id": list(range(10)) * 2,
            "physician": [0.05, 0.10, 0.15, 0.20, 0.25] * 4,
            "model": [0.10, 0.20, 0.30, 0.40, 0.50] * 4,
            "outcome": [False, False, False, True, True] * 4,
        }
    )

    result = analysis.combine_predictions(
        frame, "physician", "model", "outcome", "combined"
    )

    assert result["combined"].notna().all()
    assert result["combined"].between(0, 1).all()
    assert "combined" not in frame


def test_combine_predictions_leaves_unpaired_rows_missing():
    frame = pd.DataFrame(
        {
            "patient_id": range(6),
            "physician": [0.1, 0.2, 0.3, 0.7, 0.8, None],
            "model": [0.1, 0.2, 0.3, 0.7, 0.8, 0.9],
            "outcome": [False, False, False, True, True, True],
        }
    )

    result = analysis.combine_predictions(
        frame, "physician", "model", "outcome", "combined", n_splits=2
    )

    assert result.loc[:4, "combined"].notna().all()
    assert pd.isna(result.loc[5, "combined"])


def test_combine_predictions_treats_ordinal_categories_as_separate_levels():
    frame = pd.DataFrame(
        {
            "patient_id": list(range(12)) * 2,
            "physician_category": ["low", "medium", "high"] * 8,
            "model": [0.1, 0.4, 0.8] * 8,
            "outcome": [False, False, True] * 8,
        }
    )

    result = analysis.combine_predictions(
        frame,
        "physician_category",
        "model",
        "outcome",
        "combined",
        prediction1_is_ordinal=True,
    )

    assert result["combined"].notna().all()
    assert result["combined"].between(0, 1).all()


def test_auc_difference_table_reports_holm_adjusted_p_values():
    frame = pd.DataFrame(
        {
            "outcome": [False, True] * 10,
            "prediction_a": [0.1, 0.9] * 10,
            "prediction_b": [0.2, 0.8] * 10,
            "prediction_c": [0.3, 0.7] * 10,
        }
    )

    table, data = analysis.auc_differences_table(
        frame,
        "outcome",
        {
            "A": "prediction_a",
            "B": "prediction_b",
            "C": "prediction_c",
        },
        n_bootstraps=20,
    )

    assert "Holm-adjusted p" in table.loc["A", "B"]
    assert len(data["A"]["B"]) == 7


def test_auc_difference_table_can_leave_single_primary_contrast_unadjusted():
    frame = pd.DataFrame(
        {
            "outcome": [False, True] * 10,
            "physician": [0.2, 0.8] * 10,
            "model": [0.1, 0.9] * 10,
        }
    )

    table, data = analysis.auc_differences_table(
        frame,
        "outcome",
        {"Physician": "physician", "Model": "model"},
        n_bootstraps=20,
        adjust_holm=False,
    )

    assert "Holm-adjusted" not in table.loc["Physician", "Model"]
    assert len(data["Physician"]["Model"]) == 6


def test_nested_combination_bootstrap_refits_and_compares_predictions():
    frame = pd.DataFrame(
        {
            "patient_id": list(range(20)) * 2,
            "physician": [0.05, 0.10, 0.20, 0.70, 0.80] * 8,
            "model": [0.10, 0.20, 0.30, 0.60, 0.90] * 8,
            "outcome": [False, False, False, True, True] * 8,
        }
    )

    performance, comparisons, alert_rates, bootstrap_samples = (
        analysis.nested_bootstrap_combined_auc(
            frame,
            prediction1="physician",
            prediction2="model",
            outcome="outcome",
            combined_prediction="combined",
            comparison_predictions={"Physician": "physician", "Model": "model"},
            n_bootstraps=20,
        )
    )

    assert performance["auroc"] == pytest.approx(1.0)
    assert performance["auprc"] == pytest.approx(1.0)
    assert performance["valid_bootstraps"] == 20
    assert set(comparisons["agent_2"]) == {"Physician", "Model"}
    assert comparisons["valid_bootstraps"].eq(20).all()
    assert comparisons["p"].between(0, 1).all()
    assert set(alert_rates["target_alert_rate"]) == {0.05, 0.10, 0.15}
    assert alert_rates["sensitivity_valid_bootstraps"].eq(20).all()
    assert set(bootstrap_samples["agent_2"]) == {"Physician", "Model"}
    assert len(bootstrap_samples) == 40


def test_performance_table_omits_calibration_for_ordinal_scores():
    frame = pd.DataFrame(
        {
            "outcome": [False, False, True, True] * 5,
            "category_score": [0.0, 1.0, 1.0, 2.0] * 5,
        }
    )

    result = analysis.performance_table(
        frame,
        outcome="outcome",
        prediction_columns={"Physician (Category)": "category_score"},
        n_bootstraps=20,
        ordinal_predictions={"Physician (Category)"},
    )

    row = result.loc["Physician (Category)"]
    assert 0 <= row["auprc"] <= 1
    assert 0 <= row["auroc"] <= 1
    assert pd.isna(row["calibration_intercept"])
    assert pd.isna(row["calibration_slope"])
    assert pd.isna(row["calibration_loss"])
    assert pd.isna(row["brier_score"])
