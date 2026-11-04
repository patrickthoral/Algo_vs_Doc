import pandas as pd

from algo_vs_doc import table_presentation


def test_performance_contrasts_pivots_metrics_and_orders_comparisons():
    frame = pd.DataFrame(
        {
            "cohort": ["Combined"] * 4,
            "family": ["exploratory", "exploratory", "secondary", "primary"],
            "metric": ["auprc", "auroc", "auprc", "auroc"],
            "agent_1": [
                "Combined (Percentage)",
                "Combined (Percentage)",
                "Model at Physician Prediction",
                "Model at Physician Prediction",
            ],
            "agent_2": [
                "Combined (Category)",
                "Combined (Category)",
                "Physician (Percentage)",
                "Physician (Percentage)",
            ],
            "difference_agent_1_minus_agent_2": [0.01, 0.02, 0.03, 0.04],
            "confidence_interval_lower": [-0.01, 0.00, 0.01, 0.02],
            "confidence_interval_upper": [0.03, 0.04, 0.05, 0.06],
            "p": [0.5, 0.4, 0.2, 0.1],
            "p_holm": [float("nan"), float("nan"), 0.4, 0.2],
        }
    )

    result = table_presentation.performance_contrasts(frame)

    assert result["Analysis"].astype(str).tolist() == [
        "Primary/secondary",
        "Exploratory",
    ]
    assert result.columns.tolist() == [
        "Analysis",
        "Agent 1",
        "Agent 2",
        "Both Hospitals AUROC",
        "Both Hospitals AUPRC",
        "VUmc AUROC",
        "VUmc AUPRC",
        "LUMC AUROC",
        "LUMC AUPRC",
    ]
    assert result.loc[0, "Both Hospitals AUROC"].p_value == "0.200"
    assert result.loc[0, "Both Hospitals AUPRC"].p_value == "0.400"
    assert result.loc[1, "Both Hospitals AUROC"].p_value is None
    assert result.loc[1, "Both Hospitals AUPRC"].p_value is None
    assert result.loc[0, "VUmc AUROC"].estimate == ""


def test_factor_categories_pivots_sources_within_cohorts():
    frame = pd.DataFrame(
        {
            "cohort": ["Combined", "Combined", "VUmc", "VUmc", "LUMC", "LUMC"],
            "prediction_source": ["Physician", "Model"] * 3,
            "factor_category": ["Respiratory"] * 6,
            "count": [10, 20, 4, 8, 6, 12],
            "percentage": [25.0, 50.0, 20.0, 40.0, 30.0, 60.0],
        }
    )

    result = table_presentation.factor_categories(frame)

    assert result.columns.tolist() == [
        "Factor category",
        "Both Hospitals Physician",
        "Both Hospitals Model",
        "VUmc Physician",
        "VUmc Model",
        "LUMC Physician",
        "LUMC Model",
    ]
    assert result.iloc[0].tolist() == [
        "Respiratory",
        "10 (25.0)",
        "20 (50.0)",
        "4 (20.0)",
        "8 (40.0)",
        "6 (30.0)",
        "12 (60.0)",
    ]


def test_prediction_availability_characteristics_are_grouped_by_hospital():
    frame = pd.DataFrame(
        {
            "cohort": ["VUmc", "LUMC"],
            "variable": ["n_census_at_discharge"] * 2,
            "prediction_available": ["12.0 [10.0-14.0]", "11.0 [9.0-13.0]"],
            "prediction_missing": ["15.0 [12.0-18.0]", "14.0 [11.0-17.0]"],
        }
    )
    denominators = pd.DataFrame(
        {
            "cohort": ["VUmc", "LUMC"],
            "physician_prediction_available": [180, 266],
            "physician_prediction_missing": [26, 240],
        }
    )

    result = table_presentation.prediction_availability_characteristics(
        frame, denominators
    )

    assert result.columns.tolist() == [
        "Characteristic",
        "VUmc available",
        "VUmc missing",
        "LUMC available",
        "LUMC missing",
    ]
    assert result.loc[0].to_dict() == {
        "Characteristic": "ICU discharges, n (%)",
        "VUmc available": "180 (87.4%)",
        "VUmc missing": "26 (12.6%)",
        "LUMC available": "266 (52.6%)",
        "LUMC missing": "240 (47.4%)",
    }
    assert result.loc[1, "Characteristic"] == "ICU census at discharge, median [IQR]"
    assert result.loc[1, "LUMC missing"] == "14.0 [11.0-17.0]"


def test_prediction_availability_characteristics_omit_readiness():
    frame = pd.DataFrame(
        {
            "cohort": ["VUmc"],
            "variable": ["ready_for_discharge"],
            "prediction_available": ["171 (95.0%)"],
            "prediction_missing": ["-"],
        }
    )
    denominators = pd.DataFrame(
        {
            "cohort": ["VUmc", "LUMC"],
            "physician_prediction_available": [180, 266],
            "physician_prediction_missing": [26, 240],
        }
    )

    result = table_presentation.prediction_availability_characteristics(
        frame, denominators
    )

    assert result["Characteristic"].tolist() == ["ICU discharges, n (%)"]


def test_prediction_availability_models_group_hospitals_and_omit_intercept():
    frame = pd.DataFrame(
        {
            "cohort": ["VUmc", "VUmc", "LUMC", "LUMC"],
            "variable": ["const", "discharge_is_weekend"] * 2,
            "odds_ratio": [1.0, 0.7, 1.0, 0.8],
            "confidence_interval_lower": [0.5, 0.5, 0.5, 0.6],
            "confidence_interval_upper": [2.0, 1.0, 2.0, 1.1],
            "n": [80, 80, 100, 100],
            "available_n": [70, 70, 60, 60],
            "unavailable_n": [10, 10, 40, 40],
        }
    )

    result = table_presentation.prediction_availability_models(frame)

    assert result.columns.tolist() == [
        "Characteristic",
        "VUmc adjusted odds ratio",
        "LUMC adjusted odds ratio",
    ]
    assert result["Characteristic"].tolist() == ["Weekend discharge, n (%)"]
    assert result.loc[0, "VUmc adjusted odds ratio"].estimate == "0.70 [0.50, 1.00]"
    assert result.loc[0, "LUMC adjusted odds ratio"].estimate == "0.80 [0.60, 1.10]"
