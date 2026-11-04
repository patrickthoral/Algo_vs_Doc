import numpy as np
import pandas as pd

from algo_vs_doc import comparison

T = pd.Timestamp


def _stays(rows):
    """Rows of ``(has_doc, has_algo, icu_los, died)``."""
    return pd.DataFrame(
        [
            {
                "doc_predicted_at": T("2024-01-02") if has_doc else pd.NaT,
                "algo_predicted_at_doc_prediction": T("2024-01-02")
                if has_algo
                else pd.NaT,
                "icu_los": los,
                "died_on_icu": died,
            }
            for has_doc, has_algo, los, died in rows
        ]
    )


def test_prediction_availability_labels_all_four_combinations():
    data = _stays(
        [
            (True, True, 1, False),
            (True, False, 1, False),
            (False, True, 1, False),
            (False, False, 1, False),
        ]
    )
    assert comparison.prediction_availability(data).tolist() == [
        "both",
        "physician only",
        "model only",
        "neither",
    ]


def test_prediction_availability_is_ordered_so_tables_stay_consistent():
    data = _stays([(True, True, 1, False)])
    result = comparison.prediction_availability(data)
    assert list(result.cat.categories) == list(comparison.PREDICTION_AVAILABILITY)


def test_prediction_pair_availability_requires_both_predictions():
    data = _stays(
        [
            (True, True, 1, False),
            (True, False, 1, False),
            (False, True, 1, False),
            (False, False, 1, False),
        ]
    )

    result = comparison.prediction_pair_availability(data)

    assert result.tolist() == ["available", "unavailable", "unavailable", "unavailable"]
    assert list(result.cat.categories) == list(comparison.PREDICTION_PAIR_AVAILABILITY)
    assert result.cat.ordered


def test_prediction_source_availability_treats_sources_independently():
    data = _stays([(True, False, 1, False), (False, True, 1, False)])

    physician = comparison.prediction_source_availability(
        data, column="doc_predicted_at", source="physician"
    )
    algorithm = comparison.prediction_source_availability(
        data, column="algo_predicted_at_doc_prediction", source="algorithm"
    )

    assert physician.tolist() == ["available", "unavailable"]
    assert algorithm.tolist() == ["unavailable", "available"]


def test_compare_groups_reports_a_column_per_group_headed_with_its_n():
    data = _stays([(True, True, 2, False)] * 3 + [(False, False, 9, True)] * 4)
    table = comparison.compare_groups(
        data, comparison.prediction_availability(data), ["icu_los"]
    )
    assert "both (n=3)" in table.columns
    assert "neither (n=4)" in table.columns
    # groups that never occur are not given empty columns
    assert not any(c.startswith("model only") for c in table.columns)


def test_compare_groups_summarizes_numeric_as_median_and_iqr():
    data = _stays([(True, True, v, False) for v in (1, 2, 3, 4, 5)])
    table = comparison.compare_groups(
        data, comparison.prediction_availability(data), ["icu_los"]
    )
    assert table["both (n=5)"].iloc[0] == "3.0 [2.0-4.0]"


def test_compare_groups_summarizes_binary_as_count_and_percentage():
    data = _stays([(True, True, 1, True), (True, True, 1, False)])
    table = comparison.compare_groups(
        data,
        comparison.prediction_availability(data),
        ["died_on_icu"],
        binary=["died_on_icu"],
    )
    assert table["both (n=2)"].iloc[0] == "1 (50.0%)"


def test_compare_groups_detects_a_real_difference_between_groups():
    # stays without predictions are much longer here - the exclusion is selective, which is
    # exactly what this table exists to reveal
    rng = np.random.default_rng(0)
    with_pred = [(True, True, float(v), False) for v in rng.normal(3, 0.5, 60)]
    without = [(False, False, float(v), False) for v in rng.normal(12, 0.5, 60)]
    data = _stays(with_pred + without)
    table = comparison.compare_groups(
        data, comparison.prediction_availability(data), ["icu_los"]
    ).set_index("variable")
    assert table.loc["icu_los", "p"] < 0.001
    assert table.loc["icu_los", "p_holm"] < 0.001


def test_holm_adjust_corrects_one_family_and_preserves_missing_values():
    result = comparison.holm_adjust(pd.Series([0.01, 0.04, 0.20, np.nan]))

    assert result.iloc[:3].tolist() == [0.03, 0.08, 0.20]
    assert pd.isna(result.iloc[3])


def test_compare_groups_reports_no_difference_when_the_groups_are_alike():
    rng = np.random.default_rng(1)
    data = _stays(
        [(True, True, float(v), False) for v in rng.normal(5, 1, 60)]
        + [(False, False, float(v), False) for v in rng.normal(5, 1, 60)]
    )
    table = comparison.compare_groups(
        data, comparison.prediction_availability(data), ["icu_los"]
    ).set_index("variable")
    assert table.loc["icu_los", "p"] > 0.05


def test_an_untestable_comparison_reports_nan_rather_than_a_misleading_number():
    # one stay per group: nothing can be tested, so no p-value should be invented
    data = _stays([(True, True, 3, False), (False, False, 9, True)])
    table = comparison.compare_groups(
        data, comparison.prediction_availability(data), ["icu_los"]
    )
    assert pd.isna(table["p"].iloc[0])
    assert pd.isna(table["p_holm"].iloc[0])


def test_a_group_with_no_usable_values_shows_a_dash_not_a_fabricated_zero():
    data = _stays([(True, True, 3, False), (False, False, float("nan"), False)])
    table = comparison.compare_groups(
        data, comparison.prediction_availability(data), ["icu_los"]
    )
    assert table["neither (n=1)"].iloc[0] == "-"


def test_compare_groups_accepts_a_column_name_and_renames_for_display():
    data = _stays([(True, True, 3, False), (False, False, 9, False)])
    data["availability"] = comparison.prediction_availability(data)
    table = comparison.compare_groups(
        data,
        "availability",
        ["icu_los"],
        labels={"icu_los": "ICU length of stay (days)"},
    )
    assert table["variable"].iloc[0] == "ICU length of stay (days)"


def test_compare_groups_skips_a_variable_the_cohort_does_not_have():
    data = _stays([(True, True, 3, False)])
    table = comparison.compare_groups(
        data,
        comparison.prediction_availability(data),
        ["icu_los", "not_collected_here"],
    )
    assert table["variable"].tolist() == ["icu_los"]


def test_compare_groups_does_not_modify_the_input():
    data = _stays([(True, True, 3, False)])
    before = data.copy()
    comparison.compare_groups(
        data, comparison.prediction_availability(data), ["icu_los"]
    )
    pd.testing.assert_frame_equal(data, before)


def test_compare_groups_can_suppress_naive_tests_for_clustered_variables():
    data = pd.DataFrame(
        {"group": ["available", "missing"] * 3, "daily_pressure": [1, 1, 2, 2, 3, 3]}
    )

    table = comparison.compare_groups(
        data,
        "group",
        ["daily_pressure"],
        descriptive_only=["daily_pressure"],
    )

    assert table.loc[0, "test"] == "descriptive only"
    assert pd.isna(table.loc[0, "p"])
    assert pd.isna(table.loc[0, "p_holm"])
