import pandas as pd

from algo_vs_doc import baseline_descriptives


def _frame():
    return pd.DataFrame(
        {
            "patient_id": ["p1", "p1"],
            "hospital": ["VUmc", "VUmc"],
            "age_at_admission": [60, 80],
            "sex": ["female", "male"],
            "specialty": ["cardiology", "cardiology"],
            "discharge_care_level": ["general_ward", "high_dependency_unit"],
            "is_medical": [True, False],
            "elective_surgery": [False, False],
            "emergency_surgery": [False, True],
            "icu_los": [12, 36],
            "ventilation": [True, False],
            "vasopressors_inotropes": [False, True],
            "renal_replacement_therapy": [False, False],
            "sofa_admission": [4, 8],
            "sofa_max": [7, 12],
            "sofa_discharge": [2, 5],
            "news_discharge": [1, 6],
            "ready_for_discharge": pd.array([True, None], dtype="boolean"),
            "discharge_is_weekend": pd.array([True, False], dtype="boolean"),
            "discharge_after_18h": pd.array([False, True], dtype="boolean"),
            "n_census_at_discharge": [3, 5],
            "n_external_transfers_same_day": [0, 2],
            "n_elective_surgery_admissions_same_day": [4, 6],
            "index_episode_diagnoses_count": [0, 1],
            "palliative_care": [False, True],
            "do_not_readmit": [False, False],
            "has_icu_readmission_7days": [False, True],
            "has_death_7days": [False, False],
            "outcome": [False, True],
        }
    )


def test_cohort_summary_uses_observed_readiness_as_its_denominator():
    result = baseline_descriptives.extended_context_summary(_frame())

    assert result["ICU discharges, n"] == "2"
    assert result["Patients, n"] == "1"
    assert result.index.tolist()[:2] == ["ICU discharges, n", "Patients, n"]
    assert result["Readiness assessment available, n (%)"] == "1 (50.0%)"
    assert result["Ready for discharge, n/N observed (%)"] == "1/1 (100.0%)"
    assert result["  Ventilation"] == "1 (50.0%)"
    assert result["  Vasopressors or inotropes"] == "1 (50.0%)"
    assert result["  Renal-replacement therapy"] == "0 (0.0%)"
    assert result["  SOFA at ICU admission"] == "6.0 (5.0-7.0)"
    assert result["  Maximum SOFA during ICU stay"] == "9.5 (8.2-10.8)"
    assert result["  SOFA at ICU discharge"] == "3.5 (2.8-4.2)"
    assert result["  NEWS at ICU discharge"] == "3.5 (2.2-4.8)"
    assert result["Composite outcome within 7 days, n (%)"] == "1 (50.0%)"
    assert result["  ICU readmission within 7 days, n (%)"] == "1 (50.0%)"
    assert result["  Death within 7 days, n (%)"] == "0 (0.0%)"


def test_patient_count_keeps_site_local_identifiers_separate():
    frame = _frame()
    frame["hospital"] = ["VUmc", "LUMC"]

    assert baseline_descriptives._patient_count(frame) == 2


def test_baseline_characteristics_table_uses_reusable_publication_rows():
    result = baseline_descriptives.baseline_characteristics_table(
        {"VUmc": _frame(), "Combined": _frame()}
    )

    assert result.columns.tolist() == ["VUmc", "Combined"]
    assert result.index.tolist()[:4] == [
        "ICU discharges, n",
        "Patients, n",
        "Age, mean (SD)",
        "Female, n (%)",
    ]
    assert result.loc["ICU discharges, n", "Combined"] == "2"
    assert result.loc["Patients, n", "Combined"] == "1"
    assert result.loc["Medical admission", "Combined"] == "1 (50.0)"
    assert result.loc["Emergency surgical admission", "Combined"] == "1 (50.0)"
    assert result.loc["General ward", "Combined"] == "1 (50.0)"
    assert result.loc["High-dependency care", "Combined"] == "1 (50.0)"
    frame_with_unknown_setting = _frame()
    frame_with_unknown_setting.loc[0, "discharge_care_level"] = "unknown"
    unknown_result = baseline_descriptives.baseline_characteristics_table(
        {"Combined": frame_with_unknown_setting}
    )
    assert unknown_result.loc["General ward", "Combined"] == "1 (50.0)"
    assert unknown_result.loc["High-dependency care", "Combined"] == "1 (50.0)"
    assert result.loc["Outcome within 7 days, n (%)", "Combined"] == ""
    assert result.loc["Composite outcome, ICU discharges", "Combined"] == "1 (50.0)"
    assert result.loc["Composite outcome, patients", "Combined"] == "1 (100.0)"
    assert result.loc["ICU readmission", "Combined"] == "1 (50.0)"
    assert result.loc["Death", "Combined"] == "0 (0.0)"
    assert result.loc["ICU readmission and death", "Combined"] == "0 (0.0)"


def test_specialty_counts_are_tidy_and_include_unknown():
    frame = _frame()
    frame.loc[1, "specialty"] = pd.NA

    result = baseline_descriptives.specialty_counts(frame)

    assert set(result["specialty"]) == {"cardiology", "unknown"}
    assert result["percentage"].sum() == 100


def test_cohort_summary_lists_missing_contract_columns():
    try:
        baseline_descriptives.extended_context_summary(pd.DataFrame())
    except KeyError as error:
        assert "age_at_admission" in str(error)
        assert "outcome" in str(error)
    else:
        raise AssertionError("expected a missing-column error")


def test_outcome_group_summaries_separate_events_without_identifiers():
    result = baseline_descriptives.outcome_group_summaries({"Combined": _frame()})

    assert set(result["outcome_group"]) == {"No event", "Event"}
    assert result.columns.tolist() == [
        "cohort",
        "outcome_group",
        "characteristic",
        "value",
    ]


def test_single_age_uses_fixed_ten_year_group_midpoint():
    assert baseline_descriptives._mean_sd(pd.Series([20])) == "25 (-)"
    assert baseline_descriptives._mean_sd(pd.Series([29])) == "25 (-)"
