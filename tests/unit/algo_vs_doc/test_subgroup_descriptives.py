import pandas as pd

from algo_vs_doc import subgroup_descriptives


def _frame():
    return pd.DataFrame(
        {
            "specialty": ["cardiology", "neurology", "cardiology", pd.NA],
            "discharge_care_level": [
                "general_ward",
                "high_dependency_unit",
                "general_ward",
                "unknown",
            ],
            "patient_category_name": ["Category 3", "Category 2", "Category 3", pd.NA],
            "is_medical": [True, False, False, False],
            "emergency_surgery": [False, True, False, False],
            "elective_surgery": [False, False, True, False],
            "icu_los": [12, 48, 100, 200],
            "outcome": [True, False, True, False],
        }
    )


def test_admission_type_is_mutually_exclusive_and_preserves_unknown():
    assert subgroup_descriptives.admission_type(_frame()).tolist() == [
        "medical",
        "emergency_surgery",
        "elective_surgery",
        "other_or_unknown",
    ]


def test_length_of_stay_boundaries_are_explicit():
    result = subgroup_descriptives.length_of_stay_group(pd.Series([23.9, 24, 72, 168]))
    assert result.astype(str).tolist() == [
        "<24 hours",
        "24-<72 hours",
        "72-<168 hours",
        ">=168 hours",
    ]


def test_tables_by_cohort_reports_counts_and_event_rates():
    result = subgroup_descriptives.tables_by_cohort({"Combined": _frame()})
    cardiology = result.loc[
        result["variable"].eq("admitting_specialty") & result["level"].eq("cardiology")
    ].iloc[0]
    assert cardiology["n"] == 2
    assert cardiology["events_n"] == 2
    assert cardiology["event_rate"] == 1
    assert "patient_category" in set(result["variable"])
    assert "discharge_care_level" in set(result["variable"])
