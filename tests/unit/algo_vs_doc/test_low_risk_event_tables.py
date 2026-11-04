import pandas as pd

from algo_vs_doc import low_risk_event_tables


def _cohort():
    return pd.DataFrame(
        [
            {
                "icu_stay_id": "stay-17",
                "outcome": True,
                "doc_prediction_cat": "low",
                "algo_prediction_at_doc_prediction": 0.03,
                "has_icu_readmission_7days": True,
                "has_death_7days": False,
                "is_medical": True,
                "emergency_surgery": False,
                "elective_surgery": False,
            }
        ]
    )


def test_individual_entries_include_stable_internal_and_publication_keys():
    entries = low_risk_event_tables.individual_event_entries({"VUmc": _cohort()})

    assert entries.loc[0, "review_case_key"] == "VUmc:stay-17"
    assert entries.loc[0, "case_label"] == "V1"


def test_annotation_template_preserves_manual_edits(tmp_path):
    entries = low_risk_event_tables.individual_event_entries({"VUmc": _cohort()})
    path = tmp_path / "annotations.csv"
    annotations = low_risk_event_tables.update_annotation_template(entries, path)
    annotations.loc[0, "index_admission_diagnosis"] = "Cardiac arrest"
    annotations.to_csv(path, index=False)

    refreshed = low_risk_event_tables.update_annotation_template(entries, path)
    merged = low_risk_event_tables.add_manual_annotations(entries, refreshed)

    assert merged.loc[0, "index_admission_diagnosis"] == "Cardiac arrest"


def test_missing_reviewed_diagnoses_are_blank_in_publication_table():
    assert low_risk_event_tables._entry_value("index_admission_diagnosis", "") == ""
    assert low_risk_event_tables._entry_value("readmission_diagnosis", pd.NA) == ""
    assert low_risk_event_tables._entry_value("specialty", pd.NA) == "Not available"


def test_individual_event_length_of_stay_is_presented_in_days():
    assert low_risk_event_tables._entry_value("icu_los", 60) == "2.5"


def test_individual_event_table_uses_compact_headers():
    entries = low_risk_event_tables.individual_event_entries({"VUmc": _cohort()})
    table = low_risk_event_tables.individual_event_word_tables(entries)[0]

    assert {"M", "R", "D", "MV", "VA"}.issubset(table.data.columns)
    assert ("Prediction", ("Pp", "Pc", "M")) in table.header_groups
    assert ("Outcome", ("R", "D")) in table.header_groups
    assert ("Life Support", ("MV", "VA", "RRT")) in table.header_groups
    assert ("Diagnoses", ("Index stay", "Readmission")) in table.header_groups


def test_readmission_diagnosis_is_not_applicable_without_readmission():
    entries = low_risk_event_tables.individual_event_entries({"VUmc": _cohort()})
    entries.loc[0, "has_icu_readmission_7days"] = False
    entries.loc[0, "has_death_7days"] = True
    entries["readmission_diagnosis"] = ""

    table = low_risk_event_tables.individual_event_word_tables(entries)[0]

    assert table.data.loc[0, "R"] == "-"
    assert table.data.loc[0, "Readmission"] == "NA"


def test_individual_event_age_uses_fixed_ten_year_group_midpoints():
    assert low_risk_event_tables._entry_value("age_at_admission", 20) == "25"
    assert low_risk_event_tables._entry_value("age_at_admission", 29) == "25"
    assert low_risk_event_tables._entry_value("age_at_admission", 30) == "35"
    assert low_risk_event_tables._entry_value("age_at_admission", 87) == "85"


def test_individual_event_boolean_fields_use_plus_and_minus():
    assert low_risk_event_tables._entry_value("ventilation", True) == "+"
    assert low_risk_event_tables._entry_value("ventilation", False) == "-"


def test_severity_scores_remain_unrounded_integers():
    assert low_risk_event_tables._entry_value("sofa_admission", 7) == "7"
    assert low_risk_event_tables._entry_value("sofa_max", 13) == "13"
    assert low_risk_event_tables._entry_value("news_discharge", 4) == "4"


def test_physician_risk_categories_use_compact_labels():
    assert low_risk_event_tables._entry_value("doc_prediction_cat", "low") == "L"
    assert low_risk_event_tables._entry_value("doc_prediction_cat", "medium") == "M"
    assert low_risk_event_tables._entry_value("doc_prediction_cat", "high") == "H"


def test_individual_event_table_is_etable_11():
    entries = low_risk_event_tables.individual_event_entries({"VUmc": _cohort()})
    table = low_risk_event_tables.individual_event_word_tables(entries)[0]

    assert table.title.startswith("eTable 11.")
