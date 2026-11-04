import json
from pathlib import Path

import pandas as pd

from algo_vs_doc import ward_descriptives


def synthetic_cohorts():
    frame = pd.DataFrame(
        {
            "has_icu_readmission_7days": [True, False, True],
            "has_death_7days": [False, True, True],
            "age_at_admission": [60, 70, 80],
            "sex": ["male", "female", "female"],
            "specialty": ["cardiology", "neurology", "general_surgery"],
            "is_medical": [True, True, False],
            "elective_surgery": [False, False, False],
            "emergency_surgery": [False, False, True],
            "discharge_care_level": [
                "general_ward",
                "high_dependency_unit",
                "general_ward",
            ],
            "icu_los": [24, 48, 72],
            "ventilation": [True, False, True],
            "vasopressors_inotropes": [False, True, True],
            "renal_replacement_therapy": [False, False, True],
            "sofa_admission": [4, 5, 6],
            "sofa_max": [6, 7, 8],
            "sofa_discharge": [2, 3, 4],
            "news_discharge": [1, 2, 3],
            "readmission_origin_specialty": ["neurology", None, "cardiology"],
            "readmission_origin_unit_type": ["nursing_ward", None, None],
        }
    )
    return {"Combined": frame, "VUmc": frame.iloc[:2], "LUMC": frame.iloc[2:]}


def test_single_age_uses_fixed_ten_year_group_midpoint():
    assert ward_descriptives._mean_sd(pd.Series([60])) == "65 (-)"
    assert ward_descriptives._mean_sd(pd.Series([69])) == "65 (-)"


def test_event_summary_uses_outcomes_as_columns_and_omits_ascertainment():
    result = ward_descriptives.ward_summary(synthetic_cohorts())
    indexed = result.set_index("Characteristic")

    assert indexed.loc["Event discharges, n", "Both Hospitals Readmission only"] == "1"
    assert indexed.loc["Event discharges, n", "Both Hospitals Death only"] == "1"
    assert (
        indexed.loc["Event discharges, n", "Both Hospitals Readmission and death"]
        == "1"
    )
    cardiology = result.loc[result["Characteristic"].eq("Cardiology")]
    assert cardiology["Both Hospitals Readmission only"].tolist() == [
        "1 (100.0)",
        "0 (0.0)",
    ]
    assert cardiology["Both Hospitals Death only"].iloc[-1] == "-"
    general_ward = result.loc[result["Characteristic"].eq("General ward")]
    origin_general_ward = general_ward.iloc[-1]
    assert origin_general_ward["Both Hospitals Readmission only"] == "1 (100.0)"
    assert origin_general_ward["LUMC Readmission and death"] == "-"
    assert not result["Characteristic"].str.contains("Nursing ward").any()
    assert not result["Characteristic"].str.contains("ascertainment", case=False).any()
    unknown_setting = synthetic_cohorts()
    unknown_setting["Combined"] = unknown_setting["Combined"].copy()
    unknown_setting["Combined"].loc[0, "discharge_care_level"] = "unknown"
    unknown_result = ward_descriptives.ward_summary(unknown_setting).set_index(
        "Characteristic"
    )
    assert (
        unknown_result.loc["General ward", "Both Hospitals Readmission only"].iloc[0]
        == "1 (100.0)"
    )


def test_event_word_table_groups_columns_by_cohort():
    specification = ward_descriptives.ward_word_table(
        ward_descriptives.ward_summary(synthetic_cohorts())
    )

    assert specification.title.startswith("eTable 3.")
    assert specification.landscape
    assert specification.header_groups[0][0] == "Both Hospitals"
    assert specification.header_labels[1:4] == tuple(
        label for label, _, _ in ward_descriptives.EVENT_GROUPS
    )
    assert specification.column_widths == ()
    assert specification.preferred_column_widths == {0: 2.14}
    labels = specification.data["Characteristic"].tolist()
    assert all(label == label.strip() for label in labels)
    surgical_position = labels.index("Surgical")
    admitting_cardiology_position = labels.index("Cardiology")
    origin_heading_position = labels.index("Readmission-origin specialty, n (%)")
    origin_cardiology_position = labels.index("Cardiology", origin_heading_position)
    assert specification.row_indent_levels_by_position[surgical_position] == 1
    assert (
        specification.row_indent_levels_by_position[admitting_cardiology_position] == 2
    )
    assert specification.row_indent_levels_by_position[origin_cardiology_position] == 1
    assert (
        "* Readmission-origin care setting was not available for LUMC."
        in specification.notes
    )


def test_notebook_cells_execute_with_synthetic_data(tmp_path):
    import os

    from algo_vs_doc import word_tables

    root = Path(__file__).resolve().parents[3]
    notebook = json.loads(
        (root / "notebooks/3-baseline-characteristics.ipynb").read_text(
            encoding="utf-8"
        )
    )
    source = next(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if "ward_descriptives.ward_summary" in "".join(cell.get("source", []))
    )
    (tmp_path / "outputs/Combined").mkdir(parents=True)
    context = {
        "os": os,
        "pd": pd,
        "baseline_cohorts": synthetic_cohorts(),
        "report_outputs_folder": str(tmp_path / "outputs/Combined"),
    }
    exec(compile(source, "event characteristics cell", "exec"), context)
    assert (tmp_path / "outputs/Combined/event_characteristics_by_outcome.csv").exists()

    notebook = json.loads(
        (root / "notebooks/5-figures-and-tables.ipynb").read_text(encoding="utf-8")
    )
    source = next(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if "supplemental_tables.append(ward_descriptives"
        in "".join(cell.get("source", []))
    )
    source = source[source.index("from algo_vs_doc import ward_descriptives") :]
    source = source[: source.index("from algo_vs_doc import subgroup_tables")]
    context.update(
        supplemental_tables=[],
        word_tables=word_tables,
        outputs_folder=str(tmp_path / "outputs"),
        report_cohort="Combined",
    )
    exec(compile(source, "event presentation cell", "exec"), context)
    assert context["supplemental_tables"][0].title.startswith("eTable 3.")
