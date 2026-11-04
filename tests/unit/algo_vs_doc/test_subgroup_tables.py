from dataclasses import replace
import json
import os
from pathlib import Path

import pandas as pd
import pytest

from algo_vs_doc import subgroup_tables, word_tables


def counts():
    return pd.DataFrame(
        [
            ["VUmc", "admitting_specialty", "cardiology", 4, 1],
            ["LUMC", "admitting_specialty", "neurology", 2, 0],
            ["VUmc", "admission_type", "medical", 4, 1],
            ["VUmc", "icu_los_group", ">=168 hours", 2, 0],
            ["VUmc", "icu_los_group", "<24 hours", 2, 1],
            ["VUmc", "discharge_care_level", "general_ward", 4, 1],
        ],
        columns=["cohort", "variable", "level", "n", "events_n"],
    )


def test_subgroup_counts_order_and_empty_groups():
    table = subgroup_tables.subgroup_summary(counts())
    assert table.iloc[0].VUmc == "1/4 (25.0)"
    assert table.iloc[0].LUMC == "0/0 (NA)"
    assert (
        table.loc[table.Characteristic.eq("Admission type"), "LUMC"].item()
        == "Not available"
    )
    assert table.loc[
        table.Characteristic.eq("Index ICU length of stay"), "Subgroup"
    ].tolist() == [
        "<24 hours",
        "\u2265168 hours",
    ]
    assert len(table.Characteristic.unique()) == 3
    assert (
        "include the lower boundary and exclude the upper boundary"
        in (subgroup_tables.subgroup_word_table(table).notes[1])
    )
    with pytest.raises(ValueError, match="duplicate"):
        subgroup_tables.subgroup_summary(pd.concat([counts(), counts()]))


def test_new_notebook_sources_export_synthetic_tables(tmp_path):
    root = Path(__file__).resolve().parents[3]
    (tmp_path / "outputs/Combined").mkdir(parents=True)
    context = dict(
        os=os,
        pd=pd,
        clinical_subgroups=counts(),
        outputs_folder=str(tmp_path / "outputs"),
        cohort="Combined",
        report_cohort="Combined",
        report_outputs_folder=str(tmp_path / "outputs/Combined"),
        project_root=str(tmp_path),
    )
    notebook = json.loads(
        (root / "notebooks/3-baseline-characteristics.ipynb").read_text(
            encoding="utf-8"
        )
    )
    source = next(
        "".join(c.get("source", []))
        for c in notebook["cells"]
        if "subgroup_tables.subgroup_summary" in "".join(c.get("source", []))
    )
    exec(compile(source, "subgroup baseline cell", "exec"), context)
    assert (tmp_path / "outputs/Combined/clinical_subgroup_table.csv").exists()
    assert not (tmp_path / "tables/Combined/Clinical subgroup outcomes.docx").exists()
    notebook = json.loads(
        (root / "notebooks/5-figures-and-tables.ipynb").read_text(encoding="utf-8")
    )
    source = next(
        "".join(c.get("source", []))
        for c in notebook["cells"]
        if "supplemental_tables.append(subgroup_tables" in "".join(c.get("source", []))
    )
    source = source[source.index("subgroup_table = pd.read_csv(") :]
    source = source[: source.index("individual_case_dir =")]
    labels_source = next(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if "def publication_table_frame" in "".join(cell.get("source", []))
    )
    exec(labels_source[labels_source.index("cohort_labels =") :], context)
    context["replace"] = replace
    context.update(
        supplemental_tables=[],
        subgroup_tables=subgroup_tables,
        word_tables=word_tables,
        formatted_tables_dir=str(tmp_path / "tables/Combined"),
    )
    exec(compile(source, "subgroup presentation cell", "exec"), context)
    assert context["supplemental_tables"][0].title.startswith("eTable 4.")
