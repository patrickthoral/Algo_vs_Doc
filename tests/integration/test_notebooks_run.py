"""Execute the public notebooks end to end against synthetic data."""

from __future__ import annotations

import csv
import os
import sys

os.environ.setdefault("JUPYTER_PLATFORM_DIRS", "1")

import papermill  # noqa: E402
import pytest  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests.fixtures import synthetic_cdm  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NOTEBOOKS = os.path.join(REPO_ROOT, "notebooks")

pytestmark = pytest.mark.slow


def _analysis_data_dir(project_root: str) -> str:
    return os.path.join(project_root, "analysis_data")


def _run(notebook_name: str, project_root: str, **parameters):
    """Execute one notebook with all data paths confined to a synthetic project."""
    output = os.path.join(project_root, "out", notebook_name)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    papermill.execute_notebook(
        os.path.join(NOTEBOOKS, notebook_name),
        output,
        parameters={
            "project_root": project_root,
            "data_dir": _analysis_data_dir(project_root),
            **parameters,
        },
        cwd=NOTEBOOKS,
        progress_bar=False,
    )
    return output


def _map_and_enrich(project_root: str, hospital: str) -> None:
    _run("1-create-analysis-dataset.ipynb", project_root, hospital=hospital)


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("synthetic_project"))
    synthetic_cdm.write(root, n_stays=40)
    for folder in (
        "figures",
        "results/summaries/LUMC",
        "results/summaries/VUmc",
        "results/summaries/Combined",
        "tables",
        "tables/LUMC",
        "tables/VUmc",
        "tables/Combined",
    ):
        os.makedirs(os.path.join(root, folder), exist_ok=True)
    return root


@pytest.fixture(scope="module")
def analysis_project(tmp_path_factory):
    """Create cohorts with enough events for grouped cross-validation."""
    root = str(tmp_path_factory.mktemp("synthetic_analysis_project"))
    synthetic_cdm.write(root, n_stays=100)
    for folder in (
        "figures",
        "results/summaries/LUMC",
        "results/summaries/VUmc",
        "results/summaries/Combined",
        "tables",
        "tables/LUMC",
        "tables/VUmc",
        "tables/Combined",
    ):
        os.makedirs(os.path.join(root, folder), exist_ok=True)
    return root


@pytest.mark.parametrize(
    ("parameter", "hospital"), [("lumc", "LUMC"), ("vUmc", "VUmc")]
)
def test_value_mapping_and_enrichment_write_the_flat_analysis_csv(
    project, parameter, hospital
):
    _run("1-create-analysis-dataset.ipynb", project, hospital=parameter)

    import pandas as pd

    from algo_vs_doc import cohorts

    csv_path = cohorts.dataset_path(_analysis_data_dir(project), hospital)
    dataset = pd.read_csv(csv_path)
    assert len(dataset) == 40
    assert dataset["hospital"].unique().tolist() == [hospital]
    assert "specialty_id" not in dataset
    assert "specialty" in dataset
    for column in (
        "discharge_location_id",
        "discharge_location_name",
        "ready_for_discharge",
        "patient_category_id",
        "patient_category_name",
        "readmission_origin_location_id",
        "readmission_origin_specialty_id",
        "readmission_origin_inferred",
        "discharge_destination",
        "post_discharge_status",
        "discharge_care_level",
        "index_episode_diagnoses_count",
        "n_census_at_discharge",
        "n_external_transfers_same_day",
        "n_elective_surgery_admissions_same_day",
    ):
        assert column in dataset
    for column in (
        "sofa_admission",
        "sofa_max",
        "sofa_discharge",
        "news_discharge",
        "ventilation",
        "vasopressors_inotropes",
        "renal_replacement_therapy",
        "copd",
    ):
        assert column in dataset.columns
    assert dataset["doc_predicted_at"].isna().any()
    assert dataset["doc_predicted_at"].notna().any()

    source_data_path = cohorts.dataset_path(os.path.join(project, "data"), hospital)
    assert not os.path.exists(source_data_path)


@pytest.mark.parametrize("hospital", ["LUMC", "VUmc"])
def test_value_mapping_canonicalizes_the_flat_csv(project, hospital):
    _map_and_enrich(project, hospital)

    from algo_vs_doc import cohorts

    dataset = cohorts.load(_analysis_data_dir(project), hospital)
    assert "specialty" in dataset
    assert "specialty_id" not in dataset
    assert "readmission_origin_specialty" in dataset
    assert set(dataset["sex"].dropna()) <= {"female", "male"}


def test_cohort_enrolment_loads_the_flat_csv_files(project):
    for hospital in ("VUmc", "LUMC"):
        _map_and_enrich(project, hospital)

    from algo_vs_doc import prediction_data

    loaded = prediction_data.load_cohorts(_analysis_data_dir(project))
    assert list(loaded) == ["VUmc", "LUMC", "Combined"]
    assert len(loaded["Combined"]) == 80
    _run("2-cohort-enrolment.ipynb", project)
    for table in (
        "prediction_availability_denominators.csv",
        "physician_prediction_availability_characteristics.csv",
        "hospital_specific_physician_prediction_availability_models.csv",
    ):
        assert os.path.exists(os.path.join(project, "results", "summaries", table))


def test_analysis_notebooks_run_all_cohorts(analysis_project):
    project = analysis_project
    for hospital in ("VUmc", "LUMC"):
        _map_and_enrich(project, hospital)

    _run("2-cohort-enrolment.ipynb", project)
    _run("4-analysis.ipynb", project, n_bootstraps=10)
    _run("3-baseline-characteristics.ipynb", project)
    _run("5-figures-and-tables.ipynb", project)

    for table in (
        "fixed_alert_rate_performance.csv",
        "configured_threshold_performance.csv",
        "exploratory_youden_threshold_performance.csv",
        "risk_category_events.csv",
        "full_eligible_model_at_discharge_alert_rates.csv",
        "full_eligible_model_at_discharge_performance.csv",
        "low_risk_event_characteristics.csv",
        os.path.join("Combined", "primary_auroc_difference.csv"),
        os.path.join("Combined", "exploratory_auroc_differences.csv"),
        os.path.join("Combined", "auroc_contrasts.csv"),
        os.path.join("Combined", "physician_factor_mapping.csv"),
        os.path.join("Combined", "model_feature_mapping.csv"),
    ):
        assert os.path.exists(os.path.join(project, "results", "summaries", table))
    for cohort in ("Combined", "VUmc", "LUMC"):
        for filename in (
            "auroc_contrast_bootstraps.csv",
            "auprc_contrast_bootstraps.csv",
            "calibration_curves.csv",
            "calibration_deciles.csv",
            "calibration_histograms.csv",
        ):
            assert os.path.exists(
                os.path.join(project, "results", "plot_data", cohort, filename)
            )

    assert os.path.exists(
        os.path.join(
            project, "tables", "Table 1. Descriptive statistics for ICU discharges.docx"
        )
    )
    assert os.path.exists(
        os.path.join(
            project,
            "results",
            "word_tables",
            "Combined",
            "Supplemental Digital Content.docx",
        )
    )
    assert os.path.exists(
        os.path.join(
            project,
            "figures",
            "Combined",
            "actionable_ai.svg",
        )
    )
    for cohort in ("VUmc", "LUMC", "Combined"):
        for filename in (
            "performance_table.csv",
            "auroc_contrasts.csv",
            "correlation_table.csv",
            "feature_coverage.csv",
        ):
            assert os.path.exists(
                os.path.join(project, "results", "summaries", cohort, filename)
            )
        assert os.path.exists(
            os.path.join(
                project,
                "figures",
                cohort,
                "actionable_ai.svg",
            )
        )
        assert os.path.exists(
            os.path.join(
                project,
                "results",
                "submission_figures",
                cohort,
                "actionable_ai.pdf",
            )
        )
    secondary_path = os.path.join(
        project, "results", "summaries", "Combined", "exploratory_auroc_differences.csv"
    )
    with open(secondary_path, newline="", encoding="utf-8") as handle:
        secondary_rows = list(csv.DictReader(handle))
    assert len(secondary_rows) == 8
    assert {(row["agent_1"], row["agent_2"]) for row in secondary_rows} >= {
        ("Physician (Category)", "Physician (Percentage)")
    }


def test_baseline_characteristics_uses_the_enriched_cohorts(project):
    for hospital in ("VUmc", "LUMC"):
        _map_and_enrich(project, hospital)

    _run("3-baseline-characteristics.ipynb", project)

    for table in (
        "baseline_characteristics.csv",
        "seven_day_event_types.csv",
        "event_vs_no_event_characteristics.csv",
        "clinical_subgroup_outcomes.csv",
    ):
        assert os.path.exists(
            os.path.join(project, "results", "summaries", "Combined", table)
        )


def test_notebooks_do_not_depend_on_previous_outputs(project, tmp_path):
    clean = str(tmp_path / "clean")
    synthetic_cdm.write(clean, n_stays=20)
    _run("1-create-analysis-dataset.ipynb", clean, hospital="VUmc")

    import pandas as pd

    from algo_vs_doc import cohorts

    csv_path = cohorts.dataset_path(_analysis_data_dir(clean), "VUmc")
    assert len(pd.read_csv(csv_path)) == 20
