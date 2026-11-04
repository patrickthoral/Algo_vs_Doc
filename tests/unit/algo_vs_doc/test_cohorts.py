import matplotlib
import pandas as pd
import pytest
from xml.etree import ElementTree as ET

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from algo_vs_doc import cohorts, enrolment_plots  # noqa: E402


@pytest.fixture
def data_dir(tmp_path):
    for hospital in cohorts.HOSPITALS:
        (tmp_path / hospital).mkdir()
    pd.DataFrame(
        [
            {
                "hospital": "LUMC",
                "icu_stay_id": 1,
                "lumc_only": "x",
            }
        ]
    ).to_csv(tmp_path / "LUMC" / "algo_doc_dataset.csv", index=False)
    pd.DataFrame(
        [
            {
                "hospital": "VUmc",
                "icu_stay_id": 1,
                "vumc_only": "y",
            }
        ]
    ).to_csv(tmp_path / "VUmc" / "algo_doc_dataset.csv", index=False)
    return str(tmp_path)


def test_dataset_path_uses_a_hospital_directory(tmp_path):
    assert cohorts.dataset_path(str(tmp_path), "LUMC") == str(
        tmp_path / "LUMC" / "algo_doc_dataset.csv"
    )


@pytest.mark.parametrize(("value", "expected"), [("lumc", "LUMC"), ("vUmc", "VUmc")])
def test_canonical_hospital_is_case_insensitive(value, expected):
    assert cohorts.canonical_hospital(value) == expected


def test_canonical_hospital_rejects_unknown_values():
    with pytest.raises(ValueError, match="hospital must be one of"):
        cohorts.canonical_hospital("other")


def test_load_preserves_the_verified_hospital_label(data_dir):
    frame = cohorts.load(data_dir, "LUMC")
    assert frame["hospital"].tolist() == ["LUMC"]


def test_load_rejects_a_missing_hospital_label(tmp_path):
    (tmp_path / "LUMC").mkdir()
    pd.DataFrame([{"icu_stay_id": 1}]).to_csv(
        tmp_path / "LUMC" / "algo_doc_dataset.csv",
        index=False,
    )
    with pytest.raises(KeyError, match="has no 'hospital' column"):
        cohorts.load(str(tmp_path), "LUMC")


@pytest.mark.parametrize("label", ["VUmc", None])
def test_load_rejects_an_incorrect_or_null_hospital_label(tmp_path, label):
    (tmp_path / "LUMC").mkdir()
    pd.DataFrame([{"hospital": label, "icu_stay_id": 1}]).to_csv(
        tmp_path / "LUMC" / "algo_doc_dataset.csv",
        index=False,
    )
    with pytest.raises(ValueError, match="wrong file is at that path"):
        cohorts.load(str(tmp_path), "LUMC")


def test_load_all_returns_hospitals_and_pooled_cohort_in_order(data_dir):
    loaded = cohorts.load_all(data_dir)
    assert list(loaded) == ["VUmc", "LUMC", "Combined"]
    assert len(loaded["Combined"]) == 2


def test_combined_keeps_a_column_only_one_hospital_has(data_dir):
    combined = cohorts.load_all(data_dir)["Combined"]
    assert {"lumc_only", "vumc_only"} <= set(combined.columns)
    assert combined.loc[combined["hospital"] == "VUmc", "lumc_only"].isna().all()


def test_load_all_can_omit_the_pooled_cohort(data_dir):
    assert list(cohorts.load_all(data_dir, include_combined=False)) == ["VUmc", "LUMC"]


def test_load_all_of_one_hospital_adds_no_combined_alias(data_dir):
    assert list(cohorts.load_all(data_dir, hospitals=["LUMC"])) == ["LUMC"]


def test_split_by_hospital_round_trips_the_pooled_cohort(data_dir):
    parts = cohorts.split_by_hospital(cohorts.load_all(data_dir)["Combined"])
    assert list(parts) == ["VUmc", "LUMC"]
    assert len(parts["LUMC"]) == 1


def test_apply_to_cohorts_runs_one_function_over_all_three(data_dir):
    sizes = cohorts.apply_to_cohorts(cohorts.load_all(data_dir), len)
    assert sizes == {"LUMC": 1, "VUmc": 1, "Combined": 2}


def test_enrolment_funnel_draws_one_bar_per_step():
    report = pd.DataFrame(
        [
            {"step": "total_icu_admissions", "excluded": 0, "remaining": 100},
            {"step": "died_on_icu", "excluded": 10, "remaining": 90},
            {"step": "icu_los_lt_12h", "excluded": 5, "remaining": 85},
        ]
    )
    ax = enrolment_plots.enrolment_funnel(report, title="LUMC")
    assert len(ax.patches) == 3
    assert ax.get_title() == "LUMC"
    assert ax.get_ylim()[0] > ax.get_ylim()[1]
    plt.close(ax.figure)


def _flow_reports():
    lumc = pd.DataFrame(
        [
            {"step": "total_icu_admissions", "excluded": 0, "remaining": 100},
            {"step": "died_on_icu", "excluded": 10, "remaining": 90},
            {"step": "icu_los_lt_12h", "excluded": 5, "remaining": 85},
        ]
    )
    vumc = pd.DataFrame(
        [
            {"step": "total_icu_admissions", "excluded": 0, "remaining": 50},
            {"step": "died_on_icu", "excluded": 5, "remaining": 45},
            {"step": "icu_los_lt_12h", "excluded": 4, "remaining": 41},
        ]
    )
    combined = lumc.copy()
    combined[["excluded", "remaining"]] += vumc[["excluded", "remaining"]]
    return {"LUMC": lumc, "VUmc": vumc, "Combined": combined}


def test_enrolment_flow_svg_is_byte_deterministic():
    reports = _flow_reports()
    first = enrolment_plots.enrolment_flow_svg(reports)
    second = enrolment_plots.enrolment_flow_svg(reports)
    assert first.encode("utf-8") == second.encode("utf-8")
    assert 'id="panel-lumc"' in first
    assert 'id="panel-vumc"' in first
    assert 'id="figure-border"' in first
    assert 'id="hospital-divider-1"' in first
    assert 'id="panel-lumc-label"' not in first
    assert 'id="panel-vumc-label"' not in first
    assert ">LUMC</text>" in first
    assert ">VUmc</text>" in first
    assert 'id="stage-lumc-total_icu_admissions"' in first
    assert 'id="excluded-vumc-died_on_icu"' in first
    assert "100 admissions to intensive" in first
    assert "90 survived to discharge" in first
    assert 'fill="#ffffff"' in first
    assert "font-weight" not in first
    assert "Combined" not in first
    ET.fromstring(first)


def test_enrolment_flow_svg_xml_builder_escapes_text():
    report = _flow_reports()["LUMC"]
    svg = enrolment_plots.enrolment_flow_svg({"A&B": report}, cohort_order=("A&B",))
    assert "A&amp;B" in svg
    ET.fromstring(svg)


def test_enrolment_flow_svg_omits_steps_that_exclude_nothing():
    reports = _flow_reports()
    for report in reports.values():
        report.loc[len(report)] = {
            "step": "not_yet_discharged",
            "excluded": 0,
            "remaining": int(report.iloc[-1]["remaining"]),
        }
    svg = enrolment_plots.enrolment_flow_svg(reports)
    assert "not_yet_discharged" not in svg


def test_enrolment_flow_svg_groups_eligibility_exclusions():
    steps = ["total_icu_admissions", *enrolment_plots.NOT_ELIGIBLE_STEPS]
    lumc = pd.DataFrame(
        {
            "step": steps,
            "excluded": [0, 4, 15, 37, 14, 5],
            "remaining": [100, 96, 81, 44, 30, 25],
        }
    )
    vumc = pd.DataFrame(
        {
            "step": steps,
            "excluded": [0, 1, 13, 14, 12, 4],
            "remaining": [80, 79, 66, 52, 40, 36],
        }
    )
    svg = enrolment_plots.enrolment_flow_svg({"LUMC": lumc, "VUmc": vumc})
    assert 'id="excluded-lumc-not_eligible"' in svg
    assert "75 not eligible" in svg
    assert "4 received palliative care" in svg
    assert "15 had a do-not-readmit decision" in svg
    assert "37 transferred to another hospital" in svg
    assert "14 were discharged home" in svg
    assert "5 had length of stay under 12 hours" in svg
    assert "stage-lumc-palliative_care" not in svg
    assert "stage-lumc-icu_los_lt_12h" not in svg


def test_enrolment_flow_svg_groups_missing_predictions():
    steps = ["total_icu_admissions", *enrolment_plots.MISSING_PREDICTION_STEPS]
    lumc = pd.DataFrame(
        {
            "step": steps,
            "excluded": [0, 197, 9],
            "remaining": [463, 266, 257],
        }
    )
    vumc = pd.DataFrame(
        {
            "step": steps,
            "excluded": [0, 15, 1],
            "remaining": [163, 148, 147],
        }
    )
    svg = enrolment_plots.enrolment_flow_svg({"LUMC": lumc, "VUmc": vumc})
    assert 'id="excluded-lumc-missing_predictions"' in svg
    assert "206 missing predictions" in svg
    assert "197 missing physician prediction" in svg
    assert "9 missing algorithm prediction" in svg
    assert "257 included in analysis" in svg
    assert "stage-lumc-missing_doc_prediction" not in svg


def test_enrolment_flow_svg_vertical_arrows_have_visible_shafts():
    svg = enrolment_plots.enrolment_flow_svg(_flow_reports())
    assert 'id="arrow-vumc-died_on_icu" d="M490 94 V174"' in svg
    assert 'id="excluded-vumc-died_on_icu-connector" d="M490 134 H652"' in svg


def test_enrolment_flow_svg_applies_text_padding():
    svg = enrolment_plots.enrolment_flow_svg(_flow_reports())
    assert 'id="excluded-vumc-died_on_icu-box" x="660"' in svg
    assert 'id="excluded-vumc-died_on_icu-label" x="680"' in svg
    assert (
        'id="stage-vumc-total_icu_admissions-box" x="350" y="42" width="280" height="52"'
        in svg
    )


def test_enrolment_flow_svg_splits_included_discharge_groups():
    counts = {
        "VUmc": {"nursing_ward": 31, "high_dependency_unit": 10},
        "LUMC": {"nursing_ward": 65, "high_dependency_unit": 20},
    }
    svg = enrolment_plots.enrolment_flow_svg(
        _flow_reports(), discharge_group_counts=counts
    )
    assert "31 discharged to" in svg
    assert "10 discharged to" in svg
    assert "high-dependency unit" in svg
    assert "general ward" in svg
    assert 'id="included-vumc-high-dependency-unit-box" x="170"' in svg
    assert 'id="included-vumc-nursing-ward-box" x="520"' in svg


def test_enrolment_flow_svg_rejects_inconsistent_discharge_group_counts():
    counts = {
        "VUmc": {"nursing_ward": 30, "high_dependency_unit": 10},
        "LUMC": {"nursing_ward": 65, "high_dependency_unit": 20},
    }
    with pytest.raises(ValueError, match="VUmc discharge groups total 40, expected 41"):
        enrolment_plots.enrolment_flow_svg(
            _flow_reports(), discharge_group_counts=counts
        )


def test_enrolment_flow_svg_uses_canonical_hospital_order():
    svg = enrolment_plots.enrolment_flow_svg(_flow_reports())
    assert svg.index('id="panel-vumc"') < svg.index('id="panel-lumc"')


def test_enrolment_flow_svg_rejects_inconsistent_counts():
    reports = _flow_reports()
    reports["LUMC"].loc[1, "remaining"] = 91
    with pytest.raises(ValueError, match="counts are inconsistent"):
        enrolment_plots.enrolment_flow_svg(reports)


def test_enrolment_flow_svg_rejects_incorrect_combined_counts():
    reports = _flow_reports()
    reports["Combined"].loc[2, "excluded"] += 1
    reports["Combined"].loc[2, "remaining"] -= 1
    with pytest.raises(ValueError, match="do not equal the site totals"):
        enrolment_plots.enrolment_flow_svg(
            reports, cohort_order=("LUMC", "VUmc", "Combined")
        )
