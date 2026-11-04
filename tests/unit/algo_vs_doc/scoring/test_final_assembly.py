import pandas as pd

from algo_vs_doc.scoring import final_assembly


def _base_row(**overrides):
    row = {
        "icu_discharged_at": pd.Timestamp("2024-01-05 10:00"),
        "died_on_icu": False,
        "palliative_care": False,
        "do_not_readmit": False,
        "discharge_destination": "ward",
        "discharge_care_level": "general_ward",
        "icu_los_lt_12h": False,
        "doc_predicted_at": pd.Timestamp("2024-01-04 09:00"),
        "algo_predicted_at_doc_prediction": pd.Timestamp("2024-01-04 09:05"),
    }
    row.update(overrides)
    return row


def test_apply_enrolment_funnel_keeps_a_row_that_fails_no_step():
    algo_doc = pd.DataFrame([_base_row()])
    remaining, report = final_assembly.apply_enrolment_funnel(algo_doc)
    assert len(remaining) == 1
    assert report.set_index("step").loc["missing_algo_prediction", "remaining"] == 1


def test_apply_enrolment_funnel_excludes_each_step_independently():
    rows = [
        _base_row(),  # survives every step
        _base_row(icu_discharged_at=pd.NaT),  # not_yet_discharged
        _base_row(died_on_icu=True),
        _base_row(palliative_care=True),
        _base_row(do_not_readmit=True),
        _base_row(discharge_destination="external_transfer"),
        _base_row(discharge_destination="home"),
        _base_row(icu_los_lt_12h=True),
        _base_row(discharge_destination="ward"),
        _base_row(doc_predicted_at=pd.NaT),
        _base_row(algo_predicted_at_doc_prediction=pd.NaT),
    ]
    algo_doc = pd.DataFrame(rows)
    remaining, report = final_assembly.apply_enrolment_funnel(algo_doc)

    assert len(remaining) == 2
    report_by_step = report.set_index("step")["excluded"]
    assert report_by_step["not_yet_discharged"] == 1
    assert report_by_step["died_on_icu"] == 1
    assert report_by_step["palliative_care"] == 1
    assert report_by_step["do_not_readmit_policy"] == 1
    assert report_by_step["transferred_to_other_hospital"] == 1
    assert report_by_step["discharged_home"] == 1
    assert report_by_step["icu_los_lt_12h"] == 1
    assert "discharged_to_high_dependency_unit" not in report_by_step
    assert report_by_step["missing_doc_prediction"] == 1
    assert report_by_step["missing_algo_prediction"] == 1
    assert report.iloc[0]["remaining"] == len(rows)  # leading total_icu_admissions row


def test_apply_enrolment_funnel_does_not_double_count_a_row_excluded_by_an_earlier_step():
    # this row fails both the first step (not_yet_discharged) and a later one (died_on_icu) -
    # it must only be counted once, under the earlier step, since later steps only see what
    # remains after earlier exclusions
    algo_doc = pd.DataFrame([_base_row(icu_discharged_at=pd.NaT, died_on_icu=True)])
    remaining, report = final_assembly.apply_enrolment_funnel(algo_doc)

    assert len(remaining) == 0
    report_by_step = report.set_index("step")["excluded"]
    assert report_by_step["not_yet_discharged"] == 1
    assert report_by_step["died_on_icu"] == 0


def test_apply_enrolment_funnel_report_remaining_counts_decrease_monotonically():
    rows = [
        _base_row(),
        _base_row(died_on_icu=True),
        _base_row(discharge_destination="ward"),
    ]
    algo_doc = pd.DataFrame(rows)
    _, report = final_assembly.apply_enrolment_funnel(algo_doc)
    assert report["remaining"].tolist() == sorted(
        report["remaining"].tolist(), reverse=True
    )
    assert report["remaining"].iloc[-1] == 2


def test_apply_enrolment_funnel_preserves_surviving_row_identity():
    keep = _base_row()
    keep["icu_stay_id"] = "keep-me"
    drop = _base_row(died_on_icu=True)
    drop["icu_stay_id"] = "drop-me"
    algo_doc = pd.DataFrame([keep, drop])
    remaining, _ = final_assembly.apply_enrolment_funnel(algo_doc)
    assert remaining["icu_stay_id"].tolist() == ["keep-me"]


def test_apply_enrolment_funnel_accepts_custom_steps():
    algo_doc = pd.DataFrame([_base_row(died_on_icu=True)])
    custom_steps = [("died_on_icu", lambda df: df["died_on_icu"] == True)]  # noqa: E712
    remaining, report = final_assembly.apply_enrolment_funnel(
        algo_doc, steps=custom_steps
    )
    assert len(remaining) == 0
    assert report["step"].tolist() == ["total_icu_admissions", "died_on_icu"]
