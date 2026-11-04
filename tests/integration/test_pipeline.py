"""Verify that enrichment and scoring use only synthetic CDM-lite tables."""

import pandas as pd
import pytest

from algo_vs_doc.scoring import pipeline

T = pd.Timestamp


def _measurement(rows):
    """Rows of ``(icu_stay_id, concept, measured_at, value)``, with strings as categorical values."""
    return pd.DataFrame(
        [
            {
                "icu_stay_id": icu_stay_id,
                "measurement_concept_name": concept,
                "measured_at": T(at),
                "value_as_number": float("nan") if isinstance(value, str) else value,
                "value_as_string": value if isinstance(value, str) else None,
            }
            for icu_stay_id, concept, at, value in rows
        ]
    )


def _drug_exposure(rows):
    """Rows of ``(icu_stay_id, drug, start, end, dose)``; an empty list still yields typed columns."""
    frame = pd.DataFrame(
        [
            {
                "icu_stay_id": icu_stay_id,
                "drug_concept_name": drug,
                "drug_exposure_start_datetime": T(start),
                "drug_exposure_end_datetime": T(end),
                "dose_value": dose,
            }
            for icu_stay_id, drug, start, end, dose in rows
        ]
    )
    if frame.empty:
        frame = pd.DataFrame(
            {
                "icu_stay_id": pd.Series(dtype="object"),
                "drug_concept_name": pd.Series(dtype="object"),
                "drug_exposure_start_datetime": pd.Series(dtype="datetime64[ns]"),
                "drug_exposure_end_datetime": pd.Series(dtype="datetime64[ns]"),
                "dose_value": pd.Series(dtype="float64"),
            }
        )
    return frame


EMPTY_DRUG_EXPOSURE = _drug_exposure([])


@pytest.fixture
def icu_stays():
    return pd.DataFrame(
        [
            {
                "icu_stay_id": "s1",
                "icu_admitted_at": T("2024-01-01 00:00"),
                "icu_discharged_at": T("2024-01-03 00:00"),
                "elective_surgery": False,
            },
            {
                "icu_stay_id": "s2",  # nothing measured at all
                "icu_admitted_at": T("2024-01-01 00:00"),
                "icu_discharged_at": T("2024-01-02 00:00"),
                "elective_surgery": False,
            },
        ]
    )


def test_severity_summary_runs_from_the_intermediate_tables_alone(icu_stays):
    measurement = _measurement(
        [
            # respiratory: PaO2 12 kPa with FiO2 0.5 -> PF ~180; PEEP present -> ventilated -> 3
            ("s1", "art_po2", "2024-01-01 12:00", 12.0),
            ("s1", "fio2_fraction", "2024-01-01 12:05", 0.5),
            ("s1", "peep", "2024-01-01 12:02", 8.0),
            # neurology: fully alert -> 0
            ("s1", "gcs_eye", "2024-01-01 08:00", 4),
            ("s1", "gcs_motor", "2024-01-01 08:00", 6),
            ("s1", "gcs_verbal", "2024-01-01 08:00", 5),
            # liver: bilirubin 45 -> 2 ; coagulation: platelets 40 -> 3
            ("s1", "bilirubin", "2024-01-01 09:00", 45.0),
            ("s1", "platelets", "2024-01-01 09:00", 40.0),
            # kidneys: creatinine 200 -> 2, but 200 mL over 12h extrapolates to 400/24h -> 3
            ("s1", "creatinine", "2024-01-01 09:00", 200.0),
            ("s1", "urine_output", "2024-01-01 06:00", 80.0),
            ("s1", "urine_output", "2024-01-01 18:00", 120.0),
            # cardiovascular: MAP 65 -> 1, but sustained norepinephrine below dominates
            ("s1", "map", "2024-01-01 11:00", 65.0),
        ]
    )
    drug_exposure = _drug_exposure(
        [
            ("s1", "norepinephrine", "2024-01-01 06:00", "2024-01-01 09:00", 0.2)
        ]  # 3h > 1h gate, >0.1 -> 4
    )

    summary = pipeline.severity_summary(
        icu_stays, measurement, drug_exposure
    ).set_index("icu_stay_id")

    # resp 3 + neuro 0 + cardvasc 4 + liver 2 + coagulation 3 + kidneys 3
    assert summary.loc["s1", "sofa_admission"] == 15
    assert summary.loc["s1", "sofa_max"] == 15
    # every stay appears, and one with nothing measured scores 0 rather than going missing
    assert summary.loc["s2", "sofa_admission"] == 0
    assert summary.loc["s2", "sofa_max"] == 0
    assert set(summary.columns) >= set(pipeline.SEVERITY_COLUMNS)


def test_severity_summary_computes_news_at_discharge_from_vitals_and_gcs(icu_stays):
    measurement = _measurement(
        [
            # each vital charted at its own time, as they really are
            ("s1", "respiratory_rate", "2024-01-02 20:00", 26),  # >=25 -> 3
            ("s1", "spo2", "2024-01-02 21:00", 92),  # <=93 -> 2
            ("s1", "temperature", "2024-01-02 21:30", 35.0),  # <=35 -> 3
            ("s1", "systolic_blood_pressure", "2024-01-02 22:00", 95),  # <=100 -> 2
            ("s1", "heart_rate", "2024-01-02 22:30", 135),  # >=131 -> 3
            (
                "s1",
                "o2_flow_rate",
                "2024-01-02 23:00",
                4.0,
            ),  # any supplemental oxygen -> 2
            # GCS 12 -> not alert -> AVPU 3
            ("s1", "gcs_eye", "2024-01-02 19:00", 3),
            ("s1", "gcs_motor", "2024-01-02 19:00", 5),
            ("s1", "gcs_verbal", "2024-01-02 19:00", 4),
        ]
    )
    summary = pipeline.severity_summary(
        icu_stays, measurement, EMPTY_DRUG_EXPOSURE
    ).set_index("icu_stay_id")
    assert summary.loc["s1", "news_discharge"] == 3 + 2 + 3 + 2 + 3 + 2 + 3
    assert summary.loc["s2", "news_discharge"] == 0


def test_severity_summary_sofa_max_reflects_the_worst_day_not_the_last(icu_stays):
    measurement = _measurement(
        [
            ("s1", "platelets", "2024-01-01 09:00", 10.0),  # day 0: <20 -> 4
            ("s1", "platelets", "2024-01-02 09:00", 400.0),  # day 1: normal -> 0
        ]
    )
    summary = pipeline.severity_summary(
        icu_stays, measurement, EMPTY_DRUG_EXPOSURE
    ).set_index("icu_stay_id")
    assert summary.loc["s1", "sofa_admission"] == 4
    assert summary.loc["s1", "sofa_max"] == 4
    assert summary.loc["s1", "sofa_discharge"] == 0  # recovered by discharge


def test_severity_summary_applies_the_sedation_gate_from_drug_exposure(icu_stays):
    # a GCS of 3 taken while a sedative was running is not neurological dysfunction, so it
    # must be excluded rather than scored 4
    measurement = _measurement(
        [
            ("s1", "gcs_eye", "2024-01-01 10:00", 1),
            ("s1", "gcs_motor", "2024-01-01 10:00", 1),
            ("s1", "gcs_verbal", "2024-01-01 10:00", 1),
        ]
    )
    sedated = _drug_exposure(
        [("s1", "sedative", "2024-01-01 09:00", "2024-01-01 12:00", 5.0)]
    )
    assert (
        pipeline.severity_summary(icu_stays, measurement, sedated)
        .set_index("icu_stay_id")
        .loc["s1", "sofa_max"]
        == 0
    )
    # the same reading with no sedative running does score
    assert (
        pipeline.severity_summary(icu_stays, measurement, EMPTY_DRUG_EXPOSURE)
        .set_index("icu_stay_id")
        .loc["s1", "sofa_max"]
        == 4
    )


def test_severity_summary_forces_a_normal_day0_neurology_score_for_elective_surgery(
    icu_stays,
):
    measurement = _measurement(
        [
            ("s1", "gcs_eye", "2024-01-01 10:00", 1),
            ("s1", "gcs_motor", "2024-01-01 10:00", 1),
            ("s1", "gcs_verbal", "2024-01-01 10:00", 1),
        ]
    )
    icu_stays.loc[icu_stays["icu_stay_id"] == "s1", "elective_surgery"] = True
    summary = pipeline.severity_summary(
        icu_stays, measurement, EMPTY_DRUG_EXPOSURE
    ).set_index("icu_stay_id")
    # post-operative sedation on the admission day is not neurological dysfunction
    assert summary.loc["s1", "sofa_admission"] == 0


def test_severity_summary_needs_no_hospital_specific_column(icu_stays):
    """The inputs are the CDM tables and nothing else - the whole point of the repo split."""
    measurement = _measurement([("s1", "platelets", "2024-01-01 09:00", 40.0)])
    summary = pipeline.severity_summary(icu_stays, measurement, EMPTY_DRUG_EXPOSURE)
    assert list(summary.columns) == ["icu_stay_id", *pipeline.SEVERITY_COLUMNS]
    # no identifier from either source system leaks into the output
    for hospital_column in ("admission_id", "pat_id", "parameter_id", "flo_meas_id"):
        assert hospital_column not in summary.columns
