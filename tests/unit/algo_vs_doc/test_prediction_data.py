import os
import pandas as pd
import pytest

from algo_vs_doc import cohorts, prediction_data


def _row(hospital: str = "VUmc", **overrides) -> pd.DataFrame:
    row = {
        "hospital": hospital,
        "patient_id": 1,
        "icu_stay_id": 1,
        "icu_admitted_at": "2024-01-01 08:00",
        "icu_discharged_at": "2024-01-02 12:00",
        "doc_predicted_at": "2024-01-02 06:00",
        "algo_predicted_at_discharge": "2024-01-02 11:00",
        "algo_predicted_at_doc_prediction": "2024-01-02 06:00",
        "algo_prediction_at_discharge": 0.4,
        "algo_prediction_at_doc_prediction": 0.35,
        "doc_prediction": 0.6,
        "doc_prediction_cat": "high",
        "confidence": "medium",
        "doc_factors": "doc_factors:30||doc_factors:38",
        "doc_other_factors": "patient wants to go home",
        "age_at_admission": 70,
        "specialty": "cardiology",
        "discharge_care_level": "general_ward",
        "outcome": False,
    }
    row.update(overrides)
    return pd.DataFrame([row])


def _write_dataset(data_dir: str, hospital: str) -> None:
    path = cohorts.dataset_path(data_dir, hospital)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    _row(hospital=hospital).to_csv(path, index=False)


def test_normalize_preserves_the_enriched_contract():
    source = _row()
    result = prediction_data.normalize(source, "VUmc")

    assert result["doc_prediction"].iloc[0] == 0.6
    assert result["doc_prediction_cat"].iloc[0] == "high"
    assert result["doc_factors"].iloc[0] == "doc_factors:30||doc_factors:38"
    assert "doc_prediction_cat_id" not in result.columns
    pd.testing.assert_frame_equal(source, _row())


def test_normalize_adds_only_analysis_derived_columns():
    source = _row()
    result = prediction_data.normalize(source, "VUmc")
    added = set(result.columns) - set(source.columns)
    assert added == {
        "confidence_level",
        "doc_lead_time_hours",
        "algo_at_discharge_lead_time_hours",
        "algo_at_doc_prediction_lead_time_hours",
        "discharge_local_hour",
        "discharge_weekday",
        "discharge_is_weekend",
        "discharge_after_18h",
        "prediction_local_hour",
        "prediction_weekday",
        "prediction_is_weekend",
    }


def test_confidence_pooling_retains_the_recorded_level():
    for level, pooled in (
        ("low", "low/medium"),
        ("medium", "low/medium"),
        ("high", "high"),
    ):
        result = prediction_data.normalize(_row(confidence=level), "VUmc")
        assert result["confidence_level"].iloc[0] == level
        assert result["confidence"].iloc[0] == pooled


@pytest.mark.parametrize("column", ["doc_prediction_cat", "confidence"])
def test_unknown_ordinal_level_raises(column):
    with pytest.raises(KeyError, match="not covered by the mapping"):
        prediction_data.normalize(_row(**{column: "unknown"}), "VUmc")


def test_missing_ordinal_levels_remain_missing():
    result = prediction_data.normalize(
        _row(doc_prediction_cat=None, confidence=None),
        "VUmc",
    )
    assert pd.isna(result["doc_prediction_cat"].iloc[0])
    assert pd.isna(result["confidence_level"].iloc[0])
    assert pd.isna(result["confidence"].iloc[0])


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("doc_prediction", 1.1),
        ("algo_prediction_at_discharge", -0.1),
        ("algo_prediction_at_doc_prediction", "not-a-number"),
    ],
)
def test_predictions_must_be_proportions(column, value):
    with pytest.raises(ValueError, match="must contain proportions"):
        prediction_data.normalize(_row(**{column: value}), "VUmc")


def test_missing_enriched_column_raises():
    frame = _row().drop(columns="doc_factors")
    with pytest.raises(KeyError, match="rerun the Create Analysis Dataset notebook"):
        prediction_data.normalize(frame, "VUmc")


def test_hospital_label_must_match_the_requested_cohort():
    with pytest.raises(ValueError, match="expected only"):
        prediction_data.normalize(_row(hospital="LUMC"), "VUmc")


def test_unknown_hospital_raises():
    with pytest.raises(KeyError, match="unknown hospital"):
        prediction_data.normalize(_row(hospital="Other"), "Other")


def test_vumc_lead_times_use_the_flat_file_timestamps():
    result = prediction_data.normalize(_row(), "VUmc")
    assert result["doc_lead_time_hours"].iloc[0] == 6.0
    assert result["algo_at_doc_prediction_lead_time_hours"].iloc[0] == 6.0
    assert result["algo_at_discharge_lead_time_hours"].iloc[0] == 1.0


def test_lumc_preserves_the_existing_zero_discharge_lead_time_rule():
    result = prediction_data.normalize(_row(hospital="LUMC"), "LUMC")
    assert result["algo_at_discharge_lead_time_hours"].iloc[0] == 0.0


def test_lumc_missing_discharge_prediction_has_no_lead_time():
    result = prediction_data.normalize(
        _row(
            hospital="LUMC",
            algo_predicted_at_discharge=None,
            algo_prediction_at_discharge=None,
        ),
        "LUMC",
    )
    assert pd.isna(result["algo_at_discharge_lead_time_hours"].iloc[0])


def test_lead_time_longer_than_one_day_does_not_wrap():
    result = prediction_data.normalize(
        _row(doc_predicted_at="2024-01-01 06:00"),
        "VUmc",
    )
    assert result["doc_lead_time_hours"].iloc[0] == 30.0


def test_prediction_after_discharge_has_negative_lead_time():
    result = prediction_data.normalize(
        _row(doc_predicted_at="2024-01-02 14:00"),
        "VUmc",
    )
    assert result["doc_lead_time_hours"].iloc[0] == -2.0


def test_naive_timestamps_are_interpreted_as_amsterdam_local_wall_time():
    result = prediction_data.normalize(_row(), "VUmc")

    assert result["discharge_local_hour"].iloc[0] == 12
    assert result["discharge_weekday"].iloc[0] == 1
    assert bool(result["discharge_is_weekend"].iloc[0]) is False
    assert result["prediction_local_hour"].iloc[0] == 6


def test_timezone_aware_timestamps_are_converted_to_amsterdam_time():
    result = prediction_data.normalize(
        _row(
            icu_discharged_at="2024-07-01 10:00+00:00",
            doc_predicted_at="2024-07-01 08:00+00:00",
            algo_predicted_at_discharge="2024-07-01 09:00+00:00",
            algo_predicted_at_doc_prediction="2024-07-01 08:00+00:00",
        ),
        "VUmc",
    )

    assert result["discharge_local_hour"].iloc[0] == 12
    assert result["prediction_local_hour"].iloc[0] == 10


def test_discharge_after_18h_uses_strict_amsterdam_local_cutoff():
    at_cutoff = prediction_data.normalize(
        _row(icu_discharged_at="2024-01-02 18:00"), "VUmc"
    )
    after_cutoff = prediction_data.normalize(
        _row(icu_discharged_at="2024-01-02 18:01"), "VUmc"
    )

    assert bool(at_cutoff["discharge_after_18h"].iloc[0]) is False
    assert bool(after_cutoff["discharge_after_18h"].iloc[0]) is True


def test_missing_prediction_time_keeps_calendar_features_missing():
    result = prediction_data.normalize(
        _row(doc_predicted_at=None, doc_prediction=None),
        "VUmc",
    )

    assert pd.isna(result["prediction_local_hour"].iloc[0])
    assert pd.isna(result["prediction_weekday"].iloc[0])
    assert pd.isna(result["prediction_is_weekend"].iloc[0])


def test_load_cohorts_reads_enriched_flat_csv_files(tmp_path):
    for hospital in cohorts.HOSPITALS:
        _write_dataset(str(tmp_path), hospital)

    loaded = prediction_data.load_cohorts(str(tmp_path))

    assert list(loaded) == ["VUmc", "LUMC", "Combined"]
    assert len(loaded["Combined"]) == 2
    assert set(loaded["Combined"]["hospital"]) == {"LUMC", "VUmc"}
    assert "confidence_level" in loaded["Combined"].columns


def test_load_cohorts_can_load_one_hospital_without_combined(tmp_path):
    _write_dataset(str(tmp_path), "LUMC")
    loaded = prediction_data.load_cohorts(
        str(tmp_path),
        hospitals=("LUMC",),
        include_combined=False,
    )
    assert list(loaded) == ["LUMC"]
