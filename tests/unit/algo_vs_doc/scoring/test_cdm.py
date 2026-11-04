import pandas as pd
import pytest

from algo_vs_doc.scoring import cdm


def test_pivot_measurements_puts_each_concept_in_its_own_named_column():
    measurement = pd.DataFrame(
        [
            {
                "icu_stay_id": "s1",
                "measurement_concept_name": "platelets",
                "measured_at": pd.Timestamp("2024-01-01 08:00"),
                "value_as_number": 140.0,
                "value_as_string": None,
            },
            {
                "icu_stay_id": "s1",
                "measurement_concept_name": "bilirubin",
                "measured_at": pd.Timestamp("2024-01-01 09:00"),
                "value_as_number": 25.0,
                "value_as_string": None,
            },
        ]
    )
    wide = cdm.pivot_measurements(measurement)
    assert set(wide.columns) == {"icu_stay_id", "measured_at", "platelets", "bilirubin"}
    platelets_row = wide.loc[wide["platelets"].notna()].iloc[0]
    assert platelets_row["platelets"] == 140.0
    assert pd.isna(platelets_row["bilirubin"])


def test_pivot_measurements_pivots_a_categorical_concept_from_value_as_string():
    measurement = pd.DataFrame(
        [
            {
                "icu_stay_id": "s1",
                "measurement_concept_name": "o2_device",
                "measured_at": pd.Timestamp("2024-01-01 08:00"),
                "value_as_number": None,
                "value_as_string": "low_flow_nasal_cannula",
            },
            {
                "icu_stay_id": "s1",
                "measurement_concept_name": "o2_flow_rate",
                "measured_at": pd.Timestamp("2024-01-01 08:00"),
                "value_as_number": 2.0,
                "value_as_string": None,
            },
        ]
    )
    wide = cdm.pivot_measurements(measurement)
    row = wide.iloc[0]
    assert row["o2_device"] == "low_flow_nasal_cannula"
    assert row["o2_flow_rate"] == 2.0


def test_pivot_measurements_handles_no_categorical_rows_at_all():
    # regression: an empty categorical slice must not break the merge/columns
    measurement = pd.DataFrame(
        [
            {
                "icu_stay_id": "s1",
                "measurement_concept_name": "heart_rate",
                "measured_at": pd.Timestamp("2024-01-01 08:00"),
                "value_as_number": 90.0,
                "value_as_string": None,
            }
        ]
    )
    wide = cdm.pivot_measurements(measurement)
    assert list(wide.columns) == ["icu_stay_id", "measured_at", "heart_rate"]


def test_drug_exposure_intervals_filters_to_one_drug_and_renames_columns():
    drug_exposure = pd.DataFrame(
        [
            {
                "icu_stay_id": "s1",
                "drug_concept_name": "norepinephrine",
                "drug_exposure_start_datetime": pd.Timestamp("2024-01-01 08:00"),
                "drug_exposure_end_datetime": pd.Timestamp("2024-01-01 09:00"),
                "dose_value": 0.2,
            },
            {
                "icu_stay_id": "s1",
                "drug_concept_name": "dopamine",
                "drug_exposure_start_datetime": pd.Timestamp("2024-01-01 08:00"),
                "drug_exposure_end_datetime": pd.Timestamp("2024-01-01 09:00"),
                "dose_value": 3.0,
            },
        ]
    )
    result = cdm.drug_exposure_intervals(drug_exposure, "norepinephrine")
    assert list(result.columns) == ["icu_stay_id", "started_at", "ended_at", "dose"]
    assert len(result) == 1
    assert result.iloc[0]["dose"] == 0.2


def test_condition_flags_marks_present_conditions_true_and_everything_else_false():
    condition_occurrence = pd.DataFrame(
        [
            {"icu_stay_id": "s1", "condition_concept_name": "copd"},
            {"icu_stay_id": "s1", "condition_concept_name": "cirrhosis"},
        ]
    )
    icu_stay_id = pd.Series(["s1", "s2"])
    result = cdm.condition_flags(
        condition_occurrence,
        icu_stay_id,
        concepts=frozenset({"copd", "cirrhosis", "diabetes_mellitus"}),
    ).set_index("icu_stay_id")
    assert bool(result.loc["s1", "copd"]) is True
    assert bool(result.loc["s1", "cirrhosis"]) is True
    assert bool(result.loc["s1", "diabetes_mellitus"]) is False


def test_condition_flags_gives_a_stay_with_zero_conditions_an_all_false_row_not_a_missing_row():
    condition_occurrence = pd.DataFrame(
        [{"icu_stay_id": "s1", "condition_concept_name": "copd"}]
    )
    icu_stay_id = pd.Series(["s1", "s2"])
    result = cdm.condition_flags(
        condition_occurrence, icu_stay_id, concepts=frozenset({"copd"})
    )
    assert set(result["icu_stay_id"]) == {"s1", "s2"}
    assert bool(result.set_index("icu_stay_id").loc["s2", "copd"]) is False


def test_estimate_device_fio2_inputs_valid_flags_known_devices_only():
    o2_device = pd.Series(["room_air", "high_flow_optiflow", "o2_face_mask"])
    result = cdm.estimate_device_fio2_inputs_valid(o2_device)
    assert result.tolist() == [True, False, True]


def test_wide_conditions_to_condition_occurrence_from_boolean_columns_has_no_start_datetime():
    wide = pd.DataFrame(
        [
            {"icu_stay_id": "s1", "copd": True, "cirrhosis": False},
            {"icu_stay_id": "s2", "copd": False, "cirrhosis": True},
        ]
    )
    result = cdm.wide_conditions_to_condition_occurrence(wide)
    result = result.set_index(["icu_stay_id", "condition_concept_name"])
    assert ("s1", "copd") in result.index
    assert ("s2", "cirrhosis") in result.index
    assert (
        "s1",
        "cirrhosis",
    ) not in result.index  # False is dropped, not kept as a row
    assert pd.isna(result.loc[("s1", "copd"), "condition_start_datetime"])


def test_wide_conditions_to_condition_occurrence_from_datetime_columns_keeps_the_timestamp():
    wide = pd.DataFrame(
        [
            {
                "icu_stay_id": "s1",
                "diabetes_mellitus": pd.Timestamp("2024-01-01"),
                "copd": pd.NaT,
            },
            {"icu_stay_id": "s2", "diabetes_mellitus": pd.NaT, "copd": pd.NaT},
        ]
    )
    result = cdm.wide_conditions_to_condition_occurrence(wide)
    assert (
        len(result) == 1
    )  # only s1's diabetes_mellitus is present; everything else is NaT
    row = result.iloc[0]
    assert row["icu_stay_id"] == "s1"
    assert row["condition_concept_name"] == "diabetes_mellitus"
    assert row["condition_start_datetime"] == pd.Timestamp("2024-01-01")


def test_wide_conditions_to_condition_occurrence_rejects_a_non_boolean_non_datetime_column():
    wide = pd.DataFrame(
        [{"icu_stay_id": "s1", "copd": 1}]
    )  # a raw 0/1 int column, not yet cast
    with pytest.raises(TypeError):
        cdm.wide_conditions_to_condition_occurrence(wide)


def test_wide_conditions_to_condition_occurrence_round_trips_through_condition_flags():
    wide = pd.DataFrame(
        [
            {"icu_stay_id": "s1", "copd": True, "cirrhosis": False},
            {"icu_stay_id": "s2", "copd": False, "cirrhosis": False},
        ]
    )
    occurrence = cdm.wide_conditions_to_condition_occurrence(wide)
    round_tripped = cdm.condition_flags(
        occurrence, pd.Series(["s1", "s2"]), concepts=frozenset({"copd", "cirrhosis"})
    ).set_index("icu_stay_id")
    assert bool(round_tripped.loc["s1", "copd"]) is True
    assert bool(round_tripped.loc["s1", "cirrhosis"]) is False
    assert bool(round_tripped.loc["s2", "copd"]) is False


def test_diagnosis_windows_create_ordered_relative_timelines():
    stays = pd.DataFrame(
        {
            "icu_stay_id": ["s1"],
            "patient_id": [1],
            "icu_admitted_at": pd.to_datetime(["2024-01-08 12:00"]),
            "icu_discharged_at": pd.to_datetime(["2024-01-10 12:00"]),
            "icu_readmission_time": pd.to_datetime(["2024-01-12 12:00"]),
            "has_icu_readmission_7days": [True],
        }
    )
    diagnoses = pd.DataFrame(
        {
            "patient_id": [1] * 5,
            "diagnosis_source_code": ["old", "pre", "during", "before_re", "after_re"],
            "diagnosis_source_label": ["Old", "Pre", "During", "Before re", "After re"],
            "diagnosed_at": pd.to_datetime(
                [
                    "2023-12-31 12:00",
                    "2024-01-08 10:00",
                    "2024-01-09 12:00",
                    "2024-01-12 10:00",
                    "2024-01-12 14:00",
                ]
            ),
        }
    )

    result = cdm.diagnosis_windows(
        diagnoses, stays, condition_concepts=frozenset()
    ).iloc[0]

    assert result.index_episode_diagnoses_count == 2
    assert result.index_episode_diagnoses_timeline == "-2.0 h: Pre\n+24.0 h: During"
    assert result.readmission_episode_diagnoses_count == 2
    assert (
        result.readmission_episode_diagnoses_timeline
        == "-2.0 h: Before re\n+2.0 h: After re"
    )


def test_diagnosis_windows_add_admission_conditions_at_time_zero():
    stays = pd.DataFrame(
        {
            "icu_stay_id": ["s1", "s2"],
            "patient_id": [1, 1],
            "icu_admitted_at": pd.to_datetime(["2024-01-08 12:00", "2024-01-12 12:00"]),
            "icu_discharged_at": pd.to_datetime(
                ["2024-01-10 12:00", "2024-01-13 12:00"]
            ),
            "icu_readmission_time": pd.to_datetime(["2024-01-12 12:00", None]),
            "has_icu_readmission_7days": [True, False],
            "cpr": [True, False],
            "confirmed_infection": [False, True],
        }
    )
    diagnoses = pd.DataFrame(
        columns=[
            "patient_id",
            "diagnosis_source_code",
            "diagnosis_source_label",
            "diagnosed_at",
        ]
    )
    diagnoses["diagnosed_at"] = pd.to_datetime(diagnoses["diagnosed_at"])

    result = cdm.diagnosis_windows(
        diagnoses,
        stays,
        condition_concepts=frozenset({"cpr", "confirmed_infection"}),
    ).set_index("icu_stay_id")

    assert result.loc["s1", "index_episode_diagnoses_timeline"] == (
        "+0.0 h: CPR (admission condition)"
    )
    assert result.loc["s1", "readmission_episode_diagnoses_timeline"] == (
        "+0.0 h: Confirmed infection (admission condition)"
    )


def _prediction(rows):
    """Rows of ``(icu_stay_id, predictor, timing, predicted_at, value, features)``."""
    return pd.DataFrame(
        [
            {
                "icu_stay_id": s,
                "predictor": who,
                "prediction_timing": timing,
                "predicted_at": pd.Timestamp(at),
                "prediction_value": value,
                "prediction_category": cat,
                "confidence": conf,
                "factors": factors,
                "other_factors": None,
                "features": feats,
            }
            for s, who, timing, at, value, cat, conf, factors, feats in rows
        ]
    )


def test_predictions_wide_gives_one_row_per_stay_with_a_column_per_prediction():
    prediction = _prediction(
        [
            (
                "s1",
                "physician",
                "at_doc_prediction",
                "2024-01-02 06:00",
                0.6,
                1.0,
                "Veel",
                "Bloeddruk",
                None,
            ),
            (
                "s1",
                "model",
                "at_doc_prediction",
                "2024-01-02 05:55",
                0.35,
                float("nan"),
                None,
                None,
                "{'age': .1}",
            ),
            (
                "s1",
                "model",
                "at_discharge",
                "2024-01-02 12:00",
                0.40,
                float("nan"),
                None,
                None,
                "{'age': .2}",
            ),
        ]
    )
    wide = cdm.predictions_wide(prediction)
    assert len(wide) == 1
    row = wide.iloc[0]
    assert row["doc_prediction"] == 0.6
    assert row["confidence"] == "Veel"
    assert row["doc_factors"] == "Bloeddruk"
    assert row["algo_prediction_at_doc_prediction"] == 0.35
    assert row["algo_prediction_at_discharge"] == 0.40
    assert row["features_at_discharge"] == "{'age': .2}"


def test_predictions_wide_keeps_a_stay_whose_physician_prediction_is_missing():
    # the enrolment funnel excludes on exactly this, and the excluded-vs-analysed comparison
    # needs the stay present to compare - it must not silently disappear here
    prediction = _prediction(
        [
            (
                "s1",
                "model",
                "at_doc_prediction",
                "2024-01-02 05:55",
                0.35,
                float("nan"),
                None,
                None,
                None,
            ),
        ]
    )
    wide = cdm.predictions_wide(prediction)
    assert len(wide) == 1
    assert pd.isna(wide["doc_predicted_at"].iloc[0])
    assert wide["algo_prediction_at_doc_prediction"].iloc[0] == 0.35


def test_predictions_wide_keeps_a_stay_whose_model_prediction_is_missing():
    prediction = _prediction(
        [
            (
                "s1",
                "physician",
                "at_doc_prediction",
                "2024-01-02 06:00",
                0.6,
                1.0,
                "Veel",
                "Bloeddruk",
                None,
            ),
        ]
    )
    wide = cdm.predictions_wide(prediction)
    assert len(wide) == 1
    assert pd.isna(wide["algo_predicted_at_doc_prediction"].iloc[0])
    assert wide["doc_prediction"].iloc[0] == 0.6


def test_predictions_wide_keeps_stays_separate():
    prediction = _prediction(
        [
            (
                "s1",
                "physician",
                "at_doc_prediction",
                "2024-01-02 06:00",
                0.6,
                1.0,
                "Veel",
                None,
                None,
            ),
            (
                "s2",
                "physician",
                "at_doc_prediction",
                "2024-01-03 06:00",
                0.2,
                0.0,
                "Weinig",
                None,
                None,
            ),
        ]
    )
    wide = cdm.predictions_wide(prediction).set_index("icu_stay_id")
    assert wide.loc["s1", "doc_prediction"] == 0.6
    assert wide.loc["s2", "doc_prediction"] == 0.2
