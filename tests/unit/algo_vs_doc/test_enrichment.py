import os

import pandas as pd
import pytest

from algo_vs_doc import enrichment


def _icu_stays(n=2):
    return pd.DataFrame(
        [
            {
                "icu_stay_id": f"s{i}",
                "patient_id": 100 + i,
                "is_study_cohort": True,
                "icu_admitted_at": pd.Timestamp("2024-01-01") + pd.Timedelta(days=i),
                "icu_discharged_at": pd.Timestamp("2024-01-03") + pd.Timedelta(days=i),
                "icu_los": 48.0,
                "icu_los_lt_12h": False,
                "sex": "female",
                "age_at_admission": 70,
                "specialty_id": 9,
                "patient_category_id": 3 if i == 0 else None,
                "patient_category_name": "Category 3" if i == 0 else None,
                "discharge_location_id": 100,
                "discharge_location_name": "Synthetic ward",
                "ready_for_discharge": True if i == 0 else pd.NA,
                "readmission_origin_location_id": None,
                "readmission_origin_location_name": None,
                "readmission_origin_specialty_id": None,
                "readmission_origin_specialty_name": None,
                "readmission_origin_inferred": pd.NA,
                "is_medical": True,
                "emergency_surgery": False,
                "elective_surgery": False,
                "do_not_readmit": False,
                "palliative_care": False,
                "discharge_destination": "ward",
                "died_at": pd.NaT,
                "died_on_icu": False,
                "time_to_death": float("nan"),
                "has_death_7days": False,
                "has_icu_readmission": False,
                "icu_readmission_time": pd.NaT,
                "time_to_icu_readmission": float("nan"),
                "has_icu_readmission_7days": False,
                "outcome": False,
            }
            for i in range(n)
        ]
    )


def _tables(**overrides):
    tables = {
        "icu_stays": _icu_stays(),
        "measurement": pd.DataFrame(
            [
                {
                    "icu_stay_id": "s0",
                    "measurement_concept_name": "platelets",
                    "measured_at": pd.Timestamp("2024-01-01 08:00"),
                    "value_as_number": 40.0,
                    "value_as_string": None,
                },
            ]
        ),
        "drug_exposure": pd.DataFrame(
            columns=[
                "icu_stay_id",
                "drug_concept_name",
                "drug_exposure_start_datetime",
                "drug_exposure_end_datetime",
                "dose_value",
            ]
        ),
        "device_exposure": pd.DataFrame(
            [
                {
                    "icu_stay_id": "s0",
                    "device_concept_name": "invasive_ventilation",
                    "device_exposure_start_datetime": pd.Timestamp("2024-01-01"),
                    "device_exposure_end_datetime": pd.Timestamp("2024-01-02"),
                },
            ]
        ),
        "condition_occurrence": pd.DataFrame(
            [
                {
                    "icu_stay_id": "s0",
                    "condition_concept_name": "copd",
                    "condition_start_datetime": pd.NaT,
                },
            ]
        ),
        "diagnosis_occurrence": pd.DataFrame(
            [
                {
                    "patient_id": 100,
                    "diagnosis_source_code": "synthetic-1",
                    "diagnosis_source_label": "Synthetic diagnosis",
                    "diagnosed_at": pd.Timestamp("2024-01-02"),
                },
            ]
        ),
        "prediction": pd.DataFrame(
            [
                {
                    "icu_stay_id": "s0",
                    "predictor": "physician",
                    "prediction_timing": "at_doc_prediction",
                    "predicted_at": pd.Timestamp("2024-01-03 06:00"),
                    "prediction_value": 0.6,
                    "prediction_category": "high",
                    "confidence": "much",
                    "factors": "doc_factors:5",
                    "other_factors": None,
                    "features": None,
                },
            ]
        ),
    }
    tables.update(overrides)
    return tables


def test_enrich_rejects_hdu_or_other_events_in_primary_outcome():
    tables = _tables()
    tables["icu_stays"].loc[0, "outcome"] = True

    with pytest.raises(ValueError, match="HDU readmission is not part"):
        enrichment.enrich(tables, hospital="VUmc")


def test_enrich_produces_one_row_per_stay_with_every_derived_column():
    enriched = enrichment.enrich(_tables(), hospital="LUMC")
    assert len(enriched) == 2
    assert enriched.loc[0, "patient_category_id"] == 3
    assert enriched.loc[0, "patient_category_name"] == "Category 3"
    for column in (
        "sofa_admission",
        "sofa_max",
        "sofa_discharge",
        "news_discharge",
        "copd",
        "index_episode_diagnoses_count",
        "cirrhosis",
        "invasive_ventilation",
        "ventilation",
        "doc_prediction",
        "doc_factors",
        "hospital",
    ):
        assert column in enriched.columns, column


def test_enrich_preserves_identifiers_assigned_at_the_private_cdm_boundary():
    stays = _icu_stays()
    enriched = enrichment.enrich(_tables(icu_stays=stays), hospital="LUMC")

    assert enriched["patient_id"].tolist() == stays["patient_id"].tolist()
    assert enriched["icu_stay_id"].tolist() == stays["icu_stay_id"].tolist()


def test_enrich_keeps_discharge_location_source_values_and_separates_later_outcomes():
    stays = _icu_stays()
    readmission = stays.iloc[[1]].copy()
    readmission["icu_stay_id"] = "s2"
    readmission["patient_id"] = stays.loc[0, "patient_id"]
    readmission["icu_admitted_at"] = pd.Timestamp("2024-01-05")
    readmission["icu_discharged_at"] = pd.Timestamp("2024-01-06")
    readmission["is_study_cohort"] = False
    # Context rows are not analysis rows and need not carry their own complete outcome fields.
    readmission["outcome"] = True
    stays = pd.concat([stays, readmission], ignore_index=True)
    stays.loc[0, "has_icu_readmission"] = True
    stays.loc[0, "icu_readmission_time"] = pd.Timestamp("2024-01-05")
    stays.loc[0, "time_to_icu_readmission"] = 48.0
    stays.loc[0, "has_icu_readmission_7days"] = True
    stays.loc[0, "outcome"] = True
    enriched = enrichment.enrich(_tables(icu_stays=stays), hospital="LUMC")

    assert enriched.loc[0, "discharge_location_id"] == 100
    assert enriched.loc[0, "discharge_location_name"] == "Synthetic ward"
    assert enriched.loc[0, "discharge_destination"] == "ward"
    assert enriched.loc[0, "post_discharge_status"] == "icu_readmission"
    assert enriched["icu_stay_id"].tolist() == ["s0", "s1", "s2"]
    assert bool(enriched.loc[2, "is_study_cohort"]) is False
    assert bool(enriched.loc[2, "outcome"]) is False


def test_enrich_groups_cardiac_care_as_a_high_dependency_unit():
    tables = _tables()
    tables["icu_stays"]["discharge_unit_type"] = "cardiac_care"

    enriched = enrichment.enrich(tables, hospital="LUMC")

    assert enriched.loc[0, "discharge_care_level"] == "high_dependency_unit"


def test_elective_surgery_is_an_icu_stay_flag_not_a_cdm_table():
    assert "elective_surgery" in enrichment.STAY_COLUMNS
    assert "elective_surgery" not in enrichment.CDM_TABLES
    assert "diagnosis_occurrence" in enrichment.CDM_TABLES


def test_outcome_documentation_is_not_part_of_the_public_stay_contract():
    assert "outcome_documented" not in enrichment.STAY_COLUMNS


def test_conflated_destination_is_not_part_of_the_public_stay_contract():
    assert "destination" not in enrichment.STAY_COLUMNS
    assert "discharge_location_id" in enrichment.STAY_COLUMNS
    assert "discharge_location_name" in enrichment.STAY_COLUMNS


def test_assessment_and_readmission_provenance_are_part_of_the_stay_contract():
    assert "ready_for_discharge" in enrichment.STAY_COLUMNS
    assert "readmission_origin_location_id" in enrichment.STAY_COLUMNS
    assert "readmission_origin_specialty_id" in enrichment.STAY_COLUMNS
    assert "readmission_origin_inferred" in enrichment.STAY_COLUMNS


def test_unknown_readiness_and_origin_provenance_remain_unknown():
    enriched = enrichment.enrich(_tables(), hospital="LUMC")

    assert bool(enriched.loc[0, "ready_for_discharge"]) is True
    assert pd.isna(enriched.loc[1, "ready_for_discharge"])
    assert pd.isna(enriched.loc[0, "readmission_origin_inferred"])
    assert str(enriched["ready_for_discharge"].dtype) == "boolean"


def test_stay_duration_columns_must_contain_numeric_hours():
    stays = _icu_stays()
    stays["icu_los"] = pd.to_timedelta(stays["icu_los"], unit="h")

    with pytest.raises(TypeError, match="numeric hours.*icu_los"):
        enrichment.check_stay_columns(stays)


def test_stay_duration_hours_remain_numeric_after_csv_round_trip(tmp_path):
    stays = _icu_stays()
    expected = {
        "icu_los": 36.0,
        "time_to_death": 48.0,
        "time_to_icu_readmission": 3.5,
    }
    for column, value in expected.items():
        stays.loc[0, column] = value
    enriched = enrichment.enrich(_tables(icu_stays=stays), hospital="LUMC")
    path = tmp_path / "enriched.csv"

    enriched.to_csv(path, index=False)
    loaded = pd.read_csv(path)

    for column, value in expected.items():
        assert loaded.loc[0, column] == value
        assert pd.api.types.is_float_dtype(loaded[column].dtype)


def test_a_stay_with_no_prediction_survives_enrichment():
    # Excluded-versus-analysed comparisons require stays without predictions.
    enriched = enrichment.enrich(_tables(), hospital="LUMC").set_index("icu_stay_id")
    assert pd.isna(enriched.loc["s1", "doc_predicted_at"])
    assert enriched.loc["s0", "doc_prediction"] == 0.6


def test_a_stay_with_nothing_measured_still_gets_a_score_rather_than_disappearing():
    enriched = enrichment.enrich(_tables(), hospital="LUMC").set_index("icu_stay_id")
    assert enriched.loc["s1", "sofa_max"] == 0
    assert enriched.loc["s0", "sofa_max"] > 0  # platelets 40 scores on coagulation


def test_absent_conditions_and_support_are_false_not_missing():
    # "not recorded" means the patient did not have it; a null here would be read as unknown
    # and quietly excluded from every count
    enriched = enrichment.enrich(_tables(), hospital="LUMC").set_index("icu_stay_id")
    assert bool(enriched.loc["s0", "copd"]) is True
    assert bool(enriched.loc["s1", "copd"]) is False
    assert bool(enriched.loc["s1", "invasive_ventilation"]) is False
    assert bool(enriched.loc["s0", "ventilation"]) is True  # roll-up over the two modes


def test_vasopressor_support_is_derived_from_drug_exposure():
    drug_exposure = pd.DataFrame(
        [
            {
                "icu_stay_id": "s0",
                "drug_concept_name": "norepinephrine",
                "drug_exposure_start_datetime": pd.Timestamp("2024-01-01 08:00"),
                "drug_exposure_end_datetime": pd.Timestamp("2024-01-01 10:00"),
                "dose_value": 0.1,
            }
        ]
    )
    tables = _tables(drug_exposure=drug_exposure)
    tables["device_exposure"] = pd.concat(
        [
            tables["device_exposure"],
            pd.DataFrame(
                [{"icu_stay_id": "s1", "device_concept_name": "vasopressors_inotropes"}]
            ),
        ],
        ignore_index=True,
    )
    enriched = enrichment.enrich(tables, hospital="LUMC").set_index("icu_stay_id")
    assert bool(enriched.loc["s0", "vasopressors_inotropes"]) is True
    assert bool(enriched.loc["s1", "vasopressors_inotropes"]) is False


def test_a_missing_stay_flag_reads_as_false_but_a_missing_timestamp_stays_missing():
    # an ETL that never observed the flag hands it over as a nullable column, not a bool one
    stays = _icu_stays().astype({"outcome": object})
    stays.loc[0, "outcome"] = None
    stays.loc[0, "died_at"] = None
    enriched = enrichment.enrich(_tables(icu_stays=stays), hospital="LUMC")
    assert bool(enriched.loc[0, "outcome"]) is False
    assert pd.isna(enriched.loc[0, "died_at"])  # not turned into a date


def test_a_hospital_that_has_not_supplied_a_required_column_fails_naming_it():
    stays = _icu_stays().drop(columns=["palliative_care"])
    with pytest.raises(KeyError, match="palliative_care"):
        enrichment.enrich(_tables(icu_stays=stays), hospital="LUMC")


def test_enrich_does_not_modify_the_tables_it_was_given():
    tables = _tables()
    before = tables["icu_stays"].copy()
    enrichment.enrich(tables, hospital="LUMC")
    pd.testing.assert_frame_equal(tables["icu_stays"], before)


def test_load_cdm_says_which_notebook_to_run_when_the_etl_has_not_been_run(tmp_path):
    with pytest.raises(FileNotFoundError, match="Transform to Common Data Model VUmc"):
        enrichment.load_cdm(str(tmp_path), "VUmc")


def test_a_dataset_labelled_for_the_wrong_hospital_is_refused(tmp_path):
    # both files have the same shape, so a mix-up would otherwise attribute one site's stays
    # to the other and go unnoticed
    from algo_vs_doc import cohorts

    frame = pd.DataFrame([{"hospital": "VUmc", "patient_id": 1}])
    path = cohorts.dataset_path(str(tmp_path), "LUMC")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="wrong file is at that path"):
        cohorts.load(str(tmp_path), "LUMC")


def test_a_dataset_that_already_labels_itself_is_loaded_unchanged(tmp_path):
    from algo_vs_doc import cohorts

    frame = pd.DataFrame([{"hospital": "LUMC", "patient_id": 1}])
    path = cohorts.dataset_path(str(tmp_path), "LUMC")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    frame.to_csv(path, index=False)
    assert cohorts.load(str(tmp_path), "LUMC")["hospital"].tolist() == ["LUMC"]


def test_elective_surgery_actually_changes_the_admission_score():
    # the reason it is required rather than optional
    tables = _tables()
    tables["measurement"] = pd.concat(
        [
            tables["measurement"],
            pd.DataFrame(
                [
                    {
                        "icu_stay_id": "s0",
                        "measurement_concept_name": concept,
                        "measured_at": pd.Timestamp("2024-01-01 08:00"),
                        "value_as_number": 1.0,
                        "value_as_string": None,
                    }
                    for concept in (
                        "gcs_eye",
                        "gcs_motor",
                        "gcs_verbal",
                    )  # deeply unresponsive
                ]
            ),
        ],
        ignore_index=True,
    )

    without = enrichment.enrich(tables, hospital="LUMC").set_index("icu_stay_id")
    elective_stays = tables["icu_stays"].copy()
    elective_stays.loc[elective_stays["icu_stay_id"] == "s0", "elective_surgery"] = True
    with_elective = enrichment.enrich(
        tables | {"icu_stays": elective_stays}, hospital="LUMC"
    ).set_index("icu_stay_id")

    assert (
        with_elective.loc["s0", "sofa_admission"] < without.loc["s0", "sofa_admission"]
    )
