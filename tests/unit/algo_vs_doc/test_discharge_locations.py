import pandas as pd
import pytest

from algo_vs_doc import discharge_locations


def _stay(**overrides):
    row = {
        "icu_discharged_at": pd.Timestamp("2024-01-02"),
        "discharge_destination": "ward",
        "died_on_icu": False,
        "has_death_7days": False,
        "has_icu_readmission_7days": False,
    }
    row.update(overrides)
    return row


def test_validate_destinations_rejects_an_unknown_value():
    with pytest.raises(KeyError, match="discharge destination"):
        discharge_locations.validate_destinations(pd.Series(["somewhere_new"]))


def test_later_readmission_does_not_replace_the_physical_destination():
    frame = pd.DataFrame([_stay(has_icu_readmission_7days=True)])

    discharge_locations.validate_destinations(frame["discharge_destination"])
    assert (
        discharge_locations.derive_post_discharge_status(frame).iloc[0]
        == "icu_readmission"
    )


def test_monitored_stepdown_units_are_grouped_as_high_dependency_units():
    result = discharge_locations.derive_discharge_care_level(
        pd.Series(
            [
                "general_ward",
                "cardiac_care",
                "medium_care",
                "post_anesthesia_care",
                None,
            ]
        )
    )

    assert result.tolist()[:4] == [
        "general_ward",
        "high_dependency_unit",
        "high_dependency_unit",
        "high_dependency_unit",
    ]
    assert result.iloc[4] == "unknown"


def test_death_takes_precedence_only_in_the_enrolment_presentation_status():
    frame = pd.DataFrame([_stay(has_icu_readmission_7days=True, has_death_7days=True)])

    assert frame.loc[0, "discharge_destination"] == "ward"
    assert (
        discharge_locations.derive_post_discharge_status(frame).iloc[0]
        == "readmission_and_death"
    )


def test_map_source_locations_uses_a_reviewable_mapping_without_changing_source():
    source = pd.DataFrame(
        {
            "discharge_location_id": [10, None],
            "discharge_location_name": ["Cardiology ward", None],
        }
    )
    mapping = {
        10: {
            "name": "Cardiology ward",
            "destination": "ward",
            "specialty": "cardiology",
            "unit_type": "cardiac_care",
        }
    }

    result = discharge_locations.map_source_locations(
        source, hospital="LUMC", mapping=mapping
    )

    assert result.loc[0, "mapped_discharge_destination"] == "ward"
    assert result.loc[0, "discharge_specialty"] == "cardiology"
    assert result.loc[0, "discharge_unit_type"] == "cardiac_care"
    assert pd.isna(result.loc[1, "mapped_discharge_destination"])
    assert "discharge_specialty" not in source


def test_map_source_locations_reuses_the_mapping_for_readmission_origin():
    source = pd.DataFrame(
        {
            "readmission_origin_location_id": [10],
            "readmission_origin_location_name": ["Cardiology ward"],
        }
    )
    mapping = {
        10: {
            "name": "Cardiology ward",
            "destination": "ward",
            "specialty": "cardiology",
            "unit_type": "cardiac_care",
        }
    }

    result = discharge_locations.map_source_locations(
        source,
        hospital="LUMC",
        mapping=mapping,
        prefix="readmission_origin",
    )

    assert result.loc[0, "readmission_origin_specialty"] == "cardiology"
    assert result.loc[0, "readmission_origin_unit_type"] == "cardiac_care"
    assert result.loc[0, "mapped_readmission_origin_destination"] == "ward"


def test_map_source_locations_rejects_an_unknown_non_null_identifier():
    source = pd.DataFrame(
        {
            "discharge_location_id": [999],
            "discharge_location_name": ["Unknown ward"],
        }
    )

    with pytest.raises(KeyError, match="LUMC discharge location id"):
        discharge_locations.map_source_locations(
            source,
            hospital="LUMC",
            mapping={
                10: {
                    "name": "Cardiology ward",
                    "destination": "ward",
                    "specialty": "cardiology",
                }
            },
        )


def test_map_source_locations_rejects_a_name_that_disagrees_with_its_id():
    source = pd.DataFrame(
        {
            "discharge_location_id": [10],
            "discharge_location_name": ["Different ward"],
        }
    )

    with pytest.raises(ValueError, match="name does not match its registered id"):
        discharge_locations.map_source_locations(
            source,
            hospital="LUMC",
            mapping={
                10: {
                    "name": "Cardiology ward",
                    "destination": "ward",
                    "specialty": "cardiology",
                }
            },
        )


def test_map_source_locations_rejects_invalid_target_values():
    source = pd.DataFrame(
        {
            "discharge_location_id": [10],
            "discharge_location_name": ["Cardiology ward"],
        }
    )

    with pytest.raises(KeyError, match="mapped discharge destination"):
        discharge_locations.map_source_locations(
            source,
            hospital="LUMC",
            mapping={
                10: {
                    "name": "Cardiology ward",
                    "destination": "not_a_destination",
                    "specialty": "cardiology",
                }
            },
        )


def test_map_source_locations_rejects_invalid_unit_types():
    source = pd.DataFrame(
        {
            "discharge_location_id": [10],
            "discharge_location_name": ["Cardiology ward"],
        }
    )

    with pytest.raises(KeyError, match="discharge unit type"):
        discharge_locations.map_source_locations(
            source,
            hospital="LUMC",
            mapping={
                10: {
                    "name": "Cardiology ward",
                    "destination": "ward",
                    "specialty": "cardiology",
                    "unit_type": "not_a_unit_type",
                }
            },
        )


def test_lumc_mapping_covers_every_supplied_location_id():
    assert set(discharge_locations.SOURCE_DISCHARGE_LOCATIONS["LUMC"]) == set(
        range(1, 32)
    )
    assert (
        discharge_locations.SOURCE_DISCHARGE_LOCATIONS["LUMC"][5]["unit_type"]
        == "cardiac_care"
    )
    assert (
        discharge_locations.SOURCE_DISCHARGE_LOCATIONS["LUMC"][24]["unit_type"]
        == "medium_care"
    )
    assert (
        discharge_locations.SOURCE_DISCHARGE_LOCATIONS["LUMC"][29]["unit_type"]
        == "post_anesthesia_care"
    )


def test_vumc_included_and_excluded_department_ids_do_not_overlap():
    included = set(discharge_locations.SOURCE_DISCHARGE_LOCATIONS["VUmc"])
    assert included.isdisjoint(discharge_locations.VUMC_EXCLUDED_DEPARTMENT_IDS)
    assert (
        discharge_locations.SOURCE_DISCHARGE_LOCATIONS["VUmc"][201001057]["unit_type"]
        == "cardiac_care"
    )
    assert (
        discharge_locations.SOURCE_DISCHARGE_LOCATIONS["VUmc"][201001061]["unit_type"]
        == "medium_care"
    )
    assert (
        discharge_locations.SOURCE_DISCHARGE_LOCATIONS["VUmc"][201001061]["destination"]
        == "ward"
    )
