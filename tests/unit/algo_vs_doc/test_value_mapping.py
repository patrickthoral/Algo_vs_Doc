import pandas as pd
import pytest

from algo_vs_doc import discharge_locations, value_mapping


@pytest.mark.parametrize(
    ("hospital", "sex", "specialty_id", "expected_sex", "expected_specialty"),
    [
        ("LUMC", "female", 9, "female", "cardiology"),
        ("VUmc", "M", 31355, "male", "internal_medicine"),
    ],
)
def test_map_values_maps_flat_source_values(
    hospital, sex, specialty_id, expected_sex, expected_specialty
):
    source = pd.DataFrame({"sex": [sex], "specialty_id": [specialty_id]})

    result = value_mapping.map_values(source, hospital=hospital)

    assert result.loc[0, "sex"] == expected_sex
    assert result.loc[0, "specialty"] == expected_specialty
    assert "specialty_id" not in result
    assert "specialty" not in source


def test_map_values_requires_both_source_columns():
    with pytest.raises(KeyError, match="specialty_id"):
        value_mapping.map_values(pd.DataFrame({"sex": ["female"]}), hospital="LUMC")


def test_map_cdm_tables_maps_stays_before_enrichment_and_preserves_inputs():
    stays = pd.DataFrame({"sex": ["female"], "specialty_id": [9]})
    tables = {"icu_stays": stays, "measurement": pd.DataFrame()}

    result = value_mapping.map_cdm_tables(tables, hospital="LUMC")

    assert result["icu_stays"].loc[0, "specialty"] == "cardiology"
    assert result["icu_stays"].loc[0, "specialty_id"] == 9
    assert "specialty" not in stays
    assert result["measurement"] is tables["measurement"]


def test_map_cdm_tables_requires_icu_stays():
    with pytest.raises(KeyError, match="icu_stays"):
        value_mapping.map_cdm_tables({}, hospital="LUMC")


def test_map_values_applies_a_populated_discharge_location_mapping(monkeypatch):
    monkeypatch.setitem(
        discharge_locations.SOURCE_DISCHARGE_LOCATIONS,
        "LUMC",
        {
            5: {
                "name": "C09P - Hartbewaking",
                "destination": "ward",
                "specialty": "cardiology",
                "unit_type": "cardiac_care",
            }
        },
    )
    source = pd.DataFrame(
        {
            "sex": ["female"],
            "specialty_id": [9],
            "discharge_location_id": [5],
            "discharge_location_name": ["C09P - Hartbewaking"],
        }
    )

    result = value_mapping.map_values(source, hospital="LUMC")

    assert result.loc[0, "discharge_specialty"] == "cardiology"
    assert result.loc[0, "discharge_unit_type"] == "cardiac_care"


def test_map_values_maps_an_inferred_readmission_specialty_without_inventing_missing_values():
    source = pd.DataFrame(
        {
            "sex": ["female", "female"],
            "specialty_id": [9, 9],
            "readmission_origin_specialty_id": [4, None],
        }
    )

    result = value_mapping.map_values(source, hospital="LUMC")

    assert result.loc[0, "readmission_origin_specialty"] == "neurology"
    assert pd.isna(result.loc[1, "readmission_origin_specialty"])


def test_direct_origin_location_takes_precedence_over_inferred_specialty(monkeypatch):
    monkeypatch.setitem(
        discharge_locations.SOURCE_DISCHARGE_LOCATIONS,
        "LUMC",
        {
            5: {
                "name": "C09P - Hartbewaking",
                "destination": "ward",
                "specialty": "cardiology",
                "unit_type": "cardiac_care",
            }
        },
    )
    source = pd.DataFrame(
        {
            "sex": ["female"],
            "specialty_id": [9],
            "discharge_location_id": [5],
            "discharge_location_name": ["C09P - Hartbewaking"],
            "readmission_origin_location_id": [5],
            "readmission_origin_location_name": ["C09P - Hartbewaking"],
            "readmission_origin_specialty_id": [4],
        }
    )

    result = value_mapping.map_values(source, hospital="LUMC")

    assert result.loc[0, "readmission_origin_specialty"] == "cardiology"
    assert result.loc[0, "readmission_origin_unit_type"] == "cardiac_care"
