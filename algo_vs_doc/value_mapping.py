"""Map reviewable source values to shared analysis categories."""

from __future__ import annotations

import pandas as pd

from algo_vs_doc import demographics, discharge_locations, specialties


def map_values(
    frame: pd.DataFrame, *, hospital: str, keep_specialty_id: bool = False
) -> pd.DataFrame:
    """Return a copy with source sex and specialty values mapped to canonical values."""
    required = {"sex", "specialty_id"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(
            f"value mapping requires columns {sorted(required)}; missing {missing}"
        )

    result = frame.copy()
    result["sex"] = demographics.categorize_sex(result["sex"], hospital=hospital)
    result["specialty"] = specialties.categorize(
        result["specialty_id"], hospital=hospital
    )
    if not keep_specialty_id:
        result = result.drop(columns="specialty_id")
    if "readmission_origin_specialty_id" in result:
        result["readmission_origin_specialty"] = specialties.categorize_optional(
            result["readmission_origin_specialty_id"], hospital=hospital
        )
    if discharge_locations.SOURCE_DISCHARGE_LOCATIONS[hospital]:
        discharge_columns = {"discharge_location_id", "discharge_location_name"}
        if discharge_columns & set(result.columns):
            result = discharge_locations.map_source_locations(result, hospital=hospital)
        origin_columns = {
            "readmission_origin_location_id",
            "readmission_origin_location_name",
        }
        if origin_columns & set(result.columns):
            mapped_origin = discharge_locations.map_source_locations(
                result,
                hospital=hospital,
                prefix="readmission_origin",
            )
            mapped_specialty = mapped_origin["readmission_origin_specialty"]
            if "readmission_origin_specialty" in result:
                mapped_specialty = mapped_specialty.combine_first(
                    result["readmission_origin_specialty"]
                )
            result["readmission_origin_specialty"] = mapped_specialty
            result["readmission_origin_unit_type"] = mapped_origin[
                "readmission_origin_unit_type"
            ]
    return result


def map_cdm_tables(
    tables: dict[str, pd.DataFrame], *, hospital: str
) -> dict[str, pd.DataFrame]:
    """Map reviewable ICU-stay values before shared enrichment begins."""
    if "icu_stays" not in tables:
        raise KeyError("CDM value mapping requires the 'icu_stays' table")
    mapped = tables.copy()
    mapped["icu_stays"] = map_values(
        tables["icu_stays"], hospital=hospital, keep_specialty_id=True
    )
    return mapped
