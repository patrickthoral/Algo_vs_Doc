"""Map discharge locations and derive non-overlapping discharge and outcome statuses."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from algo_vs_doc import specialties, vocabulary

DISCHARGE_DESTINATIONS = (
    "ward",
    "home",
    "external_transfer",
    "died_on_icu",
)

DISCHARGE_UNIT_TYPES = (
    "general_ward",
    "cardiac_care",
    "medium_care",
    "post_anesthesia_care",
)

POST_DISCHARGE_STATUSES = (
    "no_event",
    "icu_readmission",
    "death",
    "readmission_and_death",
)

DISCHARGE_CARE_LEVELS = (
    "general_ward",
    "high_dependency_unit",
    "unknown",
)

# Source identifiers and labels are non-proprietary. Each record may contain ``name``,
# ``destination``, ``specialty``, and ``unit_type``.
SOURCE_DISCHARGE_LOCATIONS: dict[str, dict[object, dict[str, str | None]]] = {
    "LUMC": {
        1: {
            "name": "C06P - Hematologie (LOKH)",
            "destination": "ward",
            "specialty": "other_medical",
            "unit_type": "general_ward",
        },
        2: {
            "name": "C06Q - IHOB-boxen/zaal",
            "destination": "ward",
            "specialty": "unspecified",
            "unit_type": "general_ward",
        },
        3: {
            "name": "C08P - Oncologie short stay (LOKE)",
            "destination": "ward",
            "specialty": "other_medical",
            "unit_type": "general_ward",
        },
        4: {
            "name": "C08Q - Transplantatie (TRAX)",
            "destination": "ward",
            "specialty": "other_surgery",
            "unit_type": "general_ward",
        },
        5: {
            "name": "C09P - Hartbewaking",
            "destination": "ward",
            "specialty": "cardiology",
            "unit_type": "cardiac_care",
        },
        6: {
            "name": "C09P - Hartziekten",
            "destination": "ward",
            "specialty": "cardiology",
            "unit_type": "general_ward",
        },
        7: {
            "name": "C10P - Interne Geneeskunde (VIG1)",
            "destination": "ward",
            "specialty": "internal_medicine",
            "unit_type": "general_ward",
        },
        8: {
            "name": "C10Q - Acute Opname Afdeling",
            "destination": "ward",
            "specialty": "internal_medicine",
            "unit_type": "general_ward",
        },
        9: {
            "name": None,
            "destination": "ward",
            "specialty": "pulmonary_medicine",
            "unit_type": "general_ward",
        },
        10: {
            "name": "C11Q - Oncologie/ Hematologie long stay (LOKL)",
            "destination": "ward",
            "specialty": "other_medical",
            "unit_type": "general_ward",
        },
        11: {
            "name": "J06P - WAKZ afdeling Strand",
            "destination": "ward",
            "specialty": "other_medical",
            "unit_type": "general_ward",
        },
        12: {
            "name": "J06Q - WAKZ afdeling Bos",
            "destination": "ward",
            "specialty": "other_medical",
            "unit_type": "general_ward",
        },
        13: {
            "name": "J06R - Kinderheelkunde",
            "destination": "ward",
            "specialty": "other_surgery",
            "unit_type": "general_ward",
        },
        14: {
            "name": "J07P - Neonatologie",
            "destination": "ward",
            "specialty": "other_medical",
            "unit_type": "general_ward",
        },
        15: {
            "name": "J07Q - Verloskunde",
            "destination": "ward",
            "specialty": "other_medical",
            "unit_type": "general_ward",
        },
        16: {
            "name": "J08P - Dialyse",
            "destination": "ward",
            "specialty": "internal_medicine",
            "unit_type": "general_ward",
        },
        17: {
            "name": "J08Q - Gynaecologie /KNO /Kaak/Oog/Huid (KOHG)",
            "destination": "ward",
            "specialty": "unspecified",
            "unit_type": "general_ward",
        },
        18: {
            "name": "J08R - Dagbehandeling",
            "destination": "ward",
            "specialty": "unspecified",
            "unit_type": None,
        },
        19: {
            "name": "J09P - Thoraxchirurgie",
            "destination": "ward",
            "specialty": "cardiothoracic_surgery",
            "unit_type": "general_ward",
        },
        20: {
            "name": "J09Q - Vaat/Ortho/Trauma/Uro/Plastische (VCH2)",
            "destination": "ward",
            "specialty": "other_surgery",
            "unit_type": "general_ward",
        },
        21: {
            "name": "J10P -",
            "destination": "ward",
            "specialty": "unspecified",
            "unit_type": "general_ward",
        },
        22: {
            "name": "J10Q - G.E./ Algemene + Onco chirurgie (VCH1)",
            "destination": "ward",
            "specialty": "general_surgery",
            "unit_type": "general_ward",
        },
        23: {
            "name": "J11P - Psychiatrie (VPL3)",
            "destination": "ward",
            "specialty": "other_medical",
            "unit_type": "general_ward",
        },
        24: {
            "name": "J11Q - Neuro MC (VPL2)",
            "destination": "ward",
            "specialty": "neurology",
            "unit_type": "medium_care",
        },
        25: {
            "name": "J11Q - Neurologie/ -chirurgie (VPL1)",
            "destination": "ward",
            "specialty": "neurology",
            "unit_type": "general_ward",
        },
        26: {
            "name": "Naar andere instelling",
            "destination": "external_transfer",
            "specialty": None,
            "unit_type": None,
        },
        27: {
            "name": "Naar huis",
            "destination": "home",
            "specialty": None,
            "unit_type": None,
        },
        28: {
            "name": "Overleden",
            "destination": "died_on_icu",
            "specialty": None,
            "unit_type": None,
        },
        29: {
            "name": "PACU",
            "destination": "ward",
            "specialty": "other_surgery",
            "unit_type": "post_anesthesia_care",
        },
        30: {
            "name": "Naar ander ziekenhuis (IC/CCU)",
            "destination": "external_transfer",
            "specialty": None,
            "unit_type": None,
        },
        31: {
            "name": "Naar ander ziekenhuis (Verpleegafdeling)",
            "destination": "external_transfer",
            "specialty": None,
            "unit_type": None,
        },
    },
    "VUmc": {
        201001010: {
            "name": "VUMC SEH",
            "destination": "ward",
            "specialty": "unspecified",
            "unit_type": None,
        },
        201001019: {
            "name": "VUMC 1B KORT VERBLIJF",
            "destination": "ward",
            "specialty": "unspecified",
            "unit_type": "general_ward",
        },
        201001020: {
            "name": "VUMC 1C KNO",
            "destination": "ward",
            "specialty": "otolaryngology",
            "unit_type": "general_ward",
        },
        201001021: {
            "name": "VUMC 2B NEUROCENTRUM",
            "destination": "ward",
            "specialty": "neurology",
            "unit_type": "general_ward",
        },
        201001022: {
            "name": "VUMC 2C NEUROCENTRUM",
            "destination": "ward",
            "specialty": "neurology",
            "unit_type": "general_ward",
        },
        201001039: {
            "name": "VUMC 8B GYN",
            "destination": "ward",
            "specialty": "other_medical",
            "unit_type": "general_ward",
        },
        201001040: {
            "name": "VUMC 8C OBS",
            "destination": "ward",
            "specialty": "other_medical",
            "unit_type": "general_ward",
        },
        201001055: {
            "name": "VUMC ACUTE OPNAME",
            "destination": "ward",
            "specialty": "internal_medicine",
            "unit_type": "general_ward",
        },
        201001056: {
            "name": "VUMC CARDIOLOGIE",
            "destination": "ward",
            "specialty": "cardiology",
            "unit_type": "general_ward",
        },
        201001057: {
            "name": "VUMC 5D HARTBEWAKING EN EHH",
            "destination": "ward",
            "specialty": "cardiology",
            "unit_type": "cardiac_care",
        },
        201001059: {
            "name": "VUMC 6B HON/MDL/HGE",
            "destination": "ward",
            "specialty": "unspecified",
            "unit_type": "general_ward",
        },
        201001060: {
            "name": "VUMC HLK/TRA/PCH",
            "destination": "ward",
            "specialty": "other_surgery",
            "unit_type": "general_ward",
        },
        201001061: {
            "name": "VUMC 7D MEDIUM CARE",
            "destination": "ward",
            "specialty": "unspecified",
            "unit_type": "medium_care",
        },
        201001063: {
            "name": "VUMC 7B/7C IC",
            "destination": "ward",
            "specialty": "unspecified",
            "unit_type": None,
        },
        201001119: {
            "name": "VUMC HEMATOLOGIE",
            "destination": "ward",
            "specialty": "other_medical",
            "unit_type": "general_ward",
        },
        201001120: {
            "name": "VUMC 3C ONI",
            "destination": "ward",
            "specialty": "other_medical",
            "unit_type": "general_ward",
        },
        201001121: {
            "name": "VUMC UROLOGIE",
            "destination": "ward",
            "specialty": "other_surgery",
            "unit_type": "general_ward",
        },
        201001123: {
            "name": "VUMC 5C LON/HLO",
            "destination": "ward",
            "specialty": "pulmonary_medicine",
            "unit_type": "general_ward",
        },
        201001135: {
            "name": "VUMC INTERNE GENEESKUNDE",
            "destination": "ward",
            "specialty": "internal_medicine",
            "unit_type": "general_ward",
        },
        201001234: {
            "name": "VUMC 4C MEDISCH PSYCHIATRISCHE UNIT",
            "destination": "ward",
            "specialty": "other_medical",
            "unit_type": "general_ward",
        },
        201001301: {
            "name": "VUMC CHIRURGISCHE ONCOLOGIE EN MDL",
            "destination": "ward",
            "specialty": "general_surgery",
            "unit_type": "general_ward",
        },
    },
}

VUMC_EXCLUDED_DEPARTMENT_IDS = frozenset(
    {
        201001015,
        201001041,
        201001042,
        201001043,
        201001044,
        201001054,
        201001058,
        201001097,
        201001101,
        201001148,
        201001172,
        201001201,
        201001211,
        201001212,
        201001214,
        201001265,
        201001266,
        201001289,
    }
)


def validate_destinations(values: pd.Series) -> None:
    """Raise when immediate discharge destinations fall outside the shared vocabulary."""
    vocabulary.raise_unmapped(
        vocabulary.unmapped_values(values, DISCHARGE_DESTINATIONS),
        what="discharge destination(s)",
        known=DISCHARGE_DESTINATIONS,
    )


def derive_post_discharge_status(frame: pd.DataFrame) -> pd.Series:
    """Classify seven-day outcomes independently from the physical destination."""
    required = {
        "has_death_7days",
        "has_icu_readmission_7days",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(
            f"post-discharge status requires columns {sorted(required)}; missing {missing}"
        )

    death = frame["has_death_7days"].fillna(False)
    icu = frame["has_icu_readmission_7days"].fillna(False)
    readmission = icu
    labels = np.select(
        [death & readmission, death, icu],
        [
            "readmission_and_death",
            "death",
            "icu_readmission",
        ],
        default="no_event",
    )
    return pd.Series(
        labels, index=frame.index, name="post_discharge_status", dtype="object"
    )


def derive_discharge_care_level(unit_type: pd.Series) -> pd.Series:
    """Group receiving units as general ward, high-dependency unit, or unknown."""
    vocabulary.raise_unmapped(
        vocabulary.unmapped_values(unit_type, DISCHARGE_UNIT_TYPES),
        what="discharge unit type(s)",
        known=DISCHARGE_UNIT_TYPES,
    )
    result = unit_type.replace(
        {
            "cardiac_care": "high_dependency_unit",
            "medium_care": "high_dependency_unit",
            "post_anesthesia_care": "high_dependency_unit",
        }
    ).fillna("unknown")
    return result.rename("discharge_care_level")


def map_source_locations(
    frame: pd.DataFrame,
    *,
    hospital: str,
    mapping: Mapping[object, Mapping[str, str | None]] | None = None,
    prefix: str = "discharge",
) -> pd.DataFrame:
    """Map prefixed location identifiers to shared location concepts.

    This function is ready for the reviewed mapping lists but is not called by the enrichment
    pipeline until both hospital mappings are populated.
    """
    id_column = f"{prefix}_location_id"
    name_column = f"{prefix}_location_name"
    required = {id_column, name_column}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(
            f"{prefix} location mapping requires {sorted(required)}; missing {missing}"
        )

    if mapping is None:
        try:
            mapping = SOURCE_DISCHARGE_LOCATIONS[hospital]
        except KeyError:
            raise KeyError(
                f"no discharge location mapping for hospital {hospital!r}; "
                f"known: {sorted(SOURCE_DISCHARGE_LOCATIONS)}"
            ) from None

    identifiers = frame[id_column]
    vocabulary.raise_unmapped(
        vocabulary.unmapped_values(identifiers, mapping),
        what=f"{hospital} {prefix} location id(s)",
        known=mapping,
    )

    name_mapping = {
        source_id: record.get("name") for source_id, record in mapping.items()
    }
    expected_names = identifiers.map(name_mapping)
    supplied_names = frame[name_column]
    mismatched_names = (
        identifiers.notna()
        & expected_names.notna()
        & supplied_names.notna()
        & expected_names.ne(supplied_names)
    )
    if mismatched_names.any():
        mismatches = (
            frame.loc[
                mismatched_names,
                [id_column, name_column],
            ]
            .drop_duplicates()
            .to_dict("records")
        )
        raise ValueError(
            f"{hospital} {prefix} location name does not match its registered id: "
            f"{mismatches}"
        )

    destination_mapping = {
        source_id: record.get("destination") for source_id, record in mapping.items()
    }
    specialty_mapping = {
        source_id: record.get("specialty") for source_id, record in mapping.items()
    }
    unit_type_mapping = {
        source_id: record.get("unit_type") for source_id, record in mapping.items()
    }
    result = frame.copy()
    destination_column = f"mapped_{prefix}_destination"
    specialty_column = f"{prefix}_specialty"
    unit_type_column = f"{prefix}_unit_type"
    result[destination_column] = identifiers.map(destination_mapping)
    result[specialty_column] = identifiers.map(specialty_mapping)
    result[unit_type_column] = identifiers.map(unit_type_mapping)

    vocabulary.raise_unmapped(
        vocabulary.unmapped_values(result[destination_column], DISCHARGE_DESTINATIONS),
        what=f"mapped {prefix} destination(s)",
        known=DISCHARGE_DESTINATIONS,
    )
    specialties.validate(result[specialty_column])
    vocabulary.raise_unmapped(
        vocabulary.unmapped_values(result[unit_type_column], DISCHARGE_UNIT_TYPES),
        what=f"{prefix} unit type(s)",
        known=DISCHARGE_UNIT_TYPES,
    )
    return result
