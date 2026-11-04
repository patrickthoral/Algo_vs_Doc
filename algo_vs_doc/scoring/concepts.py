"""Shared concept names for hospital-agnostic CDM-lite tables.

Hospital ETL maps vendor-specific identifiers onto this vocabulary. These readable names are a
small project vocabulary, not the OMOP standard vocabulary.
"""

from __future__ import annotations

#: Measurement concepts consumed by scoring functions.
MEASUREMENT_CONCEPTS = frozenset(
    {
        # sofa/cardiovascular.py
        "map",  # mean arterial pressure, mmHg
        # sofa/respiratory.py
        "art_po2",  # arterial PO2, kPa
        "fio2_fraction",  # FiO2 as a fraction (0-1), from a ventilator/precise-FiO2 device
        "o2_device",  # categorical: one of the O2_DEVICE_VALUES below
        "o2_flow_rate",  # L/min, paired with o2_device
        "peep",  # cmH2O; any recorded PEEP (invasive or non-invasive) means "ventilated" for SOFA
        # sofa/coagulation.py
        "platelets",  # x10^9/L
        # sofa/liver.py
        "bilirubin",  # umol/L
        # sofa/kidneys.py
        "creatinine",  # umol/L
        "urine_output",  # mL, one interval-summed reading (not yet extrapolated to 24h)
        # A valid GCS total is derived after sedation and ventilation handling.
        "gcs_eye",
        "gcs_motor",
        "gcs_verbal",
        # news/components.py
        "respiratory_rate",  # breaths/min
        "spo2",  # %
        "temperature",  # degrees Celsius
        "systolic_blood_pressure",  # mmHg
        "heart_rate",  # beats/min
    }
)

#: Oxygen-device values supported by device-and-flow FiO2 estimation.
O2_DEVICE_VALUES = frozenset(
    {"room_air", "low_flow_nasal_cannula", "o2_face_mask", "o2_face_mask_reservoir"}
)

#: Vasopressors and inotropes scored by SOFA. Sedatives use a separate generic concept.
VASOPRESSOR_DRUG_CONCEPTS = frozenset(
    {"dopamine", "epinephrine", "norepinephrine", "dobutamine"}
)

#: Enriched per-stay flag derived from vasopressor and inotrope drug exposure.
VASOPRESSOR_INOTROPE_FLAG = "vasopressors_inotropes"

#: Generic sedative or paralytic exposure used to gate GCS readings.
SEDATIVE_DRUG_CONCEPT = "sedative"

#: Device-based life-support categories.
DEVICE_CONCEPTS = frozenset(
    {
        "invasive_ventilation",
        "non_invasive_ventilation",
        "high_flow_oxygen_therapy",
        "renal_replacement_therapy",
        "cardiac_assist_devices",
        "ecls",
    }
)

#: Enriched life-support flags derived from device and drug exposure.
LIFE_SUPPORT_CONCEPTS = DEVICE_CONCEPTS | frozenset({VASOPRESSOR_INOTROPE_FLAG})

#: Prediction producers in the purpose-built ``prediction`` table.
PREDICTORS = ("physician", "model")

#: Shared ordinal prediction and confidence labels. Hospital ETL maps source codes to these names.
PREDICTION_LEVELS = ("low", "medium", "high")

#: Prediction timings used for direct physician-model comparison and discharge scoring.
PREDICTION_TIMINGS = ("at_doc_prediction", "at_discharge")

#: Conditions recorded as presence rows. Missing rows mean the condition is absent.
CONDITION_CONCEPTS = frozenset(
    {
        "cardiovascular_insufficiency",
        "chronic_dialysis",
        "chronic_renal_insufficiency",
        "cirrhosis",
        "copd",
        "diabetes_mellitus",
        "hematological_malignancy",
        "metastatic_neoplasm",
        "confirmed_infection",
        "cpr",
    }
)
