"""Map physician factors and model features to one comparison vocabulary.

Hospital-specific factor lists remain separate. Liver-related model features map to the shared
``Abdominal`` category used by physician factors.
"""

from __future__ import annotations

from typing import Mapping

import pandas as pd

from algo_vs_doc import vocabulary

#: Canonical categories in reporting order.
CATEGORIES = (
    "General",
    "Respiratory",
    "Circulatory",
    "Neurological",
    "Renal and homeostasis",
    "Hematology",
    "Abdominal",
    "Infection",
    "Medication",
    "Logistic",
)

#: categories only a physician can cite - the model has no features of this kind, so they can
#: never overlap and are excluded when measuring coverage (see ``comparable_categories``).
PHYSICIAN_ONLY_CATEGORIES = ("Logistic", "Medication")

#: Physician-documented risk factors, keyed by the **option id** the form recorded - never by
#: its description. Two reasons, both learned the hard way:
#:
#: - LUMC's descriptions contain commas ("Demografisch (leeftijd, geslacht, etniciteit)"), so
#:   any comma-delimited handling of the text splits one factor into several and inflates the
#:   count. An id cannot be split.
#: - Descriptions also suffer encoding damage in transit ("Hematologie pati\u00ebnt" arriving as
#:   "Hematologie pati nt"), which an id is immune to.
#:
#: Keyed by hospital -> source field -> option id. **The field is part of the key, not
#: decoration:** LUMC split one question across two form sections with their own parameter
#: ids, and the option ids restart in each - id 10 is "Logistiek: zorgzwaarte voor afdeling"
#: in the first section and "Circulatie: ..." in the second. Reading an id without knowing
#: its field is how that distinction was lost once before.
PHYSICIAN_FACTOR_CATEGORIES: dict[str, dict[str, dict[int, str]]] = {
    "LUMC": {
        # MetaVision parameter 13287
        "doc_factors": {
            1: "General",  # Demografisch (leeftijd, geslacht, etniciteit)
            2: "General",  # Opname kenmerk: reden van opname
            4: "Circulatory",  # Comorbiditeit: Cardiaal
            5: "Respiratory",  # Comorbiditeit: Pulmonaal
            6: "General",  # Comorbiditeit: Maligniteit
            7: "General",  # Comorbiditeit: Kwetsbaarheid
            10: "Logistic",  # Logistiek: zorgzwaarte voor afdeling
            11: "General",  # Opname kenmerk: duur van opname of heropname
            12: "General",  # Comorbiditeit: Overig
            13: "Logistic",  # Logistiek: beddendruk op de IC
        },
        # MetaVision parameter 13306 - ids OVERLAP with doc_factors above, meaning something else
        "doc_factors_2": {
            10: "Circulatory",  # Circulatie: bv. bloeddruk, ritme, ischemie, hartfalen, ...
            11: "Respiratory",  # Sputumproductie met uitzuigbehoefte
            12: "Respiratory",  # Respiratie: bv. ventilatie, zuurstofbehoefte, beademingsduur
            13: "Neurological",  # Neurologie: bv. sub maximale EMV, IC verkregen zwakte, ...
            14: "Renal and homeostasis",  # Nierfunctie: bv. nierinsufficientie, zuur/base-stoornissen
            15: "Abdominal",  # Abdominaal: bv. leverinsufficientie
            17: "Hematology",  # Hematologie: bv. bloedingsrisico
            20: "Infection",  # Infectie: bv. infectie matig onder controle, isolatie
        },
    },
    "VUmc": {
        # choices.csv ids - one global list, so the field is not needed to disambiguate, but is
        # kept for symmetry with LUMC and so a second question could be added without ambiguity
        "doc_factors": {
            9: "General",  # leeftijd
            10: "General",  # geslacht
            11: "General",  # algemene (ziekte) indruk
            12: "General",  # reden van opname
            13: "General",  # electieve chirurgie
            14: "General",  # duur van IC-opname
            15: "General",  # eerdere heropname
            16: "Logistic",  # (verpleegkundige) zorgzwaarte
            17: "Logistic",  # Ontslag wegens beddengebrek (best patient)
            18: "Circulatory",  # Cardiaal belaste voorgeschiedenis
            19: "Respiratory",  # Pulmonaal belaste voorgeschiedenis
            20: "General",  # Maligniteit
            21: "General",  # Frailty
            22: "Respiratory",  # ademhalingsfrequentie
            23: "Respiratory",  # O2 toediening/zuurstofbehoefte
            24: "Respiratory",  # oxygenatie/PaO2
            25: "Respiratory",  # ventilatie/PaCO2
            26: "Respiratory",  # beademingsduur
            27: "Respiratory",  # hoestkracht
            28: "Respiratory",  # Sputumproductie met uitzuigbehoefte
            29: "Circulatory",  # hartfrequentie
            30: "Circulatory",  # Bloeddruk
            31: "Circulatory",  # Vochtbalans
            32: "Circulatory",  # (recent) hartfalen
            33: "Circulatory",  # (recente) cardiale ischemie
            34: "Circulatory",  # (recent) vasopressie/inotropie
            35: "Renal and homeostasis",  # nierfalen (bestaand of risico verslechtering)
            36: "Renal and homeostasis",  # Zuur-base of electrolytstoornissen
            37: "Neurological",  # verlaagd bewustzijn/lage EMV score
            38: "Neurological",  # delier
            39: "Neurological",  # ICU acquired weakness / CIPNP
            40: "Neurological",  # (recente) sedativa/anti-deliermedicatie
            41: "Abdominal",  # Leverfalen
            42: "Hematology",  # Hematologie patient
            43: "Hematology",  # (recidiverende) bloedingen
            44: "Infection",  # Moeizame source control
            45: "Infection",  # recidiverende infecties
            46: "Logistic",  # Isolatiemaatregelen
            47: "Infection",  # Immuungecompromiteerd
        },
    },
}

#: Display labels for physician-factor tokens. Analysis keys remain field and option ids because
#: description text can contain separators. Tests require one label for every mapped option.
PHYSICIAN_FACTOR_LABELS: dict[str, dict[str, dict[int, str]]] = {
    "LUMC": {
        "doc_factors": {
            1: "Demografisch (leeftijd, geslacht, etniciteit)",
            2: "Opname kenmerk: reden van opname",
            4: "Comorbiditeit: Cardiaal",
            5: "Comorbiditeit: Pulmonaal",
            6: "Comorbiditeit: Maligniteit",
            7: "Comorbiditeit: Kwetsbaarheid",
            10: "Logistiek: zorgzwaarte voor afdeling",
            11: "Opname kenmerk: duur van opname of heropname",
            12: "Comorbiditeit: Overig",
            13: "Logistiek: beddendruk op de IC",
        },
        "doc_factors_2": {
            10: "Circulatie: bv. bloeddruk, ritme, ischemie, hartfalen, ...",
            11: "Sputumproductie met uitzuigbehoefte",
            12: "Respiratie: bv. ventilatie, zuurstofbehoefte, beademingsduur",
            13: "Neurologie: bv. sub maximale EMV, IC verkregen zwakte, ...",
            14: "Nierfunctie: bv. nierinsufficientie, zuur/base-stoornissen",
            15: "Abdominaal: bv. leverinsufficientie",
            17: "Hematologie: bv. bloedingsrisico",
            20: "Infectie: bv. infectie matig onder controle, isolatie",
        },
    },
    "VUmc": {
        "doc_factors": {
            9: "leeftijd",
            10: "geslacht",
            11: "algemene (ziekte) indruk",
            12: "reden van opname",
            13: "electieve chirurgie",
            14: "duur van IC-opname",
            15: "eerdere heropname",
            16: "(verpleegkundige) zorgzwaarte",
            17: "Ontslag wegens beddengebrek (best patient)",
            18: "Cardiaal belaste voorgeschiedenis",
            19: "Pulmonaal belaste voorgeschiedenis",
            20: "Maligniteit",
            21: "Frailty",
            22: "ademhalingsfrequentie",
            23: "O2 toediening/zuurstofbehoefte",
            24: "oxygenatie/PaO2",
            25: "ventilatie/PaCO2",
            26: "beademingsduur",
            27: "hoestkracht",
            28: "Sputumproductie met uitzuigbehoefte",
            29: "hartfrequentie",
            30: "Bloeddruk",
            31: "Vochtbalans",
            32: "(recent) hartfalen",
            33: "(recente) cardiale ischemie",
            34: "(recent) vasopressie/inotropie",
            35: "nierfalen (bestaand of risico verslechtering)",
            36: "Zuur-base of electrolytstoornissen",
            37: "verlaagd bewustzijn/lage EMV score",
            38: "delier",
            39: "ICU acquired weakness / CIPNP",
            40: "(recente) sedativa/anti-deliermedicatie",
            41: "Leverfalen",
            42: "Hematologie patient",
            43: "(recidiverende) bloedingen",
            44: "Moeizame source control",
            45: "recidiverende infecties",
            46: "Isolatiemaatregelen",
            47: "Immuungecompromiteerd",
        },
    },
}

#: VUmc options absent from the trial data but retained from the complete form vocabulary.
#: Option 40 maps to Neurological because it describes sedation and anti-delirium medication.
UNSELECTED_FACTOR_OPTIONS = {
    ("VUmc", "doc_factors", option_id)
    for option_id in (
        13,
        14,
        17,
        22,
        23,
        26,
        27,
        28,
        29,
        31,
        36,
        39,
        40,
        41,
        42,
        44,
        47,
    )
}

#: Closed free-text factor vocabulary. Unknown non-null answers indicate incomplete mapping.
PHYSICIAN_FREE_TEXT_CATEGORIES: dict[str, str] = {
    "gewicht-BMI": "General",
    "beleid": "Logistic",
    "Niet-IC beleid": "Logistic",
    "schizofreen": "Logistic",
    "persoonlijkheids problematiek/borderline": "Logistic",
    "depressief en verward": "Logistic",
    "Psychiatrische voorgeschiedenis": "Logistic",
    "Psychiatrisch zoektebeeld": "Logistic",
    "Psychiatrie": "Logistic",
    "woede-uitbarsting": "Logistic",
    "st na pneumectomie": "Respiratory",
    "COVID": "Respiratory",
    "bewustzijn": "Neurological",
    # Use the decoded source spelling; strict matching detects encoding regressions.
    "Hematologie pati\u00ebnt": "Hematology",
    "pijnstilling": "Medication",
    "nog onvoldoende pijncontrole": "Medication",
    # The comma is part of this single free-text factor.
    "pancreatitis, maagperforatie": "Abdominal",
}

#: model feature name -> category.
MODEL_FEATURE_CATEGORIES: dict[str, str] = {
    # General
    "age": "General",
    "body_mass_index__mean__last_24h": "General",
    "length_of_stay_hours": "General",
    "sex__female": "General",
    "sex__male": "General",
    "time_since_start": "General",
    # Respiratory
    "cough_stimulus__mode__last_24h__cough_reflex_normal": "Respiratory",
    "cough_stimulus__mode__last_24h__cough_reflex_reasonable": "Respiratory",
    "o2_flow__change_since_first__last_24h": "Respiratory",
    "o2_flow__mean__last_24h": "Respiratory",
    "o2_saturation__mean__last_24h": "Respiratory",
    "o2_saturation__change_since_first__last_24h": "Respiratory",
    "tidal_volume_per_kg__mean__last_24h__padded": "Respiratory",
    "pao2_over_fio2__last__overall": "Respiratory",
    "fio2__mean__last_24h": "Respiratory",
    "peep__maximum__last_24h": "Respiratory",
    "pco2_arterial__mean__last_24h": "Respiratory",
    "respiratory_rate_measured__last__overall": "Respiratory",
    "respiratory_rate_measured__mean__last_24h__padded": "Respiratory",
    "tracheobronchial_toilet_quantity__total_dose__last_24h": "Respiratory",
    "tracheobronchial_toilet_quantity__total_value__last_24h": "Respiratory",
    # Circulatory
    "creatine_kinase_mb_mass__is_measured__last_24h": "Circulatory",
    "creatine_kinase_mb_mass__last__overall": "Circulatory",
    "heart_rate__mean__last_24h__padded": "Circulatory",
    "heart_rate__mean__last_24h": "Circulatory",
    "arterial_blood_pressure_diastolic__mean__last_24h": "Circulatory",
    "arterial_blood_pressure_systolic__mean__last_24h": "Circulatory",
    "arterial_blood_pressure_mean__change_since_first__last_24h": "Circulatory",
    "lactate__last__overall": "Circulatory",
    "troponin_t__maximum__last_24h": "Circulatory",
    # Neurological
    "glasgow_coma_scale_total__maximum__last_24h": "Neurological",
    "pupil_diameter_max__mean__last_24h": "Neurological",
    "glasgow_coma_scale_total__minimum__last_24h": "Neurological",
    "richmond_agitation_sedation_scale_score__maximum__last_24h": "Neurological",
    # Renal and homeostasis
    "ph_arterial__last__overall": "Renal and homeostasis",
    "glucose__minimum__last_24h": "Renal and homeostasis",
    "sodium__mean__last_24h": "Renal and homeostasis",
    "albumin__mean__last_24h__padded": "Renal and homeostasis",
    "ureum_over_creatinine__mean__last_24h": "Renal and homeostasis",
    "ureum__mean__last_24h": "Renal and homeostasis",
    "fluid_out_urine__total_dose__last_24h": "Renal and homeostasis",
    "creatinine__last__overall": "Renal and homeostasis",
    "phosphate__mean__last_24h": "Renal and homeostasis",
    "chloride__mean__last_24h": "Renal and homeostasis",
    "base_excess__mean__last_24h": "Renal and homeostasis",
    "potassium__mean__last_24h": "Renal and homeostasis",
    "albumin__mean__last_24h": "Renal and homeostasis",
    "fluid_out_urine__total_value__last_24h": "Renal and homeostasis",
    "magnesium__mean__last_24h": "Renal and homeostasis",
    # Hematology
    "thrombocytes__mean__last_24h": "Hematology",
    "hemoglobin__mean__last_24h__padded": "Hematology",
    "prothrombin_time_inr__mean__last_24h__padded": "Hematology",
    "hemoglobin__mean__last_24h": "Hematology",
    "prothrombin_time_inr__mean__last_24h": "Hematology",
    "activated_partial_thromboplastin_time__mean__last_24h": "Hematology",
    # Liver chemistry maps to the shared Abdominal category.
    "bilirubin_total__mean__last_24h": "Abdominal",
    "gamma_glutamyl_transferase__mean__last_24h": "Abdominal",
    "alanine_transaminase__mean__last_24h": "Abdominal",
    "aspartate_transaminase__mean__last_24h": "Abdominal",
    "alkaline_phosphatase__mean__last_24h": "Abdominal",
    # Infection
    # LDH remains Infection because its non-specific category requires a clinical decision.
    "lactate_dehydrogenase__mean__last_24h": "Infection",
    "temperature_skin__mean__last_24h": "Infection",
    "temperature_internal__maximum__last_24h": "Infection",
    "leukocytes__mean__last_24h__padded": "Infection",
    "c_reactive_protein__mean__last_24h": "Infection",
    "temperature__minimum__last_24h": "Infection",
    "temperature__change_since_first__last_24h": "Infection",
    "leukocytes__mean__last_24h": "Infection",
}

# Readable labels used in the supplemental mapping tables. Several site-specific model variables
# describe the same clinical feature and therefore share one display label.
MODEL_FEATURE_LABELS: dict[str, str] = {
    "age": "Age",
    "body_mass_index__mean__last_24h": "Body mass index",
    "length_of_stay_hours": "Length of stay (hours)",
    "time_since_start": "Length of stay (hours)",
    "sex__female": "Sex",
    "sex__male": "Sex",
    "cough_stimulus__mode__last_24h__cough_reflex_normal": "Cough reflex",
    "cough_stimulus__mode__last_24h__cough_reflex_reasonable": "Cough reflex",
    "o2_flow__change_since_first__last_24h": "Supplemental oxygen flow",
    "o2_flow__mean__last_24h": "Supplemental oxygen flow",
    "o2_saturation__change_since_first__last_24h": "Oxygen saturation",
    "o2_saturation__mean__last_24h": "Oxygen saturation",
    "tidal_volume_per_kg__mean__last_24h__padded": "Tidal volume (per kg)",
    "pao2_over_fio2__last__overall": "PaO2/FiO2 ratio",
    "fio2__mean__last_24h": "Fraction of inspired oxygen (FiO2)",
    "peep__maximum__last_24h": "PEEP",
    "pco2_arterial__mean__last_24h": "PaCO2",
    "respiratory_rate_measured__last__overall": "Respiratory rate",
    "respiratory_rate_measured__mean__last_24h__padded": "Respiratory rate",
    "tracheobronchial_toilet_quantity__total_dose__last_24h": (
        "Tracheobronchial suctioning (number of times)"
    ),
    "tracheobronchial_toilet_quantity__total_value__last_24h": (
        "Tracheobronchial suctioning (number of times)"
    ),
    "creatine_kinase_mb_mass__is_measured__last_24h": "Creatine kinase-MB",
    "creatine_kinase_mb_mass__last__overall": "Creatine kinase-MB",
    "heart_rate__mean__last_24h": "Heart rate",
    "heart_rate__mean__last_24h__padded": "Heart rate",
    "arterial_blood_pressure_diastolic__mean__last_24h": (
        "Diastolic arterial blood pressure"
    ),
    "arterial_blood_pressure_systolic__mean__last_24h": (
        "Systolic arterial blood pressure"
    ),
    "arterial_blood_pressure_mean__change_since_first__last_24h": (
        "Mean arterial blood pressure"
    ),
    "lactate__last__overall": "Lactate",
    "troponin_t__maximum__last_24h": "Troponin-T",
    "glasgow_coma_scale_total__maximum__last_24h": ("Glasgow Coma Scale (GCS) total"),
    "glasgow_coma_scale_total__minimum__last_24h": ("Glasgow Coma Scale (GCS) total"),
    "pupil_diameter_max__mean__last_24h": "Pupil diameter",
    "richmond_agitation_sedation_scale_score__maximum__last_24h": (
        "Richmond Agitation-Sedation Scale (RASS) score"
    ),
    "ph_arterial__last__overall": "Arterial pH",
    "glucose__minimum__last_24h": "Glucose",
    "sodium__mean__last_24h": "Sodium",
    "albumin__mean__last_24h": "Albumin",
    "albumin__mean__last_24h__padded": "Albumin",
    "ureum_over_creatinine__mean__last_24h": "Urea/creatinine ratio",
    "ureum__mean__last_24h": "Urea",
    "fluid_out_urine__total_dose__last_24h": "Urine output",
    "fluid_out_urine__total_value__last_24h": "Urine output",
    "creatinine__last__overall": "Creatinine",
    "phosphate__mean__last_24h": "Phosphate",
    "chloride__mean__last_24h": "Chloride",
    "base_excess__mean__last_24h": "Base excess",
    "potassium__mean__last_24h": "Potassium",
    "magnesium__mean__last_24h": "Magnesium",
    "thrombocytes__mean__last_24h": "Thrombocytes",
    "hemoglobin__mean__last_24h": "Hemoglobin",
    "hemoglobin__mean__last_24h__padded": "Hemoglobin",
    "prothrombin_time_inr__mean__last_24h": "Prothrombin time",
    "prothrombin_time_inr__mean__last_24h__padded": "Prothrombin time",
    "activated_partial_thromboplastin_time__mean__last_24h": (
        "Activated partial thromboplastin time (APTT)"
    ),
    "bilirubin_total__mean__last_24h": "Total bilirubin",
    "gamma_glutamyl_transferase__mean__last_24h": "Gamma-glutamyl transferase",
    "alanine_transaminase__mean__last_24h": "Alanine transaminase",
    "aspartate_transaminase__mean__last_24h": "Aspartate transaminase",
    "alkaline_phosphatase__mean__last_24h": "Alkaline phosphatase",
    "lactate_dehydrogenase__mean__last_24h": "Lactate dehydrogenase",
    "temperature_skin__mean__last_24h": "Skin temperature",
    "temperature_internal__maximum__last_24h": "Internal temperature",
    "temperature__minimum__last_24h": "Temperature",
    "temperature__change_since_first__last_24h": "Temperature",
    "leukocytes__mean__last_24h": "Leukocytes",
    "leukocytes__mean__last_24h__padded": "Leukocytes",
    "c_reactive_protein__mean__last_24h": "C-reactive protein",
}

PHYSICIAN_FACTOR_DISPLAY_LABELS: dict[str, str] = {
    "(recent) hartfalen": "Heart failure",
    "(recent) vasopressie/inotropie": "Recent vasopressor or inotrope use",
    "(recente) sedativa/anti-deliermedicatie": "Recent sedative or antidelirium medication",
    "Abdominaal: bv. leverinsufficientie": "Abdominal (e.g. liver failure)",
    "pancreatitis, maagperforatie": "Abdominal comorbidity",
    "(recente) cardiale ischemie": "Myocardial ischemia",
    "Bloeddruk": "Blood pressure",
    "Cardiaal belaste voorgeschiedenis": "Cardiovascular comorbidity",
    "Circulatie: bv. bloeddruk, ritme, ischemie, hartfalen, ...": (
        "Circulation (e.g. blood pressure, heart rhythm, ischemia, heart failure)"
    ),
    "Comorbiditeit: Cardiaal": "Cardiovascular comorbidity",
    "Comorbiditeit: Kwetsbaarheid": "Frailty",
    "Comorbiditeit: Maligniteit": "Malignancy",
    "Comorbiditeit: Overig": "Comorbidity (other)",
    "Demografisch (leeftijd, geslacht, etniciteit)": (
        "Demographics (age, sex, ethnicity)"
    ),
    "Maligniteit": "Malignancy",
    "Opname kenmerk: duur van opname of heropname": (
        "Length of stay or previous readmission"
    ),
    "Opname kenmerk: reden van opname": "Reason for admission",
    "algemene (ziekte) indruk": "General impression",
    "eerdere heropname": "Length of stay or previous readmission",
    "electieve chirurgie": "Elective surgery",
    "Frailty": "Frailty",
    "geslacht": "Demographics (age, sex, ethnicity)",
    "gewicht-BMI": "Body mass index",
    "leeftijd": "Demographics (age, sex, ethnicity)",
    "reden van opname": "Reason for admission",
    "(recidiverende) bloedingen": "Bleeding risk",
    "Hematologie patient": "Hematologic comorbidity",
    "Hematologie: bv. bloedingsrisico": "Hematology (e.g. bleeding risk)",
    "hartfrequentie": "Heart rate",
    "hoestkracht": "Cough strength",
    "ICU acquired weakness / CIPNP": "ICU-acquired weakness",
    "Immuungecompromiteerd": "Immunocompromised",
    "Infectie: bv. infectie matig onder controle, isolatie": (
        "Infection (e.g. source control, infection risk, barrier nursing)"
    ),
    "recidiverende infecties": "Recurrent infections",
    "(verpleegkundige) zorgzwaarte": "Burden of nursing care",
    "Isolatiemaatregelen": "Barrier or isolation nursing",
    "Logistiek: zorgzwaarte voor afdeling": "Burden of nursing care",
    "Logistiek: beddendruk op de IC": "ICU bed pressure",
    "Moeizame source control": "Difficult source control",
    "O2 toediening/zuurstofbehoefte": "Supplemental oxygen requirement",
    "Ontslag wegens beddengebrek (best patient)": "Discharge because of ICU bed pressure",
    "persoonlijkheids problematiek/borderline": "Psychiatric comorbidity",
    "pijnstilling": "Uncontrolled pain",
    "Neurologie: bv. sub maximale EMV, IC verkregen zwakte, ...": (
        "Neurological (e.g. consciousness, ICU-acquired weakness, deterioration)"
    ),
    "delier": "Delirium",
    "verlaagd bewustzijn/lage EMV score": "Consciousness",
    "Nierfunctie: bv. nierinsufficientie, zuur/base-stoornissen": (
        "Renal and homeostasis (e.g. kidney failure, acid-base disorders)"
    ),
    "nierfalen (bestaand of risico verslechtering)": "Kidney failure",
    "COVID": "COVID-19",
    "Comorbiditeit: Pulmonaal": "Pulmonary comorbidity",
    "Pulmonaal belaste voorgeschiedenis": "Pulmonary comorbidity",
    "Respiratie: bv. ventilatie, zuurstofbehoefte, beademingsduur": (
        "Respiration (e.g. ventilation, oxygenation, ventilation duration)"
    ),
    "Sputumproductie met uitzuigbehoefte": "Tracheal suctioning",
    "oxygenatie/PaO2": "Oxygenation",
    "ademhalingsfrequentie": "Respiratory rate",
    "beademingsduur": "Duration of mechanical ventilation",
    "duur van IC-opname": "Length of stay or previous readmission",
    "Leverfalen": "Liver failure",
    "st na pneumectomie": "Previous pneumonectomy",
    "ventilatie/PaCO2": "Ventilation and PaCO2",
    "Vochtbalans": "Fluid balance",
    "Zuur-base of electrolytstoornissen": "Acid-base or electrolyte disorder",
}


def _normalize(value):
    """Strip surrounding whitespace from a text value; leave anything else untouched."""
    return value.strip() if isinstance(value, str) else value


def _normalize_free_text(value):
    """Normalize free-text whitespace and case for matching."""
    return value.strip().casefold() if isinstance(value, str) else value


def model_feature_labels(values: pd.Series) -> pd.Series:
    """Map model variable names to readable supplemental-table labels."""
    return vocabulary.map_strict(
        values.map(_normalize), MODEL_FEATURE_LABELS, what="model feature label(s)"
    )


def physician_factor_display_labels(values: pd.Series) -> pd.Series:
    """Map source physician-factor labels to harmonized English labels."""
    return vocabulary.map_strict(
        values.map(_normalize),
        PHYSICIAN_FACTOR_DISPLAY_LABELS,
        what="physician factor display label(s)",
    )


def comparable_categories() -> tuple[str, ...]:
    """Return categories shared by physician factors and model features."""
    return tuple(c for c in CATEGORIES if c not in PHYSICIAN_ONLY_CATEGORIES)


def categorize_physician_factors(
    doc_data: pd.DataFrame,
    *,
    hospital_column: str = "hospital",
    field_column: str = "factor_field",
    id_column: str = "factor_id",
    target_column: str = "new_factors",
) -> pd.DataFrame:
    """Categorize physician factors by hospital, source field, and option id.

    Unknown hospital-field pairs or option ids raise. The input is not modified.
    """
    result = doc_data.copy()
    categories = pd.Series(pd.NA, index=result.index, dtype="object")
    unmapped: list = []

    for (hospital, field), rows in result.groupby(
        [hospital_column, field_column], dropna=False
    ):
        mapping = PHYSICIAN_FACTOR_CATEGORIES.get(hospital, {}).get(field)
        if mapping is None:
            raise KeyError(
                f"no physician factor list for hospital {hospital!r} field {field!r} - "
                f"known: {sorted((h, f) for h in PHYSICIAN_FACTOR_CATEGORIES for f in PHYSICIAN_FACTOR_CATEGORIES[h])}"
            )
        ids = pd.to_numeric(rows[id_column], errors="coerce").astype("Int64")
        # "coerce" would otherwise turn a non-numeric id into NA, which reads downstream as
        # "no factor selected" - indistinguishable from a genuinely blank answer
        unconvertible = rows[id_column][ids.isna() & rows[id_column].notna()]
        unmapped += [f"{hospital}/{field}:{value!r}" for value in unconvertible]
        unmapped += [
            f"{hospital}/{field}:{value}"
            for value in vocabulary.unmapped_values(ids, mapping)
        ]
        categories.loc[rows.index] = ids.map(mapping)

    vocabulary.raise_unmapped(
        sorted(set(unmapped)),
        what="physician factor option(s)",
        known=[
            f"{h}/{f}:{i}"
            for h, fields in PHYSICIAN_FACTOR_CATEGORIES.items()
            for f, ids in fields.items()
            for i in ids
        ],
    )
    result[target_column] = categories
    return result


#: separator between a factor's source field and its option id in a serialized factor token.
FACTOR_TOKEN_SEPARATOR = ":"


def factor_labels(
    factors: pd.DataFrame,
    *,
    hospital_column: str = "hospital",
    field_column: str = "factor_field",
    id_column: str = "factor_id",
) -> pd.Series:
    """Return readable labels for selected options, raising on unknown ids."""
    result = pd.Series(pd.NA, index=factors.index, dtype="object")
    unknown: list = []

    for (hospital, field), rows in factors.groupby(
        [hospital_column, field_column], dropna=False
    ):
        mapping = PHYSICIAN_FACTOR_LABELS.get(hospital, {}).get(field, {})
        ids = pd.to_numeric(rows[id_column], errors="coerce").astype("Int64")
        unknown += [
            f"{hospital}/{field}:{value}"
            for value in vocabulary.unmapped_values(ids, mapping)
        ]
        result.loc[rows.index] = ids.map(mapping)

    vocabulary.raise_unmapped(
        sorted(set(unknown)),
        what="physician factor label(s)",
        known=["see PHYSICIAN_FACTOR_LABELS"],
    )
    return result


def factor_token(field: str, option_id) -> str:
    """Serialize a selected factor as ``field:id`` for stable delimiter handling."""
    return f"{field}{FACTOR_TOKEN_SEPARATOR}{option_id}"


def explode_factor_tokens(
    frame: pd.DataFrame,
    *,
    hospital_column: str = "hospital",
    factors_column: str = "doc_factors",
    separator: str = "||",
) -> pd.DataFrame:
    """Expand joined ``field:id`` tokens to one row per selected factor."""
    rows = []
    for index, (hospital, tokens) in frame[
        [hospital_column, factors_column]
    ].iterrows():
        if tokens is None or (not isinstance(tokens, str) and pd.isnull(tokens)):
            continue
        for token in str(tokens).split(separator):
            if not token.strip():
                continue
            field, _, option_id = token.rpartition(FACTOR_TOKEN_SEPARATOR)
            rows.append(
                {
                    "source_index": index,
                    "hospital": hospital,
                    "factor_field": field,
                    "factor_id": option_id,
                }
            )
    return pd.DataFrame(
        rows, columns=["source_index", "hospital", "factor_field", "factor_id"]
    )


def categorize_free_text(
    free_text: pd.Series, *, mapping: Mapping[str, str] | None = None
) -> pd.Series:
    """Categorize free-text factors after whitespace and case normalization.

    Unknown non-null answers raise.
    """
    if mapping is None:
        mapping = PHYSICIAN_FREE_TEXT_CATEGORIES
    normalized = free_text.map(_normalize_free_text)
    lookup = {
        _normalize_free_text(text): category for text, category in mapping.items()
    }
    return vocabulary.map_strict(
        normalized, lookup, what="free-text physician factor(s)"
    )


def categorize_model_features(
    shap_data: pd.DataFrame,
    *,
    feature_column: str = "feature_name",
    target_column: str = "new_feature_name",
    mapping: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Add mapped model-feature categories without modifying the input."""
    if mapping is None:
        mapping = MODEL_FEATURE_CATEGORIES
    result = shap_data.copy()
    result[target_column] = vocabulary.map_strict(
        result[feature_column].map(_normalize), mapping, what="model feature(s)"
    )
    return result
