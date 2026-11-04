"""Format exported summaries of adverse events in low-risk groups."""

from pathlib import Path

import pandas as pd

from algo_vs_doc.word_tables import WordTable
from algo_vs_doc.specialties import SPECIALTY_LABELS
from algo_vs_doc.subgroup_descriptives import admission_type


def individual_event_entries(cohorts: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return one local review row per event discharge, without source identifiers.

    Diagnostic codes describe diagnoses documented in the index-stay windows,
    not established causes of the subsequent adverse event. Combined cohorts
    are excluded to avoid duplicating site entries. Case numbers are run-local.
    """
    fields = [
        "icu_stay_id",
        "age_at_admission",
        "sex",
        "specialty",
        "admission_type",
        "icu_los",
        "doc_prediction",
        "doc_prediction_cat",
        "algo_prediction_at_doc_prediction",
        "has_icu_readmission_7days",
        "has_death_7days",
        "ventilation",
        "vasopressors_inotropes",
        "renal_replacement_therapy",
        "sofa_admission",
        "sofa_max",
        "sofa_discharge",
        "news_discharge",
        "ready_for_discharge",
        "index_episode_diagnoses_timeline",
        "readmission_episode_diagnoses_timeline",
    ]
    rows = []
    for hospital, frame in cohorts.items():
        if hospital == "Combined":
            continue
        physician_low = frame.doc_prediction_cat.eq("low")
        model_low = frame.algo_prediction_at_doc_prediction.lt(0.05)
        selected = frame.loc[frame.outcome.eq(True) & (physician_low | model_low)]
        entries = selected.reindex(columns=fields).copy()
        entries["admission_type"] = admission_type(selected)
        no_readmission = ~selected.has_icu_readmission_7days.astype(bool).to_numpy()
        for column in ():
            entries.loc[no_readmission, column] = "Not applicable"
        entries.insert(0, "hospital", hospital)
        entries.insert(1, "case_number", range(1, len(entries) + 1))
        prefix = {"VUmc": "V", "LUMC": "L"}.get(hospital, hospital[:1].upper())
        entries.insert(2, "case_label", [f"{prefix}{n}" for n in entries.case_number])
        entries.insert(
            3,
            "review_case_key",
            entries["hospital"].astype(str) + ":" + entries["icu_stay_id"].astype(str),
        )
        entries["physician_low_risk"] = selected.doc_prediction_cat.eq("low").to_numpy()
        entries["model_risk_below_5_percent"] = (
            selected.algo_prediction_at_doc_prediction.lt(0.05).to_numpy()
        )
        rows.append(entries)
    return pd.concat(rows, ignore_index=True)


ANNOTATION_COLUMNS = (
    "review_case_key",
    "case_label",
    "index_admission_diagnosis",
    "readmission_diagnosis",
    "clinical_interpretation",
)


def update_annotation_template(entries: pd.DataFrame, path: str | Path) -> pd.DataFrame:
    """Add current cases to a local annotation CSV without overwriting manual edits."""
    path = Path(path)
    current = entries[["review_case_key", "case_label"]].drop_duplicates().copy()
    if current["review_case_key"].duplicated().any():
        raise ValueError("review_case_key must uniquely identify each review case")
    if path.exists():
        annotations = pd.read_csv(path, dtype=str, keep_default_na=False)
        missing = sorted(set(ANNOTATION_COLUMNS) - set(annotations.columns))
        if missing:
            raise KeyError(f"annotation file is missing {missing}")
        if annotations["review_case_key"].duplicated().any():
            raise ValueError(
                "annotation file contains duplicate review_case_key values"
            )
        annotations = annotations[list(ANNOTATION_COLUMNS)]
        additions = current.loc[
            ~current.review_case_key.isin(annotations.review_case_key)
        ].copy()
        for column in ANNOTATION_COLUMNS[2:]:
            additions[column] = ""
        annotations = pd.concat([annotations, additions], ignore_index=True)
        labels = current.set_index("review_case_key")["case_label"]
        active = annotations.review_case_key.isin(labels.index)
        annotations.loc[active, "case_label"] = annotations.loc[
            active, "review_case_key"
        ].map(labels)
    else:
        annotations = current.copy()
        for column in ANNOTATION_COLUMNS[2:]:
            annotations[column] = ""
    path.parent.mkdir(parents=True, exist_ok=True)
    annotations.to_csv(path, index=False)
    return annotations


def add_manual_annotations(
    entries: pd.DataFrame, annotations: pd.DataFrame
) -> pd.DataFrame:
    """Attach manually reviewed publication diagnoses by stable review key."""
    if annotations["review_case_key"].duplicated().any():
        raise ValueError("annotation file contains duplicate review_case_key values")
    columns = ["review_case_key", *ANNOTATION_COLUMNS[2:]]
    return entries.merge(
        annotations[columns], on="review_case_key", how="left", validate="one_to_one"
    )


ENTRY_LABELS = {
    "age_at_admission": "Age at ICU admission, 10-year group midpoint",
    "sex": "Sex",
    "specialty": "Admitting specialty",
    "admission_type": "Admission type",
    "icu_los": "Index ICU length of stay, days",
    "doc_prediction": "Physician predicted risk, %",
    "doc_prediction_cat": "Physician risk category",
    "algo_prediction_at_doc_prediction": "Model predicted risk, %",
    "has_icu_readmission_7days": "ICU readmission within seven days",
    "has_death_7days": "Death within seven days",
    "ventilation": "Mechanical ventilation",
    "vasopressors_inotropes": "Vasopressors or inotropes",
    "renal_replacement_therapy": "Renal-replacement therapy",
    "sofa_admission": "SOFA at ICU admission",
    "sofa_max": "Maximum SOFA",
    "sofa_discharge": "SOFA at ICU discharge",
    "news_discharge": "NEWS at ICU discharge",
    "ready_for_discharge": "Assessed as ready for discharge",
    "physician_low_risk": "Physician-assigned low risk",
    "model_risk_below_5_percent": "Model risk below 5%",
}
BOOLEAN_FIELDS = {
    "has_icu_readmission_7days",
    "has_death_7days",
    "ventilation",
    "vasopressors_inotropes",
    "renal_replacement_therapy",
    "ready_for_discharge",
    "physician_low_risk",
    "model_risk_below_5_percent",
}
INTEGER_FIELDS = {
    "sofa_admission",
    "sofa_max",
    "sofa_discharge",
    "news_discharge",
}


def _entry_value(column: str, value) -> str:
    if pd.isna(value) or str(value).strip() == "":
        if column in {"index_admission_diagnosis", "readmission_diagnosis"}:
            return ""
        return "Not available"
    if column in BOOLEAN_FIELDS:
        return {
            "true": "+",
            "false": "-",
            "1": "+",
            "0": "-",
            "1.0": "+",
            "0.0": "-",
        }.get(str(value).lower(), str(value))
    if column in {"doc_prediction", "algo_prediction_at_doc_prediction"}:
        return f"{100 * float(value):.1f}"
    if column == "age_at_admission":
        return str(int(float(value) // 10 * 10 + 5))
    if column == "icu_los":
        return f"{float(value) / 24:.1f}"
    if column in INTEGER_FIELDS:
        return f"{float(value):.0f}"
    if column == "specialty":
        return SPECIALTY_LABELS.get(value, str(value))
    if column == "admission_type":
        return {
            "medical": "Medical",
            "elective_surgery": "Elective surgical",
            "emergency_surgery": "Emergency surgical",
            "other_or_unknown": "Unknown",
        }.get(value, str(value))
    if column == "sex":
        return {"male": "M", "female": "F", "m": "M", "f": "F"}.get(
            str(value).lower(), str(value)
        )
    if column == "doc_prediction_cat":
        return {"low": "L", "medium": "M", "high": "H"}.get(
            str(value).lower(), str(value)
        )
    return str(value)


def local_diagnosis_review_word_tables(entries: pd.DataFrame) -> list[WordTable]:
    """Present timestamped diagnosis timelines for local review only."""
    rows = []
    for _, entry in entries.iterrows():
        rows.append(
            {
                "Case": entry["case_label"],
                "Internal review key": entry["review_case_key"],
                "Index episode diagnoses": _entry_value(
                    "index_episode_diagnoses_timeline",
                    entry.get("index_episode_diagnoses_timeline"),
                ),
                "Readmission episode diagnoses": _entry_value(
                    "readmission_episode_diagnoses_timeline",
                    entry.get("readmission_episode_diagnoses_timeline"),
                ),
            }
        )
    return [
        WordTable(
            "Diagnosis timelines for local clinical review",
            pd.DataFrame(rows),
            font_size=8,
            landscape=True,
            notes=(
                "This table is for local review and is not intended for publication.",
                "Offsets are hours relative to the applicable ICU admission; negative values precede admission and positive values follow it.",
                "The index timeline includes diagnoses first documented from seven days before index ICU admission through index ICU discharge. The readmission timeline includes diagnoses documented after index discharge through seven days after discharge, with offsets relative to the recorded ICU readmission time.",
                "Condition flags are shown at +0.0 h for the corresponding ICU admission; this denotes admission status rather than a documentation timestamp.",
                "Diagnosis timestamps reflect documentation and do not establish the onset or cause of an adverse event.",
            ),
        )
    ]


def individual_event_word_tables(entries: pd.DataFrame) -> list[WordTable]:
    """Present one row per event discharge and one value per characteristic cell."""
    groups = {
        "Predictions": [
            "doc_prediction",
            "doc_prediction_cat",
            "algo_prediction_at_doc_prediction",
        ],
        "Seven-day outcome": ["has_icu_readmission_7days", "has_death_7days"],
        "Clinical characteristics": [
            "age_at_admission",
            "sex",
            "specialty",
            "admission_type",
            "icu_los",
        ],
        "Life support": [
            "ventilation",
            "vasopressors_inotropes",
            "renal_replacement_therapy",
        ],
        "Severity": ["sofa_admission", "sofa_max", "sofa_discharge", "news_discharge"],
        "Diagnoses": [
            "index_admission_diagnosis",
            "readmission_diagnosis",
        ],
    }
    short_labels = {
        "doc_prediction": "Pp",
        "doc_prediction_cat": "Pc",
        "algo_prediction_at_doc_prediction": "M",
        "has_icu_readmission_7days": "R",
        "has_death_7days": "D",
        "age_at_admission": "Age",
        "sex": "Sex",
        "specialty": "Specialty",
        "admission_type": "Admission type",
        "icu_los": "LOS",
        "ventilation": "MV",
        "vasopressors_inotropes": "VA",
        "renal_replacement_therapy": "RRT",
        "sofa_admission": "Adm",
        "sofa_max": "Max",
        "sofa_discharge": "Dis",
        "news_discharge": "NEWS",
        "index_admission_diagnosis": "Index stay",
        "readmission_diagnosis": "Readmission",
        "ready_for_discharge": "Ready for discharge",
    }
    columns = [column for group in groups.values() for column in group]
    rows = []
    for _, entry in entries.iterrows():
        row = {"Case": entry["case_label"]}
        row.update(
            {
                short_labels[column]: _entry_value(column, entry.get(column))
                for column in columns
            }
        )
        if row["R"] == "-":
            row["Readmission"] = "NA"
        rows.append(row)
    return [
        WordTable(
            "eTable 11. Individual characteristics of discharges followed by an event despite low predicted risk",
            pd.DataFrame(rows, columns=["Case", *[short_labels[c] for c in columns]]),
            font_size=8,
            note_font_size=8,
            landscape=True,
            header_groups=(
                ("Prediction", ("Pp", "Pc", "M")),
                ("Outcome", ("R", "D")),
                ("Life Support", ("MV", "VA", "RRT")),
                ("SOFA", ("Adm", "Max", "Dis")),
                ("Diagnoses", ("Index stay", "Readmission")),
            ),
            notes=(
                "Definition: seven-day event and physician-assigned low risk or model predicted risk <5%. To reduce re-identification risk, age is grouped into 10-year categories and displayed as the midpoint of the category (for example, 65 represents 60-69 years). LOS is reported in days, predicted risks are percentages, and + and - indicate presence and absence, respectively. NA indicates not applicable. Missing documentation does not establish absence of diagnoses.",
                "Abbreviations: LOS, ICU length of stay; Pp, physician percentage prediction; Pc, physician risk category (L, low; M, medium; H, high); Mo, model prediction; R, ICU readmission within seven days; D, death within seven days; MV, mechanical ventilation; VA, vasoactive medication; RRT, renal-replacement therapy; SOFA, Sequential Organ Failure Assessment; Adm, at ICU admission; Max, maximum during ICU stay; Dis, at ICU discharge; NEWS, National Early Warning Score at ICU discharge; ICU, intensive care unit; M, male; F, female.",
            ),
        )
    ]
