"""Describe event discharges and the origins of seven-day ICU readmissions."""

from collections.abc import Mapping

import pandas as pd

from algo_vs_doc import specialties, subgroup_descriptives
from algo_vs_doc.specialties import SPECIALTY_LABELS
from algo_vs_doc.word_tables import WordTable


TITLE = "eTable 3. Characteristics of discharges followed by ICU readmission or death"
EVENT_GROUPS = (
    ("Readmission only", True, False),
    ("Death only", False, True),
    ("Readmission and death", True, True),
)
CARE_LEVEL_LABELS = {
    "general_ward": "General ward",
    "nursing_ward": "General ward",
    "high_dependency_unit": "High-dependency care",
}


def _n_percentage(values: pd.Series, denominator: int) -> str:
    count = int(values.astype("boolean").fillna(False).sum())
    return f"{count} ({100 * count / denominator:.1f})" if denominator else "-"


def _mean_sd(values: pd.Series) -> str:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return "-"
    if len(values) == 1:
        age_group_midpoint = int(values.iloc[0] // 10 * 10 + 5)
        return f"{age_group_midpoint} (-)"
    return f"{values.mean():.1f} ({values.std():.1f})"


def _median_iqr(values: pd.Series, *, divisor: float = 1) -> str:
    values = pd.to_numeric(values, errors="coerce").dropna() / divisor
    if values.empty:
        return "-"
    lower, median, upper = values.quantile([0.25, 0.5, 0.75])
    return f"{median:.1f} ({lower:.1f}-{upper:.1f})"


def _category_count(values: pd.Series, category: str, denominator: int) -> str:
    count = int(values.fillna("unknown").eq(category).sum())
    return f"{count} ({100 * count / denominator:.1f})" if denominator else "-"


def _specialty_rows(frame: pd.DataFrame, denominator: int) -> dict[str, str]:
    values = frame["specialty"].fillna("unspecified")
    rows = {
        "Surgical": _category_count(
            values.isin(specialties.SURGICAL_SPECIALTIES), True, denominator
        ),
    }
    for category in specialties.SURGICAL_SPECIALTIES:
        rows[SPECIALTY_LABELS[category]] = _category_count(
            values, category, denominator
        )
    rows["Medical"] = _category_count(
        values.isin(specialties.MEDICAL_SPECIALTIES), True, denominator
    )
    for category in specialties.MEDICAL_SPECIALTIES:
        rows[SPECIALTY_LABELS[category]] = _category_count(
            values, category, denominator
        )
    rows["Unspecified"] = _category_count(values, "unspecified", denominator)
    return rows


def _origin_rows(
    frame: pd.DataFrame,
    column: str,
    categories: list[str],
    *,
    applicable: bool,
    available: bool = True,
) -> dict[str, str]:
    if not applicable or not available:
        return {category: "-" for category in categories}
    values = frame.get(column, pd.Series(pd.NA, index=frame.index)).fillna("unknown")
    denominator = len(frame)
    return {
        category: _category_count(
            values.map(
                lambda value: SPECIALTY_LABELS.get(
                    value,
                    CARE_LEVEL_LABELS.get(
                        value, str(value).replace("_", " ").capitalize()
                    ),
                )
            ),
            category,
            denominator,
        )
        for category in categories
    }


def _event_summary(
    frame: pd.DataFrame,
    *,
    readmitted: bool,
    origin_specialties: list[str],
    origin_settings: list[str],
    origin_setting_available: bool,
) -> pd.DataFrame:
    total = len(frame)
    admission_type = subgroup_descriptives.admission_type(frame)
    care_level = frame["discharge_care_level"].fillna("unknown")
    rows = [
        ("Event discharges, n", str(total)),
        ("Age, mean (SD), years", _mean_sd(frame["age_at_admission"])),
        ("Female, n (%)", _n_percentage(frame["sex"].eq("female"), total)),
        ("Admission type, n (%)", ""),
        ("Medical admission", _category_count(admission_type, "medical", total)),
        (
            "Elective surgical admission",
            _category_count(admission_type, "elective_surgery", total),
        ),
        (
            "Emergency surgical admission",
            _category_count(admission_type, "emergency_surgery", total),
        ),
        (
            "Unknown admission type",
            _category_count(admission_type, "other_or_unknown", total),
        ),
        ("Admitting specialty, n (%)", ""),
        *_specialty_rows(frame, total).items(),
        ("Discharge care setting, n (%)", ""),
        ("General ward", _n_percentage(care_level.ne("high_dependency_unit"), total)),
        (
            "High-dependency care",
            _category_count(care_level, "high_dependency_unit", total),
        ),
        (
            "ICU length of stay, median (IQR), days",
            _median_iqr(frame["icu_los"], divisor=24),
        ),
        ("Life support during ICU stay, n (%)", ""),
        ("Mechanical ventilation", _n_percentage(frame["ventilation"], total)),
        (
            "Vasopressors or inotropes",
            _n_percentage(frame["vasopressors_inotropes"], total),
        ),
        (
            "Renal-replacement therapy",
            _n_percentage(frame["renal_replacement_therapy"], total),
        ),
        ("Severity scores, median (IQR)", ""),
        ("SOFA at ICU admission", _median_iqr(frame["sofa_admission"])),
        ("Maximum SOFA during ICU stay", _median_iqr(frame["sofa_max"])),
        ("SOFA at ICU discharge", _median_iqr(frame["sofa_discharge"])),
        ("NEWS at ICU discharge", _median_iqr(frame["news_discharge"])),
        ("Readmission-origin specialty, n (%)", ""),
        *_origin_rows(
            frame,
            "readmission_origin_specialty",
            origin_specialties,
            applicable=readmitted,
        ).items(),
        ("Readmission-origin care setting, n (%)*", ""),
        *_origin_rows(
            frame,
            "readmission_origin_unit_type",
            origin_settings,
            applicable=readmitted,
            available=origin_setting_available,
        ).items(),
    ]
    return pd.DataFrame(rows, columns=["Characteristic", "value"])


def _origin_categories(cohorts: Mapping[str, pd.DataFrame], column: str) -> list[str]:
    labels = set()
    for frame in cohorts.values():
        selected = frame.loc[frame["has_icu_readmission_7days"].astype(bool)]
        values = selected.get(column, pd.Series(pd.NA, index=selected.index)).dropna()
        labels.update(
            SPECIALTY_LABELS.get(
                value,
                CARE_LEVEL_LABELS.get(value, str(value).replace("_", " ").capitalize()),
            )
            for value in values
        )
    return sorted(labels or {"Unknown"})


def ward_summary(cohorts: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Summarize mutually exclusive seven-day event groups within each cohort."""
    origin_specialties = _origin_categories(cohorts, "readmission_origin_specialty")
    origin_settings = _origin_categories(cohorts, "readmission_origin_unit_type")
    columns = {}
    for cohort, frame in cohorts.items():
        readmission = frame["has_icu_readmission_7days"].astype(bool)
        death = frame["has_death_7days"].astype(bool)
        for label, has_readmission, has_death in EVENT_GROUPS:
            selected = readmission.eq(has_readmission) & death.eq(has_death)
            display_cohort = "Both Hospitals" if cohort == "Combined" else cohort
            setting_values = frame.loc[readmission, "readmission_origin_unit_type"]
            summary = _event_summary(
                frame.loc[selected],
                readmitted=has_readmission,
                origin_specialties=origin_specialties,
                origin_settings=origin_settings,
                origin_setting_available=setting_values.notna().any(),
            )
            if not columns:
                columns["Characteristic"] = summary["Characteristic"].tolist()
            elif columns["Characteristic"] != summary["Characteristic"].tolist():
                raise ValueError("event summary rows differ between outcome groups")
            columns[f"{display_cohort} {label}"] = summary["value"].tolist()
    return pd.DataFrame(columns)


SECTION_STARTS = (
    "Event discharges, n",
    "Admission type, n (%)",
    "Admitting specialty, n (%)",
    "Discharge care setting, n (%)",
    "Life support during ICU stay, n (%)",
    "Severity scores, median (IQR)",
    "Readmission-origin specialty, n (%)",
    "Readmission-origin care setting, n (%)*",
)
TOP_LEVEL_ROWS = {
    "Age, mean (SD), years",
    "Female, n (%)",
    "ICU length of stay, median (IQR), days",
}


def _row_indent_levels(summary: pd.DataFrame) -> dict[int, int]:
    """Return indentation levels from row position and the active table section."""
    levels = {}
    section = None
    for position, label in enumerate(summary["Characteristic"]):
        if label in SECTION_STARTS:
            section = label
            continue
        if label in TOP_LEVEL_ROWS:
            section = None
            continue
        if section == "Admitting specialty, n (%)":
            levels[position] = 1 if label in {"Surgical", "Medical"} else 2
        elif section is not None and section != "Event discharges, n":
            levels[position] = 1
    return levels


def ward_word_table(summary: pd.DataFrame) -> WordTable:
    """Wrap the event summary in the supplemental table presentation."""
    cohort_names = ("Both Hospitals", "VUmc", "LUMC")
    group_columns = tuple(label for label, _, _ in EVENT_GROUPS)
    return WordTable(
        TITLE,
        summary,
        header_groups=tuple(
            (cohort, tuple(f"{cohort} {label}" for label in group_columns))
            for cohort in cohort_names
        ),
        header_labels=(
            "Characteristic",
            *group_columns,
            *group_columns,
            *group_columns,
        ),
        preferred_column_widths={0: 2.14},
        landscape=True,
        section_starts=SECTION_STARTS,
        row_indent_levels_by_position=_row_indent_levels(summary),
        notes=(
            "Values are n (%), mean (SD), or median (IQR), as indicated.",
            "For cells containing one discharge, age is shown as the midpoint of its 10-year group and SD is shown as -.",
            "Outcome columns are mutually exclusive and describe enrolled ICU discharges followed within seven days by ICU readmission without death, death without ICU readmission, or both ICU readmission and death.",
            "Readmission-origin characteristics are not applicable to death-only events. Only the first ICU readmission is described.",
            "* Readmission-origin care setting was not available for LUMC.",
            "High-dependency care includes cardiac care, neurological high-care, and medium-care units.",
            "All included discharges were to wards; discharges not identified as high-dependency care were classified as general-ward discharges.",
            "Specialties describe clinical services and do not necessarily identify physical wards. Unknown values are retained in denominators.",
            "Abbreviations: ICU, intensive care unit; IQR, interquartile range; NEWS, National Early Warning Score; SD, standard deviation; SOFA, Sequential Organ Failure Assessment.",
        ),
    )
