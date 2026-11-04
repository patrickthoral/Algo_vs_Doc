"""Baseline and context descriptives for publication tables."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from algo_vs_doc import specialties, subgroup_descriptives
from algo_vs_doc.specialties import SPECIALTY_LABELS


def _n_percentage(values: pd.Series, *, denominator: int | None = None) -> str:
    """Format the number of true values and its percentage."""
    values = values.astype("boolean")
    if denominator is None:
        denominator = int(values.notna().sum())
    count = int(values.fillna(False).sum())
    percentage = 100 * count / denominator if denominator else 0.0
    return f"{count} ({percentage:.1f}%)"


def _mean_sd(values: pd.Series) -> str:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return "-"
    if len(values) == 1:
        age_group_midpoint = int(values.iloc[0] // 10 * 10 + 5)
        return f"{age_group_midpoint} (-)"
    return f"{values.mean():.1f} ({values.std():.1f})"


def _median_iqr(values: pd.Series) -> str:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return "NA"
    lower, median, upper = values.quantile([0.25, 0.5, 0.75])
    return f"{median:.1f} ({lower:.1f}-{upper:.1f})"


def _patient_count(frame: pd.DataFrame) -> int:
    """Count patients without conflating site-local identifiers."""
    columns = ["patient_id"]
    if "hospital" in frame:
        columns.insert(0, "hospital")
    identifiers = frame.loc[:, columns]
    if identifiers.isna().any().any():
        raise ValueError("patient identifiers must be observed for patient counts")
    return len(identifiers.drop_duplicates())


def extended_context_summary(frame: pd.DataFrame) -> pd.Series:
    """Return the prespecified baseline and discharge-context summary for one cohort."""
    required = {
        "patient_id",
        "age_at_admission",
        "sex",
        "icu_los",
        "ventilation",
        "vasopressors_inotropes",
        "renal_replacement_therapy",
        "sofa_admission",
        "sofa_max",
        "sofa_discharge",
        "news_discharge",
        "ready_for_discharge",
        "discharge_is_weekend",
        "discharge_after_18h",
        "n_census_at_discharge",
        "n_external_transfers_same_day",
        "n_elective_surgery_admissions_same_day",
        "index_episode_diagnoses_count",
        "has_icu_readmission_7days",
        "has_death_7days",
        "outcome",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"baseline summary is missing {missing}")

    total = len(frame)
    ready = frame["ready_for_discharge"].astype("boolean")
    return pd.Series(
        {
            "ICU discharges, n": str(total),
            "Patients, n": str(_patient_count(frame)),
            "Age, mean (SD), years": _mean_sd(frame["age_at_admission"]),
            "Female, n (%)": _n_percentage(
                frame["sex"].eq("female"), denominator=total
            ),
            "ICU length of stay, median (IQR), hours": _median_iqr(frame["icu_los"]),
            "Life support during ICU stay, n (%)": "",
            "  Ventilation": _n_percentage(frame["ventilation"], denominator=total),
            "  Vasopressors or inotropes": _n_percentage(
                frame["vasopressors_inotropes"], denominator=total
            ),
            "  Renal-replacement therapy": _n_percentage(
                frame["renal_replacement_therapy"], denominator=total
            ),
            "Severity scores, median (IQR)": "",
            "  SOFA at ICU admission": _median_iqr(frame["sofa_admission"]),
            "  Maximum SOFA during ICU stay": _median_iqr(frame["sofa_max"]),
            "  SOFA at ICU discharge": _median_iqr(frame["sofa_discharge"]),
            "  NEWS at ICU discharge": _median_iqr(frame["news_discharge"]),
            "Readiness assessment available, n (%)": (
                f"{int(ready.notna().sum())} "
                f"({100 * ready.notna().sum() / total if total else 0:.1f}%)"
            ),
            "Ready for discharge, n/N observed (%)": (
                f"{int(ready.fillna(False).sum())}/{int(ready.notna().sum())} "
                f"({100 * ready.mean() if ready.notna().any() else 0:.1f}%)"
            ),
            "Weekend discharge, n (%)": _n_percentage(
                frame["discharge_is_weekend"], denominator=total
            ),
            "Discharge after 18:00, n (%)": _n_percentage(
                frame["discharge_after_18h"], denominator=total
            ),
            "ICU census at discharge, median (IQR)": _median_iqr(
                frame["n_census_at_discharge"]
            ),
            "Other same-day external transfers, median (IQR)": _median_iqr(
                frame["n_external_transfers_same_day"]
            ),
            "Same-day elective surgery ICU admissions, median (IQR)": _median_iqr(
                frame["n_elective_surgery_admissions_same_day"]
            ),
            "Diagnoses documented from 7 days before through index ICU discharge, median (IQR)": _median_iqr(
                frame["index_episode_diagnoses_count"]
            ),
            "Composite outcome within 7 days, n (%)": _n_percentage(
                frame["outcome"], denominator=total
            ),
            "  ICU readmission within 7 days, n (%)": _n_percentage(
                frame["has_icu_readmission_7days"], denominator=total
            ),
            "  Death within 7 days, n (%)": _n_percentage(
                frame["has_death_7days"], denominator=total
            ),
        }
    )


def _table_1_count_percentage(count: int, denominator: int) -> str:
    percentage = 100 * count / denominator if denominator else 0.0
    return f"{count} ({percentage:.1f})"


def _table_1_n_percentage(values: pd.Series, denominator: int) -> str:
    count = int(values.astype("boolean").fillna(False).sum())
    return _table_1_count_percentage(count, denominator)


def _table_1_age(values: pd.Series) -> str:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return "-"
    if len(values) == 1:
        midpoint = int(values.iloc[0] // 10 * 10 + 5)
        return f"{midpoint} (-)"
    return f"{values.mean():.0f} ({values.std():.0f})"


def baseline_characteristics(frame: pd.DataFrame) -> pd.Series:
    """Return the publication Table 1 summary for one cohort."""
    required = {
        "patient_id",
        "age_at_admission",
        "sex",
        "specialty",
        "discharge_care_level",
        "icu_los",
        "is_medical",
        "elective_surgery",
        "emergency_surgery",
        "ventilation",
        "vasopressors_inotropes",
        "renal_replacement_therapy",
        "sofa_admission",
        "sofa_max",
        "sofa_discharge",
        "news_discharge",
        "has_icu_readmission_7days",
        "has_death_7days",
        "outcome",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"Table 1 baseline characteristics are missing {missing}")

    total = len(frame)
    total_patients = _patient_count(frame)
    event_mask = frame["outcome"].astype("boolean").fillna(False)
    event_patients = _patient_count(frame.loc[event_mask])
    admission_type = subgroup_descriptives.admission_type(frame)
    specialty = frame["specialty"].fillna("unknown")
    discharge_care_level = frame["discharge_care_level"].fillna("unknown")
    rows = {
        "ICU discharges, n": str(total),
        "Patients, n": str(total_patients),
        "Age, mean (SD)": _table_1_age(frame["age_at_admission"]),
        "Female, n (%)": _table_1_n_percentage(frame["sex"].eq("female"), total),
        "Admission type, n (%)": "",
        "Medical admission": _table_1_n_percentage(admission_type.eq("medical"), total),
        "Elective surgical admission": _table_1_n_percentage(
            admission_type.eq("elective_surgery"), total
        ),
        "Emergency surgical admission": _table_1_n_percentage(
            admission_type.eq("emergency_surgery"), total
        ),
        "Unknown": _table_1_n_percentage(admission_type.eq("other_or_unknown"), total),
        "Admitting specialty, n (%)": "",
        "Surgical": _table_1_n_percentage(
            specialty.isin(specialties.SURGICAL_SPECIALTIES), total
        ),
    }
    rows.update(
        {
            SPECIALTY_LABELS[category]: _table_1_n_percentage(
                specialty.eq(category), total
            )
            for category in specialties.SURGICAL_SPECIALTIES
        }
    )
    rows["Medical"] = _table_1_n_percentage(
        specialty.isin(specialties.MEDICAL_SPECIALTIES), total
    )
    rows.update(
        {
            SPECIALTY_LABELS[category]: _table_1_n_percentage(
                specialty.eq(category), total
            )
            for category in specialties.MEDICAL_SPECIALTIES
        }
    )
    rows.update(
        {
            "Discharge care setting, n (%)": "",
            "General ward": _table_1_n_percentage(
                discharge_care_level.ne("high_dependency_unit"), total
            ),
            "High-dependency care": _table_1_n_percentage(
                discharge_care_level.eq("high_dependency_unit"), total
            ),
            "Length of Stay, days, median (IQR)": _median_iqr(frame["icu_los"] / 24),
            "Life support during ICU stay, n (%)": "",
            "Ventilation": _table_1_n_percentage(frame["ventilation"], total),
            "Vasopressors or inotropes": _table_1_n_percentage(
                frame["vasopressors_inotropes"], total
            ),
            "Renal-replacement therapy": _table_1_n_percentage(
                frame["renal_replacement_therapy"], total
            ),
            "Severity scores, median (IQR)": "",
            "SOFA at ICU admission": _median_iqr(frame["sofa_admission"]),
            "Maximum SOFA during ICU stay": _median_iqr(frame["sofa_max"]),
            "SOFA at ICU discharge": _median_iqr(frame["sofa_discharge"]),
            "NEWS at ICU discharge": _median_iqr(frame["news_discharge"]),
            "Outcome within 7 days, n (%)": "",
            "Composite outcome, ICU discharges": _table_1_n_percentage(
                frame["outcome"], total
            ),
            "Composite outcome, patients": _table_1_count_percentage(
                event_patients, total_patients
            ),
            "ICU readmission": _table_1_n_percentage(
                frame["has_icu_readmission_7days"], total
            ),
            "Death": _table_1_n_percentage(frame["has_death_7days"], total),
            "ICU readmission and death": _table_1_n_percentage(
                frame["has_icu_readmission_7days"].astype("boolean")
                & frame["has_death_7days"].astype("boolean"),
                total,
            ),
        }
    )
    return pd.Series(rows)


def baseline_characteristics_table(
    cohorts: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    """Build publication Table 1 with one column per cohort."""
    return pd.DataFrame(
        {name: baseline_characteristics(frame) for name, frame in cohorts.items()}
    )


def specialty_counts(frame: pd.DataFrame) -> pd.DataFrame:
    """Return admitting-specialty counts, preserving missing values as unknown."""
    if "specialty" not in frame:
        raise KeyError("specialty counts require 'specialty'")
    counts = frame["specialty"].fillna("unknown").value_counts(sort=False)
    total = len(frame)
    return pd.DataFrame(
        {
            "specialty": counts.index,
            "n": counts.to_numpy(dtype="int64"),
            "percentage": 100 * counts.to_numpy() / total if total else 0.0,
        }
    ).sort_values(["n", "specialty"], ascending=[False, True], ignore_index=True)


def specialty_counts_by_cohort(
    cohorts: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    """Stack admitting-specialty counts for named cohorts."""
    return pd.concat(
        [
            specialty_counts(frame).assign(cohort=name)
            for name, frame in cohorts.items()
        ],
        ignore_index=True,
    )[["cohort", "specialty", "n", "percentage"]]


def outcome_group_summaries(
    cohorts: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    """Stack baseline summaries for event and non-event groups within each cohort."""
    rows = []
    for cohort, frame in cohorts.items():
        for outcome, label in ((False, "No event"), (True, "Event")):
            selected = frame["outcome"].astype(bool).eq(outcome)
            summary = extended_context_summary(frame.loc[selected])
            rows.append(
                summary.rename("value")
                .rename_axis("characteristic")
                .reset_index()
                .assign(cohort=cohort, outcome_group=label)
            )
    return pd.concat(rows, ignore_index=True)[
        ["cohort", "outcome_group", "characteristic", "value"]
    ]
