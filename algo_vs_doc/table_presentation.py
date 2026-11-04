"""Convert analysis outputs into publication-facing table frames."""

from __future__ import annotations

import pandas as pd

from algo_vs_doc.word_tables import RichCell


PREDICTION_LABELS = {
    "Combined (Percentage)": "Combined Model and Physician (Percentage)",
    "Combined (Category)": "Combined Model and Physician (Category)",
    "Model at Physician Prediction": "Model",
    "Model at physician prediction": "Model",
    "Model at Patient Discharge": "Model at discharge",
    "Model at discharge": "Model at discharge",
}

PREDICTION_ORDER = (
    "Physician",
    "Physician (Percentage)",
    "Physician (Category)",
    "Model",
    "Combined Model and Physician (Percentage)",
    "Combined Model and Physician (Category)",
)

EVENT_LABELS = {
    "no_event": "No event",
    "icu_readmission": "ICU readmission",
    "death": "Death",
    "readmission_and_death": "ICU readmission and death",
}

FAMILY_LABELS = {
    "primary": "Primary",
    "secondary": "Secondary",
    "exploratory": "Exploratory",
}
ANALYSIS_ORDER = ("Primary", "Secondary", "Primary/secondary", "Exploratory")
METRIC_ORDER = ("AUROC", "AUPRC")
COHORT_ORDER = ("Both Hospitals", "VUmc", "LUMC")


def _sort(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = frame.copy()
    if "Cohort" in result:
        result["Cohort"] = pd.Categorical(
            result["Cohort"].astype(object).replace({"Combined": "Both Hospitals"}),
            categories=COHORT_ORDER,
            ordered=True,
        )
    if "Analysis" in result:
        result["Analysis"] = pd.Categorical(
            result["Analysis"], categories=ANALYSIS_ORDER, ordered=True
        )
    if "Metric" in result:
        result["Metric"] = pd.Categorical(
            result["Metric"], categories=METRIC_ORDER, ordered=True
        )
    for column in ("Prediction source", "Agent 1", "Agent 2"):
        if column in result:
            result[column] = pd.Categorical(
                result[column], categories=PREDICTION_ORDER, ordered=True
            )
    return result.sort_values(columns, kind="stable").reset_index(drop=True)


def _publication_predictions(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """Apply publication labels and omit the discharge-time model analysis."""
    result = frame.copy()
    result[column] = result[column].replace(PREDICTION_LABELS)
    return result.loc[result[column].ne("Model at discharge")].reset_index(drop=True)


def _number(value, digits: int = 2) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.{digits}f}"


def _integer(value) -> str:
    return "" if pd.isna(value) else str(int(value))


def _p_value(value) -> str | None:
    if pd.isna(value):
        return None
    return "<0.001" if float(value) < 0.001 else f"{float(value):.3f}"


def estimate_interval(
    estimate,
    lower,
    upper,
    *,
    p_value=None,
    digits: int = 2,
) -> RichCell:
    """Format an estimate, optional 95% interval, and optional p-value."""
    estimate_text = _number(estimate, digits)
    if not estimate_text:
        return RichCell("")
    if pd.notna(lower) and pd.notna(upper):
        estimate_text += f" [{_number(lower, digits)}, {_number(upper, digits)}]"
    return RichCell(estimate_text, _p_value(p_value))


def _prediction_source(frame: pd.DataFrame) -> pd.Series:
    candidates = [column for column in frame if str(column).startswith("Unnamed:")]
    if not candidates:
        raise KeyError("performance table has no prediction-source column")
    return frame[candidates[0]].replace(PREDICTION_LABELS)


def _performance(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["_prediction_source"] = _prediction_source(frame)
    frame = _publication_predictions(frame, "_prediction_source")
    result = pd.DataFrame(
        {
            "Cohort": frame["cohort"],
            "Prediction source": frame["_prediction_source"],
            "n": frame["n"].map(_integer),
        }
    )
    specifications = (
        (
            "AUROC\n(95% CI)",
            "auroc",
            "auroc_confidence_interval_lower",
            "auroc_confidence_interval_upper",
        ),
        (
            "AUPRC\n(95% CI)",
            "auprc",
            "auprc_confidence_interval_lower",
            "auprc_confidence_interval_upper",
        ),
        (
            "Calibration intercept\n(95% CI)",
            "calibration_in_the_large",
            "calibration_in_the_large_ci_lower",
            "calibration_in_the_large_ci_upper",
        ),
        (
            "Calibration slope\n(95% CI)",
            "calibration_slope_estimate",
            "calibration_slope_estimate_ci_lower",
            "calibration_slope_estimate_ci_upper",
        ),
        (
            "Observed-to-expected ratio\n(95% CI)",
            "observed_expected_ratio",
            "observed_expected_ratio_ci_lower",
            "observed_expected_ratio_ci_upper",
        ),
        (
            "Brier score\n(95% CI)",
            "brier_score",
            "brier_score_ci_lower",
            "brier_score_ci_upper",
        ),
    )
    for label, estimate, lower, upper in specifications:
        result[label] = [
            estimate_interval(row[estimate], row[lower], row[upper])
            for _, row in frame.iterrows()
        ]
    return result


def performance(frame: pd.DataFrame) -> pd.DataFrame:
    """Present discrimination and calibration estimates without internal columns."""
    return _sort(_performance(frame), ["Cohort", "Prediction source"])


def performance_contrasts(frame: pd.DataFrame) -> pd.DataFrame:
    """Present each contrast once, with metric columns grouped by cohort."""
    frame = _publication_predictions(frame, "agent_1")
    frame = _publication_predictions(frame, "agent_2")
    frame = frame.copy()
    frame["Cohort"] = frame["cohort"].replace({"Combined": "Both Hospitals"})
    frame["Agent 1"] = frame["agent_1"]
    frame["Agent 2"] = frame["agent_2"]
    frame["Metric"] = frame["metric"].str.upper()
    frame["Result"] = [
        estimate_interval(
            row["difference_agent_1_minus_agent_2"],
            row["confidence_interval_lower"],
            row["confidence_interval_upper"],
            p_value=(
                row["p_holm"] if row["family"] in {"primary", "secondary"} else None
            ),
        )
        for _, row in frame.iterrows()
    ]

    rows = []
    for (agent_1, agent_2), comparison in frame.groupby(
        ["Agent 1", "Agent 2"], sort=False, observed=True
    ):
        confirmatory = comparison["family"].isin(["primary", "secondary"]).any()
        row = {
            "Analysis": "Primary/secondary" if confirmatory else "Exploratory",
            "Agent 1": agent_1,
            "Agent 2": agent_2,
        }
        for cohort in COHORT_ORDER:
            cohort_rows = comparison.loc[comparison["Cohort"].eq(cohort)].set_index(
                "Metric"
            )
            for metric in METRIC_ORDER:
                value = (
                    cohort_rows.loc[metric, "Result"]
                    if metric in cohort_rows.index
                    else RichCell("")
                )
                row[f"{cohort} {metric}"] = value
        rows.append(row)

    result = pd.DataFrame(rows)
    return _sort(result, ["Analysis", "Agent 1", "Agent 2"])


def confidence_performance(frame: pd.DataFrame) -> pd.DataFrame:
    """Present predictive performance stratified by physician confidence."""
    frame = frame.copy()
    frame["_prediction_source"] = _prediction_source(frame)
    frame = _publication_predictions(frame, "_prediction_source")
    result = (
        _performance(frame)
        .assign(Confidence=frame["confidence"].to_numpy())
        .loc[
            :,
            [
                "Cohort",
                "Confidence",
                "Prediction source",
                "n",
                "AUROC\n(95% CI)",
                "AUPRC\n(95% CI)",
                "Calibration intercept\n(95% CI)",
                "Calibration slope\n(95% CI)",
                "Observed-to-expected ratio\n(95% CI)",
                "Brier score\n(95% CI)",
            ],
        ]
    )
    return _sort(result, ["Cohort", "Confidence", "Prediction source"])


def factor_categories(frame: pd.DataFrame) -> pd.DataFrame:
    """Present one feature-category row with source columns grouped by cohort."""
    frame = frame.copy()
    frame["Cohort"] = frame["cohort"].replace({"Combined": "Both Hospitals"})
    frame["Prediction source"] = frame["prediction_source"]
    frame["Factor category"] = frame["factor_category"]
    frame["Count, n (%)"] = [
        f"{int(count)} ({float(percentage):.1f})"
        for count, percentage in zip(frame["count"], frame["percentage"])
    ]

    rows = []
    for category in sorted(frame["Factor category"].dropna().unique()):
        category_rows = frame.loc[frame["Factor category"].eq(category)]
        row = {"Factor category": category}
        for cohort in COHORT_ORDER:
            cohort_rows = category_rows.loc[category_rows["Cohort"].eq(cohort)]
            for source in ("Physician", "Model"):
                values = cohort_rows.loc[
                    cohort_rows["Prediction source"].eq(source), "Count, n (%)"
                ]
                row[f"{cohort} {source}"] = values.iloc[0] if len(values) else ""
        rows.append(row)
    return pd.DataFrame(rows)


AVAILABILITY_VARIABLE_LABELS = {
    "age_at_admission": "Age, median [IQR], years",
    "icu_los": "ICU length of stay, median [IQR], hours",
    "sofa_admission": "SOFA at ICU admission, median [IQR]",
    "sofa_max": "Maximum SOFA during ICU stay, median [IQR]",
    "sofa_discharge": "SOFA at ICU discharge, median [IQR]",
    "news_discharge": "NEWS at ICU discharge, median [IQR]",
    "discharge_local_hour": "Local hour of ICU discharge, median [IQR]",
    "new_diagnoses_count": "New diagnoses documented, median [IQR]",
    "n_census_at_discharge": "ICU census at discharge, median [IQR]",
    "n_external_transfers_same_day": "Same-day external transfers, median [IQR]",
    "n_elective_surgery_admissions_same_day": "Same-day elective surgery ICU admissions, median [IQR]",
    "ready_for_discharge": "Ready for discharge, n (%)",
    "discharge_after_18h": "Discharge after 18:00, n (%)",
    "discharge_is_weekend": "Weekend discharge, n (%)",
    "is_medical": "Medical admission, n (%)",
    "emergency_surgery": "Emergency surgical admission, n (%)",
    "elective_surgery": "Elective surgical admission, n (%)",
    "ventilation": "Mechanical ventilation, n (%)",
    "invasive_ventilation": "Invasive mechanical ventilation, n (%)",
    "renal_replacement_therapy": "Renal-replacement therapy, n (%)",
    "vasopressors_inotropes": "Vasopressors or inotropes, n (%)",
    "outcome": "Readmission or death within 7 days, n (%)",
    "admission_type_medical": "Medical admission",
    "admission_type_emergency_surgery": "Emergency surgical admission",
    "admission_type_elective_surgery": "Elective surgical admission",
}


def prediction_availability_characteristics(
    frame: pd.DataFrame, denominators: pd.DataFrame
) -> pd.DataFrame:
    """Present availability groups and their denominators beneath hospital headers."""
    required = {
        "cohort",
        "physician_prediction_available",
        "physician_prediction_missing",
    }
    missing = sorted(required - set(denominators.columns))
    if missing:
        raise KeyError(f"prediction availability denominators are missing {missing}")

    denominator_row = {"Characteristic": "ICU discharges, n (%)"}
    for cohort in ("VUmc", "LUMC"):
        cohort_denominators = denominators.loc[denominators["cohort"].eq(cohort)]
        if len(cohort_denominators) != 1:
            raise ValueError(
                f"expected one prediction availability denominator row for {cohort}"
            )
        values = cohort_denominators.iloc[0]
        available = int(values["physician_prediction_available"])
        missing = int(values["physician_prediction_missing"])
        total = available + missing
        if total == 0:
            raise ValueError(
                f"prediction availability denominator is zero for {cohort}"
            )
        denominator_row[f"{cohort} available"] = (
            f"{available} ({100 * available / total:.1f}%)"
        )
        denominator_row[f"{cohort} missing"] = (
            f"{missing} ({100 * missing / total:.1f}%)"
        )

    frame = frame.loc[frame["variable"].ne("ready_for_discharge")].copy()
    frame["Characteristic"] = frame["variable"].replace(AVAILABILITY_VARIABLE_LABELS)
    rows = [denominator_row]
    for characteristic in sorted(frame["Characteristic"].dropna().unique()):
        characteristic_rows = frame.loc[frame["Characteristic"].eq(characteristic)]
        row = {"Characteristic": characteristic}
        for cohort in ("VUmc", "LUMC"):
            cohort_rows = characteristic_rows.loc[
                characteristic_rows["cohort"].eq(cohort)
            ]
            if cohort_rows.empty:
                row[f"{cohort} available"] = ""
                row[f"{cohort} missing"] = ""
            else:
                values = cohort_rows.iloc[0]
                row[f"{cohort} available"] = (
                    ""
                    if values["prediction_available"] == "-"
                    else str(values["prediction_available"])
                )
                row[f"{cohort} missing"] = (
                    ""
                    if values["prediction_missing"] == "-"
                    else str(values["prediction_missing"])
                )
        rows.append(row)
    return pd.DataFrame(rows)


def prediction_availability_models(frame: pd.DataFrame) -> pd.DataFrame:
    """Present adjusted associations beneath hospital column headers."""
    frame = frame.loc[frame["variable"].ne("const")].copy()
    frame["Characteristic"] = frame["variable"].replace(AVAILABILITY_VARIABLE_LABELS)
    frame["Result"] = [
        estimate_interval(estimate, lower, upper)
        for estimate, lower, upper in zip(
            frame["odds_ratio"],
            frame["confidence_interval_lower"],
            frame["confidence_interval_upper"],
        )
    ]
    rows = []
    for characteristic in sorted(frame["Characteristic"].dropna().unique()):
        characteristic_rows = frame.loc[frame["Characteristic"].eq(characteristic)]
        row = {"Characteristic": characteristic}
        for cohort in ("VUmc", "LUMC"):
            values = characteristic_rows.loc[
                characteristic_rows["cohort"].eq(cohort), "Result"
            ]
            row[f"{cohort} adjusted odds ratio"] = (
                values.iloc[0] if len(values) else RichCell("")
            )
        rows.append(row)
    return pd.DataFrame(rows)


def event_types(frame: pd.DataFrame) -> pd.DataFrame:
    """Present seven-day outcome components as counts and percentages."""
    result = pd.DataFrame(
        {
            "Cohort": frame["cohort"],
            "Outcome within 7 days": frame["event_type"].replace(EVENT_LABELS),
            "n (%)": [
                f"{int(count)} ({float(percentage):.1f})"
                for count, percentage in zip(frame["n"], frame["percentage"])
            ],
        }
    )
    return _sort(result, ["Cohort"])


def fixed_selection_rates(frame: pd.DataFrame) -> pd.DataFrame:
    """Present fixed-selection operating characteristics with compact intervals."""
    frame = _publication_predictions(frame, "prediction_source")
    frame = frame.loc[frame["prediction_source"].eq("Model")].sort_values(
        ["cohort", "target_alert_rate"], kind="stable"
    )
    result = pd.DataFrame(
        {
            "Cohort": frame["cohort"],
            "Prediction source": frame["prediction_source"],
            "Selection rate": frame["target_alert_rate"].map(
                lambda value: f"{100 * value:.0f}%"
            ),
            "Patients selected": frame["flagged_n"].map(_integer)
            + "/"
            + frame["n"].map(_integer),
            "Events captured": frame["events_captured_n"].map(_integer)
            + "/"
            + frame["events_n"].map(_integer),
        }
    )
    for column, label in (
        ("sensitivity", "Sensitivity\n(95% CI)"),
        ("specificity", "Specificity\n(95% CI)"),
        ("positive_predictive_value", "Positive predictive value\n(95% CI)"),
        ("negative_predictive_value", "Negative predictive value\n(95% CI)"),
        ("f1", "F1 score\n(95% CI)"),
    ):
        result[label] = [
            estimate_interval(value, lower, upper)
            for value, lower, upper in zip(
                frame[column], frame[f"{column}_ci_lower"], frame[f"{column}_ci_upper"]
            )
        ]
    return _sort(result, ["Cohort", "Prediction source"])


def risk_thresholds(frame: pd.DataFrame) -> pd.DataFrame:
    """Present model operating characteristics at probability thresholds."""
    frame = _publication_predictions(frame, "prediction_source")
    frame = frame.loc[frame["prediction_source"].eq("Model")].copy()
    frame["_order"] = frame["operating_point"].map(
        {
            "5% risk threshold": 0,
            "10% risk threshold": 1,
            "20% risk threshold": 2,
            "Youden index": 3,
        }
    )
    frame = frame.sort_values(["cohort", "_order"], kind="stable")
    result = pd.DataFrame(
        {
            "Cohort": frame["cohort"],
            "Operating point": frame["operating_point"],
            "Risk threshold": frame["threshold"].map(
                lambda value: f"{100 * value:.1f}%"
            ),
            "Patients selected": frame["flagged_n"].map(_integer)
            + "/"
            + frame["n"].map(_integer),
            "Events captured": frame["true_positive"].map(_integer)
            + "/"
            + (frame["true_positive"] + frame["false_negative"]).map(_integer),
        }
    )
    for column, label in (
        ("sensitivity", "Sensitivity\n(95% CI)"),
        ("specificity", "Specificity\n(95% CI)"),
        ("positive_predictive_value", "Positive predictive value\n(95% CI)"),
        ("negative_predictive_value", "Negative predictive value\n(95% CI)"),
        ("f1", "F1 score\n(95% CI)"),
    ):
        result[label] = [
            estimate_interval(value, lower, upper)
            for value, lower, upper in zip(
                frame[column], frame[f"{column}_ci_lower"], frame[f"{column}_ci_upper"]
            )
        ]
    return _sort(result, ["Cohort"])
