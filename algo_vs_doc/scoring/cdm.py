"""Adapters from shared CDM-lite tables to scoring inputs.

The tables are ``measurement``, ``drug_exposure``, ``device_exposure``,
``condition_occurrence``, and ``prediction``. ``icu_stays`` is the stay table.
Hospital-specific ETL maps source fields to these tables before these adapters run.
"""

from __future__ import annotations

from typing import FrozenSet

import pandas as pd

from algo_vs_doc.scoring.concepts import CONDITION_CONCEPTS, O2_DEVICE_VALUES


def pivot_measurements(
    measurement: pd.DataFrame,
    *,
    categorical_concepts: FrozenSet[str] = frozenset({"o2_device"}),
) -> pd.DataFrame:
    """Pivot measurements by stay, timestamp, and concept.

    Categorical concepts use ``value_as_string``; other concepts use ``value_as_number``.
    Timestamp pairs remain sparse for later aggregation.
    """
    is_categorical = measurement["measurement_concept_name"].isin(categorical_concepts)
    numeric = measurement.loc[~is_categorical]
    categorical = measurement.loc[is_categorical]

    wide = numeric.pivot_table(
        index=["icu_stay_id", "measured_at"],
        columns="measurement_concept_name",
        values="value_as_number",
        aggfunc="first",
    )
    if not categorical.empty:
        wide_categorical = categorical.pivot_table(
            index=["icu_stay_id", "measured_at"],
            columns="measurement_concept_name",
            values="value_as_string",
            aggfunc="first",
        )
        wide = wide.merge(
            wide_categorical, on=["icu_stay_id", "measured_at"], how="outer"
        )

    wide = wide.reset_index()
    wide.columns.name = None
    return wide


def drug_exposure_intervals(
    drug_exposure: pd.DataFrame, drug_concept_name: str
) -> pd.DataFrame:
    """Select one drug and return ``icu_stay_id, started_at, ended_at, dose``.

    Point-in-time events must be reconstructed before calling this adapter.
    """
    rows = drug_exposure.loc[drug_exposure["drug_concept_name"] == drug_concept_name]
    return rows.rename(
        columns={
            "drug_exposure_start_datetime": "started_at",
            "drug_exposure_end_datetime": "ended_at",
            "dose_value": "dose",
        }
    )[["icu_stay_id", "started_at", "ended_at", "dose"]].reset_index(drop=True)


def condition_flags(
    condition_occurrence: pd.DataFrame,
    icu_stay_id: pd.Series,
    *,
    concepts: FrozenSet[str] = CONDITION_CONCEPTS,
) -> pd.DataFrame:
    """Return one boolean condition column per requested stay.

    Missing condition rows map to ``False``.
    """
    present = pd.crosstab(
        condition_occurrence["icu_stay_id"],
        condition_occurrence["condition_concept_name"],
    ).astype(bool)
    present = present.reindex(
        index=icu_stay_id, columns=sorted(concepts), fill_value=False
    )
    return present.reset_index().rename(columns={"index": "icu_stay_id"})


def wide_conditions_to_condition_occurrence(
    wide: pd.DataFrame, *, icu_stay_id_col: str = "icu_stay_id"
) -> pd.DataFrame:
    """Convert boolean or datetime condition columns to presence rows.

    Boolean ``True`` values produce rows with no start time. Non-null datetimes become start times.
    Other dtypes raise ``TypeError``.
    """
    concept_cols = [c for c in wide.columns if c != icu_stay_id_col]
    rows = []
    for concept in concept_cols:
        column = wide[concept]
        if pd.api.types.is_bool_dtype(column):
            present = wide.loc[column, [icu_stay_id_col]].copy()
            present["condition_start_datetime"] = pd.NaT
        elif pd.api.types.is_datetime64_any_dtype(column):
            present = wide.loc[column.notna(), [icu_stay_id_col]].copy()
            present["condition_start_datetime"] = column.loc[column.notna()]
        else:
            raise TypeError(
                f"condition column {concept!r} must be boolean or datetime, got {column.dtype}"
            )
        present["condition_concept_name"] = concept
        rows.append(present)

    if not rows:
        return pd.DataFrame(
            columns=[
                icu_stay_id_col,
                "condition_concept_name",
                "condition_start_datetime",
            ]
        )
    return pd.concat(rows, ignore_index=True)[
        [icu_stay_id_col, "condition_concept_name", "condition_start_datetime"]
    ]


CONDITION_LABELS = {
    "cardiovascular_insufficiency": "Cardiovascular insufficiency",
    "chronic_dialysis": "Chronic dialysis",
    "chronic_renal_insufficiency": "Chronic renal insufficiency",
    "cirrhosis": "Cirrhosis",
    "confirmed_infection": "Confirmed infection",
    "copd": "COPD",
    "cpr": "CPR",
    "diabetes_mellitus": "Diabetes mellitus",
    "hematological_malignancy": "Hematological malignancy",
    "metastatic_neoplasm": "Metastatic neoplasm",
}


def diagnosis_windows(
    diagnosis_occurrence: pd.DataFrame,
    icu_stays: pd.DataFrame,
    *,
    condition_concepts: FrozenSet[str] = CONDITION_CONCEPTS,
) -> pd.DataFrame:
    """Build ordered diagnosis timelines for the index and readmission episodes."""
    occurrence_columns = {
        "patient_id",
        "diagnosis_source_code",
        "diagnosis_source_label",
        "diagnosed_at",
    }
    missing = sorted(occurrence_columns - set(diagnosis_occurrence.columns))
    if missing:
        raise KeyError(f"diagnosis_occurrence is missing {missing}")
    stay_columns = {
        "icu_stay_id",
        "patient_id",
        "icu_admitted_at",
        "icu_discharged_at",
        "icu_readmission_time",
        "has_icu_readmission_7days",
    }
    missing_stays = sorted(stay_columns - set(icu_stays.columns))
    if missing_stays:
        raise KeyError(f"icu_stays is missing {missing_stays}")
    missing_conditions = sorted(set(condition_concepts) - set(icu_stays.columns))
    if missing_conditions:
        raise KeyError(f"icu_stays is missing condition flags {missing_conditions}")

    stays = icu_stays[[*stay_columns, *sorted(condition_concepts)]].copy()
    for column in ("icu_admitted_at", "icu_discharged_at", "icu_readmission_time"):
        stays[column] = pd.to_datetime(stays[column])
    occurrences = diagnosis_occurrence.copy()
    occurrences["diagnosed_at"] = pd.to_datetime(occurrences["diagnosed_at"])
    joined = (
        stays[list(stay_columns)]
        .merge(occurrences, on="patient_id", how="left", validate="many_to_many")
        .drop_duplicates(["icu_stay_id", "diagnosis_source_code", "diagnosed_at"])
    )

    index_episode = joined.diagnosed_at.ge(
        joined.icu_admitted_at - pd.Timedelta(days=7)
    ) & joined.diagnosed_at.le(joined.icu_discharged_at)
    readmission_episode = (
        joined.has_icu_readmission_7days.fillna(False).astype(bool)
        & joined.icu_readmission_time.notna()
        & joined.diagnosed_at.gt(joined.icu_discharged_at)
        & joined.diagnosed_at.le(joined.icu_discharged_at + pd.Timedelta(days=7))
    )

    target_conditions = stays[
        ["patient_id", "icu_admitted_at", *sorted(condition_concepts)]
    ].copy()
    if target_conditions.duplicated(["patient_id", "icu_admitted_at"]).any():
        raise ValueError("ICU stays must be unique by patient and admission time")
    target_conditions = target_conditions.rename(
        columns={
            "icu_admitted_at": "icu_readmission_time",
            **{concept: f"readmission_{concept}" for concept in condition_concepts},
        }
    )
    stays = stays.merge(
        target_conditions,
        on=["patient_id", "icu_readmission_time"],
        how="left",
        validate="many_to_one",
    )

    def summarize(
        mask: pd.Series, prefix: str, reference: str, *, readmission: bool
    ) -> pd.DataFrame:
        rows = []
        selected = joined.loc[mask].sort_values(
            ["icu_stay_id", "diagnosed_at", "diagnosis_source_code"]
        )
        grouped = {
            key: group for key, group in selected.groupby("icu_stay_id", sort=False)
        }
        for _, stay in stays.iterrows():
            if readmission and not bool(stay["has_icu_readmission_7days"]):
                continue
            stay_id = stay["icu_stay_id"]
            group = grouped.get(stay_id, selected.iloc[0:0])
            events = []
            for _, diagnosis in group.iterrows():
                offset = (
                    diagnosis["diagnosed_at"] - diagnosis[reference]
                ).total_seconds() / 3600
                label = diagnosis["diagnosis_source_label"]
                if pd.isna(label):
                    label = str(diagnosis["diagnosis_source_code"])
                events.append((offset, 1, str(label)))
            for concept in sorted(condition_concepts):
                column = f"readmission_{concept}" if readmission else concept
                if pd.notna(stay[column]) and bool(stay[column]):
                    label = CONDITION_LABELS.get(
                        concept, concept.replace("_", " ").capitalize()
                    )
                    events.append((0.0, 0, f"{label} (admission condition)"))
            events.sort(key=lambda event: (event[0], event[1], event[2]))
            timeline = "\n".join(
                f"{offset:+.1f} h: {label}" for offset, _, label in events
            )
            rows.append(
                {
                    "icu_stay_id": stay_id,
                    f"{prefix}_count": len(group),
                    f"{prefix}_timeline": timeline,
                }
            )
        return pd.DataFrame(
            rows, columns=["icu_stay_id", f"{prefix}_count", f"{prefix}_timeline"]
        )

    windows = (
        (index_episode, "index_episode_diagnoses", "icu_admitted_at", False),
        (
            readmission_episode,
            "readmission_episode_diagnoses",
            "icu_readmission_time",
            True,
        ),
    )
    result = stays[["icu_stay_id"]]
    for mask, prefix, reference, readmission in windows:
        result = result.merge(
            summarize(mask, prefix, reference, readmission=readmission),
            on="icu_stay_id",
            how="left",
        )
        count_column = f"{prefix}_count"
        result[count_column] = (
            pd.to_numeric(result[count_column], errors="coerce")
            .fillna(0)
            .astype("int64")
        )
    return result


def estimate_device_fio2_inputs_valid(o2_device: pd.Series) -> pd.Series:
    """Return whether each oxygen-device value supports device-and-flow FiO2 estimation."""
    return o2_device.isin(O2_DEVICE_VALUES)


#: column order of the ``prediction`` table.
PREDICTION_COLUMNS = [
    "icu_stay_id",
    "predictor",
    "prediction_timing",
    "predicted_at",
    "prediction_value",
    "prediction_category",
    "confidence",
    "factors",
    "other_factors",
    "features",
]


def predictions_wide(prediction: pd.DataFrame) -> pd.DataFrame:
    """Reshape predictions to one row per stay with one column per predictor and timing.

    Missing prediction types remain null so enrolment and group comparisons can use the same rows.
    """
    physician = prediction.loc[prediction["predictor"] == "physician"]
    wide = physician[
        [
            "icu_stay_id",
            "predicted_at",
            "prediction_value",
            "prediction_category",
            "confidence",
            "factors",
            "other_factors",
        ]
    ].rename(
        columns={
            "predicted_at": "doc_predicted_at",
            "prediction_value": "doc_prediction",
            "prediction_category": "doc_prediction_cat",
            "factors": "doc_factors",
            "other_factors": "doc_other_factors",
        }
    )

    stays = prediction[["icu_stay_id"]].drop_duplicates()
    wide = stays.merge(wide, on="icu_stay_id", how="left")

    for timing, suffix in (
        ("at_doc_prediction", "at_doc_prediction"),
        ("at_discharge", "at_discharge"),
    ):
        model = prediction.loc[
            (prediction["predictor"] == "model")
            & (prediction["prediction_timing"] == timing),
            ["icu_stay_id", "predicted_at", "prediction_value", "features"],
        ].rename(
            columns={
                "predicted_at": f"algo_predicted_{suffix}",
                "prediction_value": f"algo_prediction_{suffix}",
                "features": f"features_{suffix}",
            }
        )
        wide = wide.merge(model, on="icu_stay_id", how="left")

    return wide.reset_index(drop=True)
