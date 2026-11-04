"""Cluster-aware inference for prediction availability and operational pressure."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.linalg import qr

from algo_vs_doc import local_time, subgroup_descriptives

PRESSURE_VARIABLES = (
    "n_census_at_discharge",
    "n_external_transfers_same_day",
    "n_elective_surgery_admissions_same_day",
    "discharge_after_18h",
    "discharge_is_weekend",
)

CASE_MIX_VARIABLES = (
    "age_at_admission",
    "sofa_discharge",
)

ADMISSION_TYPE_COLUMNS = ("is_medical", "emergency_surgery", "elective_surgery")

BOOLEAN_VARIABLES = frozenset(
    {
        "discharge_after_18h",
        "discharge_is_weekend",
    }
)


def _cluster_ids(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    patient = frame["hospital"].astype(str) + "::" + frame["patient_id"].astype(str)
    discharge_date = local_time.amsterdam(frame["icu_discharged_at"]).dt.date
    hospital_date = frame["hospital"].astype(str) + "::" + discharge_date.astype(str)
    return patient, hospital_date


def prediction_availability_model(
    frame: pd.DataFrame,
    *,
    prediction_column: str,
    pressure_variables: Sequence[str] = PRESSURE_VARIABLES,
    case_mix_variables: Sequence[str] = CASE_MIX_VARIABLES,
) -> pd.DataFrame:
    """Model prediction availability with patient- and hospital-date-clustered inference."""
    required = {
        "hospital",
        "patient_id",
        "icu_discharged_at",
        prediction_column,
        *pressure_variables,
        *case_mix_variables,
        *ADMISSION_TYPE_COLUMNS,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"clustered availability model is missing {missing}")

    columns = [*pressure_variables, *case_mix_variables]
    working = frame[
        [
            "hospital",
            "patient_id",
            "icu_discharged_at",
            *columns,
            *ADMISSION_TYPE_COLUMNS,
        ]
    ].copy()
    working["available"] = frame[prediction_column].notna().astype(int)
    for column in columns:
        if column in BOOLEAN_VARIABLES:
            working[column] = working[column].astype("boolean").astype(float)
        else:
            working[column] = pd.to_numeric(working[column], errors="coerce")
    working["patient_cluster"], working["date_cluster"] = _cluster_ids(frame)
    working["admission_type"] = subgroup_descriptives.admission_type(working)
    working = working.dropna(
        subset=["available", "patient_cluster", "date_cluster", *columns]
    )
    if working["available"].nunique() < 2:
        raise ValueError(
            "prediction availability model requires available and unavailable rows"
        )
    if (
        working["patient_cluster"].nunique() < 2
        or working["date_cluster"].nunique() < 2
    ):
        raise ValueError(
            "prediction availability model requires at least two clusters per dimension"
        )

    design = working[columns].copy()
    admission_type = pd.Categorical(
        working["admission_type"],
        categories=[
            "other_or_unknown",
            "medical",
            "emergency_surgery",
            "elective_surgery",
        ],
    )
    admission_dummies = pd.get_dummies(
        admission_type, prefix="admission_type", drop_first=True, dtype=float
    ).set_axis(working.index)
    hospital_dummies = pd.get_dummies(
        working["hospital"], prefix="hospital", drop_first=True, dtype=float
    )
    design = sm.add_constant(
        pd.concat([design, admission_dummies, hospital_dummies], axis=1)
    )
    design = design.loc[
        :, design.nunique(dropna=False).gt(1) | (design.columns == "const")
    ]
    _, matrix, pivots = qr(design.to_numpy(), mode="economic", pivoting=True)
    rank = int(np.linalg.matrix_rank(matrix))
    retained = design.columns[np.sort(pivots[:rank])]
    design = design.loc[:, retained]
    fitted = sm.GLM(working["available"], design, family=sm.families.Binomial()).fit(
        cov_type="cluster",
        cov_kwds={
            "groups": np.column_stack(
                [
                    pd.factorize(working["patient_cluster"])[0],
                    pd.factorize(working["date_cluster"])[0],
                ]
            ),
            "use_correction": True,
        },
    )

    standard_error = pd.Series(fitted.bse, index=design.columns)
    coefficients = pd.Series(fitted.params, index=design.columns)
    result = pd.DataFrame(
        {
            "variable": design.columns,
            "odds_ratio": np.exp(coefficients),
            "confidence_interval_lower": np.exp(coefficients - 1.96 * standard_error),
            "confidence_interval_upper": np.exp(coefficients + 1.96 * standard_error),
        }
    ).reset_index(drop=True)
    result["n"] = len(working)
    result["available_n"] = int(working["available"].sum())
    result["unavailable_n"] = int((1 - working["available"]).sum())
    result["parameters"] = design.shape[1]
    result["patient_clusters"] = working["patient_cluster"].nunique()
    result["hospital_date_clusters"] = working["date_cluster"].nunique()
    return result
