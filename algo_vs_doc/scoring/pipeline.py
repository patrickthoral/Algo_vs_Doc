"""Run hospital-agnostic severity enrichment from shared intermediate tables.

Hospital-specific ETL maps source identifiers to CDM-lite concepts before this pipeline runs.
"""

from __future__ import annotations

import pandas as pd

from algo_vs_doc.scoring import cdm, common
from algo_vs_doc.scoring.concepts import SEDATIVE_DRUG_CONCEPT
from algo_vs_doc.scoring.news import assemble as news_assemble
from algo_vs_doc.scoring.sofa import (
    aggregate,
    cardiovascular,
    coagulation,
    kidneys,
    liver,
    neurology,
    respiratory,
)

#: Per-stay severity columns in reporting order.
SEVERITY_COLUMNS = ["sofa_admission", "sofa_max", "sofa_discharge", "news_discharge"]


def _concept(measurement: pd.DataFrame, concept: str) -> pd.DataFrame:
    """Select numeric measurements for one concept."""
    return measurement.loc[
        measurement["measurement_concept_name"] == concept,
        ["icu_stay_id", "measured_at", "value_as_number"],
    ]


def _score_simple_lab(
    measurement: pd.DataFrame,
    icu_stays: pd.DataFrame,
    concept: str,
    scorer,
    score_column: str,
) -> pd.DataFrame:
    """Score a SOFA component derived from one laboratory value."""
    wide = cdm.pivot_measurements(
        measurement.loc[measurement["measurement_concept_name"] == concept]
    )
    if wide.empty or concept not in wide.columns:
        return pd.DataFrame(
            columns=["icu_stay_id", "measured_at", "day_index", score_column]
        )
    scored = common.add_day_index(wide, icu_stays)
    scored[score_column] = scorer(scored[concept])
    return scored


def score_sofa_components(
    icu_stays: pd.DataFrame,
    measurement: pd.DataFrame,
    drug_exposure: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Score all six SOFA components from shared intermediate tables.

    Elective-surgery stays receive a normal day-0 neurology score. Results contain the score,
    ``icu_stay_id``, ``measured_at``, and ``day_index`` where applicable. Cardiovascular scores are
    reduced separately because MAP and drug doses have independent timelines.
    """
    components: dict[str, pd.DataFrame] = {}

    # Score PaO2 with recorded or device-estimated FiO2.
    resp = respiratory.score_readings(measurement)
    if not resp.empty:
        components["sofa_resp"] = common.add_day_index(resp, icu_stays)

    # Score GCS after sedation and ventilation handling.
    gcs = cdm.pivot_measurements(
        measurement.loc[
            measurement["measurement_concept_name"].isin(
                ["gcs_eye", "gcs_motor", "gcs_verbal"]
            )
        ]
    )
    if not gcs.empty:
        sedatives = drug_exposure.loc[
            drug_exposure["drug_concept_name"] == SEDATIVE_DRUG_CONCEPT,
            ["icu_stay_id", "drug_exposure_start_datetime", "dose_value"],
        ].rename(
            columns={
                "drug_exposure_start_datetime": "administered_at",
                "dose_value": "infusion_rate",
            }
        )
        neuro = neurology.flag_sedated(gcs, sedatives)
        neuro = neuro.loc[~neuro["sedated"]].drop(columns="sedated")
        neuro = neuro.loc[
            neuro[["gcs_eye", "gcs_motor", "gcs_verbal"]].notna().all(axis=1)
        ].copy()
        neuro["ventilated"] = neurology.flag_ventilated(
            neuro, _concept(measurement, "peep")
        )
        neuro["gcs_verbal"] = neurology.backfill_ventilated_verbal(
            neuro["gcs_eye"],
            neuro["gcs_motor"],
            neuro["gcs_verbal"],
            neuro["ventilated"],
            neuro["icu_stay_id"],
            neuro["measured_at"],
        )
        neuro["gcs_total"] = neuro["gcs_eye"] + neuro["gcs_motor"] + neuro["gcs_verbal"]
        neuro["sofa_neuro"] = neurology.score(neuro["gcs_total"])
        neuro = common.add_day_index(neuro, icu_stays)
        neuro = neurology.apply_elective_surgery_day0(neuro, icu_stays)
        components["sofa_neuro"] = neuro

    # Score MAP and vasoactive doses on independent timelines.
    cardvasc = cardiovascular.score(_concept(measurement, "map"), drug_exposure)
    if not cardvasc.empty:
        components["sofa_cardvasc"] = common.reduce_daily_max(
            common.add_day_index(cardvasc, icu_stays), score_col="sofa_cardvasc"
        )

    # Score liver and coagulation from their laboratory values.
    for concept, scorer, column in (
        ("bilirubin", liver.score, "sofa_liver"),
        ("platelets", coagulation.score, "sofa_coagulation"),
    ):
        scored = _score_simple_lab(measurement, icu_stays, concept, scorer, column)
        if not scored.empty:
            components[column] = scored

    # Score kidneys from creatinine and extrapolated daily urine output.
    kidney = kidneys.score_readings(
        icu_stays,
        _concept(measurement, "creatinine"),
        _concept(measurement, "urine_output"),
    )
    if not kidney.empty:
        components["sofa_kidneys"] = common.add_day_index(kidney, icu_stays)

    return components


def severity_summary(
    icu_stays: pd.DataFrame,
    measurement: pd.DataFrame,
    drug_exposure: pd.DataFrame,
) -> pd.DataFrame:
    """Return SOFA admission, maximum, and discharge scores plus NEWS at discharge.

    Every input stay remains in the result; absent component data contributes zero.
    """
    components = score_sofa_components(icu_stays, measurement, drug_exposure)

    daily = aggregate.daily_scores(components)
    summary = icu_stays[["icu_stay_id"]].drop_duplicates()
    summary = summary.merge(aggregate.at_admission(daily), on="icu_stay_id", how="left")
    summary = summary.merge(aggregate.maximum(daily), on="icu_stay_id", how="left")

    # Cardiovascular values use independent MAP and drug timelines.
    at_discharge_inputs = {k: v for k, v in components.items() if k != "sofa_cardvasc"}
    cardvasc_scored = cardiovascular.score(_concept(measurement, "map"), drug_exposure)
    if not cardvasc_scored.empty:
        at_discharge_inputs["sofa_cardvasc"] = cardiovascular.at_discharge(
            cardvasc_scored, icu_stays
        )
    sofa_discharge = aggregate.at_discharge(at_discharge_inputs, icu_stays)
    summary = summary.merge(
        sofa_discharge[["icu_stay_id", "sofa_discharge"]], on="icu_stay_id", how="left"
    )

    news_components = news_assemble.score_components(measurement)
    if "sofa_neuro" in components:
        news_components["news_avpu"] = news_assemble.score_avpu_from_gcs(
            components["sofa_neuro"]
        )
    news_discharge = news_assemble.at_discharge(news_components, icu_stays)
    summary = summary.merge(
        news_discharge[["icu_stay_id", "news_discharge"]], on="icu_stay_id", how="left"
    )

    # Convert empty merge columns before filling to avoid object downcasting.
    present = [c for c in SEVERITY_COLUMNS if c in summary.columns]
    summary[present] = summary[present].apply(pd.to_numeric, errors="coerce").fillna(0)
    return summary.reset_index(drop=True)
