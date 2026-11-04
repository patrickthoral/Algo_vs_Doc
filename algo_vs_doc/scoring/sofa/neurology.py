"""SOFA Neurology component: Glasgow Coma Scale (GCS)."""

from __future__ import annotations

import pandas as pd


def score(gcs_total: pd.Series) -> pd.Series:
    """Score SOFA Neurology from a GCS total."""
    sofa_neuro = pd.Series(0, index=gcs_total.index)
    sofa_neuro = sofa_neuro.mask(gcs_total <= 14, 1)
    sofa_neuro = sofa_neuro.mask(gcs_total <= 12, 2)
    sofa_neuro = sofa_neuro.mask(gcs_total <= 9, 3)
    sofa_neuro = sofa_neuro.mask(gcs_total <= 6, 4)
    return sofa_neuro


def flag_sedated(
    gcs: pd.DataFrame,
    sedative_doses: pd.DataFrame,
    *,
    min_infusion_rate: float = 0.1,
    lookback: pd.Timedelta = pd.Timedelta("6h"),
) -> pd.DataFrame:
    """Flag GCS readings with a preceding active sedative or paralytic infusion.

    Active doses must start within ``lookback``. Empty dose input marks all readings unsedated.
    """
    active_doses = sedative_doses[sedative_doses["infusion_rate"] > min_infusion_rate]
    if active_doses.empty:
        return gcs.assign(sedated=False)
    result = pd.merge_asof(
        gcs.sort_values("measured_at"),
        active_doses[["icu_stay_id", "administered_at"]].sort_values("administered_at"),
        left_on="measured_at",
        right_on="administered_at",
        by="icu_stay_id",
        direction="backward",
        tolerance=lookback,
    )
    result["sedated"] = result["administered_at"].notna()
    return result.drop(columns="administered_at")


def flag_ventilated(
    gcs: pd.DataFrame,
    peep: pd.DataFrame,
    *,
    tolerance: pd.Timedelta = pd.Timedelta("1h"),
) -> pd.Series:
    """Return whether each GCS reading has a nearby PEEP measurement.

    The bounded nearest match captures asynchronous charting without extending one PEEP reading
    across the full stay. Results align to the GCS index.
    """
    if peep.empty:
        return pd.Series(False, index=gcs.index)

    # Preserve the original index across the sorted merge.
    readings = gcs[["icu_stay_id", "measured_at"]].rename_axis("_index").reset_index()
    matched = pd.merge_asof(
        readings.sort_values("measured_at"),
        peep[["icu_stay_id", "measured_at"]]
        .sort_values("measured_at")
        .rename(columns={"measured_at": "peep_measured_at"}),
        left_on="measured_at",
        right_on="peep_measured_at",
        by="icu_stay_id",
        direction="nearest",
        tolerance=tolerance,
    )
    # A matched PEEP timestamp indicates ventilation.
    ventilated = matched.set_index("_index")["peep_measured_at"].notna()
    return ventilated.reindex(gcs.index)


def backfill_ventilated_verbal(
    gcs_eye: pd.Series,
    gcs_motor: pd.Series,
    gcs_verbal: pd.Series,
    ventilated: pd.Series,
    icu_stay_id: pd.Series,
    measured_at: pd.Series,
) -> pd.Series:
    """Backfill unscoreable verbal values for ventilated patients.

    Verbal=1 with maximum eye and motor responses is replaced by the next non-null verbal score
    from the same stay.
    """
    working = pd.DataFrame(
        {"icu_stay_id": icu_stay_id, "measured_at": measured_at, "verbal": gcs_verbal}
    )
    unscoreable = (gcs_verbal == 1) & (gcs_eye == 4) & (gcs_motor == 6) & ventilated
    working["verbal"] = working["verbal"].mask(unscoreable, pd.NA)
    working = working.sort_values(["icu_stay_id", "measured_at"])
    working["verbal"] = working.groupby("icu_stay_id")["verbal"].bfill()
    return working["verbal"].reindex(gcs_verbal.index)


def apply_elective_surgery_day0(
    scored: pd.DataFrame, icu_stays: pd.DataFrame
) -> pd.DataFrame:
    """Set day-0 neurology to normal for elective-surgery stays.

    A synthetic day-0 row is added when absent, and existing day-0 scores are set to zero.
    """
    stay_flags = icu_stays[["icu_stay_id", "elective_surgery"]]
    merged = scored.merge(stay_flags, on="icu_stay_id", how="left")
    merged["elective_surgery"] = merged["elective_surgery"].fillna(False)

    has_day0 = merged.groupby("icu_stay_id")["day_index"].transform(
        lambda s: (s == 0).any()
    )
    needs_synthetic_day0 = merged["elective_surgery"] & ~has_day0

    synthetic_day0 = (
        merged.loc[needs_synthetic_day0, ["icu_stay_id", "elective_surgery"]]
        .drop_duplicates()
        .assign(
            day_index=0,
            sofa_neuro=0,
            gcs_total=15,
            gcs_eye=4,
            gcs_motor=6,
            gcs_verbal=5,
            measured_at=pd.NaT,
        )
    )

    combined = pd.concat([merged, synthetic_day0], ignore_index=True)
    combined["sofa_neuro"] = combined["sofa_neuro"].mask(
        combined["elective_surgery"] & (combined["day_index"] == 0), 0
    )
    return combined
