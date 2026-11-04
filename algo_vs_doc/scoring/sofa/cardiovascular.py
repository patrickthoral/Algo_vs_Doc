"""SOFA Cardiovascular component: mean arterial pressure (MAP) and vasoactive drug doses.

MAP and drug dose are scored independently: each is scored on its own timeline and
the two are combined by taking the worse (max) of the two per day.
"""

from __future__ import annotations

import pandas as pd

from algo_vs_doc.scoring import cdm, common
from algo_vs_doc.scoring.concepts import VASOPRESSOR_DRUG_CONCEPTS

#: dose categories (mcg/kg/min for dopamine/epinephrine/norepinephrine), ascending severity.
#: dobutamine has no dose threshold - any measurable dose qualifies for category 2.
DOSE_CATEGORIES = {
    2: {"dopamine": 5.0},
    3: {"dopamine": 5.0, "epinephrine": 0.1, "norepinephrine": 0.1},
    4: {"dopamine": 15.0, "epinephrine": 0.1, "norepinephrine": 0.1},
}

#: Ascending dose thresholds and tiers for sustained-dose gating.
_DRUG_TIER_THRESHOLDS = {
    "dopamine": [(0.0, 2), (5.0, 3), (15.0, 4)],
    "epinephrine": [(0.0, 3), (0.1, 4)],
    "norepinephrine": [(0.0, 3), (0.1, 4)],
    "dobutamine": [(0.0, 2)],
}


def active_doses(
    doses: pd.DataFrame, *, min_infusion_rate: float = 0.1
) -> pd.DataFrame:
    """Keep dose records above the minimum infusion rate.

    Paused and near-zero records must not enter threshold scoring as literal zero doses.
    """
    return doses[doses["infusion_rate"] > min_infusion_rate]


def score_map(map_value: pd.Series) -> pd.Series:
    """Score MAP as zero at or above 70 mmHg and one below 70 mmHg."""
    return pd.Series(0, index=map_value.index).mask(map_value < 70, 1)


def score_drugs(
    dopamine: pd.Series,
    epinephrine: pd.Series,
    norepinephrine: pd.Series,
    dobutamine_present: pd.Series,
) -> pd.Series:
    """Score instantaneous vasoactive drug doses in mcg/kg/min.

    Non-positive doses are treated as absent. Duration requirements are applied separately.
    """
    dopamine = dopamine.mask(dopamine <= 0, float("nan"))
    epinephrine = epinephrine.mask(epinephrine <= 0, float("nan"))
    norepinephrine = norepinephrine.mask(norepinephrine <= 0, float("nan"))

    sofa = pd.Series(0, index=dopamine.index)
    sofa = sofa.mask(dobutamine_present | (dopamine <= 5), 2)
    sofa = sofa.mask((dopamine > 5) | (epinephrine <= 0.1) | (norepinephrine <= 0.1), 3)
    sofa = sofa.mask((dopamine > 15) | (epinephrine > 0.1) | (norepinephrine > 0.1), 4)
    return sofa


def sustained_dose_windows(
    dose_intervals: pd.DataFrame,
    threshold: float,
    *,
    min_duration: pd.Timedelta = pd.Timedelta("1h"),
) -> pd.DataFrame:
    """Return contiguous dose windows at or above a threshold for ``min_duration``.

    Touching and overlapping qualifying intervals are merged before duration filtering.
    """
    qualifying = dose_intervals[dose_intervals["dose"] >= threshold]
    if qualifying.empty:
        return pd.DataFrame(
            columns=["icu_stay_id", "window_started_at", "window_ended_at"]
        )

    blocks = common.merge_intervals(
        qualifying["icu_stay_id"], qualifying["started_at"], qualifying["ended_at"]
    ).rename(columns={"started_at": "window_started_at", "ended_at": "window_ended_at"})
    blocks["duration"] = blocks["window_ended_at"] - blocks["window_started_at"]
    return blocks.loc[
        blocks["duration"] >= min_duration,
        ["icu_stay_id", "window_started_at", "window_ended_at"],
    ].reset_index(drop=True)


def covered_by_windows(
    icu_stay_id: pd.Series, measured_at: pd.Series, windows: pd.DataFrame
) -> pd.Series:
    """Return whether each reading falls in a sustained dose window.

    A qualifying window includes its full duration, including the initial required interval.
    """
    if windows.empty:
        return pd.Series(False, index=icu_stay_id.index)

    readings = pd.DataFrame(
        {"icu_stay_id": icu_stay_id, "measured_at": measured_at}
    ).reset_index()
    merged = readings.merge(windows, on="icu_stay_id", how="inner")
    in_window = (merged["measured_at"] >= merged["window_started_at"]) & (
        merged["measured_at"] <= merged["window_ended_at"]
    )
    covered_original_index = merged.loc[in_window, "index"].unique()
    return pd.Series(
        icu_stay_id.index.isin(covered_original_index), index=icu_stay_id.index
    )


def gate_dose_tiers(
    dose_intervals: pd.DataFrame,
    drug: str,
    icu_stay_id: pd.Series,
    measured_at: pd.Series,
    *,
    min_duration: pd.Timedelta = pd.Timedelta("1h"),
) -> pd.Series:
    """Return the highest dose tier sustained for ``min_duration`` at each reading.

    Each tier threshold is timed independently. Non-positive doses are excluded. Results align to
    ``measured_at`` and remain null until the lowest tier qualifies.
    """
    positive_intervals = dose_intervals[dose_intervals["dose"] > 0]
    tier = pd.Series(float("nan"), index=measured_at.index)
    for threshold, drug_tier in _DRUG_TIER_THRESHOLDS[drug]:
        windows = sustained_dose_windows(
            positive_intervals, threshold, min_duration=min_duration
        )
        covered = covered_by_windows(icu_stay_id, measured_at, windows)
        tier = tier.mask(covered, drug_tier)
    return tier


def score(
    map_measurement: pd.DataFrame,
    drug_exposure: pd.DataFrame,
    *,
    min_duration: pd.Timedelta = pd.Timedelta("1h"),
) -> pd.DataFrame:
    """Score MAP and sustained drug-dose contributions on their own timelines.

    Returns ``icu_stay_id, measured_at, sofa_cardvasc, source``. Drug readings require their tier
    threshold for ``min_duration``; MAP readings do not. Reduction occurs downstream.
    """
    contributions = [
        pd.DataFrame(
            {
                "icu_stay_id": map_measurement["icu_stay_id"],
                "measured_at": map_measurement["measured_at"],
                "sofa_cardvasc": score_map(map_measurement["value_as_number"]),
                "source": "map",
            }
        )
    ]

    for drug in VASOPRESSOR_DRUG_CONCEPTS:
        drug_readings = drug_exposure[drug_exposure["drug_concept_name"] == drug]
        if drug_readings.empty:
            continue
        dose_intervals = cdm.drug_exposure_intervals(drug_exposure, drug)
        tier = gate_dose_tiers(
            dose_intervals,
            drug,
            drug_readings["icu_stay_id"],
            drug_readings["drug_exposure_start_datetime"],
            min_duration=min_duration,
        )
        contributions.append(
            pd.DataFrame(
                {
                    "icu_stay_id": drug_readings["icu_stay_id"],
                    "measured_at": drug_readings["drug_exposure_start_datetime"],
                    "sofa_cardvasc": tier,
                    "source": drug,
                }
            )
        )

    combined = pd.concat(contributions, ignore_index=True)
    return combined.dropna(subset=["sofa_cardvasc"]).reset_index(drop=True)


def at_discharge(scored: pd.DataFrame, icu_stays: pd.DataFrame) -> pd.DataFrame:
    """Return the discharge cardiovascular score per stay.

    The latest pre-discharge value is selected per MAP or drug source, then the maximum is used.
    Stays without contributing readings remain null.
    """
    per_source = []
    for source, group in scored.groupby("source"):
        nearest = common.nearest_before_discharge(
            group.rename(columns={"sofa_cardvasc": source}), icu_stays, [source]
        )
        per_source.append(nearest.set_index("icu_stay_id")[source])

    combined = pd.concat(per_source, axis=1).max(axis=1)
    return combined.rename("sofa_cardvasc").reset_index()
