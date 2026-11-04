"""SOFA Respiratory component: PaO2/FiO2 ratio, and FiO2 estimation from an O2 device + flow rate."""

from __future__ import annotations

import pandas as pd

from algo_vs_doc.scoring import cdm


def pf_ratio(art_po2: pd.Series, fio2_fraction: pd.Series) -> pd.Series:
    """Return PaO2/FiO2 with PaO2 converted from kPa to mmHg."""
    return (art_po2 / fio2_fraction) * 760 / 101.325


def estimate_device_fio2(device: pd.Series, flow_rate: pd.Series) -> pd.Series:
    """Estimate FiO2 from a mapped oxygen device (nasul cannula, masks) and
    flow rate in litres per minute.

    Precise-FiO2 devices must use a recorded FiO2 measurement.

    Thresholds follow Lambden, S., Laterre, P.F., Levy, M.M. et al. The SOFA
    score-development, utility and challenges of accurate assessment in
    clinical trials. Crit Care 23, 374 (2019).
    https://doi.org/10.1186/s13054-019-2663-7
    """
    est_fio2 = pd.Series(pd.NA, index=device.index, dtype="Float64")

    est_fio2 = est_fio2.mask(device == "room_air", 0.21)

    is_cannula = device == "low_flow_nasal_cannula"
    est_fio2 = est_fio2.mask(is_cannula, (0.21 + flow_rate * 0.03).clip(upper=0.45))

    is_mask = device == "o2_face_mask"
    est_fio2 = est_fio2.mask(is_mask & (flow_rate <= 5), 0.4)
    est_fio2 = est_fio2.mask(is_mask & flow_rate.isin([6, 7]), 0.5)
    est_fio2 = est_fio2.mask(is_mask & (flow_rate >= 8), 0.6)

    is_reservoir = device == "o2_face_mask_reservoir"
    est_fio2 = est_fio2.mask(is_reservoir & (flow_rate <= 6), 0.6)
    est_fio2 = est_fio2.mask(is_reservoir & (flow_rate == 7), 0.7)
    est_fio2 = est_fio2.mask(is_reservoir & (flow_rate == 8), 0.8)
    est_fio2 = est_fio2.mask(is_reservoir & (flow_rate == 9), 0.9)
    est_fio2 = est_fio2.mask(is_reservoir & (flow_rate >= 10), 0.95)

    return est_fio2


def score(pf: pd.Series, ventilated: pd.Series) -> pd.Series:
    """Score SOFA Respiratory from PF ratio and ventilation status."""
    ventilated = ventilated.astype("boolean").fillna(False)
    sofa_resp = pd.Series(0, index=pf.index)
    sofa_resp = sofa_resp.mask(pf < 400, 1)
    sofa_resp = sofa_resp.mask(pf < 300, 2)
    sofa_resp = sofa_resp.mask((pf < 200) & ventilated, 3)
    sofa_resp = sofa_resp.mask((pf < 100) & ventilated, 4)
    return sofa_resp


def _nearest_reading(
    readings: pd.DataFrame,
    other: pd.DataFrame,
    value_col: str,
    *,
    tolerance: pd.Timedelta | None = None,
) -> pd.DataFrame:
    """Attach the nearest per-stay reading and its signed time difference in minutes."""
    matched_at = f"{value_col}_measured_at"
    if other.empty:
        return readings.assign(
            **{value_col: float("nan"), f"minutes_to_{value_col}": float("nan")}
        )

    merged = pd.merge_asof(
        readings.sort_values("measured_at"),
        other[["icu_stay_id", "measured_at", value_col]]
        .sort_values("measured_at")
        .rename(columns={"measured_at": matched_at}),
        left_on="measured_at",
        right_on=matched_at,
        by="icu_stay_id",
        direction="nearest",
        tolerance=tolerance,
    )
    merged[f"minutes_to_{value_col}"] = (
        merged[matched_at] - merged["measured_at"]
    ).dt.total_seconds() / 60
    return merged.drop(columns=matched_at)


def score_readings(
    measurement: pd.DataFrame,
    *,
    ventilated_tolerance: pd.Timedelta = pd.Timedelta("1h"),
) -> pd.DataFrame:
    """Score SOFA Respiratory for each arterial PO2 reading.

    Each PaO2 is paired with the nearest recorded and device-estimated FiO2; the closer source is
    used. PEEP within ``ventilated_tolerance`` indicates ventilation. Readings without FiO2 are
    excluded because their PF ratio is unknown.
    """

    def _numeric(concept: str, name: str) -> pd.DataFrame:
        rows = measurement.loc[measurement["measurement_concept_name"] == concept]
        return rows[["icu_stay_id", "measured_at", "value_as_number"]].rename(
            columns={"value_as_number": name}
        )

    art_po2 = _numeric("art_po2", "art_po2")
    if art_po2.empty:
        return pd.DataFrame(
            columns=["icu_stay_id", "measured_at", "sofa_pf", "ventilated", "sofa_resp"]
        )

    # Use directly recorded FiO2 where available.
    direct_fio2 = _numeric("fio2_fraction", "fio2_fraction")

    # Estimate FiO2 from device and flow recorded at the same timestamp.
    device_wide = cdm.pivot_measurements(
        measurement.loc[
            measurement["measurement_concept_name"].isin(["o2_device", "o2_flow_rate"])
        ]
    )
    for missing in {"o2_device", "o2_flow_rate"} - set(device_wide.columns):
        device_wide[missing] = float("nan")
    device_wide["est_fio2"] = estimate_device_fio2(
        device_wide["o2_device"], device_wide["o2_flow_rate"]
    )
    est_fio2 = device_wide.loc[
        device_wide["est_fio2"].notna(), ["icu_stay_id", "measured_at", "est_fio2"]
    ]

    scored = _nearest_reading(art_po2, direct_fio2, "fio2_fraction")
    scored = _nearest_reading(scored, est_fio2, "est_fio2")
    scored = _nearest_reading(
        scored, _numeric("peep", "peep"), "peep", tolerance=ventilated_tolerance
    )

    scored["pf_direct"] = pf_ratio(scored["art_po2"], scored["fio2_fraction"])
    scored["pf_estimated"] = pf_ratio(scored["art_po2"], scored["est_fio2"])

    # Use the temporally closest FiO2 source, with the available source as fallback.
    use_direct = (
        scored["minutes_to_fio2_fraction"].abs() <= scored["minutes_to_est_fio2"].abs()
    ) | scored["pf_estimated"].isna()
    scored["sofa_pf"] = scored["pf_direct"].where(use_direct, scored["pf_estimated"])
    scored["ventilated"] = scored["peep"].notna()

    scored = scored.loc[scored["sofa_pf"].notna()].copy()
    scored["sofa_resp"] = score(scored["sofa_pf"], scored["ventilated"])
    return scored[
        ["icu_stay_id", "measured_at", "sofa_pf", "ventilated", "sofa_resp"]
    ].reset_index(drop=True)
