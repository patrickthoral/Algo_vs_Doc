import pandas as pd
import pytest

from algo_vs_doc.scoring.sofa import respiratory


def test_pf_ratio_converts_kpa_to_mmhg():
    art_po2 = pd.Series([13.3])  # kPa, ~100 mmHg
    fio2_fraction = pd.Series([1.0])
    result = respiratory.pf_ratio(art_po2, fio2_fraction)
    assert round(result.iloc[0]) == 100


def test_estimate_device_fio2_room_air_ignores_flow_rate():
    device = pd.Series(["room_air", "room_air"])
    flow_rate = pd.Series([0, 15])
    result = respiratory.estimate_device_fio2(device, flow_rate)
    assert result.tolist() == [0.21, 0.21]


def test_estimate_device_fio2_low_flow_nasal_cannula_formula_and_clip():
    device = pd.Series(["low_flow_nasal_cannula"] * 3)
    flow_rate = pd.Series([1, 5, 20])
    result = respiratory.estimate_device_fio2(device, flow_rate)
    assert result.tolist() == [
        0.24,
        0.36,
        0.45,
    ]  # 20 L/min would be 0.81 uncapped, clipped to 0.45


def test_estimate_device_fio2_face_mask_categories():
    device = pd.Series(["o2_face_mask"] * 3)
    flow_rate = pd.Series([5, 6, 8])
    result = respiratory.estimate_device_fio2(device, flow_rate)
    assert result.tolist() == [0.4, 0.5, 0.6]


def test_estimate_device_fio2_reservoir_mask_categories():
    device = pd.Series(["o2_face_mask_reservoir"] * 5)
    flow_rate = pd.Series([6, 7, 8, 9, 10])
    result = respiratory.estimate_device_fio2(device, flow_rate)
    assert result.tolist() == [0.6, 0.7, 0.8, 0.9, 0.95]


def test_score_thresholds_require_ventilated_for_severe_categories():
    pf = pd.Series([400, 399, 300, 299, 199, 199, 99, 99])
    ventilated = pd.Series([False, False, False, False, False, True, False, True])
    result = respiratory.score(pf, ventilated)
    # pf<200/100 categories only apply when ventilated; unventilated stays capped at category 2
    assert result.tolist() == [0, 1, 1, 2, 2, 3, 2, 4]


def test_score_missing_ventilated_defaults_to_false():
    pf = pd.Series([50])
    ventilated = pd.Series([pd.NA], dtype="boolean")
    result = respiratory.score(pf, ventilated)
    assert result.tolist() == [
        2
    ]  # not 4, since ventilated is treated as False when unknown


def _measurement(rows):
    return pd.DataFrame(
        [
            {
                "icu_stay_id": r[0],
                "measurement_concept_name": r[1],
                "measured_at": pd.Timestamp(r[2]),
                "value_as_number": r[3],
                "value_as_string": r[4] if len(r) > 4 else None,
            }
            for r in rows
        ]
    )


def test_score_readings_prefers_the_fio2_source_recorded_closer_in_time():
    # a directly recorded FiO2 15min away vs a device-estimated one 4h away -> direct wins.
    # art_po2 12 kPa / fio2 0.5 -> pf ~180 ; with the device estimate (0.21) it'd be ~429,
    # so the two sources land in clearly different score categories.
    measurement = _measurement(
        [
            ("s1", "art_po2", "2024-01-01 12:00", 12.0),
            ("s1", "fio2_fraction", "2024-01-01 12:15", 0.5),
            ("s1", "o2_device", "2024-01-01 08:00", float("nan"), "room_air"),
            ("s1", "peep", "2024-01-01 12:05", 8.0),
        ]
    )
    result = respiratory.score_readings(measurement)
    assert len(result) == 1
    assert result["sofa_pf"].iloc[0] == pytest.approx(12.0 / 0.5 * 760 / 101.325)
    assert bool(result["ventilated"].iloc[0]) is True


def test_score_readings_falls_back_to_the_device_estimate_when_no_direct_fio2_exists():
    measurement = _measurement(
        [
            ("s1", "art_po2", "2024-01-01 12:00", 12.0),
            ("s1", "o2_device", "2024-01-01 12:05", float("nan"), "room_air"),
        ]
    )
    result = respiratory.score_readings(measurement)
    assert len(result) == 1
    assert result["sofa_pf"].iloc[0] == pytest.approx(12.0 / 0.21 * 760 / 101.325)


def test_score_readings_falls_back_to_the_direct_fio2_when_no_device_reading_exists():
    measurement = _measurement(
        [
            ("s1", "art_po2", "2024-01-01 12:00", 12.0),
            (
                "s1",
                "fio2_fraction",
                "2024-01-01 18:00",
                0.5,
            ),  # far away, but the only source
        ]
    )
    result = respiratory.score_readings(measurement)
    assert len(result) == 1
    assert result["sofa_pf"].iloc[0] == pytest.approx(12.0 / 0.5 * 760 / 101.325)


def test_score_readings_drops_a_pao2_reading_with_no_fio2_from_either_source():
    # unscoreable is not the same as normal - it must not silently score 0
    measurement = _measurement([("s1", "art_po2", "2024-01-01 12:00", 12.0)])
    result = respiratory.score_readings(measurement)
    assert result.empty


def test_score_readings_ventilated_respects_the_tolerance():
    # the only PEEP reading for this stay is 2 days from the blood gas - without a tolerance a
    # nearest-match would still call this patient ventilated for the whole stay
    measurement = _measurement(
        [
            ("s1", "art_po2", "2024-01-01 12:00", 12.0),
            ("s1", "fio2_fraction", "2024-01-01 12:00", 0.5),
            ("s1", "peep", "2024-01-03 12:00", 8.0),
        ]
    )
    result = respiratory.score_readings(measurement)
    assert bool(result["ventilated"].iloc[0]) is False
    # pf ~180: only scores category 3 if ventilated, so this must stay at category 2
    assert result["sofa_resp"].iloc[0] == 2


def test_score_readings_does_not_match_readings_across_stays():
    measurement = _measurement(
        [
            ("s1", "art_po2", "2024-01-01 12:00", 12.0),
            ("s2", "fio2_fraction", "2024-01-01 12:00", 0.5),  # a different stay's FiO2
        ]
    )
    result = respiratory.score_readings(measurement)
    assert result.empty
