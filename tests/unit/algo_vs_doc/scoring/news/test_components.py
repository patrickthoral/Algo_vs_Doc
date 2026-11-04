import pandas as pd

from algo_vs_doc.scoring.news import components


def test_score_respiratory_rate_thresholds():
    resp_rate = pd.Series([12, 11, 9, 8, 20, 21, 24, 25])
    result = components.score_respiratory_rate(resp_rate)
    assert result.tolist() == [0, 1, 1, 3, 0, 2, 2, 3]


def test_score_spo2_thresholds():
    spo2 = pd.Series([96, 95, 94, 93, 92, 91])
    result = components.score_spo2(spo2)
    assert result.tolist() == [0, 1, 1, 2, 2, 3]


def test_score_supplemental_oxygen():
    on_oxygen = pd.Series([False, True])
    result = components.score_supplemental_oxygen(on_oxygen)
    assert result.tolist() == [0, 2]


def test_score_temperature_thresholds():
    temp = pd.Series([36.1, 36, 35, 38, 38.1, 39, 39.1])
    result = components.score_temperature(temp)
    assert result.tolist() == [0, 1, 3, 0, 1, 1, 2]


def test_score_blood_pressure_thresholds():
    systolic = pd.Series([111, 110, 100, 90, 89, 219, 220])
    result = components.score_blood_pressure(systolic)
    assert result.tolist() == [0, 1, 2, 3, 3, 0, 3]


def test_score_heart_rate_thresholds():
    hr = pd.Series([51, 50, 40, 39, 90, 91, 110, 111, 130, 131])
    result = components.score_heart_rate(hr)
    assert result.tolist() == [0, 1, 3, 3, 0, 1, 1, 2, 2, 3]


def test_score_avpu_from_gcs_total():
    gcs_total = pd.Series([15, 14, 3])
    result = components.score_avpu(gcs_total)
    assert result.tolist() == [0, 3, 3]
