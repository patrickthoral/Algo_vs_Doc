"""NEWS (National Early Warning Score) component scoring functions.

Each function scores one NEWS component from an already-normalized value. Unit/format
conversions forced by a specific source system (e.g. Fahrenheit->Celsius, or splitting a
combined "systolic/diastolic" string into a plain systolic number) are the caller's
responsibility and must happen before calling these - see ``scoring/__init__.py`` for why
that conversion logic (which is source-system-specific) does not belong in this module.
"""

from __future__ import annotations

import pandas as pd


def score_respiratory_rate(resp_rate: pd.Series) -> pd.Series:
    """0; <=11 -> 1; <=8 -> 3; >=21 -> 2; >=25 -> 3 (breaths/min)."""
    news = pd.Series(0, index=resp_rate.index)
    news = news.mask(resp_rate <= 11, 1)
    news = news.mask(resp_rate <= 8, 3)
    news = news.mask(resp_rate >= 21, 2)
    news = news.mask(resp_rate >= 25, 3)
    return news


def score_spo2(spo2: pd.Series) -> pd.Series:
    """0; <=95 -> 1; <=93 -> 2; <=91 -> 3 (%)."""
    news = pd.Series(0, index=spo2.index)
    news = news.mask(spo2 <= 95, 1)
    news = news.mask(spo2 <= 93, 2)
    news = news.mask(spo2 <= 91, 3)
    return news


def score_supplemental_oxygen(on_supplemental_oxygen: pd.Series) -> pd.Series:
    """0; any supplemental oxygen (a positive flow rate) -> 2."""
    news = pd.Series(0, index=on_supplemental_oxygen.index)
    news = news.mask(on_supplemental_oxygen, 2)
    return news


def score_temperature(temperature_celsius: pd.Series) -> pd.Series:
    """0; <=36 -> 1; <=35 -> 3; >=38.1 -> 1; >=39.1 -> 2 (degrees Celsius)."""
    news = pd.Series(0, index=temperature_celsius.index)
    news = news.mask(temperature_celsius <= 36, 1)
    news = news.mask(temperature_celsius <= 35, 3)
    news = news.mask(temperature_celsius >= 38.1, 1)
    news = news.mask(temperature_celsius >= 39.1, 2)
    return news


def score_blood_pressure(systolic: pd.Series) -> pd.Series:
    """0; <=110 -> 1; <=100 -> 2; <=90 -> 3; >=220 -> 3 (mmHg, systolic only)."""
    news = pd.Series(0, index=systolic.index)
    news = news.mask(systolic <= 110, 1)
    news = news.mask(systolic <= 100, 2)
    news = news.mask(systolic <= 90, 3)
    news = news.mask(systolic >= 220, 3)
    return news


def score_heart_rate(heart_rate: pd.Series) -> pd.Series:
    """0; <=50 -> 1; <=40 -> 3; >=91 -> 1; >=111 -> 2; >=131 -> 3 (beats/min)."""
    news = pd.Series(0, index=heart_rate.index)
    news = news.mask(heart_rate <= 50, 1)
    news = news.mask(heart_rate <= 40, 3)
    news = news.mask(heart_rate >= 91, 1)
    news = news.mask(heart_rate >= 111, 2)
    news = news.mask(heart_rate >= 131, 3)
    return news


def score_avpu(gcs_total: pd.Series) -> pd.Series:
    """0; gcs_total <15 (not fully "Alert") -> 3.

    Neither source hospital scores AVPU directly on the ICU, so it is derived from the
    SOFA Neurology GCS total instead: any patient who is not fully Alert (GCS < 15)
    scores 3.
    """
    news = pd.Series(0, index=gcs_total.index)
    news = news.mask(gcs_total < 15, 3)
    return news
