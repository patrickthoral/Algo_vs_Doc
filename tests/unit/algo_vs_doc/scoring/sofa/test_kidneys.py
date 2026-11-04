import pandas as pd

from algo_vs_doc.scoring.sofa import kidneys


def test_score_creatinine_thresholds_with_normal_urine_output():
    creatinine = pd.Series([110, 111, 170, 171, 300, 301, 440, 441])
    urine_output = pd.Series([1000] * 8)
    result = kidneys.score(creatinine, urine_output)
    assert result.tolist() == [0, 1, 1, 2, 2, 3, 3, 4]


def test_score_urine_output_thresholds_with_normal_creatinine():
    creatinine = pd.Series([50] * 4)
    urine_output = pd.Series([500, 499, 200, 199])
    result = kidneys.score(creatinine, urine_output)
    assert result.tolist() == [0, 3, 3, 4]


def test_score_takes_worse_of_creatinine_or_urine_output():
    # creatinine alone implies category 1, urine output alone implies category 4 -> category 4 wins
    creatinine = pd.Series([120])
    urine_output = pd.Series([100])
    result = kidneys.score(creatinine, urine_output)
    assert result.tolist() == [4]


def test_extrapolate_daily_urine_output():
    urine_output_sum = pd.Series([250.0])
    coverage = pd.Series([pd.Timedelta(hours=12)])
    result = kidneys.extrapolate_daily_urine_output(urine_output_sum, coverage)
    assert result.iloc[0] == 500.0


def test_score_readings_combines_creatinine_with_its_own_days_extrapolated_urine():
    icu_stays = pd.DataFrame(
        [
            {
                "icu_stay_id": "s1",
                "icu_admitted_at": pd.Timestamp("2024-01-01 00:00"),
                "icu_discharged_at": pd.Timestamp("2024-01-05 00:00"),
            }
        ]
    )
    creatinine = pd.DataFrame(
        [
            {
                "icu_stay_id": "s1",
                "measured_at": pd.Timestamp("2024-01-01 08:00"),
                "value_as_number": 50,
            }
        ]
    )
    # two urine readings on day 0, summing to 200 mL, covering 12h -> extrapolated to 400 mL/24h (< 500)
    urine_output = pd.DataFrame(
        [
            {
                "icu_stay_id": "s1",
                "measured_at": pd.Timestamp("2024-01-01 06:00"),
                "value_as_number": 80,
            },
            {
                "icu_stay_id": "s1",
                "measured_at": pd.Timestamp("2024-01-01 18:00"),
                "value_as_number": 120,
            },
        ]
    )
    result = kidneys.score_readings(icu_stays, creatinine, urine_output)
    assert len(result) == 1
    assert (
        result["sofa_kidneys"].iloc[0] == 3
    )  # urine_output_daily 400 < 500 -> category 3


def test_score_readings_only_scores_a_day_with_both_creatinine_and_urine_output():
    icu_stays = pd.DataFrame(
        [
            {
                "icu_stay_id": "s1",
                "icu_admitted_at": pd.Timestamp("2024-01-01 00:00"),
                "icu_discharged_at": pd.Timestamp("2024-01-05 00:00"),
            }
        ]
    )
    creatinine = pd.DataFrame(
        [
            {
                "icu_stay_id": "s1",
                "measured_at": pd.Timestamp("2024-01-01 08:00"),
                "value_as_number": 50,
            },  # day 0, has urine
            {
                "icu_stay_id": "s1",
                "measured_at": pd.Timestamp("2024-01-02 08:00"),
                "value_as_number": 50,
            },  # day 1, no urine
        ]
    )
    urine_output = pd.DataFrame(
        [
            {
                "icu_stay_id": "s1",
                "measured_at": pd.Timestamp("2024-01-01 12:00"),
                "value_as_number": 1000,
            }
        ]
    )
    result = kidneys.score_readings(icu_stays, creatinine, urine_output)
    assert len(result) == 1
    assert result["measured_at"].iloc[0] == pd.Timestamp("2024-01-01 08:00")
