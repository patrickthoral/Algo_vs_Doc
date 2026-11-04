import pandas as pd

from algo_vs_doc.scoring.news import assemble


def _measurement(rows):
    return pd.DataFrame(
        [
            {
                "icu_stay_id": s,
                "measurement_concept_name": concept,
                "measured_at": pd.Timestamp(at),
                "value_as_number": v,
                "value_as_string": None,
            }
            for s, concept, at, v in rows
        ]
    )


def test_score_components_scores_each_concept_on_its_own_readings():
    measurement = _measurement(
        [
            ("s1", "respiratory_rate", "2024-01-01 08:00", 26),  # >=25 -> 3
            ("s1", "heart_rate", "2024-01-01 09:00", 45),  # <=50 -> 1
            ("s1", "spo2", "2024-01-01 09:30", 92),  # <=93 -> 2
        ]
    )
    scored = assemble.score_components(measurement)
    assert scored["news_respiratory"]["news_respiratory"].tolist() == [3]
    assert scored["news_heart_rate"]["news_heart_rate"].tolist() == [1]
    assert scored["news_oxygen_saturation"]["news_oxygen_saturation"].tolist() == [2]
    # each component keeps its own timeline rather than being joined onto a shared one
    assert scored["news_respiratory"]["measured_at"].iloc[0] == pd.Timestamp(
        "2024-01-01 08:00"
    )


def test_score_components_skips_a_concept_that_was_never_measured():
    measurement = _measurement([("s1", "heart_rate", "2024-01-01 09:00", 80)])
    scored = assemble.score_components(measurement)
    assert set(scored) == {"news_heart_rate"}


def test_score_components_treats_any_positive_o2_flow_as_supplemental_oxygen():
    measurement = _measurement(
        [
            ("s1", "o2_flow_rate", "2024-01-01 08:00", 0.0),
            ("s1", "o2_flow_rate", "2024-01-01 09:00", 2.0),
        ]
    )
    scored = assemble.score_components(measurement)
    assert scored["news_supplemental_oxygen"]["news_supplemental_oxygen"].tolist() == [
        0,
        2,
    ]


def test_score_avpu_from_gcs_scores_not_alert_as_three():
    sofa_neuro = pd.DataFrame(
        [
            {
                "icu_stay_id": "s1",
                "measured_at": pd.Timestamp("2024-01-01 08:00"),
                "gcs_total": 15,
            },
            {
                "icu_stay_id": "s1",
                "measured_at": pd.Timestamp("2024-01-02 08:00"),
                "gcs_total": 12,
            },
        ]
    )
    result = assemble.score_avpu_from_gcs(sofa_neuro)
    assert result["news_avpu"].tolist() == [0, 3]


def test_score_avpu_from_gcs_drops_an_unknown_gcs_rather_than_calling_it_alert():
    sofa_neuro = pd.DataFrame(
        [
            {
                "icu_stay_id": "s1",
                "measured_at": pd.Timestamp("2024-01-01 08:00"),
                "gcs_total": float("nan"),
            },
            {
                "icu_stay_id": "s1",
                "measured_at": pd.Timestamp("2024-01-02 08:00"),
                "gcs_total": 10,
            },
        ]
    )
    result = assemble.score_avpu_from_gcs(sofa_neuro)
    assert len(result) == 1
    assert result["news_avpu"].tolist() == [3]


def test_at_discharge_sums_each_components_own_last_value_before_discharge():
    icu_stays = pd.DataFrame(
        [{"icu_stay_id": "s1", "icu_discharged_at": pd.Timestamp("2024-01-05 00:00")}]
    )
    scored = {
        "news_heart_rate": pd.DataFrame(
            [
                {
                    "icu_stay_id": "s1",
                    "measured_at": pd.Timestamp("2024-01-04 08:00"),
                    "news_heart_rate": 3,
                },
                {
                    "icu_stay_id": "s1",
                    "measured_at": pd.Timestamp("2024-01-04 20:00"),
                    "news_heart_rate": 1,
                },
            ]
        ),
        # charted hours before the heart rate - still contributes its own last value
        "news_temperature": pd.DataFrame(
            [
                {
                    "icu_stay_id": "s1",
                    "measured_at": pd.Timestamp("2024-01-04 06:00"),
                    "news_temperature": 2,
                }
            ]
        ),
    }
    result = assemble.at_discharge(scored, icu_stays).set_index("icu_stay_id")
    assert result.loc["s1", "news_heart_rate"] == 1
    assert result.loc["s1", "news_temperature"] == 2
    assert result.loc["s1", "news_discharge"] == 3


def test_at_discharge_never_selects_a_reading_logged_after_discharge():
    icu_stays = pd.DataFrame(
        [{"icu_stay_id": "s1", "icu_discharged_at": pd.Timestamp("2024-01-05 00:00")}]
    )
    scored = {
        "news_heart_rate": pd.DataFrame(
            [
                {
                    "icu_stay_id": "s1",
                    "measured_at": pd.Timestamp("2024-01-04 08:00"),
                    "news_heart_rate": 1,
                },
                {
                    "icu_stay_id": "s1",
                    "measured_at": pd.Timestamp("2024-01-05 01:00"),
                    "news_heart_rate": 3,
                },
            ]
        )
    }
    result = assemble.at_discharge(scored, icu_stays)
    assert result["news_heart_rate"].iloc[0] == 1


def test_at_discharge_covers_every_stay_including_one_with_no_readings():
    icu_stays = pd.DataFrame(
        [
            {
                "icu_stay_id": "s1",
                "icu_discharged_at": pd.Timestamp("2024-01-05 00:00"),
            },
            {
                "icu_stay_id": "s2",
                "icu_discharged_at": pd.Timestamp("2024-01-06 00:00"),
            },
        ]
    )
    scored = {
        "news_heart_rate": pd.DataFrame(
            [
                {
                    "icu_stay_id": "s1",
                    "measured_at": pd.Timestamp("2024-01-04 08:00"),
                    "news_heart_rate": 2,
                }
            ]
        )
    }
    result = assemble.at_discharge(scored, icu_stays).set_index("icu_stay_id")
    assert result.loc["s1", "news_discharge"] == 2
    assert result.loc["s2", "news_discharge"] == 0
