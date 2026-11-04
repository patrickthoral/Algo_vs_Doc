import pandas as pd

from algo_vs_doc.scoring.sofa import aggregate


def _component(column, rows):
    return pd.DataFrame(
        [{"icu_stay_id": s, "day_index": d, column: v} for s, d, v in rows]
    )


def test_daily_scores_takes_each_components_worst_value_of_the_day():
    components = {
        "sofa_resp": _component("sofa_resp", [("s1", 0, 1), ("s1", 0, 3)]),
        "sofa_liver": _component("sofa_liver", [("s1", 0, 2)]),
    }
    daily = aggregate.daily_scores(components)
    row = daily.set_index(["icu_stay_id", "day_index"]).loc[("s1", 0)]
    assert row["sofa_resp"] == 3  # worst of 1 and 3
    assert row["sofa_liver"] == 2
    assert row["sofa_total"] == 5


def test_daily_scores_fills_a_day_with_no_measurements_at_all():
    components = {"sofa_resp": _component("sofa_resp", [("s1", 0, 2), ("s1", 3, 1)])}
    daily = aggregate.daily_scores(components)
    assert daily["day_index"].tolist() == [0, 1, 2, 3]


def test_daily_scores_carries_a_missing_component_forward_for_one_day_only():
    components = {"sofa_resp": _component("sofa_resp", [("s1", 0, 4), ("s1", 3, 1)])}
    daily = aggregate.daily_scores(components).set_index("day_index")
    assert daily.loc[0, "sofa_resp"] == 4
    assert daily.loc[1, "sofa_resp"] == 4  # carried forward one day
    assert (
        daily.loc[2, "sofa_resp"] == 0
    )  # beyond the carry-forward limit -> absent -> 0
    assert daily.loc[3, "sofa_resp"] == 1


def test_daily_scores_totals_only_the_components_present():
    components = {
        "sofa_resp": _component("sofa_resp", [("s1", 0, 2)]),
        "sofa_kidneys": _component("sofa_kidneys", [("s1", 0, 3)]),
    }
    daily = aggregate.daily_scores(components)
    assert daily["sofa_total"].iloc[0] == 5
    assert "sofa_liver" not in daily.columns


def test_daily_scores_keeps_stays_independent():
    components = {"sofa_resp": _component("sofa_resp", [("s1", 0, 2), ("s2", 5, 1)])}
    daily = aggregate.daily_scores(components)
    # s2's own run starts at its own first scored day - it must not be padded back to s1's day 0
    assert daily[daily["icu_stay_id"] == "s1"]["day_index"].tolist() == [0]
    assert daily[daily["icu_stay_id"] == "s2"]["day_index"].tolist() == [5]


def test_at_admission_and_maximum():
    components = {"sofa_resp": _component("sofa_resp", [("s1", 0, 2), ("s1", 1, 4)])}
    daily = aggregate.daily_scores(components)
    assert aggregate.at_admission(daily)["sofa_admission"].tolist() == [2]
    assert aggregate.maximum(daily)["sofa_max"].tolist() == [4]


def test_at_discharge_anchors_each_component_independently_before_discharge():
    icu_stays = pd.DataFrame(
        [{"icu_stay_id": "s1", "icu_discharged_at": pd.Timestamp("2024-01-05 00:00")}]
    )
    components = {
        # last resp reading is the day before discharge
        "sofa_resp": pd.DataFrame(
            [
                {
                    "icu_stay_id": "s1",
                    "measured_at": pd.Timestamp("2024-01-02 08:00"),
                    "sofa_resp": 4,
                },
                {
                    "icu_stay_id": "s1",
                    "measured_at": pd.Timestamp("2024-01-04 08:00"),
                    "sofa_resp": 1,
                },
            ]
        ),
        # liver stopped being measured days earlier - it must still contribute its last value
        "sofa_liver": pd.DataFrame(
            [
                {
                    "icu_stay_id": "s1",
                    "measured_at": pd.Timestamp("2024-01-01 08:00"),
                    "sofa_liver": 3,
                }
            ]
        ),
    }
    result = aggregate.at_discharge(components, icu_stays).set_index("icu_stay_id")
    assert result.loc["s1", "sofa_resp"] == 1
    assert result.loc["s1", "sofa_liver"] == 3
    assert result.loc["s1", "sofa_discharge"] == 4


def test_at_discharge_never_selects_a_reading_logged_after_discharge():
    icu_stays = pd.DataFrame(
        [{"icu_stay_id": "s1", "icu_discharged_at": pd.Timestamp("2024-01-05 00:00")}]
    )
    components = {
        "sofa_liver": pd.DataFrame(
            [
                {
                    "icu_stay_id": "s1",
                    "measured_at": pd.Timestamp("2024-01-04 08:00"),
                    "sofa_liver": 1,
                },
                # a lab resulted after the patient left - closest in absolute time, but invalid
                {
                    "icu_stay_id": "s1",
                    "measured_at": pd.Timestamp("2024-01-05 02:00"),
                    "sofa_liver": 4,
                },
            ]
        )
    }
    result = aggregate.at_discharge(components, icu_stays)
    assert result["sofa_liver"].iloc[0] == 1


def test_at_discharge_gives_a_stay_with_no_readings_a_zero_not_a_missing_row():
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
    components = {
        "sofa_liver": pd.DataFrame(
            [
                {
                    "icu_stay_id": "s1",
                    "measured_at": pd.Timestamp("2024-01-04 08:00"),
                    "sofa_liver": 2,
                }
            ]
        )
    }
    result = aggregate.at_discharge(components, icu_stays).set_index("icu_stay_id")
    assert result.loc["s1", "sofa_discharge"] == 2
    assert result.loc["s2", "sofa_discharge"] == 0


def test_at_discharge_accepts_a_component_already_reduced_per_stay():
    # cardiovascular has its own at_discharge (MAP and drugs on independent timelines), so it
    # arrives here already keyed by icu_stay_id with no measured_at column
    icu_stays = pd.DataFrame(
        [{"icu_stay_id": "s1", "icu_discharged_at": pd.Timestamp("2024-01-05 00:00")}]
    )
    components = {
        "sofa_cardvasc": pd.DataFrame([{"icu_stay_id": "s1", "sofa_cardvasc": 4}]),
        "sofa_liver": pd.DataFrame(
            [
                {
                    "icu_stay_id": "s1",
                    "measured_at": pd.Timestamp("2024-01-04 08:00"),
                    "sofa_liver": 1,
                }
            ]
        ),
    }
    result = aggregate.at_discharge(components, icu_stays).set_index("icu_stay_id")
    assert result.loc["s1", "sofa_cardvasc"] == 4
    assert result.loc["s1", "sofa_discharge"] == 5
