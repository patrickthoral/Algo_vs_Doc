import pandas as pd

from algo_vs_doc.scoring import common


def merge_intervals_and_sort(*args, **kwargs):
    # merge_intervals does not guarantee row order - sort for deterministic assertions
    return (
        common.merge_intervals(*args, **kwargs)
        .sort_values("started_at")
        .reset_index(drop=True)
    )


def test_add_day_index_computes_whole_days_since_admission():
    icu_stays = pd.DataFrame(
        [
            {
                "icu_stay_id": "s1",
                "icu_admitted_at": pd.Timestamp("2024-01-01 08:00"),
                "icu_discharged_at": pd.Timestamp("2024-01-04 10:00"),
            }
        ]
    )
    measurements = pd.DataFrame(
        [
            {
                "icu_stay_id": "s1",
                "measured_at": pd.Timestamp("2024-01-01 09:00"),
                "value": 1,
            },  # day 0
            {
                "icu_stay_id": "s1",
                "measured_at": pd.Timestamp("2024-01-02 07:59"),
                "value": 2,
            },  # day 0 (< 24h)
            {
                "icu_stay_id": "s1",
                "measured_at": pd.Timestamp("2024-01-02 08:01"),
                "value": 3,
            },  # day 1
        ]
    )
    result = common.add_day_index(measurements, icu_stays)
    assert result["day_index"].tolist() == [0, 0, 1]


def test_add_day_index_clips_readings_outside_the_stay():
    icu_stays = pd.DataFrame(
        [
            {
                "icu_stay_id": "s1",
                "icu_admitted_at": pd.Timestamp("2024-01-01 08:00"),
                "icu_discharged_at": pd.Timestamp("2024-01-02 08:00"),
            }
        ]
    )
    measurements = pd.DataFrame(
        [
            {
                "icu_stay_id": "s1",
                "measured_at": pd.Timestamp("2024-01-01 07:00"),
                "value": "before_admission",
            },
            {
                "icu_stay_id": "s1",
                "measured_at": pd.Timestamp("2024-01-01 12:00"),
                "value": "during_stay",
            },
            {
                "icu_stay_id": "s1",
                "measured_at": pd.Timestamp("2024-01-02 09:00"),
                "value": "after_discharge",
            },
        ]
    )
    result = common.add_day_index(measurements, icu_stays, clip_to_stay=True)
    assert result["value"].tolist() == ["during_stay"]


def test_nearest_before_discharge_never_picks_a_post_discharge_reading():
    # regression guard: this is exactly the bug class found in several discharge-combination cells -
    # sorting by absolute time-to-discharge (rather than backward-only matching) can pick a reading
    # logged after the nominal discharge time (e.g. a late-arriving lab).
    icu_stays = pd.DataFrame(
        [{"icu_stay_id": "s1", "icu_discharged_at": pd.Timestamp("2024-01-04 10:00")}]
    )
    measurements = pd.DataFrame(
        [
            {
                "icu_stay_id": "s1",
                "measured_at": pd.Timestamp("2024-01-04 08:00"),
                "value": "real_nearest",
            },
            {
                "icu_stay_id": "s1",
                "measured_at": pd.Timestamp("2024-01-04 11:00"),
                "value": "late_arriving_straggler",
            },
        ]
    )
    result = common.nearest_before_discharge(
        measurements, icu_stays, value_cols=["value"]
    )
    assert result["value"].tolist() == ["real_nearest"]


def test_nearest_before_discharge_drops_rows_with_no_real_timestamp():
    # e.g. a synthetic default row (elective-surgery day-0 GCS) with no real measured_at
    icu_stays = pd.DataFrame(
        [{"icu_stay_id": "s1", "icu_discharged_at": pd.Timestamp("2024-01-04 10:00")}]
    )
    measurements = pd.DataFrame(
        [
            {"icu_stay_id": "s1", "measured_at": pd.NaT, "value": "synthetic_default"},
            {
                "icu_stay_id": "s1",
                "measured_at": pd.Timestamp("2024-01-02 08:00"),
                "value": "real_reading",
            },
        ]
    )
    result = common.nearest_before_discharge(
        measurements, icu_stays, value_cols=["value"]
    )
    assert result["value"].tolist() == ["real_reading"]


def test_nearest_before_discharge_supports_multiple_value_columns():
    icu_stays = pd.DataFrame(
        [{"icu_stay_id": "s1", "icu_discharged_at": pd.Timestamp("2024-01-04 10:00")}]
    )
    measurements = pd.DataFrame(
        [
            {
                "icu_stay_id": "s1",
                "measured_at": pd.Timestamp("2024-01-04 08:00"),
                "a": 1,
                "b": 2,
            }
        ]
    )
    result = common.nearest_before_discharge(
        measurements, icu_stays, value_cols=["a", "b"]
    )
    assert result[["a", "b"]].iloc[0].tolist() == [1, 2]


def test_nearest_before_discharge_supports_a_native_icu_stay_id_column_name():
    # a hospital whose own identifier is already 1:1 with a stay (e.g. admission_id) can use
    # this directly, without first renaming to "icu_stay_id" and back
    icu_stays = pd.DataFrame(
        [{"admission_id": "a1", "icu_discharged_at": pd.Timestamp("2024-01-04 10:00")}]
    )
    measurements = pd.DataFrame(
        [
            {
                "admission_id": "a1",
                "measured_at": pd.Timestamp("2024-01-04 08:00"),
                "value": 42,
            }
        ]
    )
    result = common.nearest_before_discharge(
        measurements, icu_stays, value_cols=["value"], icu_stay_id_col="admission_id"
    )
    assert result["value"].tolist() == [42]
    assert "admission_id" in result.columns


def test_merge_intervals_merges_touching_and_overlapping_but_not_gapped():
    icu_stay_id = pd.Series(["s1", "s1", "s1"])
    started_at = pd.Series(
        [
            pd.Timestamp("2024-01-01 08:00"),
            pd.Timestamp(
                "2024-01-01 08:30"
            ),  # touches the first interval's end exactly
            pd.Timestamp("2024-01-01 12:00"),  # a real gap after the second interval
        ]
    )
    ended_at = pd.Series(
        [
            pd.Timestamp("2024-01-01 08:30"),
            pd.Timestamp("2024-01-01 09:00"),
            pd.Timestamp("2024-01-01 12:30"),
        ]
    )
    result = merge_intervals_and_sort(icu_stay_id, started_at, ended_at)
    assert result["started_at"].tolist() == [
        pd.Timestamp("2024-01-01 08:00"),
        pd.Timestamp("2024-01-01 12:00"),
    ]
    assert result["ended_at"].tolist() == [
        pd.Timestamp("2024-01-01 09:00"),
        pd.Timestamp("2024-01-01 12:30"),
    ]


def test_merge_intervals_with_max_gap_bridges_a_short_gap_but_not_a_long_one():
    icu_stay_id = pd.Series(["s1", "s1", "s1"])
    # three point-in-time events (start == end), 20 minutes apart, then a 2h gap, then one more
    started_at = ended_at = pd.Series(
        [
            pd.Timestamp("2024-01-01 08:00"),
            pd.Timestamp("2024-01-01 08:20"),
            pd.Timestamp("2024-01-01 10:20"),
        ]
    )
    result = merge_intervals_and_sort(
        icu_stay_id, started_at, ended_at, max_gap=pd.Timedelta(minutes=30)
    )
    assert result["started_at"].tolist() == [
        pd.Timestamp("2024-01-01 08:00"),
        pd.Timestamp("2024-01-01 10:20"),
    ]
    assert result["ended_at"].tolist() == [
        pd.Timestamp("2024-01-01 08:20"),
        pd.Timestamp("2024-01-01 10:20"),
    ]


def test_merge_intervals_keeps_separate_stays_separate():
    icu_stay_id = pd.Series(["s1", "s2"])
    started_at = pd.Series([pd.Timestamp("2024-01-01 08:00")] * 2)
    ended_at = pd.Series([pd.Timestamp("2024-01-01 09:00")] * 2)
    result = common.merge_intervals(icu_stay_id, started_at, ended_at)
    assert sorted(result["icu_stay_id"].tolist()) == ["s1", "s2"]


def test_reduce_daily_max_takes_the_worst_reading_per_stay_per_day():
    scored = pd.DataFrame(
        [
            {"icu_stay_id": "s1", "day_index": 0, "score": 1},
            {"icu_stay_id": "s1", "day_index": 0, "score": 3},
            {"icu_stay_id": "s1", "day_index": 1, "score": 2},
            {"icu_stay_id": "s2", "day_index": 0, "score": 4},
        ]
    )
    result = common.reduce_daily_max(scored, score_col="score")
    result = result.sort_values(["icu_stay_id", "day_index"]).reset_index(drop=True)
    assert result["score"].tolist() == [3, 2, 4]  # s1/day0 takes the worse of 1 and 3
    assert result["day_index"].tolist() == [0, 1, 0]
