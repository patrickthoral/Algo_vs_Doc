import pandas as pd

from algo_vs_doc.scoring import vasopressors


def test_reconstruct_intervals_from_events_merges_a_burst_within_max_gap():
    icu_stay_id = pd.Series(["s1", "s1", "s1"])
    event_time = pd.Series(
        [
            pd.Timestamp("2024-01-01 08:00"),
            pd.Timestamp("2024-01-01 08:20"),
            pd.Timestamp("2024-01-01 08:45"),
        ]
    )
    result = vasopressors.reconstruct_intervals_from_events(
        icu_stay_id, event_time, max_gap=pd.Timedelta(minutes=30)
    )
    assert len(result) == 1
    assert result["started_at"].iloc[0] == pd.Timestamp("2024-01-01 08:00")
    assert result["ended_at"].iloc[0] == pd.Timestamp("2024-01-01 08:45")


def test_reconstruct_intervals_from_events_splits_across_a_long_gap():
    icu_stay_id = pd.Series(["s1", "s1"])
    event_time = pd.Series(
        [pd.Timestamp("2024-01-01 08:00"), pd.Timestamp("2024-01-01 14:00")]
    )
    result = vasopressors.reconstruct_intervals_from_events(
        icu_stay_id, event_time, max_gap=pd.Timedelta(minutes=30)
    )
    assert len(result) == 2


def test_total_duration_sums_intervals_and_excludes_gaps():
    # regression guard for the sum-of-intervals vs. first-to-last-span distinction: a stop-then-restart
    # must not count the gap in between as time on support.
    intervals = pd.DataFrame(
        [
            {
                "icu_stay_id": "s1",
                "started_at": pd.Timestamp("2024-01-01 08:00"),
                "ended_at": pd.Timestamp("2024-01-01 09:00"),  # 1h
            },
            {
                "icu_stay_id": "s1",
                "started_at": pd.Timestamp(
                    "2024-01-01 15:00"
                ),  # 6h gap since the previous interval ended
                "ended_at": pd.Timestamp("2024-01-01 15:30"),  # 30min
            },
        ]
    )
    result = vasopressors.total_duration(intervals)
    assert result.loc["s1"] == pd.Timedelta(
        hours=1, minutes=30
    )  # not 7.5h (first-to-last span)


def test_reconstruct_intervals_from_mar_events_ends_each_event_at_the_next_ones_start():
    icu_stay_id = pd.Series(["s1", "s1", "s1"])
    taken_at = pd.Series(
        [
            pd.Timestamp("2024-01-01 08:00"),
            pd.Timestamp("2024-01-01 10:00"),
            pd.Timestamp("2024-01-01 14:00"),
        ]
    )
    order_end_at = pd.Series([pd.NaT, pd.NaT, pd.Timestamp("2024-01-01 16:00")])
    result = vasopressors.reconstruct_intervals_from_mar_events(
        icu_stay_id, taken_at, order_end_at
    )
    result = result.sort_values("started_at").reset_index(drop=True)
    assert result["ended_at"].iloc[0] == pd.Timestamp(
        "2024-01-01 10:00"
    )  # next event's start
    assert result["ended_at"].iloc[1] == pd.Timestamp(
        "2024-01-01 14:00"
    )  # next event's start
    # last event in the group has no "next" - falls back to its own order_end_at
    assert result["ended_at"].iloc[2] == pd.Timestamp("2024-01-01 16:00")


def test_reconstruct_intervals_from_mar_events_does_not_chain_across_stays():
    icu_stay_id = pd.Series(["s1", "s2"])
    taken_at = pd.Series(
        [pd.Timestamp("2024-01-01 08:00"), pd.Timestamp("2024-01-01 09:00")]
    )
    order_end_at = pd.Series(
        [pd.Timestamp("2024-01-01 08:30"), pd.Timestamp("2024-01-01 09:30")]
    )
    result = vasopressors.reconstruct_intervals_from_mar_events(
        icu_stay_id, taken_at, order_end_at
    )
    # each stay only has one event, so each one's "last event" fallback is its own order_end_at,
    # not the other stay's event time
    s1_row = result.loc[result["icu_stay_id"] == "s1"].iloc[0]
    s2_row = result.loc[result["icu_stay_id"] == "s2"].iloc[0]
    assert s1_row["ended_at"] == pd.Timestamp("2024-01-01 08:30")
    assert s2_row["ended_at"] == pd.Timestamp("2024-01-01 09:30")


def test_reconstruct_intervals_from_mar_events_leaves_ended_at_nat_with_no_next_event_and_no_order_end():
    icu_stay_id = pd.Series(["s1"])
    taken_at = pd.Series([pd.Timestamp("2024-01-01 08:00")])
    order_end_at = pd.Series([pd.NaT])
    result = vasopressors.reconstruct_intervals_from_mar_events(
        icu_stay_id, taken_at, order_end_at
    )
    assert pd.isna(result["ended_at"].iloc[0])


def test_reconstruct_intervals_from_mar_events_result_is_aligned_to_the_input_index():
    icu_stay_id = pd.Series(["s1", "s1"], index=[5, 2])
    taken_at = pd.Series(
        [pd.Timestamp("2024-01-01 10:00"), pd.Timestamp("2024-01-01 08:00")],
        index=[5, 2],
    )
    order_end_at = pd.Series([pd.Timestamp("2024-01-01 12:00"), pd.NaT], index=[5, 2])
    result = vasopressors.reconstruct_intervals_from_mar_events(
        icu_stay_id, taken_at, order_end_at
    )
    assert list(result.index) == [5, 2]
    # index 2 is chronologically first, so its "next event" is index 5's start
    assert result.loc[2, "ended_at"] == pd.Timestamp("2024-01-01 10:00")
    # index 5 is chronologically last, so it falls back to its own order_end_at
    assert result.loc[5, "ended_at"] == pd.Timestamp("2024-01-01 12:00")


def test_total_duration_per_stay():
    intervals = pd.DataFrame(
        [
            {
                "icu_stay_id": "s1",
                "started_at": pd.Timestamp("2024-01-01 08:00"),
                "ended_at": pd.Timestamp("2024-01-01 09:00"),
            },
            {
                "icu_stay_id": "s2",
                "started_at": pd.Timestamp("2024-01-01 08:00"),
                "ended_at": pd.Timestamp("2024-01-01 08:15"),
            },
        ]
    )
    result = vasopressors.total_duration(intervals)
    assert result.loc["s1"] == pd.Timedelta(hours=1)
    assert result.loc["s2"] == pd.Timedelta(minutes=15)
