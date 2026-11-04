import pandas as pd

from algo_vs_doc.scoring.sofa import neurology


def test_score_thresholds():
    gcs_total = pd.Series([15, 14, 13, 12, 11, 9, 8, 6, 5])
    result = neurology.score(gcs_total)
    assert result.tolist() == [0, 1, 1, 2, 2, 3, 3, 4, 4]


def test_flag_sedated_within_lookback_backward_only():
    gcs = pd.DataFrame(
        [
            {
                "icu_stay_id": "s1",
                "measured_at": pd.Timestamp("2024-01-01 12:00"),
            },  # 3h after sedative start
            {
                "icu_stay_id": "s1",
                "measured_at": pd.Timestamp("2024-01-01 20:00"),
            },  # 11h after -> outside 6h lookback
            {
                "icu_stay_id": "s1",
                "measured_at": pd.Timestamp("2024-01-01 08:00"),
            },  # 1h *before* sedative start
        ]
    )
    sedative_doses = pd.DataFrame(
        [
            {
                "icu_stay_id": "s1",
                "administered_at": pd.Timestamp("2024-01-01 09:00"),
                "infusion_rate": 5.0,
            }
        ]
    )
    result = neurology.flag_sedated(gcs, sedative_doses)
    result = result.sort_values("measured_at")
    assert result["sedated"].tolist() == [False, True, False]


def test_flag_sedated_ignores_doses_at_or_below_min_infusion_rate():
    gcs = pd.DataFrame(
        [{"icu_stay_id": "s1", "measured_at": pd.Timestamp("2024-01-01 09:30")}]
    )
    sedative_doses = pd.DataFrame(
        [
            {
                "icu_stay_id": "s1",
                "administered_at": pd.Timestamp("2024-01-01 09:00"),
                "infusion_rate": 0.1,
            }
        ]
    )
    result = neurology.flag_sedated(gcs, sedative_doses)
    assert result["sedated"].tolist() == [False]


def test_backfill_ventilated_verbal_nulls_and_backfills_unscoreable_reading():
    df = pd.DataFrame(
        [
            # ventilated, E4M6V1 -> unscoreable, should be backfilled from the later reading
            {
                "icu_stay_id": "s1",
                "measured_at": pd.Timestamp("2024-01-01 08:00"),
                "gcs_eye": 4,
                "gcs_motor": 6,
                "gcs_verbal": 1,
                "ventilated": True,
            },
            # later, non-ventilated (or otherwise scoreable) reading
            {
                "icu_stay_id": "s1",
                "measured_at": pd.Timestamp("2024-01-01 14:00"),
                "gcs_eye": 4,
                "gcs_motor": 6,
                "gcs_verbal": 5,
                "ventilated": False,
            },
        ]
    )
    result = neurology.backfill_ventilated_verbal(
        df["gcs_eye"],
        df["gcs_motor"],
        df["gcs_verbal"],
        df["ventilated"],
        df["icu_stay_id"],
        df["measured_at"],
    )
    assert result.tolist() == [5, 5]


def test_backfill_ventilated_verbal_leaves_genuine_verbal_1_when_not_e4m6():
    # verbal=1 but eye/motor are NOT both maxed -> genuinely impaired, not just unscoreable; leave as-is
    df = pd.DataFrame(
        [
            {
                "icu_stay_id": "s1",
                "measured_at": pd.Timestamp("2024-01-01 08:00"),
                "gcs_eye": 2,
                "gcs_motor": 6,
                "gcs_verbal": 1,
                "ventilated": True,
            }
        ]
    )
    result = neurology.backfill_ventilated_verbal(
        df["gcs_eye"],
        df["gcs_motor"],
        df["gcs_verbal"],
        df["ventilated"],
        df["icu_stay_id"],
        df["measured_at"],
    )
    assert result.tolist() == [1]


def test_apply_elective_surgery_day0_injects_synthetic_row_when_missing():
    scored = pd.DataFrame(
        [
            {
                "icu_stay_id": "s1",
                "day_index": 1,
                "sofa_neuro": 2,
                "gcs_total": 12,
                "gcs_eye": 3,
                "gcs_motor": 5,
                "gcs_verbal": 4,
            }
        ]
    )
    icu_stays = pd.DataFrame([{"icu_stay_id": "s1", "elective_surgery": True}])
    result = neurology.apply_elective_surgery_day0(scored, icu_stays)

    day0_rows = result[result["day_index"] == 0]
    assert len(day0_rows) == 1
    assert day0_rows["sofa_neuro"].iloc[0] == 0
    assert day0_rows["gcs_total"].iloc[0] == 15
    # day 1 is untouched
    day1_rows = result[result["day_index"] == 1]
    assert day1_rows["sofa_neuro"].iloc[0] == 2


def test_apply_elective_surgery_day0_overrides_existing_day0_reading():
    # a real day-0 reading implying impairment (sofa_neuro=3) still gets forced to 0 for an elective stay
    scored = pd.DataFrame(
        [
            {
                "icu_stay_id": "s1",
                "day_index": 0,
                "sofa_neuro": 3,
                "gcs_total": 8,
                "gcs_eye": 2,
                "gcs_motor": 4,
                "gcs_verbal": 2,
            }
        ]
    )
    icu_stays = pd.DataFrame([{"icu_stay_id": "s1", "elective_surgery": True}])
    result = neurology.apply_elective_surgery_day0(scored, icu_stays)
    assert len(result) == 1
    assert result["sofa_neuro"].iloc[0] == 0


def test_apply_elective_surgery_day0_does_not_affect_non_elective_stays():
    scored = pd.DataFrame(
        [
            {
                "icu_stay_id": "s1",
                "day_index": 0,
                "sofa_neuro": 3,
                "gcs_total": 8,
                "gcs_eye": 2,
                "gcs_motor": 4,
                "gcs_verbal": 2,
            }
        ]
    )
    icu_stays = pd.DataFrame([{"icu_stay_id": "s1", "elective_surgery": False}])
    result = neurology.apply_elective_surgery_day0(scored, icu_stays)
    assert len(result) == 1
    assert (
        result["sofa_neuro"].iloc[0] == 3
    )  # untouched, genuinely impaired reading stands


def test_flag_sedated_with_no_qualifying_doses_flags_everything_unsedated():
    # a stay (or a whole hospital extract) with no sedative infusions at all must not fail in
    # merge_asof - an empty frame's administered_at can carry an object dtype, which it rejects
    gcs = pd.DataFrame(
        [
            {
                "icu_stay_id": "s1",
                "measured_at": pd.Timestamp("2024-01-01 08:00"),
                "gcs_eye": 4,
            }
        ]
    )
    no_doses = pd.DataFrame(columns=["icu_stay_id", "administered_at", "infusion_rate"])
    result = neurology.flag_sedated(gcs, no_doses)
    assert result["sedated"].tolist() == [False]
    assert result["gcs_eye"].tolist() == [4]
