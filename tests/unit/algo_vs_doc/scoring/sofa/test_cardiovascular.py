import pandas as pd

from algo_vs_doc.scoring.sofa import cardiovascular


def test_score_map_thresholds():
    map_value = pd.Series([70, 69, 40])
    result = cardiovascular.score_map(map_value)
    assert result.tolist() == [0, 1, 1]


def test_active_doses_excludes_paused_and_near_zero_infusion_rates():
    doses = pd.DataFrame(
        [
            {"icu_stay_id": "s1", "infusion_rate": 0.0},  # temporarily stopped
            {
                "icu_stay_id": "s1",
                "infusion_rate": 0.1,
            },  # at the threshold - still excluded
            {
                "icu_stay_id": "s1",
                "infusion_rate": 0.11,
            },  # just above the threshold - included to pin
            # down the boundary precisely; not a realistic rate on its own - real adult-ICU
            # vasoactive infusion rates are typically well above 0.5 mL/hour
            {"icu_stay_id": "s1", "infusion_rate": 5.0},
        ]
    )
    result = cardiovascular.active_doses(doses)
    assert result["infusion_rate"].tolist() == [0.11, 5.0]


def test_score_drugs_dobutamine_any_dose_scores_2():
    # epinephrine/norepinephrine must be NaN ("not given"), not 0 - see docstring: a literal 0
    # would incorrectly satisfy the "<=0.1" category-3 condition.
    not_given = pd.Series([float("nan")])
    result = cardiovascular.score_drugs(
        dopamine=not_given,
        epinephrine=not_given,
        norepinephrine=not_given,
        dobutamine_present=pd.Series([True]),
    )
    assert result.tolist() == [2]


def test_score_drugs_dopamine_categories():
    dopamine = pd.Series([5.0, 5.1, 15.0, 15.1])
    not_given = pd.Series([float("nan")] * 4)
    result = cardiovascular.score_drugs(
        dopamine, not_given, not_given, pd.Series([False] * 4)
    )
    assert result.tolist() == [2, 3, 3, 4]


def test_score_drugs_epinephrine_and_norepinephrine_categories():
    epinephrine = pd.Series([0.1, 0.11])
    not_given = pd.Series([float("nan"), float("nan")])
    falses = pd.Series([False, False])
    result = cardiovascular.score_drugs(not_given, epinephrine, not_given, falses)
    assert result.tolist() == [3, 4]


def test_score_drugs_normalizes_non_positive_doses_to_not_given():
    # defensive floor: a literal 0 (paused infusion) or an impossible negative (data error)
    # must not inflate the score, even if the caller forgot to run active_doses first.
    non_positive = pd.Series([0.0, -1.0])
    not_given = pd.Series([float("nan"), float("nan")])
    falses = pd.Series([False, False])
    result = cardiovascular.score_drugs(not_given, non_positive, not_given, falses)
    assert result.tolist() == [0, 0]


def test_score_drugs_does_not_guard_a_near_zero_but_positive_dose():
    # the defensive floor only catches <=0 - a near-zero-but-positive rate (e.g. a paused
    # infusion recorded as 0.05 rather than exactly 0) still satisfies "<=0.1" and scores
    # category 3. active_doses' 0.1 threshold is what actually excludes this case - this
    # test documents that score_drugs alone is not a substitute for running it first.
    near_zero = pd.Series([0.05])
    not_given = pd.Series([float("nan")])
    result = cardiovascular.score_drugs(
        not_given, near_zero, not_given, pd.Series([False])
    )
    assert result.tolist() == [3]


def test_sustained_dose_windows_drops_blocks_shorter_than_min_duration():
    intervals = pd.DataFrame(
        [
            {
                "icu_stay_id": "s1",
                "started_at": pd.Timestamp("2024-01-01 08:00"),
                "ended_at": pd.Timestamp(
                    "2024-01-01 08:45"
                ),  # 45 min, below default 1h
                "dose": 10.0,
            }
        ]
    )
    result = cardiovascular.sustained_dose_windows(intervals, threshold=5.0)
    assert result.empty


def test_sustained_dose_windows_merges_back_to_back_segments_above_threshold():
    intervals = pd.DataFrame(
        [
            {
                "icu_stay_id": "s1",
                "started_at": pd.Timestamp("2024-01-01 08:00"),
                "ended_at": pd.Timestamp("2024-01-01 08:30"),
                "dose": 6.0,
            },
            {
                "icu_stay_id": "s1",
                "started_at": pd.Timestamp(
                    "2024-01-01 08:30"
                ),  # touches the previous segment's end
                "ended_at": pd.Timestamp("2024-01-01 09:05"),
                "dose": 8.0,
            },
        ]
    )
    result = cardiovascular.sustained_dose_windows(intervals, threshold=5.0)
    assert len(result) == 1
    assert result["window_started_at"].iloc[0] == pd.Timestamp("2024-01-01 08:00")
    assert result["window_ended_at"].iloc[0] == pd.Timestamp("2024-01-01 09:05")


def test_sustained_dose_windows_a_dip_below_threshold_resets_the_clock():
    intervals = pd.DataFrame(
        [
            {
                "icu_stay_id": "s1",
                "started_at": pd.Timestamp("2024-01-01 08:00"),
                "ended_at": pd.Timestamp("2024-01-01 08:40"),
                "dose": 10.0,  # above threshold, 40 min
            },
            {
                "icu_stay_id": "s1",
                "started_at": pd.Timestamp("2024-01-01 08:40"),
                "ended_at": pd.Timestamp("2024-01-01 08:50"),
                "dose": 3.0,  # dips below threshold - not in `intervals` passed to the function at all
            },
            {
                "icu_stay_id": "s1",
                "started_at": pd.Timestamp("2024-01-01 08:50"),
                "ended_at": pd.Timestamp("2024-01-01 09:30"),
                "dose": 10.0,  # back above threshold, another 40 min - still short of 1h on its own
            },
        ]
    )
    # only the >=threshold rows are relevant input in practice; simulate the caller having already
    # filtered to qualifying doses by only passing the two above-threshold segments with a real gap
    above_threshold_only = intervals[intervals["dose"] >= 5.0]
    result = cardiovascular.sustained_dose_windows(above_threshold_only, threshold=5.0)
    assert (
        result.empty
    )  # neither 40-minute block alone reaches 1h, and they are not contiguous


def test_gate_dose_tiers_scores_at_the_previous_tier_until_the_higher_threshold_itself_has_held_1h():
    # norepinephrine: >0 for 90min (tier 3 threshold, satisfied), but only >0.1 (tier 4 threshold)
    # for the last 10min - must score tier 3, not tier 4, until the 0.1 threshold itself holds 1h
    dose_intervals = pd.DataFrame(
        [
            {
                "icu_stay_id": "s1",
                "started_at": pd.Timestamp("2024-01-01 08:00"),
                "ended_at": pd.Timestamp("2024-01-01 09:20"),
                "dose": 0.05,  # sustained >0 for 80min
            },
            {
                "icu_stay_id": "s1",
                "started_at": pd.Timestamp("2024-01-01 09:20"),
                "ended_at": pd.Timestamp("2024-01-01 09:30"),
                "dose": 0.2,  # just risen above 0.1, only 10min ago
            },
        ]
    )
    icu_stay_id = pd.Series(["s1"])
    measured_at = pd.Series([pd.Timestamp("2024-01-01 09:30")])
    result = cardiovascular.gate_dose_tiers(
        dose_intervals, "norepinephrine", icu_stay_id, measured_at
    )
    assert result.tolist() == [3]


def test_gate_dose_tiers_reaches_the_highest_tier_once_its_own_threshold_has_held_1h():
    dose_intervals = pd.DataFrame(
        [
            {
                "icu_stay_id": "s1",
                "started_at": pd.Timestamp("2024-01-01 08:00"),
                "ended_at": pd.Timestamp("2024-01-01 09:30"),
                "dose": 0.2,  # >0.1 the whole time, 90min
            }
        ]
    )
    icu_stay_id = pd.Series(["s1"])
    measured_at = pd.Series([pd.Timestamp("2024-01-01 09:30")])
    result = cardiovascular.gate_dose_tiers(
        dose_intervals, "norepinephrine", icu_stay_id, measured_at
    )
    assert result.tolist() == [4]


def test_gate_dose_tiers_a_brief_dose_that_never_reaches_1h_contributes_nothing():
    dose_intervals = pd.DataFrame(
        [
            {
                "icu_stay_id": "s1",
                "started_at": pd.Timestamp("2024-01-01 08:00"),
                "ended_at": pd.Timestamp("2024-01-01 08:20"),  # only 20min
                "dose": 5.0,
            }
        ]
    )
    icu_stay_id = pd.Series(["s1"])
    measured_at = pd.Series([pd.Timestamp("2024-01-01 08:20")])
    result = cardiovascular.gate_dose_tiers(
        dose_intervals, "dopamine", icu_stay_id, measured_at
    )
    assert result.isna().all()


def test_gate_dose_tiers_ignores_non_positive_doses_like_score_drugs_does():
    dose_intervals = pd.DataFrame(
        [
            {
                "icu_stay_id": "s1",
                "started_at": pd.Timestamp("2024-01-01 08:00"),
                "ended_at": pd.Timestamp("2024-01-01 10:00"),
                "dose": 0.0,  # paused - must not satisfy the ">=0.0" lowest-tier threshold
            }
        ]
    )
    icu_stay_id = pd.Series(["s1"])
    measured_at = pd.Series([pd.Timestamp("2024-01-01 10:00")])
    result = cardiovascular.gate_dose_tiers(
        dose_intervals, "dopamine", icu_stay_id, measured_at
    )
    assert result.isna().all()


def test_score_combines_map_and_gated_drug_contributions():
    map_measurement = pd.DataFrame(
        [
            {
                "icu_stay_id": "s1",
                "measured_at": pd.Timestamp("2024-01-01 08:00"),
                "value_as_number": 65.0,
            },
        ]
    )
    drug_exposure = pd.DataFrame(
        [
            # sustained norepinephrine >0.1 for 90min -> tier 4 at this reading
            {
                "icu_stay_id": "s1",
                "drug_concept_name": "norepinephrine",
                "drug_exposure_start_datetime": pd.Timestamp("2024-01-01 07:00"),
                "drug_exposure_end_datetime": pd.Timestamp("2024-01-01 08:30"),
                "dose_value": 0.2,
            },
            # a brief dopamine burst that never reaches 1h -> contributes nothing
            {
                "icu_stay_id": "s2",
                "drug_concept_name": "dopamine",
                "drug_exposure_start_datetime": pd.Timestamp("2024-01-01 08:00"),
                "drug_exposure_end_datetime": pd.Timestamp("2024-01-01 08:20"),
                "dose_value": 3.0,
            },
        ]
    )
    result = cardiovascular.score(map_measurement, drug_exposure)
    # MAP contributes one row (score 1, since 65 < 70)
    map_row = result[
        (result["icu_stay_id"] == "s1")
        & (result["measured_at"] == pd.Timestamp("2024-01-01 08:00"))
    ]
    assert map_row["sofa_cardvasc"].iloc[0] == 1
    # the sustained norepinephrine reading contributes tier 4
    norepi_row = result[result["measured_at"] == pd.Timestamp("2024-01-01 07:00")]
    assert norepi_row["sofa_cardvasc"].iloc[0] == 4
    # the brief, never-sustained dopamine burst contributes no row at all
    assert not (result["icu_stay_id"] == "s2").any()


def test_score_handles_a_drug_never_appearing_in_drug_exposure_at_all():
    map_measurement = pd.DataFrame(
        [
            {
                "icu_stay_id": "s1",
                "measured_at": pd.Timestamp("2024-01-01 08:00"),
                "value_as_number": 65.0,
            }
        ]
    )
    drug_exposure = pd.DataFrame(
        columns=[
            "icu_stay_id",
            "drug_concept_name",
            "drug_exposure_start_datetime",
            "drug_exposure_end_datetime",
            "dose_value",
        ]
    )
    result = cardiovascular.score(map_measurement, drug_exposure)
    assert len(result) == 1
    assert result["sofa_cardvasc"].iloc[0] == 1


def test_at_discharge_takes_the_max_across_independently_nearest_sources():
    # MAP's last reading is recent-but-mild (score 0); a drug's last reading is earlier-but-severe
    # (score 4) - the discharge value must reflect the drug's severity, not just whichever source's
    # own reading happens to be closer to discharge in absolute time.
    icu_stays = pd.DataFrame(
        [{"icu_stay_id": "s1", "icu_discharged_at": pd.Timestamp("2024-01-02 00:00")}]
    )
    scored = pd.DataFrame(
        [
            {
                "icu_stay_id": "s1",
                "measured_at": pd.Timestamp("2024-01-01 23:00"),  # 1h before discharge
                "sofa_cardvasc": 0,
                "source": "map",
            },
            {
                "icu_stay_id": "s1",
                "measured_at": pd.Timestamp("2024-01-01 12:00"),  # 12h before discharge
                "sofa_cardvasc": 4,
                "source": "norepinephrine",
            },
        ]
    )
    result = cardiovascular.at_discharge(scored, icu_stays)
    assert result.set_index("icu_stay_id").loc["s1", "sofa_cardvasc"] == 4


def test_at_discharge_covers_every_stay_even_with_no_contributing_reading():
    icu_stays = pd.DataFrame(
        [
            {
                "icu_stay_id": "s1",
                "icu_discharged_at": pd.Timestamp("2024-01-02 00:00"),
            },
            {
                "icu_stay_id": "s2",
                "icu_discharged_at": pd.Timestamp("2024-01-03 00:00"),
            },
        ]
    )
    scored = pd.DataFrame(
        [
            {
                "icu_stay_id": "s1",
                "measured_at": pd.Timestamp("2024-01-01 23:00"),
                "sofa_cardvasc": 1,
                "source": "map",
            }
        ]
    )
    result = cardiovascular.at_discharge(scored, icu_stays).set_index("icu_stay_id")
    assert result.loc["s1", "sofa_cardvasc"] == 1
    assert pd.isna(result.loc["s2", "sofa_cardvasc"])


def test_at_discharge_never_picks_a_post_discharge_reading():
    icu_stays = pd.DataFrame(
        [{"icu_stay_id": "s1", "icu_discharged_at": pd.Timestamp("2024-01-02 00:00")}]
    )
    scored = pd.DataFrame(
        [
            {
                "icu_stay_id": "s1",
                "measured_at": pd.Timestamp("2024-01-02 06:00"),  # after discharge
                "sofa_cardvasc": 4,
                "source": "map",
            }
        ]
    )
    result = cardiovascular.at_discharge(scored, icu_stays)
    assert pd.isna(result.set_index("icu_stay_id").loc["s1", "sofa_cardvasc"])


def test_covered_by_windows():
    icu_stay_id = pd.Series(["s1", "s1", "s2"])
    measured_at = pd.Series(
        [
            pd.Timestamp("2024-01-01 08:30"),  # inside the s1 window
            pd.Timestamp("2024-01-01 20:00"),  # outside the s1 window
            pd.Timestamp("2024-01-01 08:30"),  # s2 has no window at all
        ]
    )
    windows = pd.DataFrame(
        [
            {
                "icu_stay_id": "s1",
                "window_started_at": pd.Timestamp("2024-01-01 08:00"),
                "window_ended_at": pd.Timestamp("2024-01-01 09:05"),
            }
        ]
    )
    result = cardiovascular.covered_by_windows(icu_stay_id, measured_at, windows)
    assert result.tolist() == [True, False, False]
