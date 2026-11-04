import pandas as pd

from algo_vs_doc.scoring import cohort_outcomes


def test_time_to_next_admission_for_a_readmitted_patient():
    patient_key = pd.Series(["p1", "p1"])
    icu_admitted_at = pd.Series(
        [pd.Timestamp("2024-01-01 08:00"), pd.Timestamp("2024-01-10 08:00")]
    )
    icu_discharged_at = pd.Series(
        [pd.Timestamp("2024-01-04 10:00"), pd.Timestamp("2024-01-14 10:00")]
    )
    result = cohort_outcomes.time_to_next_admission(
        patient_key, icu_admitted_at, icu_discharged_at
    )
    assert result["has_next_admission"].tolist() == [True, False]
    assert result["time_to_next_admission"].iloc[0] == pd.Timestamp(
        "2024-01-10 08:00"
    ) - pd.Timestamp("2024-01-04 10:00")
    assert pd.isna(result["time_to_next_admission"].iloc[1])


def test_time_to_next_admission_does_not_leak_across_patients():
    # two different patients' admissions must never be treated as a readmission of each other
    patient_key = pd.Series(["p1", "p2"])
    icu_admitted_at = pd.Series(
        [pd.Timestamp("2024-01-01 08:00"), pd.Timestamp("2024-01-02 08:00")]
    )
    icu_discharged_at = pd.Series(
        [pd.Timestamp("2024-01-01 10:00"), pd.Timestamp("2024-01-02 10:00")]
    )
    result = cohort_outcomes.time_to_next_admission(
        patient_key, icu_admitted_at, icu_discharged_at
    )
    assert result["has_next_admission"].tolist() == [False, False]


def test_time_to_next_admission_preserves_original_row_order_and_index():
    # rows are deliberately out of chronological order and use a non-default index
    patient_key = pd.Series(["p1", "p1"], index=[7, 3])
    icu_admitted_at = pd.Series(
        [pd.Timestamp("2024-01-10 08:00"), pd.Timestamp("2024-01-01 08:00")],
        index=[7, 3],
    )
    icu_discharged_at = pd.Series(
        [pd.Timestamp("2024-01-14 10:00"), pd.Timestamp("2024-01-04 10:00")],
        index=[7, 3],
    )
    result = cohort_outcomes.time_to_next_admission(
        patient_key, icu_admitted_at, icu_discharged_at
    )
    assert list(result.index) == [7, 3]
    # row 3 (the earlier admission) is the one with a next admission; row 7 (the later one) has none
    assert bool(result.loc[3, "has_next_admission"]) is True
    assert bool(result.loc[7, "has_next_admission"]) is False


def test_classify_outcome_true_if_any_event_within_horizon():
    time_to_death = pd.Series([pd.NaT, pd.Timedelta(days=8), pd.Timedelta(days=3)])
    time_to_icu_readmission = pd.Series([pd.Timedelta(days=5), pd.NaT, pd.NaT])
    result = cohort_outcomes.classify_outcome(time_to_death, time_to_icu_readmission)
    assert result.tolist() == [True, False, True]


def test_classify_outcome_false_when_nothing_happened():
    nat = pd.Series([pd.NaT], dtype="timedelta64[ns]")
    result = cohort_outcomes.classify_outcome(nat, nat)
    assert result.tolist() == [False]


def _stays(rows):
    """Rows of ``(icu_stay_id, discharged_at, died_at|None, time_to_icu_rd, time_to_hdu_rd)``.

    The two readmission columns are built with an explicit ``timedelta64`` dtype: a column of
    nothing but NaT is otherwise inferred as ``datetime64``, which cannot be compared against
    a horizon - the very pitfall ``classify_outcome``'s docstring documents.
    """
    frame = pd.DataFrame(
        [
            {
                "icu_stay_id": s,
                "icu_discharged_at": pd.Timestamp(disch),
                "died_at": pd.NaT if died is None else pd.Timestamp(died),
            }
            for s, disch, died, _, _ in rows
        ]
    )
    frame["time_to_icu_readmission"] = pd.Series(
        [r[3] for r in rows], dtype="timedelta64[ns]"
    )
    frame["time_to_hdu_readmission"] = pd.Series(
        [r[4] for r in rows], dtype="timedelta64[ns]"
    )
    return frame


def test_summarize_flags_each_event_within_the_horizon():
    stays = _stays(
        [
            ("s1", "2024-01-01", "2024-01-04", pd.NaT, pd.NaT),  # died 3d after
            ("s2", "2024-01-01", None, pd.Timedelta(days=2), pd.NaT),  # ICU readmission
            ("s3", "2024-01-01", None, pd.NaT, pd.Timedelta(days=5)),  # HDU readmission
            ("s4", "2024-01-01", None, pd.NaT, pd.NaT),  # nothing
        ]
    )
    result = cohort_outcomes.summarize(stays).set_index("icu_stay_id")
    assert result.loc["s1", "has_death_7days"] and result.loc["s1", "outcome"]
    assert result.loc["s2", "has_icu_readmission_7days"] and result.loc["s2", "outcome"]
    assert "has_hdu_readmission_7days" not in result
    assert not result.loc["s3", "outcome"]
    assert not result.loc["s4", "outcome"]
    assert result.loc["s1", "time_to_death"] == pd.Timedelta(days=3)


def test_summarize_does_not_count_an_event_beyond_the_horizon():
    stays = _stays(
        [("s1", "2024-01-01", "2024-01-20", pd.NaT, pd.NaT)]
    )  # died 19 days later
    result = cohort_outcomes.summarize(stays)
    assert not result["has_death_7days"].iloc[0]
    assert not result["outcome"].iloc[0]


def test_summarize_returns_the_outcome_column_set():
    stays = _stays([("s1", "2024-01-01", None, pd.NaT, pd.NaT)])
    result = cohort_outcomes.summarize(stays)
    assert list(result.columns) == ["icu_stay_id", *cohort_outcomes.OUTCOME_COLUMNS]


def test_exclude_elective_readmissions_reclassifies_only_planned_return():
    stays = pd.DataFrame(
        [
            {
                "icu_stay_id": "index",
                "patient_id": "p1",
                "is_study_cohort": True,
                "icu_admitted_at": pd.Timestamp("2024-01-01"),
                "icu_readmission_time": pd.Timestamp("2024-01-05"),
                "time_to_icu_readmission": 48.0,
                "elective_surgery": False,
                "has_death_7days": False,
                "has_icu_readmission_7days": True,
                "outcome": True,
            },
            {
                "icu_stay_id": "planned",
                "patient_id": "p1",
                "is_study_cohort": False,
                "icu_admitted_at": pd.Timestamp("2024-01-05"),
                "icu_readmission_time": pd.NaT,
                "time_to_icu_readmission": float("nan"),
                "elective_surgery": True,
                "has_death_7days": False,
                "has_icu_readmission_7days": False,
                "outcome": False,
            },
        ]
    )

    result = cohort_outcomes.exclude_elective_readmissions(stays)

    assert bool(result.loc[0, "has_elective_icu_readmission_7days"]) is True
    assert bool(result.loc[0, "has_icu_readmission_7days"]) is False
    assert bool(result.loc[0, "outcome"]) is False


def test_exclude_elective_readmissions_preserves_unlinked_observed_event():
    stays = pd.DataFrame(
        [
            {
                "icu_stay_id": "index",
                "patient_id": "p1",
                "is_study_cohort": True,
                "icu_admitted_at": pd.Timestamp("2024-01-01"),
                "icu_readmission_time": pd.Timestamp("2024-01-05"),
                "time_to_icu_readmission": 48.0,
                "elective_surgery": False,
                "has_death_7days": False,
                "has_icu_readmission_7days": True,
                "outcome": True,
            }
        ]
    )

    result = cohort_outcomes.exclude_elective_readmissions(stays)

    assert bool(result.loc[0, "readmission_stay_linked"]) is False
    assert pd.isna(result.loc[0, "readmission_elective_surgery"])
    assert bool(result.loc[0, "has_icu_readmission_7days"]) is True
    assert bool(result.loc[0, "outcome"]) is True
