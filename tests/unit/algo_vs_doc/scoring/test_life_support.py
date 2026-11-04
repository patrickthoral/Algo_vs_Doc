import pandas as pd

from algo_vs_doc.scoring import life_support


def test_compute_durations_spans_first_to_last_reading_per_category():
    events = pd.DataFrame(
        [
            {
                "icu_stay_id": "s1",
                "category": "ventilation",
                "measured_at": pd.Timestamp("2024-01-01 08:00"),
            },
            {
                "icu_stay_id": "s1",
                "category": "ventilation",
                "measured_at": pd.Timestamp("2024-01-01 12:00"),
            },
            {
                "icu_stay_id": "s1",
                "category": "ventilation",
                "measured_at": pd.Timestamp("2024-01-02 08:00"),
            },
            {
                "icu_stay_id": "s1",
                "category": "ecls",
                "measured_at": pd.Timestamp("2024-01-01 09:00"),
            },
            {
                "icu_stay_id": "s1",
                "category": "ecls",
                "measured_at": pd.Timestamp("2024-01-01 10:00"),
            },
        ]
    )
    result = life_support.compute_durations(events)
    row = result[result["icu_stay_id"] == "s1"].iloc[0]
    assert row["ventilation"] == pd.Timedelta(days=1)
    assert row["ecls"] == pd.Timedelta(hours=1)


def test_compute_durations_leaves_unmeasured_categories_as_nan_not_zero():
    events = pd.DataFrame(
        [
            {
                "icu_stay_id": "s1",
                "category": "ventilation",
                "measured_at": pd.Timestamp("2024-01-01 08:00"),
            }
        ]
    )
    result = life_support.compute_durations(events)
    assert (
        "ecls" not in result.columns
    )  # never appeared at all, distinct from "appeared with 0 duration"


def test_compute_durations_separates_stays():
    events = pd.DataFrame(
        [
            {
                "icu_stay_id": "s1",
                "category": "ventilation",
                "measured_at": pd.Timestamp("2024-01-01 08:00"),
            },
            {
                "icu_stay_id": "s1",
                "category": "ventilation",
                "measured_at": pd.Timestamp("2024-01-01 10:00"),
            },
            {
                "icu_stay_id": "s2",
                "category": "ventilation",
                "measured_at": pd.Timestamp("2024-01-01 08:00"),
            },
            {
                "icu_stay_id": "s2",
                "category": "ventilation",
                "measured_at": pd.Timestamp("2024-01-01 09:00"),
            },
        ]
    )
    result = life_support.compute_durations(events).set_index("icu_stay_id")
    assert result.loc["s1", "ventilation"] == pd.Timedelta(hours=2)
    assert result.loc["s2", "ventilation"] == pd.Timedelta(hours=1)


def test_combine_duration_sources_takes_the_max_for_overlapping_categories():
    numeric_source = pd.DataFrame(
        [{"icu_stay_id": "s1", "ventilation": pd.Timedelta(hours=2)}]
    )
    categorical_source = pd.DataFrame(
        [{"icu_stay_id": "s1", "ventilation": pd.Timedelta(hours=5)}]
    )
    result = life_support.combine_duration_sources(numeric_source, categorical_source)
    assert result.set_index("icu_stay_id").loc["s1", "ventilation"] == pd.Timedelta(
        hours=5
    )


def test_combine_duration_sources_keeps_categories_unique_to_one_source():
    numeric_source = pd.DataFrame(
        [{"icu_stay_id": "s1", "ecls": pd.Timedelta(hours=1)}]
    )
    categorical_source = pd.DataFrame(
        [{"icu_stay_id": "s1", "renal_replacement_therapy": pd.Timedelta(hours=3)}]
    )
    result = life_support.combine_duration_sources(numeric_source, categorical_source)
    row = result.set_index("icu_stay_id").loc["s1"]
    assert row["ecls"] == pd.Timedelta(hours=1)
    assert row["renal_replacement_therapy"] == pd.Timedelta(hours=3)


def test_presence_flags_does_not_leak_concepts_between_stays():
    device_exposure = pd.DataFrame(
        [
            {"icu_stay_id": "s1", "device_concept_name": "invasive_ventilation"},
            {"icu_stay_id": "s2", "device_concept_name": "renal_replacement_therapy"},
        ]
    )
    result = life_support.presence_flags(
        device_exposure, pd.Series(["s1", "s2"])
    ).set_index("icu_stay_id")
    assert bool(result.loc["s1", "invasive_ventilation"]) is True
    assert bool(result.loc["s1", "renal_replacement_therapy"]) is False
    assert bool(result.loc["s1", "ecls"]) is False
    assert bool(result.loc["s2", "invasive_ventilation"]) is False
    assert bool(result.loc["s2", "renal_replacement_therapy"]) is True
    assert bool(result.loc["s2", "ecls"]) is False


def test_presence_flags_gives_a_stay_with_no_support_an_all_false_row():
    device_exposure = pd.DataFrame(
        [{"icu_stay_id": "s1", "device_concept_name": "ecls"}]
    )
    result = life_support.presence_flags(
        device_exposure, pd.Series(["s1", "s2"])
    ).set_index("icu_stay_id")
    assert bool(result.loc["s2", "ecls"]) is False
    assert set(result.index) == {"s1", "s2"}


def test_presence_flags_derives_the_combined_ventilation_flag():
    device_exposure = pd.DataFrame(
        [{"icu_stay_id": "s1", "device_concept_name": "non_invasive_ventilation"}]
    )
    result = life_support.presence_flags(device_exposure, pd.Series(["s1"])).set_index(
        "icu_stay_id"
    )
    # ventilation rolls up either mode, so non-invasive alone still counts as ventilated
    assert bool(result.loc["s1", "ventilation"]) is True
    assert bool(result.loc["s1", "invasive_ventilation"]) is False


def test_presence_flags_counts_a_concept_arriving_from_more_than_one_source_once():
    device_exposure = pd.DataFrame(
        [
            {"icu_stay_id": "s1", "device_concept_name": "ecls"},
            {
                "icu_stay_id": "s1",
                "device_concept_name": "ecls",
            },  # a second source table's interval
        ]
    )
    result = life_support.presence_flags(device_exposure, pd.Series(["s1"]))
    assert len(result) == 1
    assert bool(result["ecls"].iloc[0]) is True


def test_vasopressor_inotrope_flags_uses_drug_exposure():
    drug_exposure = pd.DataFrame(
        [
            {"icu_stay_id": "s1", "drug_concept_name": "norepinephrine"},
            {"icu_stay_id": "s2", "drug_concept_name": "sedative"},
        ]
    )
    result = life_support.vasopressor_inotrope_flags(
        drug_exposure, pd.Series(["s1", "s2", "s3"])
    ).set_index("icu_stay_id")
    assert bool(result.loc["s1", "vasopressors_inotropes"]) is True
    assert bool(result.loc["s2", "vasopressors_inotropes"]) is False
    assert bool(result.loc["s3", "vasopressors_inotropes"]) is False
