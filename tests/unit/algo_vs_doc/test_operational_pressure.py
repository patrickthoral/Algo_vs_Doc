import pandas as pd

from algo_vs_doc import operational_pressure


def test_other_active_stays_uses_half_open_intervals_and_excludes_own_stay():
    admitted = pd.Series(
        pd.to_datetime(
            [
                "2024-01-01 08:00",
                "2024-01-01 09:00",
                "2024-01-01 10:00",
            ]
        )
    )
    discharged = pd.Series(
        pd.to_datetime(
            [
                "2024-01-01 12:00",
                "2024-01-01 11:00",
                "2024-01-01 14:00",
            ]
        )
    )
    evaluated = pd.Series(
        pd.to_datetime(
            [
                "2024-01-01 10:30",
                "2024-01-01 11:00",
                "2024-01-01 12:00",
            ]
        )
    )

    result = operational_pressure.other_active_stays(admitted, discharged, evaluated)

    assert result.tolist() == [2, 2, 0]


def test_other_active_stays_keeps_a_missing_evaluation_unknown():
    result = operational_pressure.other_active_stays(
        pd.Series(pd.to_datetime(["2024-01-01 08:00"])),
        pd.Series(pd.to_datetime(["2024-01-01 12:00"])),
        pd.Series([pd.NaT]),
    )

    assert pd.isna(result.iloc[0])


def test_other_same_day_events_excludes_the_aligned_rows_own_event():
    occurred = pd.Series(
        pd.to_datetime(
            [
                "2024-01-01 10:00",
                "2024-01-01 12:00",
                "2024-01-02 09:00",
            ]
        )
    )
    is_event = pd.Series([True, True, False])

    result = operational_pressure.other_same_day_events(occurred, is_event, occurred)

    assert result.tolist() == [1, 1, 0]


def test_same_day_events_use_amsterdam_dates_for_aware_timestamps():
    occurred = pd.Series(
        pd.to_datetime(
            ["2024-01-01 23:30+00:00", "2024-01-02 00:30+00:00"],
            utc=True,
        )
    )

    result = operational_pressure.other_same_day_events(
        occurred, pd.Series([True, True]), occurred
    )

    assert result.tolist() == [1, 1]


def test_add_operational_pressure_names_the_proxies_explicitly():
    frame = pd.DataFrame(
        {
            "icu_admitted_at": pd.to_datetime(["2024-01-01 08:00", "2024-01-01 09:00"]),
            "icu_discharged_at": pd.to_datetime(
                ["2024-01-01 12:00", "2024-01-01 13:00"]
            ),
            "doc_predicted_at": pd.to_datetime(
                ["2024-01-01 10:00", "2024-01-01 11:00"]
            ),
            "discharge_destination": ["external_transfer", "ward"],
            "elective_surgery": [True, False],
        }
    )

    result = operational_pressure.add_operational_pressure(frame)

    assert result["n_census_at_discharge"].tolist() == [1, 0]
    assert result["n_external_transfers_same_day"].tolist() == [0, 1]
    assert result["n_elective_surgery_admissions_same_day"].tolist() == [1, 1]
    assert "n_census_at_discharge" not in frame
