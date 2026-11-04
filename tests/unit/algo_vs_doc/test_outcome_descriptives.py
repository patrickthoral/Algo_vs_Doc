import pandas as pd
import pytest

from algo_vs_doc import outcome_descriptives


def test_event_counts_reports_zero_count_event_types_and_percentages():
    frame = pd.DataFrame(
        {
            "post_discharge_status": [
                "no_event",
                "no_event",
                "icu_readmission",
                "readmission_and_death",
            ]
        }
    )

    result = outcome_descriptives.event_counts(frame).set_index("event_type")

    assert result.loc["no_event", "n"] == 2
    assert result.loc["icu_readmission", "percentage"] == 25.0
    assert "hdu_readmission" not in result.index


def test_event_counts_by_cohort_keeps_the_cohort_label():
    frame = pd.DataFrame({"post_discharge_status": ["no_event"]})

    result = outcome_descriptives.event_counts_by_cohort(
        {"LUMC": frame, "Combined": frame}
    )

    assert result["cohort"].drop_duplicates().tolist() == ["LUMC", "Combined"]


def test_readmission_pathways_excludes_death_without_readmission():
    frame = pd.DataFrame(
        {
            "post_discharge_status": [
                "icu_readmission",
                "readmission_and_death",
                "death",
            ],
            "discharge_specialty": ["cardiology", "neurology", "cardiology"],
            "readmission_origin_specialty": [
                "cardiology",
                None,
                "internal_medicine",
            ],
        }
    )

    result = outcome_descriptives.readmission_pathways(frame)

    assert result["n"].sum() == 2
    assert "unknown" in result["readmission_origin_specialty"].tolist()


def test_readmission_pathways_requires_mapped_specialties():
    with pytest.raises(KeyError, match="readmission pathways require"):
        outcome_descriptives.readmission_pathways(
            pd.DataFrame({"post_discharge_status": ["icu_readmission"]})
        )
