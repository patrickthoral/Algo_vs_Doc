import pandas as pd
import pytest

from algo_vs_doc import specialties


@pytest.mark.parametrize(
    ("hospital", "source_value", "expected"),
    [
        ("LUMC", 1, "cardiothoracic_surgery"),
        ("LUMC", 6, "internal_medicine"),
        ("LUMC", 7, "internal_medicine"),
        ("LUMC", 11, "gastroenterology"),
        ("VUmc", 30303, "general_surgery"),
        ("VUmc", "030200", "otolaryngology"),
        ("VUmc", "110000", "other_surgery"),
        ("VUmc", 31353, "internal_medicine"),
        ("VUmc", 31354, "internal_medicine"),
        ("VUmc", 31355, "internal_medicine"),
        ("VUmc", 31358, "internal_medicine"),
        ("VUmc", 32900, "other_medical"),
        ("VUmc", 33085, "neurology"),
        ("VUmc", 841700, "unspecified"),
    ],
)
def test_categorize_maps_source_values(hospital, source_value, expected):
    result = specialties.categorize(pd.Series([source_value]), hospital=hospital)

    assert result.tolist() == [expected]


def test_categorize_maps_missing_source_value_to_unspecified():
    result = specialties.categorize(pd.Series([None]), hospital="LUMC")

    assert result.tolist() == ["unspecified"]


def test_categorize_rejects_unknown_source_value():
    with pytest.raises(KeyError, match="LUMC specialty value"):
        specialties.categorize(pd.Series([999]), hospital="LUMC")


def test_categorize_rejects_unknown_zero_padded_source_value():
    with pytest.raises(KeyError, match="999999"):
        specialties.categorize(pd.Series(["0999999"]), hospital="VUmc")


def test_labels_map_canonical_categories_to_baseline_labels():
    values = pd.Series(["cardiothoracic_surgery", "internal_medicine", None])

    assert specialties.labels(values).tolist()[:2] == [
        "Cardiothoracic Surgery",
        "Internal Medicine",
    ]
    assert pd.isna(specialties.labels(values).iloc[2])


def test_unknown_specialty_raises():
    with pytest.raises(KeyError, match="not covered by the mapping"):
        specialties.validate(pd.Series(["unknown_source_code"]))


def test_baseline_groups_cover_every_category_once():
    grouped = [
        specialty
        for values in specialties.BASELINE_GROUPS.values()
        for specialty in values
    ]

    assert sorted(grouped) == sorted(specialties.SPECIALTY_LABELS)


def test_baseline_counts_include_group_totals_and_labels():
    frame = pd.DataFrame(
        {
            "specialty": [
                "cardiothoracic_surgery",
                "general_surgery",
                "cardiology",
                None,
            ]
        }
    )

    result = specialties.baseline_counts(frame)

    assert result["Surgical"] == 2
    assert result["Cardiothoracic Surgery"] == 1
    assert result["Medical"] == 1
    assert result["Cardiology"] == 1
    assert result["Unspecified"] == 0
