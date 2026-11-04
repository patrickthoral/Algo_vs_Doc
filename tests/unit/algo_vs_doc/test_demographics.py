import pandas as pd
import pytest

from algo_vs_doc import demographics


@pytest.mark.parametrize(
    ("hospital", "source_values", "expected"),
    [
        ("LUMC", ["female", "male"], ["female", "male"]),
        ("VUmc", ["V", "M"], ["female", "male"]),
    ],
)
def test_categorize_sex_maps_source_values(hospital, source_values, expected):
    result = demographics.categorize_sex(pd.Series(source_values), hospital=hospital)

    assert result.tolist() == expected


def test_categorize_sex_preserves_missing_values():
    result = demographics.categorize_sex(pd.Series([None]), hospital="VUmc")

    assert pd.isna(result.iloc[0])


def test_categorize_sex_rejects_unknown_source_value():
    with pytest.raises(KeyError, match="VUmc sex value"):
        demographics.categorize_sex(pd.Series(["X"]), hospital="VUmc")
