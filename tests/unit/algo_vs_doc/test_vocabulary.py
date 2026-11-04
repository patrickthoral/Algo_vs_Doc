import pandas as pd
import pytest
from algo_vs_doc import vocabulary


def test_map_strict_passes_nulls_through_and_maps_the_rest():
    result = vocabulary.map_strict(
        pd.Series([1, 2, None]), {1: "low", 2: "high"}, what="demo"
    )
    assert result.tolist()[:2] == ["low", "high"]
    assert pd.isna(result.iloc[2])


def test_map_strict_raises_naming_every_unknown_value_at_once():
    with pytest.raises(KeyError) as e:
        vocabulary.map_strict(pd.Series([1, 4, 7]), {1: "low"}, what="demo field")
    message = str(e.value)
    assert "4" in message and "7" in message and "demo field" in message


def test_the_error_reports_plain_values_not_numpy_reprs():
    with pytest.raises(KeyError) as e:
        vocabulary.map_strict(pd.Series([4], dtype="int64"), {1: "low"}, what="demo")
    assert "np.int64" not in str(e.value)


def test_a_long_known_vocabulary_is_truncated_so_the_bad_values_stay_readable():
    known = {f"feature_{i}": "x" for i in range(60)}
    with pytest.raises(KeyError) as e:
        vocabulary.map_strict(pd.Series(["nope"]), known, what="demo")
    message = str(e.value)
    assert "60 values in total" in message
    assert message.index("nope") < 60  # the offending value comes first
    assert len(message) < 900


def test_a_short_known_vocabulary_is_shown_in_full():
    with pytest.raises(KeyError) as e:
        vocabulary.map_strict(pd.Series([9]), {1: "low", 2: "high"}, what="demo")
    assert "[1, 2]" in str(e.value)


def test_raise_unmapped_does_nothing_when_everything_is_covered():
    vocabulary.raise_unmapped([], what="demo", known={1, 2})


def test_unmapped_values_ignores_nulls():
    assert vocabulary.unmapped_values(pd.Series([1, None, 5]), {1}) == [5]
