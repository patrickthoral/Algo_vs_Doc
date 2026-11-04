import pandas as pd

from algo_vs_doc.scoring.sofa import liver


def test_score_thresholds():
    bilirubin = pd.Series([19, 20, 32, 32.1, 101, 101.1, 204, 204.1])
    result = liver.score(bilirubin)
    assert result.tolist() == [0, 1, 1, 2, 2, 3, 3, 4]


def test_score_coerces_non_numeric_to_nan_and_scores_zero():
    bilirubin = pd.Series(["not_a_number", "15"])
    result = liver.score(bilirubin)
    assert result.tolist() == [0, 0]
