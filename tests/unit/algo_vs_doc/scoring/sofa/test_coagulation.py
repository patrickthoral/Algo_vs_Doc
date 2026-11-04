import pandas as pd

from algo_vs_doc.scoring.sofa import coagulation


def test_score_thresholds():
    platelets = pd.Series([150, 149, 100, 99, 50, 49, 20, 19])
    result = coagulation.score(platelets)
    assert result.tolist() == [0, 1, 1, 2, 2, 3, 3, 4]
