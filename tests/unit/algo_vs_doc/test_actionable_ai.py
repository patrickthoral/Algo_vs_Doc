import pandas as pd

from algo_vs_doc.actionable_ai import actionable_ai_svg


def _threshold_performance():
    return pd.DataFrame(
        {
            "cohort": ["Combined"] * 4,
            "n": [100] * 4,
            "prediction_source": ["Model at physician prediction"] * 4,
            "operating_point": [
                "5% risk threshold",
                "10% risk threshold",
                "20% risk threshold",
                "Youden index",
            ],
            "threshold": [0.05, 0.10, 0.20, 0.083],
            "flagged_n": [52, 19, 5, 27],
            "true_positive": [7, 4, 2, 6],
            "false_negative": [1, 4, 6, 2],
            "alert_rate": [0.52, 0.19, 0.05, 0.27],
            "sensitivity": [0.25, 0.375, 0.50, 0.75],
            "specificity": [0.50, 0.80, 0.90, 0.70],
        }
    )


def test_actionable_ai_svg_is_deterministic_and_contains_results(tmp_path):
    first = actionable_ai_svg(_threshold_performance(), tmp_path / "first.svg")
    second = actionable_ai_svg(_threshold_performance(), tmp_path / "second.svg")

    first_text = first.read_text(encoding="utf-8")
    assert first.read_bytes() == second.read_bytes()
    assert "10%" in first_text
    assert "8.3%" in first_text
    assert "Youden" in first_text
    assert "19 selected" in first_text
    assert "19 selected (" not in first_text
    assert "100" in first_text
    assert "ICU discharges" in first_text
    assert 'id="icu-bed-icon"' in first_text
    assert "4/8 events" in first_text
    assert "Se = 37.5%" in first_text
    assert "Sp = 80.0%" in first_text
    assert "Define Risk" in first_text
    assert "Threshold" in first_text
    assert "From Prediction to Action" in first_text
    assert "Clinical" in first_text
    assert "Reassessment" in first_text


def test_actionable_ai_svg_requires_all_three_thresholds(tmp_path):
    thresholds = _threshold_performance().iloc[:3]

    try:
        actionable_ai_svg(thresholds, tmp_path / "callout.svg")
    except ValueError as error:
        assert "5%, 10%, and 20%" in str(error)
        assert "Youden" in str(error)
    else:
        raise AssertionError("expected incomplete alert rates to fail")
