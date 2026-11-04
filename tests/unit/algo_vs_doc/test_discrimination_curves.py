import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from algo_vs_doc import plots


def test_precision_recall_curve_uses_average_precision_steps(monkeypatch):
    frame = pd.DataFrame(
        {
            "outcome": [True, False, True, False, False, True],
            "prediction": [0.0, 0.0, 0.5, 0.5, 1.0, 1.0],
        }
    )
    monkeypatch.setattr(plots, "_save_figure", lambda *args, **kwargs: None)

    plots.plot_combined_curves(
        frame,
        "outcome",
        {"Prediction": "prediction"},
        curve="prc",
        output_directory="unused",
    )

    axis = plt.gcf().axes[0]
    prediction_line = next(
        line for line in axis.lines if line.get_label().startswith("Prediction")
    )
    prevalence_line = next(
        line for line in axis.lines if line.get_label() == "Event prevalence"
    )
    assert prediction_line.get_drawstyle() == "steps-post"
    assert axis.get_xlabel() == "Recall (Sensitivity)"
    assert axis.get_ylabel() == "Precision (Positive Predictive Value)"
    np.testing.assert_allclose(prevalence_line.get_ydata(), [0.5, 0.5])
    assert len(axis.collections) == 1
    plt.close(axis.figure)


def test_roc_curve_remains_linearly_interpolated(monkeypatch):
    frame = pd.DataFrame(
        {
            "outcome": [True, False, True, False],
            "prediction": [0.9, 0.2, 0.7, 0.1],
        }
    )
    monkeypatch.setattr(plots, "_save_figure", lambda *args, **kwargs: None)

    plots.plot_combined_curves(
        frame,
        "outcome",
        {"Prediction": "prediction"},
        curve="roc",
        output_directory="unused",
    )

    axis = plt.gcf().axes[0]
    prediction_line = next(
        line for line in axis.lines if line.get_label().startswith("Prediction")
    )
    assert prediction_line.get_drawstyle() == "default"
    assert axis.get_xlabel() == "False Positive Rate (1 - Specificity)"
    assert axis.get_ylabel() == "True Positive Rate (Sensitivity)"
    plt.close(axis.figure)
