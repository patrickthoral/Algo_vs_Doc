import numpy as np
import pandas as pd
import pytest

from algo_vs_doc import plots


def test_matrix_orders_agents_orients_differences_and_marks_primary(monkeypatch):
    agents = [
        "Model at Physician Prediction",
        "Physician (Percentage)",
        "Physician (Category)",
        "Combined (Percentage)",
        "Combined (Category)",
    ]
    contrasts = []
    samples = []
    contrast_id = 0
    for left in range(len(agents) - 1):
        for right in range(left + 1, len(agents)):
            agent_1, agent_2 = (
                (agents[right], agents[left])
                if contrast_id % 2
                else (agents[left], agents[right])
            )
            contrasts.append(
                {
                    "contrast_id": contrast_id,
                    "agent_1": agent_1,
                    "agent_2": agent_2,
                    "difference_agent_1_minus_agent_2": 0.1,
                    "confidence_interval_lower": 0.05,
                    "confidence_interval_upper": 0.15,
                    "p": 0.2,
                    "p_holm": 0.3,
                    "family": "exploratory",
                    "metric": "auroc",
                }
            )
            samples.extend(
                {
                    "contrast_id": contrast_id,
                    "bootstrap_difference_agent_1_minus_agent_2": value,
                }
                for value in (0.08, 0.12)
            )
            contrast_id += 1

    captured = {}

    def capture(comparisons, **kwargs):
        captured["comparisons"] = comparisons
        captured.update(kwargs)
        return captured

    monkeypatch.setattr(plots, "combined_auc_difference_plot", capture)
    result = plots.performance_contrast_matrix_plot(
        pd.DataFrame(contrasts), pd.DataFrame(samples), output_directory="unused"
    )

    model = "Model"
    physician_percentage = "Physician (Percentage)"
    physician_category = "Physician (Category)"
    combined_percentage = "Combined Model and Physician (Percentage)"
    combined_category = "Combined Model and Physician (Category)"
    assert result["agent_order"] == [
        model,
        physician_percentage,
        physician_category,
        combined_percentage,
        combined_category,
    ]
    assert result["agent_colors"][model] == "tab:green"
    assert result["agent_colors"][physician_percentage] == "tab:blue"
    assert result["agent_colors"][combined_percentage] == "tab:red"
    assert sum(len(values) for values in result["comparisons"].values()) == 10

    reversed_cell = result["comparisons"][model][physician_category]
    assert reversed_cell[0:3] == (-0.1, -0.15, -0.05)
    np.testing.assert_allclose(reversed_cell[4], [-0.08, -0.12])
    assert reversed_cell[7] == "primary"


def test_matrix_marks_model_physician_auprc_comparisons_secondary(monkeypatch):
    agents = [
        "Model at Physician Prediction",
        "Physician (Percentage)",
        "Physician (Category)",
        "Combined (Percentage)",
        "Combined (Category)",
    ]
    contrasts = []
    samples = []
    contrast_id = 0
    for left in range(len(agents) - 1):
        for right in range(left + 1, len(agents)):
            contrasts.append(
                {
                    "contrast_id": contrast_id,
                    "agent_1": agents[left],
                    "agent_2": agents[right],
                    "difference_agent_1_minus_agent_2": 0.1,
                    "confidence_interval_lower": 0.05,
                    "confidence_interval_upper": 0.15,
                    "p": 0.2,
                    "p_holm": 0.3,
                    "metric": "auprc",
                }
            )
            samples.extend(
                {
                    "contrast_id": contrast_id,
                    "bootstrap_difference_agent_1_minus_agent_2": value,
                }
                for value in (0.08, 0.12)
            )
            contrast_id += 1

    monkeypatch.setattr(
        plots,
        "combined_auc_difference_plot",
        lambda comparisons, **kwargs: {"comparisons": comparisons, **kwargs},
    )
    result = plots.performance_contrast_matrix_plot(
        pd.DataFrame(contrasts), pd.DataFrame(samples), output_directory="unused"
    )

    assert result["comparisons"]["Model"]["Physician (Percentage)"][7] == ("secondary")
    assert result["comparisons"]["Model"]["Physician (Category)"][7] == ("secondary")


def test_bootstrap_cell_places_family_in_title_not_statistics_box():
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots()
    plots.plot_bootstrap_auc_difference(
        axis,
        observed_difference=0.1,
        confidence_lower=0.05,
        confidence_upper=0.15,
        p_value=0.2,
        bootstrapped_differences=np.array([0.08, 0.12]),
        curve="roc",
        adjusted_p_value=0.3,
        comparison_family="primary",
    )

    assert axis.get_title(loc="left") == "Primary"
    statistics_texts = [item.get_text() for item in axis.texts]
    assert all("Primary" not in item for item in statistics_texts)
    assert any("$p$ = 0.300" in item for item in statistics_texts)
    plt.close(fig)


def test_forest_preserves_stored_difference_and_agent_direction(monkeypatch):
    import matplotlib.pyplot as plt

    contrasts = pd.DataFrame(
        [
            {
                "agent_1": "Model at Physician Prediction",
                "agent_2": "Physician (Percentage)",
                "difference_agent_1_minus_agent_2": 0.2,
                "confidence_interval_lower": 0.1,
                "confidence_interval_upper": 0.3,
                "metric": "auroc",
            }
        ]
    )
    monkeypatch.setattr(plots, "_save_figure", lambda *args, **kwargs: None)

    fig = plots.performance_contrast_forest_plot(contrasts, output_directory="unused")
    axis = fig.axes[0]
    marker = next(line for line in axis.lines if line.get_marker() == "o")
    interval = next(
        collection.get_segments()[0]
        for collection in axis.collections
        if collection.get_segments()
    )
    agent_text = {text.get_text(): text.get_position()[0] for text in axis.texts}

    np.testing.assert_allclose(np.asarray(marker.get_xdata(), dtype=float), [0.2])
    np.testing.assert_allclose(interval[:, 0], [0.1, 0.3])
    assert agent_text["Physician (Percentage)"] < 0
    assert agent_text["Model"] > 1
    assert "Model vs physician" in agent_text
    assert axis.get_xlabel() == "$\\Delta$AUROC (right agent - left agent)"
    plt.close(fig)


def test_forest_shows_adjusted_p_only_for_formal_comparisons(monkeypatch):
    import matplotlib.pyplot as plt

    contrasts = pd.DataFrame(
        [
            {
                "agent_1": "Model at Physician Prediction",
                "agent_2": "Physician (Percentage)",
                "difference_agent_1_minus_agent_2": 0.2,
                "confidence_interval_lower": 0.1,
                "confidence_interval_upper": 0.3,
                "metric": "auroc",
                "family": "primary",
                "p_holm": 0.014,
            },
            {
                "agent_1": "Combined (Category)",
                "agent_2": "Model at Physician Prediction",
                "difference_agent_1_minus_agent_2": 0.1,
                "confidence_interval_lower": -0.1,
                "confidence_interval_upper": 0.2,
                "metric": "auroc",
                "family": "exploratory",
                "p_holm": 0.002,
            },
        ]
    )
    monkeypatch.setattr(plots, "_save_figure", lambda *args, **kwargs: None)

    fig = plots.performance_contrast_forest_plot(
        contrasts,
        output_directory="unused",
        formal_comparisons=True,
        show_adjusted_p_values=True,
    )
    labels = [item.get_text() for item in fig.axes[0].texts]

    assert "Holm-adjusted $p$ = 0.014" in labels
    assert "Holm-adjusted $p$ = 0.002" not in labels
    assert "Primary" in labels
    assert "Exploratory" in labels
    adjusted_label = next(
        item
        for item in fig.axes[0].texts
        if item.get_text().startswith("Holm-adjusted")
    )
    assert adjusted_label.get_horizontalalignment() == "left"
    assert adjusted_label.get_bbox_patch().get_facecolor()[-1] > 0
    plt.close(fig)


@pytest.mark.parametrize(
    ("agent_1", "agent_2", "expected_left", "expected_right"),
    [
        (
            "Physician (Category)",
            "Physician (Percentage)",
            "Physician (Percentage)",
            "Physician (Category)",
        ),
        (
            "Physician (Percentage)",
            "Combined (Percentage)",
            "Combined (Percentage)",
            "Physician (Percentage)",
        ),
        (
            "Combined (Percentage)",
            "Combined (Category)",
            "Combined (Category)",
            "Combined (Percentage)",
        ),
    ],
)
def test_forest_uses_requested_agent_sides(
    monkeypatch, agent_1, agent_2, expected_left, expected_right
):
    import matplotlib.pyplot as plt

    contrasts = pd.DataFrame(
        [
            {
                "agent_1": agent_1,
                "agent_2": agent_2,
                "difference_agent_1_minus_agent_2": 0.2,
                "confidence_interval_lower": 0.1,
                "confidence_interval_upper": 0.3,
                "metric": "auroc",
            }
        ]
    )
    monkeypatch.setattr(plots, "_save_figure", lambda *args, **kwargs: None)

    fig = plots.performance_contrast_forest_plot(contrasts, output_directory="unused")
    axis = fig.axes[0]
    positions = {item.get_text(): item.get_position()[0] for item in axis.texts}

    assert (
        positions[
            expected_left.replace("Model at Physician Prediction", "Model").replace(
                "Combined (", "Combined Model and\nPhysician ("
            )
        ]
        < 0
    )
    assert (
        positions[
            expected_right.replace("Model at Physician Prediction", "Model").replace(
                "Combined (", "Combined Model and\nPhysician ("
            )
        ]
        > 1
    )
    plt.close(fig)


def test_forest_preserves_interval_that_excludes_estimate(monkeypatch):
    import matplotlib.pyplot as plt

    frame = pd.DataFrame(
        [
            {
                "agent_1": "Model at Physician Prediction",
                "agent_2": "Physician (Percentage)",
                "difference_agent_1_minus_agent_2": 0.4,
                "confidence_interval_lower": 0.1,
                "confidence_interval_upper": 0.3,
                "metric": "auroc",
            }
        ]
    )
    monkeypatch.setattr(plots, "_save_figure", lambda *args, **kwargs: None)
    fig = plots.performance_contrast_forest_plot(frame, output_directory="unused")
    axis = fig.axes[0]
    np.testing.assert_allclose(axis.collections[0].get_segments()[0][:, 0], [0.1, 0.3])
    marker = next(line for line in axis.lines if line.get_marker() == "o")
    np.testing.assert_allclose(marker.get_xdata(), [0.4])
    plt.close(fig)
