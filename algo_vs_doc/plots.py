# Load required packages
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
from matplotlib.patches import Rectangle
from matplotlib.colors import to_rgba

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    auc,
)
from textwrap import wrap


def _save_figure(fig, name, output_directory, *, tight=True):
    """Save a deterministic SVG and an ignored submission PDF."""
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    bbox_inches = "tight" if tight else None
    cohort = output_directory.name
    with mpl.rc_context({"svg.hashsalt": f"{cohort}/{name}"}):
        fig.savefig(
            output_directory / f"{name}.svg",
            format="svg",
            bbox_inches=bbox_inches,
            metadata={"Date": None},
        )

    if output_directory.parent.name == "figures":
        project_root = output_directory.parent.parent
        submission_directory = project_root / "results" / "submission_figures" / cohort
    else:
        submission_directory = output_directory / "submission_figures"
    submission_directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        submission_directory / f"{name}.pdf",
        bbox_inches=bbox_inches,
        metadata={"CreationDate": None},
    )


# Default:
plt.rcParams["mathtext.fontset"] = "dejavusans"

# Serif type:
# plt.rcParams["mathtext.fontset"] = 'dejavuserif'

# Embed true outlines instead of Type 3 bitmap glyphs, so submitted PDFs pass journal
# text-editability checks.
plt.rcParams["pdf.fonttype"] = 42


def barplot_by_outcome(data, variable, chi2="", *, output_directory):
    frequencies = data.loc[:, variable].value_counts(dropna=True, sort=True)
    categories = np.array(sorted(frequencies.index))
    values = {
        "absolute": {
            "event": data.loc[data.outcome.eq(True), variable]
            .value_counts(dropna=True)
            .reindex(categories, fill_value=0)
            .values,
            "nonevent": data.loc[data.outcome.eq(False), variable]
            .value_counts(dropna=True)
            .reindex(categories, fill_value=0)
            .values,
        },
        "relative": {
            "event": 100
            * data.loc[data.outcome.eq(True), variable]
            .value_counts(dropna=True)
            .reindex(categories, fill_value=0)
            .values
            / sum(data.outcome.eq(True)),
            "nonevent": 100
            * data.loc[data.outcome.eq(False), variable]
            .value_counts(dropna=True)
            .reindex(categories, fill_value=0)
            .values
            / sum(data.outcome.eq(False)),
        },
    }
    fig = plt.figure(figsize=(10, 5))
    axes = [fig.add_subplot(121), fig.add_subplot(122)]
    for i, typ in enumerate(["absolute", "relative"]):
        ax = axes[i]
        if len(values[typ]["nonevent"]) != len(categories):
            values[typ]["nonevent"] = np.hstack((values[typ]["nonevent"], [0]))
        bar1 = ax.bar(
            categories - 0.1,
            values[typ]["nonevent"],
            width=0.2,
            color="#3cb371",  # medium sea green
            align="center",
        )
        if len(values[typ]["event"]) != len(categories):
            values[typ]["event"] = np.hstack((values[typ]["event"], [0]))
        bar2 = ax.bar(
            categories + 0.1,
            values[typ]["event"],
            width=0.2,
            color="#dc143c",  # crimson
            align="center",
        )
        ax.set_xticks(categories)
        ax.set_xticklabels(["Low Risk", "Moderate Risk", "High Risk"])
        ax.legend(
            (bar1[0], bar2[0]), ("No Readmission or Death", "Readmission or Death")
        )
    axes[0].set_ylabel("Number of Predictions as Risk category")
    axes[1].set_ylabel(
        "Physician Predictions as Risk category\n(% of Outcome Category)"
    )
    if chi2 != "":
        fig.suptitle(
            "Physician Predictions by Risk and Outcome category",
            x=fig.subplotpars.left,
            horizontalalignment="left",
            fontweight="bold",
        )
        if chi2.pvalue < 0.001:
            p_str = "$p < 0.001$"
        else:
            p_str = f"$p = {chi2.pvalue:#.2g}$"
        axes[0].set_title(
            f"Chi-square test ($\\chi^2 = {chi2.statistic:#0.3g}$, {p_str})",
            loc="left",
            fontsize="medium",
        )

    _save_figure(fig, "barplot_physician_prediction_cat", output_directory)


def barplot_confidence_cat_prediction(data, variable, chi2="", *, output_directory):
    frequencies = data.loc[:, variable].value_counts(dropna=True, sort=True)
    categories = np.array(sorted(frequencies.index))

    # only absolute will be used since the relative ones did not add much information
    values = {
        "absolute": {
            "high": data.loc[data.confidence == "high", variable]
            .value_counts(dropna=True)
            .reindex(categories, fill_value=0)
            .values,
            "medium/low": data.loc[data.confidence == "low/medium", variable]
            .value_counts(dropna=True)
            .reindex(categories, fill_value=0)
            .values,
        },
        "relative": {
            "high": 100
            * data.loc[data.confidence == "high", variable]
            .value_counts(dropna=True)
            .reindex(categories, fill_value=0)
            .values
            / sum(data.confidence == "high"),
            "medium/low": 100
            * data.loc[data.confidence == "low/medium", variable]
            .value_counts(dropna=True)
            .reindex(categories, fill_value=0)
            .values
            / sum(data.confidence == "low/medium"),
        },
    }
    fig = plt.figure(figsize=(10, 5))
    # only show the absolute values
    # axes = [fig.add_subplot(121), fig.add_subplot(122)]
    axes = [fig.add_subplot(121)]
    for i, typ in enumerate(["absolute"]):
        ax = axes[i]
        if len(values[typ]["high"]) != len(categories):
            values[typ]["high"] = np.hstack((values[typ]["high"], [0]))
        bar1 = ax.bar(
            categories - 0.2,
            values[typ]["high"],
            width=0.2,
            color="#00A36C",  # Jade
            align="center",
        )
        if len(values[typ]["medium/low"]) != len(categories):
            values[typ]["medium/low"] = np.hstack((values[typ]["medium/low"], [0]))
        bar2 = ax.bar(
            categories,
            values[typ]["medium/low"],
            width=0.2,
            color="#FFAA33",  # Yellow Orange
            align="center",
        )
        ax.set_xticks(categories)
        ax.set_xticklabels(["Low Risk", "Moderate Risk", "High Risk"])
        ax.legend((bar1[0], bar2[0]), ("High Confidence", "Medium/Low Confidence"))
        if i == 0:
            label = "Number of Predictions as Risk category\n"
        elif i == 1:
            label = "Physician Predictions as Category\n(% of Confidence Category)"
        ax.set_ylabel(label)

    # Do not show super title
    # fig.suptitle(
    #         f"Physician Predictions by Risk and self-reported Confidence",
    #         x=fig.subplotpars.left,
    #         horizontalalignment="left",
    #         fontweight="bold"
    #     )

    if chi2 != "":
        if chi2.pvalue < 0.001:
            p_str = "$p < 0.001$"
        else:
            p_str = f"$p = {chi2.pvalue:#.2g}$"
        axes[0].set_title(
            f"Chi-square test: $\\chi^2 = {chi2.statistic:#0.3g}$ ({p_str})",
            loc="left",
            fontsize="medium",
        )

    _save_figure(fig, "barplot_physician_prediction_cat_confidence", output_directory)


def histogram_by_outcome(data, variables, ttest="", *, output_directory):
    fig = plt.figure(figsize=(12, 4))
    axes = [fig.add_subplot(131), fig.add_subplot(132), fig.add_subplot(133)]

    for i, v in enumerate(variables):
        axes[i].hist(
            [
                data.loc[data.outcome.eq(False), v] * 100,
                data.loc[data.outcome.eq(True), v] * 100,
            ],
            20,
            color=[
                # medium sea green:
                "#3cb371",
                # crimson:
                "#dc143c",
            ],
            label=["No Readmission or Death", "Readmission or Death"],
        )
        axes[i].legend(loc="upper right")
        axes[i].set_xlim(left=0, right=100)
    axes[0].set_xlabel("Physician Predictions (Percentage)", fontsize=8.5)
    axes[1].set_xlabel("Model Predictions at Physician Prediction", fontsize=8.5)
    axes[2].set_xlabel("Model Predictions at Patient Discharge", fontsize=8.5)
    fig.suptitle(
        "Predictions by Outcome category",
        x=fig.subplotpars.left,
        horizontalalignment="left",
        fontweight="bold",
    )
    if ttest != "":
        for i in range(0, len(ttest)):
            if ttest[i].pvalue < 0.001:
                p_str = "$p < 0.001$"
            else:
                p_str = f"$p = {ttest[i].pvalue:#.1g}$"

            axes[i].set_title(
                f"t-test ($t = {ttest[i].statistic:#0.3g}$, {p_str})",
                loc="left",
                fontsize="medium",
            )
    _save_figure(fig, "histogram_physician_model_prediction", output_directory)


def stratified_histogram(
    data,
    variable,
    group,
    kruskal="",
    labels="",
    plotname="",
    title="",
    *,
    output_directory,
):
    groups = np.array(data[group].value_counts(dropna=True, sort=True).keys())
    colors = [
        "#00A36C",  # Jade
        "#FFAA33",  # Yellow Orange
        "#FF0000",  # Red
    ]
    if labels == "":
        labels = groups
    fig = plt.figure(figsize=(6, 3))
    axes = [fig.add_subplot()]

    axes[0].hist(
        [data.loc[data[group] == g, variable] * 100 for g in groups],
        40,
        color=colors[0 : len(groups)],
        label=groups,
    )
    axes[0].legend(loc="upper right", labels=labels)
    axes[0].set_xlim(left=0, right=100)
    axes[0].set_xlabel("Physician Predictions (Percentage)", fontsize=8.5)
    if kruskal != "":
        fig.suptitle(
            f"{title}",
            x=fig.subplotpars.left,
            y=1.02,
            horizontalalignment="left",
            fontweight="bold",
        )

        if kruskal.pvalue < 0.001:
            p_str = "$p < 0.001$"
        else:
            p_str = f"$p = {kruskal.pvalue:#.1g}$"

        axes[0].set_title(
            f"Kruskal-Wallis test ($H = {kruskal.statistic:#.4g}$, {p_str})",
            loc="left",
            fontsize="medium",
        )
    _save_figure(fig, "stratified_histogram_" + plotname, output_directory)


def scatter_plot(data, xs, ys, correlation="", *, output_directory):
    fig = plt.figure(figsize=(12, 4))
    axes = [fig.add_subplot(131), fig.add_subplot(132), fig.add_subplot(133)]

    for i, x in enumerate(xs):
        y = ys[i]
        axes[i].scatter(
            data.loc[data.outcome.eq(False), x] * 100,
            data.loc[data.outcome.eq(False), y] * 100,
            c="#3cb371",  # medium sea green
            s=10,
            label="No Readmission or Death",
        )
        axes[i].scatter(
            data.loc[data.outcome.eq(True), x] * 100,
            data.loc[data.outcome.eq(True), y] * 100,
            c="#dc143c",  # crimson
            s=10,
            label="Readmission or Death",
        )
        axes[i].set_xlim(left=0, right=100)
        axes[i].set_ylim(bottom=0, top=100)
        axes[i].yaxis.set_label_coords(-0.12, 0.5)
        axes[i].legend(loc="upper right")

    # fig.suptitle(
    #     f"Scatterplot of Prediction methods",
    #     x=fig.subplotpars.left,
    #     horizontalalignment="left",
    #     fontweight="bold"
    # )
    axes[0].set_xlabel("Physician Predictions (Percentage)", fontsize=8.5)
    axes[0].set_ylabel("Model Predictions at Physician Prediction", fontsize=8.5)
    axes[1].set_xlabel("Physician Predictions (Percentage)", fontsize=8.5)
    axes[1].set_ylabel("Model Predictions at Patient Discharge", fontsize=8.5)
    axes[2].set_xlabel("Model Predictions at Physician Prediction", fontsize=8.5)
    axes[2].set_ylabel("Model Predictions at Patient Discharge", fontsize=8.5)
    if correlation != "":
        for i in range(0, len(correlation)):
            axes[i].set_title(
                f"Spearman $\\rho = {correlation[i]:#.2g}$",
                loc="left",
                fontsize="medium",
            )
    _save_figure(fig, "scatter_plot_physician_model_prediction", output_directory)


def plot_calibration(curves, deciles, histograms, *, output_directory):
    """Render saved calibration summaries without fitting or bootstrap sampling."""
    from matplotlib.ticker import FuncFormatter

    labels = list(curves["prediction_source"].drop_duplicates())
    agent_colors = {
        "Physician (Percentage)": "tab:blue",
        "Physician (Category)": "tab:orange",
        "Model": "tab:green",
        "Model at Physician Prediction": "tab:green",
        "Model at physician prediction": "tab:green",
        "Combined (Percentage)": "tab:red",
        "Combined (Category)": "tab:purple",
        "Combined Model and Physician (Percentage)": "tab:red",
        "Combined Model and Physician (Category)": "tab:purple",
    }
    unknown = set(labels) - set(agent_colors)
    if unknown:
        raise ValueError(f"unknown agents in calibration plot: {sorted(unknown)}")
    os.makedirs(output_directory, exist_ok=True)
    n_columns = min(2, len(labels))
    n_rows = (len(labels) + n_columns - 1) // n_columns
    fig = plt.figure(figsize=(6 * n_columns, 7 * n_rows), layout="constrained")
    outer = fig.add_gridspec(n_rows, n_columns)
    for i, label in enumerate(labels):
        row, column = divmod(i, n_columns)
        inner = outer[row, column].subgridspec(2, 1, height_ratios=[5, 1], hspace=0.03)
        axis = fig.add_subplot(inner[0])
        histogram = fig.add_subplot(inner[1], sharex=axis)
        curve = curves.loc[curves.prediction_source.eq(label)]
        groups = deciles.loc[deciles.prediction_source.eq(label)]
        counts = histograms.loc[histograms.prediction_source.eq(label)]
        if groups.empty or counts.empty:
            raise ValueError(f"missing saved calibration summaries for {label}")
        color = agent_colors[label]
        axis.plot([0, 1], [0, 1], "--", color="grey", label="Ideal calibration")
        axis.fill_between(
            curve.predicted,
            curve.lower,
            curve.upper,
            color=color,
            alpha=0.2,
            label="Pointwise 95% bootstrap CI",
        )
        axis.plot(
            curve.predicted, curve.observed, color=color, label="Smoothed calibration"
        )
        axis.vlines(
            groups.predicted, groups.lower, groups.upper, color=color, linewidth=1.5
        )
        axis.scatter(
            groups.predicted,
            groups.observed,
            color=color,
            edgecolor="black",
            s=35,
            zorder=4,
            clip_on=False,
            label="Risk deciles with 95% CI",
        )
        axis.set(
            xlim=(0, 1),
            ylim=(0, 1),
            ylabel="Observed event proportion",
            title=label.replace("Model at Physician Prediction", "Model").replace(
                "Combined (", "Combined Model and Physician ("
            ),
        )
        axis.set_aspect("equal", adjustable="box")
        axis.tick_params(labelbottom=False)
        axis.legend(fontsize="small", loc="best")
        centers = (counts.bin_lower + counts.bin_upper) / 2
        width = counts.bin_upper - counts.bin_lower
        events, nonevents = counts.events, counts.nonevents
        histogram.bar(centers, events, width=width, color="tab:red", label="Event")
        histogram.bar(
            centers, -nonevents, width=width, color="tab:blue", label="No event"
        )
        histogram.axhline(0, color="black", linewidth=0.6)
        histogram.yaxis.set_major_formatter(
            FuncFormatter(lambda value, _: f"{abs(value):g}")
        )
        histogram.set(xlabel="Predicted probability", ylabel="Count")
        histogram.set_box_aspect(0.2)
        histogram.legend(fontsize="small", loc="best")
    _save_figure(fig, "calibration", output_directory)
    return fig


def plot_auc_at_max_time_difference(
    time_dif_data, prediction_columns, *, output_directory
):
    fig, ax = plt.subplots()
    x = time_dif_data["max_time_difference"]
    colors = [
        "tab:blue",
        "tab:orange",
        "tab:green",
    ]

    if any(["AUROC" in column for column in time_dif_data.columns]):
        metric = "AUROC"
    else:
        metric = "AUPRC"

    for i, c in enumerate(prediction_columns):
        ax.plot(x, time_dif_data[f"{metric} {c}"], color=colors[i], label=c)
        ax.fill_between(
            x,
            time_dif_data[f"LB {c}"],
            time_dif_data[f"UB {c}"],
            alpha=0.1,
            color=colors[i],
        )
    ax.set_xlim((4, 24))
    ax.set_ylim((0, 1))
    ax.set_xticks([4, 8, 12, 16, 20, 24])
    ax.set_xlabel("Allowed Time Difference between Prediction and Discharge\n(hours)")
    ax.set_ylabel(f"{metric}")
    ax.legend(loc="lower right")

    # super title
    fig.suptitle(
        "Performance by allowed Time Difference\nbetween Prediction and Discharge",
        x=fig.subplotpars.left,
        y=fig.subplotpars.top + 0.01,
        horizontalalignment="left",
        verticalalignment="bottom",
        fontweight="bold",
    )

    _save_figure(fig, "AUC_at_different_max_time_difference", output_directory)


def plot_coverage_at_max_features(coverage_data, *, output_directory):
    fig, ax = plt.subplots()
    x = coverage_data["max_model_features"]
    colors = ["#3454D1", "#34D1BF", "#D1345B"]
    for i, c in enumerate(["absolute_values", "no_absolute_values"]):
        ax.plot(x, coverage_data[c], color=colors[i])
    ax.set_ylim((0, 100))
    ax.set_xlabel("Number of Features used")
    ax.set_ylabel("Average Shapley value Coverage (%)")
    ax.legend(["Absolute Shapley values", "Raw Shapley values"], loc="lower right")

    # super title
    fig.suptitle(
        "Feature importance coverage by number of features",
        x=fig.subplotpars.left,
        y=fig.subplotpars.top + 0.01,
        horizontalalignment="left",
        verticalalignment="bottom",
        fontweight="bold",
    )

    _save_figure(fig, "coverage_at_different_max_features", output_directory)


def plot_combined_curves(
    predictions,
    outcome,
    prediction_columns,
    curve="roc",
    *,
    output_directory,
):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.margins(x=0, y=0)
    ax.xaxis.labelpad = 15
    ax.yaxis.labelpad = 15

    prev_x = None
    prev_y = None

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

    i = 0
    for column_desc in prediction_columns:
        column = prediction_columns[column_desc]

        y_true = predictions.loc[~pd.isnull(predictions[column]), outcome]
        y_pred = predictions.loc[~pd.isnull(predictions[column]), column]

        if curve == "roc":
            # suptitle = "Receiver Operating Characteristic"
            xlabel = "False Positive Rate (1 - Specificity)"
            ylabel = "True Positive Rate (Sensitivity)"

            # x_axis: fpr, y_axis: tpr
            x_axis, y_axis, thresholds = roc_curve(y_true, y_pred)
        elif curve == "prc":
            # suptitle = "Precision-Recall Curve"
            xlabel = "Recall (Sensitivity)"
            ylabel = "Precision (Positive Predictive Value)"

            # x_axis: recall, y_axis: precision
            y_axis, x_axis, thresholds = precision_recall_curve(y_true, y_pred)

        observed_auc = (
            average_precision_score(y_true, y_pred)
            if curve == "prc"
            else auc(x_axis, y_axis)
        )

        label = f"{column_desc} \n(AU{curve.upper()} = {observed_auc:#.2g})"
        if curve == "prc":
            ax.step(
                x_axis,
                y_axis,
                where="post",
                label=label,
                color=colors[i],
                linewidth=2,
            )
            ax.fill_between(
                x_axis,
                y_axis,
                0,
                step="post",
                color=colors[i],
                alpha=0.08,
            )
        else:
            ax.plot(x_axis, y_axis, label=label, color=colors[i], linewidth=2)
            if prev_y is None:
                ax.fill_between(x_axis, y_axis, 0, color=colors[i], alpha=0.2)
            else:
                ax.fill(
                    np.append(x_axis, prev_x[::-1]),
                    np.append(y_axis, prev_y[::-1]),
                    color=colors[i],
                    alpha=0.2,
                )

        prev_x = x_axis
        prev_y = y_axis

        i += 1

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    if curve == "roc":
        ax.axline([0, 0], slope=1, color="black", linestyle="--")
    else:
        ax.axhline(
            predictions[outcome].mean(),
            color="black",
            linestyle="--",
            label="Event prevalence",
        )
    plt.axis("square")

    ax.legend(
        loc="upper right" if curve == "prc" else "lower right",
        fontsize=10,
        facecolor="#fbf8ff",
        framealpha=1.0,
    )

    _save_figure(fig, f"combined_{curve}_curves", output_directory)


def plot_performance_by_confidence(
    performance_table_medium_low_confidence,
    performance_table_high_confidence,
    *,
    output_directory,
):
    columns = performance_table_medium_low_confidence.columns
    if "auroc" in columns:
        metric = "auroc"
    else:
        metric = "auprc"

    high_confidence_auroc = performance_table_high_confidence[metric].values
    high_confidence_error_low = np.maximum(
        0,
        high_confidence_auroc
        - performance_table_high_confidence[
            f"{metric}_confidence_interval_lower"
        ].values,
    )
    high_confidence_error_high = np.maximum(
        0,
        performance_table_high_confidence[f"{metric}_confidence_interval_upper"].values
        - high_confidence_auroc,
    )

    low_confidence_auroc = performance_table_medium_low_confidence[metric].values
    low_confidence_error_low = np.maximum(
        0,
        low_confidence_auroc
        - performance_table_medium_low_confidence[
            f"{metric}_confidence_interval_lower"
        ].values,
    )
    low_confidence_error_high = np.maximum(
        0,
        performance_table_medium_low_confidence[
            f"{metric}_confidence_interval_upper"
        ].values
        - low_confidence_auroc,
    )

    labels = performance_table_high_confidence.index.values

    labels = [
        label.replace("Combined (", "Combined Model and Physician (")
        for label in labels
    ]
    labels = [label.replace("(", "\n(") for label in labels]
    labels = ["\n".join(wrap(label, 12)) for label in labels]

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    ax.bar(
        x=x - width / 2,
        height=high_confidence_auroc,
        color="#00A36C",  # Jade
        yerr=[high_confidence_error_low, high_confidence_error_high],
        capsize=3,
        width=width,
        label="High Confidence",
    )
    ax.bar(
        x=x + width / 2,
        height=low_confidence_auroc,
        yerr=[low_confidence_error_low, low_confidence_error_high],
        color="#FFAA33",  # Yellow Orange
        capsize=3,
        width=width,
        label="Medium/Low Confidence",
    )

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(metric.upper())

    # super title
    # fig.suptitle(
    #     'Performance by Physician Confidence in Prediction',
    #     x=fig.subplotpars.left,
    #     horizontalalignment="left",
    #     fontweight="bold"
    # )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    # legend above plot:
    ax.legend(
        bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
        fontsize="small",
        loc="lower left",
        mode="expand",
        borderaxespad=0.0,
    )

    _save_figure(fig, "performance_by_confidence", output_directory)


def _format_p_value(value):
    return "< 0.001" if value < 0.001 else f"= {value:.3f}"


def plot_bootstrap_auc_difference(
    axis,
    observed_difference,
    confidence_lower,
    confidence_upper,
    p_value,
    bootstrapped_differences,
    curve,
    adjusted_p_value=None,
    comparison_family=None,
):
    """
    Plot distribution of bootstrapped AUC differences
    """
    axis.hist(bootstrapped_differences, bins="auto")
    axis.axvline(
        observed_difference,
        color="r",
        linestyle="dashed",
        linewidth=2,
        label="Observed difference",
    )
    axis.axvline(0, color="y", linestyle="dashed", linewidth=2, label="No difference")
    axis.axvline(
        confidence_lower,
        color="b",
        linestyle="dashed",
        linewidth=2,
        label="2.5 percentile",
    )
    axis.axvline(
        confidence_upper,
        color="b",
        linestyle="dashed",
        linewidth=2,
        label="97.5 percentile",
    )
    # axis.set_xlabel(f"Difference in {curve} AUC")
    # axis.set_ylabel(f"Frequency")

    # place a text box in upper right in axes coords
    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle="round,pad=0.25", facecolor="#fbf8ff", alpha=1.0)
    p_text = f"$\\Delta_{{AU{curve.upper()}}} = {observed_difference:#.2g}$"
    if comparison_family != "exploratory":
        if adjusted_p_value is not None and pd.notna(adjusted_p_value):
            p_text += f"\n$p$ {_format_p_value(adjusted_p_value)}"
        elif comparison_family is None and p_value is not None:
            p_text += f"\n$p$ {_format_p_value(p_value)}"
    if comparison_family is not None:
        family_label = {
            "primary": "Primary",
            "secondary": "Secondary",
            "exploratory": "Exploratory",
        }.get(comparison_family, comparison_family)
        title = axis.set_title(
            family_label, loc="left", fontsize="small", fontweight="bold", y=1.0, pad=4
        )
        if family_label in {"Primary", "Secondary"}:
            title.set_color("white")
            title.set_bbox(
                dict(
                    boxstyle="round,pad=0.2",
                    facecolor="black",
                    edgecolor="black",
                )
            )
    axis.annotate(
        p_text,
        xy=(1, 1),
        xycoords="axes fraction",
        xytext=(-8, -8),
        textcoords="offset points",
        fontsize=9,
        linespacing=1.1,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=props,
    )


def full_extent(axis, pad=0.0, *, draw=True):
    """Return an axis extent including tick labels, labels, and its title."""
    if draw:
        axis.figure.canvas.draw()
    renderer = axis.figure.canvas.get_renderer()
    bbox = axis.get_tightbbox(renderer)
    title_boxes = []
    for artist in axis.get_children():
        if hasattr(artist, "get_bbox_patch") and artist.get_bbox_patch() is not None:
            artist.update_bbox_position_size(renderer)
            title_boxes.append(artist.get_bbox_patch().get_window_extent(renderer))
    bbox = Bbox.union([bbox, *title_boxes])
    return bbox.expanded(1.0 + pad, 1.0 + pad)


def combined_auc_difference_plot(
    bootstrapped_data, *, output_directory, agent_order=None, agent_colors=None
):
    """
    Plot a collection of comparisons of AUC differences in a tabular format
    """
    if not bootstrapped_data or any(
        not values for values in bootstrapped_data.values()
    ):
        raise ValueError("comparison matrix requires nonempty comparison rows")
    rows = list(bootstrapped_data) if agent_order is None else agent_order[:-1]
    columns = (
        list(dict.fromkeys(column for row in rows for column in bootstrapped_data[row]))
        if agent_order is None
        else agent_order[1:]
    )
    mosaic = [[f"{row}:{column}" for column in columns] for row in rows]
    first_row = next(iter(bootstrapped_data))
    first_key = f"{first_row}:{next(iter(bootstrapped_data[first_row]))}"
    fig, axs = plt.subplot_mosaic(
        mosaic=mosaic,
        sharex=True,
        sharey=True,
        layout="constrained",
        figsize=[12, 9.5],
    )
    fig.get_layout_engine().set(h_pad=0.18, w_pad=0.08)
    for row in rows:
        for column in columns:
            if column not in bootstrapped_data.get(row, {}):
                axs[f"{row}:{column}"].set_axis_off()

    curves = {
        values[5] for row in bootstrapped_data.values() for values in row.values()
    }
    if len(curves) != 1:
        raise ValueError("select one AUC metric for the matrix")
    curve = next(iter(curves))
    axs[first_key].set_xlim((-0.25, 0.55) if curve == "roc" else (-1, 1))

    # plot bootstrapping distribution
    for row in bootstrapped_data.keys():
        for column in bootstrapped_data[row].keys():
            (
                observed_difference,
                confidence_lower,
                confidence_upper,
                p_value,
                bootstrapped_differences,
                curve,
                *extra,
            ) = bootstrapped_data[row][column]
            plot_bootstrap_auc_difference(
                axs[f"{row}:{column}"],
                observed_difference,
                confidence_lower,
                confidence_upper,
                p_value,
                bootstrapped_differences,
                curve,
                adjusted_p_value=extra[0] if extra else None,
                comparison_family=extra[1] if len(extra) > 1 else None,
            )

    # show ticks for first graph of each row
    for row in mosaic:
        first = True
        for column_name in row:
            if column_name == "." or not axs[column_name].axison:
                continue
            axs[column_name].tick_params(labelleft=True, labelbottom=True)
            if first:
                axs[column_name].set_xlabel(f"difference in AU{curve.upper()}")
                axs[column_name].set_ylabel("frequency")
            first = False

    fig.canvas.draw()
    fig.set_layout_engine("none")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for axis in axs.values():
        title = axis._left_title
        badge = title.get_bbox_patch()
        if axis.axison and badge is not None:
            title.update_bbox_position_size(renderer)
            shift = (
                axis.get_window_extent(renderer).xmin
                - badge.get_window_extent(renderer).xmin
            )
            x, y = title.get_position()
            title.set_position((x + shift / axis.get_window_extent(renderer).width, y))
    fig.canvas.draw()
    extents = {axis: full_extent(axis, draw=False) for axis in axs.values()}

    default_colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    agent_colors = agent_colors or {
        agent: default_colors[index % len(default_colors)]
        for index, agent in enumerate(agent_order or rows + [columns[-1]])
    }

    active_axes = [axs[key] for row in mosaic for key in row if axs[key].axison]
    extent_all = Bbox.union([extents[axis] for axis in active_axes])
    extent_all = extent_all.transformed(fig.transFigure.inverted())
    outer_padding = 4 / 72 / fig.get_figheight()
    horizontal_padding = 6 / 72 / fig.get_figwidth()
    extent_all = Bbox.from_extents(
        extent_all.xmin - horizontal_padding,
        extent_all.ymin - outer_padding,
        extent_all.xmax + horizontal_padding,
        extent_all.ymax + outer_padding,
    )
    cell_width = extent_all.width / len(columns)
    left_clearance = (
        max(
            axis.get_position().xmin
            - extents[axis].transformed(fig.transFigure.inverted()).xmin
            for axis in active_axes
        )
        + horizontal_padding
    )
    right_clearance = (
        max(
            extents[axis].transformed(fig.transFigure.inverted()).xmax
            - axis.get_position().xmax
            for axis in active_axes
        )
        + horizontal_padding
    )
    for row_keys in mosaic:
        for column_index, key in enumerate(row_keys):
            axis = axs[key]
            position = axis.get_position()
            axis.set_position(
                [
                    extent_all.xmin + column_index * cell_width + left_clearance,
                    position.y0,
                    cell_width - left_clearance - right_clearance,
                    position.height,
                ]
            )
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for axis in active_axes:
        title = axis._left_title
        badge = title.get_bbox_patch()
        if badge is not None:
            title.update_bbox_position_size(renderer)
            axis_bounds = axis.get_window_extent(renderer)
            shift = axis_bounds.xmin - badge.get_window_extent(renderer).xmin
            x, y = title.get_position()
            title.set_position((x + shift / axis_bounds.width, y))
    fig.canvas.draw()
    extents = {axis: full_extent(axis, draw=False) for axis in axs.values()}
    row_extents = []
    for row_keys in mosaic:
        axes = [axs[key] for key in row_keys if axs[key].axison]
        extent = Bbox.union([extents[axis] for axis in axes])
        row_extents.append(extent.transformed(fig.transFigure.inverted()))

    row_titles = []
    row_header_containers = []
    x_padding = 0.02
    # Pack full subplot bounds with equal clearance inside continuous row bands.
    title_clearance = 6 / 72 / fig.get_figheight()
    cell_height = (
        max(extent.height for extent in row_extents) + outer_padding + title_clearance
    )
    row_edges = [extent_all.ymin]
    bottom = extent_all.ymin
    for row_keys, extent in reversed(list(zip(mosaic, row_extents))):
        shift = bottom + outer_padding - extent.ymin
        for key in row_keys:
            axis = axs[key]
            position = axis.get_position()
            axis.set_position(
                [position.x0, position.y0 + shift, position.width, position.height]
            )
        bottom += cell_height
        row_edges.append(bottom)
    row_edges.reverse()
    fig.canvas.draw()
    extents = {axis: full_extent(axis, draw=False) for axis in axs.values()}
    extent_all = Bbox.from_extents(
        extent_all.xmin,
        extent_all.ymin,
        extent_all.xmax,
        row_edges[0],
    )
    row_geometry = [
        (bottom, top - bottom) for top, bottom in zip(row_edges, row_edges[1:])
    ]

    for index, (row, (bottom, height)) in enumerate(zip(rows, row_geometry)):
        row_titles.append(
            fig.text(
                extent_all.xmin - x_padding,
                bottom + height / 2,
                "and\n".join(row.split("and ")),
                fontweight="bold",
                fontsize="small",
                multialignment="center",
                color=agent_colors[row],
                ha="center",
                va="center",
                rotation=90,
                transform=fig.transFigure,
            )
        )

    fig.canvas.draw()
    label_extent = (
        row_titles[0].get_window_extent().transformed(fig.transFigure.inverted())
    )
    header_width = 2 * (extent_all.xmin - (label_extent.xmin + label_extent.xmax) / 2)
    for index, (bottom, height) in enumerate(row_geometry):
        fig.patches.append(
            Rectangle(
                (extent_all.xmin, bottom),
                extent_all.width,
                height,
                facecolor=to_rgba(agent_colors[rows[index]], 0.25),
                edgecolor="black",
                zorder=-1,
                transform=fig.transFigure,
            )
        )
        header = Rectangle(
            (extent_all.xmin - header_width, bottom),
            header_width,
            height,
            facecolor=to_rgba(agent_colors[rows[index]], 0.20),
            edgecolor=to_rgba(agent_colors[rows[index]], 0.50),
            linewidth=0.8,
            zorder=-1,
            transform=fig.transFigure,
        )
        fig.patches.append(header)
        row_header_containers.append(header)

    column_geometry = [
        (extent_all.xmin + index * cell_width, cell_width)
        for index in range(len(columns))
    ]

    header_padding = 0.008
    column_titles = []
    for index, (column, (left, width)) in enumerate(zip(columns, column_geometry)):
        column_titles.append(
            fig.text(
                left + width / 2,
                extent_all.ymax + header_padding,
                "and\n".join(column.split("and ")),
                fontweight="bold",
                fontsize="small",
                color=agent_colors[column],
                ha="center",
                va="bottom",
                transform=fig.transFigure,
            )
        )

    fig.canvas.draw()
    title_extent = Bbox.union(
        [title.get_window_extent() for title in column_titles]
    ).transformed(fig.transFigure.inverted())
    header_height = title_extent.ymax - extent_all.ymax + header_padding
    for title in column_titles:
        title.set_y(extent_all.ymax + header_height / 2)
        title.set_verticalalignment("center")
        title.set_multialignment("center")
    column_header_containers = []
    for index, (left, width) in enumerate(column_geometry):
        fig.patches.append(
            Rectangle(
                (left, extent_all.ymin),
                width,
                extent_all.height,
                facecolor=to_rgba(agent_colors[columns[index]], 0.25),
                edgecolor="black",
                zorder=-1,
                transform=fig.transFigure,
            )
        )
        header = Rectangle(
            (left, extent_all.ymax),
            width,
            header_height,
            facecolor=to_rgba(agent_colors[columns[index]], 0.20),
            edgecolor=to_rgba(agent_colors[columns[index]], 0.50),
            linewidth=0.8,
            zorder=-1,
            transform=fig.transFigure,
        )
        fig.patches.append(header)
        column_header_containers.append(header)

    row_header_extent = (
        row_header_containers[0]
        .get_window_extent()
        .transformed(fig.transFigure.inverted())
    )
    column_header_extent = (
        column_header_containers[0]
        .get_window_extent()
        .transformed(fig.transFigure.inverted())
    )
    fig.text(
        (row_header_extent.xmin + column_header_extent.xmin) / 2,
        (row_header_extent.ymax + column_header_extent.ymax) / 2,
        "vs.",
        fontsize="xx-large",
        color="black",
        ha="center",
        va="center",
        rotation=45,
        transform=fig.transFigure,
    )

    handles, labels = axs[first_key].get_legend_handles_labels()
    last_row_header_extent = (
        row_header_containers[-1]
        .get_window_extent()
        .transformed(fig.transFigure.inverted())
    )
    fig.legend(
        handles,
        labels,
        loc="lower left",
        bbox_to_anchor=(last_row_header_extent.xmax, last_row_header_extent.ymin),
        facecolor="#fbf8ff",
        framealpha=1.0,
    )

    name = "combined_auc_difference" if curve == "roc" else "combined_auprc_difference"
    _save_figure(fig, name, output_directory)
    return fig


def performance_contrast_forest_plot(
    contrasts,
    *,
    output_directory,
    formal_comparisons=False,
    show_adjusted_p_values=False,
):
    """Plot contrasts with nominal intervals and adjusted p values for formal tests."""
    data = contrasts.copy()
    metric = data["metric"].iloc[0] if "metric" in data else "auroc"
    if metric not in {"auroc", "auprc"} or (
        "metric" in data and data["metric"].nunique() != 1
    ):
        raise ValueError("select a single AUROC or AUPRC metric")
    agent_order = [
        "Model at Physician Prediction",
        "Physician (Percentage)",
        "Physician (Category)",
        "Combined (Percentage)",
        "Combined (Category)",
    ]
    rank = {agent: index for index, agent in enumerate(agent_order)}
    unknown = (set(data["agent_1"]) | set(data["agent_2"])) - set(agent_order)
    if unknown:
        raise ValueError(f"unknown agents in forest plot: {sorted(unknown)}")
    data["left_agent"] = data["agent_2"]
    data["right_agent"] = data["agent_1"]
    model_physician = data["right_agent"].eq("Model at Physician Prediction") & data[
        "left_agent"
    ].isin(["Physician (Percentage)", "Physician (Category)"])
    data["section"] = np.where(model_physician, 0, 1)
    data["left_rank"] = data["left_agent"].map(rank)
    data["right_rank"] = data["right_agent"].map(rank)
    data = data.sort_values(["section", "left_rank", "right_rank"]).reset_index(
        drop=True
    )
    y = np.empty(len(data))
    section_positions = []
    cursor = 0.0
    for section, group in data.groupby("section", sort=True):
        if section == 0 and formal_comparisons:
            label = "Primary" if metric == "auroc" else "Secondary"
        elif section == 0:
            label = "Model vs physician"
        elif formal_comparisons:
            label = "Exploratory"
        else:
            label = "Other comparisons"
        section_positions.append((cursor, label))
        cursor += 0.9
        for index in group.index:
            y[index] = cursor
            cursor += 1.0
        cursor += 0.6
    differences = data["difference_agent_1_minus_agent_2"].to_numpy(dtype=float)
    confidence_interval_lower = data["confidence_interval_lower"].to_numpy(dtype=float)
    confidence_interval_upper = data["confidence_interval_upper"].to_numpy(dtype=float)
    agent_labels = {
        "Model at Physician Prediction": "Model",
        "Physician (Percentage)": "Physician (Percentage)",
        "Physician (Category)": "Physician (Category)",
        "Combined (Percentage)": "Combined Model and\nPhysician (Percentage)",
        "Combined (Category)": "Combined Model and\nPhysician (Category)",
    }
    agent_colors = {
        "Model at Physician Prediction": "tab:green",
        "Physician (Percentage)": "tab:blue",
        "Physician (Category)": "tab:orange",
        "Combined (Percentage)": "tab:red",
        "Combined (Category)": "tab:purple",
    }

    fig, ax = plt.subplots(figsize=(12, max(5, 0.72 * len(data))))
    fig.subplots_adjust(left=0.30, right=0.70, top=0.93, bottom=0.12)
    for index, row in data.iterrows():
        color = "#444444"
        ax.plot(
            differences[index],
            y[index],
            "o",
            color=color,
            markersize=8,
            zorder=3,
        )
        ax.hlines(
            y[index],
            confidence_interval_lower[index],
            confidence_interval_upper[index],
            color=color,
            linewidth=3,
            zorder=3,
        )
        ax.vlines(
            [confidence_interval_lower[index], confidence_interval_upper[index]],
            y[index] - 0.07,
            y[index] + 0.07,
            color=color,
            linewidth=2,
            zorder=3,
        )
        family = row.get("family")
        adjusted_p = row.get("p_holm")
        if (
            show_adjusted_p_values
            and family in {"primary", "secondary"}
            and pd.notna(adjusted_p)
        ):
            p_text = "<0.001" if adjusted_p < 0.001 else f"= {adjusted_p:.3f}"
            text_direction = 1 if differences[index] >= 0 else -1
            ax.annotate(
                f"Holm-adjusted $p$ {p_text}",
                xy=(differences[index], y[index]),
                xytext=(6 * text_direction, -13),
                textcoords="offset points",
                ha="left" if text_direction > 0 else "right",
                va="top",
                fontsize="small",
                color="#444444",
                zorder=4,
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.9,
                    "pad": 1.0,
                },
            )
        for side, left in (("left_agent", -0.70), ("right_agent", 1.08)):
            agent = row[side]
            card_color = agent_colors[agent]
            ax.add_patch(
                Rectangle(
                    (left, y[index] - 0.36),
                    0.62,
                    0.72,
                    facecolor=to_rgba(card_color, 0.20),
                    edgecolor=to_rgba(card_color, 0.50),
                    linewidth=0.8,
                    transform=ax.get_yaxis_transform(),
                    clip_on=False,
                    zorder=2,
                )
            )
            ax.text(
                left + 0.31,
                y[index],
                agent_labels[agent],
                color=card_color,
                fontweight="bold",
                ha="center",
                va="center",
                transform=ax.get_yaxis_transform(),
                clip_on=False,
                zorder=3,
            )

    ax.axvline(0, color="black", linewidth=0.9, linestyle="--", zorder=1)
    ax.set_yticks(y, [""] * len(data))
    ax.tick_params(axis="y", length=0, pad=8)
    ax.set_ylim(cursor - 0.7, -0.5)
    ax.set_xlabel(f"$\\Delta$AU{metric[2:].upper()} (right agent - left agent)")
    for position, label in section_positions:
        ax.text(
            -0.70,
            position,
            label,
            transform=ax.get_yaxis_transform(),
            ha="left",
            va="center",
            fontweight="bold",
            color="black",
            clip_on=False,
        )
        ax.axhline(position + 0.35, color="#BBBBBB", linewidth=0.8)

    ax.grid(axis="x", alpha=0.2)
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.7)
    ax.set_title("Pairwise performance contrasts", pad=18, fontweight="bold")
    ax.text(
        0.0,
        -0.09,
        "Favours left agent",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize="small",
        color="#555555",
    )
    ax.text(
        1.0,
        -0.09,
        "Favours right agent",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize="small",
        color="#555555",
    )
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    name = "auroc_contrast_forest" if metric == "auroc" else "auprc_contrast_forest"
    _save_figure(fig, name, output_directory)
    return fig


def performance_contrast_matrix_plot(contrasts, bootstrap_samples, *, output_directory):
    """Render all unique pairwise contrasts in a row-minus-column matrix."""
    if contrasts.empty:
        raise ValueError("comparison matrix requires contrasts")
    if contrasts["contrast_id"].duplicated().any():
        raise ValueError("contrast IDs must be unique; select one cohort and metric")
    if contrasts.duplicated(["agent_1", "agent_2"]).any():
        raise ValueError("comparison pairs must be unique")
    metric = contrasts["metric"].iloc[0] if "metric" in contrasts else "auroc"
    if metric not in ("auroc", "auprc") or (
        "metric" in contrasts and contrasts["metric"].nunique() != 1
    ):
        raise ValueError("select a single AUROC or AUPRC metric")
    curve = metric[2:]
    display_labels = {
        "Model at Physician Prediction": "Model",
        "Combined (Percentage)": "Combined Model and Physician (Percentage)",
        "Combined (Category)": "Combined Model and Physician (Category)",
    }
    contrasts = contrasts.copy()
    for column in ("agent_1", "agent_2"):
        contrasts[column] = contrasts[column].replace(display_labels)
    if contrasts.duplicated(["agent_1", "agent_2"]).any():
        raise ValueError("comparison display labels must identify unique pairs")

    model = "Model"
    physician_percentage = "Physician (Percentage)"
    physician_category = "Physician (Category)"
    combined_percentage = "Combined Model and Physician (Percentage)"
    combined_category = "Combined Model and Physician (Category)"
    order = [
        model,
        physician_percentage,
        physician_category,
        combined_percentage,
        combined_category,
    ]
    present = set(contrasts["agent_1"]) | set(contrasts["agent_2"])
    if present != set(order):
        raise ValueError(f"comparison matrix requires {order}; found {sorted(present)}")
    rank = {agent: index for index, agent in enumerate(order)}
    comparisons = {}
    for _, row in contrasts.sort_values("contrast_id").iterrows():
        samples = bootstrap_samples.loc[
            bootstrap_samples["contrast_id"].eq(row["contrast_id"]),
            "bootstrap_difference_agent_1_minus_agent_2",
        ].dropna()
        if samples.empty:
            raise ValueError(
                f"missing bootstrap differences for contrast {row['contrast_id']}"
            )
        agent_1, agent_2 = row["agent_1"], row["agent_2"]
        matrix_row, matrix_column = sorted((agent_1, agent_2), key=rank.__getitem__)
        if agent_1 == matrix_row:
            difference = row["difference_agent_1_minus_agent_2"]
            lower = row["confidence_interval_lower"]
            upper = row["confidence_interval_upper"]
            oriented_samples = samples
        else:
            difference = -row["difference_agent_1_minus_agent_2"]
            lower = -row["confidence_interval_upper"]
            upper = -row["confidence_interval_lower"]
            oriented_samples = -samples
        primary = matrix_row == model and matrix_column in {
            physician_percentage,
            physician_category,
        }
        comparisons.setdefault(matrix_row, {})[matrix_column] = (
            difference,
            lower,
            upper,
            row["p"],
            oriented_samples,
            curve,
            row.get("p_holm"),
            (
                "primary"
                if primary and metric == "auroc"
                else "secondary"
                if primary
                else "exploratory"
            ),
        )
    expected_pairs = {
        (order[left], order[right])
        for left in range(len(order) - 1)
        for right in range(left + 1, len(order))
    }
    actual_pairs = {
        (row, column) for row, columns in comparisons.items() for column in columns
    }
    if actual_pairs != expected_pairs:
        raise ValueError(
            f"comparison matrix is missing pairs: {sorted(expected_pairs - actual_pairs)}"
        )

    agent_colors = {
        physician_percentage: "tab:blue",
        physician_category: "tab:orange",
        model: "tab:green",
        combined_percentage: "tab:red",
        combined_category: "tab:purple",
    }
    return combined_auc_difference_plot(
        comparisons,
        output_directory=output_directory,
        agent_order=order,
        agent_colors=agent_colors,
    )
