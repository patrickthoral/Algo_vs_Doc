# Load required packages
import os
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
from matplotlib.patches import Rectangle
from matplotlib.colors import to_rgba

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from textwrap import wrap

hospital = "Pacmed Medical Center"
output_folder = "../figures"


# Default:
plt.rcParams["mathtext.fontset"] = 'dejavusans'

# Serif type:
# plt.rcParams["mathtext.fontset"] = 'dejavuserif'


def barplot_by_outcome(data, variable, chi2=""):
    frequencies = data.loc[:, variable].value_counts(dropna=True, sort=True)
    categories = np.array(list(frequencies.keys() * 3))
    total = frequencies.sum()
    values = {
        "absolute": {
            "event": 
                data.loc[data.outcome == True, variable]
                .value_counts(dropna=False, sort=True)
                .values,
            "nonevent": 
                data.loc[data.outcome == False, variable]
                .value_counts(dropna=False, sort=True)
                .values
        },
        "relative": {
            "event": 100
                     * data.loc[data.outcome == True, variable]
                     .value_counts(dropna=False, sort=True)
                     .values
                     / sum(data.outcome == True),
            "nonevent": 100
                        * data.loc[data.outcome == False, variable]
                        .value_counts(dropna=False, sort=True)
                        .values
                        / sum(data.outcome == False),
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
        ax.legend((bar1[0], bar2[0]), ("No Readmission or Death", "Readmission or Death"))
    axes[0].set_ylabel("Number of Predictions as Risk category")
    axes[1].set_ylabel("Physician Predictions as Risk category\n(% of Outcome Category)")
    if chi2 != "":
        fig.suptitle(
            f"Physician Predictions by Risk and Outcome category",
            x=fig.subplotpars.left,
            horizontalalignment="left",
            fontweight="bold"
        )
        if chi2.pvalue < 0.001:
            p_str = f"$p < 0.001$"
        else:
            p_str = f"$p = {chi2.pvalue:#.2g}$"
        axes[0].set_title(
            f"Chi-square test ($\\chi^2 = {chi2.statistic:#0.3g}$, {p_str})",
            loc="left",
            fontsize="medium"
        )
        
    plt.savefig(
        os.path.join(output_folder, hospital, "barplot_physician_prediction_cat.pdf"), 
        bbox_inches="tight", 
        metadata={'CreationDate': None}
    )
    plt.savefig(
        os.path.join(output_folder, hospital, "barplot_physician_prediction_cat.png"), 
        bbox_inches="tight", 
        dpi=300
    )


def barplot_confidence_cat_prediction(data, variable, chi2=""):
    frequencies = data.loc[:, variable].value_counts(dropna=True, sort=True)
    categories = np.array(list(frequencies.keys() * 3))

    # only absolute will be used since the relative ones did not add much information
    values = {
        "absolute": {
            "high": data.loc[data.confidence == "Veel", variable]
                    .value_counts(dropna=False, sort=True)
                    .values,
            "medium/low": data.loc[data.confidence == "Gemiddeld/Weinig", variable]
                          .value_counts(dropna=False, sort=True)
                          .values,
        },
        "relative": {
            "high": 100
                    * data.loc[data.confidence == "Veel", variable]
                    .value_counts(dropna=False, sort=True)
                    .values
                    / sum(data.confidence == "Veel"),
            "medium/low": 100
                          * data.loc[data.confidence == "Gemiddeld/Weinig", variable]
                          .value_counts(dropna=False, sort=True)
                          .values
                          / sum(data.confidence == "Gemiddeld/Weinig"),
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
            align="center"
        )
        if len(values[typ]["medium/low"]) != len(categories):
            values[typ]["medium/low"] = np.hstack((values[typ]["medium/low"], [0]))
        bar2 = ax.bar(
            categories,
            values[typ]["medium/low"],
            width=0.2,
            color="#FFAA33",  # Yellow Orange
            align="center"
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
            p_str = f"$p < 0.001$"
        else:
            p_str = f"$p = {chi2.pvalue:#.2g}$"
        axes[0].set_title(
            f"Chi-square test: $\\chi^2 = {chi2.statistic:#0.3g}$ ({p_str})",
            loc="left",
            fontsize="medium"
        )
    
    plt.savefig(
        os.path.join(output_folder, hospital, "barplot_physician_prediction_cat_confidence.pdf"),
        bbox_inches="tight",
        metadata={'CreationDate': None}
    )
    plt.savefig(
        os.path.join(output_folder, hospital, "barplot_physician_prediction_cat_confidence.png"),
        bbox_inches="tight", 
        dpi=300
    )


def histogram_by_outcome(data, variables, ttest=""):
    fig = plt.figure(figsize=(12, 4))
    axes = [fig.add_subplot(131), fig.add_subplot(132), fig.add_subplot(133)]

    for i, v in enumerate(variables):
        axes[i].hist(
            [
                data.loc[(data.outcome == False), v] * 100,
                data.loc[(data.outcome == True), v] * 100,
            ],
            20,
            color=[
                # medium sea green:
                "#3cb371",
                # crimson:
                "#dc143c"
            ],
            label=["No Readmission or Death", "Readmission or Death"],
        )
        axes[i].legend(loc="upper right")
        axes[i].set_xlim(left=0, right=100)
    axes[0].set_xlabel("Physician Predictions (Percentage)", fontsize=8.5)
    axes[1].set_xlabel(
        "Model Predictions at Physician Prediction", fontsize=8.5
    )
    axes[2].set_xlabel("Model Predictions at Patient Discharge", fontsize=8.5)
    fig.suptitle(
        f"Predictions by Outcome category",
            x=fig.subplotpars.left,
            horizontalalignment="left",
            fontweight="bold"
    )
    if ttest != "":
        for i in range(0,len(ttest)):
            if ttest[i].pvalue < 0.001:
                p_str = f"$p < 0.001$"
            else:
                p_str = f"$p = {ttest[i].pvalue:#.1g}$"

            axes[i].set_title(
                f"t-test ($t = {ttest[i].statistic:#0.3g}$, {p_str})",
                loc="left",
                fontsize="medium"
            )
    plt.savefig(
        os.path.join(output_folder, hospital, "histogram_physician_model_prediction.pdf"),
        bbox_inches="tight",
        metadata={'CreationDate': None}
    )
    plt.savefig(
        os.path.join(output_folder, hospital, "histogram_physician_model_prediction.png"),
        bbox_inches="tight", 
        dpi=300
    )


def stratified_histogram(data, variable, group, kruskal="", labels="", plotname="", title=""):
    groups = np.array(data[group].value_counts(dropna=True, sort=True).keys())
    colors = [
        "#00A36C",  # Jade
        "#FFAA33",  # Yellow Orange
        "#FF0000"  # Red
    ]
    if labels == "":
        labels = groups
    fig = plt.figure(figsize=(6, 3))
    axes = [fig.add_subplot()]

    axes[0].hist(
        [data.loc[data[group] == g, variable] * 100 for g in groups],
        40,
        color=colors[0: len(groups)],
        label=groups
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
            fontweight="bold"
        )

        if kruskal.pvalue < 0.001:
            p_str = f"$p < 0.001$"
        else:
            p_str = f"$p = {kruskal.pvalue:#.1g}$"

        axes[0].set_title(
            f"Kruskal-Wallis test ($H = {kruskal.statistic:#.4g}$, {p_str})",
            loc="left",
            fontsize="medium"
        )
    plt.savefig(
        os.path.join(output_folder, hospital, "stratified_histogram_" + plotname + ".pdf"),
        bbox_inches="tight",
        metadata={'CreationDate': None}
    )
    plt.savefig(
        os.path.join(output_folder, hospital, "stratified_histogram_" + plotname + ".png"),
        bbox_inches="tight", 
        dpi=300
    )


def scatter_plot(data, xs, ys, correlation=""):
    fig = plt.figure(figsize=(12, 4))
    axes = [fig.add_subplot(131), fig.add_subplot(132), fig.add_subplot(133)]

    for i, x in enumerate(xs):
        y = ys[i]
        axes[i].scatter(
            data.loc[data.outcome == False, x] * 100,
            data.loc[data.outcome == False, y] * 100,
            c="#3cb371",  # medium sea green
            s=10,
            label="No Readmission or Death",
        )
        axes[i].scatter(
            data.loc[data.outcome == True, x] * 100,
            data.loc[data.outcome == True, y] * 100,
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
    axes[0].set_ylabel(
        "Model Predictions at Physician Prediction", fontsize=8.5
    )
    axes[1].set_xlabel("Physician Predictions (Percentage)", fontsize=8.5)
    axes[1].set_ylabel("Model Predictions at Patient Discharge", fontsize=8.5)
    axes[2].set_xlabel(
        "Model Predictions at Physician Prediction", fontsize=8.5
    )
    axes[2].set_ylabel("Model Predictions at Patient Discharge", fontsize=8.5)
    if correlation != "":
        for i in range(0, len(correlation)):
            axes[i].set_title(
                f"Spearman $\\rho = {correlation[i]:#.2g}$",
                loc="left",fontsize="medium"
            )
    plt.savefig(
        os.path.join(output_folder, hospital, "scatter_plot_physician_model_prediction.pdf"),
        bbox_inches="tight",
        metadata={'CreationDate': None}
    )
    plt.savefig(
        os.path.join(output_folder, hospital, "scatter_plot_physician_model_prediction.png"),
        bbox_inches="tight", 
        dpi=300
    )


def plot_calibration(predictions, outcome, prediction_columns):
    group_n = len(prediction_columns)
    colors = [
        'tab:blue',
        'tab:green',
        'tab:red',
        'tab:purple'
    ]
    fig, axs = plt.subplots(
        1 + group_n,
        1,
        figsize=(6, 6),
        sharex=True,
        gridspec_kw={"height_ratios": [6] + [2 / group_n] * group_n, "hspace": 0},
    )
    # plot 45 degree line
    axs[0].plot((0, 1), (0, 1), ls="--", c="#dbdbdb")

    i = 1
    for c in prediction_columns:
        y = predictions[outcome]
        p = predictions[prediction_columns[c]]

        # plot calibration curves, lowess smoothing
        lowess = sm.nonparametric.lowess
        z = lowess(endog=y, exog=p, frac=0.5, it=0, return_sorted=True)
        axs[0].plot(z[:, 0], z[:, 1], color=colors[i - 1], label=c)

        # plot histograms
        axs[i].hist(p[y == 1], range=(0, 1), color=colors[i - 1], bins=100, bottom=1)
        axs[i].get_yaxis().set_ticks([])
        axs[i].spines["top"].set_color("#dbdbdb")
        i += 1

    # legend to the left:
    # axs[0].legend(bbox_to_anchor=(1.05, 1),
    #               loc='upper left', borderaxespad=0.)
    
    # legend above plot:
    axs[0].legend(
        bbox_to_anchor=(0., 1.02, 1., .102), 
        fontsize='small',
        loc='lower left', mode="expand", 
        borderaxespad=0.)

    axs[0].set_ylabel("Observed Frequency")
    axs[int(group_n / 2)].set_ylabel(
        "Counts",
         horizontalalignment='right'
         )
    fig.align_ylabels()

    # super title
    # fig.suptitle(
    #     f"Calibration plot",
    #     x=fig.subplotpars.left,
    #     y=fig.subplotpars.top + 0.01,
    #     horizontalalignment="left",
    #     verticalalignment="bottom",
    #     fontweight="bold"
    # )

    axs[0].set_ylim(0, 1)
    plt.xlim(0, 1)
    plt.xlabel("Predicted Probability")
    plt.savefig(
        os.path.join(output_folder, hospital, "calibration.pdf"), 
        bbox_inches="tight",
        metadata={'CreationDate': None}
    )
    plt.savefig(
        os.path.join(output_folder, hospital, "calibration.png"), 
        bbox_inches="tight", 
        dpi=300
    )


def plot_auc_at_max_time_difference(time_dif_data, prediction_columns):
    fig, ax = plt.subplots()
    x = time_dif_data["max_time_difference"]
    colors = [
        'tab:blue',
        'tab:orange',
        'tab:green',
    ]

    if any(['AUROC' in column for column in time_dif_data.columns]):
        metric = 'AUROC'
    else:
        metric = 'AUPRC'

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
    ax.set_xlabel(
        "Allowed Time Difference between Prediction and Discharge\n(hours)"
    )
    ax.set_ylabel(f"{metric}")
    ax.legend(loc="lower right")

    # super title
    fig.suptitle(
        f"Performance by allowed Time Difference\nbetween Prediction and Discharge",
        x=fig.subplotpars.left,
        y=fig.subplotpars.top + 0.01,
        horizontalalignment="left",
        verticalalignment="bottom",
        fontweight="bold"
    )

    plt.savefig(
        os.path.join(output_folder, hospital, "AUC_at_different_max_time_difference.pdf"), 
        bbox_inches="tight",
        metadata={'CreationDate': None}
    )
    plt.savefig(
        os.path.join(output_folder, hospital, "AUC_at_different_max_time_difference.png"), 
        bbox_inches="tight", 
        dpi=300
    )


def plot_coverage_at_max_features(coverage_data):
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
        f"Feature importance coverage by number of features",
        x=fig.subplotpars.left,
        y=fig.subplotpars.top + 0.01,
        horizontalalignment="left",
        verticalalignment="bottom",
        fontweight="bold"
    )

    plt.savefig(
        os.path.join(output_folder, hospital, "coverage_at_different_max_features.pdf"), 
        bbox_inches="tight",
        metadata={'CreationDate': None}
    )
    plt.savefig(
        os.path.join(output_folder, hospital, "coverage_at_different_max_features.png"), 
        bbox_inches="tight", 
        dpi=300
    )


def plot_combined_curves(predictions, outcome, prediction_columns, curve="roc"):
    fig, ax = plt.subplots(
        figsize=(8, 6)
    )
    ax.margins(x=0, y=0)
    ax.xaxis.labelpad = 15
    ax.yaxis.labelpad = 15

    prev_x = None
    prev_y = None

    colors = [
        'tab:blue',
        'tab:orange',
        'tab:green',
        'tab:red',
        'tab:purple'
    ]

    i = 0
    for column_desc in prediction_columns:
        column = prediction_columns[column_desc]

        y_true = predictions.loc[
            ~pd.isnull(predictions[column]), outcome]
        y_pred = predictions.loc[
            ~pd.isnull(predictions[column]), column]

        if curve == "roc":
            suptitle = "Receiver Operating Characteristic"
            xlabel = "False Positive Rate (1 - Specificity)"
            ylabel = "True Positive Rate (Sensitivity)"
            
            # x_axis: fpr, y_axis: tpr
            x_axis, y_axis, thresholds = roc_curve(y_true, y_pred)
        elif curve == "prc":
            suptitle = "Precision-Recall Curve"
            xlabel = "Precision (Positive Predictive Value)"
            ylabel = "Recall (Sensitivity)"

            # x_axis: precion, y_axis: recall
            y_axis, x_axis, thresholds = precision_recall_curve(y_true, y_pred)
        
        observed_auc = auc(x_axis, y_axis)

        label=f"{column_desc} \n(AU{curve.upper()} = {observed_auc:#.2g})"
        ax.plot(x_axis, y_axis, label=label, color=colors[i], linewidth=2)
        if prev_y is None:
            ax.fill_between(x_axis, y_axis, 0, alpha=0.2)
        else:
            plt.fill(np.append(x_axis, prev_x[::-1]), np.append(y_axis, prev_y[::-1]),
                     color=colors[i], alpha=0.2)

        prev_x = x_axis
        prev_y = y_axis

        i += 1

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.axline([0, 0], slope=1, color='black', linestyle='--')
    plt.axis('square')

    ax.legend(
        loc='lower right',
        fontsize=10)
    
    plt.savefig(
        os.path.join(output_folder, hospital, f"combined_{curve}_curves.pdf"), 
        bbox_inches="tight",
        metadata={'CreationDate': None}
    )
    plt.savefig(
        os.path.join(output_folder, hospital, f"combined_{curve}_curves.png"), 
        bbox_inches="tight", 
        dpi=300
    )


def plot_performance_by_confidence(
        performance_table_medium_low_confidence, 
        performance_table_high_confidence):
    
    columns = performance_table_medium_low_confidence.columns
    if 'auroc' in columns: 
        metric = 'auroc'
    else:
        metric = 'auprc'
    
    high_confidence_auroc = performance_table_high_confidence[metric].values
    high_confidence_error_low = high_confidence_auroc - performance_table_high_confidence[f"{metric}_confidence_interval_lower"].values
    high_confidence_error_high = performance_table_high_confidence[f"{metric}_confidence_interval_upper"].values - high_confidence_auroc

    low_confidence_auroc = performance_table_medium_low_confidence[metric].values
    low_confidence_error_low = low_confidence_auroc - performance_table_medium_low_confidence[f"{metric}_confidence_interval_lower"].values
    low_confidence_error_high = performance_table_medium_low_confidence[f"{metric}_confidence_interval_upper"].values - low_confidence_auroc

    labels = performance_table_high_confidence.index.values

    labels = [label.replace('(', '\n(') for label in labels]
    labels = ['\n'.join(wrap(l, 12)) for l in labels]

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    ax.bar(x=x - width / 2, height=high_confidence_auroc,
           color="#00A36C",  # Jade
           yerr=[high_confidence_error_low, high_confidence_error_high],
           capsize=3,
           width=width, label='High Confidence')
    ax.bar(x=x + width / 2, height=low_confidence_auroc,
           yerr=[low_confidence_error_low, low_confidence_error_high],
           color="#FFAA33",  # Yellow Orange
           capsize=3,
           width=width, label='Medium/Low Confidence')

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
        bbox_to_anchor=(0., 1.02, 1., .102), 
        fontsize='small',
        loc='lower left', mode="expand", 
        borderaxespad=0.)

    plt.savefig(
        os.path.join(output_folder, hospital, "performance_by_confidence.pdf"), 
        bbox_inches="tight",
        metadata={'CreationDate': None}
    )
    plt.savefig(
        os.path.join(output_folder, hospital, "performance_by_confidence.png"), 
        bbox_inches="tight", 
        dpi=300
    )


def plot_bootstrap_auc_difference(
        axis,
        observed_difference,
        confidence_lower,
        confidence_upper,
        p_value,
        bootstrapped_differences,
        curve
):
    """
    Plot distribution of bootstrapped AUC differences
    """
    if p_value is None:
        return

    axis.hist(bootstrapped_differences, bins='auto')
    axis.axvline(observed_difference, color='r', linestyle='dashed', linewidth=2, label='Observed difference')
    axis.axvline(0, color='y', linestyle='dashed', linewidth=2, label='No difference')
    axis.axvline(confidence_lower, color='b', linestyle='dashed', linewidth=2, label='2.5 percentile')
    axis.axvline(confidence_upper, color='b', linestyle='dashed', linewidth=2, label='97.5 percentile')
    # axis.set_xlabel(f"Difference in {curve} AUC")
    # axis.set_ylabel(f"Frequency")
    
    # place a text box in upper right in axes coords
    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.75)
    p_text = f"$\\Delta_{{AU{curve.upper()}}} = {observed_difference:#.2g}$\n$p = {p_value:#.2g}$"
    axis.text(0.95, 0.95, p_text, transform=axis.transAxes,
    verticalalignment='top', horizontalalignment='right', bbox=props)

def full_extent(ax, pad=0.0):
    """
    Get the full extent of an axis, including axes labels, tick labels, and
    titles.
    """
    # For text objects, the figure needs to be drawn first, otherwise the extents
    # are undefined.
    ax.figure.canvas.draw()
    items = ax.get_xticklabels() + ax.get_yticklabels() 
    items += [ax, ax.title, ax.xaxis.label, ax.yaxis.label]
    # items += [ax, ax.title]
    bbox = Bbox.union([item.get_window_extent() for item in items])
    return bbox.expanded(1.0 + pad, 1.0 + pad)


def combined_auc_difference_plot(bootstrapped_data):
    """
    Plot a collection of comparisons of AUC differences in a tabular format
    """
    rows = list(bootstrapped_data.keys())
    columns = list(bootstrapped_data[rows[0]].keys())
    
    mosaic = []
    # plot bootstrapping distribution
    for row in rows:
        row_mosaic = []
        for column in bootstrapped_data[row]:
            row_mosaic.append(f"{row}:{column}")
        
        row_len_dif = len(rows) - len(row_mosaic) 
        if row_len_dif > 0:
            row_mosaic = row_len_dif * ["."] + row_mosaic
            
        mosaic.append(row_mosaic)
        
    fig, axs = plt.subplot_mosaic(
        mosaic=mosaic,
        sharex=True,
        sharey=True,
        layout="constrained",
        figsize=[12, 8]
    )

    axs[f"{rows[0]}:{columns[0]}"].set_xlim(-0.25, 0.55)

    # plot bootstrapping distribution
    for row in bootstrapped_data.keys():
        for column in bootstrapped_data[row].keys():
            
            (observed_difference, confidence_lower, confidence_upper, p_value, 
             bootstrapped_differences, curve) = bootstrapped_data[row][column]
            plot_bootstrap_auc_difference(
                axs[f"{row}:{column}"],
                observed_difference,
                confidence_lower,
                confidence_upper,
                p_value,
                bootstrapped_differences,
                curve
    )

    # show ticks for first graph of each row
    for row in mosaic:
        first = True
        for column_name in row:
            if column_name == '.':
                continue
            axs[column_name].tick_params(labelleft= True, labelbottom=True)
            if first:
                axs[column_name].set_xlabel(f"difference in AU{curve.upper()}")
                axs[column_name].set_ylabel(f"frequency")
            first = False

    
    # add_row_rectangles
    # row titles:
    x_padding = 0.02

    colors = [
        'tab:blue',
        'tab:orange',
        'tab:green',
        'tab:red',
        'tab:purple'
    ]


    row_titles = []
    row_header_containers = []
    extent_first_row = Bbox.union([full_extent(axs[col_name]) for col_name in mosaic[0]])
    extent_first_row = extent_first_row.transformed(fig.transFigure.inverted())
    extent_all = Bbox.union([full_extent(axs[col_name]) for row in mosaic for col_name in row if col_name != '.'])
    extent_all = extent_all.transformed(fig.transFigure.inverted())

    ymin_prev = np.nan
    for i, row in zip(range(0, len(bootstrapped_data.keys())), bootstrapped_data.keys()):
        column_names = [col for col in mosaic[i] if col != '.'] # remove 'empty' axis
        extent = Bbox.union([full_extent(axs[col_name]) for col_name in column_names])
        extent = extent.transformed(fig.transFigure.inverted())

        if ymin_prev != np.nan and extent.ymax > ymin_prev:
            ymin = extent.ymin
            height = extent.height - (extent.ymax - ymin_prev)
        else:
            ymin = extent.ymin
            height = extent.height

        row_titles.append(
            fig.text(
                x=extent_first_row.xmin - x_padding,
                y=(ymin + height/2),
                s="and\n".join(row.split("and ")),
                fontweight="bold",
                color=colors[i],
                transform=fig.transFigure,
                horizontalalignment='center',
                verticalalignment='center',
                rotation=90
            )
        )

        ymin_prev = extent.ymin

    ymin_prev = np.nan
    for i in range(0, len(mosaic)):
        column_names = [col for col in mosaic[i] if col != '.'] # remove 'empty' axis
        extent = Bbox.union([full_extent(axs[col_name]) for col_name in column_names])

        extent = extent.transformed(fig.transFigure.inverted())

        if ymin_prev != np.nan and extent.ymax > ymin_prev:
            ymin = extent.ymin
            height = extent.height - (extent.ymax - ymin_prev)
        else:
            ymin = extent.ymin
            height = extent.height

        # row dividers
        rect = Rectangle(
            xy=(extent_first_row.xmin, ymin),
            width=extent_all.width,
            height=height,
            facecolor=to_rgba(colors[i], 0.25),
            edgecolor=to_rgba('black', 1),
            zorder=-1, 
            transform=fig.transFigure)
        fig.patches.append(rect)

        # row header containers
        extent_label = Bbox.union([row_titles[0].get_window_extent()])
        extent_label = extent_label.transformed(fig.transFigure.inverted())
        width = 2 * (extent_first_row.xmin - (extent_label.xmin + extent_label.xmax) / 2)
        rect2 = Rectangle(
            xy=(extent_first_row.xmin - width, ymin),
            width=width,
            height=height,
            facecolor='black',
            edgecolor='white', 
            zorder=-1, 
            transform=fig.transFigure)
        fig.patches.append(rect2)
        row_header_containers.append(rect2)

        ymin_prev = extent.ymin

    # column titles:
    column_titles = []
    column_header_containers = []
    extent_last_column = Bbox.union([full_extent(axs[row[-1]]) for row in mosaic])
    extent_last_column = extent_last_column.transformed(fig.transFigure.inverted())
    y_padding = 0.025

    xmax_prev = np.nan
    for i in range(0, len(mosaic[0])):
        row_names = [row[i] for row in mosaic if row[i] != '.'] # remove 'empty' axis
        extent = Bbox.union([full_extent(axs[row_name]) for row_name in row_names])
        extent = extent.transformed(fig.transFigure.inverted())

        if xmax_prev != np.nan and extent.xmin < xmax_prev:
            xmin = xmax_prev
            width = extent.width - (xmax_prev - extent.xmin)
        else:
            xmin = extent.xmin
            width = extent.width

        column_titles.append(
            fig.text(
                x=(xmin + width/2),
                y=(extent_last_column.ymin + extent_last_column.height + y_padding),
                s="and\n".join(columns[i].split("and ")),
                fontweight="bold",
                color=colors[i + 1],
                transform=fig.transFigure,
                horizontalalignment='center',
                verticalalignment='center',
                rotation=0
            )
        )
        xmax_prev = xmin + width

    xmax_prev = np.nan
    for i in range(0, len(mosaic[0])):
        row_names = [row[i] for row in mosaic if row[i] != '.'] # remove 'empty' axis
        extent = Bbox.union([full_extent(axs[row_name]) for row_name in row_names])
        extent = extent.transformed(fig.transFigure.inverted())

        if xmax_prev != np.nan and extent.xmin < xmax_prev:
            xmin = xmax_prev
            width = extent.width - (xmax_prev - extent.xmin)
        else:
            xmin = extent.xmin
            width = extent.width

        # column dividers
        rect = Rectangle(
            xy=(xmin, extent_last_column.ymin),
            width=width,
            height=extent_last_column.height,
            facecolor=to_rgba(colors[i + 1], 0.25),
            edgecolor=to_rgba("black", 1),
            zorder=-1,
            transform=fig.transFigure)
        fig.patches.append(rect)
        
        # column header containers
        extent_label = Bbox.union([item.get_window_extent() for item in column_titles])
        extent_label = extent_label.transformed(fig.transFigure.inverted())

        height = 2 * ((extent_label.ymin + extent_label.ymax) / 2 - extent_last_column.ymax)
        rect2 = Rectangle(
            xy=(xmin, extent_last_column.ymax),
            width=width,
            height=height,
            facecolor='black',
            edgecolor='white',
            zorder=-1,
            transform=fig.transFigure)
        fig.patches.append(rect2)
        column_header_containers.append(rect2)

        xmax_prev = xmin + width

    # add "vs." label
    extent_row_header = Bbox.union([row_header_containers[0].get_window_extent()]).transformed(fig.transFigure.inverted())
    extent_column_header = Bbox.union([column_header_containers[0].get_window_extent()]).transformed(fig.transFigure.inverted())

    fig.text(
        x=(extent_row_header.xmin + extent_column_header.xmin)/2,
        y=(extent_row_header.ymax + extent_column_header.ymax)/2,
        s="vs.",
        fontsize="xx-large",
        color="black",
        transform=fig.transFigure,
        horizontalalignment='center',
        verticalalignment='center',
        rotation=45
    )

    # add legend
    extent_last_row_header = Bbox.union([row_header_containers[-1].get_window_extent()]).transformed(fig.transFigure.inverted())
    handles, labels = axs[f"{rows[0]}:{columns[0]}"].get_legend_handles_labels()  
    
    fig.legend(handles, labels, loc='lower left', bbox_to_anchor=[
        extent_last_row_header.xmax,
        extent_last_row_header.ymin,
    ])

    # # add 'super' title
    # fig.suptitle(
    #     f"Comparison of predictive performance by differences in AU{curve.upper()}",
    #     x=extent_row_header.xmin,
    #     y=extent_column_header.ymax + y_padding,
    #     verticalalignment="bottom",
    #     horizontalalignment="left",
    #     fontweight="bold"
    # )
            
    plt.savefig(
        os.path.join(output_folder, hospital, "combined_auc_difference.pdf"), 
        bbox_inches="tight",
        metadata={'CreationDate': None}
    )
    plt.savefig(
        os.path.join(output_folder, hospital, "combined_auc_difference.png"), 
        bbox_inches="tight", 
        dpi=300
    )