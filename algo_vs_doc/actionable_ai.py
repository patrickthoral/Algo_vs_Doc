"""Create an operational call-out linking risk thresholds to possible actions."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
from matplotlib.path import Path as MplPath
from matplotlib.textpath import TextPath
from matplotlib.patches import PathPatch
import pandas as pd
from matplotlib.patches import Arc, Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

from algo_vs_doc.icon_paths import ICON_PATHS


def _draw_icon(axis, name, bounds, *, color=None, transform, linewidth=None, zorder=5):
    """Draw generated SVG path data within an axes-coordinate bounding box."""
    x, y, width, height = bounds
    icon_transform = Affine2D().scale(width, height).translate(x, y) + transform
    gid = f"{name.replace('_', '-')}-icon"
    code_map = {
        "moveTo": MplPath.MOVETO,
        "lineTo": MplPath.LINETO,
        "curveTo": MplPath.CURVE4,
        "qCurveTo": MplPath.CURVE3,
    }
    for record in ICON_PATHS[name]:
        vertices, codes = [], []
        for command, operands in record["commands"]:
            if command == "endPath":
                continue
            if command == "closePath":
                vertices.append(vertices[0])
                codes.append(MplPath.CLOSEPOLY)
            else:
                vertices.extend(operands)
                codes.extend([code_map[command]] * len(operands))
        style = record["style"]
        source_fill = style.get("fill", "#000000").lower()
        source_stroke = style.get("stroke", "none").lower()

        def paint(source):
            if source == "none":
                return "none"
            if color is not None and source not in {"#ffffff", "white"}:
                return color
            return source

        fill = paint(source_fill)
        stroke = paint(source_stroke)
        patch = PathPatch(
            MplPath(vertices, codes),
            transform=icon_transform,
            facecolor=fill,
            edgecolor=stroke,
            linewidth=(
                linewidth
                if linewidth is not None
                else max(0.5, 50 * record["linewidth"])
            )
            if stroke != "none"
            else 0,
            joinstyle="round",
            capstyle="round",
            zorder=zorder,
            clip_on=False,
        )
        patch.set_gid(gid)
        axis.add_patch(patch)


REQUIRED_COLUMNS = {
    "cohort",
    "prediction_source",
    "threshold",
    "flagged_n",
    "true_positive",
    "false_negative",
    "alert_rate",
    "sensitivity",
    "specificity",
    "operating_point",
    "n",
}

COLORS = {
    "prediction": ("#5400F8", "#F1EAFF"),
    "capacity": ("#D47A00", "#FFF3E0"),
    "reassessment": ("#009E73", "#E8F6F1"),
    "actions": ("#7A5195", "#F3EDF7"),
    "evaluation": ("#4D6478", "#EEF2F5"),
}


def _card(axis, x, y, width, height, title, *, color, background) -> None:
    axis.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.008",
            linewidth=1.2,
            edgecolor=color,
            facecolor=background,
            transform=_content_transform(axis),
        )
    )
    axis.text(
        x + width / 2,
        y + height - 0.055,
        title,
        color=color,
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        multialignment="center",
        transform=_content_transform(axis),
    )


def _arrow(axis, start, end) -> None:
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=18,
            linewidth=2.2,
            color="#4D6478",
            transform=_content_transform(axis),
        )
    )


def _content_transform(axis):
    return getattr(axis, "_pathway_transform", axis.transAxes)


def _icon_canvas(axis, center):
    """Return a square inset axis so wide layouts do not distort icons."""
    x, y = center
    icon = axis.inset_axes(
        [x - 0.027, y - 0.079, 0.054, 0.158], transform=_content_transform(axis)
    )
    icon.set_xlim(0, 1)
    icon.set_ylim(0, 1)
    icon.set_aspect("equal")
    icon.axis("off")
    return icon


def _prediction_icon(axis, center, color) -> None:
    x, y = center
    icon = axis.inset_axes(
        [x - 0.033, y - 0.0957, 0.066, 0.1914],
        transform=_content_transform(axis),
    )
    icon.set(xlim=(0, 1), ylim=(0, 1), aspect="equal")
    icon.axis("off")
    gradient = mpl.colors.LinearSegmentedColormap.from_list(
        "risk", ["#2CA25F", "#F9D85E", "#D73027"]
    )
    segments = 80
    for index in range(segments):
        upper_angle = 160 - 140 * index / segments
        lower_angle = 160 - 140 * (index + 1) / segments
        icon.add_patch(
            Arc(
                (0.5, 0.46),
                0.72,
                0.72,
                theta1=lower_angle - 0.15,
                theta2=upper_angle + 0.15,
                linewidth=5.85,
                color=gradient(index / (segments - 1)),
            )
        )
    icon.plot(
        [0.5, 0.73],
        [0.46, 0.65],
        color=color,
        linewidth=2,
    )
    icon.add_patch(Circle((0.5, 0.46), 0.055, color=color))


def _reassessment_icon(axis, center, color) -> None:
    icon = _icon_canvas(axis, center)
    icon.add_patch(
        Rectangle(
            (0.18, 0.1),
            0.64,
            0.78,
            linewidth=1.7,
            edgecolor=color,
            facecolor="none",
        )
    )
    for offset in (0.68, 0.48, 0.28):
        icon.plot(
            [0.29, 0.36, 0.44],
            [offset, offset - 0.07, offset + 0.07],
            color=color,
            linewidth=1.3,
        )
        icon.plot(
            [0.51, 0.7],
            [offset, offset],
            color=color,
            linewidth=1.3,
        )


def _evaluation_icon(axis, center, color) -> None:
    icon = _icon_canvas(axis, center)
    icon.add_patch(
        Circle(
            (0.43, 0.57),
            0.28,
            fill=False,
            linewidth=1.8,
            edgecolor=color,
        )
    )
    icon.plot(
        [0.63, 0.86],
        [0.37, 0.14],
        color=color,
        linewidth=2,
    )


def _action_cards(axis, *, x, y, width) -> None:
    actions = (
        ("Enhanced Handover", "#F9D85E", "#263746"),
        ("Increased Ward Monitoring", "#F5A742", "#263746"),
        ("Critical-care outreach", "#E76F51", "white"),
        ("High-dependency setting", "#C43D3D", "white"),
    )
    for index, (label, background, text_color) in enumerate(actions):
        card_y = y + 0.045 + (3 - index) * 0.112
        axis.add_patch(
            FancyBboxPatch(
                (x + 0.012, card_y),
                width - 0.024,
                0.086,
                boxstyle="round,pad=0.005",
                linewidth=0,
                facecolor=background,
                transform=_content_transform(axis),
            )
        )
        axis.text(
            x + width / 2,
            card_y + 0.043,
            label,
            ha="center",
            va="center",
            fontsize=8.2,
            color=text_color,
            transform=_content_transform(axis),
        )


def _threshold_cards(axis, rows, *, x, y, width, color) -> None:
    badge_colors = ("#F9D85E", "#F5A742", "#E76F51", "#C43D3D")
    rows = rows.sort_values("threshold", kind="stable")
    if rows["n"].nunique() != 1:
        raise ValueError("operating points must use the same cohort denominator")
    _draw_icon(
        axis,
        "icu_bed",
        (x - 0.177, y + 0.265, 0.047, 0.10),
        color=COLORS["prediction"][0],
        transform=_content_transform(axis),
        linewidth=0.7,
    )
    axis.text(
        x - 0.085,
        y + 0.315,
        f"{int(rows['n'].iloc[0])}\nICU discharges",
        fontsize=8,
        color="black",
        ha="center",
        va="center",
        transform=_content_transform(axis),
    )
    badge_width = 0.025
    transform = _content_transform(axis)
    origin = transform.transform((0, 0))
    unit = transform.transform((1, 1)) - origin
    badge_height = badge_width * unit[0] / unit[1]
    metric_gradients = {
        "Se": mpl.colors.LinearSegmentedColormap.from_list(
            "sensitivity", ["#EFF8ED", "#83C786"]
        ),
        "Sp": mpl.colors.LinearSegmentedColormap.from_list(
            "specificity", ["#EDF5FC", "#86B9DE"]
        ),
    }
    for index, row in enumerate(rows.itertuples()):
        card_y = y + 0.045 + (3 - index) * 0.115
        events = int(row.true_positive + row.false_negative)
        background = badge_colors[index]
        badge_text_color = "#263746" if index < 2 else "white"
        axis.add_patch(
            FancyBboxPatch(
                (x + 0.014, card_y),
                width - 0.028,
                0.09,
                boxstyle="round,pad=0.006",
                linewidth=0.9,
                edgecolor=color,
                facecolor="white",
                transform=_content_transform(axis),
            )
        )
        axis.add_patch(
            FancyBboxPatch(
                (x + 0.018, card_y + (0.09 - badge_height) / 2),
                badge_width,
                badge_height,
                boxstyle="round,pad=0.004",
                linewidth=0,
                facecolor=background,
                transform=_content_transform(axis),
            )
        )
        threshold_label = f"{100 * row.threshold:.0f}%"
        if row.operating_point == "Youden index":
            threshold_label = f"{100 * row.threshold:.1f}%\nYouden"
            badge_text_color = "white"
        axis.text(
            x + 0.018 + badge_width / 2,
            card_y + 0.045,
            threshold_label,
            color=badge_text_color,
            ha="center",
            va="center",
            fontsize=6.7,
            fontweight="bold",
            linespacing=1.0,
            transform=_content_transform(axis),
        )
        count_center = (x + 0.018 + badge_width + x + 0.103) / 2
        for text, offset in (
            (f"{int(row.flagged_n)} selected", 0.065),
            (f"{int(row.true_positive)}/{events} events", 0.025),
        ):
            axis.text(
                count_center,
                card_y + offset,
                text,
                color="black",
                ha="center",
                va="center",
                fontsize=7.0,
                transform=_content_transform(axis),
            )
        for abbreviation, value, offset in (
            ("Se", row.sensitivity, 0.048),
            ("Sp", row.specificity, 0.008),
        ):
            axis.add_patch(
                FancyBboxPatch(
                    (x + 0.103, card_y + offset),
                    0.041,
                    0.034,
                    boxstyle="round,pad=0.002",
                    linewidth=0.5,
                    edgecolor="#6F7C83",
                    facecolor=metric_gradients[abbreviation](
                        max(0.0, min(1.0, float(value)))
                    ),
                    transform=_content_transform(axis),
                )
            )
            axis.text(
                x + 0.1235,
                card_y + offset + 0.017,
                f"{abbreviation} = {100 * value:.1f}%",
                color="black",
                fontsize=6.5,
                ha="center",
                va="center",
                transform=_content_transform(axis),
            )


def actionable_ai_svg(
    threshold_performance: pd.DataFrame,
    output_path,
    *,
    cohort: str = "Combined",
    prediction_source: str = "Model at physician prediction",
    pdf_path=None,
) -> Path:
    """Write SVG and optional PDF using one Matplotlib rendering path."""
    missing = sorted(REQUIRED_COLUMNS - set(threshold_performance.columns))
    if missing:
        raise KeyError(f"fixed-threshold results are missing {missing}")
    selected = threshold_performance.loc[
        threshold_performance["cohort"].eq(cohort)
        & threshold_performance["prediction_source"].eq(prediction_source)
        & threshold_performance["operating_point"].isin(
            [
                "5% risk threshold",
                "10% risk threshold",
                "20% risk threshold",
                "Youden index",
            ]
        )
    ].copy()
    required_points = {
        "5% risk threshold",
        "10% risk threshold",
        "20% risk threshold",
        "Youden index",
    }
    if set(selected["operating_point"]) != required_points:
        raise ValueError(
            "call-out requires 5%, 10%, and 20% risk thresholds and a Youden-index threshold"
        )

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with mpl.rc_context(
        {
            "font.family": "Arial",
            "svg.hashsalt": f"actionable-ai-{cohort}",
            "pdf.fonttype": 42,
        }
    ):
        figure = plt.figure(figsize=(11.5, 7.5))
        axis = figure.add_axes([0.03, 0.32, 0.94, 0.62])
        axis._pathway_transform = Affine2D().scale(1.25, 1) + axis.transAxes
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.axis("off")
        axis.text(
            0.4,
            0.95,
            "From Prediction to Action",
            ha="center",
            va="center",
            fontsize=16,
            fontweight="bold",
            color="#263746",
            transform=_content_transform(axis),
        )

        y, height = 0.22, 0.64
        stage_width = 0.16
        stages = {
            "prediction": (0.02, stage_width, "Risk\nPrediction"),
            "capacity": (0.22, stage_width, "Define Risk\nThreshold"),
            "reassessment": (0.42, stage_width, "Clinical\nReassessment"),
            "actions": (0.62, stage_width, "Possible\nActions"),
        }
        for name, (x, width, title) in stages.items():
            color, background = COLORS[name]
            _card(axis, x, y, width, height, title, color=color, background=background)

        _prediction_icon(axis, (0.10, 0.65), COLORS["prediction"][0])
        axis.text(
            0.10,
            0.405,
            "Continuous model\nrisk estimate",
            ha="center",
            va="center",
            fontsize=10,
            color="black",
            transform=_content_transform(axis),
        )
        _threshold_cards(
            axis,
            selected,
            x=0.22,
            y=y,
            width=stage_width,
            color=COLORS["capacity"][0],
        )
        _reassessment_icon(axis, (0.50, 0.65), COLORS["reassessment"][0])
        axis.text(
            0.50,
            0.445,
            "Review discharge readiness,\nreversible abnormalities,\nand goals of care",
            ha="center",
            va="center",
            fontsize=9.2,
            color="#263746",
            transform=_content_transform(axis),
        )
        axis.text(
            0.70,
            0.72,
            "Increasing Intervention Intensity",
            ha="center",
            va="center",
            fontsize=9.2,
            color="black",
            transform=_content_transform(axis),
        )
        _action_cards(axis, x=0.62, y=y, width=stage_width)
        evaluation = figure.add_axes([0.03, 0.18, 0.94, 0.25])
        evaluation.set(xlim=(0, 1), ylim=(0, 1))
        evaluation.axis("off")
        evaluation.text(
            0.5,
            0.96,
            "Evaluation",
            ha="center",
            va="top",
            fontsize=12,
            fontweight="bold",
            color=COLORS["evaluation"][0],
        )
        for index, label in enumerate(
            ("Clinical Outcomes", "Workload and Capacity", "Costs")
        ):
            left = 0.04 + index * 0.32
            evaluation.add_patch(
                FancyBboxPatch(
                    (left, 0.12),
                    0.28,
                    0.61,
                    boxstyle="round,pad=0.012",
                    edgecolor=COLORS["evaluation"][0],
                    facecolor=COLORS["evaluation"][1],
                )
            )
            center = left + 0.14
            icon = evaluation.inset_axes([center - 0.033, 0.40, 0.066, 0.30])
            icon.set(xlim=(0, 1), ylim=(0, 1), aspect="equal")
            icon.axis("off")
            color = COLORS["evaluation"][0]
            if index == 0:
                icon.add_patch(Circle((0.38, 0.76), 0.14, color=color))
                icon.add_patch(
                    FancyBboxPatch(
                        (0.16, 0.16),
                        0.44,
                        0.39,
                        boxstyle="round,pad=0.02",
                        linewidth=0,
                        facecolor=color,
                    )
                )
                _draw_icon(
                    icon,
                    "outcome_arrow",
                    (0.326, 0.086, 0.768, 0.768),
                    transform=icon.transData,
                )
            elif index == 1:
                icon.add_patch(Circle((0.5, 0.64), 0.17, color=color))
                icon.add_patch(
                    FancyBboxPatch(
                        (0.22, 0.12),
                        0.56,
                        0.30,
                        boxstyle="round,pad=0.04",
                        linewidth=0,
                        facecolor=color,
                    )
                )
                icon.add_patch(
                    Rectangle(
                        (0.33, 0.77),
                        0.34,
                        0.17,
                        facecolor="white",
                        edgecolor=color,
                        linewidth=1.5,
                    )
                )
                icon.plot([0.44, 0.56], [0.86, 0.86], color=color, linewidth=1.8)
                icon.plot([0.5, 0.5], [0.81, 0.91], color=color, linewidth=1.8)
                icon.plot(
                    [0.37, 0.5, 0.63], [0.40, 0.26, 0.40], color="white", linewidth=1.2
                )
            else:
                for coin_center in ((0.65, 0.65), (0.50, 0.50), (0.35, 0.35)):
                    icon.add_patch(
                        Circle(
                            coin_center,
                            0.29,
                            facecolor=COLORS["evaluation"][1],
                            edgecolor=color,
                            linewidth=1.7,
                        )
                    )
                euro = TextPath((0, 0), "\u20ac", size=1)
                bounds = euro.get_extents()
                scale = 0.34 / bounds.height
                transform = (
                    Affine2D()
                    .translate(
                        -bounds.x0 - bounds.width / 2, -bounds.y0 - bounds.height / 2
                    )
                    .scale(scale)
                    .translate(0.33, 0.35)
                )
                icon.add_patch(
                    PathPatch(
                        euro,
                        facecolor=color,
                        linewidth=0,
                        transform=transform + icon.transData,
                    )
                )
            evaluation.text(
                center,
                0.27,
                label,
                ha="center",
                va="center",
                fontsize=10,
                color="black",
            )

        for start, end in (
            ((0.187, 0.54), (0.213, 0.54)),
            ((0.387, 0.54), (0.413, 0.54)),
            ((0.587, 0.54), (0.613, 0.54)),
        ):
            _arrow(axis, start, end)

        figure.savefig(
            target, format="svg", bbox_inches="tight", metadata={"Date": None}
        )
        if pdf_path is not None:
            pdf_target = Path(pdf_path)
            pdf_target.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(
                pdf_target,
                format="pdf",
                bbox_inches="tight",
                metadata={"CreationDate": None},
            )
        plt.close(figure)
    return target
