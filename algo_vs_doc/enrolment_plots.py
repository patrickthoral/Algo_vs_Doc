"""Plot cohort enrolment and its outcome-aware presentation statuses."""

from __future__ import annotations

from textwrap import wrap
from typing import Mapping, Sequence
from xml.etree import ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from algo_vs_doc import cohorts

ENROLMENT_STEP_LABELS = {
    "total_icu_admissions": "ICU admissions screened",
    "not_yet_discharged": "not yet discharged",
    "died_on_icu": "died in the ICU",
    "palliative_care": "palliative care",
    "do_not_readmit_policy": "do-not-readmit decision",
    "transferred_to_other_hospital": "transferred to another hospital",
    "discharged_home": "discharged home",
    "icu_los_lt_12h": "ICU length of stay under 12 hours",
    "missing_doc_prediction": "missing physician prediction",
    "missing_algo_prediction": "missing algorithm prediction",
}

NOT_ELIGIBLE_STEPS = (
    "palliative_care",
    "do_not_readmit_policy",
    "transferred_to_other_hospital",
    "discharged_home",
    "icu_los_lt_12h",
)

NOT_ELIGIBLE_LABELS = {
    "palliative_care": "received palliative care",
    "do_not_readmit_policy": "had a do-not-readmit decision",
    "transferred_to_other_hospital": "transferred to another hospital",
    "discharged_home": "were discharged home",
    "icu_los_lt_12h": "had length of stay under 12 hours",
}

MISSING_PREDICTION_STEPS = (
    "missing_doc_prediction",
    "missing_algo_prediction",
)

MISSING_PREDICTION_LABELS = {
    "missing_doc_prediction": "missing physician prediction",
    "missing_algo_prediction": "missing algorithm prediction",
}


def _validated_flow_reports(
    reports: Mapping[str, pd.DataFrame], cohort_order: Sequence[str]
) -> dict[str, pd.DataFrame]:
    selected = {name: reports[name].reset_index(drop=True) for name in cohort_order}
    required = {"step", "excluded", "remaining"}
    for name, report in selected.items():
        missing = required - set(report.columns)
        if missing:
            raise KeyError(
                f"{name} funnel report is missing columns: {sorted(missing)}"
            )

    steps = selected[cohort_order[0]]["step"].tolist()
    if not steps or steps[0] != "total_icu_admissions":
        raise ValueError("funnel must start with total_icu_admissions")
    for name, report in selected.items():
        if report["step"].tolist() != steps:
            raise ValueError(f"{name} funnel steps differ from the first cohort")
        remaining = report["remaining"].astype(int).tolist()
        excluded = report["excluded"].astype(int).tolist()
        if excluded[0] != 0:
            raise ValueError(f"{name} initial funnel step must exclude zero stays")
        for index in range(1, len(report)):
            if remaining[index - 1] - excluded[index] != remaining[index]:
                raise ValueError(
                    f"{name} funnel counts are inconsistent at {steps[index]}"
                )

    component_names = [name for name in cohort_order if name != "Combined"]
    if "Combined" in selected and component_names:
        for column in ("excluded", "remaining"):
            component_total = sum(
                (selected[name][column].astype(int) for name in component_names),
                start=pd.Series(0, index=selected["Combined"].index),
            )
            if not component_total.equals(selected["Combined"][column].astype(int)):
                raise ValueError(
                    f"Combined {column} counts do not equal the site totals"
                )
    return selected


def enrolment_flow_svg(
    reports: Mapping[str, pd.DataFrame],
    *,
    discharge_group_counts: Mapping[str, Mapping[str, int]] | None = None,
    cohort_order: Sequence[str] = cohorts.HOSPITALS,
) -> str:
    """Return a deterministic monochrome SVG with one panel per hospital.

    The SVG contains fixed, semantic element IDs and no timestamps, random identifiers, or
    generator metadata. Steps excluding zero stays in every panel are omitted. Identical reports
    therefore produce byte-identical output.
    """
    selected = _validated_flow_reports(reports, cohort_order)
    if discharge_group_counts is not None:
        for name in cohort_order:
            counts = discharge_group_counts[name]
            displayed = int(counts["nursing_ward"]) + int(
                counts["high_dependency_unit"]
            )
            included = int(selected[name].iloc[-1]["remaining"])
            if displayed != included:
                raise ValueError(
                    f"{name} discharge groups total {displayed}, expected {included} included stays"
                )
    first = selected[cohort_order[0]]
    steps = first["step"].tolist()
    flow_items: list[tuple[str, tuple[int, ...]]] = [(steps[0], (0,))]
    index = 1
    while index < len(steps):
        if tuple(steps[index : index + len(NOT_ELIGIBLE_STEPS)]) == NOT_ELIGIBLE_STEPS:
            indices = tuple(range(index, index + len(NOT_ELIGIBLE_STEPS)))
            flow_items.append(("not_eligible", indices))
            index += len(NOT_ELIGIBLE_STEPS)
        elif (
            tuple(steps[index : index + len(MISSING_PREDICTION_STEPS)])
            == MISSING_PREDICTION_STEPS
        ):
            indices = tuple(range(index, index + len(MISSING_PREDICTION_STEPS)))
            flow_items.append(("missing_predictions", indices))
            index += len(MISSING_PREDICTION_STEPS)
        else:
            flow_items.append((steps[index], (index,)))
            index += 1
    visible_items = [flow_items[0]] + [
        item
        for item in flow_items[1:]
        if any(
            sum(int(selected[name].loc[i, "excluded"]) for i in item[1]) != 0
            for name in cohort_order
        )
    ]
    width = 1000
    panel_top = 42
    stage_height = 52
    grouped_height = 144
    missing_predictions_height = 84
    horizontal_padding = 20
    detail_indent = 20
    minimum_arrow_length = 80
    arrow_clearance = 6
    row_gap = minimum_arrow_length + arrow_clearance
    panel_gap = 52
    main_x = 350
    main_width = 280
    excluded_x = 660
    excluded_width = 310
    item_heights = [
        grouped_height
        if step == "not_eligible"
        else missing_predictions_height
        if step == "missing_predictions"
        else stage_height
        for step, _ in visible_items
    ]
    discharge_split_height = (
        row_gap + stage_height if discharge_group_counts is not None else 0
    )
    panel_height = (
        sum(item_heights) + (len(visible_items) - 1) * row_gap + discharge_split_height
    )
    height = (
        panel_top
        + len(cohort_order) * panel_height
        + (len(cohort_order) - 1) * panel_gap
        + 28
    )

    def add(parent: ET.Element, tag: str, **attributes: object) -> ET.Element:
        normalized = {
            key.replace("_", "-"): f"{value:g}"
            if isinstance(value, float)
            else str(value)
            for key, value in attributes.items()
        }
        return ET.SubElement(parent, tag, normalized)

    svg = ET.Element(
        "svg",
        {
            "xmlns": "http://www.w3.org/2000/svg",
            "width": str(width),
            "height": str(height),
            "viewBox": f"0 0 {width} {height}",
            "role": "img",
            "aria-labelledby": "title description",
        },
    )
    title = add(svg, "title", id="title")
    title.text = "Study enrolment flow"
    description = add(svg, "desc", id="description")
    description.text = "Separate hospital panels showing eligible ICU discharges after each sequential exclusion."
    definitions = add(svg, "defs")
    marker = add(
        definitions,
        "marker",
        id="arrowhead",
        markerWidth=8,
        markerHeight=8,
        refX=7,
        refY=4,
        orient="auto",
    )
    add(marker, "path", id="arrowhead-path", d="M0,0 L8,4 L0,8 Z", fill="#000000")
    flow = add(
        svg,
        "g",
        id="flow",
        font_family="Arial, Helvetica, sans-serif",
        fill="#000000",
    )
    add(
        flow,
        "rect",
        id="figure-border",
        x=1,
        y=1,
        width=width - 2,
        height=height - 2,
        fill="none",
        stroke="#000000",
        stroke_width=1.5,
    )
    for divider_index in range(len(cohort_order) - 1):
        divider_y = (
            panel_top
            + (divider_index + 1) * panel_height
            + divider_index * panel_gap
            + panel_gap / 2
        )
        add(
            flow,
            "line",
            id=f"hospital-divider-{divider_index + 1}",
            x1=1,
            y1=divider_y,
            x2=width - 1,
            y2=divider_y,
            stroke="#000000",
            stroke_width=1.5,
        )
    for panel_index, name in enumerate(cohort_order):
        panel_id = name.lower()
        panel_y = panel_top + panel_index * (panel_height + panel_gap)
        panel = add(flow, "g", id=f"panel-{panel_id}")
        panel_title = add(
            panel,
            "text",
            id=f"panel-{panel_id}-title",
            x=25,
            y=panel_y + 18,
            font_size=18,
        )
        panel_title.text = name
        previous_main_bottom = None
        row_y = panel_y
        for position, (step, indices) in enumerate(visible_items):
            final_index = indices[-1]
            row_height = item_heights[position]
            main_y = row_y + (row_height - stage_height) / 2
            if step == "not_eligible":
                label = "not eligible"
            elif step == "missing_predictions":
                label = "missing prediction"
            else:
                label = ENROLMENT_STEP_LABELS.get(
                    step, step.replace("_", " ").capitalize()
                )
            if position == 0:
                stage_label = "admissions to intensive care unit"
            elif step == "died_on_icu":
                stage_label = "survived to discharge"
            elif step in {"missing_algo_prediction", "missing_predictions"}:
                stage_label = "included in analysis"
            else:
                stage_label = "eligible ICU discharges"
            remaining = int(selected[name].loc[final_index, "remaining"])
            if position:
                vertical_arrow_end = main_y - arrow_clearance
                branch_y = (previous_main_bottom + vertical_arrow_end) / 2
                add(
                    panel,
                    "path",
                    id=f"arrow-{panel_id}-{step}",
                    d=f"M490 {previous_main_bottom:g} V{vertical_arrow_end:g}",
                    stroke="#000000",
                    stroke_width=1.5,
                    fill="none",
                    marker_end="url(#arrowhead)",
                )
            stage = add(panel, "g", id=f"stage-{panel_id}-{step}")
            add(
                stage,
                "rect",
                id=f"stage-{panel_id}-{step}-box",
                x=main_x,
                y=main_y,
                width=main_width,
                height=stage_height,
                fill="#ffffff",
                stroke="#000000",
                stroke_width=1.5,
            )
            main_text_lines = wrap(f"{remaining:,} {stage_label}", width=32)
            main_first_y = (
                main_y + stage_height / 2 - (len(main_text_lines) - 1) * 8 + 5
            )
            stage_text = add(
                stage,
                "text",
                id=f"stage-{panel_id}-{step}-label",
                x=490,
                y=main_first_y,
                text_anchor="middle",
                font_size=14,
            )
            for line_index, text in enumerate(main_text_lines):
                tspan = add(stage_text, "tspan", x=490, dy=0 if line_index == 0 else 16)
                tspan.text = text
            if position:
                excluded = sum(int(selected[name].loc[i, "excluded"]) for i in indices)
                if step == "not_eligible":
                    exclusion_lines = [f"{excluded:,} not eligible"] + [
                        f"{int(selected[name].loc[i, 'excluded']):,} "
                        f"{NOT_ELIGIBLE_LABELS[str(first.loc[i, 'step'])]}"
                        for i in indices
                    ]
                elif step == "missing_predictions":
                    exclusion_lines = [f"{excluded:,} missing predictions"] + [
                        f"{int(selected[name].loc[i, 'excluded']):,} "
                        f"{MISSING_PREDICTION_LABELS[str(first.loc[i, 'step'])]}"
                        for i in indices
                    ]
                else:
                    exclusion_lines = [f"{excluded:,} {label}"]
                grouped = step in {"not_eligible", "missing_predictions"}
                exclusion_height = row_height if grouped else stage_height
                exclusion_y = branch_y - exclusion_height / 2
                box_x = excluded_x
                box_width = excluded_width
                text_x = box_x + horizontal_padding
                detail_x = text_x + detail_indent
                text_anchor = "start"
                connector = f"M490 {branch_y:g} H652"
                wrapped_exclusion_lines = []
                for logical_index, text in enumerate(exclusion_lines):
                    max_chars = 36
                    wrapped_exclusion_lines.extend(
                        (part, logical_index > 0)
                        for part in wrap(text, width=max_chars)
                    )
                line_spacing = 18
                first_text_y = (
                    exclusion_y
                    + exclusion_height / 2
                    - (len(wrapped_exclusion_lines) - 1) * line_spacing / 2
                    + 5
                )
                exclusion_group = add(panel, "g", id=f"excluded-{panel_id}-{step}")
                add(
                    exclusion_group,
                    "path",
                    id=f"excluded-{panel_id}-{step}-connector",
                    d=connector,
                    stroke="#000000",
                    stroke_width=1.5,
                    fill="none",
                    marker_end="url(#arrowhead)",
                )
                add(
                    exclusion_group,
                    "rect",
                    id=f"excluded-{panel_id}-{step}-box",
                    x=box_x,
                    y=exclusion_y,
                    width=box_width,
                    height=exclusion_height,
                    fill="#ffffff",
                    stroke="#000000",
                    stroke_width=1.5,
                )
                exclusion_text = add(
                    exclusion_group,
                    "text",
                    id=f"excluded-{panel_id}-{step}-label",
                    x=text_x,
                    y=first_text_y,
                    text_anchor=text_anchor,
                    font_size=13,
                )
                for line_index, (text, indented) in enumerate(wrapped_exclusion_lines):
                    tspan = add(
                        exclusion_text,
                        "tspan",
                        x=detail_x if indented else text_x,
                        dy=0 if line_index == 0 else line_spacing,
                    )
                    tspan.text = text
            previous_main_bottom = main_y + stage_height
            row_y += row_height + row_gap
        if discharge_group_counts is not None:
            split_box_y = previous_main_bottom + row_gap
            fork_y = previous_main_bottom + row_gap / 2
            counts = discharge_group_counts[name]
            discharge_groups = (
                (
                    "high-dependency-unit",
                    170,
                    int(counts["high_dependency_unit"]),
                    "high-dependency unit",
                ),
                ("nursing-ward", 520, int(counts["nursing_ward"]), "general ward"),
            )
            for group_id, box_x, count, destination in discharge_groups:
                box_center_x = box_x + excluded_width / 2
                group = add(panel, "g", id=f"included-{panel_id}-{group_id}")
                add(
                    group,
                    "path",
                    id=f"included-{panel_id}-{group_id}-connector",
                    d=(
                        f"M490 {previous_main_bottom:g} V{fork_y:g} "
                        f"H{box_center_x:g} V{split_box_y - arrow_clearance:g}"
                    ),
                    stroke="#000000",
                    stroke_width=1.5,
                    fill="none",
                    marker_end="url(#arrowhead)",
                )
                add(
                    group,
                    "rect",
                    id=f"included-{panel_id}-{group_id}-box",
                    x=box_x,
                    y=split_box_y,
                    width=excluded_width,
                    height=stage_height,
                    fill="#ffffff",
                    stroke="#000000",
                    stroke_width=1.5,
                )
                group_text = add(
                    group,
                    "text",
                    id=f"included-{panel_id}-{group_id}-label",
                    x=box_center_x,
                    y=split_box_y + stage_height / 2 - 3,
                    text_anchor="middle",
                    font_size=13,
                )
                first_line = add(group_text, "tspan", x=box_center_x, dy=0)
                first_line.text = f"{count:,} discharged to"
                second_line = add(group_text, "tspan", x=box_center_x, dy=18)
                second_line.text = destination
    ET.indent(svg, space="  ")
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        + ET.tostring(svg, encoding="unicode", short_empty_elements=True)
        + "\n"
    )


def enrolment_funnel(
    report: pd.DataFrame,
    *,
    title: str | None = None,
    ax: plt.Axes | None = None,
    color: str = "steelblue",
    excluded_color: str = "lightcoral",
) -> plt.Axes:
    """Plot remaining and excluded stay counts for each enrolment step.

    ``report`` contains ``step``, ``excluded``, and ``remaining`` columns.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 0.45 * len(report) + 1.5))

    y = np.arange(len(report))
    ax.barh(y, report["remaining"], color=color, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels([step.replace("_", " ") for step in report["step"]])
    ax.invert_yaxis()  # first step at the top: the cohort narrows downward
    ax.set_xlabel("Eligible ICU discharges")
    ax.grid(axis="x", alpha=0.3, zorder=0)

    for position, (remaining, excluded) in enumerate(
        zip(report["remaining"], report["excluded"])
    ):
        label = f"{remaining:,}" + (f"  (-{excluded:,})" if excluded else "")
        ax.text(
            remaining,
            position,
            "  " + label,
            va="center",
            ha="left",
            fontsize=9,
            color=excluded_color if excluded else "black",
        )

    ax.set_xlim(0, report["remaining"].max() * 1.25)
    if title:
        ax.set_title(title)
    return ax
