"""Render analysis result frames as editable Microsoft Word tables."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
import re

import pandas as pd
from docx import Document
from docx.enum.section import WD_ORIENT, WD_SECTION
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Mm, Pt, RGBColor


@dataclass(frozen=True)
class RichCell:
    """Cell content with an optional p-value on a second line."""

    estimate: str
    p_value: str | None = None


@dataclass(frozen=True)
class WordTable:
    """Content and presentation metadata for one Word table."""

    title: str
    data: pd.DataFrame
    notes: Sequence[str] = ()
    section_starts: Sequence[str] = ()
    indented_rows: Sequence[str] = ()
    row_indent_levels: Mapping[str, int] | None = None
    row_indent_levels_by_position: Mapping[int, int] | None = None
    header_groups: Sequence[tuple[str, Sequence[str]]] = ()
    header_labels: Sequence[str] = ()
    font_size: float | None = None
    note_font_size: float = 9
    cell_margin_inches: float | None = None
    landscape: bool = False
    cohort_sections: bool = False
    multiline_intervals: bool = False
    style: str | None = "Table Grid"
    header_fill: str | None = None
    header_font_color: str | None = None
    # A sequence groups the first column (border/merge/fill); later columns are left-aligned only.
    group_column: str | Sequence[str] | None = None
    group_fill: str | None = None
    group_font_color: str | None = None
    merge_group_cells: bool = False
    merge_repeated_columns: Sequence[str] = ()
    column_widths: Sequence[float] = ()
    preferred_column_widths: Mapping[int, float] | None = None


def _set_cell_margins(cell, *, top=80, start=100, bottom=80, end=100) -> None:
    properties = cell._tc.get_or_add_tcPr()
    margins = properties.first_child_found_in("w:tcMar")
    if margins is None:
        margins = OxmlElement("w:tcMar")
        properties.append(margins)
    for name, value in (
        ("top", top),
        ("start", start),
        ("bottom", bottom),
        ("end", end),
    ):
        node = margins.find(qn(f"w:{name}"))
        if node is None:
            node = OxmlElement(f"w:{name}")
            margins.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def _set_table_cell_margins(table, margin_inches: float) -> None:
    """Set a table-wide default cell margin in inches."""
    if margin_inches < 0:
        raise ValueError("cell_margin_inches must be non-negative")
    value = str(round(margin_inches * 1440))
    properties = table._tbl.tblPr
    margins = properties.first_child_found_in("w:tblCellMar")
    if margins is None:
        margins = OxmlElement("w:tblCellMar")
        properties.append(margins)
    for name in ("top", "start", "bottom", "end"):
        node = margins.find(qn(f"w:{name}"))
        if node is None:
            node = OxmlElement(f"w:{name}")
            margins.append(node)
        node.set(qn("w:w"), value)
        node.set(qn("w:type"), "dxa")


def _set_table_column_widths(table, widths_inches: Sequence[float]) -> None:
    """Fix the Word table grid and size merged cells to their full grid span."""
    if any(width <= 0 for width in widths_inches):
        raise ValueError("column_widths must contain positive values")

    widths_twips = [round(width * 1440) for width in widths_inches]
    properties = table._tbl.tblPr
    layout = properties.first_child_found_in("w:tblLayout")
    if layout is None:
        layout = OxmlElement("w:tblLayout")
        properties.append(layout)
    layout.set(qn("w:type"), "fixed")

    table_width = properties.first_child_found_in("w:tblW")
    if table_width is None:
        table_width = OxmlElement("w:tblW")
        properties.append(table_width)
    table_width.set(qn("w:w"), str(sum(widths_twips)))
    table_width.set(qn("w:type"), "dxa")

    grid = table._tbl.tblGrid
    for grid_column in list(grid):
        grid.remove(grid_column)
    for width in widths_twips:
        grid_column = OxmlElement("w:gridCol")
        grid_column.set(qn("w:w"), str(width))
        grid.append(grid_column)

    for row in table._tbl.tr_lst:
        column_index = 0
        for cell in row.tc_lst:
            properties = cell.get_or_add_tcPr()
            grid_span = properties.find(qn("w:gridSpan"))
            span = int(grid_span.get(qn("w:val"))) if grid_span is not None else 1
            if column_index + span > len(widths_twips):
                raise ValueError("table cell span exceeds configured column widths")
            cell_width = properties.find(qn("w:tcW"))
            if cell_width is None:
                cell_width = OxmlElement("w:tcW")
                properties.append(cell_width)
            cell_width.set(
                qn("w:w"), str(sum(widths_twips[column_index : column_index + span]))
            )
            cell_width.set(qn("w:type"), "dxa")
            column_index += span
        if column_index != len(widths_twips):
            raise ValueError("table row does not cover all configured column widths")


def _set_preferred_column_widths(table, widths_inches: Mapping[int, float]) -> None:
    """Set preferred widths while allowing Word to autofit the other columns."""
    column_count = len(table.columns)
    for column_index, width in widths_inches.items():
        if column_index < 0 or column_index >= column_count:
            raise ValueError("preferred column-width index is outside the table")
        if width <= 0:
            raise ValueError("preferred column widths must be positive")
        table.columns[column_index].width = Inches(width)
        for row in table.rows:
            row.cells[column_index].width = Inches(width)


def _set_repeat_header(row) -> None:
    properties = row._tr.get_or_add_trPr()
    repeat = OxmlElement("w:tblHeader")
    repeat.set(qn("w:val"), "true")
    properties.append(repeat)


def _set_cell_border(cell, edge: str, *, size=8, color="000000") -> None:
    properties = cell._tc.get_or_add_tcPr()
    borders = properties.first_child_found_in("w:tcBorders")
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        properties.append(borders)
    border = borders.find(qn(f"w:{edge}"))
    if border is None:
        border = OxmlElement(f"w:{edge}")
        borders.append(border)
    border.set(qn("w:val"), "single")
    border.set(qn("w:sz"), str(size))
    border.set(qn("w:color"), color)


def _set_cell_fill(cell, color: str) -> None:
    properties = cell._tc.get_or_add_tcPr()
    shading = properties.find(qn("w:shd"))
    if shading is None:
        shading = OxmlElement("w:shd")
        properties.append(shading)
    shading.set(qn("w:fill"), color)


def _set_font_color(cell, color: str) -> None:
    rgb = RGBColor.from_string(color)
    for paragraph in cell.paragraphs:
        for run in paragraph.runs:
            run.font.color.rgb = rgb


def _group_ranges(values: Sequence[str]) -> list[tuple[int, int, str]]:
    """Return inclusive ranges for consecutive equal, non-empty labels."""
    ranges = []
    start = 0
    while start < len(values):
        label = values[start]
        end = start
        while end + 1 < len(values) and values[end + 1] == label:
            end += 1
        if label:
            ranges.append((start, end, label))
        start = end + 1
    return ranges


def _sanitize_display_text(value) -> str:
    """Replace formatted summaries containing an undefined numeric value."""
    text = str(value)
    return "-" if re.search(r"\bnan\b", text, flags=re.IGNORECASE) else text


def _is_rich_cell(value) -> bool:
    """Recognize RichCell instances retained across notebook module reloads."""
    value_type = type(value)
    return isinstance(value, RichCell) or (
        value_type.__module__ == RichCell.__module__
        and value_type.__name__ == RichCell.__name__
        and hasattr(value, "estimate")
        and hasattr(value, "p_value")
    )


def _display_value(value) -> str:
    if _is_rich_cell(value):
        return _sanitize_display_text(value.estimate)
    return "" if pd.isna(value) else _sanitize_display_text(value)


def _display_column(column) -> str:
    """Suppress pandas CSV placeholders for intentionally blank headers."""
    value = str(column)
    return "" if value.startswith("Unnamed:") else value


def _set_cell_value(cell, value) -> None:
    """Write plain or rich cell content, italicizing the p symbol."""
    if not _is_rich_cell(value):
        cell.text = _display_value(value)
        return

    estimate = _sanitize_display_text(value.estimate)
    cell.text = estimate
    if estimate != "-" and value.p_value is not None:
        p_value = _sanitize_display_text(value.p_value)
        paragraph = cell.paragraphs[0]
        p_run = paragraph.add_run("\np")
        p_run.italic = True
        paragraph.add_run(f" = {p_value}")


def _merge_repeated_cells(
    table, frame, column: str, parent_columns: Sequence[str] = ()
) -> None:
    if column not in frame:
        raise KeyError(f"merge column {column!r} is not present")
    column_index = frame.columns.get_loc(column)
    values = frame[column].map(_display_value).tolist()
    group_keys = [
        tuple(_display_value(frame.iloc[index][parent]) for parent in parent_columns)
        + (label,)
        for index, label in enumerate(values)
    ]
    for start, end, _ in _group_ranges(group_keys):
        if end == start:
            continue
        label = values[start]
        first_row = start + 1
        last_row = end + 1
        for row_index in range(first_row + 1, last_row + 1):
            table.cell(row_index, column_index).text = ""
        cell = table.cell(first_row, column_index).merge(
            table.cell(last_row, column_index)
        )
        cell.text = label
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        cell.paragraphs[0].alignment = (
            WD_ALIGN_PARAGRAPH.LEFT if column_index == 0 else WD_ALIGN_PARAGRAPH.CENTER
        )


def _add_table(document: Document, specification: WordTable) -> None:
    title = document.add_paragraph(specification.title)
    title.paragraph_format.space_after = Pt(4)

    frame = specification.data
    cohort_values = None
    if specification.cohort_sections and "Cohort" in frame:
        cohort_values = frame["Cohort"].map(_display_value).tolist()
        frame = frame.drop(columns="Cohort")
        specification = replace(
            specification,
            merge_repeated_columns=tuple(
                c for c in specification.merge_repeated_columns if c != "Cohort"
            ),
        )
    if specification.multiline_intervals:

        def split_interval(value):
            if _is_rich_cell(value):
                return RichCell(value.estimate.replace(" [", "\n[", 1), value.p_value)
            if isinstance(value, str) and value.lstrip()[:1] in "-0123456789":
                return value.replace(" [", "\n[", 1)
            return value

        frame = frame.map(split_interval)
    if specification.header_labels and len(specification.header_labels) != len(
        frame.columns
    ):
        raise ValueError("header_labels must contain one label per table column")
    header_labels = specification.header_labels or tuple(
        map(_display_column, frame.columns)
    )
    table = document.add_table(rows=1, cols=len(frame.columns))
    if specification.cell_margin_inches is not None:
        _set_table_cell_margins(table, specification.cell_margin_inches)

    def set_default_cell_margins(cell) -> None:
        if specification.cell_margin_inches is None:
            _set_cell_margins(cell)

    if specification.style:
        table.style = specification.style
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = not bool(specification.column_widths)
    header = table.rows[0]
    _set_repeat_header(header)

    for index, column in enumerate(frame.columns):
        cell = header.cells[index]
        cell.text = header_labels[index]
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        cell.paragraphs[0].alignment = (
            WD_ALIGN_PARAGRAPH.LEFT if index == 0 else WD_ALIGN_PARAGRAPH.CENTER
        )
        set_default_cell_margins(cell)
        _set_cell_border(cell, "top")
        _set_cell_border(cell, "bottom")
        if specification.header_fill:
            _set_cell_fill(cell, specification.header_fill)
        for run in cell.paragraphs[0].runs:
            run.bold = True
        if specification.header_font_color:
            _set_font_color(cell, specification.header_font_color)

    for row_position, values in enumerate(frame.itertuples(index=False, name=None)):
        row = table.add_row()
        for index, value in enumerate(values):
            cell = row.cells[index]
            _set_cell_value(cell, value)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            set_default_cell_margins(cell)
            cell.paragraphs[0].alignment = (
                WD_ALIGN_PARAGRAPH.LEFT if index == 0 else WD_ALIGN_PARAGRAPH.CENTER
            )
        label = _display_value(values[0]).strip() if values else ""
        indent_level = (specification.row_indent_levels_by_position or {}).get(
            row_position,
            (specification.row_indent_levels or {}).get(
                label, 1 if label in specification.indented_rows else 0
            ),
        )
        if indent_level:
            _set_cell_margins(row.cells[0], start=200 * indent_level)
            for paragraph in row.cells[0].paragraphs:
                paragraph.paragraph_format.left_indent = Inches(0.09 * indent_level)
        if label in specification.section_starts:
            for cell in row.cells:
                _set_cell_border(cell, "top", size=12)

    group_columns: tuple[str, ...] = (
        (specification.group_column,)
        if isinstance(specification.group_column, str)
        else tuple(specification.group_column or ())
    )
    primary_group_column = group_columns[0] if group_columns else None
    if group_columns:
        for column in group_columns:
            if column not in frame:
                raise KeyError(f"group column {column!r} is not present")
            header.cells[frame.columns.get_loc(column)].paragraphs[
                0
            ].alignment = WD_ALIGN_PARAGRAPH.LEFT

        left_aligned_columns = group_columns[1:]
        group_index = frame.columns.get_loc(primary_group_column)
        group_values = frame[primary_group_column].map(_display_value).tolist()
        for start, end, label in _group_ranges(group_values):
            first_row = start + 1
            last_row = end + 1
            for cell in table.rows[first_row].cells:
                _set_cell_border(cell, "top", size=12)
            group_cell = table.cell(first_row, group_index)
            if specification.merge_group_cells and last_row > first_row:
                for row_index in range(first_row + 1, last_row + 1):
                    table.cell(row_index, group_index).text = ""
                group_cell = group_cell.merge(table.cell(last_row, group_index))
                group_cell.text = label
            group_cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            group_cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.LEFT
            if specification.group_fill:
                _set_cell_fill(group_cell, specification.group_fill)
            if specification.group_font_color:
                _set_font_color(group_cell, specification.group_font_color)

        # Group columns after the first are left-aligned only: no border, merge, or fill.
        for column in left_aligned_columns:
            column_index = frame.columns.get_loc(column)
            for row in table.rows[1:]:
                row.cells[column_index].paragraphs[
                    0
                ].alignment = WD_ALIGN_PARAGRAPH.LEFT

    parent_columns: list[str] = []
    merge_frame = frame
    if cohort_values is not None:
        merge_frame = frame.assign(_cohort_section=cohort_values)
        parent_columns.append("_cohort_section")
    for column in specification.merge_repeated_columns:
        # Skip only when merge_group_cells already merged this column above;
        # otherwise a group column relies on merge_repeated_columns to merge at all.
        if column == primary_group_column and specification.merge_group_cells:
            continue
        _merge_repeated_cells(table, merge_frame, column, parent_columns)
        parent_columns.append(column)

    if cohort_values is not None:
        boundaries = [
            (i, label)
            for i, label in enumerate(cohort_values)
            if i == 0 or label != cohort_values[i - 1]
        ]
        for index, label in reversed(boundaries):
            anchor = table.rows[index + 1]._tr
            section_row = table.add_row()
            cell = section_row.cells[0]
            if len(frame.columns) > 1:
                cell = cell.merge(section_row.cells[-1])
            cell.text = label
            cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.LEFT
            for run in cell.paragraphs[0].runs:
                run.bold = True
            set_default_cell_margins(cell)
            _set_cell_border(cell, "top", size=12)
            _set_cell_border(cell, "bottom")
            anchor.addprevious(section_row._tr)

    if specification.header_groups:
        group_header = table.add_row()
        _set_repeat_header(group_header)
        occupied = set()
        for label, columns in specification.header_groups:
            indices = [frame.columns.get_loc(column) for column in columns]
            if not indices or indices != list(range(indices[0], indices[-1] + 1)):
                raise ValueError("header groups must span contiguous columns")
            if occupied.intersection(indices):
                raise ValueError("header groups must not overlap")
            occupied.update(indices)
            cell = group_header.cells[indices[0]]
            if len(indices) > 1:
                cell = cell.merge(group_header.cells[indices[-1]])
            cell.text = label
            cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            for run in cell.paragraphs[0].runs:
                run.bold = True
        header._tr.addprevious(group_header._tr)
        for index in sorted(set(range(len(frame.columns))) - occupied):
            cell = group_header.cells[index].merge(header.cells[index])
            cell.text = header_labels[index]
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            cell.paragraphs[0].alignment = (
                WD_ALIGN_PARAGRAPH.LEFT if index == 0 else WD_ALIGN_PARAGRAPH.CENTER
            )
            for run in cell.paragraphs[0].runs:
                run.bold = True
        for cell in group_header.cells:
            set_default_cell_margins(cell)
            _set_cell_border(cell, "top")
            _set_cell_border(cell, "bottom")

    if specification.font_size is not None:
        for row in table.rows:
            for cell in row.cells:
                for paragraph in cell.paragraphs:
                    for run in paragraph.runs:
                        run.font.size = Pt(specification.font_size)

    if specification.column_widths:
        if len(specification.column_widths) != len(frame.columns):
            raise ValueError("column_widths must contain one width per table column")
        _set_table_column_widths(table, specification.column_widths)
    if specification.preferred_column_widths:
        _set_preferred_column_widths(table, specification.preferred_column_widths)

    if len(table.rows) > 1:
        for cell in table.rows[-1].cells:
            _set_cell_border(cell, "bottom")

    for note in specification.notes:
        paragraph = document.add_paragraph(str(note))
        paragraph.paragraph_format.space_before = Pt(2)
        paragraph.paragraph_format.space_after = Pt(0)
        for run in paragraph.runs:
            run.italic = True
            run.font.size = Pt(specification.note_font_size)


def write_word_tables(
    tables: Iterable[WordTable],
    output_path,
    *,
    landscape: bool = False,
) -> Path:
    """Write editable tables in portrait by default and return the document path."""
    specifications = list(tables)
    if not specifications:
        raise ValueError("at least one table is required")

    def configure_section(section, use_landscape: bool) -> None:
        section.orientation = (
            WD_ORIENT.LANDSCAPE if use_landscape else WD_ORIENT.PORTRAIT
        )
        section.page_width = Mm(297 if use_landscape else 210)
        section.page_height = Mm(210 if use_landscape else 297)
        section.top_margin = Inches(0.7)
        section.bottom_margin = Inches(0.7)
        section.left_margin = Inches(0.7)
        section.right_margin = Inches(0.7)

    document = Document()
    configure_section(document.sections[0], landscape or specifications[0].landscape)

    normal = document.styles["Normal"]
    normal.font.name = "Arial"
    normal.font.size = Pt(10)
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "Arial")

    for index, specification in enumerate(specifications):
        table_landscape = landscape or specification.landscape
        if index:
            section = document.add_section(WD_SECTION.NEW_PAGE)
            configure_section(section, table_landscape)
        _add_table(document, specification)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    document.save(path)
    return path
