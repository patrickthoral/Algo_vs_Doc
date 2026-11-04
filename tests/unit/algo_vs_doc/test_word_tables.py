import pandas as pd
import pytest
from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn

from algo_vs_doc.word_tables import RichCell, WordTable, write_word_tables


def test_write_word_tables_creates_editable_document(tmp_path):
    target = tmp_path / "tables" / "results.docx"
    frame = pd.DataFrame({"Characteristic": ["Age", "SOFA"], "Combined": ["65", "5"]})

    result = write_word_tables(
        [
            WordTable(
                "Table 1. Baseline characteristics",
                frame,
                notes=("Values are summaries.",),
            )
        ],
        target,
    )

    document = Document(result)
    assert document.paragraphs[0].text == "Table 1. Baseline characteristics"
    assert document.tables[0].cell(0, 0).text == "Characteristic"
    assert document.tables[0].cell(1, 0).text == "Age"
    assert document.paragraphs[-1].text == "Values are summaries."


def test_word_tables_replace_undefined_formatted_summaries(tmp_path):
    target = tmp_path / "undefined.docx"
    frame = pd.DataFrame(
        {
            "Characteristic": ["Age", "Score", "Name"],
            "Value": ["65.0 (nan)", RichCell("nan [nan, nan]", "0.50"), "Banana"],
        }
    )

    write_word_tables([WordTable("Summary", frame)], target)

    table = Document(target).tables[0]
    assert table.cell(1, 1).text == "-"
    assert table.cell(2, 1).text == "-"
    assert table.cell(3, 1).text == "Banana"


def test_word_tables_render_rich_cells_retained_across_module_reload(tmp_path):
    target = tmp_path / "reloaded-rich-cell.docx"

    class RichCell:
        __module__ = "algo_vs_doc.word_tables"

        def __init__(self, estimate, p_value=None):
            self.estimate = estimate
            self.p_value = p_value

    frame = pd.DataFrame({"Difference": [RichCell("0.17 [0.05, 0.30]", "0.014")]})

    write_word_tables([WordTable("Contrasts", frame, multiline_intervals=True)], target)

    cell = Document(target).tables[0].cell(1, 0)
    assert cell.text == "0.17\n[0.05, 0.30]\np = 0.014"
    assert cell.paragraphs[0].runs[-2].italic


def test_write_word_tables_requires_content(tmp_path):
    with pytest.raises(ValueError, match="at least one table"):
        write_word_tables([], tmp_path / "empty.docx")


def test_word_table_suppresses_unnamed_header_and_formats_alignment(tmp_path):
    target = tmp_path / "table.docx"
    frame = pd.DataFrame(
        {"Unnamed: 0": ["Age", "Life support"], "Combined": ["65", ""]}
    )

    write_word_tables(
        [WordTable("Table 1", frame, section_starts=("Life support",))], target
    )

    document = Document(target)
    table = document.tables[0]
    assert document.sections[0].orientation == WD_ORIENT.PORTRAIT
    assert table.cell(0, 0).text == ""
    assert table.cell(0, 0).paragraphs[0].alignment == WD_ALIGN_PARAGRAPH.LEFT
    assert table.cell(0, 1).paragraphs[0].alignment == WD_ALIGN_PARAGRAPH.CENTER
    border = table.cell(2, 0)._tc.tcPr.find(
        "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}tcBorders"
    )
    assert border is not None
    assert (
        border[0].get(
            "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val"
        )
        == "single"
    )


def test_word_table_supports_hierarchical_row_indentation(tmp_path):
    target = tmp_path / "hierarchical-indentation.docx"
    frame = pd.DataFrame(
        {
            "Characteristic": ["Admitting specialty", "Surgical", "Neurosurgery"],
            "Combined": ["", "10 (20.0)", "4 (8.0)"],
        }
    )

    write_word_tables(
        [
            WordTable(
                "Table 1",
                frame,
                row_indent_levels={"Surgical": 1, "Neurosurgery": 2},
            )
        ],
        target,
    )

    table = Document(target).tables[0]
    subsection_indent = table.cell(2, 0).paragraphs[0].paragraph_format.left_indent
    detail_indent = table.cell(3, 0).paragraphs[0].paragraph_format.left_indent
    assert subsection_indent.inches == pytest.approx(0.09, abs=0.001)
    assert detail_indent.inches == pytest.approx(0.18, abs=0.001)


def test_word_table_merges_and_colors_consecutive_category_cells(tmp_path):
    target = tmp_path / "mapping.docx"
    frame = pd.DataFrame(
        {
            "Category": ["General", "General", "Respiratory"],
            "Feature Name": ["Age", "Sex", "Oxygen saturation"],
            "VUmc (n)": [2, 1, 3],
        }
    )

    write_word_tables(
        [
            WordTable(
                "eTable 1",
                frame,
                style="Table Grid",
                header_fill="1F77B4",
                header_font_color="FFFFFF",
                group_column="Category",
                group_fill="1F77B4",
                group_font_color="FFFFFF",
                merge_group_cells=True,
                column_widths=(1.2, 4.0, 0.8),
            )
        ],
        target,
    )

    table = Document(target).tables[0]
    namespace = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    assert table.style.name == "Table Grid"
    assert (
        table.cell(0, 0)._tc.tcPr.find(f"{namespace}shd").get(f"{namespace}fill")
        == "1F77B4"
    )
    assert table.cell(1, 0)._tc is table.cell(2, 0)._tc
    assert table.cell(1, 0).text == "General"
    assert table.cell(3, 0).text == "Respiratory"


def test_word_table_left_aligns_additional_group_columns(tmp_path):
    target = tmp_path / "left-aligned-groups.docx"
    frame = pd.DataFrame(
        {
            "Category": ["General", "General", "Respiratory"],
            "Physician factor": ["Workload", "Workload", "Workload"],
            "Feature Name": ["Age", "Sex", "Oxygen saturation"],
            "VUmc (n)": [2, 1, 3],
        }
    )

    write_word_tables(
        [
            WordTable(
                "eTable 1",
                frame,
                group_column=("Category", "Physician factor"),
                group_fill="1F77B4",
                merge_group_cells=True,
            )
        ],
        target,
    )

    table = Document(target).tables[0]
    namespace = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    # The first group column keeps the original merge/border/fill treatment.
    assert table.cell(1, 0)._tc is table.cell(2, 0)._tc
    assert table.cell(1, 0).text == "General"
    assert table.cell(3, 0).text == "Respiratory"
    assert (
        table.cell(1, 0)._tc.tcPr.find(f"{namespace}shd").get(f"{namespace}fill")
        == "1F77B4"
    )
    assert table.cell(0, 1).paragraphs[0].alignment == WD_ALIGN_PARAGRAPH.LEFT


def test_word_table_merge_repeated_columns_still_merges_primary_group_column(tmp_path):
    """A group column without merge_group_cells must still merge via merge_repeated_columns."""
    target = tmp_path / "merge-repeated-group-column.docx"
    frame = pd.DataFrame(
        {
            "Confidence": ["High", "High", "Low", "Low"],
            "Prediction source": ["Model", "Physician", "Model", "Physician"],
            "Value": ["1", "2", "3", "4"],
        }
    )

    write_word_tables(
        [
            WordTable(
                "eTable 5",
                frame,
                merge_repeated_columns=("Confidence",),
                group_column=("Confidence", "Prediction source"),
            )
        ],
        target,
    )

    table = Document(target).tables[0]
    assert table.cell(1, 0)._tc is table.cell(2, 0)._tc
    assert table.cell(1, 0).text == "High"
    assert table.cell(3, 0).text == "Low"
    assert table.cell(1, 1)._tc is not table.cell(2, 1)._tc
    assert table.cell(1, 1).paragraphs[0].alignment == WD_ALIGN_PARAGRAPH.LEFT


def test_word_table_supports_grouped_headers_with_row_spans(tmp_path):
    target = tmp_path / "grouped-headers.docx"
    frame = pd.DataFrame(
        {
            "Analysis": ["Primary/secondary"],
            "Agent 1": ["Model"],
            "Agent 2": ["Physician"],
            "Both Hospitals AUROC": ["0.10"],
            "Both Hospitals AUPRC": ["0.05"],
        }
    )
    write_word_tables(
        [
            WordTable(
                "eTable 4",
                frame,
                header_groups=(
                    (
                        "Both Hospitals",
                        ("Both Hospitals AUROC", "Both Hospitals AUPRC"),
                    ),
                ),
                header_labels=(
                    "Analysis",
                    "Agent 1",
                    "Agent 2",
                    "\u0394AUROC",
                    "\u0394AUPRC",
                ),
                column_widths=(2.0, 1.0, 1.0, 1.2, 1.2),
            )
        ],
        target,
    )

    table = Document(target).tables[0]
    assert table.cell(0, 0)._tc is table.cell(1, 0)._tc
    assert table.cell(0, 1)._tc is table.cell(1, 1)._tc
    assert table.cell(0, 2)._tc is table.cell(1, 2)._tc
    assert table.cell(0, 3)._tc is table.cell(0, 4)._tc
    assert table.cell(0, 3).text == "Both Hospitals"
    assert table.cell(1, 3).text == "\u0394AUROC"
    assert table.cell(1, 4).text == "\u0394AUPRC"
    grid_widths = [
        int(column.get(qn("w:w"))) for column in table._tbl.tblGrid.gridCol_lst
    ]
    assert grid_widths == [2880, 1440, 1440, 1728, 1728]
    merged_width = table.cell(0, 3)._tc.tcPr.find(qn("w:tcW"))
    assert int(merged_width.get(qn("w:w"))) == 3456
    layout = table._tbl.tblPr.find(qn("w:tblLayout"))
    assert layout.get(qn("w:type")) == "fixed"


def test_word_table_repeated_cell_merges_respect_parent_groups(tmp_path):
    target = tmp_path / "hierarchical.docx"
    frame = pd.DataFrame(
        {
            "Cohort": ["VUmc", "VUmc", "LUMC", "LUMC"],
            "Metric": ["AUROC", "AUROC", "AUROC", "AUROC"],
            "Predictor": ["Model", "Physician", "Model", "Physician"],
        }
    )

    write_word_tables(
        [
            WordTable(
                "Performance",
                frame,
                merge_repeated_columns=("Cohort", "Metric"),
            )
        ],
        target,
    )

    table = Document(target).tables[0]
    assert table.cell(1, 0)._tc is table.cell(2, 0)._tc
    assert table.cell(3, 0)._tc is table.cell(4, 0)._tc
    assert table.cell(2, 0)._tc is not table.cell(3, 0)._tc
    assert table.cell(1, 1)._tc is table.cell(2, 1)._tc
    assert table.cell(3, 1)._tc is table.cell(4, 1)._tc
    assert table.cell(2, 1)._tc is not table.cell(3, 1)._tc


def test_word_table_supports_preferred_width_with_autofit(tmp_path):
    target = tmp_path / "preferred-width.docx"
    frame = pd.DataFrame(
        {"Characteristic": ["Age"], "Outcome 1": ["1"], "Outcome 2": ["2"]}
    )

    write_word_tables(
        [WordTable("eTable 8", frame, preferred_column_widths={0: 2.14})], target
    )

    table = Document(target).tables[0]
    assert table.autofit
    assert round(table.cell(0, 0).width.inches, 2) == 2.14


def test_word_tables_use_one_explicitly_sized_section_per_table(tmp_path):
    target = tmp_path / "mixed.docx"
    frame = pd.DataFrame({"Characteristic": ["Age"], "Value": ["65"]})

    write_word_tables(
        [
            WordTable("eTable 1", frame),
            WordTable("eTable 4", frame, landscape=True),
            WordTable("eTable 5", frame),
            WordTable("eTable 11", frame, landscape=True),
        ],
        target,
    )

    document = Document(target)
    assert len(document.sections) == 4
    assert [section.orientation for section in document.sections] == [
        WD_ORIENT.PORTRAIT,
        WD_ORIENT.LANDSCAPE,
        WD_ORIENT.PORTRAIT,
        WD_ORIENT.LANDSCAPE,
    ]
    assert [round(section.page_width.mm) for section in document.sections] == [
        210,
        297,
        210,
        297,
    ]
    assert [round(section.page_height.mm) for section in document.sections] == [
        297,
        210,
        297,
        210,
    ]


def test_word_table_supports_a_per_table_note_font_size(tmp_path):
    target = tmp_path / "note-size.docx"
    write_word_tables(
        [
            WordTable(
                "eTable 12",
                pd.DataFrame({"Case": ["V1"]}),
                notes=("Compact note.",),
                note_font_size=7,
            )
        ],
        target,
    )

    note_run = Document(target).paragraphs[-1].runs[0]
    assert note_run.font.size.pt == 7


def test_multiline_intervals_does_not_split_characteristic_labels(tmp_path):
    target = tmp_path / "interval-layout.docx"
    write_word_tables(
        [
            WordTable(
                "eTable",
                pd.DataFrame(
                    {
                        "Characteristic": ["ICU census at discharge, median [IQR]"],
                        "Available": ["10.0 [8.0-11.0]"],
                    }
                ),
                multiline_intervals=True,
            )
        ],
        target,
    )

    table = Document(target).tables[0]
    assert table.cell(1, 0).text == "ICU census at discharge, median [IQR]"
    assert table.cell(1, 1).text == "10.0\n[8.0-11.0]"


def test_word_table_supports_table_wide_cell_margins(tmp_path):
    target = tmp_path / "compact-margins.docx"
    write_word_tables(
        [
            WordTable(
                "eTable 1",
                pd.DataFrame({"Feature": ["Age"], "Category": ["General"]}),
                cell_margin_inches=0.03,
            )
        ],
        target,
    )

    table = Document(target).tables[0]
    namespace = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    margins = table._tbl.tblPr.find(f"{namespace}tblCellMar")
    assert margins is not None
    for name in ("top", "start", "bottom", "end"):
        assert margins.find(f"{namespace}{name}").get(f"{namespace}w") == "43"
    assert table.cell(0, 0)._tc.tcPr.find(f"{namespace}tcMar") is None
