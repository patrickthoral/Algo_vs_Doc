r"""Convert the project SVG icons into normalized Matplotlib path data.

Run from the repository root:

    .venv\Scripts\python.exe scripts/convert_svg_icons.py

The generated module is committed and has no runtime SVG parsing dependency.
"""

from __future__ import annotations

import re
from pathlib import Path
import xml.etree.ElementTree as ET

from fontTools.misc.transform import Transform
from fontTools.pens.recordingPen import RecordingPen
from fontTools.pens.transformPen import TransformPen
from fontTools.svgLib.path import parse_path


ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "icu_bed": ROOT / "img" / "icu_bed.svg",
    "outcome_arrow": ROOT / "img" / "outcome_arrow.svg",
}
TARGET = ROOT / "algo_vs_doc" / "icon_paths.py"
NUMBER = r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?"


def _transform(value: str | None) -> Transform:
    result = Transform()
    if not value:
        return result
    for name, arguments in re.findall(r"([A-Za-z]+)\s*\(([^)]*)\)", value):
        values = [float(item) for item in re.findall(NUMBER, arguments)]
        if name == "matrix":
            local = Transform(*values)
        elif name == "translate":
            local = Transform().translate(
                values[0], values[1] if len(values) > 1 else 0
            )
        elif name == "scale":
            local = Transform().scale(
                values[0], values[1] if len(values) > 1 else values[0]
            )
        elif name == "rotate":
            if len(values) == 1:
                local = Transform().rotate(values[0])
            else:
                local = (
                    Transform()
                    .translate(values[1], values[2])
                    .rotate(values[0])
                    .translate(-values[1], -values[2])
                )
        else:
            raise ValueError(f"unsupported SVG transform {name!r}")
        result = result.transform(local)
    return result


def _style(element: ET.Element, inherited: dict[str, str]) -> dict[str, str]:
    result = inherited.copy()
    result.update(
        item.split(":", 1)
        for item in element.get("style", "").split(";")
        if ":" in item
    )
    for name in ("fill", "stroke", "stroke-width", "fill-opacity", "stroke-opacity"):
        if element.get(name) is not None:
            result[name] = element.get(name)
    return result


def _paths(path: Path) -> list[dict]:
    root = ET.parse(path).getroot()
    records = []

    def visit(element, parent_transform=Transform(), inherited=None):
        tag = element.tag.rsplit("}", 1)[-1]
        if tag in {"defs", "clipPath", "mask", "filter", "metadata", "namedview"}:
            return
        inherited = inherited or {}
        combined = parent_transform.transform(_transform(element.get("transform")))
        style = _style(element, inherited)
        if tag == "path" and element.get("d"):
            recording = RecordingPen()
            parse_path(element.get("d"), TransformPen(recording, combined))
            records.append({"commands": recording.value, "style": style})
        for child in element:
            visit(child, combined, style)

    visit(root)
    points = [
        point
        for record in records
        for _, operands in record["commands"]
        for point in operands
    ]
    minimum_x = min(point[0] for point in points)
    maximum_x = max(point[0] for point in points)
    minimum_y = min(point[1] for point in points)
    maximum_y = max(point[1] for point in points)
    width, height = maximum_x - minimum_x, maximum_y - minimum_y
    for record in records:
        normalized = []
        for command, operands in record["commands"]:
            normalized.append(
                (
                    command,
                    tuple(
                        ((x - minimum_x) / width, 1 - (y - minimum_y) / height)
                        for x, y in operands
                    ),
                )
            )
        record["commands"] = normalized
        stroke_width = float(record["style"].get("stroke-width", 0) or 0)
        record["linewidth"] = stroke_width / max(width, height)
    return records


def main() -> None:
    icons = {name: _paths(path) for name, path in SOURCES.items()}
    lines = [
        '"""Generated Matplotlib path data. Run scripts/convert_svg_icons.py to update."""',
        "",
        "ICON_PATHS = " + repr(icons),
        "",
    ]
    TARGET.write_text("\n".join(lines), encoding="ascii")
    print(f"Wrote {TARGET.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
