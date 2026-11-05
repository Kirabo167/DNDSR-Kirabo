#!/usr/bin/env python3
"""Scale all DOCX tables to the landscape page width without reflowing tokens."""

from __future__ import annotations

import argparse
import re
import zipfile
from pathlib import Path


TABLE_RE = re.compile(r"<w:tbl>.*?</w:tbl>", re.DOTALL)


def scale_widths(table_xml: str, target_width: int) -> str:
    width_match = re.search(r'<w:tblW w:w="(\d+)" w:type="dxa"/>', table_xml)
    if width_match is None:
        raise RuntimeError("table has no absolute tblW")
    source_width = int(width_match.group(1))
    scale = target_width / source_width

    table_xml = re.sub(
        r'<w:tblW w:w="\d+" w:type="dxa"/>',
        f'<w:tblW w:w="{target_width}" w:type="dxa"/><w:tblLayout w:type="fixed"/>',
        table_xml,
        count=1,
    )

    def scaled(match: re.Match[str]) -> str:
        tag, value = match.group(1), int(match.group(2))
        return f'<w:{tag} w:w="{round(value * scale)}"/>'

    table_xml = re.sub(r'<w:(gridCol) w:w="(\d+)"/>', scaled, table_xml)

    def scaled_cell(match: re.Match[str]) -> str:
        value = int(match.group(1))
        return f'<w:tcW w:w="{round(value * scale)}" w:type="dxa"/>'

    table_xml = re.sub(
        r'<w:tcW w:w="(\d+)" w:type="dxa"/>', scaled_cell, table_xml
    )
    table_xml = table_xml.replace("</w:tcPr>", "<w:noWrap/></w:tcPr>")
    return table_xml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--width", type=int, default=15000)
    parser.add_argument(
        "--repeat-headers",
        default="0",
        help="comma-separated zero-based table indexes whose first row repeats",
    )
    parser.add_argument("--expected-tables", type=int, default=3)
    parser.add_argument(
        "--page-break-marker",
        default="内存口径：",
        help="exact paragraph text before which to insert a page break; empty disables",
    )
    args = parser.parse_args()
    repeat_headers = {
        int(value) for value in args.repeat_headers.split(",") if value.strip()
    }

    if args.destination.exists():
        raise RuntimeError(f"refusing to overwrite {args.destination}")

    with zipfile.ZipFile(args.source, "r") as source:
        document_xml = source.read("word/document.xml").decode("utf-8")
        tables = TABLE_RE.findall(document_xml)
        if len(tables) != args.expected_tables:
            raise RuntimeError(
                f"expected {args.expected_tables} tables, found {len(tables)}"
            )

        table_index = 0

        def transform_table(match: re.Match[str]) -> str:
            nonlocal table_index
            table_xml = scale_widths(match.group(0), args.width)
            if table_index not in repeat_headers:
                table_xml = table_xml.replace('<w:tblHeader w:val="true"/>', "")
            table_index += 1
            return table_xml

        transformed, count = TABLE_RE.subn(
            transform_table, document_xml
        )
        if count != args.expected_tables:
            raise RuntimeError(
                f"expected to transform {args.expected_tables} tables, transformed {count}"
            )

        if args.page_break_marker:
            marker = f"<w:t>{args.page_break_marker}</w:t>"
            marker_position = transformed.find(marker)
            if marker_position < 0:
                raise RuntimeError(
                    f"cannot locate page-break paragraph: {args.page_break_marker}"
                )
            paragraph_start = transformed.rfind("<w:p>", 0, marker_position)
            properties_start = transformed.find("<w:pPr>", paragraph_start, marker_position)
            if paragraph_start < 0 or properties_start < 0:
                raise RuntimeError("cannot locate page-break paragraph properties")
            insertion = properties_start + len("<w:pPr>")
            transformed = (
                transformed[:insertion]
                + "<w:keepLines/><w:pageBreakBefore/>"
                + transformed[insertion:]
            )

        with zipfile.ZipFile(args.destination, "w") as destination:
            for item in source.infolist():
                data = (
                    transformed.encode("utf-8")
                    if item.filename == "word/document.xml"
                    else source.read(item.filename)
                )
                destination.writestr(item, data)

    print(args.destination)


if __name__ == "__main__":
    main()
