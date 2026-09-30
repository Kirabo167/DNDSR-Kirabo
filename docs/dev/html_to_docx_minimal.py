"""Convert the generated NCFV report HTML into a self-contained DOCX.

This deliberately uses only the Python standard library because the workspace
does not provide a Writer DOCX export filter or python-docx. Supported report
elements are headings, paragraphs and simple tables.
"""

from __future__ import annotations

import sys
import zipfile
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from xml.etree import ElementTree as ET


W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
XML = "http://www.w3.org/XML/1998/namespace"
ET.register_namespace("w", W)
ET.register_namespace("r", R)


def w(name: str) -> str:
    return f"{{{W}}}{name}"


def node(parent: ET.Element, name: str, **attributes: object) -> ET.Element:
    return ET.SubElement(parent, w(name), {w(key): str(value) for key, value in attributes.items()})


class ReportParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[tuple[str, object, str]] = []
        self.kind: str | None = None
        self.block_class = ""
        self.fragments: list[str] = []
        self.table: list[list[str]] | None = None
        self.table_class = ""
        self.row: list[str] | None = None
        self.cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag in {"h1", "h2", "h3", "p"}:
            self.kind = tag
            self.block_class = attributes.get("class") or ""
            self.fragments = []
        elif tag == "table":
            self.table = []
            self.table_class = attributes.get("class") or ""
        elif tag == "tr" and self.table is not None:
            self.row = []
        elif tag in {"th", "td"} and self.row is not None:
            self.cell = []
        elif tag == "div" and attributes.get("class") == "pagebreak":
            self.blocks.append(("pagebreak", "", ""))

    def handle_data(self, data: str) -> None:
        if self.cell is not None:
            self.cell.append(data)
        elif self.kind is not None:
            self.fragments.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"th", "td"} and self.cell is not None and self.row is not None:
            self.row.append("".join(self.cell).strip())
            self.cell = None
        elif tag == "tr" and self.row is not None and self.table is not None:
            self.table.append(self.row)
            self.row = None
        elif tag == "table" and self.table is not None:
            self.blocks.append(("table", self.table, self.table_class))
            self.table = None
        elif tag in {"h1", "h2", "h3", "p"} and self.kind == tag:
            value = "".join(self.fragments).strip()
            if value:
                self.blocks.append((tag, value, self.block_class))
            self.kind = None
            self.fragments = []


def add_text_run(paragraph: ET.Element, value: str, *, bold: bool = False) -> None:
    run = node(paragraph, "r")
    if bold:
        run_properties = node(run, "rPr")
        node(run_properties, "b")
    text = node(run, "t")
    text.set(f"{{{XML}}}space", "preserve")
    text.text = value


def add_paragraph(body: ET.Element, kind: str, value: str, css_class: str) -> None:
    paragraph = node(body, "p")
    properties = node(paragraph, "pPr")
    style_name = {"h1": "Title", "h2": "Heading1", "h3": "Heading2"}.get(kind, "Normal")
    if css_class in {"subtitle", "source", "formula", "note"}:
        style_name = {"subtitle": "Subtitle", "source": "Source", "formula": "Formula", "note": "Note"}[css_class]
    node(properties, "pStyle", val=style_name)
    add_text_run(paragraph, value)


def add_table(body: ET.Element, rows: list[list[str]]) -> None:
    if not rows:
        return
    columns = max(len(row) for row in rows)
    width = 15100
    column_width = width // columns
    table = node(body, "tbl")
    properties = node(table, "tblPr")
    node(properties, "tblW", w=width, type="dxa")
    node(properties, "tblLayout", type="fixed")
    borders = node(properties, "tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        node(borders, edge, val="single", sz="4", color="B9CBD3")
    grid = node(table, "tblGrid")
    for _ in range(columns):
        node(grid, "gridCol", w=column_width)

    for row_index, values in enumerate(rows):
        tr = node(table, "tr")
        if row_index == 0:
            tr_properties = node(tr, "trPr")
            node(tr_properties, "tblHeader", val="1")
        for cell_text in values:
            tc = node(tr, "tc")
            tc_properties = node(tc, "tcPr")
            node(tc_properties, "tcW", w=column_width, type="dxa")
            if row_index == 0:
                node(tc_properties, "shd", fill="E4F0F4")
            paragraph = node(tc, "p")
            paragraph_properties = node(paragraph, "pPr")
            node(paragraph_properties, "pStyle", val="TableHeader" if row_index == 0 else "TableText")
            add_text_run(paragraph, cell_text, bold=row_index == 0)


def make_style(styles: ET.Element, style_id: str, *, size: int, color: str = "17212B", bold: bool = False,
               before: int = 0, after: int = 80) -> None:
    style = node(styles, "style", type="paragraph", styleId=style_id)
    node(style, "name", val=style_id)
    paragraph_properties = node(style, "pPr")
    node(paragraph_properties, "spacing", before=before, after=after, line="260", lineRule="auto")
    run_properties = node(style, "rPr")
    node(run_properties, "rFonts", ascii="Noto Sans CJK SC", hAnsi="Noto Sans CJK SC", eastAsia="Noto Sans CJK SC")
    node(run_properties, "sz", val=size)
    node(run_properties, "szCs", val=size)
    node(run_properties, "color", val=color)
    if bold:
        node(run_properties, "b")


def xml_bytes(element: ET.Element) -> bytes:
    return b'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>' + ET.tostring(element, encoding="utf-8")


def write_docx(html_path: Path, docx_path: Path) -> None:
    parser = ReportParser()
    parser.feed(html_path.read_text(encoding="utf-8"))
    assert sum(kind == "table" for kind, _, _ in parser.blocks) >= 8
    document = ET.Element(w("document"))
    body = node(document, "body")
    for kind, content, css_class in parser.blocks:
        if kind == "table":
            add_table(body, content)  # type: ignore[arg-type]
        elif kind == "pagebreak":
            paragraph = node(body, "p")
            run = node(paragraph, "r")
            node(run, "br", type="page")
        else:
            add_paragraph(body, kind, str(content), css_class)
    section = node(body, "sectPr")
    node(section, "pgSz", w="16838", h="11906", orient="landscape")
    node(section, "pgMar", top="850", right="850", bottom="850", left="850", header="450", footer="450", gutter="0")

    styles = ET.Element(w("styles"))
    make_style(styles, "Normal", size=19)
    make_style(styles, "Title", size=36, color="12384B", bold=True, after=170)
    make_style(styles, "Subtitle", size=20, color="5B6671", after=140)
    make_style(styles, "Heading1", size=26, color="155D74", bold=True, before=220, after=100)
    make_style(styles, "Heading2", size=21, color="205369", bold=True, before=150, after=70)
    make_style(styles, "TableText", size=15, after=0)
    make_style(styles, "TableHeader", size=15, color="193C4C", bold=True, after=0)
    make_style(styles, "Source", size=15, color="405363", after=30)
    make_style(styles, "Formula", size=21, color="12384B", after=130)
    make_style(styles, "Note", size=19, color="463918", after=130)

    content_types = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
</Types>"""
    root_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
</Relationships>"""
    word_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>"""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    core = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
 xmlns:dc="http://purl.org/dc/elements/1.1/"
 xmlns:dcterms="http://purl.org/dc/terms/"
 xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
<dc:title>NCFV 三棱柱网格：体积加权误差综合报告</dc:title>
<dc:creator>Codex</dc:creator>
<dcterms:created xsi:type="dcterms:W3CDTF">{timestamp}</dcterms:created>
</cp:coreProperties>"""

    docx_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(docx_path, "w", zipfile.ZIP_DEFLATED) as package:
        package.writestr("[Content_Types].xml", content_types)
        package.writestr("_rels/.rels", root_rels)
        package.writestr("word/document.xml", xml_bytes(document))
        package.writestr("word/styles.xml", xml_bytes(styles))
        package.writestr("word/_rels/document.xml.rels", word_rels)
        package.writestr("docProps/core.xml", core)
    print(f"{docx_path}: {len(parser.blocks)} blocks")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: html_to_docx_minimal.py INPUT.html OUTPUT.docx")
    write_docx(Path(sys.argv[1]), Path(sys.argv[2]))
