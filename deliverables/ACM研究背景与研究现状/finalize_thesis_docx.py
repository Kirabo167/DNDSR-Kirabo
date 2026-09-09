#!/usr/bin/env python3
"""Apply Tsinghua doctoral-thesis chapter formatting to a Pandoc DOCX."""

from __future__ import annotations

import argparse
import re
import zipfile
from pathlib import Path

from lxml import etree


W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
M = "http://schemas.openxmlformats.org/officeDocument/2006/math"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PR = "http://schemas.openxmlformats.org/package/2006/relationships"
CT = "http://schemas.openxmlformats.org/package/2006/content-types"
CP = "http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
DC = "http://purl.org/dc/elements/1.1/"
NS = {"w": W, "m": M, "r": R, "cp": CP, "dc": DC}


def qn(namespace: str, tag: str) -> str:
    return f"{{{namespace}}}{tag}"


def child(parent: etree._Element, namespace: str, tag: str) -> etree._Element:
    found = parent.find(qn(namespace, tag))
    if found is None:
        found = etree.SubElement(parent, qn(namespace, tag))
    return found


def set_attr(element: etree._Element, name: str, value: str) -> None:
    element.set(qn(W, name), value)


def paragraph_text(paragraph: etree._Element) -> str:
    return "".join(paragraph.xpath(".//w:t/text()", namespaces=NS)).strip()


def paragraph_properties(paragraph: etree._Element) -> etree._Element:
    ppr = paragraph.find(qn(W, "pPr"))
    if ppr is None:
        ppr = etree.Element(qn(W, "pPr"))
        paragraph.insert(0, ppr)
    return ppr


def run_properties(run: etree._Element) -> etree._Element:
    rpr = run.find(qn(W, "rPr"))
    if rpr is None:
        rpr = etree.Element(qn(W, "rPr"))
        run.insert(0, rpr)
    return rpr


def set_fonts_and_size(run: etree._Element, half_points: int) -> None:
    rpr = run_properties(run)
    fonts = child(rpr, W, "rFonts")
    set_attr(fonts, "ascii", "Times New Roman")
    set_attr(fonts, "hAnsi", "Times New Roman")
    set_attr(fonts, "eastAsia", "宋体")
    set_attr(fonts, "cs", "Times New Roman")
    set_attr(child(rpr, W, "sz"), "val", str(half_points))
    set_attr(child(rpr, W, "szCs"), "val", str(half_points))


def set_paragraph_spacing(
    paragraph: etree._Element,
    *,
    before: int,
    after: int,
    line: int,
    line_rule: str,
) -> None:
    spacing = child(paragraph_properties(paragraph), W, "spacing")
    set_attr(spacing, "before", str(before))
    set_attr(spacing, "after", str(after))
    set_attr(spacing, "line", str(line))
    set_attr(spacing, "lineRule", line_rule)


def set_paragraph_alignment(paragraph: etree._Element, value: str) -> None:
    set_attr(child(paragraph_properties(paragraph), W, "jc"), "val", value)


def set_indent(
    paragraph: etree._Element,
    *,
    left: int | None = None,
    hanging: int | None = None,
    first_line: int | None = None,
) -> None:
    ind = child(paragraph_properties(paragraph), W, "ind")
    for key in ("left", "hanging", "firstLine"):
        ind.attrib.pop(qn(W, key), None)
    if left is not None:
        set_attr(ind, "left", str(left))
    if hanging is not None:
        set_attr(ind, "hanging", str(hanging))
    if first_line is not None:
        set_attr(ind, "firstLine", str(first_line))


def set_style(paragraph: etree._Element, style_id: str) -> None:
    set_attr(child(paragraph_properties(paragraph), W, "pStyle"), "val", style_id)


def set_page_break_before(paragraph: etree._Element) -> None:
    set_attr(child(paragraph_properties(paragraph), W, "pageBreakBefore"), "val", "1")


def set_keep_next(paragraph: etree._Element) -> None:
    set_attr(child(paragraph_properties(paragraph), W, "keepNext"), "val", "1")


def set_cell_width(cell: etree._Element, width: int) -> None:
    tcpr = child(cell, W, "tcPr")
    tcw = child(tcpr, W, "tcW")
    set_attr(tcw, "type", "dxa")
    set_attr(tcw, "w", str(width))
    set_attr(child(tcpr, W, "vAlign"), "val", "center")


def set_border(parent: etree._Element, side: str, val: str, size: int = 0) -> None:
    border = child(parent, W, side)
    set_attr(border, "val", val)
    if val != "nil":
        set_attr(border, "sz", str(size))
        set_attr(border, "space", "0")
        set_attr(border, "color", "000000")


def patch_styles(xml: bytes) -> bytes:
    root = etree.fromstring(xml)

    def style(style_id: str) -> etree._Element | None:
        return root.find(f".//w:style[@w:styleId='{style_id}']", namespaces=NS)

    normal = style("Normal")
    if normal is not None:
        ppr = child(normal, W, "pPr")
        set_attr(child(ppr, W, "jc"), "val", "both")
        ind = child(ppr, W, "ind")
        set_attr(ind, "firstLine", "480")
        spacing = child(ppr, W, "spacing")
        for key, value in {
            "before": "0",
            "after": "0",
            "line": "400",
            "lineRule": "exact",
        }.items():
            set_attr(spacing, key, value)
        rpr = child(normal, W, "rPr")
        fonts = child(rpr, W, "rFonts")
        for key, value in {
            "ascii": "Times New Roman",
            "hAnsi": "Times New Roman",
            "eastAsia": "宋体",
            "cs": "Times New Roman",
        }.items():
            set_attr(fonts, key, value)
        set_attr(child(rpr, W, "sz"), "val", "24")
        set_attr(child(rpr, W, "szCs"), "val", "24")

    heading_specs = {
        "Heading1": ("32", "480", "360", "center"),
        "Heading2": ("28", "480", "120", "left"),
        "Heading3": ("26", "240", "120", "left"),
        "Heading4": ("24", "240", "120", "left"),
    }
    for style_id, (size, before, after, alignment) in heading_specs.items():
        current = style(style_id)
        if current is None:
            continue
        ppr = child(current, W, "pPr")
        set_attr(child(ppr, W, "jc"), "val", alignment)
        spacing = child(ppr, W, "spacing")
        for key, value in {
            "before": before,
            "after": after,
            "line": "400" if style_id != "Heading1" else "240",
            "lineRule": "exact" if style_id != "Heading1" else "auto",
        }.items():
            set_attr(spacing, key, value)
        set_attr(child(ppr, W, "keepNext"), "val", "1")
        set_attr(child(ppr, W, "keepLines"), "val", "1")
        rpr = child(current, W, "rPr")
        fonts = child(rpr, W, "rFonts")
        for key, value in {
            "ascii": "Arial",
            "hAnsi": "Arial",
            "eastAsia": "黑体",
            "cs": "Arial",
        }.items():
            set_attr(fonts, key, value)
        set_attr(child(rpr, W, "sz"), "val", size)
        set_attr(child(rpr, W, "szCs"), "val", size)
        set_attr(child(rpr, W, "b"), "val", "1")
        set_attr(child(rpr, W, "color"), "val", "000000")

    return etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)


def patch_header(xml: bytes, title: str) -> bytes:
    root = etree.fromstring(xml)
    texts = root.xpath(".//w:t", namespaces=NS)
    if texts:
        texts[0].text = title
        for item in texts[1:]:
            item.text = ""
    for run in root.xpath(".//w:r", namespaces=NS):
        set_fonts_and_size(run, 21)
    for paragraph in root.xpath(".//w:p", namespaces=NS):
        set_paragraph_alignment(paragraph, "center")
    return etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)


def patch_settings(xml: bytes) -> bytes:
    root = etree.fromstring(xml)
    update = root.find("w:updateFields", namespaces=NS)
    if update is None:
        update = etree.SubElement(root, qn(W, "updateFields"))
    set_attr(update, "val", "true")
    return etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)


def patch_core(xml: bytes, document_title: str) -> bytes:
    root = etree.fromstring(xml)
    title = root.find("dc:title", namespaces=NS)
    if title is None:
        title = etree.SubElement(root, qn(DC, "title"))
    title.text = document_title
    subject = root.find("dc:subject", namespaces=NS)
    if subject is None:
        subject = etree.SubElement(root, qn(DC, "subject"))
    subject.text = "博士学位论文研究背景与研究现状章节草稿"
    creator = root.find("dc:creator", namespaces=NS)
    if creator is None:
        creator = etree.SubElement(root, qn(DC, "creator"))
    creator.text = "DNDSR研究写作"
    return etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)


def patch_relationships(xml: bytes) -> tuple[bytes, str, str]:
    root = etree.fromstring(xml)
    ids = []
    for relationship in root.findall(qn(PR, "Relationship")):
        match = re.fullmatch(r"rId(\d+)", relationship.get("Id", ""))
        if match:
            ids.append(int(match.group(1)))
    next_id = max(ids, default=0) + 1

    def ensure(kind: str, target: str) -> str:
        nonlocal next_id
        rel_type = (
            "http://schemas.openxmlformats.org/officeDocument/2006/relationships/"
            + kind
        )
        existing = [
            item
            for item in root.findall(qn(PR, "Relationship"))
            if item.get("Type") == rel_type
        ]
        if existing:
            existing[0].set("Target", target)
            return existing[0].get("Id")
        rel_id = f"rId{next_id}"
        next_id += 1
        etree.SubElement(
            root,
            qn(PR, "Relationship"),
            Id=rel_id,
            Type=rel_type,
            Target=target,
        )
        return rel_id

    header_id = ensure("header", "header1.xml")
    footer_id = ensure("footer", "footer1.xml")
    return (
        etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True),
        header_id,
        footer_id,
    )


def patch_content_types(xml: bytes) -> bytes:
    root = etree.fromstring(xml)

    def ensure(part_name: str, content_type: str) -> None:
        existing = root.xpath(
            "./ct:Override[@PartName=$part]", namespaces={"ct": CT}, part=part_name
        )
        if not existing:
            etree.SubElement(
                root,
                qn(CT, "Override"),
                PartName=part_name,
                ContentType=content_type,
            )

    ensure(
        "/word/header1.xml",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.header+xml",
    )
    ensure(
        "/word/footer1.xml",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.footer+xml",
    )
    return etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)


def build_header(title: str) -> bytes:
    root = etree.Element(qn(W, "hdr"), nsmap={"w": W, "r": R})
    paragraph = etree.SubElement(root, qn(W, "p"))
    ppr = paragraph_properties(paragraph)
    set_paragraph_alignment(paragraph, "center")
    set_paragraph_spacing(paragraph, before=0, after=0, line=240, line_rule="auto")
    borders = child(ppr, W, "pBdr")
    set_border(borders, "bottom", "single", 6)
    run = etree.SubElement(paragraph, qn(W, "r"))
    set_fonts_and_size(run, 21)
    text = etree.SubElement(run, qn(W, "t"))
    text.text = title
    return etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)


def build_footer() -> bytes:
    root = etree.Element(qn(W, "ftr"), nsmap={"w": W, "r": R})
    paragraph = etree.SubElement(root, qn(W, "p"))
    set_paragraph_alignment(paragraph, "center")
    set_paragraph_spacing(paragraph, before=0, after=0, line=240, line_rule="auto")

    begin = etree.SubElement(paragraph, qn(W, "r"))
    set_fonts_and_size(begin, 21)
    set_attr(child(begin, W, "fldChar"), "fldCharType", "begin")
    instruction = etree.SubElement(paragraph, qn(W, "r"))
    set_fonts_and_size(instruction, 21)
    instr_text = etree.SubElement(instruction, qn(W, "instrText"))
    instr_text.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    instr_text.text = " PAGE "
    separate = etree.SubElement(paragraph, qn(W, "r"))
    set_fonts_and_size(separate, 21)
    set_attr(child(separate, W, "fldChar"), "fldCharType", "separate")
    value = etree.SubElement(paragraph, qn(W, "r"))
    set_fonts_and_size(value, 21)
    value_text = etree.SubElement(value, qn(W, "t"))
    value_text.text = "1"
    end = etree.SubElement(paragraph, qn(W, "r"))
    set_fonts_and_size(end, 21)
    set_attr(child(end, W, "fldChar"), "fldCharType", "end")
    return etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)


def patch_document(xml: bytes, header_id: str, footer_id: str) -> bytes:
    root = etree.fromstring(xml)

    # Use bookmark names accepted reliably by desktop Word.
    for bookmark in root.xpath(".//w:bookmarkStart", namespaces=NS):
        name = bookmark.get(qn(W, "name"), "")
        match = re.fullmatch(r"ref-(\d+)", name)
        if match:
            bookmark.set(qn(W, "name"), f"ref_{int(match.group(1)):03d}")
    for hyperlink in root.xpath(".//w:hyperlink[@w:anchor]", namespaces=NS):
        anchor = hyperlink.get(qn(W, "anchor"), "")
        match = re.fullmatch(r"ref-(\d+)", anchor)
        if match:
            hyperlink.set(qn(W, "anchor"), f"ref_{int(match.group(1)):03d}")
            for run in hyperlink.xpath(".//w:r", namespaces=NS):
                rpr = run_properties(run)
                set_attr(child(rpr, W, "color"), "val", "000000")
                set_attr(child(rpr, W, "u"), "val", "none")

    reference_paragraphs: set[etree._Element] = set()
    for bookmark in root.xpath(
        ".//w:bookmarkStart[starts-with(@w:name, 'ref_')]", namespaces=NS
    ):
        paragraph = bookmark.xpath("ancestor::w:p[1]", namespaces=NS)
        if paragraph:
            reference_paragraphs.add(paragraph[0])
        else:
            following = bookmark.getnext()
            if following is not None and following.tag == qn(W, "p"):
                reference_paragraphs.add(following)

    for paragraph in root.xpath(".//w:p", namespaces=NS):
        text = paragraph_text(paragraph)
        ppr = paragraph_properties(paragraph)
        in_table = bool(paragraph.xpath("ancestor::w:tbl", namespaces=NS))
        style_element = ppr.find(qn(W, "pStyle"))
        style_id = (
            style_element.get(qn(W, "val"), "") if style_element is not None else ""
        )

        if paragraph in reference_paragraphs:
            set_style(paragraph, "Normal")
            set_paragraph_alignment(paragraph, "both")
            set_paragraph_spacing(
                paragraph, before=60, after=0, line=320, line_rule="exact"
            )
            set_indent(paragraph, left=420, hanging=420)
            set_attr(child(ppr, W, "keepLines"), "val", "1")
            for run in paragraph.xpath(".//w:r", namespaces=NS):
                set_fonts_and_size(run, 21)
            continue

        if text == "参考文献":
            set_style(paragraph, "Heading1")
            set_page_break_before(paragraph)
            set_keep_next(paragraph)
            set_paragraph_alignment(paragraph, "center")
            set_indent(paragraph)
            continue

        if text.startswith("第1章") and style_id == "Heading1":
            set_keep_next(paragraph)
            set_paragraph_alignment(paragraph, "center")
            set_indent(paragraph)
            continue

        if re.match(r"^表1[-.]\d+", text):
            set_style(paragraph, "Caption")
            set_paragraph_alignment(paragraph, "center")
            set_paragraph_spacing(
                paragraph, before=240, after=120, line=240, line_rule="auto"
            )
            set_indent(paragraph)
            set_page_break_before(paragraph)
            set_keep_next(paragraph)
            for run in paragraph.xpath(".//w:r", namespaces=NS):
                set_fonts_and_size(run, 22)
                rpr = run_properties(run)
                for tag in ("i", "iCs"):
                    for italic in rpr.findall(qn(W, tag)):
                        rpr.remove(italic)
                    set_attr(child(rpr, W, tag), "val", "0")
                set_attr(child(rpr, W, "color"), "val", "000000")
            continue

        if not in_table and style_id not in {
            "Heading1",
            "Heading2",
            "Heading3",
            "Heading4",
        }:
            has_numbering = ppr.find(qn(W, "numPr")) is not None
            set_paragraph_alignment(paragraph, "both")
            set_paragraph_spacing(
                paragraph, before=0, after=0, line=400, line_rule="exact"
            )
            if not has_numbering and text:
                set_indent(paragraph, first_line=480)

    for table in root.xpath(".//w:tbl", namespaces=NS):
        contains_math = bool(table.xpath(".//m:oMath", namespaces=NS))
        tblpr = child(table, W, "tblPr")
        tblw = child(tblpr, W, "tblW")
        set_attr(tblw, "type", "pct")
        set_attr(tblw, "w", "5000")
        set_attr(child(tblpr, W, "tblLayout"), "type", "fixed")
        borders = child(tblpr, W, "tblBorders")

        if contains_math:
            table_style = tblpr.find(qn(W, "tblStyle"))
            if table_style is not None:
                tblpr.remove(table_style)
            for side in ("top", "left", "bottom", "right", "insideH", "insideV"):
                set_border(borders, side, "nil")
            look = child(tblpr, W, "tblLook")
            for key in ("firstRow", "lastRow", "firstColumn", "lastColumn"):
                set_attr(look, key, "0")
            for header in table.xpath(".//w:tblHeader", namespaces=NS):
                header.getparent().remove(header)

            row = table.find(qn(W, "tr"))
            if row is not None:
                existing_cells = row.findall(qn(W, "tc"))
                if len(existing_cells) == 2:
                    spacer = etree.Element(qn(W, "tc"))
                    spacer_p = etree.SubElement(spacer, qn(W, "p"))
                    set_paragraph_alignment(spacer_p, "left")
                    row.insert(row.index(existing_cells[0]), spacer)

            grid = child(table, W, "tblGrid")
            for grid_col in list(grid):
                grid.remove(grid_col)
            for width in (1200, 6104, 1200):
                item = etree.SubElement(grid, qn(W, "gridCol"))
                set_attr(item, "w", str(width))

            cells = table.xpath(".//w:tr[1]/w:tc", namespaces=NS)
            if len(cells) >= 3:
                set_cell_width(cells[0], 1200)
                set_cell_width(cells[1], 6104)
                set_cell_width(cells[2], 1200)
                set_attr(child(child(cells[2], W, "tcPr"), W, "noWrap"), "val", "1")
                for cell in cells:
                    cell_properties = child(cell, W, "tcPr")
                    cell_borders = child(cell_properties, W, "tcBorders")
                    for side in (
                        "top",
                        "left",
                        "bottom",
                        "right",
                        "insideH",
                        "insideV",
                    ):
                        set_border(cell_borders, side, "nil")
                    margins = child(cell_properties, W, "tcMar")
                    for side in ("top", "left", "bottom", "right"):
                        margin = child(margins, W, side)
                        set_attr(margin, "w", "0")
                        set_attr(margin, "type", "dxa")
                for paragraph in cells[1].xpath(".//w:p", namespaces=NS):
                    set_style(paragraph, "Normal")
                    set_paragraph_alignment(paragraph, "center")
                    set_paragraph_spacing(
                        paragraph, before=120, after=120, line=240, line_rule="auto"
                    )
                    set_indent(paragraph)
                for paragraph in cells[2].xpath(".//w:p", namespaces=NS):
                    set_style(paragraph, "Normal")
                    set_paragraph_alignment(paragraph, "right")
                    set_paragraph_spacing(
                        paragraph, before=120, after=120, line=240, line_rule="auto"
                    )
                    set_indent(paragraph)
                    for run in paragraph.xpath(".//w:r", namespaces=NS):
                        rpr = run_properties(run)
                        for bold in rpr.findall(qn(W, "b")):
                            rpr.remove(bold)
                        set_fonts_and_size(run, 21)
        else:
            set_border(borders, "top", "single", 12)
            set_border(borders, "bottom", "single", 12)
            for side in ("left", "right", "insideH", "insideV"):
                set_border(borders, side, "nil")
            first_row_cells = table.xpath("./w:tr[1]/w:tc", namespaces=NS)
            for cell in first_row_cells:
                tcpr = child(cell, W, "tcPr")
                tcborders = child(tcpr, W, "tcBorders")
                set_border(tcborders, "bottom", "single", 8)
            for paragraph in table.xpath(".//w:p", namespaces=NS):
                set_paragraph_spacing(
                    paragraph, before=60, after=60, line=240, line_rule="auto"
                )
                set_indent(paragraph)
                for run in paragraph.xpath(".//w:r", namespaces=NS):
                    set_fonts_and_size(run, 22)

    for section in root.xpath(".//w:sectPr", namespaces=NS):
        for reference in section.xpath(
            "./w:headerReference | ./w:footerReference", namespaces=NS
        ):
            section.remove(reference)
        header_reference = etree.Element(qn(W, "headerReference"))
        set_attr(header_reference, "type", "default")
        header_reference.set(qn(R, "id"), header_id)
        footer_reference = etree.Element(qn(W, "footerReference"))
        set_attr(footer_reference, "type", "default")
        footer_reference.set(qn(R, "id"), footer_id)
        section.insert(0, footer_reference)
        section.insert(0, header_reference)
        pgsz = child(section, W, "pgSz")
        set_attr(pgsz, "w", "11906")
        set_attr(pgsz, "h", "16838")
        pgmar = child(section, W, "pgMar")
        for key, value in {
            "top": "1701",
            "right": "1701",
            "bottom": "1701",
            "left": "1701",
            "header": "1247",
            "footer": "1247",
            "gutter": "0",
        }.items():
            set_attr(pgmar, key, value)
        pgnum = child(section, W, "pgNumType")
        set_attr(pgnum, "fmt", "decimal")
        set_attr(pgnum, "start", "1")

    return etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)


def finalize(
    source: Path, target: Path, *, document_title: str, header_title: str
) -> None:
    with zipfile.ZipFile(source, "r") as input_zip:
        members = {name: input_zip.read(name) for name in input_zip.namelist()}

    relationships, header_id, footer_id = patch_relationships(
        members["word/_rels/document.xml.rels"]
    )
    members["word/_rels/document.xml.rels"] = relationships
    members["[Content_Types].xml"] = patch_content_types(members["[Content_Types].xml"])
    members["word/document.xml"] = patch_document(
        members["word/document.xml"], header_id, footer_id
    )
    if "word/styles.xml" in members:
        members["word/styles.xml"] = patch_styles(members["word/styles.xml"])
    if "word/settings.xml" in members:
        members["word/settings.xml"] = patch_settings(members["word/settings.xml"])
    if "docProps/core.xml" in members:
        members["docProps/core.xml"] = patch_core(
            members["docProps/core.xml"], document_title
        )
    members["word/header1.xml"] = build_header(header_title)
    members["word/footer1.xml"] = build_footer()

    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as output_zip:
        for name, data in members.items():
            info = zipfile.ZipInfo(name)
            info.date_time = (2026, 9, 9, 0, 0, 0)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            output_zip.writestr(info, data)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument(
        "--document-title",
        default="ACM预处理方法及其他不可压缩流体计算方法：研究背景与研究现状",
    )
    parser.add_argument("--header-title", default="第1章　绪论")
    args = parser.parse_args()
    finalize(
        args.source,
        args.target,
        document_title=args.document_title,
        header_title=args.header_title,
    )


if __name__ == "__main__":
    main()
