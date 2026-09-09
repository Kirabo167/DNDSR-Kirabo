#!/usr/bin/env python3
"""Validate structural and formatting invariants of a thesis DOCX."""

from __future__ import annotations

import argparse
import re
import sys
import zipfile
from pathlib import Path

from lxml import etree


W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
M = "http://schemas.openxmlformats.org/officeDocument/2006/math"
NS = {"w": W, "m": M}


def attr(element: etree._Element, name: str, namespace: str = W) -> str:
    return element.get(f"{{{namespace}}}{name}", "")


def print_report(
    path: Path, errors: list[str], warnings: list[str], stats: dict[str, int]
) -> None:
    print(f"DOCX: {path}")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    for message in warnings:
        print(f"WARNING: {message}")
    for message in errors:
        print(f"ERROR: {message}")
    print("RESULT: PASS" if not errors else "RESULT: FAIL")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("docx", type=Path)
    parser.add_argument("--expect-formulas", type=int)
    parser.add_argument("--expect-references", type=int)
    args = parser.parse_args()

    errors: list[str] = []
    warnings: list[str] = []
    stats: dict[str, int] = {}

    try:
        archive = zipfile.ZipFile(args.docx)
        bad = archive.testzip()
    except (OSError, zipfile.BadZipFile) as exc:
        print(f"ERROR: invalid DOCX package: {exc}")
        return 2
    if bad:
        errors.append(f"corrupt ZIP member: {bad}")

    required = {
        "[Content_Types].xml",
        "word/document.xml",
        "word/styles.xml",
        "word/_rels/document.xml.rels",
    }
    missing = sorted(required - set(archive.namelist()))
    if missing:
        errors.append("missing required OOXML parts: " + ", ".join(missing))
        print_report(args.docx, errors, warnings, stats)
        return 1

    try:
        document = etree.fromstring(archive.read("word/document.xml"))
        styles = etree.fromstring(archive.read("word/styles.xml"))
    except etree.XMLSyntaxError as exc:
        errors.append(f"invalid OOXML: {exc}")
        print_report(args.docx, errors, warnings, stats)
        return 1

    formulas = document.xpath(".//m:oMathPara", namespaces=NS)
    stats["display_formulas"] = len(formulas)
    stats["all_math_objects"] = len(document.xpath(".//m:oMath", namespaces=NS))
    if args.expect_formulas is not None and len(formulas) != args.expect_formulas:
        errors.append(
            f"expected {args.expect_formulas} display formulas, found {len(formulas)}"
        )
    if not formulas:
        warnings.append("no native display OMML equations found")

    bookmarks = {
        attr(item, "name")
        for item in document.xpath(".//w:bookmarkStart", namespaces=NS)
        if re.fullmatch(r"ref_\d{3,}", attr(item, "name"))
    }
    all_internal_links = document.xpath(".//w:hyperlink[@w:anchor]", namespaces=NS)
    links = [
        item
        for item in all_internal_links
        if re.fullmatch(r"ref_\d{3,}", attr(item, "anchor"))
    ]
    anchor_sequence = [attr(item, "anchor") for item in links]
    anchors = set(anchor_sequence)
    stats["reference_bookmarks"] = len(bookmarks)
    stats["internal_citations"] = len(links)
    stats["other_internal_links"] = len(all_internal_links) - len(links)
    broken = sorted(anchors - bookmarks)
    if broken:
        errors.append("internal citation targets missing: " + ", ".join(broken))
    if args.expect_references is not None and len(bookmarks) != args.expect_references:
        errors.append(
            f"expected {args.expect_references} reference bookmarks, "
            f"found {len(bookmarks)}"
        )

    uncited = sorted(bookmarks - anchors)
    if uncited:
        errors.append("bibliography entries are never cited: " + ", ".join(uncited))

    first_seen: list[int] = []
    for anchor in anchor_sequence:
        number = int(anchor.removeprefix("ref_"))
        if number not in first_seen:
            first_seen.append(number)
    expected_first_seen = list(range(1, len(first_seen) + 1))
    if first_seen != expected_first_seen:
        errors.append(
            "references are not numbered by first citation; got "
            + ", ".join(map(str, first_seen))
        )

    for index, link in enumerate(links, start=1):
        runs = link.xpath(".//w:r", namespaces=NS)
        if not runs:
            errors.append(f"internal citation {index} has no text run")
            continue
        displayed = "".join(link.xpath(".//w:t/text()", namespaces=NS))
        displayed_match = re.fullmatch(r"\[(\d+)\]", displayed.strip())
        target_number = int(attr(link, "anchor").removeprefix("ref_"))
        if displayed_match is None:
            errors.append(
                f"internal citation {index} has unexpected display text: {displayed!r}"
            )
        elif int(displayed_match.group(1)) != target_number:
            errors.append(
                f"internal citation {index} displays [{displayed_match.group(1)}] "
                f"but targets reference {target_number}"
            )
        if not link.xpath(
            ".//w:vertAlign[@w:val='superscript']", namespaces=NS
        ):
            warnings.append(f"internal citation {index} is not superscript")
        if not link.xpath(".//w:color[@w:val='000000']", namespaces=NS):
            warnings.append(f"internal citation {index} is not explicitly black")
        if not link.xpath(".//w:u[@w:val='none']", namespaces=NS):
            warnings.append(f"internal citation {index} may be underlined")

    sections = document.xpath(".//w:sectPr", namespaces=NS)
    stats["sections"] = len(sections)
    if not sections:
        errors.append("no section properties found")
    for index, section in enumerate(sections, start=1):
        size = section.find("w:pgSz", namespaces=NS)
        margins = section.find("w:pgMar", namespaces=NS)
        if size is None or (attr(size, "w"), attr(size, "h")) != ("11906", "16838"):
            errors.append(f"section {index} is not A4 portrait")
        if margins is None:
            errors.append(f"section {index} has no page margins")
        else:
            for side in ("top", "right", "bottom", "left"):
                hip = int(attr(margins, side) or 0)
                if abs(hip - 1701) > 20:
                    errors.append(
                        f"section {index} {side} margin is {hip} twips, not 3 cm"
                    )
        if section.find("w:headerReference", namespaces=NS) is None:
            warnings.append(f"section {index} has no header relationship")
        if section.find("w:footerReference", namespaces=NS) is None:
            warnings.append(f"section {index} has no footer relationship")

    normal = styles.xpath(".//w:style[@w:styleId='Normal']", namespaces=NS)
    if normal:
        east_asia = normal[0].xpath(".//w:rFonts/@w:eastAsia", namespaces=NS)
        ascii_font = normal[0].xpath(".//w:rFonts/@w:ascii", namespaces=NS)
        size = normal[0].xpath(".//w:sz/@w:val", namespaces=NS)
        if not east_asia or east_asia[-1] != "宋体":
            warnings.append("Normal style East Asian font is not explicitly SimSun")
        if not ascii_font or ascii_font[-1] != "Times New Roman":
            warnings.append("Normal style Latin font is not Times New Roman")
        if not size or size[-1] != "24":
            warnings.append("Normal style is not explicitly 12 pt")
    else:
        errors.append("Normal paragraph style is missing")

    stats["media_parts"] = len(
        [name for name in archive.namelist() if name.startswith("word/media/")]
    )
    header_parts = [
        name for name in archive.namelist() if re.fullmatch(r"word/header\d+\.xml", name)
    ]
    footer_parts = [
        name for name in archive.namelist() if re.fullmatch(r"word/footer\d+\.xml", name)
    ]
    stats["header_parts"] = len(header_parts)
    stats["footer_parts"] = len(footer_parts)
    if not header_parts:
        warnings.append("DOCX package has no header part")
    if not footer_parts:
        warnings.append("DOCX package has no footer part")
    elif not any(
        "PAGE" in "".join(
            etree.fromstring(archive.read(name)).xpath(".//w:instrText/text()", namespaces=NS)
        )
        for name in footer_parts
    ):
        warnings.append("footer parts do not contain a PAGE field")

    if "word/settings.xml" in archive.namelist():
        settings = etree.fromstring(archive.read("word/settings.xml"))
        update_fields = settings.find("w:updateFields", namespaces=NS)
        if update_fields is None or attr(update_fields, "val").lower() not in {
            "true",
            "1",
        }:
            warnings.append("field updating on open is not enabled")
    else:
        warnings.append("DOCX package has no settings.xml")

    if document.xpath(".//w:altChunk", namespaces=NS):
        warnings.append("document contains altChunk content that may not be portable")

    print_report(args.docx, errors, warnings, stats)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
