# Word, citation-link, and equation delivery checks

## DOCX implementation

A `.docx` file is an OOXML ZIP package. Validate these parts:

- `word/document.xml`: headings, body, tables, bookmarks, hyperlinks, and OMML.
- `word/styles.xml`: Normal and heading typography.
- `word/settings.xml`: field-update setting when live fields are present.
- `word/header*.xml` and `word/footer*.xml`: current chapter and page field.
- `word/_rels/document.xml.rels`: external DOI/URL and header/footer relations.
- `[Content_Types].xml`: declared header/footer and media part types.

## Internal citations

Use bookmark names compatible with Word, such as `ref_001`, not spaces or punctuation.
An in-text citation should be:

1. a `w:hyperlink` with `w:anchor="ref_001"`;
2. a superscript run;
3. black (`000000`) with underline disabled;
4. paired with an existing `w:bookmarkStart` named `ref_001`.

Bookmarks may surround a paragraph as sibling elements instead of appearing inside it.
Validators must support both structures. Test actual clicking in desktop Word because
PDF viewers and office suites differ in internal-link handling.

## Editable equations

Accept:

- native Word OMML (`m:oMath` / `m:oMathPara`);
- preserved MathType OLE with an `Equation Native` stream.

Reject:

- screenshots or rasterized formula images;
- SVG-only formula renderings;
- raw TeX left as visible text.

For broad compatibility, avoid unnecessarily complex aligned-array constructs when
simple single-line OMML equations express the same mathematics. Use a borderless
three-column table for page-centered equations and right-aligned numbers. Verify long
equations in an exported PDF preview; a valid OMML object can still overflow a narrow
cell.

## Practical QA sequence

1. Test ZIP integrity.
2. Parse all XML parts.
3. Count OMML objects and equation-number labels.
4. Compare reference hyperlink anchors with bookmarks, match visible numbers to
   targets, and verify that first appearances are consecutively numbered.
5. Check citation superscript/color/underline properties.
6. Verify A4 and 3 cm margins in every section.
7. Verify header/footer relationships and a `PAGE` field.
8. Convert to PDF without resaving the source DOCX.
9. Inspect chapter opening, dense formula pages, tables, page breaks, and references.
10. Extract PDF text and search for replacement glyphs, lost symbols, truncated
    equations, duplicate headings, or stale sample headers.

Passing structural checks does not prove that a particular MathType version can edit
every OMML construct. State that limitation and request a final local Word/MathType
spot check when the user's installed version is unknown.
