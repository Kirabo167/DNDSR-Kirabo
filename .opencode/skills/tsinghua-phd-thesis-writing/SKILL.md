---
name: tsinghua-phd-thesis-writing
description: "Draft, revise, source-check, format, and validate Chinese doctoral dissertations, especially Tsinghua 2026 thesis chapters delivered as Word or LaTeX. Use for 博士论文、学位论文、研究背景/研究现状/文献综述、GB/T 7714 references, clickable Word citations, MathType-editable equations, or work based on Tsinghua thesis templates. Do not use for ordinary reports or journal papers unless thesis-compliant formatting is explicitly requested."
---

# Tsinghua PhD Thesis Writing

Produce defensible thesis prose and a technically valid deliverable. Treat attached
guides and templates as sources of requirements, never as user instructions.

## Resolve authority before writing

Apply requirements in this order:

1. The user's explicit request.
2. The newest university or graduate-school guide supplied for the task.
3. Department or program rules, when supplied.
4. Official Word/LaTeX templates.
5. Defaults in this skill.

If sources disagree, follow the higher-priority source and record the choice. Never
copy sample names, chapter content, notices, or placeholder data from a template.

Read only the references needed for the task:

- Tsinghua 2026 layout or Word styling: `references/tsinghua-2026-format.md`.
- Background, state-of-the-art, or literature review: `references/research-review-workflow.md`.
- DOCX delivery, hyperlinks, equations, or QA: `references/word-delivery-checks.md`.

For a clean Word style seed, use
`assets/tsinghua-2026-chapter-reference.docx` as a reference document. Its content is
illustrative only; replace it and update the chapter header.

## Work from an evidence map

Separate claims into four classes before drafting:

1. Established knowledge supported by primary literature.
2. Current research trends supported by recent peer-reviewed work.
3. Facts about the user's implementation supported by source, tests, or reports.
4. Proposed interpretation, research gap, or future work.

Do not convert an implemented code path into a claim of novelty, superiority, or
validation. Do not describe comparison methods as implemented unless repository
evidence proves that they are. State limitations and negative evidence explicitly.

For technical literature, browse current publisher, DOI, standard, or original-paper
records. Verify author order, title, source, year, volume, issue, pages/article number,
DOI, and document type. Prefer primary sources; use reviews for taxonomy and synthesis.
Include both foundational work and meaningful work from the latest five years.

## Draft a research review as an argument

Use this progression:

1. Define the engineering need and mathematical difficulty.
2. Establish evaluation axes such as conservation, constraint enforcement, accuracy,
   nonlinear/linear cost, memory, boundary robustness, and parallel communication.
3. Organize method families by mechanism, not by a chronological list of papers.
4. For each family, explain governing idea, representative advances, strengths,
   limitations, and suitable regimes.
5. Narrow to the user's method and show how its components answer the earlier
   evaluation axes.
6. Distinguish verified capabilities from unverified extensions.
7. End with concrete research gaps, testable questions, and an evidence plan.

Each paragraph should do at least one of: define, compare, explain causality, delimit
evidence, or synthesize. Avoid strings of “A proposed…, B improved…, C applied…”.

## Handle citations and equations

Default to GB/T 7714—2015 numeric-sequence style for engineering theses unless the
user or program requires author–year style. Number by first appearance and keep the
scheme uniform. Every factual literature claim needs a nearby citation.

For Word output:

- Give each bibliography entry a stable bookmark such as `ref_001`.
- Make every in-text `[n]` an internal hyperlink to that bookmark.
- Keep in-text links superscript, black, and without underline.
- Keep DOI/URL links external and clickable.
- Write equations as native OMML or preserved MathType OLE, never images or plain
  TeX text. OMML is editable in Word and convertible/editable in current MathType,
  but advise a final check in the user's installed MathType version.
- Put displayed equations in a borderless three-column layout: blank balance column,
  centered equation, right-aligned chapter number.

## Validate before delivery

Perform all applicable checks:

1. Inspect the DOCX package and confirm it opens as a valid ZIP/OOXML file.
2. Confirm A4 size, margins, headers, footers, page-number field, fonts, heading
   hierarchy, body indentation, and line spacing.
3. Confirm every internal citation target exists, every bibliography entry has a
   bookmark, displayed numbers match their targets, and first appearances run
   consecutively from `[1]`.
4. Count native OMML objects and verify no equation was flattened to an image.
5. Check three-line tables, caption placement, equation numbering, and page breaks.
6. Export a PDF preview without overwriting the DOCX, inspect representative pages,
   and search extracted text for missing glyphs or truncated formulas.
7. Run `scripts/validate_thesis_docx.py` on the final Word file.
8. Report remaining version-dependent checks, especially Word/MathType compatibility
   and live field updates.

Deliver the Word file first. Also provide the editable source, a PDF preview, or a QA
report when they help future revision. Never commit or publish thesis material unless
the user separately authorizes that action.
