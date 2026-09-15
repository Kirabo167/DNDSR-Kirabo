# NCFV JCP-style manuscript draft

This directory contains a manuscript expanded on 15 September 2026 using
Runzhi Ma's April 2026 master's thesis, Chapters 3–5, and the NCFV
efficient/differential implementation and its existing diagnostic results.
It uses numerical citations and the standard sections expected in a Journal of
Computational Physics submission, while remaining portable in this workspace.

## Deliverables

- `NCFV_JCP_manuscript.tex` — primary LaTeX source.
- `references.bib` — BibTeX database, including the supplied thesis (no DOI).
- `NCFV_JCP_manuscript.docx` — Word rendition with editable native math.
- `NCFV_JCP_manuscript.pdf` — compiled LaTeX preview.
- `NCFV_JCP_manuscript.md` — generated companion text.
- `THESIS_SOURCE_MAP.md` — Chinese source map, page locations and corrections.
- `thesis_reported_convergence.csv` — transcribed thesis data, with provenance.
- `figures/` — manuscript figures, including the redrawn thesis data.
- `NCFV_JCP_thesis_supplement_20260915.zip` — complete deliverable/source bundle.

LaTeX is the single text source. Word and Markdown are generated from it so
equations, tables and citations remain synchronized.

## Rebuild all artifacts

From the repository root:

    venv/bin/python docs/reports/NCFV_JCP_manuscript/build_manuscript.py

Requires matplotlib, pandoc, latexmk, XeLaTeX and BibTeX. The script checks
transcribed convergence rates, creates the figure, compiles LaTeX, resolves
cross-references for Word, applies numeric citations, verifies native Word
equations and tables, and packages all files. After extracting the ZIP
elsewhere, use a Python interpreter that has matplotlib.

## Compile LaTeX

```bash
cd docs/reports/NCFV_JCP_manuscript
latexmk -xelatex -interaction=nonstopmode NCFV_JCP_manuscript.tex
```

The source is deliberately self-contained and compiles with the workspace TeX
installation.  For an official Elsevier upload, replace the `article` class
with `elsarticle` and use `elsarticle-num.bst`; retain the manuscript text,
numbered bibliography, declarations and data-availability statement.

## Evidence boundary

The thesis reports near-third-order L1 convergence for one-dimensional and
extruded-vortex tests. These are attributed source results, not new runs.
The separate repository full-vortex sequence has decreasing component-wise
L2 errors but does not yet demonstrate stable third-order convergence.
The thesis arithmetic/storage model and repository measured time/PSS are
reported with their own definitions and quadrature baselines.
See THESIS_SOURCE_MAP.md for precise qualifications and source locations.
