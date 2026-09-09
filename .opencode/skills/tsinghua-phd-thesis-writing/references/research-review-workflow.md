# Evidence-led research review workflow

## Build the source matrix

Record one row per source:

| Field | Purpose |
|---|---|
| Full verified metadata | Prevent fabricated or malformed references |
| Primary/review/standard/software | Clarify evidentiary role |
| Method family and governing mechanism | Support thematic organization |
| Core contribution | State what the source actually establishes |
| Assumptions and regime | Prevent overgeneralization |
| Evidence type | Theory, manufactured solution, benchmark, experiment, scaling |
| Limitation | Enable critical synthesis |
| Thesis claim supported | Keep citations local and relevant |

Use DOI landing pages, publisher metadata, original PDFs, standards, and official
software records. Do not rely on search-result snippets for final metadata.

## Recommended review structure

1. Engineering context and mathematical structure.
2. A compact statement of governing equations and symbols.
3. Method taxonomy based on how the constraint/coupling is handled.
4. Cross-method comparison table using common criteria.
5. Development of the focal method from foundational to current work.
6. Mapping from literature to the user's actual implementation.
7. Evidence boundaries and unresolved problems.
8. Testable research questions and chapter conclusion.

## Comparison dimensions for CFD methods

- Continuous and discrete mass conservation.
- Pressure uniqueness and null-space treatment.
- Spatial and physical-time accuracy.
- Splitting, pseudo-time, nonlinear, and linear-solve errors.
- Robustness on skewed/non-orthogonal or moving meshes.
- Boundary-condition consistency.
- Memory, global synchronization, and communication pattern.
- Extensibility to variable density, turbulence, multiphysics, and adaptivity.
- Verification level: identities, unit tests, order tests, benchmark, experiment,
  uncertainty quantification, and strong/weak scaling.

## Language controls

Use “the implementation provides” for code-backed capability, “tests verify” for the
exact property covered by tests, and “may/can be investigated” for hypotheses.
Reserve “novel”, “first”, “superior”, “validated”, and “production-ready” for claims
supported by a documented literature search and proportionate evidence.

Explicitly distinguish:

- equation-level preconditioning from algebraic solver preconditioning;
- nominal discretization order from measured convergence order;
- a high-order residual with frozen low-order Jacobian from a consistent high-order
  Newton method;
- regression consistency from physical validation;
- qualitative visualization from quantitative benchmark agreement.

## Final citation audit

Check that:

1. Every numbered source is cited and numbers follow first appearance.
2. Every factual comparison has a source close to the claim.
3. Reviews are not used in place of original landmark papers.
4. Recent literature is substantive, not added only to satisfy a date quota.
5. Code/software citations do not substitute for method papers.
6. The bibliography contains no source that was not inspected.
