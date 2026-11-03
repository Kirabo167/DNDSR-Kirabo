# KT2002 Configuration 12 — five NCFV limiter methods

Reference: Kurganov & Tadmor (2002), DOI 10.1002/num.10025,
Configuration 12 (initial states p.600), Fig.3.12 (p.602), final time **0.25**.

| Quadrant | rho | u | v | w | p |
| --- | ---: | ---: | ---: | ---: | ---: |
| TR (1) | 0.5313 | 0 | 0 | 0 | 0.4 |
| TL (2) | 1 | 0.7276 | 0 | 0 | 1 |
| BL (3) | 0.8 | 0 | 0 | 0 | 1 |
| BR (4) | 1 | 0 | 0.7276 | 0 | 1 |

Interfaces: x=y=0.5; gamma=1.4. Initial data are exact conservative
averages over clipped median dual volumes, including mixed nodes.

- 800×800×5 cells; domain 1×1×0.05; dz=0.01.
- z faces periodic; x/y faces use exterior extrapolation (SupersonicOutlet).
- Original stencil, EfficientDifferential, SSPRK3, CFL=0.475.
- Reuse the frozen Config3/RP34 executable and source snapshot; no rebuild.
- SG_BJ_PRES, SG_SIGN_BUDGET, FC_IDP_BINARY, FC_IDP_CONT, LLF_FIRST_ORDER.
- Each case uses 128 MPI ranks on distinct physical cores; two slots, no SMT.
- The detached controller waits until the existing RP3/RP4 batch is complete
  and existing owned NCFV solver ranks have exited, then runs all five
  40×40×5 preflights to t=0.25 before launching the production cases.
- Every 200 steps save a full HDF5 checkpoint. VTK is initial/final only.
  Native five-layer exports and positivity/z-span summaries are automatic.
- Output: data/out/NCFV/KT2002_Config12_800x800x5_np128_20261102/.
  suite_status.json records queue state, progress, timing, configuration and
  executable hashes; controller.log records execution. Inputs are frozen.

No existing calculations are stopped or modified. Duplicate launch is refused.
