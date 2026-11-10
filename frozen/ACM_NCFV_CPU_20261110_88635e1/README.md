# CPU baseline synchronized with GitHub main (2026-11-10)

The user authorized updating both CPU and GPU versions. This snapshot uses
`kirabo/main` at `88635e1`, including the RANS implicit transport and SST-1994
fixes. It preserves the three NCFV diagnostic-hook files carried by GPUACM
`e113c2f`. The SST guide now states the implemented `1e-20` cross-diffusion floor.

`cpu_sources.sha256` covers all four CPU source/application directories.
`metadata.json` records provenance and validation; `preserved_cpu.patch` records
the retained NCFV additions. Source archives and six compiled executables are
local artifacts listed in `artifacts.sha256`, excluded from Git.

Run `python3 scripts/verify_cpu_freeze.py --sources-only` after cloning.
The default check additionally requires the local archive and binaries.
Earlier snapshots remain in GPUACM history and in the original workspace.
