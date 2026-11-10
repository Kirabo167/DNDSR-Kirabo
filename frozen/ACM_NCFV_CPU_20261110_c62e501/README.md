# CPU reference for explicit isotropic k stress

This snapshot follows GitHub main `c62e501` and preserves the local NCFV
diagnostic hooks. The user authorized updating the local CPU contents and GPU
implementation, with the resulting commit published only to `GPUACM`.

Two-equation ACM now stores static pressure by default and includes isotropic
kinetic-energy stress. `includeIsotropicStress=false` selects the legacy
modified-pressure convention. Old field-only restarts need an explicit
`pressureIsModified=true` conversion when switching to the new convention.

The 63-source manifest covers `src/ACM`, `src/NCFV`, `app/ACM`, `app/NCFV`.
The archive and six binaries are local artifacts; metadata, CMake cache and
hash manifests are versioned. Earlier CPU snapshots are retained unchanged.
Run `python3 scripts/verify_cpu_freeze.py --sources-only` after cloning, or the
default command on this machine to also check the nine local artifacts.
