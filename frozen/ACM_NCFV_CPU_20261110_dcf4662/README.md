# Documentation refresh after main dcf4662

Main added README updates after the numerical `c62e501` synchronization.
This snapshot records those CPU documentation changes and corrects the ACM
README description of existing neighbour turbulence transport face blocks.
Numerical sources are unchanged from the validated `c62e501` snapshot.

The source archive and 63-source manifest are specific to this snapshot.
The artifact manifest reuses the previous snapshot's six validated binaries
and CMake cache through relative paths; keep that snapshot available locally.
No new compilation or numerical testing is needed for these Markdown changes.
The original test evidence remains under `data/out/project_sync_20261110_c62e501`.

Only GPUACM is to be published. The pure-CPU GitHub main branch is not modified
by this delivery. Source-only verification remains available after cloning.
