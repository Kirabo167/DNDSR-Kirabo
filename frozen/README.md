# Frozen CPU provenance

`active_cpu_snapshot.txt` selects the source baseline used by the independent
ACMGPU/NCFVGPU modules. The original snapshot manifests and build metadata are
versioned here. CPU sources remain in `src/ACM`, `src/NCFV`, `app/ACM`, and
`app/NCFV` and must match the selected `cpu_sources.sha256`.

After cloning the repository, verify those sources with:

```bash
python3 scripts/verify_cpu_freeze.py --sources-only
```

Compiled executables, source archives, and simulation outputs remain local and
are excluded from Git. The strict default check also verifies those local
snapshot artifacts and is intended for the original development workspace:

```bash
python3 scripts/verify_cpu_freeze.py
```

The `artifacts.sha256` files describe the complete local snapshots; they do not
imply that the binaries or archives are included in this Git branch. Preserve
the selected manifests rather than regenerating them from changed CPU sources.
