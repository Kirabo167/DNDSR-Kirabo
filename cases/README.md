# Solver cases

Run solvers from a build directory. Mesh, mechanism, restart and output paths
inside case JSON/JSONC files remain relative to that working directory;
`$schema` links are relative to the configuration file.

| Family | Dimension | Executables | Cases |
| --- | --- | --- | --- |
| Euler/CFV | 2D | `euler`, `euler2D`, `eulerSA`, `euler2EQ`, `eulerEX` | `euler/2D`, with `SA`, `2EQ`, `EX` subdirectories |
| Euler/CFV | 3D | `euler3D`, `eulerSA3D`, `euler2EQ3D`, `eulerEX3D` | `euler/3D`, with `SA`, `2EQ`, `EX` subdirectories |
| ACM | 2D | `acm2D`, `acmVariable2D` | `acm/2D`, with `variable_density` for variable density |
| ACM | 3D | `acm3D`, `acmVariable3D`, `ACM` (3D compatibility name) | `acm/3D`, with `variable_density` for variable density |
| NCFV Euler | 2D | `ncfv_euler2D` | `ncfv_euler/2D` |
| NCFV Euler | 3D | `ncfv_euler3D` | `ncfv_euler/3D` |

`euler` retains the upstream NS model: 2D geometry with three velocity
components and five conserved variables. `euler2D` uses the NS_2D model,
two velocity components and four conserved variables. Their configurations
cannot be interchanged. NCFV stores `dimension + 2` conserved variables;
constant-density ACM always stores `[u,v,w,p]` and variable-density ACM
stores `[rho,mx,my,mz,p]`.

Euler reads a case-adjacent `<target>_default_config.json` when present and
merges the case plus CLI overrides without overwriting that file; missing
defaults are constructed in memory. ACM uses complete single-file cases;
NCFV merges typed defaults. These differences matter when reproducing an
upstream run. See [the migration and upstream comparison guide](../docs/guides/solver_split_zh.md).

The top-level `solver` object must match the selected executable. For NCFV, the metadata fragment is:

```json
{
    "solver": {
        "type": "ncfv_euler",
        "discretization": "NCFV",
        "model": "IdealGas",
        "fieldNVariables": 4
    },
    "dimension": 2
}
```

Use `--check-config` to validate typed parameters without mesh reads or result
output. Passing this check does not establish mesh availability or convergence. Use `--emit-schema` for the executable's schema. EulerEX also accepts
the upstream positional state size, such as `eulerEX.exe 14 config.json`;
`solver.fieldNVariables` in JSON is the preferred setting.

```bash
cmake --build build --target all_solvers --parallel 4
python3 scripts/check_solver_cases.py --build-dir build
(cd build && ./app/ncfv_euler3D.exe \
    ../cases/ncfv_euler/3D/NCFV_periodic_hex_iv10.json)
(cd build && ./app/acm2D.exe ../cases/acm/2D/acm2D.json)
```

For a complete audit including reaction cases, add
`--reactive-build-dir build-reactive`. A CPU build without Cantera reports
reaction cases as requiring another build rather than treating them as
nonreactive simulations.

Legacy cases using the retired NCFV `Roe_M2` selector now use `Roe`.
Archived timing/error data still describe their original flux setting and
must be recomputed before comparing with the updated configurations.

Regenerate all 15 schemas after building the required executables:

```bash
cmake --build --preset schemas --parallel 4
bash cases/update_schemas.sh build-reactive build
python3 cases/validate_configs.py --quiet
```

`DNDS_SCHEMA_CHECK_ONLY=1` generates and validates without replacing schemas.
The update tool uses the Cantera build for Euler and optionally the second
CPU build for ACM/NCFV. Mesh manifest JSON files describe data and are skipped
by the solver schema validator.

Meshes and diagnostic tools previously in `cases/NCFV` are now under
`ncfv_euler/3D`; mesh manifests are data descriptions rather than solver
configurations. Historical result files remain with their cases.

## Large mesh generation

The smaller generated periodic fixtures are versioned alongside their cases.
The 107 MiB `periodic_tet_iv80.cgns` fixture is generated locally rather than
stored in Git. After building the external CGNS library, create it from the
repository root when needed:

```bash
python3 cases/ncfv_euler/3D/diagnostics/generate_periodic_tet_hex_meshes.py \
    --topologies tet --sizes 80
```

The generator preserves existing meshes and refuses to overwrite an existing
file. Its manifest remains versioned. The larger refined prism meshes are
also local generated data, rather than bundled Git fixtures.
