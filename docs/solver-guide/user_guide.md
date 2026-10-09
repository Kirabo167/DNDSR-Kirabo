# User Guide {#user_guide}

DNDSR provides three CPU solver families: compressible Euler/Navier-Stokes
with CFV reconstruction, artificial-compressibility ACM, and node-centred
NCFV Euler/Navier-Stokes. Choose the family, dimension and model first, then
build its executable and use a matching JSON/JSONC case. The Python/CUDA
EulerP path has separate modules and build options; see @ref building.

## Choose an executable and case

| Flow/model | 2D target | 3D target | Case location |
| --- | --- | --- | --- |
| Euler / laminar Navier-Stokes | `euler` or `euler2D` | `euler3D` | `cases/euler/{2D,3D}` |
| Spalart-Allmaras | `eulerSA` | `eulerSA3D` | Euler dimension directory, `SA/` |
| Two-equation RANS | `euler2EQ` | `euler2EQ3D` | Euler dimension directory, `2EQ/` |
| Extended/multi-species Euler | `eulerEX` | `eulerEX3D` | Euler dimension directory, `EX/` |
| Constant-density ACM | `acm2D` | `acm3D` | `cases/acm/{2D,3D}` |
| Variable-density ACM | `acmVariable2D` | `acmVariable3D` | ACM dimension directory, `variable_density/` |
| NCFV Euler / laminar viscous flow | `ncfv_euler2D` | `ncfv_euler3D` | `cases/ncfv_euler/{2D,3D}` |

`ACM` is a compatibility target for constant-density 3D ACM. `euler` retains
upstream's five-variable NS model on 2D geometry; `euler2D` uses the four-variable
NS_2D model. They require different state-array lengths. The top-level `solver`
metadata is checked against the selected executable; it does not dispatch to
another family. See @ref solver_config for fields and loading rules.

## Build and validate

From the repository root, configure the CPU build and select the targets:

```bash
cmake --preset release-test
cmake --build build --target euler2D acm2D ncfv_euler3D --parallel 4

# Optional: all variants, or one whole family
cmake --build --preset solvers --parallel 4
cmake --build --preset acm --parallel 4
cmake --build --preset ncfv_euler --parallel 4
```

Only `euler` belongs to the default solver build. Group targets are
`all_euler`, `all_acm`, `all_ncfv_euler` and `all_solvers`. Limit parallel
compiler jobs to fit the available memory; separate executables allow
selective builds but do not limit concurrent jobs. Dependencies, Python
installation and Cantera configuration are covered in @ref building.

Check parameters before a full simulation:

```bash
(cd build && ./app/euler2D.exe ../cases/euler/2D/euler2D_config.json --check-config)
(cd build && ./app/acm2D.exe ../cases/acm/2D/acm2D.json --check-config)
(cd build && ./app/ncfv_euler3D.exe \
    ../cases/ncfv_euler/3D/NCFV_periodic_hex_iv10.json --check-config)
python3 cases/validate_configs.py --quiet
```

`--check-config` validates typed parameters without mesh loading or solver
result output. Check that the referenced mesh and other inputs are present
before launching. Some engineering cases depend on external mesh fixtures.

## Mesh and path conventions

Fresh mesh input uses DNDSR's CGNS mesh infrastructure. Euler can also read
serialized partitioned meshes as configured by `dataIOControl.readMeshMode`.
Preserve the intended boundary-zone names and periodic pairing when preparing
a mesh. Mesh utilities and Python geometry usage are described in
@ref python_geom_guide.

Case paths passed to the CLI, and mesh/output/restart paths inside the case,
are relative to the launch working directory. Maintained examples normally
run from `build/` or `build-reactive/`; `$schema` alone is relative to the JSON
file. Moving a case to a new subdirectory does not change how its mesh paths
are resolved.

Euler reads an existing case-adjacent `<target>_default_config.json`, then
merges the case and CLI overrides; absent defaults are constructed in memory.
It does not overwrite existing defaults at startup. ACM uses complete
single-file cases, and NCFV merges its typed defaults. Match final merged
parameters when comparing to upstream.

## Run a simulation

After confirming the input mesh is available, launch the selected program
from its build directory:

```bash
(cd build && OMP_NUM_THREADS=1 mpirun -np 4 ./app/euler2D.exe \
    ../cases/euler/2D/euler2D_config.json)
(cd build && OMP_NUM_THREADS=1 mpirun -np 4 ./app/acm2D.exe \
    ../cases/acm/2D/acm2D.json)
(cd build && OMP_NUM_THREADS=1 mpirun -np 4 ./app/ncfv_euler3D.exe \
    ../cases/ncfv_euler/3D/NCFV_periodic_hex_iv10.json)
```

Adjust MPI rank count and OpenMP threads for the mesh and machine. Output
directories and physical/pseudo-time controls come from the selected case.
Review residuals and the requested stopping criteria; successful startup or
a short smoke test does not establish a converged solution.

All three families support JSON-pointer overrides. This NCFV example changes
the explicit step count while validating the resulting configuration:

```bash
(cd build && ./app/ncfv_euler3D.exe \
    ../cases/ncfv_euler/3D/NCFV_periodic_hex_iv10.json \
    -k /time/iterations -v 10 --check-config)
```

### Reactive EulerEX

Reaction-enabled cases need Cantera:

```bash
cmake --preset reactive-test
cmake --build build-reactive --target eulerEX --parallel 4
(cd build-reactive && \
    DNDS_MECH_PATH=../external/cfd_externals/install/data \
    ./app/eulerEX.exe ../cases/euler/2D/EX/react_test.json --check-config)
```

Use the same command without `--check-config`, optionally under `mpirun`, to
run after supplying the mesh and mechanism. `solver.fieldNVariables` sets
the dynamic state size; the upstream positional form such as
`eulerEX.exe 14 config.json` remains available. Cantera-free EX builds reject
reaction-enabled configurations.

## Results and restart

Output formats and checkpoint controls differ by family:

| Family | Result format | Main controls |
| --- | --- | --- |
| Euler/CFV | VTK, VTK-HDF and/or Tecplot, according to the case | `outputControl`, `dataIOControl` |
| Constant-density ACM | Parallel VTK-HDF (`.vtkhdf`) | `outputSettings`, `restartSettings` |
| ACMVariable | Parallel VTK-HDF (`.vtkhdf`) | `outputSettings`; restart loading is not implemented |
| NCFV Euler | Parallel VTK (`.pvtu` with `.vtu` pieces) | `io` |

Open the selected output in a reader that supports that format, such as
ParaView for VTK/VTK-HDF or Tecplot for Tecplot output. Available fields depend
on the model. Result output is not a universal restart format; follow the
family's restart settings and partition/history constraints, particularly
for dual-time integration. The [ACM](../../src/ACM/README.md),
[ACMVariable](../../src/ACMVariable/README.md) and
[NCFV](../../src/NCFV/README.md) guides describe their current limitations.

## Regression checks

Build solver entry points as well as unit tests before running configuration
CTest checks:

```bash
cmake --build build --target all_solvers all_unit_tests --parallel 4
ctest --test-dir build -L solver_config --output-on-failure
ctest --test-dir build -R '^(acm|ncfv|euler)_' --output-on-failure

# Audit all cases, including reaction cases from a separate Cantera build
python3 scripts/check_solver_cases.py --build-dir build \
    --reactive-build-dir build-reactive
```

The native audit requires all corresponding Euler targets in the Cantera
build. See @ref solver_split_zh for the dated validation results, unavailable
mesh fixtures and the limits of the upstream comparison.

## See Also

- @ref building — build instructions
- @ref solver_config — JSON configuration reference
- @ref solver_split_zh — solver migration and upstream comparison (中文)
- @ref troubleshooting — common issues
