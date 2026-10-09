# Solver Configuration {#solver_config}

DNDSR CPU executables read JSON/JSONC cases. Choose a case from the matching
family, dimension and model directory; the three families use different
configuration structures. The generated schema beside each model's cases
is the field reference for that executable.

## Executable selection

Maintained cases carry a top-level `solver` object. This is a metadata fragment
for `euler2D`, rather than a complete simulation configuration:

```json
{
    "solver": {
        "type": "Euler",
        "discretization": "CFV",
        "model": "NS_2D",
        "fieldNVariables": 4
    },
    "$schema": "euler2D_schema.json"
}
```

| Family | Selection and state size | Case directories |
| --- | --- | --- |
| Euler/CFV | `Euler`, `CFV`; model selects NS, SA, 2EQ or EX | `cases/euler/{2D,3D}`, with `SA`, `2EQ`, `EX` subdirectories |
| Constant-density ACM | `ACM`, `CFV`; `ConstantDensity2D` / `ConstantDensity3D`; four variables | `cases/acm/{2D,3D}` |
| Variable-density ACM | `ACMVariable`, `CFV`; `VariableDensity2D` / `VariableDensity3D`; five variables | `cases/acm/{2D,3D}/variable_density` |
| NCFV Euler | `ncfv_euler`, `NCFV`, `IdealGas`; `dimension + 2` variables | `cases/ncfv_euler/{2D,3D}` |

`euler` retains the upstream NS convention: two-dimensional geometry with
three velocity components and five conserved variables. `euler2D` uses
NS_2D, two velocity components and four conserved variables. Selecting a
different executable requires a matching model and all state-array lengths;
changing only the metadata does not convert a case.

EulerEX state size is dynamic. Prefer `solver.fieldNVariables` in JSON;
the upstream form `eulerEX.exe 14 config.json` is also accepted. A reaction
case requires an EX executable compiled with Cantera and a compatible species
layout. See @ref solver_split_zh for the complete executable/model table.

## Loading defaults and applying overrides

Euler checks the case directory for `<target>_default_config.json`, such as
`euler2D_default_config.json`. If the file exists, it is read; otherwise the
model's compiled defaults are used in memory. The case and CLI overrides are
then merged. Startup does not overwrite an existing default JSON file.

ACM and ACMVariable read complete single-file configurations and do not
search for adjacent defaults. Their loaders supply implemented compatibility
fields. NCFV merges its typed `Configuration` defaults with the case, including
defaults for boundary and initial-field entries. These loading rules differ;
compare final merged parameters when reproducing an upstream run.

All native entry points accept paired `-k` / `-v` arguments: keys are JSON
pointers; values are parsed as JSON literals, with a plain-string fallback. For example, validate a changed NCFV
iteration count without starting a simulation:

```bash
(cd build && ./app/ncfv_euler2D.exe \
    ../cases/ncfv_euler/2D/ncfv_euler2D.json \
    -k /time/iterations -v 10 --check-config)
```

Use the actual field names below. A generic top-level `meshFile`, `dt` or
`outputDirectory` is not a portable substitute for a family's nested fields.

## Main configuration sections

| Purpose | Euler/CFV | ACM / ACMVariable | NCFV Euler |
| --- | --- | --- | --- |
| Mesh and partitioning | `dataIOControl` | `meshSettings` | `mesh` |
| Physical model | `eulerSettings` | `acmSettings`, `turbulenceSettings` | `physics`, including `viscous` |
| Initial field | `eulerSettings` initializers and state values | `initialState` and family-specific controls | `initialField`, `physics.initialPrimitive` |
| Spatial reconstruction | `vfvSettings`, `implicitReconstructionControl`, `limiterControl` | `reconstructionSettings`, `vfvSettings` | `algorithm`, `reconstruction` |
| Time integration | `timeMarchControl`, `implicitCFLControl`, `convergenceControl` | `timeMarchSettings` | `time` |
| Linear solve | `linearSolverControl` | `timeMarchSettings` | Explicit SSPRK3 driver |
| Boundary conditions | `bcSettings`, `boundaryDefinition` | `boundaryConditions`, `defaultBoundaryType` | `physics.boundaryZones`, `physics.boundaryMode` |
| Output and restart | `outputControl`, `dataIOControl`, `restartState` | `outputSettings`; `restartSettings` for constant-density ACM | `io` |

For Euler, representative fields are `dataIOControl.meshFile`,
`timeMarchControl.nTimeStep`, `timeMarchControl.dtImplicit` and
`eulerSettings.idealGasProperty`. Reaction controls are nested under
`eulerSettings.reactiveFlow`. SA and two-equation settings belong to the
matching executable/model; their presence does not switch the executable.

ACM uses `meshSettings.meshFile`, `timeMarchSettings.nSteps` and
`outputSettings.directory`. Constant-density state order is `[u,v,w,p]`,
including in 2D. Variable-density state order is `[rho,mx,my,mz,p]`.
Pseudo-time and physical dual-time modes have different meanings and
requirements; consult the [ACM](../../src/ACM/README.md) and
[variable-density ACM](../../src/ACMVariable/README.md) module guides before
changing integrators. `restartSettings` belongs to constant-density ACM;
ACMVariable currently has result output without restart loading.

NCFV uses `mesh.meshFile`, `time.iterations`, `time.timeStep` and
`io.outputPrefix`. Primitive states are `[rho,u,v,p]` in 2D and
`[rho,u,v,w,p]` in 3D; conserved states end in total energy density.
`algorithm.mode` selects `EfficientDifferential` or `TraditionalQuadrature`.
The legacy disabled `Roe_M2` selector was replaced by `Roe` in migrated cases;
historical timing/error results need recomputation. See the
[NCFV module guide](../../src/NCFV/README.md).

Boundary names must match the mesh zones. Euler's `bcSettings`, ACM's
`boundaryConditions` and NCFV's `physics.boundaryZones` have distinct entry
formats and type names. Copy a boundary example from the same family and
check its schema; do not transplant an entry from another family's case.

## Paths and complete examples

Mesh, output and restart paths are interpreted relative to the process working
directory, normally `build/` or `build-reactive/`. `$schema` is relative to the
case file. Reaction mechanism lookup also supports `DNDS_MECH_PATH` and
`CANTERA_DATA`. Relocating a JSON file does not make its mesh paths relative
to that file.

Start with a maintained complete configuration:

- [Euler NS_2D](../../cases/euler/2D/euler2D_config.json)
- [Constant-density ACM 2D](../../cases/acm/2D/acm2D.json)
- [Variable-density ACM 2D](../../cases/acm/2D/variable_density/acmVariable2D.json)
- [NCFV Euler 2D](../../cases/ncfv_euler/2D/ncfv_euler2D.json)
- [NCFV Euler 3D periodic vortex](../../cases/ncfv_euler/3D/NCFV_periodic_hex_iv10.json)

Preserve comments and case notes when editing. Some configurations require
external source meshes or partitioned mesh files that are not supplied locally.

## Configuration checks and schema generation

Build the selected executable, then use `--check-config` to load and validate
its typed parameters and model selection. This skips mesh reads and result
output, so it does not establish input availability or convergence.

```bash
(cd build && ./app/euler2D.exe ../cases/euler/2D/euler2D_config.json --check-config)
(cd build && ./app/acmVariable2D.exe \
    ../cases/acm/2D/variable_density/acmVariable2D.json --check-config)
(cd build && ./app/ncfv_euler3D.exe --emit-schema)

python3 cases/validate_configs.py --quiet
python3 scripts/check_solver_cases.py --build-dir build \
    --reactive-build-dir build-reactive
```

The native audit requires all solver entry points; reaction cases require the
Cantera build. Without one, it explicitly reports those cases as requiring
another build. Mesh manifests are data descriptions and are excluded from
solver-configuration validation. These scripts do not import DNDSR's Python
extension modules.

Regenerate all 15 per-executable schemas with the Cantera Euler build. ACM and
NCFV can use the same build or the optional CPU fallback argument:

```bash
cmake --preset reactive-test
cmake --build --preset schemas --parallel 4
bash cases/update_schemas.sh build-reactive build
python3 cases/validate_configs.py --quiet

# Generate/check without replacing the existing schemas
DNDS_SCHEMA_CHECK_ONLY=1 bash cases/update_schemas.sh build-reactive build
```

Configuration CTests require `all_solvers` in addition to the unit-test targets:

```bash
cmake --build build --target all_solvers all_unit_tests --parallel 4
ctest --test-dir build -L solver_config --output-on-failure
```

## See Also

- @ref building — build targets and dependencies
- @ref user_guide — selecting and running a case
- @ref solver_split_zh — migration and upstream comparison (中文)
- @ref euler_unit_tests — Euler regression tests
