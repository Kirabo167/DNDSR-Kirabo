# DNDSR

DNDSR is a C++17/Python research framework for unstructured-grid
computational fluid dynamics. It combines compact finite-volume (CFV)
discretizations, distributed-memory MPI parallelism, optional CUDA execution,
and optional Cantera-based multi-species reacting-flow models.

This repository is the Kirabo fork of the
[upstream DNDSR project](https://github.com/CFDLAB-THU/DNDSR). It tracks the
upstream v0.3.1 code line and organizes CPU solvers into Euler/CFV, ACM
(constant and variable density), and NCFV Euler. The latter implements the
third-order **Node Center Finite Volume Method (NCFV)**. Project target,
namespace, and Python-package names remain `DNDSR`.

The native mesh and Cauchy CPU framework are additionally migrated from
[`611b0a0`](https://github.com/harryzhou2000/DNDSR/commit/611b0a0b7a6b3ac9a54be891919f9016a8756471).
They share the Runtime/Algebra data layer with this fork and expose separate
`NativeGeom`/`NativeCFV` interfaces alongside the established solver interfaces.

See the [v0.3.1 feature and migration guide](docs/guides/v0.3.1_new_features_zh.md)
for compatibility changes and the validation matrix. The upstream manuals and
API reference remain available at
[cfdlab-thu.github.io/DNDSR](https://cfdlab-thu.github.io/DNDSR/).

## Project Overview

| Module | Responsibility |
|---|---|
| `DNDS` | Runtime arrays, leased storage, DOF algebra, MPI plans, legacy array bridges, configuration, serialization, and host/device utilities |
| `Geom` | CGNS/HDF5 mesh I/O, Metis/ParMetis partitioning, connectivity, periodic topology, and ghost construction |
| `CFV` | Compact finite-volume geometry, variational reconstruction, limiters, and quadrature support |
| `NativeGeom` | Native mesh/entity storage, adjacency registry, snapshots/edits, CGNS/OpenFOAM/archive I/O, partitioning, refinement, and ghost preparation |
| `NativeCFV` | Native finite-volume, gradient, polynomial sampling, limited/variational reconstruction, and interface sampling plans |
| `Physics` | Shared state input, thermodynamics, transport, RANS closures, and optional chemistry resources |
| `Cauchy` | Native CPU Domain/Simulation framework, explicit/implicit stepping, linear/nonlinear solves, multidomain coupling, and restart/output |
| `Euler` | Compressible Euler/Navier-Stokes, SA-IDDES, and k-omega RANS solvers |
| `EulerP` | Alternative evaluator and optional CUDA/Python execution path |
| `Solver` | Header-only ODE, Krylov, direct-solver, and nonlinear iteration utilities |
| `ACM` | Constant-density artificial-compressibility flow solver |
| `ACMVariable` | Variable-density artificial-compressibility flow solver |
| `NCFV` | Third-order Node Center finite-volume solver with efficient differential and traditional quadrature modes |

The [module interface guide (中文)](docs/architecture/SolverInterfaces_zh.md) traces the two solver paths and describes how a new module can reuse case overrides, field access and distributed residual diagnostics.

### Main capabilities

- Compressible Euler and laminar/RANS Navier-Stokes calculations in 2D and 3D.
- SA-IDDES and k-omega two-equation turbulence models.
- Cantera-coupled multi-species transport, thermodynamics, chemistry source
  terms, and state repair in `EulerEX`.
- Compact variational reconstruction, Roe/HLLE-family numerical fluxes,
  limiters, p-multigrid, and explicit/implicit time integration.
- Constant- and variable-density artificial-compressibility formulations.
- A standalone third-order NCFV solver sharing DNDSR mesh/configuration/I/O
  infrastructure without changing the established Euler/CFV modules.
- MPI owner/ghost communication throughout the distributed mesh pipeline,
  OpenMP where enabled, and optional CUDA support for the applicable EulerP
  path.
- JSON/JSONC runtime configuration, generated JSON schemas, restart files,
  VTKHDF output, and pybind11 modules for core, mesh, CFV, and EulerP access.

### Independent solver executables

Build each solver family and dimension independently. Common mesh, MPI and
numerical infrastructure is shared; a selected target does not link other
solver families or the other dimension's template instantiations.

| Family | 2D executables | 3D executables |
|---|---|---|
| Euler / CFV | `euler`, `euler2D`, `eulerSA`, `euler2EQ`, `eulerEX` | `euler3D`, `eulerSA3D`, `euler2EQ3D`, `eulerEX3D` |
| ACM | `acm2D`, `acmSA2D`, `acm2EQ2D`, `acmVariable2D` | `acm3D`, `acmSA3D`, `acm2EQ3D`, `acmVariable3D`, `ACM` (compatibility name) |
| NCFV Euler | `ncfv_euler2D` | `ncfv_euler3D` |

The original upstream Euler executables remain available. `euler` uses the
upstream NS model (2D geometry, five variables); `euler2D` uses NS_2D (four
variables). Each case's top-level `solver` metadata must match its executable.
NCFV uses `solver.type="ncfv_euler"`, `discretization="NCFV"`,
`model="IdealGas"` and `fieldNVariables=dimension+2`.

Cases and their schemas are grouped under `cases/euler/{2D,3D}`,
`cases/acm/{2D,3D}` and `cases/ncfv_euler/{2D,3D}`. See
[the case guide](cases/README.md) for variant subdirectories and configuration checks.
Cauchy has its own layered configuration under `cases/cauchy/`; `cauchy_solver`
builds `cauchy.exe`, with mesh dimension and physical model selected at runtime.

### Euler compatibility and validation scope

The executable split preserves Euler's evaluator, residual and time-integration
implementation files. Launch/configuration behavior changes: existing adjacent
default JSON files are read instead of overwritten, and solver metadata is
checked against the selected executable. The final merged parameters must be
matched when comparing runs.

The local branch already differed from upstream in the Wilcox production cap,
3D polynomial WBAP limiting, and some shared periodic-mesh handling. Those
differences predate this split. The 2026-10-09 checks passed 29 CPU module tests,
15 Cantera Euler tests, 142 native configuration checks, 11 selection-rejection
checks and nine two-rank short runs. They do not establish identical upstream
field solutions or convergence of every engineering case; some external mesh
fixtures are unavailable locally. See the
[solver migration and upstream comparison guide (中文)](docs/guides/solver_split_zh.md)
for the pinned upstream revision, test scope and reproduction commands.

Supporting tools include `eulerState` for state conversion/inspection and,
when Cantera is enabled, `canteraConstVolTrajectory` for constant-volume
chemistry trajectories.

## Native Mesh and Cauchy

`DNDS::NativeGeom` and `DNDS::NativeCFV` retain the upstream native mesh and
field implementation under separate namespaces. Existing Euler, ACM and NCFV
continue to use their established `Geom`/`CFV` interfaces and case formats.
Cauchy uses the native mesh, entity spaces and shared Runtime/Algebra fields
throughout its numerical path.

The mesh path includes persistent entity IDs, owner/ghost distribution,
connectivity and periodic metadata, retained read scopes, transactional edits,
partition/order changes, refinement, and native/legacy archive import. Cauchy
adds equation-block fields, finite-volume and reconstruction operators, RANS,
SSPRK and implicit time methods, Krylov/SGS/ILU solves, polynomial multigrid,
multidomain interfaces, accepted-state checkpoints, CSV histories and VTKHDF.
Cantera-dependent profiles require a separate Cantera-enabled build.

```bash
# Fresh dependency installs include Cauchy's pinned header-only logger.
# Existing checkouts can add it without replacing other dependencies:
bash scripts/install_spdlog_deps.sh

CC=mpicc CXX=mpicxx cmake -S . -B build \
    -DDNDS_BUILD_TESTS=ON -DDNDS_USE_CANTERA=OFF
cmake --build build --target cauchy_solver native_framework_tests all_pybind11 -j16
cmake --install build --component py

# Run Cauchy from the source root: mesh/output paths are relative to CWD.
build/app/cauchy.exe cases/cauchy/smoke.json --check-config
mpirun -np 4 build/app/cauchy.exe cases/cauchy/smoke.json
# Print the canonical layered-input schema.
build/app/cauchy.exe --emit-schema
```

The smoke case performs two constant-reconstruction SSPRK3 steps on a periodic
mesh. Its output goes to `data/cauchy-smoke/`, outside the maintained case
configuration tree. The other [Cauchy templates](cases/cauchy/README.md) are
numerical experiment inputs; a short smoke run does not qualify their accuracy
or convergence.

The Python API adds `DNDS.Runtime`, `DNDSR.NativeGeom`, `DNDSR.NativeCFV` and
`DNDSR.Cauchy`. Runtime's Python layer supplies tracked array slicing and live
debugging views; `numpy()` returns a snapshot. Mesh read scopes and field leases
must be released before mutation or stepping. For example, after building and
installing the extensions:

```python
import json
from mpi4py import MPI
from DNDSR import DNDS, NativeGeom, NativeCFV, Cauchy

mpi = DNDS.MPIInfo()
mpi.setWorld()
mesh = NativeGeom.with_native_topology(
    NativeGeom.read_native_cgns(mpi, "data/mesh/Uniform_3x3_wall.cgns", 2)
)
with mesh.acquire_read(faces=True) as read:
    local_zones = {int(read.boundary_info(i).zone) for i in range(mesh.NumBnd())}
zones = set().union(*MPI.COMM_WORLD.allgather(local_zones))
config = {"defaults": {
    "initialPrimitive": [1, 0, 0, 0, 1],
    "boundaries": {str(zone): {"kind": "slipWall"} for zone in zones},
}}
domain = Cauchy.Domain(mesh, json.dumps(config))
assert domain.advance(1e-5).accepted
flow_snapshot = domain.fields["flow"].numpy()
```

```bash
# Native CPU/MPI integration (default MPI sizes: 1, 2, 4, 8).
ctest --test-dir build -L native --output-on-failure -j8
# Include the separately built cross-extension test module for Python tests.
PYTHONPATH="$PWD/python:$PWD/build/test/cpp" mpirun -np 2 \
    venv/bin/python -m pytest test/DNDS test/NativeGeom test/Cauchy -q
```

The supported Cauchy frontend in this migration is CPU. After the shared RANS
kernel and test-fixture corrections, CUDA operator qualification passes 8 of 9
tests on the local RTX A5000/CUDA 13.1 build. The remaining `state` test rejects
direct device halo binding with the non-CUDA-aware local MPI. This does not
establish a working Cauchy GPU solver.
Their tests are opt-in with `DNDS_BUILD_CAUCHY_CUDA_TESTS=ON` in a CUDA build,
then target `cauchy_cuda_tests`. Existing Runtime/Algebra and EulerP CUDA paths
remain separate. See the [migration and validation record](docs/dev/native_framework_migration_20261113.txt)
for the precise tested scope and remaining limitations.

## NCFV: Third-Order Node Center Finite Volume

`src/NCFV` is an independent solver module that reuses DNDSR mesh reading,
configuration, MPI-array, boundary-zone, restart, and output facilities. It
does not alter the existing Euler/CFV numerical kernels. Both NCFV algorithms
use the same node-centered median-dual control volume. For every primal cell,
the dual sub-control-volume geometry is built during initialization from:

1. each primal edge midpoint;
2. the arithmetic mean of all vertices on each primal face; and
3. the arithmetic mean of all vertices in the primal cell.

Dual faces and volumes, orientation, reconstruction normalization, sparse
dependency lists, and all geometry-only integration weights are constructed
and stored before time marching. The quadratic zero-mean basis is normalized
with each dual volume's directional half-span before the weighted-SVD system is
formed. Each efficient macro face then stores one deduplicated list of local
`NodeHalo` indices together with the value, state-gradient,
physical-flux-gradient, and viscous-gradient weights that are actually
nonzero. Runtime
assembly addresses those entries directly, whether the support node is owned
or ghosted.

Two runtime-selectable discretization modes are implemented in one module:

| `algorithm.mode` | Integration path |
|---|---|
| `EfficientDifferential` | Uses precomputed value/gradient weights and replaces Hessian contractions by the total differential of first derivatives; it stores no Gauss-point coordinates |
| `TraditionalQuadrature` | Stores conventional face/volume quadrature points: Gauss-Legendre on line entities, Hammer-type rules on triangles, and simplex rules in tetrahedra |

Both modes construct the reconstruction from the complete quadratic,
zero-mean least-squares problem. Efficient mode retains only the pseudoinverse
rows needed for first derivatives after that full system has been solved;
therefore its gradient is not obtained from a reduced linear reconstruction.

Both paths support Euler and laminar viscous terms, per-boundary-zone
configuration, local/global CFL evaluation, SSPRK3 time advancement, initial
field input, restart, and result output. At each efficient residual evaluation,
physical-flux gradients are computed once per local/ghost support node and
then contracted through precomputed geometry weights. This production path is
always active in efficient mode and has no separate compatibility switch.

For smooth flows with unlimited efficient reconstruction, `algorithm.macroFaceBoundary=true`
integrates the convective boundary flux with quadratic-exact moments. Far fields and
named inlets and outlets use the local normal Mach number to select incoming data.
Pressure outlets impose the configured static pressure at subsonic speeds. Walls
pair the interior reconstruction with the normal wall velocity to compute wall pressure.
`Extrapolation` takes the full reconstructed interior trace at every Mach number.
In viscous runs, each original boundary piece supplies its own wall stress and heat flux.
The 3-D exponential manufactured solution adds a volume-averaged source
and supplies incoming characteristic data or a prescribed exterior state at each SSPRK3 stage time. The example
includes a perturbed unit-cube prism mesh and reports the final density and
source errors against degree-six quadrature. [Prism case](cases/ncfv_euler/3D/NCFV_exponential_mms_prism12.json) · [NCFV boundary and source integration](src/NCFV/README.md#非周期三棱柱制造解)

For distributed runs, periodic node pairs are merged into the global topology
before partitioning. Initialization then builds a point-neighbour-complete
geometry cell halo and a dependency-exact sparse `NodeHalo` for reconstruction
and integration. Runtime field exchange is ordinary owner/ghost sparse
communication, including corner and body-diagonal dependencies in domains
periodic in all three directions; NCFV does not replicate the global nodal
field with `MPI_Allgatherv`.

Detailed theory, implementation notes, and validation material:

- [NCFV module guide](src/NCFV/README.md)
- [Mathematical derivation and MPI-design report source](docs/reports/DNDSR_efficient_NCFV_report.tex)
- [NCFV manuscript workspace](docs/reports/NCFV_JCP_manuscript/README.md)

## Artificial-Compressibility Solvers

The two artificial-compressibility implementations are independent of the
compressible Euler equation module while reusing `Geom`, `CFV`, and DNDS MPI
arrays:

- `ACM` advances the constant-density flow state `[u,v,w,p]`, extended with
  conservative turbulence variables in the SA and two-equation executables.
  It includes direct
  Green-Gauss or arbitrary-order variational reconstruction, Roe/Rusanov
  fluxes, laminar viscosity, SSPRK3 pseudo-time marching, implicit backward
  Euler, and laminar/RANS BDF2 dual-time operation with block-Jacobi, LU-SGS, or
  GMRES solution paths. Laminar, SA, Wilcox k-omega, SST, and realizable
  k-epsilon selections are available for steady and BDF2 dual-time calculations.
- `ACMVariable` advances `[rho,rho*u,rho*v,rho*w,p]` with pressure treated as
  the algebraic component of a DAE mass matrix. It supplies explicit and
  implicit pseudo-/physical-time adapters, conservative variable-density
  transport, viscous terms, limiters, and segregated turbulence support.

Constant-density `acmSA2D/3D` stores `[u,v,w,p,rho*nuTilde]`;
`acm2EQ2D/3D` stores `[u,v,w,p,rho*k,rho*omega]` (or `rho*epsilon` for
realizable k-epsilon). Flow and turbulence advance in the same RK stages or
nonlinear iterations. The implicit system includes local turbulence source
blocks and isotropic-pressure coupling, with other cross derivatives and
turbulence spatial off-diagonal terms approximated or frozen. Variable-density
ACM retains segregated turbulence updates.

Both ACM modules enable `turbulenceSettings.includeIsotropicStress=true` by
default for two-equation models. The isotropic stress tensor
`T_k = (2/3) rho*k I` enters the CFV face flux through effective pressure
`p + (2/3) rho*k`, including Riemann dissipation and boundary treatment.
Input and output `Pressure` then represents static pressure. SA and laminar
models add no k stress. Setting the option to `false` retains the legacy
modified-pressure convention.

For a constant-density VTK-HDF restart that stores modified pressure, set
`restartSettings.pressureIsModified=true` together with explicit isotropic
stress; conversion to static pressure requires loaded turbulence fields.
Keep this restart option `false` for new static-pressure files. Initial states
and pressure boundaries must use the selected pressure convention.
See the [ACM state, pressure, and restart guide (中文)](docs/solver-guide/acm_model_state_zh.md)
for model metadata, configuration examples, and coupling limits.

See the [constant-density ACM guide](src/ACM/README.md), the
[variable-density ACM guide](src/ACMVariable/README.md), and the
[incompressible-method comparison](docs/reports/DNDSR_incompressible_methods_zh.md)
for the implemented method matrices and current limitations.

ACM, variable-density ACM, Euler and migrated Physics/Cauchy use the original
SST (1994) closure: the Omega viscosity limiter, `CD_komega >= 1e-20`, and a
20-times production cap for k only. Omega production uses the signed full
Reynolds-stress production divided by `nu_t`, including dilatation, with no
dimensional `1e-8` viscosity floor. The Cauchy source and diffusion kernels have
independent weak-turbulence and dilatation regressions. See the
[gs Suboff comparison and corrections](docs/dev/turbulence_gs_comparison_20261117.txt)
for source provenance, retained discretization differences and validation limits.

## Quick Start

### 1. System dependencies

```bash
# Debian/Ubuntu
sudo apt install build-essential cmake ninja-build openmpi-bin libopenmpi-dev doxygen

# RHEL/Fedora
sudo dnf install gcc-c++ cmake ninja-build openmpi-devel doxygen
```

### 2. Clone the repository

```bash
git clone --recursive https://github.com/Kirabo167/DNDSR-Kirabo.git
cd DNDSR-Kirabo
```

### 3. Create the Python environment and build dependencies

```bash
python3.12 -m venv venv
source venv/bin/activate

# Install Python packages needed to build cfd_externals (Cantera)
pip install -r external/cfd_externals/requirements.txt

# Download, verify, and extract the pinned header-only libraries
# (Eigen, Boost, CGAL, fmt, pybind11, ...)
bash scripts/install_headeronly_deps.sh

# Build binary external libraries (HDF5, CGNS, Metis, ParMetis, Cantera)
cd external/cfd_externals
CC=mpicc CXX=mpicxx python cfd_externals_build.py
cd ../..

# Install remaining Python dependencies (h5py against the project's HDF5, etc.)
bash scripts/install_python_deps.sh
```

> **Why system Python?** Conda Python embeds an RPATH to conda's bundled
> libstdc++, which may be too old for the compiler used to build DNDSR.
> System Python uses the system libstdc++ and avoids this conflict.

### 4. Build C++ solvers

```bash
# Baseline Release CPU build with all unit-test targets available
cmake --preset release-test
cmake --build build --target euler --parallel 4

# Other solver families and all model variants
cmake --build --preset acm --parallel 4
cmake --build --preset ncfv_euler --parallel 4
cmake --build --preset solvers --parallel 4

# Equivalent manual configuration
CC=mpicc CXX=mpicxx cmake -S . -B build \
    -DDNDS_BUILD_TESTS=ON -DDNDS_USE_CANTERA=OFF
cmake --build build --target euler --parallel 4
```

The default build includes the upstream `euler` executable. Other solvers
and diagnostic tools are opt-in targets. Build `all_euler`, `all_acm`,
`all_ncfv_euler`, or `all_solvers` when needed. Limit `--parallel` according
to available memory; splitting executables does not limit concurrent jobs.
The baseline `release-test` preset disables Cantera and CUDA. For reactive flow:

```bash
cmake --preset reactive-test
cmake --build build-reactive --target all_solvers --parallel 4
cmake --build --preset reactive --parallel 4
ctest --preset reactive
# Or rerun only the focused chemistry tests
ctest --preset reactive-focused
```

For the CUDA/EulerP configuration:

```bash
cmake --preset cuda
cmake --build build-cuda --target all_solvers --parallel 4
cmake --build --preset cuda --parallel 4
ctest --preset cuda
```

The CUDA preset intentionally keeps reactive CFD CPU-only and disables
Cantera. A complete option and dependency reference is available in the
[building guide](docs/guides/building.md).

### 5. Run a solver

Solver paths in case files are interpreted relative to the process working
directory. The maintained examples therefore normally run from `build/`:

```bash
# Compressible Euler, serial and MPI
(cd build && ./app/euler.exe ../cases/euler/2D/euler_config_IV.json)
(cd build && mpirun -np 4 ./app/euler.exe ../cases/euler/2D/euler_config_IV.json)

# NCFV efficient differential mode on the included 3-D periodic mesh
(cd build && ./app/ncfv_euler3D.exe ../cases/ncfv_euler/3D/NCFV_periodic_hex_iv10.json)
(cd build && mpirun -np 4 ./app/ncfv_euler3D.exe \
    ../cases/ncfv_euler/3D/NCFV_periodic_hex_iv10.json)

# Use the same NCFV case with traditional quadrature
(cd build && ./app/ncfv_euler3D.exe \
    ../cases/ncfv_euler/3D/NCFV_periodic_hex_iv10.json \
    -k /algorithm/mode -v TraditionalQuadrature)

# Artificial-compressibility examples
(cd build && ./app/acm2D.exe ../cases/acm/2D/acm2D.json)
(cd build && ./app/acmVariable2D.exe \
    ../cases/acm/2D/variable_density/acmVariable2D.json)
```

The smaller generated periodic NCFV meshes are stored in this repository.
The largest Tet4 fixture is generated locally; see [large mesh generation](cases/README.md#large-mesh-generation).
Several Euler, ACM, and legacy NCFV cases require the separately versioned
`cfd_meshes`/`data/mesh` fixtures described in the migration guide.

### 6. Configure a solver

Input parameters are defined in JSONC config files. First choose the executable
from the table above; that executable fixes the solver family, geometry
dimension and equation/model specialization.

Maintained cases include a `solver` metadata object so the program can check
that the case matches the chosen executable. It does not select or switch to
another solver. The following metadata fragment belongs to `eulerSA3D.exe`:

```json
{
    "solver": {
        "type": "Euler",
        "discretization": "CFV",
        "model": "NS_SA_3D",
        "fieldNVariables": 6
    }
}
```

Running this case with `euler3D.exe` fails the model-compatibility check;
use `eulerSA3D.exe` for `NS_SA_3D`. Changing the JSON metadata alone cannot
turn an executable into a different solver. The metadata can appear anywhere
in the top-level JSON object; it need not be the first property.

For fixed-size models, `fieldNVariables` records the compiled state size and
must match it. For `NS_EX` and `NS_EX_3D`, it supplies the dynamic state size
used by the extended/multi-species model. Run the chosen executable with
`--check-config` to validate the case before mesh loading.

Start with
[the commented Euler defaults](cases/euler/defaults/euler_default_config_commented.json) and
the schema stored beside the selected solver's cases;
EulerEX reactive and model-specific RANS fields are documented by their own
schemas. After changing configuration types, regenerate them from the
Cantera-enabled `schemas` build preset and validate them with
`cases/validate_configs.py` (see `CONTRIBUTING.md`).

For NCFV, select the integration algorithm in the case file:

```json
{
    "algorithm": {
        "mode": "EfficientDifferential"
    }
}
```

Set the value to `TraditionalQuadrature` for the conventional third-order
integration path. NCFV also supports JSON-pointer command-line overrides and
can emit its current schema directly:

```bash
(cd build && ./app/ncfv_euler2D.exe --emit-schema)
(cd build && ./app/ncfv_euler2D.exe ../cases/ncfv_euler/2D/ncfv_euler2D.json \
    -k /algorithm/mode -v TraditionalQuadrature)
```

Validate native configuration parameters without loading a mesh:

```bash
(cd build && ./app/euler2D.exe ../cases/euler/2D/euler2D_config.json --check-config)
python3 cases/validate_configs.py --quiet
python3 scripts/check_solver_cases.py --build-dir build \
    --reactive-build-dir build-reactive
```

Build the required executables first. The audit reports reactive cases as
requiring another build when Cantera support is unavailable. Neither schema
nor parameter checks prove that external meshes are present or a simulation
has converged. Euler reads a case-adjacent `<target>_default_config.json` when
available, otherwise starts from compiled defaults; ACM uses a complete case,
and NCFV merges its typed defaults. See the
[configuration guide](docs/solver-guide/solver_config.md).

Reactive runs may also use `DNDS_MECH_PATH` and `CANTERA_DATA` to locate
Cantera mechanism/data files.

### 7. Install the Python package

```bash
CC=mpicc CXX=mpicxx CMAKE_BUILD_PARALLEL_LEVEL=32 \
    pip install -e . --no-build-isolation
```

The persistent scikit-build directory is separated by Python wheel tag, so
different interpreter ABIs cannot reuse one CMake cache. After a C++ change,
rerun the editable-install command above, or use the already configured
in-place CMake build:

```bash
cmake --build --preset python -j32
cmake --install build --component py
```

### 8. Run tests

Some Geom/CFV/Euler regressions require the separately versioned
`cfd_meshes` fixtures. Fetch them first using the pinned instructions in the
[migration guide](docs/guides/v0.3.1_new_features_zh.md#72-测试网格).

```bash
# Build solver entry points required by configuration CTests, then unit tests
cmake --build --preset solvers --parallel 4
cmake --build --preset tests --parallel 4
ctest --preset unit
# Configuration/model/dimension regressions only
ctest --test-dir build -L solver_config --output-on-failure

# One module only (examples)
cmake --build build -t ncfv_unit_tests -j32
ctest --test-dir build -R '^ncfv_' --output-on-failure
cmake --build build -t geom_unit_tests -j32
ctest --test-dir build -R '^geom_' --output-on-failure

# Python tests (always rebuild and install all bindings first)
cmake --build --preset python -j32
cmake --install build --component py
PYTHONPATH="$PWD/python:$PWD/build/test/cpp" venv/bin/python -m pytest test/
```

Available C++ category targets are `dnds_unit_tests`, `geom_unit_tests`,
`cfv_unit_tests`, `euler_unit_tests`, `acm_unit_tests`,
`acm_variable_unit_tests`, `solver_unit_tests`, `ncfv_unit_tests`,
`native_geom_unit_tests`, `native_cfv_unit_tests`, and `cauchy_unit_tests`.
`native_framework_tests` also builds native I/O and Python interop fixtures. MPI
tests are registered for 1, 2, 4, and 8 ranks by default; set
`DNDS_TEST_NP_LIST` when configuring to choose another list. Use
`scripts/ctest_summary.py` when an aggregated doctest assertion summary is
preferred.

The native Python extensions must be rebuilt **and installed** after every
relevant C++ source change. Running pytest against stale files in
`python/DNDSR/` can otherwise look like a numerical or memory defect.

### 9. Build and serve documentation locally

```bash
pip install -r docs/sphinx/requirements.txt
cmake --preset release-test
cmake --build build -t serve-docs
# Then: cd build/docs/sphinx && python3 -m http.server 8000
```

## Repository Layout

```text
DNDSR/
├── app/                 # Thin executable entry points
├── src/
│   ├── DNDS/            # Core MPI, arrays, config, serialization, device utilities
│   ├── Geom/            # Mesh I/O, topology, partitioning, and ghost layers
│   ├── CFV/             # Compact finite-volume reconstruction and quadrature
│   ├── NativeGeom/      # Runtime mesh/entity storage and mesh preparation
│   ├── NativeCFV/       # Native finite-volume/reconstruction plans
│   ├── Physics/         # Shared material and closure resources
│   ├── Cauchy/          # Native CPU Domain/Simulation framework
│   ├── Euler/            # Compressible and reactive-flow solver kernels
│   ├── EulerP/           # Alternative/CUDA-capable evaluator
│   ├── Solver/           # Generic algebraic and time-integration utilities
│   ├── ACM/              # Constant-density artificial compressibility
│   ├── ACMVariable/      # Variable-density artificial compressibility
│   └── NCFV/             # Node Center Finite Volume Method
├── cases/               # euler/, acm/, ncfv_euler/ (2D/3D), plus cauchy/
├── test/                # C++ doctest and Python pytest suites
├── python/DNDSR/        # Python package and installed native extensions
├── docs/                # Guides, architecture notes, theory, and research reports
├── external/            # cfd_externals submodule and third-party sources
└── scripts/             # Dependency, validation, formatting, and documentation tools
```

## Documentation

The upstream documentation is hosted at
**[cfdlab-thu.github.io/DNDSR](https://cfdlab-thu.github.io/DNDSR/)**.
Fork-specific material is maintained in the local [`docs/`](docs) tree.

### Guides

- [Build and dependency guide](docs/guides/building.md)
- [Solver split, configuration migration and upstream comparison (中文)](docs/guides/solver_split_zh.md)
- [Project structure and target map](docs/guides/project_structure.md)
- [Python Geom API guide](docs/guides/python_geom_guide.md)
- [Geometry and CFV usage](docs/guides/geom_usage.md)
- [Array usage](docs/guides/array_usage.md)
- [Serialization usage](docs/guides/serialization_usage.md)
- [Code style](docs/guides/style_guide.md)
- [Documentation authoring](docs/guides/doc_authoring.md)

### Architecture

- [Mesh connectivity and adjacency-state pipeline](docs/architecture/MeshConnectivity.md)
- [Mesh DAG design](docs/architecture/MeshDAGDesign.md)
- [Array infrastructure](docs/architecture/array_infrastructure.md)
- [Serialization](docs/architecture/Serialization.md)
- [Programming paradigm](docs/architecture/Paradigm.md)

### Solver and verification material

- [NCFV implementation guide](src/NCFV/README.md)
- [First-reconstruction accuracy report](docs/reports/ncfv_first_reconstruction/report.md)
- [Gradient implementation audit](docs/reports/ncfv_gradient_audit/report.md)
- [Quadratic reconstruction verification](docs/reports/ncfv_quadratic_reconstruction_20260909/report.md)
- [Efficient NCFV `t=2`, common-CFL convergence report](docs/reports/ncfv_t2_cfl05_thesis_20260907/report.md)
- [Constant-density ACM implementation guide](src/ACM/README.md)
- [Variable-density ACM implementation guide](src/ACMVariable/README.md)
- [Incompressible-method analysis](docs/reports/DNDSR_incompressible_methods_zh.md)

The hosted upstream site additionally provides the Sphinx theory pages,
Python autodoc, Breathe API pages, and full
[Doxygen C++ reference](https://cfdlab-thu.github.io/DNDSR/doxygen/).

## Current Scope and Boundaries

- Linux/POSIX with MPI is the primary supported development environment;
  GCC 9+ or Clang 8+, CMake 3.21+, and C++17 are required.
- Python 3.10+ is supported by package metadata; Python 3.12 is the current CI
  and development baseline.
- Cantera/reactive-flow targets are CPU-only in the maintained presets. CUDA
  applies to the Runtime/Algebra and EulerP/device paths. Cauchy CPU is the
  supported frontend here; its CUDA operator qualification remains incomplete.
  NCFV and ACM have no CUDA solver path.
- NCFV currently accepts O1 primal meshes. O2 curved elements, rotational
  periodicity, NCFV turbulence models, and implicit NCFV time integration are
  not implemented.
- NCFV geometry, polynomial-exactness, free-stream, halo, and I/O tests are
  regression evidence; they do not replace a problem-specific mesh-refinement
  study for nonlinear viscous-flow accuracy.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the CPU/Cantera/CUDA validation
matrix, Python native-module rebuild rule, configuration-schema workflow, and
pull-request checklist.

## Type Stubs

Type stubs (`.pyi`) are generated automatically during `cmake --install`
(and therefore during `pip install`). They are placed in `python/DNDSR/`
for PEP 561 compliance and do not need to be committed.

## VS Code Setup

For Pylance / clangd to work correctly, the project ships `.vscode/`
configs. Set the Python interpreter to `venv/bin/python` via the
VS Code Python extension.

## License

DNDSR is licensed under [GPL-3.0-only](LICENSE). Local Python wheels also
contain separately licensed binary dependencies; see
[THIRD_PARTY_DEPENDENCIES.md](THIRD_PARTY_DEPENDENCIES.md) for their notices
and the ParMETIS restriction that currently prevents public redistribution.
