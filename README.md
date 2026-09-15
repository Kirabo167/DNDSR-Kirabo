# DNDSR

DNDSR is a C++17/Python research framework for unstructured-grid
computational fluid dynamics. It combines compact finite-volume (CFV)
discretizations, distributed-memory MPI parallelism, optional CUDA execution,
and optional Cantera-based multi-species reacting-flow models.

This repository is the Kirabo fork of the
[upstream DNDSR project](https://github.com/harryzhou2000/DNDSR). It tracks the
upstream v0.3.1 code line while maintaining three additional solver families:
the constant-density ACM solver, the variable-density ACM solver, and the
third-order **Node Center Finite Volume Method (NCFV)** solver. Project target,
namespace, and Python-package names remain `DNDSR`.

See the [v0.3.1 feature and migration guide](docs/guides/v0.3.1_new_features_zh.md)
for compatibility changes and the validation matrix. The upstream manuals and
API reference remain available at
[cfdlab-thu.github.io/DNDSR](https://cfdlab-thu.github.io/DNDSR/).

## Project Overview

| Module | Responsibility |
|---|---|
| `DNDS` | MPI arrays and mappings, configuration, profiling, serialization, HDF5, and host/device utilities |
| `Geom` | CGNS/HDF5 mesh I/O, Metis/ParMetis partitioning, connectivity, periodic topology, and ghost construction |
| `CFV` | Compact finite-volume geometry, variational reconstruction, limiters, and quadrature support |
| `Euler` | Compressible Euler/Navier-Stokes, SA-IDDES, and k-omega RANS solvers |
| `EulerP` | Alternative evaluator and optional CUDA/Python execution path |
| `Solver` | Header-only ODE, Krylov, direct-solver, and nonlinear iteration utilities |
| `ACM` | Constant-density artificial-compressibility flow solver |
| `ACMVariable` | Variable-density artificial-compressibility flow solver |
| `NCFV` | Third-order Node Center finite-volume solver with efficient differential and traditional quadrature modes |

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

### Solver executables

| CMake target / executable | Model |
|---|---|
| `euler`, `euler3D` | Compressible Navier-Stokes |
| `euler2D` | Two-velocity-component Navier-Stokes on a 2D mesh |
| `eulerSA`, `eulerSA3D` | Spalart-Allmaras RANS/IDDES |
| `euler2EQ`, `euler2EQ3D` | k-omega two-equation RANS |
| `eulerEX`, `eulerEX3D` | Cantera-based multi-species reactive Navier-Stokes |
| `ACM`, `acm2D`, `acm3D` | Constant-density artificial-compressibility flow |
| `acmVariable2D`, `acmVariable3D` | Variable-density artificial-compressibility flow |
| `NCFV` | Third-order Node Center Finite Volume Method |

Supporting tools include `eulerState` for state conversion/inspection and,
when Cantera is enabled, `canteraConstVolTrajectory` for constant-volume
chemistry trajectories.

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

- `ACM` advances the constant-density state `[u,v,w,p]`. It includes direct
  Green-Gauss or arbitrary-order variational reconstruction, Roe/Rusanov
  fluxes, laminar viscosity, SSPRK3 pseudo-time marching, implicit backward
  Euler, and laminar BDF2 dual-time operation with block-Jacobi, LU-SGS, or
  GMRES solution paths. Laminar, SA, Wilcox k-omega, SST, and realizable
  k-epsilon selections are available for steady calculations.
- `ACMVariable` advances `[rho,rho*u,rho*v,rho*w,p]` with pressure treated as
  the algebraic component of a DAE mass matrix. It supplies explicit and
  implicit pseudo-/physical-time adapters, conservative variable-density
  transport, viscous terms, limiters, and segregated turbulence support.

See the [constant-density ACM guide](src/ACM/README.md), the
[variable-density ACM guide](src/ACMVariable/README.md), and the
[incompressible-method comparison](docs/reports/DNDSR_incompressible_methods_zh.md)
for the implemented method matrices and current limitations.

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
cmake --build build -t euler NCFV ACM acmVariable3D -j32

# Equivalent manual configuration
CC=mpicc CXX=mpicxx cmake -S . -B build \
    -DDNDS_BUILD_TESTS=ON -DDNDS_USE_CANTERA=OFF
cmake --build build -t euler NCFV -j32
```

Most applications are excluded from the default `all` target, so build the
required solver target explicitly. The baseline `release-test` preset disables
Cantera and CUDA. To build and test the reactive-flow targets with Cantera:

```bash
cmake --preset reactive-test
cmake --build --preset reactive -j32
ctest --preset reactive
# Or rerun only the focused chemistry tests
ctest --preset reactive-focused
```

For the CUDA/EulerP configuration:

```bash
cmake --preset cuda
cmake --build --preset cuda -j32
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
(cd build && ./app/euler.exe ../cases/euler/euler_config_IV.json)
(cd build && mpirun -np 4 ./app/euler.exe ../cases/euler/euler_config_IV.json)

# NCFV efficient differential mode on the included 3-D periodic mesh
(cd build && ./app/NCFV.exe ../cases/NCFV/NCFV_periodic_hex_iv10.json)
(cd build && mpirun -np 4 ./app/NCFV.exe \
    ../cases/NCFV/NCFV_periodic_hex_iv10.json)

# Use the same NCFV case with traditional quadrature
(cd build && ./app/NCFV.exe \
    ../cases/NCFV/NCFV_periodic_hex_iv10.json \
    -k /algorithm/mode -v TraditionalQuadrature)

# Artificial-compressibility examples
(cd build && ./app/acm2D.exe ../cases/acm2D/acm2D.json)
(cd build && ./app/acmVariable2D.exe \
    ../cases/acmVariable2D/acmVariable2D.json)
```

The generated periodic NCFV meshes are stored in this repository. Several
Euler, ACM, and legacy NCFV cases instead require the separately versioned
`cfd_meshes`/`data/mesh` fixtures described in the migration guide.

### 6. Configure a solver

Input parameters are defined in JSONC config files. Start with
[the commented Euler defaults](cases/euler_default_config_commented.json) and
the generated `cases/*_schema.json` file for the selected solver variant;
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
(cd build && ./app/NCFV.exe --emit-schema)
(cd build && ./app/NCFV.exe ../cases/NCFV/NCFV.json \
    -k /algorithm/mode -v TraditionalQuadrature)
```

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
# All C++ unit tests (doctest, via CTest)
cmake --build --preset tests -j32
ctest --preset unit

# One module only (examples)
cmake --build build -t ncfv_unit_tests -j32
ctest --test-dir build -R '^ncfv_' --output-on-failure
cmake --build build -t geom_unit_tests -j32
ctest --test-dir build -R '^geom_' --output-on-failure

# Python tests (always rebuild and install all bindings first)
cmake --build --preset python -j32
cmake --install build --component py
PYTHONPATH="$PWD/python" venv/bin/python -m pytest test/
```

Available C++ category targets are `dnds_unit_tests`, `geom_unit_tests`,
`cfv_unit_tests`, `euler_unit_tests`, `acm_unit_tests`,
`acm_variable_unit_tests`, `solver_unit_tests`, and `ncfv_unit_tests`. MPI
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
│   ├── Euler/            # Compressible and reactive-flow solver kernels
│   ├── EulerP/           # Alternative/CUDA-capable evaluator
│   ├── Solver/           # Generic algebraic and time-integration utilities
│   ├── ACM/              # Constant-density artificial compressibility
│   ├── ACMVariable/      # Variable-density artificial compressibility
│   └── NCFV/             # Node Center Finite Volume Method
├── cases/               # JSON/JSONC cases, schemas, and NCFV validation cases
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
  currently applies to the EulerP/device path, not to NCFV or ACM.
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
