# DNDSR

DNDSR is a C++17 / Python CFD research code implementing Compact Finite Volume
methods with MPI parallelism, optional CUDA GPU support, and optional
Cantera-based multi-species reactive flow.

This fork preserves the ACM, ACMVariable, and NCFV modules while integrating
the official v0.3.1 line. See the
[v0.3.1 feature and migration guide](docs/guides/v0.3.1_new_features_zh.md)
for the compatibility adaptations and validation matrix.

**Upstream documentation**: [cfdlab-thu.github.io/DNDSR](https://cfdlab-thu.github.io/DNDSR/)
(includes guides, architecture, API reference, and embedded
[Doxygen C++ docs](https://cfdlab-thu.github.io/DNDSR/doxygen/))

## Features

- **Solvers**: Euler / Navier-Stokes (2D/3D), SA-IDDES, k-omega RANS
- **Reactive flow**: EulerEX/EulerEX3D multi-species transport, Cantera
  thermodynamics and chemistry, source coupling, and state repair
- **Additional modules**: constant/variable-density ACM and third-order NCFV
- **Numerics**: Compact Finite Volume with variational reconstruction,
  Roe / HLLE+ Riemann solvers, ESDIRK / HM3 time integration, p-Multigrid
- **Parallelism**: MPI with ghost communication, optional CUDA GPU support
- **Python bindings**: pybind11 modules for mesh, CFV, and solver access
- **Config system**: typed JSON configs with schema validation

Solver executables:

| Executable | Model |
|------------|-------|
| `euler` / `euler3D` | Navier-Stokes |
| `euler2D` | Two-velocity-component Navier-Stokes on a 2D mesh |
| `eulerSA` / `eulerSA3D` | Spalart-Allmaras RANS (IDDES) |
| `euler2EQ` / `euler2EQ3D` | k-omega two-equation RANS |
| `eulerEX` / `eulerEX3D` | Multi-species reactive Navier-Stokes (Cantera) |
| `ACM` / `acm2D` / `acm3D` | Constant-density artificial-compressibility solver |
| `acmVariable2D` / `acmVariable3D` | Variable-density artificial-compressibility solver |
| `NCFV` | Third-order node-centred finite-volume solver |

Supporting tools include `eulerState` for state conversion/inspection and
`canteraConstVolTrajectory` for constant-volume chemistry trajectories.

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
cd DNDSR
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
# Using CMake presets (recommended)
cmake --preset release-test
cmake --build build -t euler -j32

# Or manually
CC=mpicc CXX=mpicxx cmake -S . -B build \
    -DDNDS_BUILD_TESTS=ON -DDNDS_USE_CANTERA=OFF
cmake --build build -t euler -j32
```

The baseline `release-test` preset explicitly disables Cantera and CUDA. To
build the v0.3.1 reactive-flow targets and validate every C++ module with
Cantera enabled:

```bash
cmake --preset reactive-test
cmake --build --preset reactive -j32
ctest --preset reactive
# Or rerun only the eight focused chemistry tests:
ctest --preset reactive-focused
```

### 5. Run a solver

```bash
# Serial (the subshell keeps the caller at the repository root)
(cd build && ./app/euler.exe ../cases/euler/euler_config_IV.json)

# Parallel
(cd build && mpirun -np 4 ./app/euler.exe ../cases/euler/euler_config_IV.json)
```

The maintained case files resolve mesh and output paths relative to the
`build/` working directory. The IV example also requires the pinned external
test meshes described in the migration guide.

Input parameters are defined in JSONC config files. Start with
[the commented Euler defaults](cases/euler_default_config_commented.json) and
the generated `cases/*_schema.json` file for the selected solver variant;
EulerEX reactive and model-specific RANS fields are documented by their own
schemas. After changing configuration types, regenerate them from the
Cantera-enabled `schemas` build preset and validate them with
`cases/validate_configs.py` (see `CONTRIBUTING.md`).

Reactive runs may also use `DNDS_MECH_PATH` and `CANTERA_DATA` to locate
Cantera mechanism/data files.

### 6. Install the Python package

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

### 7. Run tests

Some Geom/CFV/Euler regressions require the separately versioned
`cfd_meshes` fixtures. Fetch them first using the pinned instructions in the
[migration guide](docs/guides/v0.3.1_new_features_zh.md#72-测试网格).

```bash
# All C++ unit tests (doctest, via CTest)
cmake --build --preset tests -j32
ctest --preset unit

# Python tests (always rebuild and install all bindings first)
cmake --build --preset python -j32
cmake --install build --component py
PYTHONPATH="$PWD/python" venv/bin/python -m pytest test/
```

### 8. Build and serve documentation locally

```bash
pip install -r docs/sphinx/requirements.txt
cmake --preset release-test
cmake --build build -t serve-docs
# Then: cd build/docs/sphinx && python3 -m http.server 8000
```

## Documentation

The upstream documentation is hosted at
**[cfdlab-thu.github.io/DNDSR](https://cfdlab-thu.github.io/DNDSR/)**.
Fork-specific documentation is maintained in this repository's
[`docs/` tree](https://github.com/Kirabo167/DNDSR/tree/main/docs). Together
they include:

- **Guides**: [Building](https://cfdlab-thu.github.io/DNDSR/guides/building.html),
  [Style Guide](https://cfdlab-thu.github.io/DNDSR/guides/style_guide.html),
  [Array Usage](https://cfdlab-thu.github.io/DNDSR/guides/array_usage.html),
  [Geometry + CFV](https://cfdlab-thu.github.io/DNDSR/guides/geom_usage.html),
  [Doc Authoring](https://cfdlab-thu.github.io/DNDSR/guides/doc_authoring.html)
- **Architecture**: Array infrastructure, Serialization, Paradigm
- **Theory**: Variational Reconstruction, Shape Functions
- **C++ API**: via [Breathe](https://cfdlab-thu.github.io/DNDSR/sphinx/api_cpp.html)
  and [Doxygen](https://cfdlab-thu.github.io/DNDSR/doxygen/) (with class diagrams)
- **Python API**: [autodoc](https://cfdlab-thu.github.io/DNDSR/sphinx/api_python.html)
- **Unit Tests**: suite overview plus detailed pages for DNDS, Geom, CFV,
  Euler, and Solver; ACM, ACMVariable, NCFV, and EulerP coverage is summarized
  in the overview

For local-only references:
- [Project Structure](docs/guides/project_structure.md)
- [Building](docs/guides/building.md)
- [v0.3.1 Features and Migration](docs/guides/v0.3.1_new_features_zh.md)
- [Doc Authoring Guide](docs/guides/doc_authoring.md)

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
