# Project Structure {#project_structure}

## Directory Layout

```
DNDSR/
├── src/                        C++ source code
│   ├── DNDS/                   Core: MPI arrays, serialization, profiling, CUDA
│   │   ├── Config/             Runtime configuration enums, parameters, registry
│   │   ├── Device/             Host-device memory transfer, CUDA utilities
│   │   ├── Serializer/         JSON and HDF5 serialization framework
│   │   └── ArrayDerived/       Specialized array types (adjacency, Eigen matrices)
│   ├── Geom/                   Unstructured mesh, CGNS I/O, partitioning
│   │   └── Mesh/               Mesh data structures, connectivity, ghost management
│   ├── CFV/                    Compact Finite Volume, variational reconstruction
│   ├── Euler/                  Compressible N-S, RANS, and reactive EulerEX
│   ├── EulerP/                 Alternative evaluator with CUDA GPU support
│   ├── Solver/                 ODE integrators, Krylov solvers (GMRES, PCG)
│   ├── ACM/                    Constant-density artificial compressibility
│   ├── ACMVariable/            Variable-density artificial compressibility
│   └── NCFV/                   Third-order node-centred finite volume
│
├── python/DNDSR/               Python package (pip-installable)
│   ├── __init__.py             Top-level: imports DNDS, Geom, CFV, EulerP
│   ├── _loader.py              Shared native library preloader (ctypes.CDLL)
│   ├── py.typed                PEP 561 typed package marker
│   ├── DNDS/                   Python bindings + factory functions for core
│   │   ├── __init__.py         Preload, import bindings, MPI init, factories
│   │   ├── _ext/               Compiled pybind11 .so (cmake --install target)
│   │   ├── Debug_Py.py         Pure-Python debug helpers
│   │   └── Wrapper.py          Python introspection utilities
│   ├── Geom/                   Mesh bindings
│   │   ├── __init__.py
│   │   ├── _ext/
│   │   └── utils.py            Mesh creation utilities
│   ├── CFV/                    FV bindings
│   │   ├── __init__.py
│   │   └── _ext/
│   └── EulerP/                 Euler evaluator bindings
│       ├── __init__.py
│       ├── _ext/
│       └── EulerP_Solver.py    High-level solver wrapper
│
├── app/                        C++ application entry points
│   ├── Euler/                  Euler/RANS/reactive solvers and state tools
│   ├── ACM/                    Constant-density ACM applications
│   ├── ACMVariable/            Variable-density ACM applications
│   ├── NCFV/                   ncfv_euler2D / ncfv_euler3D entry points
│   ├── DNDS/                   Old standalone test apps
│   ├── Geom/                   Mesh tool apps
│   └── CFV/                    FV test apps
│
├── test/                       Tests
│   ├── cpp/                    C++ unit tests (doctest, registered with CTest)
│   │   ├── DNDS/               DNDS core tests
│   │   ├── Geom/               Geom mesh tests
│   │   ├── CFV/                CFV reconstruction tests
│   │   ├── Euler/              Euler, RANS, and reactive evaluator tests
│   │   ├── Solver/             Solver (ODE, linear, direct, scalar) tests
│   │   ├── ACM/                Constant-density ACM tests
│   │   ├── ACMVariable/        Variable-density ACM tests
│   │   └── NCFV/               NCFV geometry, MPI, and I/O tests
│   ├── DNDS/                   Python tests (pytest)
│   ├── Geom/                   Python Geom tests
│   ├── CFV/                    Python CFV tests
│   ├── Euler/                  Python Euler restart/redistribution tests
│   └── EulerP/                 Python EulerP binding/solver tests
│
├── external/                   Third-party dependencies
│   ├── cfd_externals/          C libraries submodule (HDF5, CGNS, Metis, ...)
│   ├── doctest/                C++ test framework (header-only)
│   ├── eigen/                  Eigen linear algebra (header-only)
│   ├── fmt/                    fmt formatting library
│   ├── pybind11/               Python binding framework
│   ├── nlohmann/               JSON for Modern C++ (header-only)
│   └── ...                     boost, CGAL, nanoflann, exprtk, etc.
│
├── docs/                       Documentation sources
│   ├── doxygen/                Doxygen configuration and main page
│   ├── sphinx/                 Sphinx configuration, toctree stubs, extensions
│   ├── architecture/           Architecture design documents
│   ├── guides/                 Developer and usage guides
│   ├── theory/                 Mathematical background
│   ├── tests/                  Test suite documentation
│   ├── dev/                    Development notes and design proposals
│   └── index.md                Documentation root page
│
├── cases/                      JSON/JSONC cases and per-model schemas
│   ├── euler/                  2D/, 3D/; SA/, 2EQ/, EX/ variants; defaults/
│   ├── acm/                    2D/, 3D/; variable_density/ variants
│   └── ncfv_euler/             2D/, 3D/; meshes and diagnostics in 3D/
├── scripts/                    Utility scripts
│   └── generate-stubs.sh       Type stub generator (pybind11-stubgen)
├── stubs/                      Generated .pyi stubs (intermediate output)
│
├── CMakeLists.txt              Root CMake build file
├── CMakePresets.json           CPU, reactive, CUDA, Python, and CI presets
├── cmakeCommonUtils.cmake       Shared CMake helper functions
├── pyproject.toml              Python package build configuration
└── AGENTS.md                   Agentic coding guide
```

## C++ Source Organization (`src/`)

Each module under `src/` contains:

- `*.hpp` / `*.cpp` — C++ headers and implementation files.
- `*_pybind.cpp` / `*_bind.cpp` — Pybind11 binding definitions.
- `_explicit_instantiation/` — Explicit template instantiation files
  (one per model variant, compiled as separate translation units).
- `CMakeLists.txt` — Module-level build rules.

### Module dependency graph

Each module depends on those above it:

```
DNDS
├── Solver (header-only)
└── Geom
    └── CFV
        ├── Euler / EulerEX (also uses Solver; Cantera optional)
        ├── EulerP (CUDA optional)
        ├── ACM
        ├── ACMVariable
        └── NCFV
```

## Python Package Organization (`python/DNDSR/`)

The Python package is separate from C++ sources.  Compiled pybind11
extension modules (`.so` / `.pyd`) are installed into `_ext/`
subdirectories by `cmake --install`.

### Import chain

```python
from DNDSR import DNDS
  # → python/DNDSR/__init__.py
  #     → from . import DNDS
  #       → python/DNDSR/DNDS/__init__.py
  #         → _loader.preload("dnds")           # ctypes.CDLL for native libs
  #         → from ._ext.dnds_pybind11 import *  # load C++ bindings
  #         → _init_mpi()                        # MPI_Init_thread at import time
```

### Native Library Preloader (`_loader.py`)

Before pybind11 extensions can be imported, their shared library
dependencies must be loaded.  `_loader.py` provides a single
`preload(module)` function that:

1. Locates the library directory (`python/DNDSR/_lib/` or legacy paths).
2. Loads bundled external dependencies (zlib, HDF5, CGNS, Metis and
   ParMetis) via `ctypes.CDLL` with `RTLD_GLOBAL`. MPI and the C++ standard
   library remain system-provided and are not bundled.
3. Loads module-specific shared libraries (libdnds_shared, libgeom_shared,
   etc.).

## Test Organization

| Directory        | Framework | Runner            | What it tests          |
|------------------|-----------|-------------------|------------------------|
| `test/cpp/DNDS/` | doctest   | CTest (np=1,2,4,8) | DNDS core C++ classes  |
| `test/cpp/Geom/` | doctest   | CTest (np=1,2,4,8) | Geom mesh C++ classes  |
| `test/cpp/CFV/`  | doctest   | CTest (np=1,2,4,8) | CFV reconstruction     |
| `test/cpp/Euler/`| doctest   | CTest (np=1,2,4,8) | Euler evaluator        |
| `test/cpp/Solver/`| doctest  | CTest (serial)      | ODE, linear, direct, scalar |
| `test/cpp/ACM/`  | doctest   | CTest (serial/MPI) | Constant-density ACM   |
| `test/cpp/ACMVariable/` | doctest | CTest (serial/MPI) | Variable-density ACM |
| `test/cpp/NCFV/` | doctest   | CTest (serial/MPI) | NCFV geometry and I/O  |
| `test/DNDS/`     | pytest    | pytest             | DNDS Python bindings   |
| `test/Geom/`     | pytest    | pytest             | Geom Python bindings   |
| `test/CFV/`      | pytest    | pytest             | CFV Python bindings    |
| `test/Euler/`    | pytest    | pytest             | Euler restart redistribution |
| `test/EulerP/`   | pytest    | pytest             | EulerP Python bindings |

See @ref test_overview for the full C++ test suite overview. Detailed pages are
currently provided for DNDS, Geom, CFV, Euler, and Solver; the overview covers
ACM, ACMVariable, NCFV, and EulerP.

## CMake Build Targets

| Target              | Description                                 |
|---------------------|---------------------------------------------|
| `euler`             | Euler N-S solver (2D)                       |
| `euler2D`           | Two-component Euler N-S solver (2D)         |
| `euler3D`           | Euler N-S solver (3D)                       |
| `eulerSA`           | Spalart-Allmaras RANS solver (2D)           |
| `eulerSA3D`         | Spalart-Allmaras RANS solver (3D)           |
| `euler2EQ`          | k-omega RANS solver (2D)                    |
| `euler2EQ3D`        | k-omega RANS solver (3D)                    |
| `eulerEX` / `eulerEX3D` | Cantera-based reactive N-S solvers       |
| `eulerState`        | Euler state conversion and inspection tool  |
| `canteraConstVolTrajectory` | Constant-volume chemistry trajectory tool (Cantera only) |
| `ACM` / `acm2D` / `acm3D` | Constant-density ACM applications       |
| `acmVariable2D` / `acmVariable3D` | Variable-density ACM applications |
| `ncfv_euler2D` / `ncfv_euler3D` | Independent 2D/3D node-centred Euler solvers |
| `all_euler` | All retained Euler models and associated state/chemistry tools |
| `all_acm` | Constant-/variable-density ACM variants and the ACM compatibility target |
| `all_ncfv_euler` | Both NCFV Euler dimensions |
| `all_solvers` | All three solver families |
| `dnds_pybind11`     | DNDS Python binding module                  |
| `geom_pybind11`     | Geom Python binding module                  |
| `cfv_pybind11`      | CFV Python binding module                   |
| `eulerP_pybind11`   | EulerP Python binding module                |
| `all_unit_tests`    | All C++ unit test executables (aggregate)   |
| `dnds_unit_tests`   | DNDS C++ unit tests (aggregate)             |
| `geom_unit_tests`   | Geom C++ unit tests (aggregate)             |
| `cfv_unit_tests`    | CFV C++ unit tests (aggregate)              |
| `euler_unit_tests`  | Euler C++ unit tests (aggregate)            |
| `acm_unit_tests`    | ACM C++ unit tests (aggregate)              |
| `acm_variable_unit_tests` | ACMVariable C++ tests (aggregate)     |
| `solver_unit_tests` | Solver C++ unit tests (aggregate)           |
| `ncfv_unit_tests`   | NCFV C++ unit tests (aggregate)             |
| `docs`              | Build all documentation (Doxygen + Sphinx)  |
| `sphinx`            | Build Sphinx documentation only             |
| `doxygen`           | Build Doxygen XML + HTML only               |

Only `euler` is part of the default solver build. The `acm` and `acmVariable`
module targets aggregate both dimensions for tests; executable targets link
only their dimension-specific libraries. Shared infrastructure is reused.
See @ref solver_split_zh for the dependency and case migration details.

> **See also:** @ref building for full build instructions, CMake preset
> descriptions, and troubleshooting.
