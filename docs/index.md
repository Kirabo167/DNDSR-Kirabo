(index_page)=
# DNDSR Documentation

DNDSR is a C++17 / Python CFD (Computational Fluid Dynamics) research code
implementing Compact Finite Volume methods with MPI parallelism and optional
CUDA GPU support. The EulerEX family optionally adds Cantera-based
multi-species thermodynamics, transport, and chemistry.

## Core Modules

| Module     | Directory    | Description                                              |
|------------|-------------|----------------------------------------------------------|
| **DNDS**   | `src/DNDS`  | MPI arrays, serialization (JSON, HDF5), profiling, CUDA  |
| **Geom**   | `src/Geom`  | Unstructured mesh, CGNS I/O, partitioning (Metis/ParMetis) |
| **CFV**    | `src/CFV`   | Compact Finite Volume, variational reconstruction        |
| **Euler**  | `src/Euler` | Compressible Navier-Stokes solvers (2D/3D, SA, k-omega)  |
| **EulerP** | `src/EulerP`| Alternative evaluator with CUDA GPU support              |
| **Solver** | `src/Solver`| ODE integrators and Krylov solvers (GMRES, PCG)          |
| **ACM** | `src/ACM` | Constant-density artificial-compressibility solver |
| **ACMVariable** | `src/ACMVariable` | Variable-density artificial-compressibility solver |
| **NCFV** | `src/NCFV` | Third-order node-centred finite-volume solver |

CPU solvers use separate Euler/CFV, ACM and NCFV Euler executables, each
with 2D/3D variants. Cases and schemas follow the same grouping. See
[the solver migration and upstream comparison guide](guides/solver_split_zh.md)
for target names, configuration precedence and validation scope.

## Quick Start

```sh
# 1. Build external dependencies
git submodule update --init --recursive --depth=1
python3.12 -m venv venv && source venv/bin/activate
pip install -r external/cfd_externals/requirements.txt

bash scripts/install_headeronly_deps.sh

cd external/cfd_externals && CC=mpicc CXX=mpicxx python cfd_externals_build.py && cd ../..
bash scripts/install_python_deps.sh

# 2. Build C++ solvers
cmake --preset release-test
cmake --build build --target euler --parallel 4
# All families: cmake --build --preset solvers --parallel 4

# 3. Install Python package (editable)
CC=mpicc CXX=mpicxx CMAKE_BUILD_PARALLEL_LEVEL=32 \
  pip install -e . --no-build-isolation

# 4. Run tests
# Fetch the pinned cfd_meshes fixtures from the v0.3.1 guide first.
cmake --build --preset solvers --parallel 4
cmake --build --preset tests --parallel 4
ctest --preset unit
cmake --build --preset python -j32
cmake --install build --component py
PYTHONPATH="$PWD/python" venv/bin/python -m pytest test/

# 5. Build and serve documentation
pip install -r docs/sphinx/requirements.txt
cmake --preset release-test
cmake --build build -t serve-docs
```

> **See also:** @ref building for the complete build guide (CMake presets,
> Python editable installs, external dependencies, and common issues).
> The [v0.3.1 integration guide](guides/v0.3.1_new_features_zh.md) records the
> compatibility changes, validation results, and current CUDA/reactive scope.

## C++ API (Doxygen)

The Sphinx C++ API pages provide namespace outlines and key class documentation
via Breathe. For the full Doxygen HTML documentation with interactive
**call graphs**, **class inheritance diagrams**, and **include dependency graphs**,
build and serve the Doxygen output separately:

```sh
cmake --build build -t serve-doxygen
```

Once built, the full Doxygen reference is at
<a href="doxygen/index.html">Doxygen C++ API Reference</a> within the same site.

```{toctree}
:maxdepth: 2
:caption: Contents

/sphinx/guides
/sphinx/solver-guide
/sphinx/architecture
/sphinx/theory
/sphinx/tests
/sphinx/dev
/sphinx/api_cpp
/sphinx/api_python
/sphinx/presentations
```
