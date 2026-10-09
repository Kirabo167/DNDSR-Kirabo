# Unit Test Suite Overview {#test_overview}

@tableofcontents

DNDSR uses [doctest](https://github.com/doctest/doctest) for C++ unit
tests and [pytest](https://docs.pytest.org/) with
[pytest-timeout](https://pypi.org/project/pytest-timeout/) for Python tests.
MPI Python runs invoke `mpirun` explicitly rather than relying on a pytest MPI
plugin.
MPI-aware C++ tests are registered with CTest at multiple process counts
to verify parallel correctness.

## Module Test Pages

| Module | Aggregate target | C++ executables | Additional coverage |
|---|---|---:|---|
| **DNDS** | `dnds_unit_tests` | 8 | Python core tests |
| **Geom** | `geom_unit_tests` | 10 | Python mesh tests and multi-zone CGNS |
| **CFV** | `cfv_unit_tests` | 3 + 1 CUDA-only | Python FV/reconstruction tests |
| **Euler** | `euler_unit_tests` | 4 baseline + 5 Cantera | Reactive evaluator, chemistry, thermo, RANS |
| **ACM** | `acm_unit_tests` | 5 | MPI and periodic self-face regression |
| **ACMVariable** | `acm_variable_unit_tests` | 2 | Core and MPI algebra |
| **Solver** | `solver_unit_tests` | 4 | ODE, linear, direct, and scalar solvers |
| **NCFV** | `ncfv_unit_tests` | 3 | Geometry, MPI, and I/O |
| **EulerP** | — | — | Python host/CUDA evaluator tests |

Doctest case/assertion counts and MPI-expanded CTest entries change as coverage
is added. Use `scripts/ctest_summary.py` and `ctest -N` for the configured
build instead of relying on a hard-coded total in this document.

## Quick Start

```sh
# 1. Configure the baseline CPU matrix (Cantera/CUDA off)
CC=mpicc CXX=mpicxx cmake --preset release-test

# 2. Build native entry points for configuration CTests and all test executables
cmake --build --preset solvers --parallel 4
cmake --build --preset tests --parallel 4

# 3. Run the full C++ test suite
# (fetch the pinned cfd_meshes fixtures from the v0.3.1 guide first)
ctest --preset unit

# 4. Run Python tests (requires pybind11 shared libraries)
cmake --build --preset python -j32
cmake --install build --component py
PYTHONPATH="$PWD/python" venv/bin/python -m pytest test/ -v

# 5. Validate every C++ module with Cantera enabled
CC=mpicc CXX=mpicxx cmake --preset reactive-test
cmake --build build-reactive --target all_solvers --parallel 4
cmake --build --preset reactive --parallel 4
ctest --preset reactive

# Optional: rerun only the eight chemistry/reactive checks
ctest --preset reactive-focused
```

## Aggregate CMake Targets

| Target | Contents |
|---|---|
| `dnds_unit_tests` | All DNDS core test executables |
| `geom_unit_tests` | All Geom test executables |
| `cfv_unit_tests` | All CFV test executables |
| `euler_unit_tests` | All Euler test executables |
| `acm_unit_tests` | Constant-density ACM tests |
| `acm_variable_unit_tests` | Variable-density ACM tests |
| `solver_unit_tests` | All Solver test executables |
| `ncfv_unit_tests` | All NCFV test executables |
| `all_unit_tests` | All of the above |

All test executables are `EXCLUDE_FROM_ALL` and must be built explicitly.

## MPI Test Registration

MPI-aware tests are registered at multiple process counts, and the CTest name
encodes the count. All MPI registration helpers use `DNDS_TEST_NP_LIST`, whose
default is np = 1, 2, 4, and 8. The NCFV I/O test is intentionally registered
only at np = 1 and 2. Configure a smaller matrix when needed:

```sh
DNDS_TEST_NP_LIST="1;2;4" cmake --preset release-test
```

`DNDS_TEST_TIMEOUT` defaults to 1800 seconds and scales upward for larger MPI
jobs and evaluator pipelines. `DNDS_TEST_OMP_THREADS` defaults to 2. Both may
be set in the environment at configure time.

## Configuration and executable-selection regressions

`solver_case_configurations` calls the native `--check-config` entry for the
classified case files. `solver_executable_selection` verifies rejection of
incorrect models and dimensions. Both are labelled `solver_config` and need
the relevant executables from `all_solvers`, which `all_unit_tests` does not build.

```bash
ctest --test-dir build -L solver_config --output-on-failure
python3 scripts/check_solver_cases.py --build-dir build \
    --reactive-build-dir build-reactive
python3 cases/validate_configs.py --quiet
```

Without a Cantera build, the native audit lists reaction cases as requiring
another build. Supplying `--reactive-build-dir` checks them with that build.
Mesh manifests are data descriptions and are excluded from solver validation.
Parameter checks do not confirm input-file availability or simulation convergence.
The dated split validation record and upstream comparison limits are in
@ref solver_split_zh.

## Naming Conventions

### C++ (CTest)

```
<module>_<test_name>             # serial
<module>_<test_name>_np<N>       # MPI at N ranks
```

Examples: `cfv_limiters`, `euler_evaluator_pipeline_np4`,
`solver_ode`.

### Python (pytest)

```
test/<Module>/test_<name>.py::TestClass::test_method
```

Examples: `test/CFV/test_fv_correctness.py::TestCellVolumes::test_wall_mesh_cell_volumes`.

## Golden Values and Regression

Many tests compare computed results against pre-captured **golden
values** with a relative tolerance (typically 1e-6 to 1e-8).  Golden
values are deterministic because:

- Iterative VR uses **Jacobi iteration** (not SOR) to avoid
  partition-dependent update ordering.
- Euler pipeline tests use **Jacobi update** (not LU-SGS) for the same
  reason.
- Metis partitioning uses a fixed seed (`metisSeed = 42`).

When a golden value has not yet been captured, the sentinel `1e300`
(or `0.0` in older tests) is stored.  In this mode the test only prints
the value and checks it is finite and non-negative — no regression
assertion.

## POSIX `index()` Ambiguity

Because doctest includes `<cstring>`, which transitively pulls in the
POSIX `index()` function from `<strings.h>`, the bare name `index` is
ambiguous when `using namespace DNDS;` is active.  All C++ test files
qualify the DNDS type aliases as `DNDS::index`, `DNDS::real`, and
`DNDS::rowsize` in variable declarations.

## Rebuilding Before Testing

Before running Python tests, always rebuild and reinstall the pybind11
shared libraries:

```sh
cmake --build build -t dnds_pybind11 geom_pybind11 cfv_pybind11 eulerP_pybind11 -j32
cmake --install build --component py
```

If C++ source changed since the last build, stale `.so` files produce
misleading crashes (segfaults, aborts, wrong results) that look like code
bugs.
