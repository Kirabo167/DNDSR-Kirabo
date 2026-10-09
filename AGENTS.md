# AGENTS.md — DNDSR

> **CRITICAL RULE — NEVER COMMIT WITHOUT AUTHORIZATION.**
> `git commit`, `git commit --amend`, and `git push` require an explicit
> "proceed", "go ahead", "authorized", or similar directive from the user.
> "Looks good", "ok", or passive approval is NOT sufficient.  If in doubt,
> ask: "Proceed with commit?"

DNDSR is a C++17 / Python CFD (Computational Fluid Dynamics) research code implementing
Compact Finite Volume methods with MPI parallelism and optional CUDA GPU support.

## Project Structure

- `src/` — C++ and Python source, organized by module:
  - `DNDS/` — Core: MPI arrays, serialization (JSON, HDF5), profiling, CUDA
    - `Config/` — Runtime configuration enums, parameters, registry
    - `Device/` — Host-device memory transfer, CUDA utilities
    - `Serializer/` — JSON and HDF5 serialization framework
    - `ArrayDerived/` — Specialized array types (adjacency, Eigen matrices)
  - `Geom/` — Unstructured mesh, CGNS I/O, partitioning (Metis/ParMetis)
    - `Mesh/` — Mesh data structures, connectivity, ghost management, state tracking
    - `Elements/` — Per-element-type shape functions
    - `Quadratures/` — Numerical integration rules
  - `CFV/` — Compact Finite Volume, variational reconstruction
  - `Euler/` — Compressible N-S solvers (2D/3D, SA, k-omega RANS)
  - `EulerP/` — Alternative evaluator with CUDA GPU support
  - `Solver/` — Header-only ODE, Krylov, and direct-solver utilities
  - `ACM/` — Constant-density artificial-compressibility solver
  - `ACMVariable/` — Variable-density artificial-compressibility solver
  - `NCFV/` — Third-order node-centred finite-volume solver
- `app/` — C++ application entry points (solver executables)
- `test/` — Test suite
  - `cpp/` — C++ unit tests (doctest, registered with CTest)
  - `DNDS/`, `Geom/`, `CFV/`, `Euler/`, `EulerP/` — Python tests
    (pytest + pytest-timeout; use `mpirun` explicitly)
- `cases/` — JSON/JSONC configurations grouped by `euler`, `acm`, `ncfv_euler`, then `2D`/`3D`
- `external/` — Git submodule (`cfd_externals`) and header-only libraries

## Build Commands

> **For humans:** The canonical build guide with full explanations,
> troubleshooting, and platform notes lives in `docs/guides/building.md`.
> This section is a condensed agent reference.

### C++ (CMake)

```bash
# Configure (from project root)
CC=mpicc CXX=mpicxx cmake -S . -B build -DDNDS_USE_CANTERA=OFF
# ^ Use CC=mpicc CXX=mpicxx when unsure which MPI CMake will find.

# Or use CMake presets (see CMakePresets.json)
cmake --preset release-test   # Release with tests enabled
cmake --preset debug          # Debug with tests enabled
cmake --preset reactive-test  # Release with Cantera and reactive tests
cmake --preset cuda           # Release with CUDA and tests

# Build a specific target (-j for parallel)
cmake --build build -t euler -j 8
# Solver targets: euler, euler2D, euler3D, eulerSA, eulerSA3D, euler2EQ, euler2EQ3D,
# eulerEX, eulerEX3D, ACM, acm2D, acm3D, acmVariable2D,
# acmVariable3D, ncfv_euler2D, ncfv_euler3D
# Solver groups: all_euler, all_acm, all_ncfv_euler, all_solvers
# Build presets: solvers, acm, ncfv_euler (release-test configuration)
# Tools: eulerState; with Cantera: canteraConstVolTrajectory, cantera_Test
# Python modules: dnds_pybind11, geom_pybind11, cfv_pybind11, eulerP_pybind11
```

### Python Package (scikit-build-core)

```bash
# Full install
CC=mpicc CXX=mpicxx CMAKE_BUILD_PARALLEL_LEVEL=16 pip install . --no-build-isolation --verbose

# Editable install
CC=mpicc CXX=mpicxx CMAKE_BUILD_PARALLEL_LEVEL=16 pip install -e . --no-build-isolation --verbose

# Or rebuild/install the in-place modules from the release-test preset
cmake --build --preset python -j32 && cmake --install build --component py
```

### Using the DNDSR Python Module (from build/)

To use the pybind11-based Python module from a CMake build directory:

1. **Build the pybind11 targets:**

   ```bash
   cmake --build build -t dnds_pybind11 geom_pybind11 cfv_pybind11 eulerP_pybind11 -j32
   ```

2. **Install the Python component** (copies `.so` files into `python/`):

   ```bash
   cmake --install build --component py
   ```

3. **Run with the project venv and `PYTHONPATH`:**

   ```bash
   source venv/bin/activate
   PYTHONPATH=<project_root>/python python my_script.py
   ```

The `.so` files are built against the active virtual environment. Python 3.12
is the CI-tested development baseline (package metadata permits Python 3.10+).
Always use that environment's interpreter (normally `venv/bin/python`) rather
than an unrelated system or Conda Python.

### External Dependencies

```bash
git submodule update --init --recursive --depth=1

# Cantera/external build requirements (inside the active venv)
pip install -r external/cfd_externals/requirements.txt

# Download and extract the header-only bundle.
bash scripts/install_headeronly_deps.sh

cd external/cfd_externals
CC=mpicc CXX=mpicxx python cfd_externals_build.py
cd ../..
bash scripts/install_python_deps.sh
```

## Test Commands

### Python Tests

**IMPORTANT: Before running ANY Python test, you MUST build the pybind11
shared libraries AND install them.** The Python modules load `.so` files
from `python/DNDSR/`, which are only placed there by `cmake --install`.
Running Python tests against stale or missing `.so` files will produce
misleading crashes (segfaults, aborts, wrong results) that look like code
bugs but are actually stale-binary problems. **Every time C++ source
changes, repeat both steps before running Python tests:**

```bash
# Step 1: Build pybind11 targets
cmake --build build -t dnds_pybind11 geom_pybind11 cfv_pybind11 eulerP_pybind11 -j32

# Step 2: Install into python/ (MANDATORY — do not skip)
cmake --install build --component py

# Step 3: Now run Python tests
source venv/bin/activate
PYTHONPATH=<project_root>/python pytest test/
```

If you switch git branches or checkout different commits, you MUST rebuild
and reinstall before running Python tests. A `git checkout` changes source
files but does NOT rebuild binaries — the installed `.so` files will be
from the previous build and will silently produce wrong behavior.

### HARD RULE — never discard changes you did not make

**Every modified tracked file belongs to the user** (or another agent).
The agent is a visitor in the user's working tree.  It may edit files,
stage changes, or create new files, but it must **never** destroy the
user's uncommitted work.

**Prohibited operations on files/state the agent did NOT modify:**
- `git checkout -- <file>`
- `git restore <file>`
- `git checkout <file>`
- `git reset --hard` / `git clean -f`
- overwriting a file with `git show <rev>:<path> > <path>`
- any command whose effect is to discard or revert the user's changes

If a clean working tree is genuinely needed (e.g. for a benchmark
baseline, or to isolate the agent's own edits from pre-existing noise),
**push a stash with a brief message first** — a stash is harmless and
the user can pop it later:

    git stash push -m "baseline: user changes before <task description>"

Always check `git stash list` after popping to confirm the stash was
consumed.  A popped stash with conflicts or a dirty tree leaves the
stash intact — the changes are NOT applied.

**Before running ANY `git checkout`, `git switch`, `git restore`,
`git reset`, `git stash`, `git stash pop`, or `git checkout -- <file>`
command, ALWAYS run `git status` first.** Verify the working tree is
clean or that all valuable changes are committed/stashed.
    Do not guess — a "clang-format: N file(s)" pre-commit message does
    not guarantee the changes are formatting-only.

### Running Solver Executables

Solver executables (`euler`, `eulerEX`, `eulerSA`, etc.) are run from the
`build/` directory.  **All file paths in JSON config files (mesh, output,
mechanism) are relative to the CWD at invocation time** — typically
`build/`.  Use `../` to reach the project root:

```bash
# From the Cantera-enabled build directory
(cd build-reactive && \
  DNDS_MECH_PATH=../external/cfd_externals/install/data \
  ./app/eulerEX.exe 14 ../cases/euler/2D/EX/react_test.json)
```

Cases live under `cases/{euler,acm,ncfv_euler}/{2D,3D}`; Euler RANS/EX
variants and variable-density ACM have their own subdirectories. Schemas are
stored beside their model's cases. `euler` is the upstream five-variable NS
model on 2D geometry; `euler2D` is the four-variable NS_2D model.

Euler reads an existing case-adjacent `<target>_default_config.json` and does
not overwrite it; absent files use compiled defaults. Match final merged
parameters when comparing to upstream. ACM loads complete single-file cases;
NCFV merges typed defaults. `--check-config` performs parameter/selection
checks without reading meshes or writing simulation results. See
`docs/guides/solver_split_zh.md` for the pinned upstream comparison and limits.

**Path conventions in configs:**
- `meshFile`: relative to CWD (e.g. `../data/mesh/IV10_10.cgns`)
- `outPltName`: relative to CWD (e.g. `../data/out/react_test/react_`)
- `mechanismFile` (in `reactiveFlow`): relative to `DNDS_MECH_PATH` env var or CWD

**Editing solver JSON configs:** must use the edit tool.  Read and
understand the file first, then edit in place — never rewrite the
whole config with Python `json.dump`.  Python is acceptable for
read-only queries (`json.load`, `json.tool` validation).  Configs may
contain hand-maintained `caseNotes["/**/"]` sections and inline
comments that must be preserved across edits.

Tests use **pytest** with **pytest-timeout**; MPI Python runs invoke `mpirun`
explicitly. Test files live under `test/`. A default 120-second timeout is
configured in `pyproject.toml` to prevent hung tests from blocking CI.

```bash
# Run all tests
pytest test/

# Run a single test file
pytest test/DNDS/test_basic.py

# Run a single test function
pytest test/DNDS/test_basic.py::test_all_reduce_scalar

# Run with MPI (multiple ranks)
mpirun -np 4 python -m pytest test/DNDS/test_basic.py

# Run a test file as a script (some tests support this)
python test/DNDS/test_basic.py
mpirun -np 2 python test/DNDS/test_basic.py
```

### C++ Unit Tests (doctest)

C++ tests live under `test/cpp/` and use the [doctest](https://github.com/doctest/doctest)
framework. They are built when `DNDS_BUILD_TESTS=ON` and registered with CTest.
MPI tests are registered at np=1, np=2, np=4, and np=8 by default (configurable via
`DNDS_TEST_NP_LIST` environment variable at configure time). All tests run with
`OMP_NUM_THREADS=2` by default (configurable via `DNDS_TEST_OMP_THREADS` environment
variable at configure time).

```bash
# Configure with tests enabled (from build directory)
cmake .. -DDNDS_BUILD_TESTS=ON -DDNDS_USE_CANTERA=OFF

# Configure with custom OMP threads (optional)
DNDS_TEST_OMP_THREADS=4 cmake .. \
  -DDNDS_BUILD_TESTS=ON -DDNDS_USE_CANTERA=OFF

# Build solver entry points needed by solver_config CTests, plus all unit tests
cmake --build . -t all_solvers all_unit_tests -j8

# Build only specific category
cmake --build . -t dnds_unit_tests -j8   # DNDS/ tests only
cmake --build . -t geom_unit_tests -j8   # Geom/ tests only
cmake --build . -t cfv_unit_tests -j8    # CFV/ tests only
cmake --build . -t euler_unit_tests -j8  # Euler/ tests only
cmake --build . -t acm_unit_tests -j8    # ACM/ tests only
cmake --build . -t acm_variable_unit_tests -j8 # ACMVariable/ tests only
cmake --build . -t solver_unit_tests -j8 # Solver/ tests only
cmake --build . -t ncfv_unit_tests -j8   # NCFV/ tests only

# Run all C++ tests via CTest (exclude separately managed pytest entries)
ctest --test-dir . -LE python --output-on-failure

# Run with aggregated doctest summary (shows total test cases + assertions)
python ../scripts/ctest_summary.py --output-on-failure
python ../scripts/ctest_summary.py -R "^dnds_"   # filter by category

# Configuration and wrong-model/dimension checks
ctest --test-dir . -L solver_config --output-on-failure

# Run tests by category prefix
ctest --test-dir . -R "^dnds_" --output-on-failure   # DNDS tests
ctest --test-dir . -R "^geom_" --output-on-failure   # Geom tests
ctest --test-dir . -R "^cfv_" --output-on-failure    # CFV tests
ctest --test-dir . -R "^euler_" --output-on-failure  # Euler tests
ctest --test-dir . -R "^acm_" --output-on-failure    # ACM/ACMVariable tests
ctest --test-dir . -R "^solver_" --output-on-failure # Solver tests
ctest --test-dir . -R "^ncfv_" --output-on-failure   # NCFV tests

# Run only np=2 MPI tests across all categories
ctest --test-dir . -R "_np2$" --output-on-failure

# Run a single test executable directly
./test/cpp/dnds_test_array
mpirun -np 4 ./test/cpp/dnds_test_mpi
```

**Test categories and targets:**

- **DNDS:** `dnds_test_array`, `dnds_test_mpi`, `dnds_test_array_transformer`,
  `dnds_test_array_derived`, `dnds_test_array_dof`, `dnds_test_index_mapping`,
  `dnds_test_serializer`, `dnds_test_permutation_transfer`
- **Geom:** `geom_test_elements`, `geom_test_quadrature`, `geom_test_mesh_index_conversion`,
  `geom_test_mesh_pipeline`, `geom_test_mesh_distributed_read`, `geom_test_mesh_connectivity`,
  `geom_test_mesh_connectivity_ghost`, `geom_test_mesh_connectivity_interpolate`,
  `geom_test_mesh_reorder`, `geom_test_mesh_cgns_multizone`
- **CFV:** `cfv_test_reconstruction`, `cfv_test_limiters`, `cfv_test_reconstruction3d`,
  `cfv_test_device_transferable` (CUDA only)
- **Euler:** `euler_test_gas_thermo`, `euler_test_riemann_solvers`, `euler_test_rans`,
  `euler_test_evaluator_pipeline`; with Cantera: `euler_test_source_chemical`,
  `euler_test_uv`, `euler_test_physics_properties`, `euler_test_chem_ode`,
  `euler_test_evaluator_reactive`
- **ACM:** `acm_test_core`, `acm_test_time`, `acm_test_turbulence`,
  `acm_test_self_periodic`, `acm_test_parallel`
- **ACMVariable:** `acm_test_variable_core`, `acm_test_variable_mpi`
- **Solver:** `solver_test_ode`, `solver_test_linear`, `solver_test_direct`, `solver_test_scalar`
- **NCFV:** `ncfv_test_geometry`, `ncfv_test_parallel`, `ncfv_test_io`

**Note:** When writing new C++ tests with `using namespace DNDS;`, always qualify
`DNDS::index`, `DNDS::real`, and `DNDS::rowsize` in declarations to avoid ambiguity
with POSIX `index()` from `<strings.h>` (pulled in by doctest).

### Python Geom Module

The Python `DNDSR.Geom` module provides mesh reading and manipulation capabilities.
See the comprehensive guide at `docs/guides/python_geom_guide.md` for full API
details, including all parameters, read modes, and notes on which C++ methods
are (and are not) exposed in the Python bindings.

**Quick Example:**

```python
from DNDSR.Geom.utils import read_mesh, prepare_mesh
from DNDSR import DNDS

mpi = DNDS.MPIInfo()
mpi.setWorld()

# Read mesh with elevation and bisection
result = read_mesh(
    "data/mesh/UniformSquare_10.cgns",
    mpi=mpi,
    dim=2,
    elevation="O2",         # Elevate O1→O2
    bisect=1,               # Bisect once
)
prepare_mesh(result.mesh, result.reader)
```

The legacy `create_mesh_from_CGNS` wrapper is still available for backward
compatibility.

**Key Features:**

- CGNS mesh reading (`ReadFromCGNSSerial`)
- H5 distributed reading with ParMetis repartition
- Order elevation: Quad4→Quad9, Hex8→Hex27, etc. (`BuildO2FromO1Elevation`)
- Mesh bisection for h-refinement (`BuildBisectO1FormO2`)
- Boundary mesh extraction (`build_bnd_mesh`)
- Multi-layer ghost cells via `BuildGhostPrimary(nGhostLayers)`
- VTK output generation
- Wall distance computation (`BuildNodeWallDist`)
- CUDA device offloading (`to_device` / `to_host`)

## Code Style

Full style guide (naming, formatting, includes, error handling, Doxygen,
Python conventions): **`docs/guides/style_guide.md`**

Formatting is handled by the pre-commit hook; do not run formatters manually.
If an explicit formatting run is needed, use `scripts/run-clang-format.sh`.

When reporting audit or review findings to a human user, use numbered findings
(`1.`, `2.`, `3.`) rather than bullets so each issue can be referenced
unambiguously in follow-up discussion.

Quick reference for C++:

- **Braces:** Allman (opening brace on its own line)
- **Naming:** `PascalCase` classes/methods, `_` prefix for private members,
  `DNDS_ALL_CAPS` macros, `t_` prefix type aliases
- **Headers:** `#pragma once`, preserve include order (no auto-sort)
- **Errors:** `DNDS_assert` / `DNDS_check_throw` from `DNDS/Errors.hpp`; never raw
  `assert()`.
- DNDS currently builds with `DNDS_assert*` active by default at the maximum
  assertion level, including release-style builds. Still prefer
  `DNDS_check_throw_info` for non-hot user config, file input, CLI, and runtime
  validation that should report recoverable errors rather than aborting.
- **Core types:** `real = double`, `index = int64_t`, `rowsize = int32_t`,
  `ssp<T> = std::shared_ptr<T>`

Quick reference for Python:

- `snake_case` functions/variables; C++ wrapper classes match C++ name
- Plain `assert`; `@pytest.fixture` for MPI; numpy for array comparisons

### Clang-tidy sanitation

DNDS is clean as of 2026-04-29 (26-pass cleanup, 24 597 → 1
diagnostics; the remaining one is an unrelated Eigen PCH
`omp.h` include issue). Full per-pass record, `.clang-tidy`
disable rationale, and NOLINT placement gotchas:
**`docs/dev/clang_tidy_plan.md`**.

Other modules (`Solver`, `Geom`, `CFV`, `Euler`, `EulerP`) are
not yet sanitised. Apply the same recipe in that order. Run
`scripts/run_clang_tidy.py <module>` to get the per-check
histogram; the `.clang-tidy` disables carry forward unchanged.

## Geom Module Architecture

Mesh connectivity, ghost management, and the build pipeline are documented
in **`docs/architecture/MeshConnectivity.md`**.

Key concepts agents should know:

- **Adjacency state:** Each adjacency array (e.g. `cell2node`, `face2cell`)
  is wrapped in `AdjPairTracked<TPair>` (inherits from `TPair`, adds an
  `AdjIndexInfo idx` member). All `idx` fields are private; state
  transitions go through `markGlobal()`, `markLocal()`,
  `wireTargetMapping()`, `toLocal()`/`toGlobal()`, and
  `bootstrapToLocal()`. See `src/Geom/Mesh/AdjIndexInfo.hpp`.

- **Three-layer architecture:**
  1. `MeshConnectivity` (DSL) -- bare `ArrayAdjacencyPair<rs>`, no state
  2. `Mesh/MeshConnectivity_StateChecked.hpp` -- asserts `idx.state()`, forwards to DSL
  3. `UnstructuredMesh` (Mesh/Mesh.cpp) -- owns `AdjPairTracked` members, calls checked wrappers

- **Conversion methods:** `AdjGlobal2Local*` / `AdjLocal2Global*` delegate
  to `adj.toLocal()` / `adj.toGlobal()`, which use the stored target
  mapping. The five group state variables (`adjPrimaryState`, etc.)
  still exist and are updated in parallel with per-adj states.

- **State discipline:** Every site that sets a group state variable
  (`adjXState = ...`) must also call the corresponding `idx` method
  (`markGlobal()`, `markLocal()`) on the governed adjacencies. Every
  site that builds ghost mappings must wire them to the relevant
  adjacencies via `wireTargetMapping()`. Both DSL and legacy code
  paths maintain this invariant.

## Key Dependencies

- **Compiler:** GCC 9+ / Clang 8+, C++17 required
- **MPI:** MPI-3 compatible (OpenMPI or MPICH)
- **CMake:** >= 3.21
- **C++ libs:** Eigen, Boost, CGAL, nlohmann_json, fmt, pybind11, HDF5, CGNS, Metis, ParMetis
- **Python:** package metadata permits >= 3.10; CI and development use 3.12;
  numpy, scipy, pytest, pytest-timeout, mpi4py, h5py;
  `mpi4py` and `h5py` must use the selected MPI and cfd_externals HDF5
- **Optional:** CUDA toolkit, Cantera 3.2, SuperLU_dist

## GitHub CLI (`gh`) Policy

## Git Commit Messages

Prefer verbose commit messages for non-trivial changes. Keep the subject concise
and conventional, then use the body to explain the problem, the root cause, the
fix, and the verification performed. Mention subtle behavioral changes,
edge-cases, and test coverage so future readers can understand why the change
was made without reconstructing the investigation from the diff.

For very small mechanical changes, a short message is acceptable.

Every commit message must end with a separate trailing line:

    committed by <agent harness> with model <model name>

Use the actual agent harness name (for example, `codex` or `opencode`) and model
identifier. This line must appear after a blank line separating it from the body
(or subject, if no body).

**Read-only by default.** You may use `gh` freely for read operations (viewing
issues, PRs, checks, releases, diffs, comments). **Do NOT use `gh` for any
write operation** (creating/closing issues, creating/merging PRs, posting
comments, approving reviews, creating releases, editing labels, deleting
caches, etc.) **unless the user explicitly requests that specific write
action.** One-time explicit permission does not carry over to other write
actions — ask each time.

**Operations requiring explicit user authorization (non-exhaustive):**
- `git commit` / `git commit --amend`
- `git push` / `git push --force` / `git push --force-with-lease`
- `gh pr create/merge/close/edit`
- `gh issue create/close/edit`
- `gh pr comment` / `gh issue comment`
- `gh pr review`
- `gh release create/delete`
- `gh cache delete`
- `gh api` with non-GET methods (POST, PUT, PATCH, DELETE)

Explicit authorization means a direct imperative like "commit", "proceed",
"go ahead", or "authorized".  Passive or ambiguous statements ("looks good",
"ok", "fine", "nice") are NOT authorization to commit.  A single-commit
authorization is consumed by the first commit — do NOT commit again without
a fresh authorization, even for follow-up fixes.

**Draft PR by default** You must use --draft on new prs.
