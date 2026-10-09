# Contributing to DNDSR

Thank you for your interest in contributing to DNDSR. This is a C++17/Python
research CFD code with MPI parallelism, optional CUDA support, and optional
Cantera-based reactive flow. Bug reports, documentation improvements, tests,
and well-scoped features are welcome.

## Before You Start

- Read the [build guide](docs/guides/building.md),
  [style guide](docs/guides/style_guide.md), and relevant documents under
  `docs/architecture/`.
- For a large numerical-method or public-API change, open an issue first so
  the model, validation data, and compatibility expectations can be agreed.
- Keep unrelated changes in separate pull requests. Do not commit build
  directories, generated Python extensions, solver output, or private meshes.

## Development Setup

Clone the repository with submodules, create a Python 3.12 virtual environment
(the CI-tested development baseline; package metadata permits Python 3.10+),
and build the external libraries before configuring DNDSR:

```bash
git clone --recursive https://github.com/Kirabo167/DNDSR.git
cd DNDSR
python3.12 -m venv venv
source venv/bin/activate
pip install -r external/cfd_externals/requirements.txt

# Fetch and verify the pinned header-only dependency bundle, then build
# cfd_externals.
bash scripts/install_headeronly_deps.sh

cd external/cfd_externals
CC=mpicc CXX=mpicxx python cfd_externals_build.py
cd ../..
bash scripts/install_python_deps.sh
```

Use a descriptive branch name such as `feature/reactive-bc`,
`fix/periodic-connectivity`, or `docs/build-guide`. Automated Codex work uses
the `codex/` prefix.

## Development Workflow

1. Create a branch from the current `main` and keep it focused.
2. Implement the change together with tests and user-facing documentation.
3. Format only the files you changed. The repository hook can be installed
   with `ln -sf ../../scripts/pre-commit .git/hooks/pre-commit`.
4. Run the smallest relevant test group, then the required compatibility
   matrix below.
5. Use a short imperative commit subject. Explain numerical or compatibility
   decisions in the commit body or pull-request description.
6. Open a pull request against `main` and include the commands and results used
   for validation. A maintainer can trigger repository CI with `/ci-run`.

Follow the existing Allman brace style and naming conventions. Use
`DNDS_assert`/`DNDS_check_throw` rather than raw `assert()` in C++ production
code. Full conventions and formatting commands are in
`docs/guides/style_guide.md`.

## Required Validation

### Baseline CPU compatibility

All changes that touch shared C++, mesh, solver, or build infrastructure should
pass the non-reactive CPU configuration:

```bash
CC=mpicc CXX=mpicxx cmake --preset release-test
cmake --build --preset solvers --parallel 4
cmake --build --preset tests --parallel 4
ctest --preset unit
```

`solvers` builds the native entry points used by `solver_config` CTests.
`tests` builds `all_unit_tests`; `unit` runs the registered native tests while
excluding the separately managed Python CTest entries.
The full Geom/CFV/Euler matrix also needs the pinned external meshes described
in [the migration guide](docs/guides/v0.3.1_new_features_zh.md#72-测试网格).

### Reactive-flow changes

Changes under reactive Euler, thermodynamics, chemistry, source integration,
or common Gas APIs must also pass the Cantera-enabled configuration:

```bash
CC=mpicc CXX=mpicxx cmake --preset reactive-test
cmake --build build-reactive --target all_solvers --parallel 4
cmake --build --preset reactive --parallel 4
ctest --preset reactive
```

The `reactive` build/test pair covers all C++ modules, not just EulerEX. Code
shared by reactive and non-reactive solvers must compile in both
`DNDS_USE_CANTERA=OFF` and `DNDS_USE_CANTERA=ON` configurations. Use
`ctest --preset reactive-focused` only for a fast rerun of the eight dedicated
chemistry checks after the full matrix has passed.

### Python changes

Python tests must never be run against missing or stale extension modules.
After every C++ change—and after switching commits or branches—rebuild and
install all four pybind11 modules first:

```bash
cmake --build --preset python -j8
cmake --install build --component py
PYTHONPATH="$PWD/python" venv/bin/python -m pytest test/
```

The project's `mpi4py` and `h5py` must be built against the selected MPI and
the HDF5 from `cfd_externals`; use `scripts/install_python_deps.sh` rather than
unrelated binary wheels.

### CUDA changes

CUDA/EulerP changes additionally require the `cuda` configure and build
presets on a machine with a supported toolkit and device:

```bash
CC=mpicc CXX=mpicxx cmake --preset cuda
cmake --build --preset cuda -j8
```

Reactive CFD is currently a CPU-only EulerEX/EulerEX3D feature; enabling CUDA
does not provide a reactive EulerP implementation.

## Adding Tests

- C++ tests use doctest and live under `test/cpp/<Module>/`. Register new
  executables in `test/cpp/CMakeLists.txt` and attach them to the relevant
  `<module>_unit_tests` aggregate target.
- MPI tests should use the shared registration helpers so the configured
  `DNDS_TEST_NP_LIST` and timeout policy are applied consistently.
- Python tests live under `test/<Module>/` and use pytest. Mark device-only or
  slow coverage with the markers declared in `pyproject.toml`.
- Prefer generated temporary files for focused regressions. If an external
  mesh fixture is necessary, document its immutable source and checksum.

See [the test overview](docs/tests/overview.md) for module targets and commands.

## Adding a JSON Configuration Parameter

1. Add the field to the appropriate typed config object and register its JSON
   serialization/default value.
2. Thread the concrete config object through every kernel that consumes it;
   avoid hidden hard-coded fallbacks.
3. Add default-value, JSON round-trip, and physical-effect tests where
   applicable.
4. Build all 15 schema-producing solver variants. Euler schemas require
   Cantera; ACM/NCFV can use that build or a separate CPU build. The `schemas`
   preset builds `all_solvers` in `build-reactive`. Regenerate schemas in their
   classified case directories:

   ```bash
   cmake --preset reactive-test
   cmake --build --preset schemas -j8
   bash cases/update_schemas.sh build-reactive build
   venv/bin/python cases/validate_configs.py --quiet
   ```

   Native checks additionally validate model/state-size compatibility:

   ```bash
   python3 scripts/check_solver_cases.py --build-dir build \
       --reactive-build-dir build-reactive
   ctest --test-dir build -L solver_config --output-on-failure
   ```

   JSON tools do not import the Python extension modules. Parameter checks
   do not read meshes; simulations still require the referenced fixtures.

5. Update the relevant commented example configuration and user documentation.
   Preserve backward-compatible defaults unless the change is explicitly
   documented as a migration.

## Documentation Changes

Documentation lives under `docs/` and is consumed by Sphinx and Doxygen. Add a
new page to the appropriate toctree and follow the checklist in
`docs/guides/doc_authoring.md`. User-visible behavior changes should also update
`README.md`, `RELEASE_NOTES.md`, or a versioned migration guide as appropriate.

## Python Packaging

Build wheels only from a prepared checkout containing the pinned submodule,
header-only dependency bundle, and compiled `cfd_externals`. The current
external-dependency model is not a self-contained PEP 517 source build, so do
not publish the generated sdist as an installable release artifact. Wheel
installs deliberately retain the host MPI and C++ runtime while staging the
four extension modules, project shared libraries, generated type stubs, and
the bundled CGNS/HDF5/Metis/ParMetis/zlib libraries.

Treat locally built wheels as development/internal artifacts. In particular,
the bundled ParMETIS license restricts redistribution; do not publish or
redistribute a wheel until the intended distribution has completed a
third-party license review and obtained any required permission.
See [the binary dependency inventory](THIRD_PARTY_DEPENDENCIES.md) for the
current bundled components and release checklist.

## Pull Request Checklist

- [ ] The change is focused and preserves unrelated user work.
- [ ] New behavior has regression coverage.
- [ ] Baseline CPU tests pass; Cantera/CUDA matrices were run when applicable.
- [ ] Python extensions were rebuilt and installed before pytest.
- [ ] Config schemas and examples were regenerated when parameters changed.
- [ ] User-facing documentation and dependency metadata are current.
- [ ] The pull-request description records any untested hardware path.

## Code of Conduct

Be respectful and constructive. DNDSR is a research code, and design decisions
often reflect numerical-method requirements rather than general
software-engineering preferences. Technical discussion is welcome.

## See Also

- [Building DNDSR](docs/guides/building.md)
- [v0.3.1 features and migration notes](docs/guides/v0.3.1_new_features_zh.md)
- [Solver split and upstream comparison (中文)](docs/guides/solver_split_zh.md)
- [Code style](docs/guides/style_guide.md)
- [Documentation authoring](docs/guides/doc_authoring.md)
- [Test-suite overview](docs/tests/overview.md)
- [Active development tasks](docs/dev/TODO.md)
