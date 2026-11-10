<!--
File description: implementation status and integration boundary of the ACM module.
Modifier: Runzhi Ma
Last modified: 2026-11-10
-->

# ACM high-order solver

This directory contains an independent constant-density ACM solver for two- and
three-dimensional unstructured meshes. It reuses `Geom`, `CFV`, and the existing
MPI/OpenMP array infrastructure, but it does not modify or inherit the Euler equation model.

Implemented now:

- model-specific state layouts `[u,v,w,p]`, `[u,v,w,p,rho*nuTilde]`, and
  `[u,v,w,p,rho*k,rho*omega]` (or `rho*epsilon`);
- physical flux and its conservative Jacobian;
- Scheme-A preconditioning matrices and eigenvalues;
- general-Turkel-`alpha` Rusanov and Roe fluxes;
- laminar viscous flux kernel with a linearly exact, center-connection face-gradient correction;
- all Euler-named boundary families, implemented with ACM `[u,v,w,p]` semantics;
- per-CGNS-zone boundary configuration and reserved Euler zone-name mapping;
- OpenMP face-buffer evaluation and MPI checksum reduction;
- explicit three-stage SSPRK3 pseudo-time integration with `Gamma^{-1} R`;
- implicit backward-Euler pseudo-time integration with nonlinear 4x4 block-Jacobi corrections;
- backward-Euler-started, constant-step BDF2 physical dual-time integration with either
  distributed ACM LU-SGS or left-preconditioned GMRES inner solves;
- CGNS mesh reading, METIS partitioning, ghost construction, and periodic translation reuse;
- direct second-order Green-Gauss reconstruction;
- CFV variational reconstruction with polynomial degrees 1--3 selected by `vfvSettings.maxOrder`;
  degree zero selects piecewise-constant reconstruction;
- selectable local-extrema, WBAP, and CWBAP limiting;
- ACM-specific general-`alpha` 4x4 characteristic transforms for WBAP/CWBAP, dimension-aware
  two-/three-dimensional polynomial norms, and a dedicated 3-D four-variable CFV instantiation;
- face-quadrature evaluation of reconstructed left/right states and viscous gradients;
- frozen-reconstruction face-flux diagonal Jacobians for implicit block-Jacobi updates;
- ACM 4x4 first-order face Jacobians including viscous jump terms and boundary chain rules;
- parallel block LU-SGS forward/backward sweeps with lagged off-rank ghost coupling;
- direct reuse of generic left-preconditioned GMRES with block-Jacobi or ACM LU-SGS preconditioning;
- Euler-style local/global CFL pseudo-time steps with convective and viscous spectral radii;
- runtime-selectable laminar, SA, Wilcox k-omega, SST k-omega, and Realizable k-epsilon modes;
- configurable parallel VTK-HDF output for flow and active turbulence fields;
- unified conservative RANS transport with wall distance, MPI ghost exchange,
  limited second-order turbulence reconstruction, positivity bounds, local implicit source
  blocks, and explicit isotropic turbulent kinetic energy stress;
- DNDS configuration registration, self-contained single-case JSON loading, CLI overrides, and schema output.

Applications use independent dimension-specific entry points:

- `app/ACM/acm2D.cpp` and `app/ACM/acm3D.cpp`: select the compiled ACM dimension;
- `app/ACM/ACM.cpp`: compatibility entry for the 3D model;
- `SingleBlockApp.hpp`: ACM CLI, configuration checks and solve workflow;
- `ACMSolver.*`: mesh/reconstruction/time-loop assembly;
- `ACMRANSSolver.*`: unified SA/two-equation state, time integration, output and restart;
- `ACMEvaluator.*`: high-order spatial residual and frozen-reconstruction Jacobian;
- `ACMTurbulence.*`: model-local viscosity, diffusion, source, and boundary kernels;
- `ACMTurbulenceTransport.*`: dimension-generic finite-volume transport and flow coupling;
- `acm2D.cpp` and `acm3D.cpp`: explicit template instantiations for four ACM variables.

Case configuration uses one file per case. `cases/acm/2D/acm2D.json` and
`cases/acm/3D/acm3D.json` each contain the complete physical, numerical, mesh, reconstruction,
boundary, and initial-state configuration. The application does not search for or merge an
adjacent base file. The executable fixes the dimension; the case's `solver.type="ACM"` and
`solver.model` must match it. `-k/-v` overrides remain available for short studies.
From the repository root:

```bash
cmake --build build --target acm2D acm3D --parallel 4
(cd build && ./app/acm2D.exe ../cases/acm/2D/acm2D.json --check-config)
(cd build && ./app/acm2D.exe ../cases/acm/2D/acm2D.json)
(cd build && ./app/acm3D.exe --emit-schema)
```

`acm_core` contains shared kernels; `acm_2D` and `acm_3D` contain separate
instantiations. The `acm` module target aggregates both for tests; `all_acm`
builds constant-/variable-density variants and the compatibility executable.

The executable selects the compiled state layout; the runtime closure must match it:

- `acm2D/3D`: `[u,v,w,p]`, `Laminar`, and `mu_t=0`;
- `acmSA2D/3D`: `[u,v,w,p,rho*nuTilde]`, `SpalartAllmaras`;
- `acm2EQ2D/3D`: `[u,v,w,p,rho*k,rho*omega]` for `KOmegaWilcox`/`KOmegaSST`,
  or `[u,v,w,p,rho*k,rho*epsilon]` for `RealizableKEpsilon`.

The same implementation is instantiated for both two and three dimensions. It reuses Geom wall
distance/mesh metrics, CFV face quadrature, and the existing MPI arrays, but it neither includes,
links, nor modifies the Euler module. Flow and conservative turbulence fields advance in the
same RK stage or nonlinear iteration. The implicit system includes local turbulence source
blocks, approximate transport diagonals, and face derivatives from `rho*k` into the flow
operator. Eddy viscosity and other cross derivatives remain frozen; turbulence spatial
off-diagonal Jacobians are absent. This is an approximate coupled system, not a full Newton
linearization. See the [state and integration guide](../../docs/solver-guide/acm_model_state_zh.md).

Two-equation models default to `turbulenceSettings.includeIsotropicStress=true`.
The tensor `T_k=(2/3)rho*k I` enters momentum flux through temporary effective
pressure `p+(2/3)rho*k`, also used for Riemann dissipation and characteristic
boundaries. Stored/input/output `Pressure` is static pressure; unified RANS output
also includes `TurbulentPressure=(2/3)rho*k`. SA and laminar models add no k stress.
Setting `includeIsotropicStress=false` retains the old modified-pressure convention.

For old constant-density VTK-HDF restarts containing modified pressure, enable
`restartSettings.pressureIsModified=true` with explicit stress to convert using
loaded k. Missing turbulence fields are rejected. New static-pressure restarts
use `false`; initial states and pressure boundary values must already follow the
selected convention. Steady same-partition restart is supported; complete BDF2
history restart is not available.

The governing-equation derivation, SST blending formulas, and implementation map are in
[`docs/solver-guide/acm_rans_zh.md`](../../docs/solver-guide/acm_rans_zh.md).

A non-laminar model requires `acmSettings.enableViscousFlux=true` and positive
`acmSettings.dynamicViscosity`. A typical three-dimensional SST selection is:

```json
"turbulenceSettings": {
  "model": "KOmegaSST",
  "includeIsotropicStress": true,
  "initialValue": [0.001, 10.0],
  "farFieldValue": [0.001, 10.0],
  "minimumValue": [1e-12, 1e-10],
  "maximumValue": [1000000.0, 1000000000000.0],
  "maximumEddyViscosityRatio": 100000.0,
  "wallOmegaCoefficient": 800.0,
  "enableSourceTerms": true,
  "secondOrderReconstruction": true,
  "transportSubsteps": 1,
  "transportTimeScale": 0.25,
  "wallDistanceMethod": 1,
  "wallDistanceExecution": 0,
  "wallDistanceSubdivide": 0,
  "minimumWallDistance": 1e-10,
  "wallDistanceVerbose": 0
}
```

This is a `turbulenceSettings` fragment for `acm2EQ3D`, whose metadata must select
`ConstantDensity2EQ3D` with `fieldNVariables=6`. Set the complete `initialState`
tail to `[rho0*k,rho0*omega]`; `initialValue` does not initialize the unified state.
`farFieldValue`, `minimumValue`, and `maximumValue` use primitive `[k,omega]` values
(or `[k,epsilon]` / `[nuTilde,unused]` for their models). `maximumEddyViscosityRatio`
caps only `mu_t/mu`. Unified RANS requires `transportSubsteps=1`;
`transportTimeScale` controls turbulence pseudo-time scaling, not physical BDF derivatives.
For high-order RANS calculations, enabling the main ACM reconstruction limiter is
recommended as well. Far-field turbulence values are imposed only where `BCFar` is locally inflow;
outflow uses zero normal gradient. Wall states impose zero `nuTilde`/`k`, with
`omega=wallOmegaCoefficient*nu/d^2` for the two k-omega models and
`epsilon=2*nu*k/d^2` for Realizable k-epsilon.

Boundary-model notes:

- `BCFar` uses incoming characteristics of the general-`alpha` ACM preconditioned system;
- at the isolated condition `alpha*q_n^2=beta2/rho0`, the normal operator can be defective; Roe
  evaluates the entropy-fixed matrix absolute value by a confluent-Hermite polynomial and retains
  the exact Jordan derivative term; `BCFar` uses a finite spectral cluster, while WBAP/CWBAP falls
  back to component space because no complete characteristic basis exists there;
- `BCWallIsothermal` is equivalent to `BCWall`, because constant-density ACM has no temperature
  or energy variable;
- `BCInPsTs` interprets the configured velocity as face data and extrapolates pressure, because
  total pressure/temperature cannot be reconstructed from `[u,v,w,p]`;
- `BCSpecial` currently supports `specialOption=0`, meaning a prescribed complete ACM state;
- legacy names (`FarField`, `NoSlipWall`, `SlipWall`, `PressureOutlet`, `VelocityInlet`,
  `Symmetry`) remain accepted by the JSON reader.

One named outlet can be configured without changing the mesh reader or solver driver:

```json
"defaultBoundaryType": "BCFar",
"boundaryConditions": [
  {
    "type": "BCOutP",
    "name": "OUTLET",
    "value": [0.0, 0.0, 0.0, 1.0],
    "frameOption": 0,
    "anchorOption": 0,
    "integrationOption": 0,
    "specialOption": 0,
    "rectifyOption": 0,
    "valueExtra": []
  }
]
```

The `name` must match the CGNS boundary-zone name. Set `useCFLTimeStep=true` in
`timeMarchSettings` to enable local spectral-radius stepping; `useLocalTimeStep=false` replaces
all local values with the MPI-global minimum.

## BDF2 physical dual-time marching

Select `BDF2DualTimeLUSGS` or `BDF2DualTimeGMRES` as the `integrator`. In this mode, `nSteps`
counts physical-time steps, `physicalTimeStep` is the uniform physical step, and
`maxImplicitIterations` limits the pseudo-time corrections inside each physical step. The first
physical step uses backward Euler; every later step uses constant-step BDF2. The physical-time
mass matrix is `diag(1,1,1,0)`, so BDF differentiates velocity but never artificial-compressibility
pressure. All four RANS closures transport their active primitive variables with matching
completed-step physical histories: SA `nuTilde`, Wilcox/SST `[k,omega]`, and realizable
`[k,epsilon]`. Inactive storage is zeroed and excluded from positivity checks and BDF defects.
Their spatial-minus-BDF defects are relaxed at each flow inner iteration.
CFL controls continue to set only the inner pseudo-time step.

The new setting is available to JSON configuration and command-line JSON-pointer overrides. Case
files created before this option remain loadable: when `physicalTimeStep` is absent, the loader
inserts the default `0.01` into the in-memory normalized configuration. No existing case JSON must
be edited merely to retain its previous steady integration behavior.

An eventual BDF2 selection has the following form (this documentation example does not modify an
existing case file):

```json
"timeMarchSettings": {
  "integrator": "BDF2DualTimeLUSGS",
  "nSteps": 4000,
  "physicalTimeStep": 0.01,
  "maxImplicitIterations": 20,
  "implicitTolerance": 1e-8,
  "dualTimeConvergenceCriterion": "AbsoluteOrRelative",
  "dualTimeFlowRelativeTolerance": 1e-4,
  "linearAbsoluteTolerance": 1e-10,
  "linearRelativeTolerance": 1e-6
}
```

Laminar BDF2 also accepts `dualTimeConvergenceCriterion="RelativeUpdate"`,
`dualTimeUpdateRelativeTolerance=1e-5`, and `dualTimeUpdateMinimumIterations=2`.
It tests the global owned-cell L2 norm of the applied damped update divided by the
updated `[u,v,w,p]` state norm. A zero tolerance disables update-based early exit
and requires diagnostic budget-limited advancement. Inner and physical-step logs
keep the true BDF defect and a separate `residualConverged` flag; a small update
does not certify that defect. Existing configurations retain their previous criterion.

See `docs/solver-guide/acm_bdf2_dual_time_zh.md` for the governing defect, implicit matrix, source
mapping, and usage details.

## Build and focused RANS/BDF2 regression tests

From the project root:

```bash
cmake -S . -B build -DDNDS_USE_CANTERA=OFF -DDNDS_BUILD_TESTS=ON
cmake --build build --target all_acm acm_rans_bdf2_tests --parallel 4
ctest --test-dir build -L '^acm_rans_bdf2$' --output-on-failure
```

The `acm_rans_bdf2_tests` build target is available when `DNDS_BUILD_TESTS=ON`.
The CTest label selects the RANS closure tests, BDF2 kernels and active-variable histories,
and physical-time regressions for Wilcox, SA, and realizable k-epsilon. The latter include
independent BE/BDF2 discrete solutions and time-step refinement for SA and k-epsilon.
These tests set `HWLOC_COMPONENTS=-gl` to avoid GL/X11 probing on headless compute nodes.
The executable selects dimension/density family; flow and turbulence settings
are selected at runtime through that ACM variant's JSON. See
[the migration guide](../../docs/guides/solver_split_zh.md) for the upstream
comparison and verification scope.

Current limits:

- periodic translations are configurable, while rotational periodic setup is not exposed yet;
- WBAP/CWBAP requires `Variational` reconstruction;
- the independent turbulence field uses first-order or limited second-order Green-Gauss
  reconstruction, controlled by `turbulenceSettings.secondOrderReconstruction`, with every
  supported flow reconstruction and time integrator;
- LU-SGS/GMRES uses a first-order frozen face linearization as the implicit operator while the
  nonlinear residual retains the selected high-order reconstruction;
- unified RANS uses approximate implicit coupling rather than a full turbulence spatial Jacobian;
- BDF2 physical dual-time marching accepts laminar flow and all four RANS models;
  physical-time accuracy requires converging both the flow and
  turbulence defects within each physical step;
- the supplied SA model is baseline RANS; DES, transition, and rotation/curvature corrections are
  not enabled;
- unified RANS supports same-partition steady field restart and parallel VTK-HDF flow/turbulence
  output; complete BDF2-history serialization is not connected yet.

Relevant selections are:

```json
"acmSettings": {
  "rho0": 1.0,
  "beta2": 4.0,
  "alpha": 0.5,
  "riemannSolverType": "Roe"
},
"reconstructionSettings": {
  "type": "Variational",
  "enableLimiter": true,
  "limiterType": "CWBAP"
},
"timeMarchSettings": {
  "integrator": "ImplicitEulerGMRES",
  "lusgsSweeps": 2,
  "gmresSubspace": 10,
  "gmresRestarts": 3,
  "linearRelativeTolerance": 1e-6,
  "linearAbsoluteTolerance": 1e-10,
  "gmresPreconditioner": "LUSGS"
}
```

The source-level comparison and implemented integration route for WBAP/CWBAP and LU-SGS/GMRES
is documented in `docs/dev/acm_euler_reuse_comparison.md`.

## Absolute-or-relative residual convergence

`AbsoluteOrRelative` accepts a physical defect when
`R <= max(implicitTolerance, relativeTolerance * R_initial)`. The initial reference
is reset at each physical step. Flow and active RANS groups must each pass one
of their thresholds; one group's success never bypasses another group. Setting
the grouped relative coefficient to zero selects the existing scaled per-equation
controls. `Absolute` and `Relative` remain available for controlled comparisons.
The supplied cases use nonlinear absolute RMS `1e-8` and relative reduction `1e-4`.

GMRES and standalone LU-SGS check the unpreconditioned true linear RMS
`||b-A*delta||_2/sqrt(N*nVars)` before nonlinear damping. Their OR target is
`max(linearAbsoluteTolerance, linearRelativeTolerance * initialLinearRMS)`, with
case values `1e-10` and `1e-6`. GMRES checks at cycle boundaries. Standalone
LU-SGS checks after each sweep and uses `lusgsSweeps` as its maximum budget;
GMRES preconditioning still applies that fixed sweep count. The linear log
reports `converged=0` if its budget is spent without meeting either tolerance.
An inexact correction may still be used by the outer nonlinear iteration;
physical history acceptance remains controlled by `dualTimeRequireConvergence`.

The legacy JSON key `gmresRelativeTolerance` and its CLI override are translated
to `linearRelativeTolerance` on input. Conflicting old/new values are rejected.
Threshold rationale and source links are in
[`acm_residual_tolerances.txt`](../../docs/solver-guide/acm_residual_tolerances.txt).

## Converged reconstruction and steady pseudo-time efficiency

Legacy configurations retain fixed reconstruction sweeps and absolute inner tolerances.
`LoadConfiguration` inserts defaults for the following optional controls in memory.
For convergence-controlled VR, add these entries to `reconstructionSettings`:

```json
"variationalTolerance": 1e-13,
"variationalMaxIterations": 30000,
"variationalCheckInterval": 10,
"variationalRelaxation": 0.7
```

A positive `variationalTolerance` enables simultaneous relaxed updates of the
block-Jacobi-preconditioned reconstruction equation defect
`a - A_i^{-1}(sum_j B_ij a_j + b_i(a_i,U))`. Boundary contributions and the defect
check use the same, unmodified coefficient field. The legacy SOR path can evaluate
boundaries using a partially overwritten local polynomial, so simply increasing
its sweep count is not equivalent to converging this equation.

The stopping norm is the MPI-global maximum absolute defect over owned coefficients,
scaled by `sqrt(beta2/rho0)` for velocity and `beta2` for pressure. It does not depend
on the initial warm-start residual or on the update relaxation. A converged warm
start is accepted without further sweeps. Otherwise, `variationalIterations` is the
minimum sweep count before periodic checks; the maximum cap is always checked.
Exceeding the cap raises an error on all ranks instead of using an unchecked residual.
Setting the tolerance to zero restores legacy fixed sweeps. Controlled updates use
`variationalRelaxation`, independently of `vfvSettings.SORInstead`/`jacobiRelax`.

This is a reconstruction-equation tolerance, **not** a bound on the flow residual
error: mesh stretching, polynomial scaling and the spatial operator amplify errors.
Verify it by a tighter or independently converged reconstruction on the target mesh.
The example cap/tolerance are diagnostic settings, not universal production values.

For steady implicit Euler only, optional `timeMarchSettings` entries are:

```json
"steadyRelativeTolerance": 0.01,
"steadyAdaptiveCFL": true,
"steadyCFLMin": 0.1,
"steadyCFLMax": 20.0,
"steadyCFLGrowth": 1.5,
"steadyCFLReduction": 0.5
```

The inner target is `max(implicitTolerance, steadyRelativeTolerance * initialSpatialRMS)`.
Thus early pseudo-time steps can be solved inexactly while the target tightens as
the outer residual falls. The absolute tolerance remains the final accuracy floor.
The raw spatial RMS is reevaluated after the complete flow/turbulence step;
it is distinct from the implicit history defect. CFL increases only after inner
success and spatial residual reduction, by at most `steadyCFLGrowth` and the square
root of the before/after residual ratio. Inner failure or spatial residual growth
above 5% reduces CFL. Stagnation holds CFL fixed; bounds always apply. This changes
the next step and does not roll back an accepted state or certify steady convergence.

Logs expose `innerTarget`, `steadyResidual`, used/next CFL, and reconstruction sweep
count/last equation defect. `converged` still refers to the inner target, not to the
global steady solution. Steady-only controls reject explicit and BDF2 integrators;
physical dual-time accuracy is unchanged. Both steady controls default to disabled.
