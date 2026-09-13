<!--
File description: implementation status and integration boundary of the initial ACM module.
Modifier: Runzhi Ma
Last modified: 2026-09-02
-->

# ACM high-order initial solver

This directory contains an independent constant-density ACM solver for two- and
three-dimensional unstructured meshes. It reuses `Geom`, `CFV`, and the existing
MPI/OpenMP array infrastructure, but it does not modify or inherit the Euler equation model.

Implemented now:

- state layout `[u,v,w,p]`;
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
- arbitrary-order CFV variational reconstruction selected by `vfvSettings.maxOrder`;
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
- configurable parallel VTK-HDF cell-field output for velocity and pressure;
- segregated limited second-order turbulence transport with wall distance, MPI ghost exchange,
  positivity-bounded SSPRK3 substeps, and frozen eddy-viscosity coupling to the flow equations;
- DNDS configuration registration, self-contained single-case JSON loading, CLI overrides, and schema output.

Application organization follows the compact Euler entry-point style:

- `app/ACM/ACM.cpp`: short default 3-D launcher;
- `app/ACM/acm2D.cpp` and `app/ACM/acm3D.cpp`: short dimension-specific launchers;
- `SingleBlockApp.hpp`: shared CLI and configuration workflow;
- `ACMSolver.*`: mesh/reconstruction/time-loop assembly;
- `ACMEvaluator.*`: high-order spatial residual and frozen-reconstruction Jacobian;
- `ACMTurbulence.*`: model-local viscosity, diffusion, source, and boundary kernels;
- `ACMTurbulenceTransport.*`: dimension-generic finite-volume transport and flow coupling;
- `acm2D.cpp` and `acm3D.cpp`: explicit template instantiations for four ACM variables.

Case configuration uses one file per case. `cases/acm2D/acm2D.json` and
`cases/acm3D/acm3D.json` each contain the complete physical, numerical, mesh, reconstruction,
boundary, and initial-state configuration. The application does not search for or merge an
adjacent base file. A user-supplied positional JSON path completely selects the case; `-k/-v`
overrides remain available for short parameter studies.

Turbulence is modular and does not enlarge the ACM flow state. The flow solver always stores
`[u,v,w,p]`; a separate distributed two-entry field stores only the active turbulence variables:

- `Laminar`: no turbulence equation and `mu_t=0`;
- `SpalartAllmaras`: `[nuTilde, unused]`;
- `KOmegaWilcox` and `KOmegaSST`: `[k, omega]`;
- `RealizableKEpsilon`: `[k, epsilon]`.

The same implementation is instantiated for both two and three dimensions. It reuses Geom wall
distance/mesh metrics, CFV face quadrature, and the existing MPI arrays, but it neither includes,
links, nor modifies the Euler module. At each flow residual evaluation, the transport module freezes
face `mu_t`; the flow viscous stress, viscous CFL radius, and frozen 4x4 implicit operator then use
`mu+mu_t`. After each completed flow pseudo-time step, the turbulence field advances segregatedly
with positivity-bounded SSPRK3 substeps. Therefore LU-SGS/GMRES remains the four-variable ACM flow
solve; turbulence source Jacobians are not inserted into that implicit system.

A non-laminar model requires `acmSettings.enableViscousFlux=true` and positive
`acmSettings.dynamicViscosity`. A typical three-dimensional SST selection is:

```json
"turbulenceSettings": {
  "model": "KOmegaSST",
  "initialValue": [0.001, 10.0],
  "farFieldValue": [0.001, 10.0],
  "minimumValue": [1e-12, 1e-10],
  "maximumValue": [1000000.0, 1000000000000.0],
  "maximumEddyViscosityRatio": 100000.0,
  "wallOmegaCoefficient": 800.0,
  "enableSourceTerms": true,
  "secondOrderReconstruction": true,
  "transportSubsteps": 4,
  "transportTimeScale": 0.25,
  "wallDistanceMethod": 1,
  "wallDistanceExecution": 0,
  "wallDistanceSubdivide": 0,
  "minimumWallDistance": 1e-10,
  "wallDistanceVerbose": 0
}
```

`initialValue` and `farFieldValue` use the model-specific layouts above. `maximumValue` bounds the
explicit stages independently of `maximumEddyViscosityRatio`, which caps only `mu_t/mu`. Increasing
`transportSubsteps` or decreasing `transportTimeScale` is the first stability adjustment for stiff
wall-omega cases. For high-order RANS calculations, enabling the main ACM reconstruction limiter is
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
pressure. CFL controls continue to set only the inner pseudo-time step.

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
  "implicitTolerance": 1e-10
}
```

See `docs/solver-guide/acm_bdf2_dual_time_zh.md` for the governing defect, implicit matrix, source
mapping, and usage details.

Current initial-version limits:

- periodic translations are configurable, while rotational periodic setup is not exposed yet;
- WBAP/CWBAP requires `Variational` reconstruction;
- LU-SGS/GMRES uses a first-order frozen face linearization as the implicit operator while the
  nonlinear residual retains the selected high-order reconstruction;
- turbulence transport is segregated and explicit even when the four-variable flow integrator is
  implicit; no coupled turbulence Jacobian or implicit turbulence source linearization is present;
- BDF2 physical dual-time marching currently accepts `Laminar` only; turbulence physical-time
  histories and a coupled/segregated unsteady update have not yet been implemented;
- the supplied SA model is baseline RANS; DES, transition, and rotation/curvature corrections are
  not enabled;
- restart output and BDF2-history serialization are not connected yet; flow-field output currently
  contains cell-centered velocity and pressure in parallel VTK-HDF format.

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
  "gmresRelativeTolerance": 1e-6,
  "gmresPreconditioner": "LUSGS"
}
```

The source-level comparison and implemented integration route for WBAP/CWBAP and LU-SGS/GMRES
is documented in `docs/dev/acm_euler_reuse_comparison.md`.

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
The raw spatial RMS is reevaluated after the step and any segregated turbulence update;
it is distinct from the implicit history defect. CFL increases only after inner
success and spatial residual reduction, by at most `steadyCFLGrowth` and the square
root of the before/after residual ratio. Inner failure or spatial residual growth
above 5% reduces CFL. Stagnation holds CFL fixed; bounds always apply. This changes
the next step and does not roll back an accepted state or certify steady convergence.

Logs expose `innerTarget`, `steadyResidual`, used/next CFL, and reconstruction sweep
count/last equation defect. `converged` still refers to the inner target, not to the
global steady solution. Steady-only controls reject explicit and BDF2 integrators;
physical dual-time accuracy is unchanged. Both steady controls default to disabled.
