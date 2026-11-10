# CPU/GPU synchronization with main, 2026-11-10

Local CPU reference: commit `03d020f`, based on `kirabo/main` `88635e1`.
GPU branch: `GPUACM`, merging that local reference while retaining local restart,
single-card capacity, steady local-time-step, audit and postprocessing work.
The user clarified that only `GPUACM` is to be committed and published for this
delivery. The pure-CPU GitHub `main` branch must remain unchanged; the local CPU
reference commit is included only through the GPU branch's merge history.
The previous CPU snapshots and all production fields remain available.

## Main changes reviewed

1. `f11d20a`: coupled frozen turbulence transport face blocks, flow-flux/scalar
   coupling, stable SST destruction Jacobian, equilibrated block solves,
   equation-wise linear convergence, staged restart and optional RANS acceleration.
2. `88635e1`: original SST-1994 cross-diffusion floor `1e-20`, exact zero wall
   eddy viscosity, Euler omega coefficient blending and k-epsilon indexing.
3. Main also removes obsolete SUBOFF mesh experiment scripts and configurations.
   These were unmodified locally; their old versions remain in Git history.

## GPU implementation

The ACMGPU host reference incorporates the CPU changes while keeping explicit
resident dispatch and its prohibition on host CFD residuals in CUDA mode.
The device closure uses `1e-20` and exact zero no-slip eddy viscosity.
`SSTImplicitFaces` assembles frozen advection/diffusion blocks and face upwind
states. The coupled operator includes neighbour scalar corrections, derivatives
of the shared continuity flux and physical-BDF pseudo-time pressure mass.
Device forward/backward scalar SGS follows the same ordering as the CPU
preconditioner. Its schedule is captured once in a CUDA graph; correction halos
are exchanged on the normal device/MPI path.

The source iteration matrix freezes production/blending and treats destruction
implicitly; the nonlinear physical source is unchanged. BDF startup budgets are
supported on CUDA. `initializeBDF2History` is accepted alongside the existing
GPU `resetBDFHistory`; field-only restarts explicitly perform BE before BDF2.
The verification case explicitly disables Anderson, which older GPU versions
silently ignored even when its JSON value was nonzero.

CPU-only Anderson, mass continuation and matrix-free GMRES remain unavailable
on CUDA and are rejected explicitly. This update does not claim that every
optional CPU accelerator has been implemented on the device.

## Verification

Results and exact executable hashes are stored locally under
`data/out/project_sync_20261110/`:

- CPU: 27/27 ACM, ACMVariable, Euler RANS and NCFV CTests passed, including
  relevant MPI runs at 1, 2, 4 and 8 ranks.
- Four additional 2D/3D CFV reconstruction CTests passed at 1/2 ranks after
  linking the existing mesh fixtures into the isolated CPU worktree.
- 3D SST/BDF2: 78/78 CPU/host/CUDA records passed across 13 modes and 1/2 ranks;
  maximum scaled field error `2.6645352591003757e-15`.
- 2D SST/BDF2: 18/18 records passed for wall, three linear sweeps and density
  scaling at 1/2 ranks; maximum error `2.831068712794149e-15`.
- Six further CPU/host/CUDA records passed for the extended first-step RANS
  iteration budget at 1/2 ranks; maximum error `6.938893903907228e-16`.
- Steady SST: five checks passed; host/CUDA error `1.6414924974839096e-12`,
  changing unused physical dt has exactly zero effect, and convergence-window
  and nonmonotone line-search controls passed.
- Field restart: four host/CUDA checks passed at 1/2 ranks, including exact
  initial-field import, changed dt, correct absolute clocks and BE/BDF2 orders.
- CUDA analytic closure probes passed. Nsight recorded 364 device kernels in
  four physical steps, zero numerical-field host transfers, zero step-time
  device allocations and zero single-rank halo staging. Compute Sanitizer
  reported zero errors.
- The new CPU snapshot verifies 63 source files and nine local artifacts.

The two early test directories whose names include `before_link_invalid` were
run before the 3D binary had finished linking and are not validation evidence.
`steady_rejected_unsupported_config` records the expected rejection of the
previously ignored nonzero Anderson setting. Final evidence is in the named
directories above without those suffixes.

## Remaining SST findings

1. The previously diagnosed mirrored diffusion ghost is still reused as the
   incoming advective turbulence state in CPU main and CUDA. Eight direct
   kernel probes still reproduce that boundary issue; it was not fixed by the
   two main commits. Matching CPU/GPU fields does not rule out a shared defect.
2. The existing nonorthogonal scalar diffusion operator can violate a discrete
   maximum principle. The analytical counterexample still reproduces. Earlier
   SUBOFF diagnosis also identified lower-bound clipping and a concentrated
   omega residual. The new implicit operator has not yet been shown to resolve
   those spatial/discrete problems on the complete SUBOFF grid.

The full SUBOFF calculation remains stopped at saved iteration 15850. No
production run was resumed and no convergence/physical-validity claim is made.
