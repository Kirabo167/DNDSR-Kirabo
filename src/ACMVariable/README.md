# Variable-density ACM module

Author and maintainer: Runzhi Ma (2026-09-04)

This directory is an independent incompressible variable-density artificial-
compressibility solver. It does not modify `Euler` or constant-density `ACM`.
It is launched through the unified `build/app/euler.exe`; the case JSON selects
`solver.type="ACMVariable"` and the 2-D or 3-D model.
The solved flow state is `U=[rho,rho*u,rho*v,rho*w,p]`; a 2-D mesh uses the
same five-entry storage and keeps the out-of-plane equation for modular 2-D/3-D
assembly. Physical-time histories contain only density and momentum. Pressure
is an algebraic Lagrange multiplier for the discrete divergence constraint.

## File map

- `ACM.cpp/.hpp`, `ACMState.hpp`, `ACMFlux.hpp`: state transforms, exact physical
  flux Jacobian, general-alpha Gamma/eigensystem, Roe/Rusanov flux, collision-safe
  matrix absolute value, viscosity, boundaries and JSON loading.
- `ACMEvaluator.hpp/.hxx`: MPI/CFV face reconstruction, local-extrema and
  characteristic WBAP/CWBAP limiting, residual, CFL and first-order implicit
  face linearization.
- `ACMTime.cpp/.hpp`, `ACMBDF2.cpp/.hpp`, `ACMPhysicalTime.hxx`: SSPRK3,
  backward Euler, LU-SGS/GMRES, BDF1/BDF2, ESDIRK/SDIRK/trapezoidal/Hermite
  adapters. Physical time uses a DAE mass matrix `diag(1,1,1,1,0)`.
- `ACMTurbulence*.{hpp,hxx,cpp}`: segregated laminar, SA, Wilcox k-omega,
  SST and project-compatible realizable k-epsilon transport. The conservative
  histories are `rho*phi`, and the mass flux is the flow solver's face mass flux.
- `ACMSolver.hpp/.hxx`: mesh/parallel setup and short modular solve driver.
- `acm2D.cpp`, `acm3D.cpp`: template instantiations only.

The implementation derivation, source-document errata, method matrix and
verification procedure are in `docs/reports/acm_variable_density_implementation.pdf`.
