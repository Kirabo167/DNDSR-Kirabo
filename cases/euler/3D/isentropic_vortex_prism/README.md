# CFV: two-dimensional isentropic vortex on three-dimensional Prism6 meshes

These cases use the `NS_3D` Euler/CFV solver on the existing extruded
`cases/ncfv_euler/3D/iv{10,20,40,80}_3d_1_dnds.cgns` meshes. The vortex depends on
`x,y` and is constant in `z`. The domain is `[0,10] x [0,10] x [0,4]`,
periodic in all three directions. The original grids have 1,024; 7,792;
60,832; and 479,936 Prism6 cells, respectively.

The built-in initializer `specialBuiltinInitializer=2` uses the exact
isentropic vortex with strength 5 and center `(5,5)`, background velocity
`(1,1,0)`, density and pressure 1, and `gamma=1.4`. Viscosity is set to
`1e-200`, so this is effectively an inviscid test. The source extension in
`src/Euler/EulerEvaluator.hxx` permits the same cell-average initializer for
`NS_3D`; the existing solver error output compares the reconstructed density
to the analytic solution at cell quadrature points.

`maxOrder=2` is quadratic reconstruction and targets third-order spatial
accuracy. `intOrder=3` selects the 6-point degree-4 rule on triangular faces
and the 2x2 Gauss rule on quadrilateral faces. `odeCode=2` selects SSPRK4.
The final time is `t=0.01`; the time step halves when nominal resolution
doubles. Time error therefore also decreases with refinement. The error
reported here is the volume-normalized mean `L1` error of reconstructed
density, not the error of cell means alone.

The existing NCFV meshes use boundary names `bc-2`, `bc-2-1`, `bc-3`,
`bc-3-1`, `bc-4`, `bc-4-1`. The CFV CGNS reader requires `PERIODIC_1`,
`PERIODIC_1_DONOR`, `PERIODIC_2`, `PERIODIC_2_DONOR`, `PERIODIC_3`,
`PERIODIC_3_DONOR`. `prepare_meshes.py` makes copies with those names,
preserving coordinates and connectivity. Generated meshes and solver output
are ignored by git and can be regenerated.

Run from the project root:

```sh
/home/mrz/.venvs/DNDSR-Kirabo/bin/python cases/euler/3D/isentropic_vortex_prism/prepare_meshes.py
cmake --build build -t euler3D -j8
cd build
OMP_NUM_THREADS=2 ./app/euler3D.exe ../cases/euler/3D/isentropic_vortex_prism/iv10.json
OMP_NUM_THREADS=2 ./app/euler3D.exe ../cases/euler/3D/isentropic_vortex_prism/iv20.json
OMP_NUM_THREADS=2 ./app/euler3D.exe ../cases/euler/3D/isentropic_vortex_prism/iv40.json
OMP_NUM_THREADS=2 ./app/euler3D.exe ../cases/euler/3D/isentropic_vortex_prism/iv80.json
cd ..
/home/mrz/.venvs/DNDSR-Kirabo/bin/python cases/euler/3D/isentropic_vortex_prism/analyze.py --meshes 10 20 40 80
```

`analyze.py` checks that each run reached its configured time and step count,
then computes `log(error_coarse/error_fine)/log(N_fine/N_coarse)` using the
nominal in-plane resolution. All physical error measurements are taken at
the same time. IV80 is substantially more expensive than the first three
meshes; it is optional if the available memory or runtime is insufficient.

Observed results for the three completed runs (single MPI rank, IV10/20 with
two OpenMP threads, IV40 with four):

| Mesh | Prism6 cells | Mean density L1 | Observed L1 order | Density Linf | Observed Linf order |
| --- | ---: | ---: | ---: | ---: | ---: |
| IV10 | 1,024 | 5.3793614e-4 | — | 1.0584144e-2 | — |
| IV20 | 7,792 | 4.6421161e-5 | 3.535 | 1.9085116e-3 | 2.471 |
| IV40 | 60,832 | 4.1590702e-6 | 3.480 | 1.6433642e-4 | 3.538 |

The two mean-L1 rates support the expected third-order spatial behavior in
this test. Rates slightly above three on these meshes do not establish formal
fourth order; the time error, quadrature error, and pre-asymptotic effects
also contribute. The final VTK cell fields independently show that the
nominally two-dimensional solution stayed constant across the extruded
direction: maximum same-`(x,y)` density differences were 1.33e-15,
1.78e-15, and 2.66e-15 for IV10/20/40, respectively; maximum `|w|` was
7.11e-16 across these runs.
