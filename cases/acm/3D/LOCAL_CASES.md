# Local sphere and SUBOFF AFF8 cases

These configurations were migrated from `cases/acm3D/` to `cases/acm/3D/`.
Run the following checks from `build/`; they validate parameters without
loading meshes or advancing the solution:

```sh
./app/acm3D.exe ../cases/acm/3D/acm3D_sphere_Re100_1M.json --check-config
./app/acm3D.exe ../cases/acm/3D/acm3D_sphere_Re100_1M_restart10000_add20000.json --check-config
./app/acm2EQ3D.exe ../cases/acm/3D/acm3D_suboff_AFF8_Re12000000_SST_uniform_5000.json --check-config
./app/acm2EQ3D.exe ../cases/acm/3D/acm3D_suboff_AFF8_Re12000000_SST_uniform_20000_out200.json --check-config
./app/acm2EQ3D.exe ../cases/acm/3D/acm3D_suboff_AFF8_Re12000000_SST_uniform_30000_out200.json --check-config
```

The sphere cases remain laminar and use `acm3D`. The AFF8 cases retain their
filenames but now require `acm2EQ3D`, `ConstantDensity2EQ3D`, and six state
variables. Their initial state is `[u,v,w,p,rho*k,rho*omega]`; with `rho0=1`,
the two added entries preserve the original `k=1e-5`, `omega=1` initialization.
The unified RANS implementation requires `transportSubsteps=1`; the original
`transportTimeScale=0.1` is retained. This is a change to the turbulence update
algorithm, so the migrated run is not guaranteed to reproduce the old
segregated solver's iteration history.

Mesh, output, and restart paths under `/fs2/home/mrz/DNDSR-Kirabo/` are retained
for the original cluster. Those mesh and restart files are not available on
this workstation; passing `--check-config` does not check their availability.
The sphere restart still requires the original 2240-rank partition and starts
from completed step 10000. Its Slurm check script is
`scripts/check_sphere_restart_cp6.sh`, now using the migrated configuration path.
