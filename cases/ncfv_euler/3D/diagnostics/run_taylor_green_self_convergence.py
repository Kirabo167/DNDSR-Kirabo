"""Run the unforced compressible Taylor--Green vortex on periodic prism meshes."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[4]
CONFIG = ROOT / "cases/ncfv_euler/3D/NCFV_traditional_variational_vortex.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--levels", type=int, nargs="+", default=[10, 20, 40, 80])
    parser.add_argument("--modes", nargs="+", choices=["Traditional", "Efficient"],
                        default=["Traditional", "Efficient"])
    parser.add_argument("--weight", type=float, default=2)
    parser.add_argument("--mach", type=float, default=0.1)
    parser.add_argument("--end-time", type=float, default=0.4)
    parser.add_argument("--dt-numerator", type=float, default=0.5)
    parser.add_argument("--stencil", type=int, default=16)
    parser.add_argument("--ranks", type=int, default=8)
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    if min(args.levels) <= 0 or min(args.weight, args.mach, args.end_time,
                                     args.dt_numerator) <= 0:
        parser.error("Levels, weight, Mach, end time, and time step must be positive")
    if args.stencil < 9 or args.ranks < 1 or args.jobs < 1:
        parser.error("Invalid stencil, ranks, or jobs")

    output = args.output.resolve()
    raw = output / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    binary = args.binary.resolve()
    binary_hash = hashlib.sha256(binary.read_bytes()).hexdigest()
    config_hash = hashlib.sha256(CONFIG.read_bytes()).hexdigest()
    env = dict(os.environ, OMP_NUM_THREADS="1", DNDS_DIST_OMP_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               HWLOC_COMPONENTS="-gl,-cuda,-nvml")

    def run(case):
        level, mode = case
        name = f"tgv_prism{level}_{mode.lower()}_p{args.weight:g}_m{args.mach:g}"
        result = raw / f"{name}.json"
        manifest_file = raw / f"{name}.manifest.json"
        mesh = ROOT / f"cases/ncfv_euler/3D/iv{level}_3d_1_dnds.cgns"
        dt = args.dt_numerator / level
        steps = round(args.end_time / dt)
        if abs(steps * dt - args.end_time) >= 1e-12:
            raise ValueError(f"End time {args.end_time} is not a multiple of {dt}")
        command = ["mpirun", "--mca", "pml", "ob1", "--mca", "btl", "self,vader",
                   "--bind-to", "none", "-np", str(args.ranks), str(binary),
                   str(CONFIG), str(result), str(args.mach), str(dt),
                   str(args.end_time), str(args.stencil), mode, str(args.weight),
                   str(mesh)]
        manifest = {"command": command, "binary_sha256": binary_hash,
                    "config_sha256": config_hash,
                    "mesh_sha256": hashlib.sha256(mesh.read_bytes()).hexdigest()}
        if result.exists():
            if not manifest_file.exists() or json.loads(manifest_file.read_text()) != manifest:
                raise RuntimeError(f"Refusing stale result: {result}")
        else:
            if manifest_file.exists() and json.loads(manifest_file.read_text()) != manifest:
                raise RuntimeError(f"Refusing to replace different run: {manifest_file}")
            manifest_file.write_text(json.dumps(manifest, indent=2) + "\n")
            print(f"Start {name}", flush=True)
            with (raw / f"{name}.log").open("w") as log:
                subprocess.run(command, cwd=ROOT / "build", env=env,
                               stdout=log, stderr=subprocess.STDOUT, check=True)
        data = json.loads(result.read_text())
        if data["problem"] != "compressible_taylor_green" or data["mode"] != mode:
            raise RuntimeError(f"Unexpected output metadata: {result}")
        for rank in range(args.ranks):
            field = raw / f"{name}.json.rank{rank:04}.csv"
            if not field.exists():
                raise RuntimeError(f"Missing nodal field: {field}")
        print(f"Done {name}: N={data['nodes']}, mass drift="
              f"{data['final_conserved'][0]-data['initial_conserved'][0]:.3e}",
              flush=True)

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        list(pool.map(run, itertools.product(args.levels, args.modes)))


if __name__ == "__main__":
    main()
