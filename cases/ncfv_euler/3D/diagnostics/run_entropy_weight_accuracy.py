"""Run the exact 3-D periodic entropy-wave test for weighted NCFV LS."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import itertools
import json
import math
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
    parser.add_argument("--weights", type=float, nargs="+", default=[1, 2])
    parser.add_argument("--ranks", type=int, default=8)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--end-time", type=float, default=1.6)
    parser.add_argument("--stencil", type=int, default=16)
    args = parser.parse_args()
    if not args.levels or min(args.levels) <= 0 or min(args.weights) < 0:
        parser.error("Levels and weights must be nonnegative, with positive levels")
    if args.ranks < 1 or args.jobs < 1 or args.end_time <= 0 or args.stencil < 9:
        parser.error("Invalid ranks, jobs, end time, or stencil size")

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
        level, mode, power = case
        weight_name = f"{power:g}".replace(".", "p")
        name = f"entropy_prism{level}_{mode.lower()}_p{weight_name}"
        result = raw / f"{name}.json"
        manifest_file = raw / f"{name}.manifest.json"
        mesh = ROOT / f"cases/ncfv_euler/3D/iv{level}_3d_1_dnds.cgns"
        dt = 2 / level
        steps = round(args.end_time / dt)
        if not math.isclose(steps * dt, args.end_time, abs_tol=1e-12):
            raise ValueError(f"End time {args.end_time} is not a multiple of dt={dt}")
        command = ["mpirun", "--mca", "pml", "ob1", "--mca", "btl", "self,vader",
                   "--bind-to", "none", "-np", str(args.ranks), str(binary),
                   str(CONFIG), str(result), "entropy", str(dt), str(args.end_time),
                   str(args.stencil), mode, str(power), str(mesh)]
        manifest = {"command": command, "binary_sha256": binary_hash,
                    "config_sha256": config_hash,
                    "mesh_sha256": hashlib.sha256(mesh.read_bytes()).hexdigest()}
        if result.exists():
            if not manifest_file.exists() or json.loads(manifest_file.read_text()) != manifest:
                raise RuntimeError(f"Refusing to use result with a different manifest: {result}")
        else:
            if manifest_file.exists() and json.loads(manifest_file.read_text()) != manifest:
                raise RuntimeError(f"Refusing to replace a different run: {manifest_file}")
            manifest_file.write_text(json.dumps(manifest, indent=2) + "\n")
            print(f"Start {name}", flush=True)
            with (raw / f"{name}.log").open("w") as log:
                subprocess.run(command, cwd=ROOT / "build", env=env,
                               stdout=log, stderr=subprocess.STDOUT, check=True)
        data = json.loads(result.read_text())
        if data["problem"] != "entropy" or data["mode"] != mode or data["mpi_ranks"] != args.ranks:
            raise RuntimeError(f"Unexpected result metadata: {result}")
        density_mean = data["mean_conservative"]["L2"][0]
        density_point = data["point_primitive"]["L2"][0]
        print(f"Done {name}: mean={density_mean:.8e}, point={density_point:.8e}", flush=True)
        return {"level": level, "mode": mode, "weight_power": power,
                "nodes": data["nodes"], "dt": data["dt"],
                "mean_density_L2": density_mean, "point_density_L2": density_point,
                "point_density_L1": data["point_primitive"]["L1"][0],
                "point_density_Linf": data["point_primitive"]["Linf"][0],
                "mass_drift": data["final_conserved"][0] - data["initial_conserved"][0],
                "march_seconds": data["march_seconds"], "mean_order": "", "point_order": ""}

    cases = itertools.product(args.levels, args.modes, args.weights)
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        rows = list(pool.map(run, cases))
    rows.sort(key=lambda row: (row["mode"], row["weight_power"], row["level"]))
    previous = {}
    for row in rows:
        key = row["mode"], row["weight_power"]
        if key in previous:
            coarse = previous[key]
            h_ratio = (row["nodes"] / coarse["nodes"]) ** (1 / 3)
            row["mean_order"] = math.log(coarse["mean_density_L2"] / row["mean_density_L2"]) / math.log(h_ratio)
            row["point_order"] = math.log(coarse["point_density_L2"] / row["point_density_L2"]) / math.log(h_ratio)
        previous[key] = row
    with (output / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
