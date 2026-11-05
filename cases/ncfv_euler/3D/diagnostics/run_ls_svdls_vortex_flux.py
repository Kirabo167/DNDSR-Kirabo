"""Compare production LS/SVDLS vortex errors for quadrature and edge-midpoint fluxes."""
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


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--levels", nargs="+", type=int, default=[10, 20, 40, 80])
    parser.add_argument("--methods", nargs="+", choices=["LeastSquares", "SVDLeastSquares"],
                        default=["LeastSquares", "SVDLeastSquares"])
    parser.add_argument("--fluxes", nargs="+", choices=["quadrature", "midpoint"],
                        default=["quadrature", "midpoint"])
    parser.add_argument("--distance-weight-power", type=float, default=2)
    parser.add_argument("--dt-scale", type=float, default=1)
    parser.add_argument("--ranks", type=int, default=8)
    parser.add_argument("--edge-threads", type=int, default=1)
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    if min(args.dt_scale, args.ranks, args.jobs, args.edge_threads, *args.levels) <= 0:
        parser.error("time-step scale, ranks, jobs and levels must be positive")
    if not 0 <= args.distance_weight_power <= 8:
        parser.error("distance exponent must be between zero and eight")
    binary = args.binary.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    binary_hash = digest(binary)
    config_hash = digest(CONFIG)
    base_environment = dict(os.environ, OMP_NUM_THREADS="1", DNDS_DIST_OMP_NUM_THREADS="1",
                            OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                            HWLOC_COMPONENTS="-gl,-cuda,-nvml")
    for key in ("NCFV_TRADITIONAL_EDGE_MIDPOINT", "NCFV_REFERENCE_ROE_HARTEN",
                "NCFV_REFERENCE_DOCUMENT_FLUX", "NCFV_AUDIT_FLUX_BUDGET"):
        base_environment.pop(key, None)
    base_environment.pop("NCFV_EDGE_OMP_THREADS", None)
    if args.edge_threads > 1:
        base_environment["NCFV_EDGE_OMP_THREADS"] = str(args.edge_threads)

    def run(case):
        method, flux, level = case
        name = f"prism{level}_{method}_{flux}"
        mesh = ROOT / f"cases/ncfv_euler/3D/iv{level}_3d_1_dnds.cgns"
        result_path = output / f"{name}.json"
        manifest_path = output / f"{name}.manifest.json"
        environment = dict(base_environment)
        if flux == "midpoint":
            environment["NCFV_TRADITIONAL_EDGE_MIDPOINT"] = "1"
        dt = 2 / level * args.dt_scale
        command = ["mpirun", "--mca", "pml", "ob1", "--mca", "btl", "self,vader",
                   "--bind-to", "none", "-np", str(args.ranks), str(binary),
                   str(CONFIG), str(mesh), str(result_path), "Traditional", method,
                   str(dt), "1.6", "Roe", str(args.distance_weight_power)]
        manifest = {"command": command, "binary_sha256": binary_hash,
                    "config_sha256": config_hash, "mesh_sha256": digest(mesh),
                    "midpoint": flux == "midpoint", "distance_weight_power": args.distance_weight_power}
        if args.edge_threads > 1:
            manifest["edge_threads"] = args.edge_threads
        if result_path.exists():
            if not manifest_path.exists() or json.loads(manifest_path.read_text()) != manifest:
                raise RuntimeError(f"Refusing stale result: {result_path}")
        else:
            if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
                raise RuntimeError(f"Refusing stale manifest: {manifest_path}")
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
            print(f"Start {name}", flush=True)
            with (output / f"{name}.log").open("w") as log:
                subprocess.run(command, cwd=ROOT / "build", env=environment,
                               stdout=log, stderr=subprocess.STDOUT, check=True)
        data = json.loads(result_path.read_text())
        assert data["method"] == method and data["mode"] == "Traditional"
        assert data["traditional_edge_midpoint_flux"] == (flux == "midpoint")
        assert data["configuration"]["reconstruction"]["distanceWeightPower"] == args.distance_weight_power
        assert abs(data["final"]["time"] - 1.6) < 1e-12
        final = data["final"]
        print(f"Done {name}: L2={final['rho_point_L2V']:.9e}", flush=True)
        return {"method": method, "flux": flux, "level": level, "nodes": data["nodes"],
                "dt": dt, "L1": final["rho_point_L1V"], "L2": final["rho_point_L2V"],
                "Linf": final["rho_point_Linf"], "initial_L2": data["initial"]["rho_point_L2V"],
                "mass_drift": final["mass_drift"],
                "last_step_riemann_calls": data["last_step_riemann_calls"],
                "order_L1": "", "order_L2": "", "order_Linf": "",
                "order_L2_volume_h": ""}

    cases = itertools.product(args.methods, args.fluxes, sorted(args.levels))
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        rows = list(pool.map(run, cases))
    rows.sort(key=lambda row: (row["method"], row["flux"], row["level"]))
    previous = {}
    for row in rows:
        key = row["method"], row["flux"]
        if key in previous:
            coarse = previous[key]
            for norm in ("L1", "L2", "Linf"):
                row[f"order_{norm}"] = math.log(coarse[norm] / row[norm]) / math.log(row["level"] / coarse["level"])
            row["order_L2_volume_h"] = math.log(coarse["L2"] / row["L2"]) / (math.log(row["nodes"] / coarse["nodes"]) / 3)
        previous[key] = row
    with (output / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
