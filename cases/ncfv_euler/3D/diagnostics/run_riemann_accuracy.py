"""Compare NCFV fluxes using the production SSPRK3 solver and fresh binaries."""
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
MESHES = {
    "prism": "cases/ncfv_euler/3D/iv{level}_3d_1_dnds.cgns",
    "tet": "cases/ncfv_euler/3D/generated_periodic_meshes/periodic_tet_iv{level}.cgns",
    "hex": "cases/ncfv_euler/3D/generated_periodic_meshes/periodic_hex_iv{level}.cgns",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--binary", type=Path, default=ROOT / "build/src/NCFV/ncfv_reconstruction_benchmark")
    parser.add_argument("--meshes", nargs="+", choices=MESHES, default=["prism"])
    parser.add_argument("--levels", nargs="+", type=int, default=[10, 20, 40])
    parser.add_argument("--modes", nargs="+", choices=["Traditional", "Efficient"], default=["Traditional", "Efficient"])
    parser.add_argument("--methods", nargs="+", choices=["LeastSquares", "SVDLeastSquares", "Variational"], default=["LeastSquares", "Variational"])
    parser.add_argument("--solvers", nargs="+", default=["Roe", "HLLC", "HLLEP", "HLLEP_V1", "Roe_M6"])
    parser.add_argument("--dt-scale", type=float, default=1)
    parser.add_argument("--end-time", type=float, default=1.6)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--ranks", type=int, default=4)
    parser.add_argument("--flux-budget", action="store_true")
    parser.add_argument("--distance-weight-power", type=float,
                        help="Override LS row-weight exponent; zero gives unweighted LS")
    args = parser.parse_args()
    if min(args.dt_scale, args.end_time, args.jobs, args.ranks) <= 0:
        parser.error("time, jobs, and ranks must be positive")
    if args.distance_weight_power is not None and not 0 <= args.distance_weight_power <= 8:
        parser.error("distance-weight-power must be between 0 and 8")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    binary = args.binary.resolve()
    digest = hashlib.sha256(binary.read_bytes()).hexdigest()
    environment = dict(os.environ, OMP_NUM_THREADS="1", DNDS_DIST_OMP_NUM_THREADS="1",
                       OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", HWLOC_COMPONENTS="-gl,-cuda,-nvml")
    environment.pop("NCFV_AUDIT_FLUX_BUDGET", None)
    if args.flux_budget:
        environment["NCFV_AUDIT_FLUX_BUDGET"] = "1"

    def run(case):
        mesh, level, mode, method, flux = case
        name = f"{mesh}_{level}_{mode}_{method}_{flux}"
        result = output / f"{name}.json"
        dt = 2 / level * args.dt_scale
        command = ["mpirun", "--mca", "pml", "ob1", "--mca", "btl", "self,vader",
                   "--bind-to", "none", "-np", str(args.ranks), str(binary),
                   str(ROOT / "cases/ncfv_euler/3D/NCFV_traditional_variational_vortex.json"),
                   str(ROOT / MESHES[mesh].format(level=level)), str(result), mode, method,
                   str(dt), str(args.end_time), flux]
        if args.distance_weight_power is not None:
            command.append(str(args.distance_weight_power))
        manifest = {"command": command, "binary_sha256": digest, "dt": dt, "end_time": args.end_time,
                    "flux_budget": args.flux_budget,
                    "distance_weight_power": args.distance_weight_power}
        manifest_path = output / f"{name}.manifest.json"
        if result.exists():
            if not manifest_path.exists() or json.loads(manifest_path.read_text()) != manifest:
                raise RuntimeError(f"Refusing stale result: {result}")
        else:
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
            print(f"Start {name}", flush=True)
            with (output / f"{name}.log").open("w") as log:
                subprocess.run(command, cwd=ROOT / "build", env=environment,
                               stdout=log, stderr=subprocess.STDOUT, check=True)
        raw = json.loads(result.read_text())
        if args.flux_budget and "initial_flux_budget" not in raw:
            raise RuntimeError(f"Binary did not produce the requested flux budget: {result}")
        print(f"Done {name}: {raw['final']['rho_point_L2V']:.8e}", flush=True)
        return {"mesh": mesh, "level": level, "mode": mode, "method": method, "flux": flux,
                "nodes": raw["nodes"], "dt": raw["dt"], "end_time": raw["end_time"],
                "initial_L2": raw["initial"]["rho_point_L2V"],
                "L1": raw["final"]["rho_point_L1V"], "L2": raw["final"]["rho_point_L2V"],
                "Linf": raw["final"]["rho_point_Linf"], "mass_drift": raw["final"]["mass_drift"],
                "order": ""}

    cases = itertools.product(args.meshes, args.levels, args.modes, args.methods, args.solvers)
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        rows = list(pool.map(run, cases))
    rows.sort(key=lambda r: (r["mesh"], r["mode"], r["method"], r["flux"], r["level"]))
    previous = {}
    for row in rows:
        key = tuple(row[k] for k in ("mesh", "mode", "method", "flux"))
        if key in previous:
            coarse = previous[key]
            row["order"] = math.log(coarse["L2"] / row["L2"]) / (math.log(row["nodes"] / coarse["nodes"]) / 3)
        previous[key] = row
    with (output / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
