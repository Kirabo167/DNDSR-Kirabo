"""Run the complete static periodic reference chain on the original prism meshes."""
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--levels", type=int, nargs="+", default=[10, 20, 40, 80])
    parser.add_argument("--problems", nargs="+", choices=["entropy", "entropy_xy", "vortex", "tgv"], default=["entropy", "vortex"])
    parser.add_argument("--write-fields", action="store_true")
    parser.add_argument("--entropy-moments", action="store_true")
    parser.add_argument("--surface-quadrature", action="store_true")
    parser.add_argument("--uncached-quadrature", action="store_true")
    parser.add_argument("--sweeps", type=int, default=30)
    parser.add_argument("--ranks", type=int, default=8)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--end-time", type=float, default=1.6)
    parser.add_argument("--dt-numerator", type=float, default=2)
    args = parser.parse_args()
    if args.uncached_quadrature and not args.surface_quadrature:
        parser.error("Uncached quadrature requires --surface-quadrature")
    if args.surface_quadrature and args.entropy_moments:
        parser.error("Surface quadrature and entropy moment correction are separate diagnostics")
    if args.entropy_moments and any(p not in ("entropy", "entropy_xy") for p in args.problems):
        parser.error("Entropy moment correction only supports constant-velocity entropy waves")
    if min(args.levels) <= 0 or min(args.sweeps, args.ranks, args.jobs) < 1:
        parser.error("Levels, sweeps, ranks and jobs must be positive")
    if min(args.end_time, args.dt_numerator) <= 0:
        parser.error("Time inputs must be positive")
    raw = args.output.resolve() / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    binary = args.binary.resolve()
    digest = hashlib.sha256(binary.read_bytes()).hexdigest()
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", HWLOC_COMPONENTS="-gl,-cuda,-nvml")
    env.pop("HOCFV_AUDIT_RHS", None)
    env.pop("HOCFV_WRITE_FIELDS", None)
    env.pop("HOCFV_ENTROPY_MOMENT_CORRECTION", None)
    env.pop("HOCFV_SURFACE_QUADRATURE", None)
    env.pop("HOCFV_UNCACHED_QUADRATURE", None)
    if args.write_fields:
        env["HOCFV_WRITE_FIELDS"] = "1"
    if args.entropy_moments:
        env["HOCFV_ENTROPY_MOMENT_CORRECTION"] = "1"
    if args.surface_quadrature:
        env["HOCFV_SURFACE_QUADRATURE"] = "1"
    if args.uncached_quadrature:
        env["HOCFV_UNCACHED_QUADRATURE"] = "1"

    def run(case):
        level, problem = case
        mesh = ROOT / f"cases/ncfv_euler/3D/iv{level}_3d_1_dnds.cgns"
        name = f"{problem}_iv{level}"
        result = raw / f"{name}.json"
        manifest_path = raw / f"{name}.manifest.json"
        dt = args.dt_numerator / level
        command = ["mpirun", "--mca", "pml", "ob1", "--mca", "btl", "self,vader",
                   "--bind-to", "none", "-np", str(args.ranks), str(binary), str(mesh),
                   str(result), problem, str(dt), str(args.end_time), str(args.sweeps)]
        manifest = {"command": command, "binary_sha256": digest,
                    "write_fields": args.write_fields,
                    "mesh_sha256": hashlib.sha256(mesh.read_bytes()).hexdigest()}
        if args.entropy_moments:
            manifest["entropy_moments"] = True
        if args.surface_quadrature:
            manifest["surface_quadrature"] = True
            manifest["quadrature_cached"] = not args.uncached_quadrature
        if result.exists():
            if not manifest_path.exists() or json.loads(manifest_path.read_text()) != manifest:
                raise RuntimeError(f"Different existing run: {result}")
        else:
            if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
                raise RuntimeError(f"Different partial run: {manifest_path}")
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
            print(f"Start {name}, sweeps={args.sweeps}, dt={dt}", flush=True)
            with (raw / f"{name}.log").open("w") as log:
                subprocess.run(command, cwd=ROOT / "build", env=env, check=True,
                               stdout=log, stderr=subprocess.STDOUT)
        data = json.loads(result.read_text())
        if data.get("entropy_moment_correction", False) != args.entropy_moments:
            raise RuntimeError(f"Unexpected entropy moment correction: {result}")
        if data.get("surface_flux_mode", "midpoint") != ("surface6" if args.surface_quadrature else "midpoint"):
            raise RuntimeError(f"Unexpected surface flux mode: {result}")
        if args.surface_quadrature and data["quadrature_cached"] != (not args.uncached_quadrature):
            raise RuntimeError(f"Unexpected quadrature cache mode: {result}")
        if data["problem"] != problem or data["sweeps"] != args.sweeps or data["ranks"] != args.ranks:
            raise RuntimeError(f"Unexpected reference metadata: {result}")
        if data["rhs_evaluations"] != 3 * data["steps"] or data["variational_weight"] != 1:
            raise RuntimeError(f"Unexpected reconstruction or RK chain: {result}")
        message = f"Done {name}: residual={data['last_coupled_residual']:.3e}"
        if data["density_L2"] is not None:
            message += f", L2={data['density_L2']:.9e}"
        print(message, flush=True)
        return {"problem": problem, "level": level, "nodes": data["nodes"], "dt": dt,
                "sweeps": args.sweeps, "L1": data["density_L1"], "L2": data["density_L2"],
                "Linf": data["density_Linf"], "L2_high_reference": data["density_L2_higher_reference"],
                "order_L1": "", "order_L2": "", "order_Linf": "",
                "coupled_residual": data["last_coupled_residual"],
                "fallbacks": data["mean_fallback_count"], "mass_drift": data["mass_drift"]}

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        rows = list(pool.map(run, itertools.product(args.levels, args.problems)))
    rows.sort(key=lambda row: (row["problem"], row["level"]))
    previous = {}
    for row in rows:
        if row["problem"] in previous:
            coarse = previous[row["problem"]]
            for field in ("L1", "L2", "Linf"):
                if row[field] is not None and coarse[field] is not None:
                    row[f"order_{field}"] = math.log(coarse[field] / row[field]) / math.log(row["level"] / coarse["level"])
        previous[row["problem"]] = row
    with (args.output / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, rows[0].keys())
        writer.writeheader(); writer.writerows(rows)


if __name__ == "__main__":
    main()
