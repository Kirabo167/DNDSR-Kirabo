"""Run supported NCFV vortex comparisons and write raw JSON/CSV."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "cases/ncfv_euler/3D/NCFV_traditional_variational_vortex.json"
PROBE = ROOT / "build/src/NCFV/ncfv_reconstruction_benchmark"
DEFAULT_OUTPUT = ROOT / "docs/reports/ncfv_reconstruction_resource_accuracy_20260924"
LEVELS = {10: (1, 0.1), 20: (4, 0.05), 40: (8, 0.025), 80: (16, 0.0125)}
MESHES = {
    "tet": "cases/ncfv_euler/3D/generated_periodic_meshes/periodic_tet_iv{level}.cgns",
    "prism": "cases/ncfv_euler/3D/iv{level}_3d_1_dnds.cgns",
    "hex": "cases/ncfv_euler/3D/generated_periodic_meshes/periodic_hex_iv{level}.cgns",
}
MODES = {"traditional": "Traditional", "efficient": "Efficient"}
METHODS = {"ls": "LeastSquares", "svdls": "SVDLeastSquares", "variational": "Variational"}


def summarize(output: Path, expected: int = 72) -> None:
    rows = []
    for mesh in MESHES:
        for mode in MODES:
            for method in METHODS:
                previous = None
                previous_final = None
                for level in LEVELS:
                    name = f"{mesh}_iv{level}_{mode}_{method}"
                    path = output / f"{name}.json"
                    if not path.exists():
                        continue
                    raw = json.loads(path.read_text())
                    error = raw.get("initial", raw["final"])["rho_point_L2V"]
                    final_error = raw["final"]["rho_point_L2V"]
                    h = (400 / raw["nodes"]) ** (1 / 3)
                    order = (math.log(previous[1] / error) /
                             math.log(previous[0] / h)) if previous else ""
                    final_order = (math.log(previous_final[1] / final_error) /
                                   math.log(previous_final[0] / h)) if (
                                       previous_final and
                                       math.isclose(previous_final[2], raw["end_time"],
                                                    abs_tol=1e-12)) else ""
                    rows.append({
                        "mesh_type": mesh, "level": level, "mode": mode,
                        "method": method, "nodes": raw["nodes"],
                        "cells": raw["cells"], "mpi_ranks": raw["mpi_ranks"],
                        "dt": raw["dt"], "steps": raw["steps"],
                        "end_time": raw["end_time"], "h": h,
                        "rho_point_L2V_t0": error,
                        "rho_point_L1V_final": raw["final"]["rho_point_L1V"],
                        "rho_point_L2V_final": final_error,
                        "rho_point_Linf_final": raw["final"]["rho_point_Linf"],
                        "entropy_L1V_final": raw["final"]["entropy_L1V"],
                        "observed_order_t0": order,
                        "observed_order_final": final_order,
                        "mass_drift": raw["final"].get("mass_drift", ""),
                        "initialization_seconds": raw["initialization_seconds"],
                        "march_seconds": raw["march_seconds"],
                        "last_step_core_seconds": raw.get("last_step_core_seconds", ""),
                        "initialization_peak_rss_sum_mib": raw["initialization_peak_rss_sum_mib"],
                        "final_peak_rss_sum_mib": raw["final_peak_rss_sum_mib"],
                    })
                    previous = (h, error)
                    previous_final = (h, final_error, raw["end_time"])
    if rows:
        with (output / "metrics.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        by_case = {(row["mesh_type"], row["mode"], row["method"], row["level"]): row
                   for row in rows}
        order_rows = []
        final_order_rows = []
        for mesh in MESHES:
            for mode in MODES:
                for method in METHODS:
                    if not any((mesh, mode, method, level) in by_case for level in LEVELS):
                        continue
                    result = {"mesh_type": mesh, "mode": mode, "method": method}
                    final_result = result.copy()
                    for coarse, fine in ((10, 20), (20, 40), (40, 80)):
                        lower = by_case.get((mesh, mode, method, coarse))
                        upper = by_case.get((mesh, mode, method, fine))
                        column = f"order_iv{coarse}_iv{fine}"
                        result[column] = (math.log(lower["rho_point_L2V_t0"] /
                                                   upper["rho_point_L2V_t0"]) /
                                          math.log(lower["h"] / upper["h"])) if lower and upper else ""
                        final_result[column] = (
                            math.log(lower["rho_point_L2V_final"] /
                                     upper["rho_point_L2V_final"]) /
                            math.log(lower["h"] / upper["h"])) if (
                                lower and upper and
                                math.isclose(lower["end_time"], upper["end_time"],
                                             abs_tol=1e-12)) else ""
                    order_rows.append(result)
                    final_order_rows.append(final_result)
        with (output / "convergence_orders.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=order_rows[0].keys())
            writer.writeheader()
            writer.writerows(order_rows)
        with (output / "convergence_orders_final.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=final_order_rows[0].keys())
            writer.writeheader()
            writer.writerows(final_order_rows)
    print(f"Completed {len(rows)}/{expected} runs; summary: {output / 'metrics.csv'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--levels", type=int, nargs="+", choices=LEVELS, default=list(LEVELS))
    parser.add_argument("--meshes", nargs="+", choices=MESHES, default=list(MESHES))
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--one-step", action="store_true",
                        help="Run one SSPRK3 step per case and compare t=0 reconstruction errors")
    parser.add_argument("--time-step-scale", type=float, default=1.0,
                        help="Multiply the four default time steps by this positive factor")
    parser.add_argument("--end-time", type=float, default=1.6,
                        help="Common final time for transient comparisons")
    parser.add_argument("--ranks", type=int,
                        help="Override the default MPI rank count for every selected level")
    args = parser.parse_args()
    if (args.time_step_scale <= 0 or args.end_time <= 0 or
            (args.ranks is not None and args.ranks <= 0)):
        parser.error("time-step-scale, end-time and ranks must be positive")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update({"OMP_NUM_THREADS": "1", "DNDS_DIST_OMP_NUM_THREADS": "1",
                        "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                        "HWLOC_COMPONENTS": "-gl,-cuda,-nvml"})
    expected = (len(args.meshes) * len(args.levels) *
                len(args.modes) * len(args.methods))
    for mesh in args.meshes:
        for level in args.levels:
            ranks, dt = LEVELS[level]
            if args.ranks is not None:
                ranks = args.ranks
            dt *= args.time_step_scale
            mesh_path = ROOT / MESHES[mesh].format(level=level)
            for mode in args.modes:
                for method in args.methods:
                    name = f"{mesh}_iv{level}_{mode}_{method}"
                    result = output / f"{name}.json"
                    if result.exists():
                        print(f"Skip {name}", flush=True)
                        continue
                    command = ["mpirun", "--mca", "pml", "ob1", "--mca", "btl",
                               "self,vader", "--bind-to", "core", "--map-by", "core",
                               "-np", str(ranks), str(PROBE), str(BASE), str(mesh_path),
                               str(result), MODES[mode], METHODS[method], str(dt),
                               str(dt if args.one_step else args.end_time)]
                    print(f"Start {name} ({ranks} ranks, dt={dt})", flush=True)
                    with (output / f"{name}.log").open("w") as log:
                        status = subprocess.run(command, cwd=ROOT / "build", env=environment,
                                                stdout=log, stderr=subprocess.STDOUT)
                    if status.returncode:
                        print(f"FAILED {name}: exit {status.returncode}", flush=True)
                    else:
                        raw = json.loads(result.read_text())
                        print(f"Done {name}: {raw['march_seconds']:.2f}s, "
                              f"{raw['final']['rho_point_L2V']:.6e} final L2, "
                              f"{raw['final_peak_rss_sum_mib']:.1f} MiB", flush=True)
                    summarize(output, expected)
    summarize(output, expected)


if __name__ == "__main__":
    main()
