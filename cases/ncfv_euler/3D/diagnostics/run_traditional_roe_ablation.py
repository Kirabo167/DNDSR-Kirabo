"""Compare traditional NCFV reconstruction with and without its Roe correction."""

import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parents[4]
CONFIG = ROOT / "cases/ncfv_euler/3D/NCFV_traditional_variational_vortex.json"
PROBE = Path("/tmp/ncfv_traditional_roe_ablation_20260928")
BASELINES = {
    "ls": ROOT / "docs/reports/ncfv_ls_svdls_t1p6_dt_20260924/metrics.csv",
    "variational": (ROOT / "docs/reports/ncfv_traditional_variational_w5_t1p6_dt_20260924"
                    / "metrics.csv"),
}
DEFAULT_OUTPUTS = {
    "ls": ROOT / "docs/reports/ncfv_traditional_ls_roe_ablation_t1p6_20260928",
    "variational": ROOT / "docs/reports/ncfv_traditional_variational_roe_ablation_t1p6_20260928",
}
METHODS = {"ls": "LeastSquares", "variational": "Variational"}
LEVELS = {10: (1, 0.2), 20: (4, 0.1), 40: (8, 0.05), 80: (16, 0.025)}
MESHES = {
    "tet": "cases/ncfv_euler/3D/generated_periodic_meshes/periodic_tet_iv{level}.cgns",
    "prism": "cases/ncfv_euler/3D/iv{level}_3d_1_dnds.cgns",
    "hex": "cases/ncfv_euler/3D/generated_periodic_meshes/periodic_hex_iv{level}.cgns",
}


def summarize(output: Path, baseline: dict, method: str) -> None:
    rows = []
    for mesh in MESHES:
        previous = None
        for level in LEVELS:
            path = output / f"{mesh}_iv{level}_alpha0.json"
            if not path.exists():
                continue
            result = json.loads(path.read_text())
            full = baseline[(mesh, level)]
            center_error = result["transient"]["rho_point_L2V"]
            full_error = float(full["rho_point_L2V_final"])
            h = (400 / result["nodes"]) ** (1 / 3)
            order = (math.log(previous[1] / center_error) /
                     math.log(previous[0] / h)) if (
                         previous and previous[2] == level // 2) else ""
            full_order = (math.log(previous[3] / full_error) /
                          math.log(previous[0] / h)) if (
                              previous and previous[2] == level // 2) else ""
            rows.append({
                "mesh_type": mesh,
                "level": level,
                "method": method,
                "nodes": result["nodes"],
                "mpi_ranks": result["mpi_ranks"],
                "dt": result["dt"],
                "steps": result["steps"],
                "end_time": result["final_time"],
                "h": h,
                "full_roe_rho_point_L2V": full_error,
                "no_roe_rho_point_L2V": center_error,
                "error_ratio_no_roe_to_full": center_error / full_error,
                "error_reduction_fraction": 1 - center_error / full_error,
                "observed_order_full_roe": full_order,
                "observed_order_no_roe": order,
                "rho_mean_L2V_no_roe": result["transient"]["rho_mean_L2V"],
                "mass_drift_no_roe": result["transient"]["mass_drift"],
                "rhs_decomposition_max_mismatch_t0":
                    result["initial_static"]["rhs_decomposition_max_mismatch"],
            })
            previous = (h, center_error, level, full_error)
    if rows:
        with (output / "metrics.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    print(f"Completed {len(rows)}/12 cases", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, default="ls")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--meshes", nargs="+", choices=MESHES, default=list(MESHES))
    parser.add_argument("--levels", type=int, nargs="+", choices=LEVELS,
                        default=list(LEVELS))
    args = parser.parse_args()
    output = (args.output or DEFAULT_OUTPUTS[args.method]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with BASELINES[args.method].open() as stream:
        baseline = {(row["mesh_type"], int(row["level"])): row
                    for row in csv.DictReader(stream)
                    if row["mode"] == "traditional" and row["method"] == args.method}
    if len(baseline) != len(MESHES) * len(LEVELS) or not PROBE.exists():
        parser.error("Traditional baseline CSV or compiled Roe probe is incomplete")
    environment = os.environ.copy()
    environment.update({"HWLOC_COMPONENTS": "-gl,-cuda,-nvml",
                        "OMP_NUM_THREADS": "1", "DNDS_DIST_OMP_NUM_THREADS": "1",
                        "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    for mesh in args.meshes:
        for level in args.levels:
            ranks, dt = LEVELS[level]
            name = f"{mesh}_iv{level}_alpha0"
            result = output / f"{name}.json"
            if result.exists():
                print(f"Skip {name}", flush=True)
                continue
            mesh_path = ROOT / MESHES[mesh].format(level=level)
            command = ["mpirun", "--mca", "pml", "ob1", "--mca", "btl",
                       "self,vader", "--bind-to", "core", "--map-by", "core",
                       "-np", str(ranks), str(PROBE), str(CONFIG), str(result),
                       METHODS[args.method], "0", str(dt), "1.6", "0", "Traditional",
                       str(mesh_path)]
            print(f"Start {name}: {ranks} ranks, dt={dt}", flush=True)
            start = time.perf_counter()
            with (output / f"{name}.log").open("w") as log:
                status = subprocess.run(command, cwd=ROOT / "build", env=environment,
                                        stdout=log, stderr=subprocess.STDOUT)
            elapsed = time.perf_counter() - start
            if status.returncode:
                print(f"FAILED {name}: exit {status.returncode}, {elapsed:.1f}s",
                      flush=True)
            else:
                raw = json.loads(result.read_text())
                print(f"Done {name}: E2={raw['transient']['rho_point_L2V']:.6e}, "
                      f"{elapsed:.1f}s", flush=True)
            summarize(output, baseline, args.method)
    summarize(output, baseline, args.method)


if __name__ == "__main__":
    main()
