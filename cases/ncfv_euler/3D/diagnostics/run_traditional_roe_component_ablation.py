"""Run one-component Roe-correction ablations for traditional NCFV + LS."""

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
PROBE = Path("/tmp/ncfv_roe_component_probe_lean_20260928")
OUTPUT = ROOT / "docs/reports/ncfv_traditional_ls_roe_component_ablation_t1p6_20260928"
PREVIOUS = ROOT / "docs/reports/ncfv_traditional_ls_roe_ablation_t1p6_20260928/metrics.csv"
LEVELS = {10: (1, 0.2), 20: (4, 0.1), 40: (8, 0.05), 80: (16, 0.025)}
MESHES = {
    "tet": "cases/ncfv_euler/3D/generated_periodic_meshes/periodic_tet_iv{level}.cgns",
    "prism": "cases/ncfv_euler/3D/iv{level}_3d_1_dnds.cgns",
    "hex": "cases/ncfv_euler/3D/generated_periodic_meshes/periodic_hex_iv{level}.cgns",
}
COMPONENTS = ("full", "rho", "mx", "my", "mz", "E")


def raw_path(output: Path, mesh: str, level: int, component: str) -> Path:
    return output / f"{mesh}_iv{level}_{component}.json"


def summarize(output: Path, meshes: tuple[str, ...]) -> None:
    with PREVIOUS.open() as stream:
        previous = {(r["mesh_type"], int(r["level"])): r
                    for r in csv.DictReader(stream)}
    rows = []
    for mesh in meshes:
        for component in COMPONENTS:
            earlier = None
            for level in LEVELS:
                path = raw_path(output, mesh, level, component)
                reference_path = raw_path(output, mesh, level, "full")
                if not path.exists() or not reference_path.exists():
                    earlier = None
                    continue
                result = json.loads(path.read_text())
                reference = json.loads(reference_path.read_text())
                rho = result["transient"]["rho_point_L2V"]
                pressure = result["transient"]["pressure_point_L2V"]
                full_rho = reference["transient"]["rho_point_L2V"]
                full_pressure = reference["transient"]["pressure_point_L2V"]
                h = (400 / result["nodes"]) ** (1 / 3)
                order = (math.log(earlier[1] / rho) /
                         math.log(earlier[0] / h)) if earlier else ""
                old_rho = float(previous[(mesh, level)]["full_roe_rho_point_L2V"])
                rows.append({
                    "mesh_type": mesh,
                    "level": level,
                    "ablated_component": component,
                    "nodes": result["nodes"],
                    "mpi_ranks": result["mpi_ranks"],
                    "dt": result["dt"],
                    "steps": result["steps"],
                    "final_time": result["final_time"],
                    "h": h,
                    "full_rho_point_L2V": full_rho,
                    "rho_point_L2V": rho,
                    "rho_error_change_percent": 100 * (rho / full_rho - 1),
                    "full_pressure_point_L2V": full_pressure,
                    "pressure_point_L2V": pressure,
                    "pressure_error_change_percent":
                        100 * (pressure / full_pressure - 1),
                    "observed_order": order,
                    "baseline_rho_difference": full_rho - old_rho,
                    "mass_drift": result["transient"]["mass_drift"],
                    "component_rhs_validation_max_mismatch_t0":
                        result["component_rhs_validation_max_mismatch_t0"],
                })
                earlier = (h, rho)
    if rows:
        suffix = meshes[0] if len(meshes) == 1 else "all"
        target = output / f"metrics_{suffix}.csv"
        with target.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {len(rows)} rows to {target}", flush=True)
        failures = []
        for mesh in meshes:
            for level in LEVELS:
                for component in COMPONENTS:
                    path = raw_path(output, mesh, level, component)
                    failed = path.with_suffix(".failed")
                    if failed.exists() and not path.exists():
                        failures.append({
                            "mesh_type": mesh,
                            "level": level,
                            "ablated_component": component,
                            "status": failed.read_text().strip(),
                            "log": path.with_suffix(".log").name,
                        })
        if failures:
            failure_target = output / f"failed_cases_{suffix}.csv"
            with failure_target.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, failures[0].keys())
                writer.writeheader()
                writer.writerows(failures)


def run_case(output: Path, mesh: str, level: int, component: str) -> bool:
    path = raw_path(output, mesh, level, component)
    failed = path.with_suffix(".failed")
    expected_component = "none" if component == "full" else component
    if path.exists():
        result = json.loads(path.read_text())
        if (result["mode"] != "Traditional" or
                result["method"] != "LeastSquares" or
                result["ablated_component"] != expected_component):
            raise ValueError(f"Existing result has unexpected settings: {path}")
        print(f"Skip {path.stem}", flush=True)
        return True
    if failed.exists():
        print(f"Skip known failure {path.stem}: {failed.read_text().strip()}",
              flush=True)
        return False
    ranks, dt = LEVELS[level]
    mesh_path = ROOT / MESHES[mesh].format(level=level)
    command = ["mpirun", "--mca", "pml", "ob1", "--mca", "btl",
               "self,vader", "--bind-to", "core", "--map-by", "core",
               "-np", str(ranks), str(PROBE), str(CONFIG), str(path),
               "LeastSquares", "0", str(dt), "1.6", "1", "Traditional",
               str(mesh_path)]
    command.extend(["none" if component == "full" else component, "lean"])
    environment = os.environ.copy()
    environment.update({"HWLOC_COMPONENTS": "-gl,-cuda,-nvml",
                        "OMP_NUM_THREADS": "1", "DNDS_DIST_OMP_NUM_THREADS": "1",
                        "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    print(f"Start {path.stem}: {ranks} ranks, dt={dt}", flush=True)
    start = time.perf_counter()
    with path.with_suffix(".log").open("w") as log:
        status = subprocess.run(command, cwd=ROOT / "build", env=environment,
                                stdout=log, stderr=subprocess.STDOUT)
    elapsed = time.perf_counter() - start
    if status.returncode or not path.exists():
        failed.write_text(f"exit {status.returncode}; see {path.stem}.log\n")
        print(f"FAILED {path.stem}: exit {status.returncode}, {elapsed:.1f}s",
              flush=True)
        return False
    result = json.loads(path.read_text())
    print(f"Done {path.stem}: rho={result['transient']['rho_point_L2V']:.6e}, "
          f"pressure={result['transient']['pressure_point_L2V']:.6e}, "
          f"{elapsed:.1f}s", flush=True)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", choices=MESHES)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    meshes = (args.mesh,) if args.mesh else tuple(MESHES)
    if not args.summarize_only:
        if not PROBE.exists():
            parser.error(f"Missing compiled component probe: {PROBE}")
        for mesh in meshes:
            for level in LEVELS:
                for component in COMPONENTS:
                    run_case(output, mesh, level, component)
                summarize(output, (mesh,))
    summarize(output, meshes)


if __name__ == "__main__":
    main()
