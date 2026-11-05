#!/usr/bin/env python3
"""Run the four Riemann states on the small mesh using CLI overrides only.

The production JSON files are read without modification.  Every run records
the exact command, executable/config hashes, elapsed time, exit status, and
log-derived time-step sum.  VTK, restarts, and solver diagnostics are disabled.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time


ROOT = Path(__file__).resolve().parents[4]
CASE_DIR = Path(__file__).resolve().parent
ITERATION = re.compile(
    r"NCFV iteration=(\d+), residual RMS=([^,]+), dt\[min,max\]=\[([^,]+),([^]]+)\]")


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=ROOT / "build/app/ncfv_euler3D.exe")
    parser.add_argument("--mesh", type=Path,
                        default=ROOT / "data/mesh/riemann_barth_pilot40x40x4.cgns")
    parser.add_argument("--ranks", type=int, default=8)
    parser.add_argument("--cases", nargs="+", choices=["RP1", "RP2", "RP3", "RP4"],
                        default=["RP1", "RP2", "RP3", "RP4"])
    parser.add_argument("--modes", nargs="+",
                        choices=["TraditionalQuadrature", "EfficientDifferential"],
                        default=["TraditionalQuadrature", "EfficientDifferential"])
    parser.add_argument("--max-dt", type=float, default=0.001)
    parser.add_argument("--cfl", type=float, default=0.5)
    parser.add_argument("--timeout", type=float, default=1200)
    parser.add_argument("--tag", default="np8_cfl05")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "data/out/NCFV/riemann_barth_400/pilots")
    args = parser.parse_args()
    if args.ranks < 1 or min(args.max_dt, args.cfl, args.timeout) <= 0:
        parser.error("ranks, max-dt, cfl, and timeout must be positive")
    binary, mesh, output = args.binary.resolve(), args.mesh.resolve(), args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    binary_hash = digest(binary)
    env = dict(os.environ, OMP_NUM_THREADS="1", DNDS_DIST_OMP_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               HWLOC_COMPONENTS="-gl,-cuda,-nvml")
    for name in list(env):
        if name.startswith("NCFV_"):
            del env[name]
    all_results = []
    for mode in args.modes:
        for name in args.cases:
            config = CASE_DIR / f"{name}.json"
            settings = json.loads(config.read_text())
            end_time = float(settings["time"]["endTime"])
            label = f"{name}_{mode}_{args.tag}"
            log_path, result_path = output / f"{label}.log", output / f"{label}.json"
            if log_path.exists() or result_path.exists():
                raise FileExistsError(f"refusing to overwrite {label}")
            overrides = {
                "/mesh/meshFile": str(mesh), "/algorithm/mode": mode,
                "/time/cfl": args.cfl, "/time/maximumTimeStep": args.max_dt,
                "/time/reportInterval": 1, "/time/iterations": 1000000,
                "/io/writeVTK": False, "/io/writeInitial": False,
                "/io/writeFinal": False, "/io/outputInterval": 0,
                "/io/restartInterval": 0, "/io/writeFinalRestart": False,
                "/io/writeResolvedConfiguration": False,
                "/io/outputPrefix": str(output / label),
                "/initialField/isentropicVortex": False,
            }
            command = ["mpirun", "--mca", "pml", "ob1", "--mca", "btl", "self,vader",
                       "--bind-to", "none", "-np", str(args.ranks), str(binary), str(config)]
            for key, value in overrides.items():
                command.extend(["-k", key, "-v", json.dumps(value)])
            result = {"case": name, "mode": mode, "ranks": args.ranks,
                      "target_end_time": end_time, "command": command,
                      "binary_sha256": binary_hash, "configuration_sha256": digest(config),
                      "mesh_sha256": digest(mesh), "cli_overrides": overrides,
                      "log_file": str(log_path)}
            print(f"Start {label}", flush=True)
            start = time.monotonic()
            with log_path.open("x") as log:
                try:
                    process = subprocess.run(command, cwd=ROOT / "build", env=env,
                                             stdout=log, stderr=subprocess.STDOUT,
                                             timeout=args.timeout, check=False)
                    result["returncode"] = process.returncode
                    result["timed_out"] = False
                except subprocess.TimeoutExpired:
                    result["returncode"] = None
                    result["timed_out"] = True
            result["elapsed_seconds"] = time.monotonic() - start
            contents = log_path.read_text(errors="replace")
            steps = ITERATION.findall(contents)
            result["completed_steps"] = int(steps[-1][0]) if steps else 0
            result["logged_time_step_sum"] = sum(float(step[2]) for step in steps)
            result["all_logged_residuals_finite"] = bool(steps) and all(
                math.isfinite(float(step[1])) for step in steps)
            result["failure_message_detected"] = any(token in contents.lower() for token in (
                "nonphysical", "non-physical", "top-level exception", "assertion", "sigabrt"))
            result["reached_target"] = (
                result["returncode"] == 0 and not result["timed_out"] and
                abs(result["logged_time_step_sum"] - end_time) < 1e-9 and
                result["all_logged_residuals_finite"] and
                not result["failure_message_detected"])
            result_path.write_text(json.dumps(result, indent=2) + "\n")
            all_results.append(result)
            print(f"Done {label}: rc={result['returncode']}, "
                  f"steps={result['completed_steps']}, t={result['logged_time_step_sum']:.12g}, "
                  f"seconds={result['elapsed_seconds']:.3f}, "
                  f"reached_target={result['reached_target']}", flush=True)
    summary = output / f"summary_{args.tag}.json"
    if summary.exists():
        raise FileExistsError(f"refusing to overwrite {summary}")
    summary.write_text(json.dumps(all_results, indent=2) + "\n")


if __name__ == "__main__":
    main()
