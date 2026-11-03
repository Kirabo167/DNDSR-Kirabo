"""Run serial/colored RP1 comparisons sequentially on distinct physical cores.

Generate the mesh/dual-mean CSV with the existing riemann_barth_400x400x5
generators first: --nx 40 --ny 40 --nz 20 --depth 2. The stretched z/x ratio
is four, matching the user's 400x400x20, Lz=0.2 experiment. These are short
strong-scaling measurements, not a prediction of the production run's speed.
"""

from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import argparse
import hashlib
import json
import os
import re
import resource
import shutil
import statistics
import subprocess
import time

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
CASE = Path(__file__).with_name("RP1_small.json")


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def save(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def checkpoint_report(directory):
    path = directory / "restart_00000012.dnds.h5"
    with h5py.File(path, "r") as handle:
        values = handle["state/father/array/data"][:].reshape(-1, 5)
        identities = handle["state/origIndex"][:]
        physical_time = float(handle.attrs["simulationTime"])
        iteration = int(handle.attrs["iteration"])
    expected = 41 * 41 * 20
    if values.shape != (expected, 5) or not np.array_equal(np.sort(identities), np.arange(expected)):
        raise ValueError("Incomplete periodic checkpoint")
    if iteration != 12 or abs(physical_time - .0024) > 1e-13 or not np.isfinite(values).all():
        raise ValueError("Incorrect checkpoint time or nonfinite values")
    ordered = np.empty_like(values)
    ordered[identities] = values
    cube = ordered.reshape(20, 41, 41, 5)
    rho = cube[..., 0]
    velocity = cube[..., 1:4] / rho[..., None]
    pressure = .4 * (cube[..., 4] - .5 * rho * np.sum(velocity**2, axis=-1))
    if rho.min() <= 0 or pressure.min() <= 0:
        raise ValueError("Nonphysical saved means")
    fields = {"density": rho, "pressure": pressure, "velocity_x": velocity[..., 0],
              "velocity_y": velocity[..., 1], "velocity_z": velocity[..., 2]}
    np.savez_compressed(directory / "final_layers.npz", **fields, z=np.arange(20)*.1)
    report = {"physical_time": physical_time, "iteration": iteration, "fields": {}}
    for name, field in fields.items():
        span = np.ptp(field, axis=0)
        report["fields"][name] = {"minimum": float(field.min()), "maximum": float(field.max()),
                                  "maximum_absolute": float(np.abs(field).max()),
                                  "maximum_cross_z_range": float(span.max()),
                                  "rms_cross_z_range": float(np.sqrt(np.mean(span**2)))}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ranks", type=int, nargs="+", default=[64, 128, 256])
    parser.add_argument("--modes", nargs="+", choices=["serial", "parallel"], default=["serial", "parallel"])
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    inputs = output / "inputs"
    inputs.mkdir()
    executable = inputs / "ncfv_euler3D.exe"
    shutil.copy2(ROOT / "build/app/ncfv_euler3D.exe", executable)
    shutil.copy2(CASE, inputs / "case.json")
    shutil.copytree(ROOT / "src/NCFV", inputs / "NCFV")
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(65536, hard), hard))
    environment = os.environ.copy()
    environment.pop("DISPLAY", None)
    environment.pop("WAYLAND_DISPLAY", None)
    environment.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       DNDS_DIST_OMP_NUM_THREADS="1", OMP_DYNAMIC="FALSE")
    cores = {}
    for cpu in sorted(os.sched_getaffinity(0)):
        topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        key = ((topology/"physical_package_id").read_text().strip(), (topology/"core_id").read_text().strip())
        cores.setdefault(key, cpu)
    cpus = list(cores.values())
    if max(args.ranks) > len(cpus):
        raise ValueError("Insufficient distinct physical cores")
    state = {"started_at": now(), "phase": "running", "grid": [40, 40, 20],
             "spacing": [.025, .025, .1], "domain": [[0, 1], [0, 1], [0, 2]],
             "fixed_physical_step": .0002, "steps": 12, "warmup_steps_excluded": 2,
             "io_during_measured_steps": False, "threads_per_rank": 1,
             "executable_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(), "runs": []}
    save(output/"results.json", state)
    progress = re.compile(r"NCFV iteration=(\d+).*?time=([^,]+), target=([^,]+), step wall\[s\]=([^\s]+)")
    for ranks in args.ranks:
        for mode in args.modes:
            directory = output / f"{mode}_np{ranks}"
            directory.mkdir()
            command = ["mpirun", "--mca", "pml", "ob1", "--mca", "btl", "self,vader",
                       "--use-hwthread-cpus", "-np", str(ranks), "--cpu-list", ",".join(map(str, cpus[:ranks])),
                       "--bind-to", "cpu-list:ordered", "--report-bindings", str(executable), str(inputs/"case.json"),
                       "-k", "/reconstruction/sharedGradientFaceSweepParallel", "-v", json.dumps(mode == "parallel"),
                       "-k", "/io/restartPrefix", "-v", json.dumps(str(directory/"restart"))]
            state.update(active_run=f"{mode}_np{ranks}", updated_at=now())
            save(output/"results.json", state)
            save(directory/"command.json", {"command": command, "cwd": str(ROOT/"build"), "started_at": now()})
            print(f"START {mode} np={ranks} {now()}", flush=True)
            started = time.monotonic()
            with (directory/"solver.log").open("x") as log:
                process = subprocess.Popen(command, cwd=ROOT/"build", env=environment,
                                           stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                state["active_mpi_pid"] = process.pid
                save(output/"results.json", state)
                returncode = process.wait()
            wall = time.monotonic() - started
            log = (directory/"solver.log").read_text(errors="replace")
            if returncode:
                state.update(phase="failed", returncode=returncode, failed_run=str(directory))
                save(output/"results.json", state)
                raise RuntimeError(f"Failed {directory}; inspect solver.log")
            records = [{"iteration": int(m[1]), "physical_time": float(m[2]), "seconds": float(m[4])}
                       for m in progress.finditer(log)]
            if len(records) != 12 or "NCFV rejected iteration=" in log:
                raise RuntimeError("Missing steps or a rejected step; timing comparison invalid")
            measured = [record["seconds"] for record in records[2:]]
            grouping = re.search(r"full-support coloring, (\d+) colors, (\d+) proposal rounds, initialization\[s\]=([^;]+)", log)
            limiter = re.findall(r"limiter stages\[s\]=([^,\s]+)", log)
            result = {"mode": mode, "mpi_ranks": ranks, "physical_cores": ranks,
                      "directory": str(directory), "cpu_list": cpus[:ranks], "returncode": returncode,
                      "total_wall_seconds_including_initialization_and_final_io": wall,
                      "step_seconds": records, "measured_step_mean_seconds": statistics.mean(measured),
                      "measured_step_median_seconds": statistics.median(measured),
                      "measured_step_minimum_seconds": min(measured), "measured_step_maximum_seconds": max(measured),
                      "color_groups": int(grouping[1]) if grouping else None,
                      "coloring_proposal_rounds": int(grouping[2]) if grouping else None,
                      "coloring_initialization_seconds_rank0": float(grouping[3]) if grouping else None,
                      "limiter_stages_mean_seconds_rank0": statistics.mean(map(float, limiter[2:])) if len(limiter) == 12 else None,
                      "checkpoint": checkpoint_report(directory)}
            save(directory/"result.json", result)
            state["runs"].append(result)
            state.update(updated_at=now())
            save(output/"results.json", state)
            print(f"DONE {mode} np={ranks}: mean step {result['measured_step_mean_seconds']:.6f}s", flush=True)
    state.update(phase="complete", finished_at=now(), active_run=None, active_mpi_pid=None)
    save(output/"results.json", state)


if __name__ == "__main__":
    main()
