#!/usr/bin/env python3
"""Run the saved 400 x 400 x 10 Riemann cases and verify their final files.

Case JSON files are passed directly to the production executable.  Run logs,
commands, input hashes, exit codes, and final restart times are preserved in
the output directory.  A pre-existing result directory is never overwritten.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

import h5py


ROOT = Path(__file__).resolve().parents[4]
CASE_DIR = Path(__file__).resolve().parent
OUTPUT = ROOT / "data/out/NCFV/riemann_barth_400"


def pipeline_status(status: str, **details) -> None:
    path = OUTPUT / "pipeline_status.json"
    previous = json.loads(path.read_text()) if path.exists() else {}
    if previous.get("pid") != os.getpid():
        previous = {"pid": os.getpid(), "started_utc": datetime.now(timezone.utc).isoformat()}
    previous.update(status=status, updated_utc=datetime.now(timezone.utc).isoformat(), **details)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial.json")
    temporary.write_text(json.dumps(previous, indent=2) + "\n")
    temporary.replace(path)


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def final_results(directory: Path, target: float) -> dict:
    series_path = directory / "series.pvtu.series"
    series = json.loads(series_path.read_text())
    frames = sorted(series["files"], key=lambda entry: entry["time"])
    if not frames or abs(float(frames[-1]["time"]) - target) > 1e-11:
        raise RuntimeError(f"{directory.name}: output series did not reach {target}")
    for entry in frames:
        if not (series_path.parent / entry["name"]).is_file():
            raise FileNotFoundError(entry["name"])
    restarts = sorted(directory.glob("restart_*.dnds.h5"))
    if not restarts:
        raise RuntimeError(f"{directory.name}: no final HDF5 restart")
    restart = restarts[-1]
    with h5py.File(restart, "r") as data:
        saved_time = float(data.attrs["simulationTime"])
        iteration = int(data.attrs["iteration"])
        configuration = json.loads(data.attrs["configuration"].decode())
    if not math.isclose(saved_time, target, rel_tol=0, abs_tol=1e-11):
        raise RuntimeError(f"{directory.name}: restart time {saved_time} != {target}")
    if not configuration["reconstruction"]["enableLimiter"]:
        raise RuntimeError("the saved computation did not enable the Barth limiter")
    return {"simulation_time": saved_time, "iteration": iteration,
            "snapshots": len(frames), "series": str(series_path),
            "restart": str(restart), "restart_sha256": digest(restart),
            "integration_mode": configuration["algorithm"]["mode"],
            "barth_enabled": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranks", type=int, default=64)
    parser.add_argument("--binary", type=Path, default=ROOT / "build/app/ncfv_euler3D.exe")
    parser.add_argument("--case-directory", type=Path, default=CASE_DIR,
                        help="Directory containing RP1.json through RP4.json")
    parser.add_argument("--cases", nargs="+", choices=["RP1", "RP2", "RP3", "RP4"],
                        default=["RP1", "RP2", "RP3", "RP4"])
    parser.add_argument("--skip-completed", action="store_true",
                        help="verify and skip cases that already have a completion manifest")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    if args.ranks < 1:
        parser.error("ranks must be positive")
    binary = args.binary.resolve()
    env = dict(os.environ, OMP_NUM_THREADS="1", DNDS_DIST_OMP_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               HWLOC_COMPONENTS="-gl,-cuda,-nvml", OMPI_MCA_mpi_yield_when_idle="1")
    for name in list(env):
        if name.startswith("NCFV_"):
            del env[name]
    binary_hash = digest(binary)
    pipeline_status("computing", requested_cases=args.cases, ranks=args.ranks,
                    binary=str(binary), case_directory=str(args.case_directory.resolve()),
                    completed_cases=[])
    completed_cases = []
    for name in args.cases:
        pipeline_status("computing", current_case=name)
        config_path = args.case_directory.resolve() / f"{name}.json"
        settings = json.loads(config_path.read_text())
        target = float(settings["time"]["endTime"])
        mesh = (ROOT / "build" / settings["mesh"]["meshFile"]).resolve()
        initial_file = settings.get("initialField", {}).get("nodeFile", "")
        initial_hash = digest((ROOT / "build" / initial_file).resolve()) if initial_file else None
        directory = OUTPUT / name
        manifest_path = directory / "run_manifest.json"
        if args.skip_completed and manifest_path.exists():
            previous = json.loads(manifest_path.read_text())
            if previous.get("status") == "complete":
                expected_hashes = {"configuration_sha256": digest(config_path),
                                   "binary_sha256": binary_hash, "mesh_sha256": digest(mesh),
                                   "initial_field_sha256": initial_hash}
                if any(previous.get(key) != value for key, value in expected_hashes.items()):
                    raise RuntimeError(f"{name}: saved results use different inputs; refusing to skip")
                print(f"Verified {name}: {final_results(directory, target)}", flush=True)
                completed_cases.append(name)
                pipeline_status("computing", completed_cases=completed_cases)
                continue
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError(f"refusing to overwrite existing results in {directory}")
        directory.mkdir(parents=True, exist_ok=True)
        command = ["mpirun", "--use-hwthread-cpus", "--mca", "pml", "ob1", "--mca", "btl", "self,vader",
                   "--bind-to", "none", "-np", str(args.ranks), str(binary), str(config_path)]
        manifest = {"case": name, "status": "running", "command": command,
                    "working_directory": str(ROOT / "build"), "ranks": args.ranks,
                    "started_utc": datetime.now(timezone.utc).isoformat(),
                    "target_end_time": target, "configuration_sha256": digest(config_path),
                    "binary_sha256": binary_hash, "mesh_sha256": digest(mesh),
                    "initial_field_sha256": initial_hash}
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"Starting {name}, end time {target}, {args.ranks} MPI ranks", flush=True)
        started = time.monotonic()
        try:
            with (directory / "solver.log").open("x") as log:
                process = subprocess.run(command, cwd=ROOT / "build", env=env,
                                         stdout=log, stderr=subprocess.STDOUT, check=False)
            manifest["returncode"] = process.returncode
            if process.returncode:
                raise RuntimeError(f"{name} exited with status {process.returncode}; see solver.log")
            manifest.update(final_results(directory, target))
            manifest["status"] = "complete"
        except BaseException as error:
            manifest["status"] = "failed"
            manifest["error"] = str(error)
            raise
        finally:
            manifest["elapsed_seconds"] = time.monotonic() - started
            manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"Completed {name}: {manifest['iteration']} steps, "
              f"{manifest['snapshots']} snapshots, {manifest['elapsed_seconds']:.1f} s", flush=True)
        completed_cases.append(name)
        pipeline_status("computing", completed_cases=completed_cases)
    if args.render:
        pipeline_status("rendering", current_case=None)
        for name in ("RP1", "RP2", "RP3", "RP4"):
            config = json.loads((args.case_directory / f"{name}.json").read_text())
            final_results(OUTPUT / name, float(config["time"]["endTime"]))
        subprocess.run([str(ROOT / "venv/bin/python"), str(CASE_DIR / "render_results.py"),
                        "--root", str(OUTPUT), "--workers", "2"], cwd=ROOT, env=env, check=True)
    pipeline_status("complete", current_case=None, videos_rendered=args.render)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        pipeline_status("failed", error=str(error))
        raise
