#!/usr/bin/env python3
"""Run four frozen Riemann inputs concurrently and render verified results."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import h5py


ROOT = Path(__file__).resolve().parents[4]
CASES = Path(__file__).resolve().parent
OUTPUT = ROOT / "data/out/NCFV/riemann_barth_400x400x5"
NAMES = ["RP1", "RP2", "RP3", "RP4"]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    temporary = path.with_suffix(".partial.json")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def verify(directory, target, initial_time=0.):
    series_path = directory / "series.pvtu.series"
    entries = sorted(json.loads(series_path.read_text())["files"], key=lambda e: e["time"])
    if not entries or not math.isclose(entries[-1]["time"], target, abs_tol=1e-11):
        raise RuntimeError(f"{directory.name}: output did not reach t={target}")
    if abs(entries[0]["time"] - initial_time) > 1e-12:
        raise RuntimeError("Missing initial output")
    for entry in entries:
        master = directory / entry["name"]
        for piece in ET.parse(master).getroot().iter("Piece"):
            if not (master.parent / piece.attrib["Source"]).is_file():
                raise FileNotFoundError(piece.attrib["Source"])
    restart = sorted(directory.glob("restart_*.dnds.h5"))[-1]
    with h5py.File(restart) as data:
        saved_time = float(data.attrs["simulationTime"])
        iteration = int(data.attrs["iteration"])
        raw = data.attrs["configuration"]
        configuration = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
    if not math.isclose(saved_time, target, abs_tol=1e-11):
        raise RuntimeError(f"Restart time {saved_time} != {target}")
    if not configuration["reconstruction"]["enableLimiter"]:
        raise RuntimeError("Barth limiter disabled in saved restart")
    return {"simulation_time": saved_time, "iteration": iteration,
            "snapshots": len(entries), "restart": str(restart),
            "restart_sha256": digest(restart), "barth_enabled": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranks", type=int, default=64)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    if args.ranks < 1 or args.jobs < 1:
        parser.error("ranks and jobs must be positive")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frozen = OUTPUT / "inputs"
    frozen.mkdir(exist_ok=True)
    executable = frozen / "ncfv_euler3D.exe"
    if not executable.exists():
        shutil.copy2(ROOT / "build/app/ncfv_euler3D.exe", executable)
    for name in NAMES:
        destination = frozen / f"{name}.json"
        if not destination.exists():
            shutil.copy2(CASES / destination.name, destination)
        elif destination.read_bytes() != (CASES / destination.name).read_bytes():
            raise RuntimeError(f"Frozen configuration differs: {name}")
    environment = dict(os.environ, OMP_NUM_THREADS="1", DNDS_DIST_OMP_NUM_THREADS="1",
                       OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       HWLOC_COMPONENTS="-gl,-cuda,-nvml", OMPI_MCA_mpi_yield_when_idle="1",
                       EGL_PLATFORM="surfaceless")
    for key in list(environment):
        if key.startswith("NCFV_"):
            del environment[key]
    status = {"status": "computing", "pid": os.getpid(), "mpi_ranks_per_case": args.ranks,
              "concurrent_cases": min(args.jobs, 4), "completed_cases": [],
              "started_utc": datetime.now(timezone.utc).isoformat(),
              "note": "MPI process count does not override inherited CPU quota"}
    save(OUTPUT / "pipeline_status.json", status)

    def run(name):
        config = frozen / f"{name}.json"
        settings = json.loads(config.read_text())
        target = settings["time"]["endTime"]
        directory = OUTPUT / name
        manifest_path = directory / "run_manifest.json"
        hashes = {"binary_sha256": digest(executable), "config_sha256": digest(config),
                  "mesh_sha256": digest((ROOT / "build" / settings["mesh"]["meshFile"]).resolve()),
                  "initial_sha256": digest((ROOT / "build" / settings["initialField"]["nodeFile"]).resolve())}
        if args.skip_completed and manifest_path.exists():
            previous = json.loads(manifest_path.read_text())
            if previous.get("status") == "complete" and all(previous.get(k) == v for k, v in hashes.items()):
                verify(directory, target)
                return name
        directory.mkdir(exist_ok=True)
        if any(directory.iterdir()):
            raise FileExistsError(f"Refusing to overwrite {directory}")
        command = ["mpirun", "--mca", "pml", "ob1", "--mca", "btl", "self,vader",
                   "--bind-to", "none", "-np", str(args.ranks), str(executable), str(config)]
        manifest = dict(hashes, case=name, status="running", command=command,
                        working_directory=str(ROOT / "build"), target_end_time=target,
                        started_utc=datetime.now(timezone.utc).isoformat())
        started = time.monotonic()
        save(manifest_path, manifest)
        print(f"Starting {name}: {args.ranks} MPI ranks, target t={target}", flush=True)
        try:
            with (directory / "solver.log").open("x") as log:
                process = subprocess.Popen(command, cwd=ROOT / "build", env=environment,
                                           stdout=log, stderr=subprocess.STDOUT)
                manifest["launcher_pid"] = process.pid
                save(manifest_path, manifest)
                manifest["returncode"] = process.wait()
            if manifest["returncode"]:
                raise RuntimeError(f"{name} failed; see {directory / 'solver.log'}")
            manifest.update(verify(directory, target))
            manifest["status"] = "complete"
        except BaseException as error:
            manifest.update(status="failed", error=str(error))
            raise
        finally:
            manifest["elapsed_seconds"] = time.monotonic() - started
            manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
            save(manifest_path, manifest)
        print(f"Completed {name}: {manifest['iteration']} steps", flush=True)
        return name

    failures = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(run, name): name for name in NAMES}
        for future in as_completed(futures):
            try:
                status["completed_cases"].append(future.result())
            except Exception as error:
                failures.append(str(error))
            status["failures"] = failures
            status["updated_utc"] = datetime.now(timezone.utc).isoformat()
            save(OUTPUT / "pipeline_status.json", status)
    if failures:
        raise RuntimeError("; ".join(failures))
    if args.render:
        status["status"] = "rendering"
        save(OUTPUT / "pipeline_status.json", status)
        subprocess.run([sys.executable, str(CASES / "render_results.py"), "--root", str(OUTPUT),
                        "--z", "0.00625", "--workers", "2"],
                       cwd=ROOT, env=environment, check=True)
        for video in (OUTPUT / "videos").glob("*.mp4"):
            if video.stat().st_size == 0:
                raise RuntimeError(f"Empty video: {video}")
    status.update(status="complete", finished_utc=datetime.now(timezone.utc).isoformat(),
                  videos_rendered=args.render)
    save(OUTPUT / "pipeline_status.json", status)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        path = OUTPUT / "pipeline_status.json"
        status = json.loads(path.read_text()) if path.exists() else {}
        status.update(status="failed", error=str(error))
        save(path, status)
        raise
