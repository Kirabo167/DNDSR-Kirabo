"""Run an independent RP1 continuation and join it to the existing result series.

The supplied configuration must write into a new segment directory. Preserve the
original failure manifest and series. After all four computations finish, take
over final verification/rendering from the initial batch, which saw RP1 fail.
"""
from datetime import datetime, timezone
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import h5py

from run_cases import digest, save, verify, ROOT, OUTPUT, CASES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    config = args.config.resolve()
    settings = json.loads(config.read_text())
    directory = OUTPUT / "RP1"
    segment = (ROOT / "build" / settings["io"]["outputPrefix"]).resolve().parent
    if segment.parent != directory or segment.exists():
        raise RuntimeError("Continuation must use a new directory inside RP1")
    restart = (ROOT / "build" / (settings["io"]["restartInput"] + ".dnds.h5")).resolve()
    with h5py.File(restart) as data:
        start_time = float(data.attrs["simulationTime"])
        start_iteration = int(data.attrs["iteration"])
    segment.mkdir()
    frozen = segment / "inputs"
    frozen.mkdir()
    shutil.copy2(config, frozen / config.name)
    executable = frozen / "ncfv_euler3D.exe"
    shutil.copy2(ROOT / "build/app/ncfv_euler3D.exe", executable)
    command = ["mpirun", "--mca", "pml", "ob1", "--mca", "btl", "self,vader",
               "--bind-to", "none", "-np", "64", str(executable), str(frozen / config.name)]
    environment = dict(os.environ, OMP_NUM_THREADS="1", DNDS_DIST_OMP_NUM_THREADS="1",
                       OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       HWLOC_COMPONENTS="-gl,-cuda,-nvml", OMPI_MCA_mpi_yield_when_idle="1",
                       EGL_PLATFORM="surfaceless")
    for key in list(environment):
        if key.startswith("NCFV_"):
            del environment[key]
    previous = directory / "run_manifest.initial_attempt.json"
    if not previous.exists():
        shutil.copy2(directory / "run_manifest.json", previous)
    manifest = {"case": "RP1", "status": "running", "controller_pid": os.getpid(),
                "command": command, "working_directory": str(ROOT / "build"),
                "started_utc": datetime.now(timezone.utc).isoformat(),
                "restart": str(restart), "restart_sha256": digest(restart),
                "start_time": start_time, "start_iteration": start_iteration,
                "target_end_time": settings["time"]["endTime"], "cfl": settings["time"]["cfl"],
                "config_sha256": digest(config), "binary_sha256": digest(executable),
                "segment_directory": str(segment), "active_solver_log": str(segment / "solver.log"),
                "previous_attempt_manifest": str(previous)}
    save(directory / "run_manifest.json", manifest)
    save(OUTPUT / "RP1_recovery_status.json", manifest)
    started = time.monotonic()
    try:
        with (segment / "solver.log").open("x") as log:
            process = subprocess.Popen(command, cwd=ROOT / "build", env=environment,
                                       stdout=log, stderr=subprocess.STDOUT)
            manifest["launcher_pid"] = process.pid
            save(directory / "run_manifest.json", manifest)
            manifest["returncode"] = process.wait()
        if manifest["returncode"]:
            raise RuntimeError(f"Continuation failed: {segment / 'solver.log'}")
        result = verify(segment, settings["time"]["endTime"], initial_time=start_time)
        original_series = directory / "series.pvtu.series"
        backup = directory / "series.before_resume.pvtu.series"
        if not backup.exists():
            shutil.copy2(original_series, backup)
        original = json.loads(backup.read_text())
        continuation = json.loads((segment / "series.pvtu.series").read_text())
        frames = {float(e["time"]): e for e in original["files"] if float(e["time"]) <= start_time + 1e-12}
        for entry in continuation["files"]:
            frames[float(entry["time"])] = {"name": str(Path(segment.name) / entry["name"]), "time": entry["time"]}
        original["files"] = [frames[t] for t in sorted(frames)]
        save(original_series, original)
        final_restart = Path(result["restart"])
        link = directory / final_restart.name
        if link.exists():
            raise FileExistsError(link)
        link.symlink_to(final_restart.relative_to(directory))
        manifest.update(verify(directory, settings["time"]["endTime"]), status="complete")
        print("RP1 continuation complete; joined series verified", flush=True)
    except BaseException as error:
        manifest.update(status="failed", error=str(error))
        raise
    finally:
        manifest["elapsed_seconds"] = time.monotonic() - started
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        save(directory / "run_manifest.json", manifest)
        save(OUTPUT / "RP1_recovery_status.json", manifest)

    # The first batch has already recorded RP1 as failed. Wait for its other
    # workers to finish before updating shared pipeline status or rendering.
    while True:
        states = {name: json.loads((OUTPUT / name / "run_manifest.json").read_text())["status"]
                  for name in ("RP2", "RP3", "RP4")}
        if any(value == "failed" for value in states.values()):
            raise RuntimeError(f"Other case failed; automatic rendering deferred: {states}")
        if all(value == "complete" for value in states.values()):
            break
        time.sleep(30)
    while True:
        initial = json.loads((OUTPUT / "pipeline_status.json").read_text())
        initial_pid = initial.get("pid")
        try:
            command_line = Path(f"/proc/{initial_pid}/cmdline").read_bytes()
        except FileNotFoundError:
            break
        if b"run_cases.py" not in command_line:
            break
        time.sleep(5)
    for name in ("RP1", "RP2", "RP3", "RP4"):
        target = json.loads((OUTPUT / "inputs" / f"{name}.json").read_text())["time"]["endTime"]
        verify(OUTPUT / name, target)
    pipeline = {"status": "rendering", "pid": os.getpid(), "completed_cases": ["RP1", "RP2", "RP3", "RP4"],
                "RP1_recovered": True, "updated_utc": datetime.now(timezone.utc).isoformat()}
    save(OUTPUT / "pipeline_status.json", pipeline)
    subprocess.run([sys.executable, str(CASES / "render_results.py"), "--root", str(OUTPUT),
                    "--z", ".00625", "--workers", "2"], cwd=ROOT, env=environment, check=True)
    pipeline.update(status="complete", videos_rendered=True, finished_utc=datetime.now(timezone.utc).isoformat())
    save(OUTPUT / "pipeline_status.json", pipeline)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        save(OUTPUT / "RP1_recovery_controller_error.json", {"error": str(error), "time_utc": datetime.now(timezone.utc).isoformat()})
        raise
