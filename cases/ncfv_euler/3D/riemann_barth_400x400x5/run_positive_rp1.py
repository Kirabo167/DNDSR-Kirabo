"""Recompute full RP1 with mean-anchored positivity limiting and render results."""
from datetime import datetime, timezone
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import h5py
import numpy as np

from run_cases import ROOT, CASES, digest, save, verify


OUTPUT = ROOT / "data/out/NCFV/riemann_barth_400x400x5_positive"


def main():
    global OUTPUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranks", type=int, default=64)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--config", type=Path, default=CASES / "RP1_positive.json")
    parser.add_argument("--executable", type=Path, default=ROOT / "build/app/ncfv_euler3D.exe")
    args = parser.parse_args()
    if args.ranks < 1:
        parser.error("ranks must be positive")
    OUTPUT = args.output.resolve()
    directory = OUTPUT / "RP1"
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    frozen = OUTPUT / "inputs"
    frozen.mkdir(exist_ok=True)
    config = frozen / args.config.name
    executable = frozen / "ncfv_euler3D.exe"
    for source, destination in ((args.config.resolve(), config),
                                (args.executable.resolve(), executable),
                                (CASES / "render_results.py", frozen / "render_results.py")):
        if destination.exists():
            raise FileExistsError(destination)
        shutil.copy2(source, destination)
    settings = json.loads(config.read_text())
    for key in ("outputPrefix", "vtkSeriesName", "restartPrefix"):
        path = (ROOT / "build" / settings["io"][key]).resolve()
        if path.parent != directory:
            raise ValueError(f"{key} must point into {directory}, got {path}")
    assert settings["reconstruction"]["enableLimiter"]
    assert settings["reconstruction"]["limitPointRecovery"]
    assert settings["time"]["rejectNonPhysicalSteps"]
    assert settings["time"]["endTime"] == .25
    environment = dict(os.environ, OMP_NUM_THREADS="1", DNDS_DIST_OMP_NUM_THREADS="1",
                       OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       HWLOC_COMPONENTS="-gl,-cuda,-nvml", OMPI_MCA_mpi_yield_when_idle="1",
                       EGL_PLATFORM="surfaceless")
    for key in list(environment):
        if key.startswith("NCFV_"):
            del environment[key]
    mpi = ["mpirun", "--mca", "pml", "ob1", "--mca", "btl", "self,vader", "--bind-to", "none"]
    subprocess.run([*mpi, "-np", "1", str(executable), str(config), "--check-config"],
                   cwd=ROOT / "build", env=environment, check=True, timeout=60)
    command = [*mpi, "-np", str(args.ranks), str(executable), str(config)]
    status = {"status": "computing", "case": "RP1", "pid": os.getpid(),
              "mpi_ranks": args.ranks, "command": command,
              "started_utc": datetime.now(timezone.utc).isoformat(),
              "target_end_time": .25, "binary_sha256": digest(executable),
              "config_sha256": digest(config),
              "mesh_sha256": digest((ROOT / "build" / settings["mesh"]["meshFile"]).resolve()),
              "initial_sha256": digest((ROOT / "build" / settings["initialField"]["nodeFile"]).resolve()),
              "solver_log": str(directory / "solver.log"),
              "positivity_strategy": "One mean-anchored factor for point and interface states; reject and halve nonphysical RK steps"}
    save(OUTPUT / "pipeline_status.json", status)
    save(directory / "run_manifest.json", status)
    started = time.monotonic()
    print("Starting full 400x400x5 RP1 from t=0 to t=.25", flush=True)
    with (directory / "solver.log").open("x") as log:
        process = subprocess.Popen(command, cwd=ROOT / "build", env=environment,
                                   stdout=log, stderr=subprocess.STDOUT)
        status["launcher_pid"] = process.pid
        save(directory / "run_manifest.json", status)
        while True:
            try:
                status["returncode"] = process.wait(timeout=30)
                break
            except subprocess.TimeoutExpired:
                try:
                    frames = json.loads((directory / "series.pvtu.series").read_text())["files"]
                    status["latest_output_time"] = max(e["time"] for e in frames)
                    status["progress_percent"] = 100 * status["latest_output_time"] / .25
                except (OSError, ValueError, KeyError):
                    pass
                status["elapsed_seconds"] = time.monotonic() - started
                status["updated_utc"] = datetime.now(timezone.utc).isoformat()
                save(OUTPUT / "pipeline_status.json", status)
    if status["returncode"]:
        raise RuntimeError(f"Solver failed: {directory / 'solver.log'}")
    status.update(verify(directory, .25))
    solver_text = (directory / "solver.log").read_text(errors="replace")
    initialized_nodes = re.search(r"global nodes=(\d+)", solver_text)
    if initialized_nodes is None:
        raise ValueError("Missing initialized global node count in solver log")
    expected_nodes = int(initialized_nodes.group(1))
    with h5py.File(status["restart"]) as data:
        values = data["state/father/array/data"][:].reshape(-1, 5)
        pressure = .4 * (values[:, 4] - .5 * np.sum(values[:, 1:4]**2, axis=1) / values[:, 0])
        assert len(values) == expected_nodes and np.isfinite(values).all()
        assert np.all(values[:, 0] > 1e-12) and np.all(pressure > 1e-12)
        raw = data.attrs["configuration"]
        actual = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
        assert actual["reconstruction"]["limitPointRecovery"] and actual["time"]["rejectNonPhysicalSteps"]
    status.update(status="complete", nodes=expected_nodes, latest_output_time=status["simulation_time"],
                  progress_percent=100.0, minimum_final_density=float(values[:, 0].min()),
                  minimum_final_pressure=float(pressure.min()),
                  rejected_steps=solver_text.count("NCFV rejected iteration="),
                  elapsed_seconds=time.monotonic() - started,
                  finished_utc=datetime.now(timezone.utc).isoformat())
    save(directory / "run_manifest.json", status)
    status["status"] = "rendering"
    save(OUTPUT / "pipeline_status.json", status)
    print(f"Reached t=.25 in {status['iteration']} steps; rendering", flush=True)
    with (OUTPUT / "render.log").open("x") as log:
        subprocess.run([sys.executable, str(frozen / "render_results.py"), "--root", str(OUTPUT),
                        "--cases", "RP1", "--workers", "2", "--contours", "25", "--cmap", "coolwarm"],
                       cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
    status.update(status="complete", videos_rendered=True,
                  finished_utc=datetime.now(timezone.utc).isoformat(),
                  video=str(OUTPUT / "videos/RP1_density_pressure.mp4"),
                  final_figure=str(OUTPUT / "videos/RP1_final_density_pressure.png"))
    save(OUTPUT / "pipeline_status.json", status)
    print(json.dumps(status, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        path = OUTPUT / "pipeline_status.json"
        status = json.loads(path.read_text()) if path.exists() else {}
        status.update(status="failed", error=str(error),
                      updated_utc=datetime.now(timezone.utc).isoformat())
        save(path, status)
        raise
