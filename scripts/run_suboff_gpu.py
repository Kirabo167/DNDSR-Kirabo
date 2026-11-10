#!/usr/bin/env python3
"""Run the supplied SUBOFF mesh with resident SST/BDF2 and persist job status."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "data/out/GPU/SUBOFF_SST_BDF2_20261104"
CONFIG = ROOT / "cases/acm/GPU/3D/suboff/suboff_SST_BDF2.json"
PATTERN = re.compile(r"ACM resident physical step\s+(\d+) time=(\S+) BDF(\d+) residual (\S+) -> (\S+) dtauMin=(\S+) turbResidual=(\S+) inner=(\d+) converged=(\d+)")


def save_status(status):
    temporary = DIRECTORY / "status.json.tmp"
    temporary.write_text(json.dumps(status, indent=2, allow_nan=False) + "\n")
    temporary.replace(DIRECTORY / "status.json")


def audit_final():
    files = sorted((DIRECTORY / "fields").glob("*.vtkhdf"))
    if not files:
        raise RuntimeError("No saved flow field")
    fields = {}
    with h5py.File(files[-1]) as hf:
        for name in ["Pressure", "Velocity", "Density", "TurbulenceK", "TurbulenceOmega", "rho_TurbulenceK", "rho_TurbulenceOmega"]:
            ds = hf["VTKHDF/CellData/" + name]
            low, high, finite = float("inf"), -float("inf"), True
            for start in range(0, len(ds), 100000):
                values = ds[start:start + 100000]
                finite = finite and bool(np.isfinite(values).all())
                low, high = min(low, float(values.min())), max(high, float(values.max()))
            if not finite or ("Turbulence" in name and low <= 0):
                raise RuntimeError(f"Invalid saved {name}")
            fields[name] = {"minimum": low, "maximum": high, "all_finite": finite, "shape": list(ds.shape)}
        cells = int(hf["VTKHDF/CellData/Pressure"].shape[0])
    if cells != 3914184:
        raise RuntimeError(f"Saved cell count {cells} differs from the input mesh")
    result = dict(file=str(files[-1]), cells=cells, fields=fields, steady_convergence_verified=False, accuracy_verified=False)
    (DIRECTORY / "final_field_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    global DIRECTORY, CONFIG
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--ranks", type=int, default=2)
    parser.add_argument("--gpu-device", default="auto", help="CUDA device index, or automatic local-rank assignment")
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output", type=Path, default=DIRECTORY)
    args = parser.parse_args()
    DIRECTORY = args.output.resolve()
    CONFIG = args.config.resolve()
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    status_file = DIRECTORY / "status.json"
    if status_file.exists():
        prior = json.loads(status_file.read_text())
        if prior.get("state") in ["starting", "running", "initializing", "paused"]:
            try:
                os.kill(prior["worker_pid"], 0)
            except ProcessLookupError:
                pass
            else:
                raise RuntimeError(f"This run is already managed by PID {prior['worker_pid']}")
        if prior.get("state") == "completed":
            raise RuntimeError("The requested run has already completed")
    if any((DIRECTORY / "fields").glob("*.vtkhdf")):
        raise RuntimeError("Saved fields already exist; choose a new output directory before a fresh run")
    config = json.loads(CONFIG.read_text())
    configured_fields = (ROOT / "build-gpu-solvers" / config["outputSettings"]["directory"]).resolve()
    if configured_fields != DIRECTORY / "fields":
        raise RuntimeError("Configured field directory must equal the selected output directory / fields")
    if args.detach:
        with (DIRECTORY / "worker.log").open("a") as log:
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--ranks", str(args.ranks),
                                        "--gpu-device", args.gpu_device,
                                        "--config", str(CONFIG), "--output", str(DIRECTORY)],
                                       cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps({"worker_pid": process.pid, "directory": str(DIRECTORY)}))
        return
    restart = config.get("restartSettings", {})
    initial_step = restart.get("completedSteps", 0)
    initial_time = restart.get("physicalTime", 0.)
    exe = ROOT / "build-gpu-solvers/app/acm2EQ_gpu3D.exe"
    environment = os.environ.copy()
    environment.pop("DISPLAY", None)
    environment.update(OMP_NUM_THREADS="8", OPENBLAS_NUM_THREADS="1", HWLOC_COMPONENTS="-opencl",
                       OMPI_MCA_pml="ob1", OMPI_MCA_btl="self,vader,tcp", OMPI_MCA_btl_tcp_if_include="lo",
                       OMPI_MCA_oob_tcp_if_include="lo")
    command = ["mpirun", "--bind-to", "none", "-np", str(args.ranks), str(exe), str(CONFIG), "--backend", "cuda", "--gpu-device", args.gpu_device]
    status = dict(state="starting", worker_pid=os.getpid(), created_utc=datetime.now(timezone.utc).isoformat(),
                  config=str(CONFIG), executable=str(exe), executable_sha256=hashlib.file_digest(exe.open("rb"), "sha256").hexdigest(),
                  requested_physical_steps=initial_step + config["timeMarchSettings"]["nSteps"],
                  initial_physical_steps=initial_step, initial_physical_time=initial_time,
                  additional_physical_steps=config["timeMarchSettings"]["nSteps"],
                  physical_time_step=config["timeMarchSettings"]["physicalTimeStep"],
                  max_inner_iterations=config["timeMarchSettings"]["maxImplicitIterations"],
                  target_physical_time=initial_time + config["timeMarchSettings"]["nSteps"] * config["timeMarchSettings"]["physicalTimeStep"],
                  ranks=args.ranks, gpu_device=args.gpu_device, command=command, completed_physical_steps=initial_step, physical_time=initial_time,
                  log=str(DIRECTORY / "run.log"),
                  restart_flow_file=restart.get("flowFile", ""), reset_bdf_history=restart.get("resetBDFHistory", False),
                  communication="single rank: no halo; multiple ranks: GPU packing with pinned-host MPI transport",
                  scientific_scope=(f"SST/BDF2; configured inner cap={config['timeMarchSettings']['maxImplicitIterations']}; "
                                    f"require inner convergence={config['timeMarchSettings']['dualTimeRequireConvergence']}"))
    save_status(status)
    started = time.monotonic()
    last_step_time = None
    try:
        for attempt in range(3):
            with (DIRECTORY / "run.log").open("w") as log:
                process = subprocess.Popen(command, cwd=ROOT / "build-gpu-solvers", env=environment,
                                           stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                status.update(solver_pid=process.pid, startup_attempt=attempt + 1)
                save_status(status)
                attempt_started = time.monotonic()
                while process.poll() is None:
                    with (DIRECTORY / "run.log").open("rb") as reader:
                        reader.seek(0, 2)
                        reader.seek(max(0, reader.tell() - 128000))
                        tail = reader.read().decode(errors="replace")
                    if "Reading complete ACM" in tail or "CGNS ===" in tail or "Unstructured" in tail:
                        status["state"] = "initializing"
                    matches = list(PATTERN.finditer(tail))
                    if matches:
                        latest = matches[-1].groups()
                        step = int(latest[0])
                        if step > status["completed_physical_steps"]:
                            now = time.monotonic()
                            if last_step_time is not None:
                                status["last_observed_seconds_per_step"] = (now - last_step_time) / (step - status["completed_physical_steps"])
                            last_step_time = now
                        status.update(state="running", completed_physical_steps=step, physical_time=float(latest[1]),
                                      completed_additional_physical_steps=step - initial_step,
                                      bdf_order=int(latest[2]), initial_flow_defect=float(latest[3]), flow_defect=float(latest[4]),
                                      minimum_pseudo_step=float(latest[5]), turbulence_defect=float(latest[6]),
                                      inner_iterations=int(latest[7]), last_inner_converged=bool(int(latest[8])))
                    status["elapsed_seconds"] = time.monotonic() - started
                    status["updated_utc"] = datetime.now(timezone.utc).isoformat()
                    save_status(status)
                    if status["state"] == "starting" and time.monotonic() - attempt_started > 90:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                        break
                    time.sleep(5)
                code = process.wait()
            if status["state"] != "starting" or code == 0:
                break
            (DIRECTORY / "run.log").rename(DIRECTORY / f"mpi_startup_attempt_{attempt + 1}.log")
        status["exit_code"] = code
        if code != 0:
            raise RuntimeError(f"Solver exited with code {code}; inspect {DIRECTORY / 'run.log'}")
        # Read the final diagnostic even when the process exited between polls.
        text = (DIRECTORY / "run.log").read_text(errors="replace")
        matches = list(PATTERN.finditer(text))
        if not matches or int(matches[-1].group(1)) != status["requested_physical_steps"]:
            raise RuntimeError("Solver exited before completing every requested physical step")
        status.update(completed_physical_steps=int(matches[-1].group(1)), physical_time=float(matches[-1].group(2)))
        status["inner_converged_steps"] = sum(int(m.group(9)) for m in matches)
        status["final_field_audit"] = audit_final()
        status["state"] = "completed"
    except Exception as error:
        status.update(state="failed", error=str(error))
        raise
    finally:
        status["elapsed_seconds"] = time.monotonic() - started
        status["updated_utc"] = datetime.now(timezone.utc).isoformat()
        save_status(status)


if __name__ == "__main__":
    main()
