"""Run a frozen solver and retain its log, exit status, and physical-time progress."""

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
from zoneinfo import ZoneInfo


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cwd", type=Path, required=True)
    parser.add_argument("--ranks", type=int, default=64)
    parser.add_argument("--cpu-list", default="144-159,192-223,240-255")
    args = parser.parse_args()
    executable, config, output, cwd = (
        path.resolve() for path in (args.executable, args.config, args.output, args.cwd)
    )
    output.mkdir(parents=True, exist_ok=True)
    status_file = output / "run_status.json"
    log_file = output / "solver.log"
    if status_file.exists() or log_file.exists():
        raise FileExistsError("Run status or log already exists; use a new output directory")
    case = json.loads(config.read_text())
    target = case["time"]["endTime"]
    command = [
        "mpirun", "--mca", "pml", "ob1", "--mca", "btl", "self,vader",
        "-np", str(args.ranks), "--cpu-list", args.cpu_list,
        "--bind-to", "cpu-list:ordered", "--report-bindings",
        str(executable), str(config),
    ]
    environment = os.environ.copy()
    environment.pop("DISPLAY", None)
    environment.pop("WAYLAND_DISPLAY", None)
    environment.update({
        "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
    })
    started = time.monotonic()
    status = {
        "status": "starting",
        "started_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "supervisor_pid": os.getpid(),
        "command": command,
        "cwd": str(cwd),
        "mpi_ranks": args.ranks,
        "threads_per_rank": 1,
        "cpu_list": args.cpu_list,
        "executable_sha256": digest(executable),
        "configuration_sha256": digest(config),
        "target_physical_time": target,
        "physical_time": 0.0,
        "physical_time_fraction": 0.0,
        "iteration": 0,
        "rejected_steps": 0,
        "log": str(log_file),
    }

    def save():
        status["elapsed_seconds"] = time.monotonic() - started
        temporary = status_file.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(status_file)

    save()
    progress = re.compile(
        r"NCFV iteration=(\d+).*?, time=([0-9.eE+\-]+), target=([0-9.eE+\-]+)"
    )
    with log_file.open("x", buffering=1) as log:
        process = subprocess.Popen(
            command, cwd=cwd, env=environment, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        status.update(status="running", mpi_pid=process.pid)
        save()
        for line in process.stdout:
            log.write(line)
            match = progress.search(line)
            if match:
                status.update(iteration=int(match[1]), physical_time=float(match[2]))
                status["physical_time_fraction"] = status["physical_time"] / target
                status["last_progress_at"] = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
                save()
            elif "NCFV rejected iteration=" in line:
                status["rejected_steps"] += 1
                save()
        returncode = process.wait()

    status.update(
        status="completed" if returncode == 0 else "failed", returncode=returncode,
        finished_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
    )
    # The last reported step need not coincide with the final checkpoint.
    checkpoints = sorted(output.glob("restart_*.dnds.h5"))
    if checkpoints:
        try:
            import h5py

            with h5py.File(checkpoints[-1], "r") as checkpoint:
                status.update(
                    final_checkpoint=str(checkpoints[-1]),
                    physical_time=float(checkpoint.attrs["simulationTime"]),
                    iteration=int(checkpoint.attrs["iteration"]),
                )
                status["physical_time_fraction"] = status["physical_time"] / target
        except (OSError, KeyError, ImportError) as error:
            status["checkpoint_metadata_error"] = str(error)
    save()


if __name__ == "__main__":
    main()
