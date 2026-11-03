"""Keep two existing physical-core jobs and queue all remaining cases.

Adopted launcher identity includes kernel start time. Its original exit code is
unavailable, so completion requires launcher exit and a validated final state.
"""
import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import re
import resource
import shutil
import signal
import statistics
import subprocess
import time
import traceback
import xml.etree.ElementTree as ET

import compare_results as post
from legacy_runtime import ROOT, affinity, digest, now, save, topology

DEFAULT = ROOT / "data/out/NCFV/RP1_efficient_limiter_derivation_800x800x5_np128_smt_20261029"
STOP = False


def stop(*unused):
    global STOP
    STOP = True


def alive(pid, starttime):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return fields[0] != "Z" and fields[19] == starttime
    except FileNotFoundError:
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT)
    root = parser.parse_args().output.resolve()
    lock = (root / "suite.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    handoff = json.loads((root / "smt_to_physical_queue_handoff_20261029.json").read_text())
    state = handoff["old_state"]
    executable = Path(state["executable"])
    if digest(executable) != state["executable_sha256"]:
        raise RuntimeError("Frozen executable hash mismatch")
    environment = os.environ.copy()
    for key in list(environment):
        if key.startswith(("BJ_", "NCFV_")) or key in ("DISPLAY", "WAYLAND_DISPLAY"):
            environment.pop(key, None)
    environment.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       DNDS_DIST_OMP_NUM_THREADS="1", OMP_DYNAMIC="FALSE", HWLOC_COMPONENTS="-gl,-cuda,-nvml")
    _, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(65536, hard), hard))
    physical = topology()
    slots = [set(range(start, start + 128)) for start in (0, 128)]
    if len({physical[c] for cpus in slots for c in cpus}) != 256:
        raise RuntimeError("Physical-core slots overlap")
    for cpus in slots:
        if not cpus.issubset(os.sched_getaffinity(0)) or len({physical[c] for c in cpus}) != 128:
            raise RuntimeError("Invalid 128-core slot")
    tree = ET.fromstring(subprocess.check_output(["lstopo-no-graphics", "--of", "xml", "-"], text=True, env=environment))
    pus = [int(o.attrib["os_index"]) for o in tree.iter("object") if o.attrib.get("type") == "PU"]
    mapping = {cpu: pu for pu, cpu in enumerate(pus)}
    active = {}
    for name, identity in handoff["adopted_processes"].items():
        if not alive(identity["pid"], identity["starttime"]):
            raise RuntimeError("Adopted launcher identity changed: " + name)
        slot = state["cases"][name]["slot"]
        active[slot] = dict(name=name, pid=identity["pid"], starttime=identity["starttime"], child=None, log=None)
    pending = [name for name in post.NAMES if state["cases"][name]["status"] == "queued"]
    for name in pending:
        status = state["cases"][name]
        if digest(Path(status["configuration"])) != status["configuration_sha256"]:
            raise RuntimeError("Frozen configuration changed: " + name)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    state.update(status="running", controller_pid=os.getpid(), previous_controller_pid=handoff["old_controller_pid"],
                 concurrent_cases=2, use_smt_siblings=False, total_logical_cpus=256, total_physical_cores=256,
                 physical_queue_enabled_at=now())
    shutil.copy2(Path(__file__), root / "inputs_physical_20261029/run_suite_queue_physical.py")
    pattern = re.compile(r"NCFV iteration=(\d+).*?dt\[min,max\]=\[([0-9.eE+-]+),[0-9.eE+-]+\], time=([0-9.eE+-]+), target=([0-9.eE+-]+), step wall\[s\]=([0-9.eE+-]+)")
    try:
        while (active or pending) and not STOP:
            if shutil.disk_usage(root).free < 10 * 1024**3:
                raise RuntimeError("Disk free below 10 GiB")
            for slot in range(2):
                if slot in active or not pending or STOP:
                    continue
                name = pending.pop(0)
                status = state["cases"][name]
                cpus = slots[slot]
                if affinity(executable, cpus, physical)["rank_count"]:
                    raise RuntimeError("CPU slot already contains solver ranks")
                pu_list = ",".join(str(mapping[c]) for c in sorted(cpus))
                command = ["mpirun", "--mca", "pml", "ob1", "--mca", "btl", "self,vader", "--use-hwthread-cpus",
                           "-np", "128", "--cpu-list", pu_list, "--bind-to", "cpu-list:ordered", "--report-bindings",
                           str(executable), status["configuration"]]
                log = (root / name / "solver.log").open("x")
                child = subprocess.Popen(command, cwd=ROOT / "build", env=environment, stdin=subprocess.DEVNULL,
                                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                starttime = Path(f"/proc/{child.pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
                active[slot] = dict(name=name, pid=child.pid, starttime=starttime, child=child, log=log)
                status.update(status="running", mpi_pid=child.pid, started_at=now(), slot=slot,
                              linux_cpu_list=sorted(cpus), hwloc_pu_list=pu_list, command=command,
                              logical_binding_verified=False, physical_binding_verified=False)
                save(root / name / "run_status.json", status)
                print(now(), "started", name, "PID", child.pid, "Linux CPUs", min(cpus), max(cpus), flush=True)
            for slot, job in list(active.items()):
                name = job["name"]
                case, status = root / name, state["cases"][name]
                text = (case / "solver.log").read_text(errors="replace")
                rows = list(pattern.finditer(text))
                if rows:
                    r = rows[-1]
                    status.update(iteration=int(r[1]), physical_time=float(r[3]), physical_time_fraction=float(r[3]) / .25,
                                  last_reported_dt=float(r[2]), last_reported_step_seconds=float(r[5]))
                    recent = [float(r[5]) for r in rows[-30:] if int(r[1]) >= 100]
                    if recent:
                        status["recent_step_median_seconds"] = statistics.median(recent)
                        status["remaining_compute_seconds_estimate"] = (.25 - status["physical_time"]) / status["last_reported_dt"] * statistics.median(recent)
                status.update(updated_at=now(), rejected_steps=text.count("NCFV rejected iteration="),
                              elapsed_seconds=(datetime.fromisoformat(now()) - datetime.fromisoformat(status["started_at"])).total_seconds())
                if not status.get("logical_binding_verified"):
                    evidence = affinity(executable, slots[slot], physical)
                    if evidence["verified"]:
                        save(case / "cpu_affinity.json", evidence)
                        status.update(logical_binding_verified=True, physical_binding_verified=True)
                child = job["child"]
                finished = child.poll() is not None if child is not None else not alive(job["pid"], job["starttime"])
                if finished:
                    returncode = child.returncode if child is not None else None
                    status.update(solver_finished_at=now(), returncode=returncode)
                    try:
                        checkpoints = sorted(case.glob("restart_*.dnds.h5"))
                        metric = post.export(case, checkpoints[-1]) if checkpoints else None
                        complete = metric is not None and abs(metric["time"] - .25) < 1e-12 and (child is None or returncode == 0)
                        if metric:
                            status.update(physical_time=metric["time"], physical_time_fraction=metric["time"] / .25, iteration=metric["iteration"])
                    except Exception:
                        complete = False
                        status["final_validation_error"] = traceback.format_exc()
                    status.update(status="completed" if complete else "failed", finished_at=now())
                    if child is None:
                        status["completion_evidence"] = "Adopted launcher exited; final checkpoint validated; original exit code unavailable"
                    if job["log"] is not None:
                        job["log"].close()
                    del active[slot]
                    print(now(), name, status["status"], flush=True)
                save(case / "run_status.json", status)
            validation = root / "physical_queue_affinity_validation_20261029.json"
            if len(active) == 2 and not validation.exists():
                evidence = {j["name"]: affinity(executable, slots[slot], physical) for slot, j in active.items()}
                if all(e["verified"] for e in evidence.values()):
                    cpus = [r["logical_cpus"][0] for e in evidence.values() for r in e["ranks"]]
                    if len(cpus) != 256 or len(set(cpus)) != 256 or len({physical[c] for c in cpus}) != 256:
                        raise RuntimeError("Logical or physical CPU overlap")
                    save(validation, dict(verified_at=now(), logical_cpus=256, physical_cores=len({physical[c] for c in cpus}),
                                          logical_cpu_overlap=False, cases={n: {"ranks": e["rank_count"]} for n, e in evidence.items()}))
            state["updated_at"] = now()
            save(root / "suite_status.json", state)
            if active:
                time.sleep(5)
        state["status"] = "stopped" if STOP else ("completed" if all(s["status"] == "completed" for s in state["cases"].values()) else "failed")
    except BaseException:
        state.update(status="failed", error=traceback.format_exc())
        raise
    finally:
        for job in active.values():
            if alive(job["pid"], job["starttime"]):
                os.killpg(job["pid"], signal.SIGTERM)
        deadline = time.monotonic() + 30
        while any(alive(j["pid"], j["starttime"]) for j in active.values()) and time.monotonic() < deadline:
            time.sleep(1)
        for job in active.values():
            if alive(job["pid"], job["starttime"]):
                os.killpg(job["pid"], signal.SIGKILL)
            if job["log"] is not None:
                job["log"].close()
            status = state["cases"][job["name"]]
            status.update(status="stopped", stopped_at=now())
            save(root / job["name"] / "run_status.json", status)
        state["finished_at"] = now()
        save(root / "suite_status.json", state)
    post.compare(root)


if __name__ == "__main__":
    main()
