"""Adopt the running RP1 jobs and start the two queued flux jobs on SMT siblings.

No solver is restarted. The handoff identifies existing MPI launchers by PID and
kernel start time. Newly launched jobs retain ordinary child exit codes; adopted
jobs use launcher exit plus a validated final checkpoint as completion evidence.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import resource
import shutil
import signal
import subprocess
import time
import xml.etree.ElementTree as ET

import compare_results as post
from run_suite import ROOT, HERE, DEFAULT, affinity, digest, now, save, topology

STOP = False


def stop(*unused):
    global STOP
    STOP = True


def process_alive(pid, started):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().split()
        return fields[2] != "Z" and fields[21] == started
    except FileNotFoundError:
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT)
    args = parser.parse_args()
    root = args.output.resolve()
    handoff = json.loads((root / "smt_controller_handoff.json").read_text())
    lock = (root / "suite.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    state = handoff["old_state"]
    executable = Path(state["executable"])
    if digest(executable) != state["executable_sha256"]:
        raise RuntimeError("Frozen executable hash mismatch")
    physical = topology()
    slots = [set(range(a, a + 128)) for a in (0, 128, 256, 384)]
    allowed = os.sched_getaffinity(0)
    for slot in slots:
        if not slot.issubset(allowed) or len({physical[c] for c in slot}) != 128:
            raise RuntimeError("Invalid CPU slot")
    environment = os.environ.copy()
    for key in list(environment):
        if key.startswith(("BJ_", "NCFV_")) or key in ("DISPLAY", "WAYLAND_DISPLAY"):
            environment.pop(key, None)
    environment.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       DNDS_DIST_OMP_NUM_THREADS="1", OMP_DYNAMIC="FALSE", HWLOC_COMPONENTS="-gl,-cuda,-nvml")
    _, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(65536, hard), hard))
    tree = ET.fromstring(subprocess.check_output(["lstopo-no-graphics", "--of", "xml", "-"], text=True, env=environment))
    pus = [int(o.attrib["os_index"]) for o in tree.iter("object") if o.attrib.get("type") == "PU"]
    mapping = {cpu: i for i, cpu in enumerate(pus)}
    active = {}
    for name, identity in handoff["adopted_processes"].items():
        active[name] = {"pid": identity["pid"], "starttime": identity["starttime"], "child": None,
                        "milestone": .05 * (int(state["cases"][name]["physical_time"] / .05) + 1),
                        "adopted": True}
    state.update(controller_pid=os.getpid(), status="running", concurrent_cases=4,
                 use_smt_siblings=True, total_logical_cpus=512, total_physical_cores=256,
                 smt_handoff_at=now(), previous_controller_pid=handoff["old_controller_pid"])
    save(root / "suite_status.json", state)
    # Preserve the scheduler used after handoff separately from the original
    # frozen source manifest; the numerical executable/configuration stay fixed.
    shutil.copy2(Path(__file__), root / "inputs/run_suite_smt.py")
    for name, slot_index in (("FC_IDP_BINARY", 2), ("FC_IDP_CONT", 3)):
        status = state["cases"][name]
        if status["status"] != "queued":
            raise RuntimeError("Refusing duplicate start: " + name)
        config = Path(status["configuration"])
        if digest(config) != status["configuration_sha256"]:
            raise RuntimeError("Configuration hash mismatch: " + name)
        cpus = slots[slot_index]
        if affinity(executable, cpus, physical)["rank_count"]:
            raise RuntimeError("Logical CPU slot already contains solver ranks")
        cpu_list = ",".join(str(mapping[c]) for c in sorted(cpus))
        command = ["mpirun", "--mca", "pml", "ob1", "--mca", "btl", "self,vader",
                   "--use-hwthread-cpus", "-np", "128", "--cpu-list", cpu_list,
                   "--bind-to", "cpu-list:ordered", "--report-bindings", str(executable), str(config)]
        log = (root / name / "solver.log").open("x")
        child = subprocess.Popen(command, cwd=ROOT / "build", env=environment, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        active[name] = {"pid": child.pid, "child": child, "adopted": False, "log": log,
                        "starttime": Path(f"/proc/{child.pid}/stat").read_text().split()[21],
                        "milestone": 0, "started_monotonic": time.monotonic()}
        status.update(status="running", mpi_pid=child.pid, started_at=now(), linux_cpu_list=sorted(cpus),
                      hwloc_pu_list=cpu_list, command=command, uses_other_hardware_thread=True,
                      logical_cpus_per_case=128, physical_cores_per_case=128,
                      shares_physical_cores_with="SG_BJ_PRES" if slot_index == 2 else "SG_SIGN_BUDGET")
        save(root / name / "run_status.json", status)
        save(root / "suite_status.json", state)
    progress = re.compile(r"NCFV iteration=(\d+).*?, time=([0-9.eE+-]+), target=([0-9.eE+-]+)")
    try:
        while active and not STOP:
            for name, job in list(active.items()):
                status = state["cases"][name]
                case = root / name
                text = (case / "solver.log").read_text(errors="replace")
                matches = list(progress.finditer(text))
                if matches:
                    m = matches[-1]
                    status.update(iteration=int(m[1]), physical_time=float(m[2]), physical_time_fraction=float(m[2]) / .25)
                status.update(updated_at=now(), rejected_steps=text.count("NCFV rejected iteration="))
                if not job["adopted"]:
                    status["elapsed_seconds"] = time.monotonic() - job["started_monotonic"]
                if not status.get("logical_binding_verified"):
                    evidence = affinity(executable, set(status["linux_cpu_list"]), physical)
                    if evidence["verified"]:
                        save(case / "cpu_affinity.json", evidence)
                        status.update(logical_binding_verified=True, physical_binding_verified=True,
                                      binding_note="128 distinct physical cores within this job; sibling threads may run another job")
                child = job["child"]
                finished = child.poll() is not None if child is not None else not process_alive(job["pid"], job["starttime"])
                checkpoints = sorted(p for p in case.glob("restart_*.dnds.h5") if finished or time.time() - p.stat().st_mtime > 5)
                final_metric = None
                if checkpoints and (finished or status["physical_time"] >= job["milestone"]):
                    try:
                        final_metric = post.export(case, checkpoints[-1])
                        status["last_exported_time"] = final_metric["time"]
                        job["milestone"] = .05 * (int(final_metric["time"] / .05) + 1)
                        post.compare(root)
                    except (OSError, ValueError, KeyError) as error:
                        status["export_retry_reason"] = str(error)
                if finished:
                    reached = final_metric is not None and abs(final_metric["time"] - .25) < 1e-12
                    returncode = child.returncode if child is not None else None
                    complete = reached and (child is None or returncode == 0)
                    status.update(status="completed" if complete else "failed", finished_at=now(), returncode=returncode)
                    if child is None:
                        status["completion_evidence"] = "Adopted launcher exited; final physical checkpoint checked; original exit code unavailable"
                    else:
                        job["log"].close()
                    if complete: status.update(physical_time=.25, physical_time_fraction=1.)
                    del active[name]
                save(case / "run_status.json", status)
            if not (root / "smt_affinity_validation.json").exists() and len(active) == 4:
                evidence = {name: affinity(executable, set(state["cases"][name]["linux_cpu_list"]), physical)
                            for name in active}
                if all(v["verified"] for v in evidence.values()):
                    cpus = [r["logical_cpus"][0] for v in evidence.values() for r in v["ranks"]]
                    if len(cpus) != 512 or len(set(cpus)) != 512:
                        raise RuntimeError("Multiple ranks assigned to the same logical CPU")
                    save(root / "smt_affinity_validation.json", {"verified_at": now(), "logical_cpus": 512,
                         "physical_cores": len({physical[c] for c in cpus}), "logical_cpu_overlap": False,
                         "four_cases": {n: {"ranks": e["rank_count"], "physical_cores_within_case": e["physical_core_count"]}
                                        for n, e in evidence.items()}})
            save(root / "suite_status.json", state)
            if active: time.sleep(5)
    finally:
        if STOP:
            for name, job in active.items():
                if process_alive(job["pid"], job["starttime"]): os.killpg(job["pid"], signal.SIGTERM)
                state["cases"][name].update(status="stopped", stopped_at=now())
                save(root / name / "run_status.json", state["cases"][name])
        state.update(status="stopped" if STOP else ("completed" if all(v["status"] == "completed" for v in state["cases"].values()) else "failed"),
                     controller_updated_at=now())
        save(root / "suite_status.json", state)
        post.compare(root)


if __name__ == "__main__": main()
