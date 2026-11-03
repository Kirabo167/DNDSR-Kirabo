"""Run the five frozen RP1 variants on four disjoint 128-PU SMT slots.

The fifth variant starts when a slot is released. SIGTERM stops only this
controller's process groups. Numerical completion requires a valid checkpoint.
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
import statistics
import subprocess
import time
import traceback
import xml.etree.ElementTree as ET

import compare_results as post
from legacy_runtime import ROOT, HERE, affinity, digest, now, save, topology

PREVIOUS = ROOT / "data/out/NCFV/RP1_efficient_limiter_derivation_400x400x5_np128_20261029"
DEFAULT = ROOT / "data/out/NCFV/RP1_efficient_limiter_derivation_800x800x5_np128_smt_20261029"
STOP = False


def request_stop(*unused):
    global STOP
    STOP = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / "suite.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (output / "suite_status.json").exists():
        raise FileExistsError("Existing suite: refusing duplicate launch")
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    environment = os.environ.copy()
    for key in list(environment):
        if key.startswith(("BJ_", "NCFV_")) or key in ("DISPLAY", "WAYLAND_DISPLAY"):
            environment.pop(key, None)
    environment.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       DNDS_DIST_OMP_NUM_THREADS="1", OMP_DYNAMIC="FALSE", HWLOC_COMPONENTS="-gl,-cuda,-nvml")
    _, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(65536, hard), hard))
    if shutil.disk_usage(output).free < 175 * 1024**3:
        raise RuntimeError("Need 175 GiB available for approximately 150 GiB of output plus reserve")
    physical = topology()
    slots = [set(range(start, start + 128)) for start in (0, 128, 256, 384)]
    hardware = ET.fromstring(subprocess.check_output(["lstopo-no-graphics", "--of", "xml", "-"], text=True, env=environment))
    pus = [int(obj.attrib["os_index"]) for obj in hardware.iter("object") if obj.attrib.get("type") == "PU"]
    mapping = {cpu: logical_pu for logical_pu, cpu in enumerate(pus)}
    for cpus in slots:
        if not cpus.issubset(os.sched_getaffinity(0)) or len({physical[c] for c in cpus}) != 128:
            raise RuntimeError("Each slot must contain 128 distinct available physical cores")
    if len(set.union(*slots)) != 512 or len({physical[c] for c in set.union(*slots)}) != 256:
        raise RuntimeError("Expected 512 disjoint logical CPUs over 256 physical cores")
    previous_state = json.loads((PREVIOUS / "suite_status.json").read_text())
    old_executable = Path(previous_state["executable"])
    binary_sha256 = digest(old_executable)
    if binary_sha256 != previous_state["executable_sha256"]:
        raise RuntimeError("Previous executable hash mismatch")
    preflight_path = PREVIOUS / "diagnostics/preflight/results.json"
    preflight = json.loads(preflight_path.read_text())
    if preflight["status"] != "complete" or any(digest(Path(v["frozen_executable"])) != binary_sha256 for v in preflight["variants"]):
        raise RuntimeError("Previous numerical preflight does not match frozen executable")
    inputs = output / "inputs"
    inputs.mkdir()
    executable = inputs / old_executable.name
    shutil.copy2(old_executable, executable)
    frozen_sources = inputs / "solver_sources"
    frozen_sources.mkdir()
    for source in (PREVIOUS / "inputs").iterdir():
        if source.suffix in (".cpp", ".hpp"):
            shutil.copy2(source, frozen_sources / source.name)
    for source in HERE.glob("*.py"):
        shutil.copy2(source, inputs / source.name)
    shutil.copy2(preflight_path, inputs / "previous_preflight_results.json")
    shutil.copy2(PREVIOUS / "suite_status.json", inputs / "previous_suite_status.json")
    state = dict(status="preparing", created_at=now(), controller_pid=os.getpid(), target_time=.25,
                 mesh_cells=[800, 800, 5], spacing=[.00125, .00125, .01], ranks_per_case=128,
                 concurrent_cases=4, use_smt_siblings=True, total_logical_cpus=512, total_physical_cores=256,
                 executable=str(executable), executable_sha256=binary_sha256, previous_suite=str(PREVIOUS),
                 previous_preflight_sha256=digest(preflight_path), linux_to_hwloc_pu=mapping, cases={})
    active = {}
    pending = list(post.NAMES)
    pattern = re.compile(r"NCFV iteration=(\d+).*?dt\[min,max\]=\[([0-9.eE+-]+),[0-9.eE+-]+\], time=([0-9.eE+-]+), target=([0-9.eE+-]+), step wall\[s\]=([0-9.eE+-]+)")
    try:
        for name in post.NAMES:
            case = output / name
            case.mkdir()
            config = inputs / (name + ".json")
            shutil.copy2(HERE / config.name, config)
            current = json.loads(config.read_text())
            baseline_status = json.loads((PREVIOUS / name / "run_status.json").read_text())
            baseline = json.loads(Path(baseline_status["configuration"]).read_text())
            if baseline_status["status"] != "completed":
                raise RuntimeError("Previous case was not completed: " + name)
            # Compare the actual frozen input, permitting only grid and output changes.
            for section in current:
                if section not in ("mesh", "initialField", "io") and current[section] != baseline[section]:
                    raise RuntimeError("Unexpected numerical parameter change: " + name + "/" + section)
            for section, changed_keys in (("mesh", {"meshFile"}), ("initialField", {"nodeFile"}),
                                          ("io", {"outputInterval", "outputPrefix", "vtkSeriesName", "restartInterval", "restartPrefix"})):
                if {k: v for k, v in current[section].items() if k not in changed_keys} != {k: v for k, v in baseline[section].items() if k not in changed_keys}:
                    raise RuntimeError("Unexpected boundary or output setting change: " + name)
            check = subprocess.run([str(executable), str(config), "--check-config"], cwd=ROOT / "build", env=environment,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=90)
            (case / "config_check.log").write_text(check.stdout)
            if check.returncode:
                raise RuntimeError("Configuration check failed: " + name)
            status = dict(status="queued", configuration=str(config), configuration_sha256=digest(config),
                          physical_time=0., iteration=0, mpi_ranks=128, baseline_configuration_sha256=digest(Path(baseline_status["configuration"])))
            state["cases"][name] = status
            save(case / "run_status.json", status)
        sample = json.loads((inputs / "SG_BJ_PRES.json").read_text())
        state["input_files"] = {}
        for path in ((ROOT / "build" / sample["mesh"]["meshFile"]).resolve(), (ROOT / "build" / sample["initialField"]["nodeFile"]).resolve()):
            state["input_files"][str(path)] = digest(path)
        save(output / "suite_status.json", state)
        state["status"] = "running"
        while (active or pending) and not STOP:
            if shutil.disk_usage(output).free < 10 * 1024**3:
                raise RuntimeError("Available disk below 10 GiB; stopping own jobs before filesystem fills")
            for slot in range(4):
                if slot in active or not pending or STOP:
                    continue
                name = pending.pop(0)
                status = state["cases"][name]
                cpus = slots[slot]
                if affinity(executable, cpus, physical)["rank_count"]:
                    raise RuntimeError("Slot already contains ranks; refusing duplicate launch")
                pu_list = ",".join(str(mapping[c]) for c in sorted(cpus))
                command = ["mpirun", "--mca", "pml", "ob1", "--mca", "btl", "self,vader", "--use-hwthread-cpus",
                           "-np", "128", "--cpu-list", pu_list, "--bind-to", "cpu-list:ordered", "--report-bindings",
                           str(executable), status["configuration"]]
                log = (output / name / "solver.log").open("x")
                process = subprocess.Popen(command, cwd=ROOT / "build", env=environment, stdin=subprocess.DEVNULL,
                                           stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                active[slot] = dict(name=name, process=process, log=log, start=time.monotonic())
                status.update(status="running", mpi_pid=process.pid, started_at=now(), command=command,
                              slot=slot, linux_cpu_list=sorted(cpus), hwloc_pu_list=pu_list)
                save(output / name / "run_status.json", status)
                print(now(), "started", name, "PID", process.pid, "Linux CPUs", min(cpus), max(cpus), flush=True)
            for slot, job in list(active.items()):
                name, process = job["name"], job["process"]
                case, status = output / name, state["cases"][name]
                text = (case / "solver.log").read_text(errors="replace")
                matches = list(pattern.finditer(text))
                if matches:
                    m = matches[-1]
                    status.update(iteration=int(m[1]), physical_time=float(m[3]), physical_time_fraction=float(m[3]) / .25,
                                  last_reported_dt=float(m[2]), last_reported_step_seconds=float(m[5]))
                    recent = [float(m[5]) for m in matches if int(m[1]) >= 100][-30:]
                    if recent:
                        status["recent_step_median_seconds"] = statistics.median(recent)
                        status["remaining_compute_seconds_estimate"] = (.25 - status["physical_time"]) / status["last_reported_dt"] * statistics.median(recent)
                status.update(updated_at=now(), elapsed_seconds=time.monotonic() - job["start"], rejected_steps=text.count("NCFV rejected iteration="))
                if not status.get("logical_binding_verified"):
                    evidence = affinity(executable, slots[slot], physical)
                    if evidence["verified"]:
                        save(case / "cpu_affinity.json", evidence)
                        status.update(logical_binding_verified=True, physical_binding_verified=True)
                if process.poll() is not None:
                    job["log"].close()
                    status.update(returncode=process.returncode, solver_finished_at=now())
                    try:
                        checkpoints = sorted(case.glob("restart_*.dnds.h5"))
                        metric = post.export(case, checkpoints[-1]) if checkpoints else None
                        complete = process.returncode == 0 and metric is not None and abs(metric["time"] - .25) < 1e-12
                        if metric:
                            status.update(physical_time=metric["time"], iteration=metric["iteration"], physical_time_fraction=metric["time"] / .25)
                    except Exception:
                        complete = False
                        status["final_validation_error"] = traceback.format_exc()
                    status.update(status="completed" if complete else "failed", finished_at=now())
                    del active[slot]
                    print(now(), name, status["status"], "returncode", process.returncode, flush=True)
                save(case / "run_status.json", status)
            if len(active) == 4 and not (output / "smt_affinity_validation.json").exists():
                evidence = {job["name"]: affinity(executable, slots[slot], physical) for slot, job in active.items()}
                if all(e["verified"] for e in evidence.values()):
                    actual = [r["logical_cpus"][0] for e in evidence.values() for r in e["ranks"]]
                    if len(actual) != 512 or len(set(actual)) != 512:
                        raise RuntimeError("Logical CPU overlap detected")
                    save(output / "smt_affinity_validation.json", dict(verified_at=now(), logical_cpus=512,
                         physical_cores=len({physical[c] for c in actual}), logical_cpu_overlap=False,
                         cases={n: {"ranks": e["rank_count"], "physical_cores_within_case": e["physical_core_count"]} for n, e in evidence.items()}))
            state["updated_at"] = now()
            save(output / "suite_status.json", state)
            if active:
                time.sleep(5)
        state["status"] = "stopped" if STOP else ("completed" if all(s["status"] == "completed" for s in state["cases"].values()) else "failed")
    except BaseException:
        state.update(status="failed", error=traceback.format_exc())
        raise
    finally:
        for job in active.values():
            process = job["process"]
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for job in active.values():
            process = job["process"]
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
            job["log"].close()
            status = state["cases"][job["name"]]
            status.update(status="stopped", stopped_at=now(), returncode=process.returncode)
            save(output / job["name"] / "run_status.json", status)
        state["finished_at"] = now()
        save(output / "suite_status.json", state)
    post.compare(output)


if __name__ == "__main__":
    main()
