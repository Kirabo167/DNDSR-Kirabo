"""Run four named NCFV limiter comparisons, two concurrent 128-physical-core jobs.

The required small-grid gate is the separately frozen preflight result. All jobs
use a frozen executable/configuration and retain logs, physical-time progress,
CPU-affinity evidence and automatic real-layer exports. SIGTERM stops its jobs.
"""
import argparse
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import shutil
import signal
import subprocess
import time
import traceback
import xml.etree.ElementTree as ET
from zoneinfo import ZoneInfo

from compare_results import NAMES, compare, export

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
DEFAULT = ROOT / "data/out/NCFV/RP1_efficient_limiter_derivation_400x400x5_np128_20261029"
STOP = False


def now(): return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def save(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def digest(path):
    with path.open("rb") as f: return hashlib.file_digest(f, "sha256").hexdigest()


def request_stop(*unused):
    global STOP
    STOP = True


def topology():
    result = {}
    for line in subprocess.check_output(["lscpu", "-p=CPU,CORE,SOCKET"], text=True).splitlines():
        if line.startswith("#"): continue
        cpu, core, socket = map(int, line.split(","))
        result[cpu] = (socket, core)
    return result


def affinity(executable, cpus, physical):
    ranks = []
    for p in Path("/proc").iterdir():
        if not p.name.isdigit(): continue
        try:
            if (p / "exe").resolve() != executable: continue
            environment = (p / "environ").read_bytes().split(b"\0")
            rank = next(int(v.split(b"=", 1)[1]) for v in environment if v.startswith(b"OMPI_COMM_WORLD_RANK="))
            assigned = sorted(os.sched_getaffinity(int(p.name)))
            ranks.append({"rank": rank, "pid": int(p.name), "logical_cpus": assigned})
        except (OSError, StopIteration, ValueError): continue
    # Both jobs share the frozen binary; filter by this slot's CPU allocation.
    ranks = [r for r in ranks if set(r["logical_cpus"]).issubset(cpus)]
    actual = [r["logical_cpus"][0] for r in ranks if len(r["logical_cpus"]) == 1]
    valid = (len(ranks) == 128 and len(actual) == 128 and len(set(actual)) == 128 and
             len({physical[c] for c in actual}) == 128 and {r["rank"] for r in ranks} == set(range(128)))
    return {"verified": valid, "rank_count": len(ranks), "physical_core_count": len({physical[c] for c in actual}), "ranks": ranks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / "suite.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    if (output / "suite_status.json").exists():
        raise FileExistsError("Existing suite status; do not overwrite or silently duplicate jobs")
    physical = topology()
    # OpenMPI --cpu-list uses hwloc logical PU indices, not Linux CPU IDs.
    # Enumerate the same hwloc tree and map every selected Linux CPU explicitly.
    hardware = ET.fromstring(subprocess.check_output(["lstopo-no-graphics", "--of", "xml", "-"], text=True))
    pu_order = [int(obj.attrib["os_index"]) for obj in hardware.iter("object") if obj.attrib.get("type") == "PU"]
    logical_pu = {cpu: pu for pu, cpu in enumerate(pu_order)}
    slots = [set(range(128)), set(range(128, 256))]
    for cpus in slots:
        if not cpus.issubset(os.sched_getaffinity(0)) or len({physical[c] for c in cpus}) != 128:
            raise RuntimeError("Requested 128 independent physical CPUs are not available")
    if {physical[c] for c in slots[0]} & {physical[c] for c in slots[1]}:
        raise RuntimeError("The two slots overlap physical cores")
    inputs = output / "inputs"
    inputs.mkdir(exist_ok=True)
    source_exe = ROOT / "build/app/ncfv_euler3D.exe"
    executable = inputs / ("solver_" + digest(source_exe)[:16] + ".exe")
    shutil.copy2(source_exe, executable)
    frozen = {}
    sources = list((ROOT / "src/NCFV").glob("*.hpp")) + list((ROOT / "src/NCFV").glob("*.cpp"))
    sources += list(HERE.glob("*.py")) + [HERE / "variants.json"]
    for path in sources:
        destination = inputs / path.name
        shutil.copy2(path, destination)
        frozen[path.name] = {"source": str(path), "sha256": digest(destination)}
    state = {"status": "waiting_preflight", "created_at": now(), "controller_pid": os.getpid(),
             "target_time": .25, "ranks_per_case": 128, "physical_cores_per_case": 128,
             "concurrent_cases": 2, "executable": str(executable), "executable_sha256": digest(executable),
             "frozen_sources": frozen, "cases": {}}
    for name in NAMES:
        case = output / name
        case.mkdir(exist_ok=True)
        config = inputs / (name + ".json")
        shutil.copy2(HERE / (name + ".json"), config)
        c = json.loads(config.read_text())
        for pointer in ("meshFile",):
            path = (ROOT / "build" / c["mesh"][pointer]).resolve()
            state.setdefault("input_files", {})[str(path)] = digest(path)
        initial = (ROOT / "build" / c["initialField"]["nodeFile"]).resolve()
        state.setdefault("input_files", {})[str(initial)] = digest(initial)
        state["cases"][name] = {"status": "queued", "configuration": str(config),
                                "configuration_sha256": digest(config), "physical_time": 0., "iteration": 0}
        save(case / "run_status.json", state["cases"][name])
    save(output / "suite_status.json", state)
    gate = output / "diagnostics/preflight/results.json"
    while not STOP:
        if gate.exists():
            result = json.loads(gate.read_text())
            if result.get("status") == "failed":
                state.update(status="preflight_failed", finished_at=now())
                save(output / "suite_status.json", state)
                return
            if result.get("status") == "complete":
                expected = {(name, problem, cells) for name in NAMES for problem in ("RP1", "planar_shock", "contact", "smooth")
                            for cells in ((20, 40) if problem == "smooth" else (40,))}
                actual = {(r["variant"], r["problem"], r["cells"]) for r in result["runs"] if r["returncode"] == 0}
                if actual != expected: raise RuntimeError("Incomplete preflight matrix")
                # A successful return code alone is insufficient: the preflight
                # binary must be byte-identical to the production executable.
                binaries = {v["frozen_executable"] for v in result["variants"]}
                if any(digest(Path(p)) != state["executable_sha256"] for p in binaries):
                    raise RuntimeError("Preflight and production executable differ")
                state["preflight_result_sha256"] = digest(gate)
                break
        time.sleep(10)
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(65536, hard), hard))
    environment = os.environ.copy()
    for key in list(environment):
        if key.startswith(("BJ_", "NCFV_")) or key in ("DISPLAY", "WAYLAND_DISPLAY"):
            environment.pop(key, None)
    environment.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       DNDS_DIST_OMP_NUM_THREADS="1", OMP_DYNAMIC="FALSE", HWLOC_COMPONENTS="-gl,-cuda,-nvml")
    active, pending = {}, list(NAMES)
    pattern = re.compile(r"NCFV iteration=(\d+).*?, time=([0-9.eE+-]+), target=([0-9.eE+-]+)")
    state["status"] = "running" if not STOP else "stopped"
    try:
        while (active or pending) and not STOP:
            for slot in range(2):
                if slot in active or not pending: continue
                name = pending.pop(0)
                case = output / name
                log = (case / "solver.log").open("x", buffering=1)
                linux_cpus = sorted(slots[slot])
                cpu_list = ",".join(str(logical_pu[cpu]) for cpu in linux_cpus)
                command = ["mpirun", "--mca", "pml", "ob1", "--mca", "btl", "self,vader",
                           "--use-hwthread-cpus", "-np", "128", "--cpu-list", cpu_list,
                           "--bind-to", "cpu-list:ordered", "--report-bindings", str(executable),
                           state["cases"][name]["configuration"]]
                process = subprocess.Popen(command, cwd=ROOT / "build", env=environment, stdin=subprocess.DEVNULL,
                                           stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                active[slot] = {"name": name, "process": process, "log": log, "milestone": 0, "start": time.monotonic()}
                state["cases"][name].update(status="running", mpi_pid=process.pid, started_at=now(),
                                             hwloc_pu_list=cpu_list, linux_cpu_list=linux_cpus, command=command)
            for slot, job in list(active.items()):
                name, process = job["name"], job["process"]
                case, status = output / name, state["cases"][name]
                text = (case / "solver.log").read_text(errors="replace")
                matches = list(pattern.finditer(text))
                if matches:
                    m = matches[-1]
                    status.update(iteration=int(m[1]), physical_time=float(m[2]), physical_time_fraction=float(m[2]) / .25)
                status.update(rejected_steps=text.count("NCFV rejected iteration="), elapsed_seconds=time.monotonic() - job["start"], updated_at=now())
                binding_path = case / "cpu_affinity.json"
                if not binding_path.exists():
                    evidence = affinity(executable, slots[slot], physical)
                    if evidence["verified"]:
                        save(binding_path, evidence)
                        status["physical_binding_verified"] = True
                checkpoints = sorted(p for p in case.glob("restart_*.dnds.h5") if time.time() - p.stat().st_mtime > 5)
                finished = process.poll() is not None
                if checkpoints and (status["physical_time"] >= job["milestone"] or finished):
                    try:
                        metric = export(case, checkpoints[-1])
                        status["last_exported_time"] = metric["time"]
                        job["milestone"] = .05 * (int(metric["time"] / .05) + 1)
                        compare(output)
                    except (OSError, ValueError, KeyError) as error:
                        status["export_retry_reason"] = str(error)
                if finished:
                    job["log"].close()
                    # Wait for completed final HDF5 metadata, then require the
                    # target time instead of inferring completion from logs.
                    time.sleep(1)
                    checkpoints = sorted(case.glob("restart_*.dnds.h5"))
                    try:
                        metric = export(case, checkpoints[-1]) if checkpoints else None
                        completed = process.returncode == 0 and metric is not None and abs(metric["time"] - .25) < 1e-12
                    except Exception as error:
                        completed = False
                        status["final_export_error"] = str(error)
                    status.update(status="completed" if completed else "failed", returncode=process.returncode, finished_at=now())
                    if completed:
                        status.update(physical_time=.25, physical_time_fraction=1.)
                    del active[slot]
                    compare(output)
                save(case / "run_status.json", status)
            save(output / "suite_status.json", state)
            if active: time.sleep(10)
    finally:
        for job in active.values():
            if job["process"].poll() is None:
                os.killpg(job["process"].pid, signal.SIGTERM)
        for job in active.values():
            try: job["process"].wait(timeout=30)
            except subprocess.TimeoutExpired: os.killpg(job["process"].pid, signal.SIGKILL)
            state["cases"][job["name"]].update(status="stopped", stopped_at=now())
            save(output / job["name"] / "run_status.json", state["cases"][job["name"]])
        state.update(status="stopped" if STOP else ("completed" if all(c["status"] == "completed" for c in state["cases"].values()) else "failed"), finished_at=now())
        save(output / "suite_status.json", state)
        compare(output)


if __name__ == "__main__":
    try: main()
    except Exception:
        traceback.print_exc()
        raise
