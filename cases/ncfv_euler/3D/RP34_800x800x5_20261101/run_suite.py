"""Run RP3/RP4 with five frozen NCFV methods, two 128-physical-core slots."""
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

import numpy as np
import compare_results as post
from legacy_runtime import ROOT, HERE, affinity, digest, now, save, topology

PREVIOUS = ROOT / "data/out/NCFV/KT2002_Config3_800x800x5_np128_20261031"
OUTPUT = ROOT / "data/out/NCFV/RP34_800x800x5_np128_20261101"
TARGETS = {"RP3": .3, "RP4": .2}
JOBS = [(problem, method) for method in post.NAMES for problem in TARGETS]
STOP = False


def stop(*unused):
    global STOP
    STOP = True


def command(executable, config, cpus, mapping):
    return ["mpirun", "--mca", "pml", "ob1", "--mca", "btl", "self,vader", "--use-hwthread-cpus",
            "-np", str(len(cpus)), "--cpu-list", ",".join(str(mapping[c]) for c in sorted(cpus)),
            "--bind-to", "cpu-list:ordered", "--report-bindings", str(executable), str(config)]


def preflight(executable, inputs, environment, mapping, output):
    directory = output / "preflight"
    directory.mkdir()
    result = dict(status="running", started_at=now(), executable_sha256=digest(executable),
                  mesh_cells=[40, 40, 5], depth=1., spacing_ratio_dz_dx=8., targets=TARGETS, runs=[])
    save(directory / "results.json", result)
    for problem, method in JOBS:
        case = directory / problem / method
        case.mkdir(parents=True)
        overrides = {
            "/mesh/meshFile": str(ROOT / "data/mesh/KT2002_Config13_preflight40x40x5_Lz1.cgns"),
            "/mesh/periodicLengths": [0, 0, 1.],
            "/initialField/nodeFile": str(ROOT / f"data/mesh/RP34_preflight40x40x5_initial/{problem}.csv"),
            "/time/maximumTimeStep": .002, "/time/reportInterval": 50,
            "/io/writeVTK": False, "/io/writeInitial": False, "/io/writeFinal": False,
            "/io/restartInterval": 0, "/io/writeFinalRestart": True,
            "/io/outputPrefix": str(case / "solution"), "/io/vtkSeriesName": str(case / "series"),
            "/io/restartPrefix": str(case / "restart"),
        }
        config = inputs / problem / (method + ".json")
        args = command(executable, config, set(range(4)), mapping)
        for key, value in overrides.items():
            args += ["-k", key, "-v", json.dumps(value)]
        save(case / "command.json", dict(command=args))
        started = time.monotonic()
        with (case / "solver.log").open("x") as log:
            process = subprocess.Popen(args, cwd=ROOT / "build", env=environment, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                while process.poll() is None:
                    if STOP or time.monotonic() - started > 600:
                        raise RuntimeError("Preflight interrupted or exceeded 600 seconds: " + problem + "/" + method)
                    time.sleep(1)
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=10)
        if process.returncode:
            raise RuntimeError("Preflight solver failed: " + problem + "/" + method)
        checkpoint = sorted(case.glob("restart_*.dnds.h5"))[-1]
        fields, t, iteration, actual = post.read_checkpoint(checkpoint)
        expected = json.loads(config.read_text())
        if abs(t - TARGETS[problem]) > 1e-12 or fields["rho"].shape != (5, 41, 41):
            raise RuntimeError("Preflight time or independent node shape mismatch")
        if any(actual["reconstruction"][k] != v for k, v in expected["reconstruction"].items()):
            raise RuntimeError("Preflight limiter configuration mismatch")
        record = dict(problem=problem, method=method, returncode=process.returncode, physical_time=t,
                      iteration=iteration, seconds=time.monotonic() - started,
                      checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint),
                      rho_min=float(fields["rho"].min()), p_min=float(fields["p"].min()),
                      rho_z_span_max=float(np.ptp(fields["rho"], axis=0).max()),
                      p_z_span_max=float(np.ptp(fields["p"], axis=0).max()),
                      max_abs_w=float(np.abs(fields["w"]).max()),
                      rejected_steps=(case / "solver.log").read_text().count("NCFV rejected iteration="))
        result["runs"].append(record)
        save(directory / "results.json", result)
        print(now(), "preflight passed", problem, method, record, flush=True)
    result.update(status="completed", finished_at=now())
    save(directory / "results.json", result)


def main():
    output = OUTPUT
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / "suite.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (output / "suite_status.json").exists():
        raise FileExistsError("Refusing duplicate suite launch")
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    environment = os.environ.copy()
    for key in list(environment):
        if key.startswith(("BJ_", "NCFV_")) or key in ("DISPLAY", "WAYLAND_DISPLAY"):
            environment.pop(key, None)
    environment.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       DNDS_DIST_OMP_NUM_THREADS="1", OMP_DYNAMIC="FALSE", HWLOC_COMPONENTS="-gl,-cuda,-nvml")
    _, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(65536, hard), hard))
    if shutil.disk_usage(output).free < 70 * 1024**3:
        raise RuntimeError("Need 70 GiB free for checkpoint-based RP3/RP4 output and reserve")
    physical = topology()
    slots = [set(range(a, a + 128)) for a in (0, 128)]
    tree = ET.fromstring(subprocess.check_output(["lstopo-no-graphics", "--of", "xml", "-"], text=True, env=environment))
    pus = [int(o.attrib["os_index"]) for o in tree.iter("object") if o.attrib.get("type") == "PU"]
    mapping = {cpu: pu for pu, cpu in enumerate(pus)}
    for cpus in slots:
        if not cpus.issubset(os.sched_getaffinity(0)) or len({physical[c] for c in cpus}) != 128:
            raise RuntimeError("Invalid physical-core slot")
    if len({physical[c] for cpus in slots for c in cpus}) != 256:
        raise RuntimeError("The two slots share physical cores")
    previous = json.loads((PREVIOUS / "suite_status.json").read_text())
    if previous["status"] != "completed":
        raise RuntimeError("Previous Configuration 3 suite must be complete")
    source_executable = Path(previous["executable"])
    if digest(source_executable) != previous["executable_sha256"]:
        raise RuntimeError("Frozen Config3 binary hash mismatch")
    inputs = output / "inputs"
    inputs.mkdir()
    executable = inputs / source_executable.name
    shutil.copy2(source_executable, executable)
    shutil.copytree(source_executable.parent / "solver_sources", inputs / "solver_sources")
    shutil.copy2(source_executable.parent / "CMakeCache.txt", inputs / "CMakeCache.txt")
    for source in HERE.glob("*.py"):
        shutil.copy2(source, inputs / source.name)
    state = dict(status="preparing", created_at=now(), controller_pid=os.getpid(), targets=TARGETS,
                 mesh_cells=[800, 800, 5], spacing=[.00125, .00125, .01], cfl=.475, ranks_per_case=128,
                 concurrent_cases=2, use_smt_siblings=False, total_logical_cpus=256, total_physical_cores=256,
                 executable=str(executable), executable_sha256=digest(executable), previous_suite=str(PREVIOUS),
                 stencil_selection="Original", output_interval=1000000, restart_interval=200,
                 saved_states="Full HDF5 every 200 steps; initial/final VTK", cases={})
    active = {}
    pending = list(JOBS)
    pattern = re.compile(r"NCFV iteration=(\d+).*?dt\[min,max\]=\[([0-9.eE+-]+),[0-9.eE+-]+\], time=([0-9.eE+-]+), target=([0-9.eE+-]+), step wall\[s\]=([0-9.eE+-]+)")
    try:
        for problem, method in JOBS:
            case = output / problem / method
            case.mkdir(parents=True)
            config = inputs / problem / (method + ".json")
            config.parent.mkdir(exist_ok=True)
            shutil.copy2(HERE / problem / config.name, config)
            current = json.loads(config.read_text())
            baseline = json.loads(Path(previous["cases"][method]["configuration"]).read_text())
            for section in ("solver", "dimension", "mesh", "algorithm", "reconstruction"):
                if current[section] != baseline[section]:
                    raise RuntimeError("Unexpected method change: " + problem + "/" + method + "/" + section)
            check = subprocess.run([str(executable), str(config), "--check-config"], cwd=ROOT / "build", env=environment,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=90)
            (case / "config_check.log").write_text(check.stdout)
            if check.returncode:
                raise RuntimeError("Invalid configuration: " + problem + "/" + method)
            key = problem + "/" + method
            status = dict(status="queued", problem=problem, method=method, target_time=TARGETS[problem],
                          configuration=str(config), configuration_sha256=digest(config),
                          physical_time=0., physical_time_fraction=0., iteration=0, mpi_ranks=128)
            state["cases"][key] = status
            save(case / "run_status.json", status)
        state["input_files"] = {}
        for problem in TARGETS:
            first = json.loads((inputs / problem / "SG_BJ_PRES.json").read_text())
            for path in [(ROOT / "build" / first["mesh"]["meshFile"]).resolve(),
                         (ROOT / "build" / first["initialField"]["nodeFile"]).resolve(),
                         ROOT / f"data/mesh/RP34_preflight40x40x5_initial/{problem}.csv"]:
                state["input_files"][str(path)] = digest(path)
        state["status"] = "preflight"
        save(output / "suite_status.json", state)
        preflight(executable, inputs, environment, mapping, output)
        state.update(status="running", preflight_result_sha256=digest(output / "preflight/results.json"))
        while (active or pending) and not STOP:
            if shutil.disk_usage(output).free < 15 * 1024**3:
                raise RuntimeError("Disk free below 15 GiB")
            for slot in range(2):
                if slot in active or not pending or STOP:
                    continue
                problem, method = pending.pop(0)
                key = problem + "/" + method
                status = state["cases"][key]
                if affinity(executable, slots[slot], physical)["rank_count"]:
                    raise RuntimeError("CPU slot already contains solver ranks")
                args = command(executable, status["configuration"], slots[slot], mapping)
                log = (output / problem / method / "solver.log").open("x")
                child = subprocess.Popen(args, cwd=ROOT / "build", env=environment, stdin=subprocess.DEVNULL,
                                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                active[slot] = dict(key=key, process=child, log=log, started=time.monotonic())
                status.update(status="running", mpi_pid=child.pid, started_at=now(), command=args,
                              slot=slot, linux_cpu_list=sorted(slots[slot]))
                print(now(), "started", key, child.pid, flush=True)
            for slot, job in list(active.items()):
                key, child = job["key"], job["process"]
                case, status = output / key, state["cases"][key]
                text = (case / "solver.log").read_text(errors="replace")
                rows = list(pattern.finditer(text))
                if rows:
                    row = rows[-1]
                    status.update(iteration=int(row[1]), physical_time=float(row[3]),
                                  physical_time_fraction=float(row[3]) / status["target_time"],
                                  last_reported_dt=float(row[2]), last_reported_step_seconds=float(row[5]))
                    recent = [float(row[5]) for row in rows[-30:] if int(row[1]) >= 100]
                    if recent:
                        status["recent_step_median_seconds"] = statistics.median(recent)
                status.update(updated_at=now(), elapsed_seconds=time.monotonic() - job["started"],
                              rejected_steps=text.count("NCFV rejected iteration="))
                if not status.get("logical_binding_verified"):
                    evidence = affinity(executable, slots[slot], physical)
                    if evidence["verified"]:
                        save(case / "cpu_affinity.json", evidence)
                        status.update(logical_binding_verified=True, physical_binding_verified=True)
                if child.poll() is not None:
                    job["log"].close()
                    status.update(returncode=child.returncode, solver_finished_at=now())
                    try:
                        checkpoints = sorted(case.glob("restart_*.dnds.h5"))
                        metric = post.export(case, checkpoints[-1]) if checkpoints else None
                        complete = child.returncode == 0 and metric is not None and abs(metric["time"] - status["target_time"]) < 1e-12
                        if metric:
                            status.update(physical_time=metric["time"], physical_time_fraction=metric["time"] / status["target_time"],
                                          iteration=metric["iteration"])
                    except Exception:
                        complete = False
                        status["final_validation_error"] = traceback.format_exc()
                    status.update(status="completed" if complete else "failed", finished_at=now())
                    del active[slot]
                save(case / "run_status.json", status)
            if len(active) == 2:
                evidence = {job["key"]: affinity(executable, slots[slot], physical) for slot, job in active.items()}
                if all(record["verified"] for record in evidence.values()):
                    cpus = [rank["logical_cpus"][0] for record in evidence.values() for rank in record["ranks"]]
                    if len(cpus) != 256 or len(set(cpus)) != 256 or len({physical[c] for c in cpus}) != 256:
                        raise RuntimeError("Physical or logical CPU overlap")
                    save(output / "physical_affinity_validation.json", dict(verified_at=now(), active_cases=list(evidence),
                         logical_cpus=256, physical_cores=256, logical_cpu_overlap=False, physical_core_overlap=False))
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
            if job["process"].poll() is None:
                os.killpg(job["process"].pid, signal.SIGTERM)
        for job in active.values():
            try:
                job["process"].wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(job["process"].pid, signal.SIGKILL)
                job["process"].wait(timeout=10)
            job["log"].close()
            status = state["cases"][job["key"]]
            status.update(status="stopped", stopped_at=now())
            save(output / job["key"] / "run_status.json", status)
        state["finished_at"] = now()
        save(output / "suite_status.json", state)
    for problem in TARGETS:
        post.compare(output / problem)


if __name__ == "__main__":
    main()
