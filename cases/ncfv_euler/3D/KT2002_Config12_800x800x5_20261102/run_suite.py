"""Queue five KT Configuration 12 cases after RP3/RP4; 128 physical cores each."""
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
DEPENDENCY = ROOT / "data/out/NCFV/RP34_800x800x5_np128_20261101/suite_status.json"
OUTPUT = ROOT / "data/out/NCFV/KT2002_Config12_800x800x5_np128_20261102"
TARGET = .25
STOP = False


def stop(*unused):
    global STOP
    STOP = True


def command(executable, config, cpus, mapping):
    return ["mpirun", "--mca", "pml", "ob1", "--mca", "btl", "self,vader", "--use-hwthread-cpus",
            "-np", str(len(cpus)), "--cpu-list", ",".join(str(mapping[c]) for c in sorted(cpus)),
            "--bind-to", "cpu-list:ordered", "--report-bindings", str(executable), str(config)]


def owned_solver_ranks():
    ranks = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            if proc.stat().st_uid != os.getuid():
                continue
            name = (proc/'comm').read_text().strip()
            if name.startswith(('solver_', 'ncfv_euler')):
                ranks.append(int(proc.name))
        except OSError:
            continue
    return ranks


def wait_for_predecessor(output, state):
    last_reason = None
    while not STOP:
        predecessor = json.loads(DEPENDENCY.read_text())
        occupied = owned_solver_ranks()
        reason = ('Waiting for RP3/RP4 suite completion' if predecessor['status'] != 'completed'
                  else 'Waiting for existing owned NCFV solver ranks to exit' if occupied else None)
        state.update(status='waiting_resources', updated_at=now(), waiting_reason=reason,
                     dependency_status=predecessor['status'], occupied_solver_ranks=len(occupied))
        save(output/'suite_status.json', state)
        if reason is None:
            return True
        if reason != last_reason:
            print(now(), reason, flush=True)
            last_reason = reason
        time.sleep(30)
    return False


def preflight(executable, inputs, environment, mapping, output):
    directory = output / "preflight"
    directory.mkdir()
    result = dict(status="running", started_at=now(), executable_sha256=digest(executable),
                  mesh_cells=[40, 40, 5], depth=1., spacing_ratio_dz_dx=8., target=TARGET, runs=[])
    save(directory / "results.json", result)
    for name in post.NAMES:
        case = directory / name
        case.mkdir()
        overrides = {
            "/mesh/meshFile": str(ROOT / "data/mesh/KT2002_Config13_preflight40x40x5_Lz1.cgns"),
            "/mesh/periodicLengths": [0, 0, 1.],
            "/initialField/nodeFile": str(ROOT / "data/mesh/KT2002_Config12_preflight40x40x5_initial/KT12.csv"),
            "/time/maximumTimeStep": .002,
            "/time/reportInterval": 50,
            "/io/writeVTK": False, "/io/writeInitial": False, "/io/writeFinal": False,
            "/io/restartInterval": 0, "/io/writeFinalRestart": True,
            "/io/outputPrefix": str(case / "solution"), "/io/vtkSeriesName": str(case / "series"),
            "/io/restartPrefix": str(case / "restart"),
        }
        config = inputs / (name + ".json")
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
                        raise RuntimeError("Preflight interrupted or exceeded 600 seconds: " + name)
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
            raise RuntimeError("Preflight solver failed: " + name)
        checkpoint = sorted(case.glob("restart_*.dnds.h5"))[-1]
        fields, t, iteration, actual = post.read_checkpoint(checkpoint)
        expected = json.loads(config.read_text())
        if abs(t - TARGET) > 1e-12 or fields["rho"].shape != (5, 41, 41):
            raise RuntimeError("Preflight time or independent node shape mismatch: " + name)
        if any(actual["reconstruction"][k] != v for k, v in expected["reconstruction"].items()):
            raise RuntimeError("Preflight limiter configuration mismatch: " + name)
        record = dict(name=name, returncode=process.returncode, physical_time=t, iteration=iteration,
                      seconds=time.monotonic() - started, checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint),
                      rho_min=float(fields["rho"].min()), p_min=float(fields["p"].min()),
                      rho_z_span_max=float(np.ptp(fields["rho"], axis=0).max()),
                      p_z_span_max=float(np.ptp(fields["p"], axis=0).max()), max_abs_w=float(np.abs(fields["w"]).max()),
                      rejected_steps=(case / "solver.log").read_text().count("NCFV rejected iteration="))
        result["runs"].append(record)
        save(directory / "results.json", result)
        print(now(), "preflight passed", name, record, flush=True)
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
    if shutil.disk_usage(output).free < 40 * 1024**3:
        raise RuntimeError("Need 40 GiB free for checkpoint-based Config12 output and reserve")
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
    source_executable = Path(previous['executable'])
    if digest(source_executable) != previous['executable_sha256']:
        raise RuntimeError('Frozen Config3 executable hash mismatch')
    inputs = output / "inputs"
    inputs.mkdir()
    executable = inputs / ("solver_" + digest(source_executable)[:16] + ".exe")
    shutil.copy2(source_executable, executable)
    shutil.copytree(source_executable.parent / "solver_sources", inputs / "solver_sources")
    shutil.copy2(source_executable.parent / "CMakeCache.txt", inputs / "CMakeCache.txt")
    for source in HERE.glob("*.py"):
        shutil.copy2(source, inputs / source.name)
    state = dict(status="preparing", created_at=now(), controller_pid=os.getpid(), target_time=TARGET,
                 problem="Kurganov--Tadmor 2002 Configuration 12", reference_figure="3.12", reference_pages=[600, 602],
                 mesh_cells=[800, 800, 5], spacing=[.00125, .00125, .01], cfl=.475, ranks_per_case=128,
                 concurrent_cases=2, use_smt_siblings=False, total_logical_cpus=256, total_physical_cores=256,
                 executable=str(executable), executable_sha256=digest(executable), previous_suite=str(PREVIOUS),
                 stencil_selection="Original", output_interval=1000000, restart_interval=200,
                 saved_states="Full HDF5 every 200 steps; initial/final VTK", dependency=str(DEPENDENCY), cases={})
    state["frozen_source_hashes"] = {str(p.relative_to(inputs)): digest(p)
                                     for p in sorted((inputs / "solver_sources").rglob("*")) if p.is_file()}
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
            baseline_path = Path(previous["cases"][name]["configuration"])
            baseline = json.loads(baseline_path.read_text())
            # The new executable exposes the prior strategy as an explicit
            # option; older frozen configurations omit this default key.
            baseline["reconstruction"].setdefault("stencilSelectionStrategy", "Original")
            for section in ("solver", "dimension", "mesh", "algorithm", "reconstruction"):
                if current[section] != baseline[section]:
                    raise RuntimeError("Unexpected method change: " + name + "/" + section)
            check = subprocess.run([str(executable), str(config), "--check-config"], cwd=ROOT / "build", env=environment,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=90)
            (case / "config_check.log").write_text(check.stdout)
            if check.returncode:
                raise RuntimeError("Invalid configuration: " + name)
            status = dict(status="queued", configuration=str(config), configuration_sha256=digest(config),
                          physical_time=0., physical_time_fraction=0., iteration=0, mpi_ranks=128)
            state["cases"][name] = status
            save(case / "run_status.json", status)
        state["input_files"] = {}
        first = json.loads((inputs / "SG_BJ_PRES.json").read_text())
        for path in [(ROOT / "build" / first["mesh"]["meshFile"]).resolve(), (ROOT / "build" / first["initialField"]["nodeFile"]).resolve()]:
            state["input_files"][str(path)] = digest(path)
        small_initial = ROOT / 'data/mesh/KT2002_Config12_preflight40x40x5_initial/KT12.csv'
        state['input_files'][str(small_initial)] = digest(small_initial)
        state['status'] = 'waiting_resources'
        save(output / "suite_status.json", state)
        if not wait_for_predecessor(output, state):
            state.update(status='stopped', stopped_at=now())
            return
        if shutil.disk_usage(output).free < 40 * 1024**3:
            raise RuntimeError('Need 40 GiB free before launching Config12 preflight')
        state.update(status='preflight', updated_at=now(), resources_acquired_at=now())
        save(output / 'suite_status.json', state)
        preflight(executable, inputs, environment, mapping, output)
        state.update(status="running", preflight_result_sha256=digest(output / "preflight/results.json"))
        while (active or pending) and not STOP:
            if shutil.disk_usage(output).free < 10 * 1024**3:
                raise RuntimeError("Disk free below 10 GiB")
            for slot in range(2):
                if slot in active or not pending or STOP:
                    continue
                name = pending.pop(0)
                status = state["cases"][name]
                if affinity(executable, slots[slot], physical)["rank_count"]:
                    raise RuntimeError("CPU slot already contains solver ranks")
                args = command(executable, status["configuration"], slots[slot], mapping)
                log = (output / name / "solver.log").open("x")
                child = subprocess.Popen(args, cwd=ROOT / "build", env=environment, stdin=subprocess.DEVNULL,
                                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                active[slot] = dict(name=name, process=child, log=log, started=time.monotonic())
                status.update(status="running", mpi_pid=child.pid, started_at=now(), command=args,
                              slot=slot, linux_cpu_list=sorted(slots[slot]))
                print(now(), "started", name, child.pid, flush=True)
            for slot, job in list(active.items()):
                name, child = job["name"], job["process"]
                case, status = output / name, state["cases"][name]
                text = (case / "solver.log").read_text(errors="replace")
                rows = list(pattern.finditer(text))
                if rows:
                    r = rows[-1]
                    status.update(iteration=int(r[1]), physical_time=float(r[3]), physical_time_fraction=float(r[3]) / TARGET,
                                  last_reported_dt=float(r[2]), last_reported_step_seconds=float(r[5]))
                    recent = [float(r[5]) for r in rows[-30:] if int(r[1]) >= 100]
                    if recent:
                        status["recent_step_median_seconds"] = statistics.median(recent)
                status.update(updated_at=now(), elapsed_seconds=time.monotonic() - job["started"], rejected_steps=text.count("NCFV rejected iteration="))
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
                        complete = child.returncode == 0 and metric is not None and abs(metric["time"] - TARGET) < 1e-12
                        if metric:
                            status.update(physical_time=metric["time"], physical_time_fraction=metric["time"] / TARGET, iteration=metric["iteration"])
                    except Exception:
                        complete = False
                        status["final_validation_error"] = traceback.format_exc()
                    status.update(status="completed" if complete else "failed", finished_at=now())
                    del active[slot]
                save(case / "run_status.json", status)
            if len(active) == 2 and not (output / "physical_affinity_validation.json").exists():
                evidence = {j["name"]: affinity(executable, slots[slot], physical) for slot, j in active.items()}
                if all(e["verified"] for e in evidence.values()):
                    cpus = [r["logical_cpus"][0] for e in evidence.values() for r in e["ranks"]]
                    if len(cpus) != 256 or len(set(cpus)) != 256 or len({physical[c] for c in cpus}) != 256:
                        raise RuntimeError("Physical or logical CPU overlap")
                    save(output / "physical_affinity_validation.json", dict(verified_at=now(), logical_cpus=256,
                         physical_cores=256, logical_cpu_overlap=False, physical_core_overlap=False))
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
            status = state["cases"][job["name"]]
            status.update(status="stopped", stopped_at=now())
            save(output / job["name"] / "run_status.json", status)
        state["finished_at"] = now()
        save(output / "suite_status.json", state)
    post.compare(output)


if __name__ == "__main__":
    main()
