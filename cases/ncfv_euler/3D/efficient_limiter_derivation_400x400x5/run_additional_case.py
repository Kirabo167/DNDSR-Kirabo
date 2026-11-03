"""Queue an additional 128-physical-core case behind the existing four cases.

The original queue keeps priority. The job starts only once no original case
is queued and an entire physical-core slot is free. Existing jobs are untouched.
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

import h5py
import numpy as np
import compare_results as post
from run_suite import ROOT, DEFAULT, HERE, affinity, digest, now, save, topology

STOP = False


def stop(*unused):
    global STOP
    STOP = True


def preflight(executable, configuration, root, environment, name):
    out = root / "diagnostics" / ("additional_" + name)
    out.mkdir(exist_ok=False)
    overrides = {
        "/mesh/meshFile": str(ROOT / "data/out/NCFV/riemann_barth_400x400x5/limiter_audit/mesh40.cgns"),
        "/mesh/periodicLengths": [0, 0, .125],
        "/initialField/nodeFile": str(ROOT / "data/out/NCFV/riemann_barth_400x400x5/limiter_audit/initial40/RP1.csv"),
        "/time/maximumTimeStep": .001,
        "/io/writeVTK": False, "/io/restartInterval": 0,
        "/io/outputPrefix": str(out / "solution"), "/io/vtkSeriesName": str(out / "series"),
        "/io/restartPrefix": str(out / "restart"),
    }
    command = ["mpirun", "--mca", "pml", "ob1", "--mca", "btl", "self,vader", "--bind-to", "none",
               "-np", "4", str(executable), str(configuration)]
    for key, value in overrides.items(): command += ["-k", key, "-v", json.dumps(value)]
    save(out / "command.json", {"command": command, "executable_sha256": digest(executable)})
    with (out / "solver.log").open("x") as log:
        process = subprocess.Popen(command, cwd=ROOT / "build", env=environment, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        while process.poll() is None:
            if STOP:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=30)
                raise InterruptedError("Stopped additional case preflight")
            time.sleep(1)
    if process.returncode: raise RuntimeError("Additional case preflight failed")
    checkpoint = sorted(out.glob("restart_*.dnds.h5"))[-1]
    with h5py.File(checkpoint) as f:
        u = f["state/father/array/data"][:].reshape(-1, 5)
        ids = f["state/origIndex"][:].astype(int)
        t = float(f.attrs["simulationTime"])
        c = json.loads(f.attrs["configuration"])
    if len(u) != 8405 or not np.array_equal(np.sort(ids), np.arange(8405)):
        raise RuntimeError("Unexpected small-grid independent nodes")
    ordered = np.empty_like(u)
    ordered[ids] = u
    u = ordered.reshape(5, 41, 41, 5)
    rho = u[..., 0]
    p = .4 * (u[..., 4] - .5 * np.sum(u[..., 1:4] ** 2, axis=-1) / rho)
    if not np.isfinite(u).all() or np.any(rho <= 0) or np.any(p <= 0) or abs(t - .25) > 1e-12:
        raise RuntimeError("Additional case preflight state failed validation")
    expected = json.loads(configuration.read_text())["reconstruction"]
    if any(c["reconstruction"][key] != value for key, value in expected.items()):
        raise RuntimeError("Expected reconstruction switches were not active")
    result = {"status": "passed", "physical_time": t, "rho_min": float(rho.min()), "p_min": float(p.min()),
              "rho_z_span_max": float(np.ptp(rho, axis=0).max()), "p_z_span_max": float(np.ptp(p, axis=0).max()),
              "max_abs_w": float(np.abs(u[..., 3] / rho).max()), "checkpoint": str(checkpoint),
              "executable_sha256": digest(executable), "definition": "40x40x5, Lz=.125, four ranks, t=.25"}
    save(out / "metrics.json", result)
    return result


def settings_match(actual, expected):
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(k in actual and settings_match(actual[k], v) for k, v in expected.items())
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(settings_match(a, e) for a, e in zip(actual, expected))
    return actual == expected


def reuse_preflight(path, executable, configuration):
    """Reuse verified CFD evidence when only the output name changes."""
    result = json.loads(path.read_text())
    command = json.loads((path.parent / "command.json").read_text())
    if result.get("returncode") != 0 or command["binary_sha256"] != digest(executable):
        raise RuntimeError("Preflight return code or executable hash does not match")
    checkpoint = Path(result["restart"])
    if digest(checkpoint) != result["restart_sha256"]:
        raise RuntimeError("Preflight checkpoint hash changed")
    with h5py.File(checkpoint) as f:
        actual = json.loads(f.attrs["configuration"])
        if abs(float(f.attrs["simulationTime"]) - .25) > 1e-12:
            raise RuntimeError("Preflight did not reach t=0.25")
    expected = json.loads(configuration.read_text())
    for section in ("reconstruction", "algorithm", "physics"):
        if not settings_match(actual[section], expected[section]):
            raise RuntimeError("Preflight settings differ: " + section)
    return {"status": "passed", "reused_from": str(path), "checkpoint": str(checkpoint),
            "executable_sha256": digest(executable), "physical_time": .25,
            "rho_min": result["rho_min"], "p_min": result["p_min"],
            "rho_z_span_max": result["rho_z_span_max"], "p_z_span_max": result["p_z_span_max"],
            "reuse_scope": "Identical numerical switches and executable; new independent output directory"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT)
    parser.add_argument("--name", default="LLF_FIRST_ORDER")
    parser.add_argument("--wait-for", nargs="*", default=[])
    parser.add_argument("--wait-for-started", nargs="*", default=[])
    parser.add_argument("--preflight-result", type=Path)
    args = parser.parse_args()
    root = args.output.resolve()
    name = args.name
    if not re.fullmatch(r"[A-Za-z0-9_]+", name): raise ValueError("Invalid case name")
    case = root / name
    case.mkdir(exist_ok=True)
    lock = (case / "queue.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (case / "run_status.json").exists(): raise FileExistsError("Additional case already has a status")
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    suite = json.loads((root / "suite_status.json").read_text())
    executable = Path(suite["executable"])
    if digest(executable) != suite["executable_sha256"]: raise RuntimeError("Frozen binary hash mismatch")
    inputs = case / "inputs"
    inputs.mkdir()
    configuration = inputs / (name + ".json")
    shutil.copy2(HERE / (name + ".json"), configuration)
    for path in (Path(__file__), HERE / "compare_results.py", HERE / "run_suite.py"):
        shutil.copy2(path, inputs / path.name)
    state = {"name": name, "status": "preflight", "created_at": now(), "controller_pid": os.getpid(),
             "target_time": .25, "physical_time": 0., "iteration": 0,
             "mpi_ranks": 128, "physical_cores": 128, "threads_per_rank": 1,
             "configuration": str(configuration), "configuration_sha256": digest(configuration),
             "executable": str(executable), "executable_sha256": digest(executable),
             "queue_policy": "Existing four cases and named dependencies have priority; wait for a free 128-core slot",
             "wait_for": args.wait_for,
             "wait_for_started": args.wait_for_started,
             "note": "Independent named run using the original frozen executable and unchanged solver settings."}
    save(case / "run_status.json", state)
    environment = os.environ.copy()
    for key in list(environment):
        if key.startswith(("BJ_", "NCFV_")) or key in ("DISPLAY", "WAYLAND_DISPLAY"): environment.pop(key, None)
    environment.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       DNDS_DIST_OMP_NUM_THREADS="1", OMP_DYNAMIC="FALSE", HWLOC_COMPONENTS="-gl,-cuda,-nvml")
    _, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(65536, hard), hard))
    process = None
    try:
        state["preflight"] = (reuse_preflight(args.preflight_result.resolve(), executable, configuration)
                              if args.preflight_result else preflight(executable, configuration, root, environment, name))
        state["status"] = "queued"
        save(case / "run_status.json", state)
        physical = topology()
        chosen = None
        while not STOP:
            suite = json.loads((root / "suite_status.json").read_text())
            cases = list(suite["cases"].values())
            dependencies = []
            started_dependencies = []
            for dependency in args.wait_for + args.wait_for_started:
                path = root / dependency / "run_status.json"
                if not path.exists(): raise RuntimeError("Missing queue dependency: " + dependency)
                target = dependencies if dependency in args.wait_for else started_dependencies
                target.append(json.loads(path.read_text()))
            if any(v["status"] == "stopped" for v in dependencies + started_dependencies):
                raise InterruptedError("Earlier additional case stopped; keep dependent computation stopped")
            dependency_ready = all(v["status"] in ("completed", "failed") for v in dependencies)
            dependency_ready = dependency_ready and all(v["status"] in ("completed", "failed") or
                                                        (v["status"] == "running" and v.get("physical_binding_verified"))
                                                        for v in started_dependencies)
            if suite["status"] in ("stopped", "preflight_failed"):
                raise InterruptedError("Original suite stopped; do not restart computation implicitly")
            if dependency_ready and not any(v["status"] in ("queued", "starting") for v in cases):
                busy = {c for v in cases if v["status"] == "running" for c in v["linux_cpu_list"]}
                for path in root.glob("*/run_status.json"):
                    other = json.loads(path.read_text())
                    if other.get("status") == "running": busy.update(other.get("linux_cpu_list", []))
                for slot in [set(range(128)), set(range(128, 256))]:
                    if not slot & busy and affinity(executable, slot, physical)["rank_count"] == 0:
                        chosen = slot
                        break
            if chosen is not None: break
            state["updated_at"] = now()
            save(case / "run_status.json", state)
            time.sleep(10)
        if STOP: raise InterruptedError("Additional case stopped while queued")
        if len({physical[c] for c in chosen}) != 128: raise RuntimeError("Physical core collision")
        hardware = ET.fromstring(subprocess.check_output(["lstopo-no-graphics", "--of", "xml", "-"], text=True, env=environment))
        pus = [int(o.attrib["os_index"]) for o in hardware.iter("object") if o.attrib.get("type") == "PU"]
        mapping = {cpu: i for i, cpu in enumerate(pus)}
        cpu_list = ",".join(str(mapping[c]) for c in sorted(chosen))
        command = ["mpirun", "--mca", "pml", "ob1", "--mca", "btl", "self,vader", "--use-hwthread-cpus",
                   "-np", "128", "--cpu-list", cpu_list, "--bind-to", "cpu-list:ordered", "--report-bindings",
                   str(executable), str(configuration)]
        post.NAMES = list(dict.fromkeys(post.NAMES + [p.parent.name for p in root.glob("*/run_status.json")]))
        with (case / "solver.log").open("x") as log:
            process = subprocess.Popen(command, cwd=ROOT / "build", env=environment, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            state.update(status="running", mpi_pid=process.pid, started_at=now(), linux_cpu_list=sorted(chosen), command=command)
            save(case / "run_status.json", state)
            pattern = re.compile(r"NCFV iteration=(\d+).*?, time=([0-9.eE+-]+), target=([0-9.eE+-]+)")
            milestone = 0
            while not STOP:
                text = (case / "solver.log").read_text(errors="replace")
                matches = list(pattern.finditer(text))
                if matches:
                    m = matches[-1]
                    state.update(iteration=int(m[1]), physical_time=float(m[2]), physical_time_fraction=float(m[2]) / .25)
                if not state.get("physical_binding_verified"):
                    evidence = affinity(executable, chosen, physical)
                    if evidence["verified"]:
                        save(case / "cpu_affinity.json", evidence)
                        state["physical_binding_verified"] = True
                finished = process.poll() is not None
                checkpoints = sorted(p for p in case.glob("restart_*.dnds.h5") if finished or time.time() - p.stat().st_mtime > 5)
                if checkpoints and (state["physical_time"] >= milestone or finished):
                    metric = post.export(case, checkpoints[-1])
                    state["last_exported_time"] = metric["time"]
                    milestone = .05 * (int(metric["time"] / .05) + 1)
                    post.compare(root)
                state.update(updated_at=now(), rejected_steps=text.count("NCFV rejected iteration="))
                save(case / "run_status.json", state)
                if finished:
                    complete = process.returncode == 0 and abs(state.get("last_exported_time", 0) - .25) < 1e-12
                    state.update(status="completed" if complete else "failed", returncode=process.returncode, finished_at=now())
                    if complete: state.update(physical_time=.25, physical_time_fraction=1.)
                    save(case / "run_status.json", state)
                    return
                time.sleep(10)
    except InterruptedError as error:
        state.update(status="stopped", reason=str(error), stopped_at=now())
    except Exception as error:
        state.update(status="failed", reason=repr(error), finished_at=now())
        raise
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try: process.wait(timeout=30)
            except subprocess.TimeoutExpired: os.killpg(process.pid, signal.SIGKILL)
        save(case / "run_status.json", state)


if __name__ == "__main__": main()
