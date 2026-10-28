"""Controlled z-periodic RP1 limiter and flux comparisons on 40x40x5.

Only private diagnostic C++ copies are built. Production sources and existing
results are preserved. Scalar maxima are descriptive, not oscillation tests.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import h5py
import numpy as np

from test_barth_forms import ROOT, CASE, build


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    temporary = path.with_suffix(".partial.json")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def metrics(path):
    with h5py.File(path) as data:
        states = data["state/father/array/data"][:].reshape(-1, 5)
        ids = data["state/origIndex"][:].astype(np.int64)
        result = {"time": float(data.attrs["simulationTime"]),
                  "iteration": int(data.attrs["iteration"])}
    count = 5 * 41 * 41
    if len(states) != count or not np.array_equal(np.sort(ids), np.arange(count)):
        raise ValueError("Unexpected periodic quotient node identities")
    ordered = np.empty_like(states)
    ordered[ids] = states
    cube = ordered.reshape(5, 41, 41, 5)
    rho = cube[..., 0]
    pressure = .4 * (cube[..., 4] - .5 * np.sum(cube[..., 1:4] ** 2, axis=-1) / rho)
    if not np.isfinite(cube).all() or np.any(rho <= 0) or np.any(pressure <= 0):
        raise ValueError("Nonfinite or nonphysical final states")
    result.update(nodes=count, restart=str(path), restart_sha256=digest(path),
                  w_abs_max=float(np.abs(cube[..., 3] / rho).max()))
    for name, values in (("rho", rho), ("p", pressure)):
        span = np.ptp(values, axis=0)
        chain_tv = np.abs(np.diff(values, axis=0)).sum(axis=0)
        cyclic_tv = chain_tv + np.abs(values[0] - values[-1])
        result.update({name + "_min": float(values.min()),
                       name + "_max": float(values.max()),
                       name + "_z_span_max": float(span.max()),
                       name + "_z_span_rms": float(np.sqrt(np.mean(span ** 2))),
                       name + "_z_TV_excess_max": float(np.maximum(0, chain_tv - span).max()),
                       name + "_z_cyclic_TV_excess_max": float(np.maximum(0, cyclic_tv - 2 * span).max()),
                       name + "_xy_symmetry_abs_max": float(np.abs(values - values.transpose(0, 2, 1)).max())})
        for label, cut in (("midplane", values[2, 38]),
                           ("zmean", values[:, 38].mean(axis=0))):
            result[name + "_cut_y095_" + label] = {
                "TV": float(np.abs(np.diff(cut)).sum()),
                "range": float(np.ptp(cut)),
                "TV_minus_range": float(max(0, np.abs(np.diff(cut)).sum() - np.ptp(cut))),
                "values": cut.tolist()}
    result["momentum_xy_swap_abs_max"] = float(np.abs(cube[..., 1] - cube[..., 2].transpose(0, 2, 1)).max())
    result["cut_x"] = (np.arange(41) / 40).tolist()
    return result


def compare_saved(out):
    """Quantify whether stricter limiting mostly reproduces first order."""
    def load(name):
        path = sorted((out / name).glob("restart_*.dnds.h5"))[-1]
        with h5py.File(path) as data:
            values = data["state/father/array/data"][:].reshape(-1, 5)
            ids = data["state/origIndex"][:].astype(np.int64)
        ordered = np.empty_like(values)
        ordered[ids] = values
        return ordered

    reference = load("first_order")
    reference_p = .4 * (reference[:, 4] - .5 * np.sum(reference[:, 1:4] ** 2, axis=1) / reference[:, 0])
    result = {}
    variants = ["baseline", "pair_bounds", "LLF_only", "face_Roe", "first_order_local"]
    if (out / "cfl_half/metrics.json").exists():
        variants.append("cfl_half")
    for name in variants:
        candidate = load(name)
        pressure = .4 * (candidate[:, 4] - .5 * np.sum(candidate[:, 1:4] ** 2, axis=1) / candidate[:, 0])
        result[name + "_vs_first_order"] = {
            "conservative_abs_max": np.abs(candidate - reference).max(axis=0).tolist(),
            "conservative_abs_mean": np.abs(candidate - reference).mean(axis=0).tolist(),
            "pressure_abs_max": float(np.abs(pressure - reference_p).max()),
            "pressure_abs_mean": float(np.abs(pressure - reference_p).mean())}
    save(out / "first_order_comparison.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--ranks", type=int, default=4)
    args = parser.parse_args()
    out = (args.output or ROOT / "data/out/NCFV/periodic_limiter_analysis" /
           datetime.now().strftime("%Y%m%d_%H%M%S")).resolve()
    out.mkdir(parents=True, exist_ok=False)
    print("Output:", out, flush=True)
    config = CASE / "RP1_zperiodic_tol1e10.json"
    audit = ROOT / "data/out/NCFV/riemann_barth_400x400x5/limiter_audit"
    result = {"diagnostic_only": True, "threshold": 1e-10,
              "started_utc": datetime.now(timezone.utc).isoformat(),
              "grid": [40, 40, 5], "periodic_lengths": [0, 0, .125],
              "mpi_ranks": args.ranks, "base_config": str(config),
              "config_sha256": digest(config), "mesh_sha256": digest(audit / "mesh40.cgns"),
              "initial_sha256": digest(audit / "initial40/RP1.csv"),
              "production_source_sha256": {name: digest(ROOT / "src/NCFV" / name)
                  for name in ("NCFVSpatial.cpp", "NCFVReconstruction.cpp", "NCFVSolver.cpp", "NCFVConfig.cpp")},
              "metric_notes": "All scalars are conservative means; positive maxima above initial values alone are not an oscillation criterion. z TV uses the five quotient planes. Cut y=.95 uses z-plane index 2 and z average.",
              "runs": []}
    save(out / "results.json", result)
    started = time.monotonic()
    executable = build(out)
    result.update(binary_sha256=digest(executable), build_elapsed_seconds=time.monotonic() - started)
    variants = {
        "baseline": {},
        "pair_bounds": {"BJ_PAIR_BOUNDS": "1"},
        "LLF_only": {"BJ_LLF": "1"},
        "face_Roe": {"BJ_FACE_ONLY": "1"},
        "first_order": {"BJ_FIRST_ORDER": "1"},
        "first_order_local": {"BJ_FIRST_ORDER": "1", "BJ_LOCAL_BOUNDARY": "1"},
    }
    environment = dict(os.environ, OMP_NUM_THREADS="1", DNDS_DIST_OMP_NUM_THREADS="1",
                       OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       HWLOC_COMPONENTS="-gl,-cuda,-nvml", OMPI_MCA_mpi_yield_when_idle="1")
    for key in list(environment):
        if key.startswith(("BJ_", "NCFV_")):
            del environment[key]
    for name, flags in variants.items():
        directory = out / name
        directory.mkdir()
        command = ["mpirun", "--mca", "pml", "ob1", "--mca", "btl", "self,vader",
                   "--bind-to", "none", "-np", str(args.ranks), str(executable), str(config)]
        overrides = {"/mesh/meshFile": str(audit / "mesh40.cgns"),
                     "/mesh/periodicLengths": [0, 0, .125],
                     "/initialField/nodeFile": str(audit / "initial40/RP1.csv"),
                     "/time/endTime": .25, "/time/cfl": .5,
                     "/time/maximumTimeStep": .001, "/io/writeVTK": False,
                     "/io/restartInterval": 0, "/io/writeFinalRestart": True,
                     "/io/outputPrefix": str(directory / "solution"),
                     "/io/vtkSeriesName": str(directory / "series"),
                     "/io/restartPrefix": str(directory / "restart")}
        for key, value in overrides.items():
            command += ["-k", key, "-v", json.dumps(value)]
        save(directory / "command.json", {"command": command, "flags": flags,
                                          "binary_sha256": result["binary_sha256"],
                                          "working_directory": str(ROOT / "build")})
        started = time.monotonic()
        with (directory / "solver.log").open("x") as log:
            process = subprocess.run(command, cwd=ROOT / "build", env=dict(environment, **flags),
                                     stdout=log, stderr=subprocess.STDOUT)
        run = {"variant": name, "flags": flags, "returncode": process.returncode,
               "elapsed_seconds": time.monotonic() - started}
        if process.returncode == 0:
            run.update(metrics(sorted(directory.glob("restart_*.dnds.h5"))[-1]))
            run["rejected_steps"] = (directory / "solver.log").read_text().count("NCFV rejected iteration=")
        result["runs"].append(run)
        save(out / "results.json", result)
        save(directory / "metrics.json", run)
        print(json.dumps({key: value for key, value in run.items()
                          if "cut" not in key}), flush=True)
    result["finished_utc"] = datetime.now(timezone.utc).isoformat()
    compare_saved(out)
    save(out / "results.json", result)


if __name__ == "__main__":
    main()
