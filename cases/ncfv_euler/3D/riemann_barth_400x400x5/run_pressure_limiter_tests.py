"""Compare pressure limiting off/on using one frozen production solver.

Runs periodic RP1 and its planar shock on 40x40x5 to t=.25. The only
numerical-setting difference within each pair is reconstruction.limitPressure.
No diagnostic BJ flags or production-source mutations are used.
"""

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/ncfv-pressure-limiter-matplotlib")

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np

from test_periodic_limiter import ROOT, CASE, digest, metrics, save
from test_planar_shock import exact_shock, conservative


def read_states(path):
    with h5py.File(path) as data:
        values = data["state/father/array/data"][:].reshape(-1, 5)
        ids = data["state/origIndex"][:].astype(np.int64)
    ordered = np.empty_like(values)
    ordered[ids] = values
    cube = ordered.reshape(5, 41, 41, 5)
    rho = cube[..., 0]
    pressure = .4 * (cube[..., 4] - .5 * np.sum(cube[..., 1:4] ** 2, axis=-1) / rho)
    return cube, rho, pressure


def shock_reference(time):
    x = np.linspace(0, 1, 41)
    dx = .025
    lower, upper = np.maximum(0, x - dx / 2), np.minimum(1, x + dx / 2)
    boundaries, states = exact_shock(time)
    reference = np.zeros((41, 5))
    edges = [0, *boundaries, 1]
    for region, state in enumerate(states):
        fraction = np.maximum(0, np.minimum(upper, edges[region + 1]) -
                              np.maximum(lower, edges[region])) / (upper - lower)
        reference += fraction[:, None] * conservative(state)
    pressure = .4 * (reference[:, 4] - .5 * reference[:, 1] ** 2 / reference[:, 0])
    # Exclude three cell spacings around both acoustic shock fronts.
    away = (np.abs(x - boundaries[0]) > 3 * dx) & (np.abs(x - boundaries[2]) > 3 * dx)
    return {"x": x, "widths": upper - lower, "boundaries": boundaries, "states": states,
            "reference": reference, "rho": reference[:, 0], "pressure": pressure,
            "away": away, "dx": dx}


def shock_metrics(path):
    cube, rho, pressure = read_states(path)
    exact = shock_reference(.25)
    states = exact["states"]
    rho_min, rho_max = min(s[0] for s in states), max(s[0] for s in states)
    p_min, p_max = min(s[2] for s in states), max(s[2] for s in states)
    error = np.abs(rho - exact["rho"][None, None, :])
    p_error = np.abs(pressure - exact["pressure"][None, None, :])
    weights = exact["widths"][None, :, None] * exact["widths"][None, None, :] / 5
    return {
        "exact_wave_locations": exact["boundaries"], "exact_states_rho_u_p": states,
        "exact_density_range": [rho_min, rho_max], "exact_pressure_range": [p_min, p_max],
        "rho_overshoot_exact_states": max(0, float(rho.max() - rho_max)),
        "rho_undershoot_exact_states": max(0, float(rho_min - rho.min())),
        "p_overshoot_exact_states": max(0, float(pressure.max() - p_max)),
        "p_undershoot_exact_states": max(0, float(p_min - pressure.min())),
        "p_overshoot_exact_dualmeans": max(0, float(pressure.max() - exact["pressure"].max())),
        "p_undershoot_exact_dualmeans": max(0, float(exact["pressure"].min() - pressure.min())),
        "p_overshoot_away_shocks_3dx": max(0, float(pressure[..., exact["away"]].max() - p_max)),
        "p_undershoot_away_shocks_3dx": max(0, float(p_min - pressure[..., exact["away"]].min())),
        "rho_L1_exact_dualmean": float(error.mean()),
        "rho_L1_volume_weighted_exact_dualmean": float(np.sum(weights * error)),
        "p_L1_exact_dualmean": float(p_error.mean()),
        "p_L1_volume_weighted_exact_dualmean": float(np.sum(weights * p_error)),
        "rho_midline_TV": float(np.abs(np.diff(rho[2, 20])).sum()),
        "p_midline_TV": float(np.abs(np.diff(pressure[2, 20])).sum()),
        "shock_front_exclusion_distance": 3 * exact["dx"],
    }


def plot_results(out, records):
    lookup = {(r["problem"], r["limitPressure"]): r for r in records}
    x = np.linspace(0, 1, 41)
    states = {(problem, flag): read_states(lookup[problem, flag]["restart"])
              for problem in ("RP1", "planar_shock") for flag in (False, True)}
    figure, axes = plt.subplots(2, 2, figsize=(11, 10), layout="constrained", facecolor="white")
    for column, field in enumerate(("Density", "Pressure")):
        field_index = column + 1
        minimum = min(states["RP1", flag][field_index].min() for flag in (False, True))
        maximum = max(states["RP1", flag][field_index].max() for flag in (False, True))
        normalizer = Normalize(minimum, maximum)
        levels = np.linspace(minimum, maximum, 27)[1:-1]
        for row, flag in enumerate((False, True)):
            axis = axes[row, column]
            values = states["RP1", flag][field_index][2]
            record = lookup["RP1", flag]
            image = axis.pcolormesh(x, x, values, shading="gouraud", cmap="coolwarm", norm=normalizer)
            axis.contour(x, x, values, levels=levels, colors="black", linewidths=.45, alpha=.65)
            span = record["rho_z_span_max" if column == 0 else "p_z_span_max"]
            axis.set_title(f"Pressure limiter {'on' if flag else 'off'}: {field}\n"
                           f"max z span = {span:.3g}; max |w| = {record['w_abs_max']:.3g}", fontsize=11)
            axis.set(xlabel="x", ylabel="y", xlim=(0, 1), ylim=(0, 1),
                     xticks=[0, .5, 1], yticks=[0, .5, 1], aspect="equal")
        figure.colorbar(image, ax=axes[:, column], shrink=.85, label=field)
    figure.suptitle("RP1: pressure limiter off/on, otherwise identical production solver\n"
                    r"$t=0.25$, $40\times40\times5$ cells, Roe, CFL=0.5, z-periodic; node plane z=0.05"
                    "\nBlue-to-red colors and 25 fixed contours; shared scale per physical variable", fontsize=13)
    figure.savefig(out / "RP1_pressure_limiter_comparison.png", dpi=180, facecolor="white")
    plt.close(figure)
    exact = shock_reference(.25)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained", facecolor="white")
    for column, field in enumerate(("Density", "Pressure")):
        axis = axes[column]
        reference = exact["rho" if column == 0 else "pressure"]
        axis.plot(x, reference, "k--", linewidth=1.7, label="Exact conservative dual mean")
        for flag, color in ((False, "#3366aa"), (True, "#bb3322")):
            values = states["planar_shock", flag][column + 1]
            axis.plot(x, values[2, 20], ".-", color=color, markersize=4,
                      label=f"Pressure limiter {'on' if flag else 'off'}")
            axis.fill_between(x, values.min(axis=(0, 1)), values.max(axis=(0, 1)),
                              color=color, alpha=.10, linewidth=0)
        for position in (exact["boundaries"][0], exact["boundaries"][2]):
            axis.axvline(position, color="gray", linestyle=":", linewidth=.7)
        axis.set(xlabel="x", ylabel=field, xlim=(0, 1), title=field + " at y=0.5, node plane z=0.05")
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
    figure.suptitle("Planar shock: pressure limiter is the only numerical-setting change\n"
                    "Shading: range over y,z; dotted lines: exact shock fronts", fontsize=12)
    figure.savefig(out / "planar_shock_pressure_limiter.png", dpi=180, facecolor="white")
    plt.close(figure)
    columns = ["problem", "limitPressure", "iteration", "rho_z_span_max", "p_z_span_max",
               "w_abs_max", "rho_min", "rho_max", "p_min", "p_max", "rejected_steps",
               "rho_overshoot_exact_states", "p_overshoot_exact_states", "p_overshoot_away_shocks_3dx",
               "rho_L1_exact_dualmean", "rho_L1_volume_weighted_exact_dualmean"]
    with (out / "metrics_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--executable", type=Path, default=ROOT / "build/app/ncfv_euler3D.exe")
    parser.add_argument("--config", type=Path, default=CASE / "RP1_pressure_zperiodic_tol1e10.json")
    parser.add_argument("--ranks", type=int, default=4)
    args = parser.parse_args()
    out = (args.output or ROOT / "data/out/NCFV/pressure_limiter_tests" /
           datetime.now().strftime("%Y%m%d_%H%M%S")).resolve()
    out.mkdir(parents=True, exist_ok=False)
    inputs = out / "inputs"
    inputs.mkdir()
    source_inputs = {
        "ncfv_euler3D.exe": args.executable.resolve(), "RP1_pressure.json": args.config.resolve(),
        "mesh40.cgns": ROOT / "data/out/NCFV/riemann_barth_400x400x5/limiter_audit/mesh40.cgns",
        "RP1_initial.csv": ROOT / "data/out/NCFV/riemann_barth_400x400x5/limiter_audit/initial40/RP1.csv",
        "shock_initial.csv": ROOT / "data/out/NCFV/periodic_planar_shock_analysis/20261026/initial.csv",
        "run_pressure_limiter_tests.py": Path(__file__).resolve(),
        "test_periodic_limiter.py": CASE / "test_periodic_limiter.py",
        "test_planar_shock.py": CASE / "test_planar_shock.py",
        "test_barth_forms.py": CASE / "test_barth_forms.py",
        "NCFVPressureLimiter.hpp": ROOT / "src/NCFV/NCFVPressureLimiter.hpp",
    }
    for stem in ("NCFVConfig", "NCFVReconstruction", "NCFVSpatial", "NCFVSolver"):
        for suffix in (".cpp", ".hpp"):
            source_inputs[stem + suffix] = ROOT / "src/NCFV" / (stem + suffix)
    for name, source in source_inputs.items():
        shutil.copy2(source, inputs / name)
    result = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
              "binary_sha256": digest(inputs / "ncfv_euler3D.exe"), "mpi_ranks": args.ranks,
              "grid": [40, 40, 5], "periodic_lengths": [0, 0, .125], "time": .25, "cfl": .5,
              "controlled_parameter": "reconstruction.limitPressure",
              "frozen_inputs": {name: {"source": str(source), "sha256": digest(inputs / name)}
                                for name, source in source_inputs.items()},
              "metric_notes": "All states are conservative means. RP1 extrema relative to initial pressure are descriptive, not a maximum principle. Planar shock errors use exact conservative dual-volume means; away-shock extrema exclude 3dx around both acoustic shocks.",
              "runs": []}
    save(out / "results.json", result)
    print("Output:", out, flush=True)
    environment = dict(os.environ, OMP_NUM_THREADS="1", DNDS_DIST_OMP_NUM_THREADS="1",
                       OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", HWLOC_COMPONENTS="-gl,-cuda,-nvml",
                       OMPI_MCA_mpi_yield_when_idle="1")
    for key in list(environment):
        if key.startswith(("BJ_", "NCFV_")):
            del environment[key]
    for problem, initial_file in (("RP1", "RP1_initial.csv"), ("planar_shock", "shock_initial.csv")):
        for flag in (False, True):
            directory = out / problem / ("pressureOn" if flag else "pressureOff")
            directory.mkdir(parents=True)
            command = ["mpirun", "--mca", "pml", "ob1", "--mca", "btl", "self,vader", "--bind-to", "none",
                       "-np", str(args.ranks), str(inputs / "ncfv_euler3D.exe"), str(inputs / "RP1_pressure.json")]
            overrides = {"/mesh/meshFile": str(inputs / "mesh40.cgns"), "/mesh/periodicLengths": [0, 0, .125],
                         "/initialField/nodeFile": str(inputs / initial_file),
                         "/reconstruction/limitPressure": flag, "/time/endTime": .25, "/time/cfl": .5,
                         "/time/maximumTimeStep": .001, "/time/rejectNonPhysicalSteps": True,
                         "/physics/riemannSolver": "Roe", "/io/writeVTK": False,
                         "/io/restartInterval": 0, "/io/writeFinalRestart": True,
                         "/io/outputPrefix": str(directory / "solution"),
                         "/io/vtkSeriesName": str(directory / "series"),
                         "/io/restartPrefix": str(directory / "restart")}
            for key, value in overrides.items():
                command += ["-k", key, "-v", json.dumps(value)]
            save(directory / "command.json", {"command": command, "flags": {}, "limitPressure": flag,
                                               "binary_sha256": result["binary_sha256"],
                                               "working_directory": str(ROOT / "build")})
            started = time.monotonic()
            with (directory / "solver.log").open("x") as log:
                process = subprocess.run(command, cwd=ROOT / "build", env=environment,
                                         stdout=log, stderr=subprocess.STDOUT)
            record = {"problem": problem, "limitPressure": flag, "returncode": process.returncode,
                      "elapsed_seconds": time.monotonic() - started}
            if process.returncode == 0:
                restart = sorted(directory.glob("restart_*.dnds.h5"))[-1]
                with h5py.File(restart) as data:
                    raw = data.attrs["configuration"]
                    actual = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
                if actual["reconstruction"].get("limitPressure") != flag:
                    raise ValueError("Saved configuration does not match requested pressure limiter setting")
                record.update(metrics(restart))
                record["resolved_config_sha256"] = digest(directory / "solution.resolved.json")
                record["rejected_steps"] = (directory / "solver.log").read_text().count("NCFV rejected iteration=")
                if problem == "planar_shock":
                    record.update(shock_metrics(restart))
            save(directory / "metrics.json", record)
            result["runs"].append(record)
            save(out / "results.json", result)
            print(json.dumps({key: value for key, value in record.items() if "cut" not in key}), flush=True)
    result["status"] = "complete" if all(r["returncode"] == 0 for r in result["runs"]) else "failed"
    result["finished_utc"] = datetime.now(timezone.utc).isoformat()
    save(out / "results.json", result)
    if result["status"] == "complete":
        plot_results(out, result["runs"])
    else:
        raise RuntimeError("A controlled solver run failed; see results.json and per-run solver.log")


if __name__ == "__main__":
    main()
