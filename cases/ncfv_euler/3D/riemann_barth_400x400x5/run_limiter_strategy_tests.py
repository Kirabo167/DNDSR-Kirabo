"""Controlled discontinuity and accuracy tests for optional limiter strategies.

This runner never compiles or edits production C++. Each variant supplies an
executable, JSON-pointer overrides, and optional private BJ diagnostic flags.
The executable and source/configuration inputs are frozen before computation.

Variant file format::

    {"variants": [
      {"name": "baseline", "overrides": {}},
      {"name": "candidate", "overrides": {"/reconstruction/example": true}}
    ]}

Only numerical switches differ between strategies. RP1/planar-shock tests use
z periodicity; contact and smooth entropy-wave tests use a fully periodic box
to avoid measuring physical-boundary error as limiter dissipation.
"""

import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/ncfv-limiter-strategy-matplotlib")

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np

from generate_mesh import write_mesh
from run_pressure_limiter_tests import shock_metrics, shock_reference
from test_periodic_limiter import ROOT, CASE, digest, save


DEPTH = .125
SPEED = .5
END_TIME = .25


def read_output(path, cells, fully_periodic):
    with h5py.File(path) as data:
        states = data["state/father/array/data"][:].reshape(-1, 5)
        ids = data["state/origIndex"][:].astype(np.int64)
        raw = data.attrs["configuration"]
        settings = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
        actual_time = float(data.attrs["simulationTime"])
        iteration = int(data.attrs["iteration"])
    side = cells if fully_periodic else cells + 1
    expected = 5 * side * side
    if len(states) != expected or not np.array_equal(np.sort(ids), np.arange(expected)):
        raise ValueError("Unexpected quotient node count or identity permutation")
    ordered = np.empty_like(states)
    ordered[ids] = states
    cube = ordered.reshape(5, side, side, 5)
    rho = cube[..., 0]
    pressure = .4 * (cube[..., 4] - .5 * np.sum(cube[..., 1:4] ** 2, axis=-1) / rho)
    if not np.isfinite(cube).all() or np.any(rho <= 0) or np.any(pressure <= 0):
        raise ValueError("Nonphysical or nonfinite output state")
    if abs(actual_time - END_TIME) > 1e-12:
        raise ValueError("Solver did not reach the requested physical end time")
    return cube, rho, pressure, settings, iteration


def transport_reference(problem, cells, time):
    dx = 1 / cells
    x = np.arange(cells) * dx
    if problem == "smooth":
        return 1 + .2 * np.sinc(dx) * np.sin(2 * np.pi * (x - SPEED * time))
    lower, upper = x - dx / 2, x + dx / 2
    fraction = np.zeros(cells)
    for shift in (-1, 0, 1):
        left, right = .25 + SPEED * time + shift, .5 + SPEED * time + shift
        fraction += np.maximum(0, np.minimum(upper, right) - np.maximum(lower, left)) / dx
    return 1 + fraction


def transport_expression(problem, cells):
    dx = 1 / cells
    if problem == "smooth":
        density = f"1 + {(.2 * np.sinc(dx)):.17g} * sin(6.283185307179586*x[0])"
    else:
        density = f"1 + max(0,min(x[0]+{dx/2:.17g},0.5)-max(x[0]-{dx/2:.17g},0.25))/{dx:.17g}"
    return [{"program": ["inRegion := 1;", f"UPrim[0] := {density};",
                          "UPrim[1] := 0.5; UPrim[2] := 0; UPrim[3] := 0; UPrim[4] := 1;", "0;"]}]


def compute_metrics(path, problem, cells, fully_periodic):
    cube, rho, pressure, settings, iteration = read_output(path, cells, fully_periodic)
    result = {"time": END_TIME, "iteration": iteration, "nodes": int(rho.size),
              "restart": str(path), "restart_sha256": digest(path),
              "rho_min": float(rho.min()), "rho_max": float(rho.max()),
              "p_min": float(pressure.min()), "p_max": float(pressure.max()),
              "w_abs_max": float(np.abs(cube[..., 3] / rho).max())}
    for name, values in (("rho", rho), ("p", pressure)):
        span = np.ptp(values, axis=0)
        cyclic_tv = np.abs(np.roll(values, -1, axis=0) - values).sum(axis=0)
        result[name + "_z_span_max"] = float(span.max())
        result[name + "_z_span_rms"] = float(np.sqrt(np.mean(span ** 2)))
        result[name + "_z_cyclic_TV_excess_max"] = float(np.maximum(0, cyclic_tv - 2 * span).max())
        result[name + "_xy_symmetry_abs_max"] = float(np.abs(values - values.transpose(0, 2, 1)).max())
    if problem == "planar_shock":
        result.update(shock_metrics(path))
    if problem in ("contact", "smooth"):
        reference = transport_reference(problem, cells, END_TIME)
        density_error = rho - reference[None, None, :]
        average = rho.mean(axis=(0, 1))
        result.update(rho_L1=float(np.abs(density_error).mean()),
                      rho_Linf=float(np.abs(density_error).max()),
                      pressure_error_max=float(np.abs(pressure - 1).max()),
                      velocity_error_max=float(np.abs(cube[..., 1] / rho - SPEED).max()),
                      transverse_velocity_abs_max=float(np.abs(cube[..., 2:4] / rho[..., None]).max()),
                      rho_yz_span_max=float(np.ptp(rho.reshape(-1, rho.shape[-1]), axis=0).max()),
                      mass_mean_error=float(abs(rho.mean() - transport_reference(problem, cells, 0).mean())),
                      cut_x=(np.arange(cells) / cells).tolist(),
                      cut_density_yz_average=average.tolist(), reference_density=reference.tolist())
        initial_mean_rho = transport_reference(problem, cells, 0).mean()
        initial_mean = np.array([initial_mean_rho, SPEED * initial_mean_rho, 0, 0,
                                 2.5 + .5 * SPEED ** 2 * initial_mean_rho])
        result["conservative_mean_error"] = (cube.mean(axis=(0, 1, 2)) - initial_mean).tolist()
        result["conservative_mean_error_max"] = float(np.abs(cube.mean(axis=(0, 1, 2)) - initial_mean).max())
        if problem == "contact":
            result.update(rho_overshoot=max(0, float(rho.max() - 2)),
                          rho_undershoot=max(0, float(1 - rho.min())),
                          mean_contact_10_90_width=float(np.count_nonzero((average > 1.1) & (average < 1.9)) / cells / 2),
                          contact_10_90_width_valid=bool(average.max() >= 1.9 and average.min() <= 1.1))
        else:
            fourier = 2 * np.mean((average - 1) * np.exp(-2j * np.pi * np.arange(cells) / cells))
            exact_fourier = -1j * .2 * np.sinc(1 / cells) * np.exp(-2j * np.pi * SPEED * END_TIME)
            result.update(fundamental_amplitude_ratio=float(abs(fourier / exact_fourier)),
                          fundamental_phase_error_radians=float(np.angle(fourier / exact_fourier)),
                          rho_overshoot=max(0, float(rho.max() - (1 + .2 * np.sinc(1 / cells)))),
                          rho_undershoot=max(0, float((1 - .2 * np.sinc(1 / cells)) - rho.min())))
    return result, settings


def plot_results(out, runs):
    successful = [record for record in runs if record["returncode"] == 0]
    rp1 = [record for record in successful if record["problem"] == "RP1"]
    if rp1:
        arrays = [read_output(Path(r["restart"]), r["cells"], False)[1:3] for r in rp1]
        ncols = min(3, len(rp1))
        nrows = math.ceil(len(rp1) / ncols)
        for variable, label in enumerate(("density", "pressure")):
            figure, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows),
                                        squeeze=False, layout="constrained", facecolor="white")
            low = min(a[variable].min() for a in arrays)
            high = max(a[variable].max() for a in arrays)
            norm = Normalize(low, high)
            levels = np.linspace(low, high, 27)[1:-1]
            for axis, record, fields in zip(axes.flat, rp1, arrays):
                values = fields[variable][2]
                x = np.linspace(0, 1, values.shape[-1])
                image = axis.pcolormesh(x, x, values, shading="gouraud", norm=norm, cmap="coolwarm")
                axis.contour(x, x, values, levels=levels, colors="black", linewidths=.45, alpha=.65)
                span = record[("rho" if variable == 0 else "p") + "_z_span_max"]
                axis.set_title(f"{record['variant']}\nz span={span:.3g}; max |w|={record['w_abs_max']:.3g}", fontsize=10)
                axis.set(xlabel="x", ylabel="y", xlim=(0, 1), ylim=(0, 1), aspect="equal")
            for axis in list(axes.flat)[len(rp1):]:
                axis.set_visible(False)
            figure.colorbar(image, ax=axes, shrink=.85, label=label.capitalize())
            figure.suptitle("RP1: controlled limiter strategies, t=0.25; shared scale and 25 contours")
            figure.savefig(out / f"RP1_{label}_strategies.png", dpi=170)
            plt.close(figure)
    for problem in ("planar_shock", "contact"):
        selected = [r for r in successful if r["problem"] == problem]
        if not selected:
            continue
        figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
        if problem == "planar_shock":
            exact = shock_reference(END_TIME)
            x = exact["x"]
            exact_rho, exact_p = exact["rho"], exact["pressure"]
        else:
            x = np.arange(selected[0]["cells"]) / selected[0]["cells"]
            exact_rho, exact_p = transport_reference(problem, selected[0]["cells"], END_TIME), np.ones(len(x))
        for axis, values in zip(axes, (exact_rho, exact_p)):
            axis.plot(x, values, "k--", linewidth=1.6, label="Exact dual mean")
        for record in selected:
            _, rho, pressure, _, _ = read_output(Path(record["restart"]), record["cells"], problem == "contact")
            for axis, values in zip(axes, (rho, pressure)):
                axis.plot(x, values.mean(axis=(0, 1)), ".-", markersize=3, label=record["variant"])
        for axis, name in zip(axes, ("Density", "Pressure")):
            axis.set(xlabel="x", ylabel=name, title=f"{problem}: y,z averaged {name.lower()}")
            axis.grid(alpha=.2)
            axis.legend(fontsize=8)
        figure.savefig(out / f"{problem}_strategies.png", dpi=170)
        plt.close(figure)
    smooth = [r for r in successful if r["problem"] == "smooth"]
    if smooth:
        figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
        for name in dict.fromkeys(record["variant"] for record in smooth):
            selected = sorted((r for r in smooth if r["variant"] == name), key=lambda r: r["cells"])
            dx = [1 / r["cells"] for r in selected]
            axes[0].loglog(dx, [r["rho_L1"] for r in selected], "o-", label=name)
            axes[1].plot(dx, [r["fundamental_amplitude_ratio"] for r in selected], "o-", label=name)
        axes[0].set(xlabel="dx", ylabel="Density L1 error", title="Smooth entropy wave: spatial accuracy")
        axes[1].set(xlabel="dx", ylabel="Fundamental amplitude / exact amplitude", title="Smooth-wave dissipation")
        axes[1].axhline(1, color="black", linestyle="--", linewidth=.8)
        for axis in axes:
            axis.grid(alpha=.2)
            axis.legend(fontsize=8)
        figure.savefig(out / "smooth_accuracy_dissipation.png", dpi=170)
        plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--executable", type=Path, default=ROOT / "build/app/ncfv_euler3D.exe")
    parser.add_argument("--config", type=Path, default=CASE / "RP1_pressure_zperiodic_tol1e10.json")
    parser.add_argument("--variant-file", type=Path)
    parser.add_argument("--variants", nargs="+", help="Run selected names from the variant definition")
    parser.add_argument("--resume", action="store_true", help="Continue an existing directory using its frozen binaries and inputs")
    parser.add_argument("--problems", nargs="+", choices=["RP1", "planar_shock", "contact", "smooth"],
                        default=["RP1", "planar_shock", "contact", "smooth"])
    parser.add_argument("--ranks", type=int, default=4)
    args = parser.parse_args()
    defaults = [
        {"name": "base", "overrides": {}},
        {"name": "pressure", "overrides": {"/reconstruction/limitPressure": True}},
        {"name": "pair", "overrides": {"/reconstruction/directionalPairBounds": True}},
        {"name": "flux", "overrides": {"/reconstruction/limitFluxCorrection": True}},
        {"name": "flux_pair", "overrides": {"/reconstruction/limitFluxCorrection": True,
                                             "/reconstruction/fluxCorrectionPairBounds": True}},
        {"name": "flux_only", "overrides": {"/reconstruction/limitFluxCorrection": True,
                                             "/reconstruction/enableLimiter": False,
                                             "/reconstruction/limitPointRecovery": False}},
        {"name": "flux_pair_only", "overrides": {"/reconstruction/limitFluxCorrection": True,
                                                  "/reconstruction/fluxCorrectionPairBounds": True,
                                                  "/reconstruction/enableLimiter": False,
                                                  "/reconstruction/limitPointRecovery": False}},
        {"name": "low_order", "overrides": {"/reconstruction/limitFluxCorrection": True,
                                              "/reconstruction/maximumFluxCorrection": 0}},
        {"name": "complete_rings", "overrides": {"/reconstruction/completeStencilRings": True}},
        {"name": "complete_rings_flux_only", "overrides": {"/reconstruction/completeStencilRings": True,
                                                           "/reconstruction/limitFluxCorrection": True,
                                                           "/reconstruction/enableLimiter": False,
                                                           "/reconstruction/limitPointRecovery": False}},
    ]
    variants = json.loads(args.variant_file.read_text())["variants"] if args.variant_file else defaults
    if args.variants:
        selected = set(args.variants)
        if not selected.issubset({variant["name"] for variant in variants}):
            parser.error("Requested an undefined variant")
        variants = [variant for variant in variants if variant["name"] in selected]
    if any(variant.get("overrides", {}).get("/reconstruction/preserveCoordinateInvariance", False)
           for variant in variants):
        parser.error("Coordinate-derivative modification is excluded from the current strategy experiments")
    names = [variant["name"] for variant in variants]
    if len(set(names)) != len(names) or any(not name or "/" in name for name in names):
        parser.error("Variant names must be nonempty, unique, and contain no slash")
    out = (args.output or ROOT / "data/out/NCFV/limiter_strategy_tests" /
           datetime.now().strftime("%Y%m%d_%H%M%S")).resolve()
    out.mkdir(parents=True, exist_ok=args.resume)
    inputs = out / "inputs"
    inputs.mkdir(exist_ok=args.resume)
    previous = json.loads((out / "results.json").read_text()) if args.resume else None
    if previous:
        for name, information in previous["frozen_inputs"].items():
            if digest(inputs / name) != information["sha256"]:
                raise ValueError("Frozen input changed before resume: " + name)
        prior_variants = {variant["name"]: variant for variant in previous["variants"]}
        for variant in variants:
            if variant["name"] not in prior_variants:
                raise ValueError("Resume may only select variants from the frozen original definition")
            variant["executable"] = prior_variants[variant["name"]]["frozen_executable"]
    frozen = dict(previous["frozen_inputs"]) if previous else {}
    sources = {"case.json": args.config.resolve(), "run_limiter_strategy_tests.py": Path(__file__).resolve()}
    for pattern in ("*.cpp", "*.hpp"):
        for path in (ROOT / "src/NCFV").glob(pattern):
            sources[path.name] = path
    for name in ("generate_mesh.py", "test_periodic_limiter.py", "test_barth_forms.py",
                 "run_pressure_limiter_tests.py", "test_planar_shock.py"):
        sources[name] = CASE / name
    audit = ROOT / "data/out/NCFV/riemann_barth_400x400x5/limiter_audit"
    sources.update({"mesh40.cgns": audit / "mesh40.cgns", "RP1.csv": audit / "initial40/RP1.csv",
                    "shock.csv": ROOT / "data/out/NCFV/periodic_planar_shock_analysis/20261026/initial.csv"})
    if args.variant_file:
        sources["variants.json"] = args.variant_file.resolve()
    for variant in variants:
        executable = Path(variant.get("executable", args.executable)).resolve()
        binary_name = "solver_" + digest(executable)[:16] + ".exe"
        sources[binary_name] = executable
        variant["frozen_executable"] = str(inputs / binary_name)
    for name, source in sources.items():
        if args.resume:
            if name in frozen and digest(inputs / name) != frozen[name]["sha256"]:
                raise ValueError("Frozen input changed before resume: " + name)
            continue
        shutil.copy2(source, inputs / name)
        frozen[name] = {"source": str(source), "sha256": digest(inputs / name)}
    if "smooth" in args.problems and not args.resume:
        write_mesh(inputs / "mesh20.cgns", 20, 20, 5, DEPTH)
        frozen["mesh20.cgns"] = {"source": "Generated Cartesian 20x20x5 mesh", "sha256": digest(inputs / "mesh20.cgns")}
    result = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
              "frozen_inputs": frozen, "variants": variants, "mpi_ranks": args.ranks,
              "time": END_TIME, "cfl": .5, "maximum_time_step": .001, "runs": [],
              "notes": "RP1/planar shock: z periodic. Contact/smooth: fully periodic exact dual means from coordinate expressions. Smooth order refines x,y at fixed 5 z cells and fixed physical depth. RP1 scalar maxima alone are not an oscillation criterion."}
    if previous:
        result = previous
        result.update(status="running", active_variants=[variant["name"] for variant in variants],
                      resumed_utc=datetime.now(timezone.utc).isoformat(),
                      resume_driver_sha256=digest(Path(__file__)))
    save(out / "results.json", result)
    print("Output:", out, flush=True)
    environment = dict(os.environ, OMP_NUM_THREADS="1", DNDS_DIST_OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                       MKL_NUM_THREADS="1", HWLOC_COMPONENTS="-gl,-cuda,-nvml", OMPI_MCA_mpi_yield_when_idle="1")
    environment = {name: value for name, value in environment.items() if not name.startswith(("BJ_", "NCFV_"))}
    for problem in args.problems:
        for cells in ((20, 40) if problem == "smooth" else (40,)):
            fully_periodic = problem in ("contact", "smooth")
            for variant in variants:
                directory = out / problem / str(cells) / variant["name"]
                directory.mkdir(parents=True, exist_ok=args.resume)
                if args.resume and any(r["problem"] == problem and r["cells"] == cells and
                                       r["variant"] == variant["name"] and r["returncode"] == 0
                                       for r in result["runs"]):
                    continue
                overrides = {"/mesh/meshFile": str(inputs / f"mesh{cells}.cgns"),
                             "/mesh/periodicLengths": [1, 1, DEPTH] if fully_periodic else [0, 0, DEPTH],
                             "/mesh/periodicBoundaryPairs": ["XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"] if fully_periodic else ["ZMIN", "ZMAX"],
                             "/physics/boundaryZones": [{"name": name, "mode": "Periodic" if fully_periodic or name.startswith("Z") else "SupersonicOutlet"}
                                                        for name in ("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX")],
                             "/initialField/nodeFile": "" if fully_periodic else str(inputs / ("RP1.csv" if problem == "RP1" else "shock.csv")),
                             "/initialField/expressions": transport_expression(problem, cells) if fully_periodic else [],
                             "/reconstruction/enableLimiter": True, "/reconstruction/limitPointRecovery": True,
                             "/reconstruction/limitPressure": False,
                             "/reconstruction/directionalPairBounds": False,
                             "/reconstruction/limitFluxCorrection": False,
                             "/reconstruction/fluxCorrectionPairBounds": False,
                             "/reconstruction/maximumFluxCorrection": 1,
                             "/reconstruction/completeStencilRings": False,
                             "/time/endTime": END_TIME, "/time/cfl": .5, "/time/maximumTimeStep": .001,
                             "/time/rejectNonPhysicalSteps": True, "/physics/riemannSolver": "Roe",
                             "/io/writeVTK": False, "/io/restartInterval": 0, "/io/writeFinalRestart": True,
                             "/io/outputPrefix": str(directory / "solution"), "/io/vtkSeriesName": str(directory / "series"),
                             "/io/restartPrefix": str(directory / "restart")}
                overrides.update(variant.get("overrides", {}))
                flags = variant.get("environment", {})
                if any(not key.startswith("BJ_") for key in flags):
                    raise ValueError("Only private BJ diagnostic flags may be supplied in variant environment")
                command = ["mpirun", "--mca", "pml", "ob1", "--mca", "btl", "self,vader", "--bind-to", "none",
                           "-np", str(args.ranks), variant["frozen_executable"], str(inputs / "case.json")]
                for key, value in overrides.items():
                    command += ["-k", key, "-v", json.dumps(value)]
                recovered = args.resume and bool(list(directory.glob("restart_*.dnds.h5")))
                if not recovered:
                    save(directory / "command.json", {"command": command, "environment_flags": flags,
                                                       "overrides": overrides, "binary_sha256": digest(variant["frozen_executable"])})
                    started = time.monotonic()
                    with (directory / "solver.log").open("x") as log:
                        process = subprocess.run(command, cwd=ROOT / "build", env=dict(environment, **flags),
                                                 stdout=log, stderr=subprocess.STDOUT)
                record = {"problem": problem, "cells": cells, "variant": variant["name"],
                          "fully_periodic": fully_periodic, "returncode": 0 if recovered else process.returncode,
                          "elapsed_seconds": None if recovered else time.monotonic() - started,
                          "environment_flags": flags, "recovered_completed_output": recovered}
                if record["returncode"] == 0:
                    restart = sorted(directory.glob("restart_*.dnds.h5"))[-1]
                    measured, settings = compute_metrics(restart, problem, cells, fully_periodic)
                    for pointer, expected in variant.get("overrides", {}).items():
                        actual = settings
                        for name in pointer.strip("/").split("/"):
                            actual = actual[name]
                        if actual != expected:
                            raise ValueError(f"Inactive variant override: {pointer}")
                    record.update(measured)
                    record["rejected_steps"] = (directory / "solver.log").read_text().count("NCFV rejected iteration=")
                    previous_directory = ROOT / "data/out/NCFV/pressure_limiter_tests/20261026_120750" / problem / "pressureOff"
                    previous_files = sorted(previous_directory.glob("restart_*.dnds.h5"))
                    if variant["name"] == "base" and problem in ("RP1", "planar_shock") and previous_files:
                        previous = read_output(previous_files[-1], cells, False)[0]
                        current = read_output(restart, cells, False)[0]
                        record["base_compatibility_previous"] = str(previous_files[-1])
                        record["base_compatibility_bitwise_equal"] = bool(np.array_equal(previous, current))
                        record["base_compatibility_max_state_difference"] = float(np.abs(previous - current).max())
                save(directory / "metrics.json", record)
                result["runs"].append(record)
                save(out / "results.json", result)
                print(json.dumps({key: value for key, value in record.items() if "cut" not in key and key != "reference_density"}), flush=True)
    orders = {}
    for variant in variants:
        smooth = {r["cells"]: r for r in result["runs"] if r["variant"] == variant["name"] and r["problem"] == "smooth" and r["returncode"] == 0}
        if 20 in smooth and 40 in smooth:
            orders[variant["name"]] = math.log(smooth[20]["rho_L1"] / smooth[40]["rho_L1"]) / math.log(2)
    result.update(status="complete" if all(r["returncode"] == 0 for r in result["runs"]) else "failed",
                  smooth_observed_L1_order_20_to_40=orders, finished_utc=datetime.now(timezone.utc).isoformat())
    save(out / "results.json", result)
    columns = ["problem", "cells", "variant", "returncode", "iteration", "rho_z_span_max", "p_z_span_max", "w_abs_max",
               "rho_min", "rho_max", "p_min", "p_max", "rho_L1", "rho_Linf", "pressure_error_max",
               "fundamental_amplitude_ratio", "mean_contact_10_90_width", "rho_overshoot", "rho_undershoot",
               "rho_overshoot_exact_states", "p_overshoot_away_shocks_3dx", "rho_L1_exact_dualmean"]
    with (out / "metrics_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(result["runs"])
    plot_results(out, result["runs"])


if __name__ == "__main__":
    main()
