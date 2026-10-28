"""Compare general limiter strategies on genuinely three-dimensional fields.

The entropy wave has nonzero x/y/z gradients, velocity (.3,.2,.1), and exact
Cartesian dual means. The oblique two-shock problem has normal (1,2,3)/sqrt(14)
and exact box/half-space intersection fractions. No derivative is overwritten.
Only a frozen production executable is run; this script never compiles C++.

--prepare-only generates meshes/initial fields and validates the references.
--run-prepared --output DIR freezes the executable and runs that prepared set.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

import h5py
import numpy as np

from generate_mesh import write_mesh
from test_planar_shock import LEFT, RIGHT, exact_shock


ROOT = Path(__file__).resolve().parents[4]
CASE = Path(__file__).resolve().parent
VELOCITY = np.array([.3, .2, .1])
NORMAL = np.array([1., 2., 3.]) / np.sqrt(14.)
PLANE_CENTER = float(NORMAL.sum() / 2)
GAMMA = 1.4
VARIANTS = {
    "base": {},
    "pair": {"/reconstruction/directionalPairBounds": True},
    "flux": {"/reconstruction/limitFluxCorrection": True},
    "flux_pair": {"/reconstruction/limitFluxCorrection": True,
                  "/reconstruction/fluxCorrectionPairBounds": True},
    "flux_only": {"/reconstruction/limitFluxCorrection": True,
                  "/reconstruction/enableLimiter": False,
                  "/reconstruction/limitPointRecovery": False},
    "flux_pair_only": {"/reconstruction/limitFluxCorrection": True,
                       "/reconstruction/fluxCorrectionPairBounds": True,
                       "/reconstruction/enableLimiter": False,
                       "/reconstruction/limitPointRecovery": False},
    "low_order": {"/reconstruction/limitFluxCorrection": True,
                  "/reconstruction/maximumFluxCorrection": 0},
    "complete_rings": {"/reconstruction/completeStencilRings": True},
    "complete_rings_flux_only": {"/reconstruction/completeStencilRings": True,
                                 "/reconstruction/limitFluxCorrection": True,
                                 "/reconstruction/enableLimiter": False,
                                 "/reconstruction/limitPointRecovery": False},
}
DEFAULT_VARIANTS = ["base", "pair", "flux_pair", "flux_pair_only", "low_order"]


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def matches_override(actual, expected):
    """Resolved typed objects may contain defaults absent from an override."""
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(
            key in actual and matches_override(actual[key], value) for key, value in expected.items())
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(
            matches_override(left, right) for left, right in zip(actual, expected))
    return actual == expected


def coordinates(cells, periodic):
    side = cells if periodic else cells + 1
    z, y, x = np.indices((side, side, side), dtype=float) / cells
    return np.column_stack([x.ravel(), y.ravel(), z.ravel()])


def primitive(states):
    density = states[:, 0]
    velocity = states[:, 1:4] / density[:, None]
    pressure = (GAMMA - 1) * (states[:, 4] - .5 * density * np.sum(velocity ** 2, axis=1))
    return density, velocity, pressure


def uniform_velocity_states(density, velocity=VELOCITY, pressure=1):
    return np.column_stack([density, density[:, None] * velocity,
                            pressure / (GAMMA - 1) + .5 * density * np.dot(velocity, velocity)])


def smooth_reference(cells, physical_time):
    xyz = coordinates(cells, True)
    amplitude = .2 * np.sinc(1 / cells) ** 3
    phase = 2 * np.pi * (xyz.sum(axis=1) - VELOCITY.sum() * physical_time)
    density = 1 + amplitude * np.sin(phase)
    return uniform_velocity_states(density)


def smooth_expression(cells):
    amplitude = .2 * np.sinc(1 / cells) ** 3
    return [{"program": ["inRegion := 1;",
                         f"UPrim[0] := 1 + {amplitude:.17g} * sin(6.283185307179586*(x[0]+x[1]+x[2]));",
                         "UPrim[1] := 0.3; UPrim[2] := 0.2; UPrim[3] := 0.1; UPrim[4] := 1;", "0;"]}]


def box_projection_cdf(lower, upper, threshold, normal=NORMAL):
    """Exact fraction n.dot(x)<=threshold in each axis-aligned box.

    Positive normal components are required. Long-double inclusion/exclusion
    evaluates the eight truncated cubes; no numerical volume quadrature is used.
    """
    lower = np.asarray(lower, dtype=np.longdouble)
    upper = np.asarray(upper, dtype=np.longdouble)
    normal = np.asarray(normal, dtype=np.longdouble)
    if np.any(normal <= 0) or np.any(upper <= lower):
        raise ValueError("CDF requires positive projection lengths")
    lengths = (upper - lower) * normal
    distance = np.asarray(threshold, dtype=np.longdouble) - lower @ normal
    support = lengths.sum(axis=1)
    fraction = np.zeros(len(lower), dtype=np.longdouble)
    fraction[distance >= support] = 1
    active = (distance > 0) & (distance < support)
    if np.any(active):
        projected = lengths[active]
        partial = np.zeros(int(active.sum()), dtype=np.longdouble)
        for mask in range(8):
            shift = sum(projected[:, axis] for axis in range(3) if mask & (1 << axis))
            partial += (-1 if mask.bit_count() % 2 else 1) * np.maximum(distance[active] - shift, 0) ** 3
        fraction[active] = partial / (6 * np.prod(projected, axis=1))
    return np.asarray(np.clip(fraction, 0, 1), dtype=float)


def oblique_conservative(state):
    density, normal_velocity, pressure = state
    velocity = normal_velocity * NORMAL
    return np.array([density, *(density * velocity),
                     pressure / (GAMMA - 1) + .5 * density * normal_velocity ** 2])


def oblique_reference(cells, physical_time):
    xyz = coordinates(cells, False)
    lower, upper = np.maximum(0, xyz - .5 / cells), np.minimum(1, xyz + .5 / cells)
    fronts, states = exact_shock(physical_time)
    fronts = np.asarray(fronts) - .5 + PLANE_CENTER
    cumulative = [np.zeros(len(xyz))]
    cumulative.extend(box_projection_cdf(lower, upper, front) for front in fronts)
    cumulative.append(np.ones(len(xyz)))
    reference = np.zeros((len(xyz), 5))
    for region, state in enumerate(states):
        fraction = cumulative[region + 1] - cumulative[region]
        if fraction.min() < -1e-12:
            raise ValueError("Nonmonotone exact front fractions")
        reference += fraction[:, None] * oblique_conservative(state)
    weights = np.prod(upper - lower, axis=1)
    return reference, xyz, weights, fronts, states


def validate_references(cells):
    lower = np.zeros((3, 3))
    upper = np.ones((3, 3))
    equal_normal = np.ones(3) / np.sqrt(3)
    measured = box_projection_cdf(lower, upper, np.array([1, 1.5, 2]) / np.sqrt(3), equal_normal)
    np.testing.assert_allclose(measured, [1 / 6, .5, 5 / 6], atol=2e-14, rtol=0)
    np.testing.assert_array_equal(box_projection_cdf(lower, upper, -1), np.zeros(3))
    np.testing.assert_array_equal(box_projection_cdf(lower, upper, 10), np.ones(3))
    np.testing.assert_allclose(box_projection_cdf(lower, upper, PLANE_CENTER), .5, atol=2e-14, rtol=0)
    smooth = smooth_reference(cells, 0)
    smooth_expected_mean = uniform_velocity_states(np.ones(1))[0]
    smooth_conservation = np.max(np.abs(smooth.mean(axis=0) - smooth_expected_mean))
    initial, _, weights, _, _ = oblique_reference(cells, 0)
    oblique_expected_mean = .5 * (oblique_conservative(LEFT) + oblique_conservative(RIGHT))
    oblique_conservation = np.max(np.abs(weights @ initial - oblique_expected_mean))
    for values in (smooth, initial):
        density, _, pressure = primitive(values)
        if not np.isfinite(values).all() or density.min() <= 0 or pressure.min() <= 0:
            raise ValueError("Invalid exact initial conservative means")
    if smooth_conservation > 1e-12 or oblique_conservation > 1e-12:
        raise ValueError("Initial reference failed its analytic volume integral")
    return {"cells": cells, "CDF_equal_projection_values": measured.tolist(),
            "CDF_limits_and_center_pass": True, "smooth_initial_mean_error": float(smooth_conservation),
            "oblique_initial_integral_error": float(oblique_conservation),
            "oblique_minimum_initial_pressure": float(primitive(initial)[2].min()),
            "smooth_true_w": .1, "oblique_normal": NORMAL.tolist(),
            "smooth_gradient_rms_each_axis": float(.2 * np.sinc(1 / cells) ** 3 * 2 * np.pi / np.sqrt(2)),
            "reference_integration": "Analytic smooth sinc factor and eight-term box CDF; only floating-point roundoff"}


def prepare(out, smooth_sizes, shock_size, problems):
    inputs = out / "inputs"
    inputs.mkdir(parents=True)
    sizes = sorted(set(smooth_sizes if "smooth" in problems else []) |
                   ({shock_size} if "oblique_shock" in problems else set()))
    checks = []
    for cells in sizes:
        checks.append(validate_references(cells))
        write_mesh(inputs / f"mesh{cells}.cgns", cells, cells, cells, 1)
    if "oblique_shock" in problems:
        initial = oblique_reference(shock_size, 0)[0]
        np.savetxt(inputs / f"oblique_initial{shock_size}.csv",
                   np.column_stack([np.arange(len(initial)), initial]), delimiter=",",
                   header="globalNode,rho,rhou,rhov,rhow,rhoE", comments="", fmt=["%d"] + ["%.17g"] * 5)
    record = {"smooth_sizes": smooth_sizes, "shock_size": shock_size, "problems": problems,
              "reference_validation": checks, "status": "prepared",
              "prepared_utc": datetime.now(timezone.utc).isoformat()}
    save(out / "prepared.json", record)
    return record


def load_states(path, cells, periodic, target_time):
    with h5py.File(path) as data:
        values = data["state/father/array/data"][:].reshape(-1, 5)
        ids = data["state/origIndex"][:].astype(np.int64)
        actual_time = float(data.attrs["simulationTime"])
        iteration = int(data.attrs["iteration"])
        raw = data.attrs["configuration"]
        configuration = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
    side = cells if periodic else cells + 1
    if len(values) != side ** 3 or not np.array_equal(np.sort(ids), np.arange(side ** 3)):
        raise ValueError("Unexpected original-index permutation after periodic quotient")
    ordered = np.empty_like(values)
    ordered[ids] = values
    density, _, pressure = primitive(ordered)
    if abs(actual_time - target_time) > 1e-12:
        raise ValueError("Run did not reach its target time")
    if not np.isfinite(ordered).all() or density.min() <= 0 or pressure.min() <= 0:
        raise ValueError("Nonphysical final state")
    return ordered, configuration, iteration


def measure(path, cells, problem):
    target_time = .25 if problem == "smooth" else .05
    values, configuration, iteration = load_states(path, cells, problem == "smooth", target_time)
    density, velocity, pressure = primitive(values)
    record = {"time": target_time, "iteration": iteration, "nodes": len(values),
              "restart": str(path), "restart_sha256": digest(path),
              "rho_min": float(density.min()), "rho_max": float(density.max()),
              "pressure_min": float(pressure.min()), "pressure_max": float(pressure.max()),
              "w_min": float(velocity[:, 2].min()), "w_max": float(velocity[:, 2].max())}
    if problem == "smooth":
        exact = smooth_reference(cells, target_time)
        xyz = coordinates(cells, True)
        error = density - exact[:, 0]
        amplitude = .2 * np.sinc(1 / cells) ** 3
        coefficient = 2 * np.mean((density - 1) * np.exp(-2j * np.pi * xyz.sum(axis=1)))
        exact_coefficient = -1j * amplitude * np.exp(-2j * np.pi * VELOCITY.sum() * target_time)
        initial_mean = uniform_velocity_states(np.ones(1))[0]
        conservation = values.mean(axis=0) - initial_mean
        record.update(rho_L1=float(np.abs(error).mean()), rho_Linf=float(np.abs(error).max()),
                      fundamental_amplitude_ratio=float(abs(coefficient / exact_coefficient)),
                      fundamental_phase_error_radians=float(np.angle(coefficient / exact_coefficient)),
                      velocity_error_max_by_component=np.max(np.abs(velocity - VELOCITY), axis=0).tolist(),
                      pressure_error_max=float(np.max(np.abs(pressure - 1))),
                      conservative_mean_error=conservation.tolist(),
                      conservative_mean_error_max=float(np.abs(conservation).max()),
                      exact_w=.1, w_error_max=float(np.max(np.abs(velocity[:, 2] - .1))))
    else:
        exact, xyz, weights, fronts, states = oblique_reference(cells, target_time)
        exact_density, exact_velocity, exact_pressure = primitive(exact)
        interior = np.all((xyz >= .2 - 1e-12) & (xyz <= .8 + 1e-12), axis=1)
        roi_weights = weights[interior] / weights[interior].sum()
        normal_speed = velocity @ NORMAL
        tangential = velocity - normal_speed[:, None] * NORMAL
        exact_density_range = [min(state[0] for state in states), max(state[0] for state in states)]
        exact_pressure_range = [min(state[2] for state in states), max(state[2] for state in states)]
        record.update(interior_box=[.2, .8], interior_nodes=int(interior.sum()),
                      rho_L1=float(roi_weights @ np.abs(density[interior] - exact_density[interior])),
                      rho_Linf=float(np.abs(density[interior] - exact_density[interior]).max()),
                      pressure_L1=float(roi_weights @ np.abs(pressure[interior] - exact_pressure[interior])),
                      pressure_Linf=float(np.abs(pressure[interior] - exact_pressure[interior]).max()),
                      velocity_error_max_by_component=np.max(np.abs(velocity[interior] - exact_velocity[interior]), axis=0).tolist(),
                      tangential_velocity_abs_max=float(np.linalg.norm(tangential[interior], axis=1).max()),
                      w_error_L1=float(roi_weights @ np.abs(velocity[interior, 2] - exact_velocity[interior, 2])),
                      rho_overshoot=max(0., float(density[interior].max() - exact_density_range[1])),
                      rho_undershoot=max(0., float(exact_density_range[0] - density[interior].min())),
                      pressure_overshoot=max(0., float(pressure[interior].max() - exact_pressure_range[1])),
                      pressure_undershoot=max(0., float(exact_pressure_range[0] - pressure[interior].min())),
                      exact_normal_fronts=fronts.tolist(), exact_states_rho_un_p=states,
                      volume_integral_error_vs_exact=(weights @ (values - exact)).tolist(),
                      open_boundary_note="Total state is not conserved in an outlet box; integral difference is measured against the exact moving fronts")
    np.savez_compressed(path.parent / "comparison_arrays.npz", actual=values, exact=exact, coordinates=xyz)
    return record, configuration


def run(out, prepared, args):
    inputs = out / "inputs"
    binary = args.executable.resolve()
    frozen = {"solver.exe": binary, "case.json": args.config.resolve(),
              "run_true3d_limiter_tests.py": Path(__file__).resolve(),
              "generate_mesh.py": CASE / "generate_mesh.py", "test_planar_shock.py": CASE / "test_planar_shock.py"}
    for pattern in ("*.cpp", "*.hpp"):
        frozen.update({path.name: path for path in (ROOT / "src/NCFV").glob(pattern)})
    manifest = {}
    for name, source in frozen.items():
        target = inputs / name
        if target.exists():
            raise FileExistsError(f"refusing to replace frozen input {target}")
        shutil.copy2(source, target)
        manifest[name] = {"source": str(source), "sha256": digest(target)}
    for name in ("case.json", "NCFVConfig.hpp", "NCFVReconstruction.hpp", "NCFVReconstruction.cpp"):
        if "preserveCoordinateInvariance" in (inputs / name).read_text():
            raise ValueError("Frozen inputs still contain the removed derivative-overwrite experiment")
    results = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
               "mpi_ranks": args.ranks, "variants": args.variants, "frozen_inputs": manifest,
               "reference_validation": prepared["reference_validation"], "runs": [],
               "notes": "All three coordinates vary. Smooth t=.25 fully periodic; oblique shock t=.05 outlet boundaries, errors in [.2,.8]^3. CFL=.5, no imposed small maximum time step. No coordinate derivative modification."}
    save(out / "results.json", results)
    environment = dict(os.environ, OMP_NUM_THREADS="1", DNDS_DIST_OMP_NUM_THREADS="1",
                       OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", HWLOC_COMPONENTS="-gl,-cuda,-nvml",
                       OMPI_MCA_mpi_yield_when_idle="1")
    environment = {key: value for key, value in environment.items() if not key.startswith(("BJ_", "NCFV_"))}
    for problem in prepared["problems"]:
        sizes = prepared["smooth_sizes"] if problem == "smooth" else [prepared["shock_size"]]
        for cells in sizes:
            for name in args.variants:
                directory = out / problem / str(cells) / name
                directory.mkdir(parents=True)
                periodic = problem == "smooth"
                overrides = {"/mesh/meshFile": str(inputs / f"mesh{cells}.cgns"),
                             "/mesh/periodicLengths": [1, 1, 1] if periodic else [0, 0, 0],
                             "/mesh/periodicBoundaryPairs": ["XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"] if periodic else [],
                             "/physics/boundaryZones": [{"name": side, "mode": "Periodic" if periodic else "SupersonicOutlet"}
                                                        for side in ("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX")],
                             "/initialField/nodeFile": "" if periodic else str(inputs / f"oblique_initial{cells}.csv"),
                             "/initialField/expressions": smooth_expression(cells) if periodic else [],
                             "/reconstruction/enableLimiter": True, "/reconstruction/limitPointRecovery": True,
                             "/reconstruction/limitPressure": False, "/reconstruction/directionalPairBounds": False,
                             "/reconstruction/limitFluxCorrection": False, "/reconstruction/fluxCorrectionPairBounds": False,
                             "/reconstruction/maximumFluxCorrection": 1,
                             "/physics/riemannSolver": "Roe", "/time/endTime": .25 if periodic else .05,
                             "/time/iterations": 1000000, "/time/cfl": .5, "/time/maximumTimeStep": 1,
                             "/time/useCFLTimeStep": True, "/time/useLocalTimeStep": False,
                             "/time/rejectNonPhysicalSteps": True, "/time/maximumStepRetries": 20,
                             "/io/writeVTK": False, "/io/writeInitial": False, "/io/writeFinal": False,
                             "/io/restartInterval": 0, "/io/writeFinalRestart": True,
                             "/io/writeResolvedConfiguration": True,
                             "/io/outputPrefix": str(directory / "solution"),
                             "/io/vtkSeriesName": str(directory / "series"), "/io/restartPrefix": str(directory / "restart")}
                overrides.update(VARIANTS[name])
                command = ["mpirun", "--mca", "pml", "ob1", "--mca", "btl", "self,vader", "--bind-to", "none",
                           "-np", str(args.ranks), str(inputs / "solver.exe"), str(inputs / "case.json")]
                for pointer, value in overrides.items():
                    command.extend(["-k", pointer, "-v", json.dumps(value)])
                save(directory / "command.json", {"command": command, "overrides": overrides})
                started = time.monotonic()
                with (directory / "solver.log").open("x") as log:
                    process = subprocess.run(command, cwd=ROOT / "build", env=environment,
                                             stdout=log, stderr=subprocess.STDOUT)
                record = {"problem": problem, "cells": cells, "variant": name,
                          "returncode": process.returncode, "elapsed_seconds": time.monotonic() - started}
                if process.returncode == 0:
                    restart = sorted(directory.glob("restart_*.dnds.h5"))[-1]
                    metrics, configuration = measure(restart, cells, problem)
                    for pointer, expected in overrides.items():
                        actual = configuration
                        for field in pointer.strip("/").split("/"):
                            actual = actual[field]
                        if not matches_override(actual, expected):
                            raise ValueError(f"Inactive override: {pointer}")
                    record.update(metrics)
                    record["rejected_steps"] = (directory / "solver.log").read_text().count("NCFV rejected iteration=")
                save(directory / "metrics.json", record)
                results["runs"].append(record)
                save(out / "results.json", results)
                print(json.dumps(record), flush=True)
    orders = {}
    for variant in args.variants:
        records = sorted((record for record in results["runs"] if record["problem"] == "smooth" and
                          record["variant"] == variant and record["returncode"] == 0), key=lambda record: record["cells"])
        if len(records) > 1:
            orders[variant] = [{"sizes": [coarse["cells"], fine["cells"]],
                                "observed_L1_order": math.log(coarse["rho_L1"] / fine["rho_L1"]) /
                                math.log(fine["cells"] / coarse["cells"])} for coarse, fine in zip(records, records[1:])]
    results.update(status="complete" if all(record["returncode"] == 0 for record in results["runs"]) else "failed",
                   smooth_observed_orders=orders, finished_utc=datetime.now(timezone.utc).isoformat())
    save(out / "results.json", results)
    fields = ["problem", "cells", "variant", "returncode", "iteration", "elapsed_seconds", "rho_L1", "rho_Linf",
              "fundamental_amplitude_ratio", "pressure_error_max", "pressure_L1", "pressure_overshoot",
              "rho_overshoot", "conservative_mean_error_max", "w_error_max", "tangential_velocity_abs_max"]
    with (out / "metrics_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results["runs"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--executable", type=Path, default=ROOT / "build/app/ncfv_euler3D.exe")
    parser.add_argument("--config", type=Path, default=CASE / "RP1_zperiodic_tol1e10.json")
    parser.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=DEFAULT_VARIANTS)
    parser.add_argument("--problems", nargs="+", choices=["smooth", "oblique_shock"], default=["smooth", "oblique_shock"])
    parser.add_argument("--smooth-sizes", nargs="+", type=int, default=[20])
    parser.add_argument("--shock-size", type=int, default=20)
    parser.add_argument("--ranks", type=int, default=4)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--run-prepared", action="store_true")
    args = parser.parse_args()
    if min([args.shock_size, *args.smooth_sizes]) < 4 or args.ranks < 1 or len(set(args.variants)) != len(args.variants):
        parser.error("Use at least four cells per axis, positive ranks, and unique variants")
    out = (args.output or ROOT / "data/out/NCFV/true3d_limiter_tests" /
           datetime.now().strftime("%Y%m%d_%H%M%S")).resolve()
    if args.run_prepared:
        if args.prepare_only:
            parser.error("Choose either preparation or running prepared inputs")
        prepared = json.loads((out / "prepared.json").read_text())
    else:
        out.mkdir(parents=True, exist_ok=False)
        prepared = prepare(out, args.smooth_sizes, args.shock_size, args.problems)
    print("Output:", out, flush=True)
    if not args.prepare_only:
        run(out, prepared, args)


if __name__ == "__main__":
    main()
