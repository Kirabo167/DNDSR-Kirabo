"""Audit iv10/20/40/80 efficient vortex runs and report observed convergence.

No solver or pybind11 module is imported. Error norms use recovered nodal density,
not dual-cell averages, and the reference density is evaluated independently.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.ticker import NullFormatter
import numpy as np

from analyze_iv40 import load_nodes


ROOT = Path(__file__).resolve().parents[3]
NORMS = ("L1", "L2", "Linf")


def exact_density(xyz, time):
    delta = xyz[:, :2] - 5.0 - time
    delta -= 10.0 * np.floor(delta / 10.0 + 0.5)
    temperature = 1.0 - 0.4 * 25.0 / (8.0 * 1.4 * np.pi**2) * np.exp(
        1.0 - np.sum(delta**2, axis=1))
    return temperature**2.5


def norms(error, weights):
    return dict(zip(NORMS, (float(np.sum(weights * np.abs(error)) / weights.sum()),
                           float(np.sqrt(np.sum(weights * error**2) / weights.sum())),
                           float(np.max(np.abs(error))))))


def audit_mesh(n):
    path = ROOT / f"cases/ncfv_euler/3D/iv{n}_3d_1_dnds.cgns"
    manifest = json.loads(path.with_suffix(".manifest.json").read_text())
    with h5py.File(path) as mesh:
        zone = mesh["Base/blk-1"]
        coords = []
        for axis in ("CoordinateX", "CoordinateY", "CoordinateZ"):
            array = zone[f"GridCoordinates/{axis}/ data"][:].ravel()
            if hashlib.sha256(array.tobytes()).hexdigest() != manifest["coordinates"][axis]["sha256"]:
                raise ValueError(f"iv{n}: coordinate conversion mismatch")
            coords.append(array)
        for section in manifest["sections"]:
            array = zone[f"{section['name']}/ElementConnectivity/ data"][:].ravel()
            if hashlib.sha256(array.tobytes()).hexdigest() != section["sha256"]:
                raise ValueError(f"iv{n}: connectivity conversion mismatch")
        cells = zone["PrismElements/ElementConnectivity/ data"][:].reshape(-1, 6) - 1
    return manifest, np.column_stack(coords), cells


def audit_run(n, directory, log_path=None):
    prefix = directory / "solution"
    diagnostics = np.atleast_1d(np.genfromtxt(str(prefix) + ".diagnostics.csv", delimiter=",", names=True))
    if (len(diagnostics) != 2 or abs(diagnostics[-1]["time"] - 2.0) > 1e-12
            or diagnostics[0]["time"] != 0 or diagnostics[0]["iteration"] != 0):
        raise ValueError(f"iv{n}: run is not complete at t=2")
    configuration = json.loads(Path(str(prefix) + ".resolved.json").read_text())
    if configuration["algorithm"]["mode"] != "EfficientDifferential":
        raise ValueError("Only efficient runs may enter this convergence study")
    if configuration["time"]["useLocalTimeStep"]:
        raise ValueError("Local pseudo-time is invalid for this unsteady comparison")
    manifest, xyz, cells = audit_mesh(n)
    data = []
    errors = []
    for diagnostic in diagnostics:
        nodes = load_nodes(prefix, int(diagnostic["iteration"]))
        if not np.array_equal(nodes["original_node"], np.arange(len(xyz))):
            raise ValueError("Original mesh node ownership is incomplete")
        output_xyz = np.column_stack([nodes[k] for k in ("x", "y", "z")])
        if not np.allclose(output_xyz, xyz, atol=1e-13, rtol=0):
            raise ValueError("Output node coordinates do not match original mesh")
        if not np.isfinite(nodes.view(np.float64)).all() or np.min(nodes["partial_volume"]) <= 0:
            raise ValueError("Non-finite data or nonpositive dual volume")
        exact = exact_density(xyz, float(diagnostic["time"]))
        if not np.allclose(exact, nodes["rho_exact"], rtol=0, atol=3e-14):
            raise ValueError("Independent vortex reference disagrees with solver")
        error = norms(nodes["rho_point"] - exact, nodes["partial_volume"])
        for name in NORMS:
            if not np.isclose(error[name], diagnostic[f"rho_point_{name}"], rtol=3e-12, atol=2e-15):
                raise ValueError("Independent error reduction disagrees with solver")
        data.append(nodes)
        errors.append(error)
    first, final = data
    volume = float(np.sum(final["partial_volume"]))
    if abs(volume - 400.0) > 1e-9:
        raise ValueError("Dual volumes do not sum to domain volume")
    lengths = np.array([10.0, 10.0, 4.0])
    wrapped = xyz.copy()
    for d, length in enumerate(lengths):
        wrapped[np.isclose(wrapped[:, d], length, rtol=0, atol=1e-8), d] = 0
    _, inverse = np.unique(np.round(wrapped, 8), axis=0, return_inverse=True)
    n_periodic = int(inverse.max() + 1)
    n_xy = len(np.unique(np.round(wrapped[:, :2], 8), axis=0))
    periodic_difference = 0.0
    for field in ("rho_mean", "rhou", "rhov", "rhow", "rhoE"):
        minimum, maximum = np.full(n_periodic, np.inf), np.full(n_periodic, -np.inf)
        np.minimum.at(minimum, inverse, final[field])
        np.maximum.at(maximum, inverse, final[field])
        periodic_difference = max(periodic_difference, float(np.max(maximum - minimum)))
    if periodic_difference > 1e-11:
        raise ValueError("Periodic state copies disagree")
    conserved = ("mass", "momentum_x", "momentum_y", "momentum_z", "total_energy")
    drift = {key: float((diagnostics[-1][key] - diagnostics[0][key]) /
                       max(abs(diagnostics[0][key]), 1.0)) for key in conserved}
    momentum_squared = sum(final[k]**2 for k in ("rhou", "rhov", "rhow"))
    pressure_of_mean = 0.4 * (final["rhoE"] - 0.5 * momentum_squared / final["rho_mean"])
    if final["rho_mean"].min() <= 0 or pressure_of_mean.min() <= 0:
        raise ValueError("Nonphysical final conservative means")
    metrics = {
        "mesh": f"iv{n}", "nominal_resolution": n, "nodes": len(final), "cells": manifest["cells"],
        "periodic_unknowns": n_periodic, "periodic_xy_nodes": n_xy,
        "z_layers": len(np.unique(xyz[:, 2])) - 1,
        "h_3d": float((volume / n_periodic)**(1.0 / 3.0)),
        "h_xy": float(np.sqrt(100.0 / n_xy)), "h_nominal": 10.0 / n,
        "steps": int(diagnostics[-1]["iteration"]), "time": float(diagnostics[-1]["time"]),
        "average_dt": float(2.0 / diagnostics[-1]["iteration"]),
        "cfl": configuration["time"]["cfl"], "dt_cap": configuration["time"]["maximumTimeStep"],
        "mpi_ranks": len(list(directory.glob("solution_00000000.nodes.rank*.csv"))),
        "volume": volume, "density_initial": errors[0], "density_final": errors[1],
        "conservation_scaled_drift": drift, "periodic_state_difference": periodic_difference,
        "entropy_L1": float(diagnostics[-1]["entropy_L1"]),
        "min_density_mean": float(final["rho_mean"].min()),
        "min_pressure_of_mean": float(pressure_of_mean.min()),
        "source_directory": str(directory.relative_to(ROOT)),
        "mesh_arrays_sha256_verified": True,
    }
    if log_path and log_path.exists():
        log = log_path.read_text()
        for field in ("volume", "internal-surface", "boundary-surface"):
            if f"stored {field} quadrature=0" not in log:
                raise ValueError("High-efficiency zero-quadrature invariant was not logged")
        metrics["stored_gauss_points"] = 0
        metrics["closure_max"] = float(re.search(r"max closure=([\deE.+-]+)", log).group(1))
        wall = re.search(r"^real ([\d.]+)$", log, re.MULTILINE)
        metrics["wall_seconds"] = float(wall.group(1)) if wall else None
    return metrics, final, cells, configuration


def plot_results(runs, output):
    plt.rcParams.update({"text.usetex": False, "font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    h = np.array([run[0]["h_3d"] for run in runs])
    for norm, marker in zip(NORMS, ("o", "s", "^")):
        axes[0].loglog(h, [run[0]["density_final"][norm] for run in runs], marker + "-", label=norm)
    l2 = runs[-1][0]["density_final"]["L2"]
    for order, style in ((2, "--"), (3, ":")):
        axes[0].loglog(h, l2 * (h / h[-1])**order, style, color="gray", label=f"slope {order}")
    axes[0].set(xlabel="Effective spacing h = (V/N_periodic)^(1/3)", ylabel="Density error", title="Final error at t=2")
    axes[0].legend(fontsize=8)
    axes[1].loglog(h, [run[0]["density_initial"]["L2"] for run in runs], "o-", label="initial recovered-point L2")
    init = runs[-1][0]["density_initial"]["L2"]
    axes[1].loglog(h, init * (h / h[-1])**3, ":", color="gray", label="slope 3")
    axes[1].set(xlabel="Effective spacing h", ylabel="Density error", title="Initialization / point recovery")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(which="both", alpha=0.2)
        ax.set_xticks(h, labels=[f"{value:.3f}" for value in h])
        ax.xaxis.set_minor_formatter(NullFormatter())
    fig.savefig(output / "convergence.png", dpi=190)
    fig.savefig(output / "convergence.pdf")
    plt.close(fig)
    fig, axes = plt.subplots(2, 4, figsize=(13.3, 6.6), constrained_layout=True)
    for column, (metrics, data, cells, _) in enumerate(runs):
        cut = np.flatnonzero(np.isclose(data["z"], 2.0, rtol=0, atol=1e-9))
        node_map = np.full(len(data), -1, dtype=int)
        node_map[cut] = np.arange(len(cut))
        faces = node_map[cells[:, :3]]
        faces = faces[np.all(faces >= 0, axis=1)]
        tri = mtri.Triangulation(data["x"][cut], data["y"][cut], faces)
        rho = data["rho_point"][cut]
        error = rho - data["rho_exact"][cut]
        density_plot = axes[0, column].tripcolor(tri, rho, shading="gouraud", cmap="viridis", vmin=0.45, vmax=1.05, rasterized=True)
        error_limit = max(float(np.max(np.abs(error))), 1e-12)
        error_plot = axes[1, column].tripcolor(tri, error, shading="gouraud", cmap="RdBu_r", vmin=-error_limit, vmax=error_limit, rasterized=True)
        axes[0, column].set_title(f"{metrics['mesh']}: density")
        axes[1, column].set_title(f"Error (scale +/-{error_limit:.2e})", fontsize=9)
        fig.colorbar(error_plot, ax=axes[1, column], shrink=0.75, format="%.1e")
        for row in (0, 1):
            axes[row, column].set(xlim=(0, 10), ylim=(0, 10), aspect="equal", xlabel="x", ylabel="y")
    fig.colorbar(density_plot, ax=axes[0, :].tolist(), shrink=0.75)
    fig.suptitle("EfficientDifferential: recovered nodal density at t=2, z=2")
    fig.savefig(output / "density_grid_comparison.png", dpi=190)
    fig.savefig(output / "density_grid_comparison.pdf")
    plt.close(fig)


def write_report_tables(result, output):
    def number(value):
        return f"{value:.6e}"

    def rows(name, content):
        (output / name).write_text("\n".join(" & ".join(row) + r" \\" for row in content) + "\n")

    runs = result["runs"]
    rows("mesh_rows.tex", [[r["mesh"], f"{r['nodes']:,}", f"{r['cells']:,}",
         f"{r['periodic_unknowns']:,}", f"{r['h_3d']:.6f}", str(r["steps"]), str(r["mpi_ranks"])] for r in runs])
    rows("error_rows.tex", [[r["mesh"]] + [item for norm in NORMS for item in
         [number(r["density_final"][norm]), f"{r['order_3d'][norm]:.3f}" if norm in r["order_3d"] else "--"]] for r in runs])
    rows("check_rows.tex", [[r["mesh"], number(r["density_initial"]["L2"]),
         number(abs(r["conservation_scaled_drift"]["mass"])), number(abs(r["conservation_scaled_drift"]["total_energy"])),
         number(r["periodic_state_difference"]), number(r["closure_max"])] for r in runs])
    rows("time_rows.tex", [[r["mesh"], number(r["baseline_L2"]), number(r["half_dt_L2"]),
         number(r["solution_difference"]["L2"]), number(100 * r["difference_over_baseline_L2"])] for r in result["time_step_checks"]])
    order = runs[-1]["order_3d"]["L2"]
    interpretation = ("最细两级的密度 $L_2$ 误差呈现接近三阶的收敛趋势；粗网格仍未进入这一收敛区间。"
                      if 2.8 <= order <= 3.2 else
                      "最细两级的实测收敛阶尚不能直接认定为稳定三阶；应以表中实测值为准。")
    macros = {"FineOrderLtwo": f"{order:.3f}", "FineOrderNominal": f"{runs[-1]['order_nominal']['L2']:.3f}",
              "FineErrorLtwo": number(runs[-1]["density_final"]["L2"]),
              "FineReduction": f"{runs[0]['density_final']['L2'] / runs[-1]['density_final']['L2']:.2f}",
              "FineInterpretation": interpretation}
    (output / "summary.tex").write_text("\n".join("\\newcommand{\\" + key + "}{" + value + "}" for key, value in macros.items()) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/reports/vertexfv_convergence")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    runs = []
    for n in (10, 20, 40, 80):
        directory = ROOT / ("data/out/vertexFV/iv40_efficient" if n == 40 else f"data/out/vertexFV/convergence/iv{n}")
        log = directory / "run.log"
        runs.append(audit_run(n, directory, log))
    for run in runs[1:]:
        for section in ("physics", "algorithm", "reconstruction", "initialField", "time"):
            if run[3][section] != runs[0][3][section]:
                raise ValueError(f"Inconsistent configuration section {section}")
    for i, (metrics, _, _, _) in enumerate(runs):
        metrics["order_3d"] = {}
        metrics["order_nominal"] = {}
        if i:
            previous = runs[i - 1][0]
            for norm in NORMS:
                ratio = previous["density_final"][norm] / metrics["density_final"][norm]
                metrics["order_3d"][norm] = float(np.log(ratio) / np.log(previous["h_3d"] / metrics["h_3d"]))
                metrics["order_nominal"][norm] = float(np.log(ratio) / np.log(previous["h_nominal"] / metrics["h_nominal"]))
    time_tests = []
    for n in (20, 40):
        reference = next(run for run in runs if run[0]["nominal_resolution"] == n)
        directory = ROOT / f"data/out/vertexFV/convergence/iv{n}_halfdt"
        metrics, data, _, half_configuration = audit_run(n, directory, directory / "run.log")
        for section in ("mesh", "physics", "algorithm", "reconstruction", "initialField"):
            if half_configuration[section] != reference[3][section]:
                raise ValueError(f"Time sensitivity test changed section {section}")
        if (metrics["cfl"] != reference[0]["cfl"] / 2
                or metrics["dt_cap"] != reference[0]["dt_cap"] / 2):
            raise ValueError("Time sensitivity test did not halve CFL and time-step cap")
        difference = norms(data["rho_point"] - reference[1]["rho_point"], data["partial_volume"])
        time_tests.append({"mesh": f"iv{n}", "baseline_L2": reference[0]["density_final"]["L2"],
                           "half_dt_L2": metrics["density_final"]["L2"], "solution_difference": difference,
                           "difference_over_baseline_L2": difference["L2"] / reference[0]["density_final"]["L2"],
                           "half_dt_steps": metrics["steps"], "cfl": metrics["cfl"], "dt_cap": metrics["dt_cap"]})
    result = {"algorithm": "EfficientDifferential", "time": 2.0, "period": 10.0,
              "error_definition": "partial-dual-volume weighted recovered nodal density minus analytic nodal density",
              "order_spacing": "h=(400/N_periodic)^(1/3)", "runs": [run[0] for run in runs],
              "time_step_checks": time_tests,
              "note": "Observed order is measured, not imposed; original thesis error values are not acceptance thresholds."}
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    with (args.output / "convergence.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("mesh", "nodes", "cells", "periodic_unknowns", "h_3d", "steps", "time", "L1", "L2", "Linf", "p_L1", "p_L2", "p_Linf", "initial_L2"))
        for metrics, _, _, _ in runs:
            writer.writerow([metrics[k] for k in ("mesh", "nodes", "cells", "periodic_unknowns", "h_3d", "steps", "time")] +
                            [metrics["density_final"][k] for k in NORMS] +
                            [metrics["order_3d"].get(k, "") for k in NORMS] + [metrics["density_initial"]["L2"]])
    plot_results(runs, args.output)
    write_report_tables(result, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
