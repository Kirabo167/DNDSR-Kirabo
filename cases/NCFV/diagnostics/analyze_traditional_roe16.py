"""Analyze the Roe/QR16 traditional NCFV static and t=2 accuracy runs.

The expected input layout is::

    data/out/NCFV/traditional_roe16_surface3_20260917/
      static/{iv10_np1,iv10_np2,iv20_np4,iv40_np8,iv80_np16,...}
      transient/{iv10_np1,iv10_np2,iv20_np4,iv40_np8,iv80_np64}

The script is deliberately separate from the historical Roe_M2 analyzers.  It
only reads completed probe output and writes CSV/JSON/Markdown summaries to the
directory selected with ``--output``.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA = ROOT / "data/out/NCFV/traditional_roe16_surface3_20260917"
NORMS = ("L1", "L2", "Linf")
COMPONENTS = ("rho", "rhou", "rhov", "rhow", "rhoE")
CONSERVATION_FIELDS = ("mass", "momentum_x", "momentum_y", "momentum_z", "total_energy")

STATIC_RUNS = {
    10: ("iv10_np1", 1),
    20: ("iv20_np4", 4),
    40: ("iv40_np8", 8),
    80: ("iv80_np16", 16),
}
TRANSIENT_RUNS = {
    10: ("iv10_np1", 1),
    20: ("iv20_np4", 4),
    40: ("iv40_np8", 8),
    80: ("iv80_np64", 64),
}
HIGHER_SURFACE_RUNS = {
    5: {
        40: "iv40_np8_sq5",
        80: "iv80_np16_sq5",
    },
    7: {
        40: "iv40_np8_sq7",
        80: "iv80_np16_sq7",
    },
}
SURFACE_POINTS = {3: 3, 5: 6, 7: 12}

# label, section in metrics.json, key template
STATIC_QUANTITIES = (
    ("jump_quadrature", "quadrature_norms", "jump.{component}"),
    ("jump_face", "face_mean_norms", "jump.{component}"),
    ("roe_quadrature", "quadrature_norms", "Roe.Fnum.{component}"),
    ("roe_face", "face_mean_norms", "Roe.Fnum.{component}"),
    ("central_rhs", "rhs_norms", "central.error.{component}"),
    ("roe_rhs", "rhs_norms", "Roe.dissipation.{component}"),
    ("total_rhs", "rhs_norms", "total.error.{component}"),
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def close(actual: float, expected: float, message: str, *, rtol: float = 5e-10,
          atol: float = 5e-13) -> None:
    require(math.isclose(float(actual), float(expected), rel_tol=rtol, abs_tol=atol),
            f"{message}: got {actual!r}, expected {expected!r}")


def read_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"Missing completed result: {path}")
    return json.loads(path.read_text())


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_configuration(configuration: dict[str, Any], surface_order: int) -> None:
    algorithm = configuration["algorithm"]
    physics = configuration["physics"]
    reconstruction = configuration["reconstruction"]
    require(configuration["dimension"] == 3, "Expected a three-dimensional run")
    require(algorithm["mode"] == "TraditionalQuadrature", "Run is not TraditionalQuadrature")
    require(algorithm["quadratureOrder"] == 4, "Volume quadrature order is not q4")
    require(algorithm["surfaceQuadratureOrder"] == surface_order,
            f"Surface quadrature order is not q{surface_order}")
    require(physics["riemannSolver"] == "Roe", "Run did not use the standard Roe solver")
    require(not physics["viscous"]["enabled"], "Viscous terms are enabled")
    require(not reconstruction["enableLimiter"], "Limiter is enabled in the smooth-vortex audit")


def comparable_configuration(configuration: dict[str, Any], *, keep_mesh_file: bool = False) -> dict[str, Any]:
    """Return only settings that must remain common across grids and MPI sizes."""
    result = {
        key: copy.deepcopy(configuration[key])
        for key in ("dimension", "mesh", "algorithm", "reconstruction", "physics", "initialField")
    }
    if not keep_mesh_file:
        result["mesh"].pop("meshFile", None)
    return result


def comparable_transient_configuration(configuration: dict[str, Any], *,
                                       keep_mesh_file: bool = False) -> dict[str, Any]:
    result = comparable_configuration(configuration, keep_mesh_file=keep_mesh_file)
    result["time"] = copy.deepcopy(configuration["time"])
    return result


def validate_stencil(record: dict[str, Any], context: str) -> None:
    require(record["stencil_size_min"] == 16, f"{context}: minimum stencil size is not 16")
    close(record["stencil_size_mean"], 16.0, f"{context}: mean stencil size is not 16",
          rtol=0.0, atol=1e-12)
    require(record["stencil_size_max"] == 16, f"{context}: maximum stencil size is not 16")


def observed_order(coarse_h: float, fine_h: float, coarse_error: float,
                   fine_error: float) -> float | None:
    if not (coarse_h > fine_h > 0.0 and coarse_error > 1e-14 and fine_error > 1e-14):
        return None
    return float(math.log(coarse_error / fine_error) / math.log(coarse_h / fine_h))


def least_squares_order(runs: list[dict[str, Any]], values: list[float],
                        finest_count: int | None = None) -> dict[str, Any]:
    selected_runs = runs[-finest_count:] if finest_count is not None else runs
    selected_values = values[-finest_count:] if finest_count is not None else values
    if len(selected_runs) < 2 or any(value <= 1e-14 for value in selected_values):
        return {"points": len(selected_runs), "order": None, "r_squared": None}
    x = np.log(np.asarray([run["h"] for run in selected_runs], dtype=float))
    y = np.log(np.asarray(selected_values, dtype=float))
    slope, intercept = np.polyfit(x, y, 1)
    residual = y - (slope * x + intercept)
    total = y - np.mean(y)
    sum_total = float(np.dot(total, total))
    r_squared = 1.0 if sum_total == 0.0 else 1.0 - float(np.dot(residual, residual)) / sum_total
    return {"points": len(selected_runs), "order": float(slope), "r_squared": r_squared}


def static_l2(metrics: dict[str, Any]) -> dict[str, dict[str, float]]:
    values: dict[str, dict[str, float]] = {}
    for label, section, key_template in STATIC_QUANTITIES:
        values[label] = {}
        for component in COMPONENTS:
            key = key_template.format(component=component)
            require(key in metrics[section], f"Missing static metric {section}.{key}")
            item = metrics[section][key]
            require(all(norm in item for norm in NORMS), f"Incomplete norms for {section}.{key}")
            require(float(item["weight"]) > 0.0, f"Nonpositive norm weight for {section}.{key}")
            value = float(item["L2"])
            require(math.isfinite(value) and value >= 0.0, f"Invalid L2 value for {section}.{key}")
            values[label][component] = value
    return values


def audit_static(directory: Path, expected_ranks: int, surface_order: int) -> dict[str, Any]:
    path = directory / "metrics.json"
    metrics = read_json(path)
    validate_configuration(metrics["configuration"], surface_order)
    require(metrics["mpi_ranks"] == expected_ranks,
            f"{directory.name}: expected {expected_ranks} MPI ranks")
    require(metrics["time"] == 0 and metrics["time_steps"] == 0,
            f"{directory.name}: static state was advanced")
    require(metrics["reconstruction_calls"] == 1 and metrics["rhs_evaluations"] == 1,
            f"{directory.name}: expected exactly one reconstruction and one RHS evaluation")
    require(metrics["coefficient_rows_min"] == 9 and metrics["coefficient_rows_max"] == 9,
            f"{directory.name}: full nine-row quadratic coefficients were not retained")
    validate_stencil(metrics, directory.name)
    require(metrics["surface_rho_min"] > 0.0 and metrics["surface_pressure_min"] > 0.0,
            f"{directory.name}: nonphysical reconstructed interface state")
    require(metrics["volume_quadrature_points"] > 0 and metrics["surface_quadrature_points"] > 0,
            f"{directory.name}: quadrature points were not constructed")
    require(metrics["points_per_micro_tetrahedron"] == 14,
            f"{directory.name}: q4 volume rule is not the expected 14-point tetrahedron rule")
    require(surface_order in SURFACE_POINTS, f"Unsupported surface-order audit q{surface_order}")
    require(metrics["points_per_micro_triangle"] == SURFACE_POINTS[surface_order],
            f"{directory.name}: surface q{surface_order} should use "
            f"{SURFACE_POINTS[surface_order]} points per micro triangle; this may be stale pre-11:30 data")
    close(metrics["volume"], 400.0, f"{directory.name}: unexpected domain volume",
          rtol=0.0, atol=2e-9)
    for key in ("M2_formula_max_mismatch", "face_area_relative_mismatch",
                "unit_normal_max_mismatch", "rhs_decomposition_max_mismatch"):
        require(float(metrics[key]) < 1e-10, f"{directory.name}: failed identity check {key}")
    require(max(abs(float(value)) for value in metrics["global_rhs_parts_integral"]) < 1e-9,
            f"{directory.name}: assembled RHS is not globally conservative")
    edge_files = sorted(directory.glob("edges.rank*.csv"))
    require(len(edge_files) == expected_ranks, f"{directory.name}: missing per-rank edge files")
    result = {
        "directory": str(directory),
        "mpi_ranks": int(metrics["mpi_ranks"]),
        "nodes": int(metrics["nodes"]),
        "cells": int(metrics["cells"]),
        "edges": int(metrics["edges"]),
        "periodic_unknowns": int(metrics["periodic_unknowns"]),
        "h": float(metrics["h_3d"]),
        "volume": float(metrics["volume"]),
        "coefficient_rows": [int(metrics["coefficient_rows_min"]),
                             int(metrics["coefficient_rows_max"])],
        "stencil_size": [int(metrics["stencil_size_min"]),
                         float(metrics["stencil_size_mean"]),
                         int(metrics["stencil_size_max"])],
        "volume_quadrature_points": int(metrics["volume_quadrature_points"]),
        "surface_quadrature_points": int(metrics["surface_quadrature_points"]),
        "boundary_quadrature_points": int(metrics["boundary_quadrature_points"]),
        "points_per_micro_tetrahedron": int(metrics["points_per_micro_tetrahedron"]),
        "points_per_micro_triangle": int(metrics["points_per_micro_triangle"]),
        "surface_rho_min": float(metrics["surface_rho_min"]),
        "surface_pressure_min": float(metrics["surface_pressure_min"]),
        "l2": static_l2(metrics),
        "configuration": metrics["configuration"],
        "input_sha256": file_sha256(path),
    }
    return result


def load_snapshot(directory: Path, label: str, ranks: int, expected_nodes: int,
                  gamma: float, expected_errors: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    files = sorted(directory.glob(f"points_{label}.rank*.csv"))
    require(len(files) == ranks, f"{directory.name}: incomplete {label} point snapshots")
    arrays = [np.atleast_1d(np.genfromtxt(path, delimiter=",", names=True)) for path in files]
    data = np.concatenate(arrays)
    require(data.dtype.names is not None, f"{directory.name}: snapshot has no named columns")
    required = {
        "original_node", "x", "y", "z", "partial_volume", "mean_rho", "mean_rhou",
        "mean_rhov", "mean_rhow", "mean_rhoE", "point_rho", "point_rhou", "point_rhov",
        "point_rhow", "point_rhoE", "rho_exact", "p_point", "p_exact",
    }
    require(required.issubset(data.dtype.names), f"{directory.name}: snapshot columns are incomplete")
    data = np.sort(data, order="original_node")
    node_ids = data["original_node"].astype(np.int64)
    require(len(data) == expected_nodes and np.array_equal(node_ids, np.arange(expected_nodes)),
            f"{directory.name}: duplicate or missing original node IDs")
    numeric = np.column_stack([data[name] for name in data.dtype.names])
    require(np.isfinite(numeric).all(), f"{directory.name}: non-finite snapshot value")
    require(np.min(data["partial_volume"]) > 0.0, f"{directory.name}: nonpositive dual volume")

    rho = data["point_rho"]
    point_momentum_squared = sum(data[name] ** 2 for name in ("point_rhou", "point_rhov", "point_rhow"))
    pressure = (gamma - 1.0) * (data["point_rhoE"] - point_momentum_squared / (2.0 * rho))
    mean_rho = data["mean_rho"]
    mean_momentum_squared = sum(data[name] ** 2 for name in ("mean_rhou", "mean_rhov", "mean_rhow"))
    pressure_of_mean = (gamma - 1.0) * (data["mean_rhoE"] - mean_momentum_squared / (2.0 * mean_rho))
    require(np.min(rho) > 0.0 and np.min(pressure) > 0.0,
            f"{directory.name}: nonphysical recovered point state")
    require(np.min(mean_rho) > 0.0 and np.min(pressure_of_mean) > 0.0,
            f"{directory.name}: nonphysical conservative mean state")
    require(np.allclose(pressure, data["p_point"], rtol=2e-12, atol=2e-14),
            f"{directory.name}: saved recovered pressure is inconsistent")

    volume = data["partial_volume"]
    total_volume = float(np.sum(volume))
    close(total_volume, 400.0, f"{directory.name}: snapshot volume", rtol=0.0, atol=2e-9)

    def norms(error: np.ndarray) -> dict[str, float]:
        return {
            "L1": float(np.dot(volume, np.abs(error)) / total_volume),
            "L2": float(np.sqrt(np.dot(volume, error * error) / total_volume)),
            "Linf": float(np.max(np.abs(error))),
        }

    errors = {
        "rho": norms(rho - data["rho_exact"]),
        "pressure": norms(pressure - data["p_exact"]),
    }
    expected_keys = {"rho": "rho_point", "pressure": "pressure_point"}
    for field, key in expected_keys.items():
        for norm in NORMS:
            close(errors[field][norm], expected_errors[key][norm],
                  f"{directory.name}: independent {label} {field} {norm}", rtol=3e-11,
                  atol=2e-14)
    summary = {
        "volume": total_volume,
        "errors": errors,
        "min_density_mean": float(np.min(mean_rho)),
        "min_pressure_of_mean": float(np.min(pressure_of_mean)),
        "min_density_point": float(np.min(rho)),
        "min_pressure_point": float(np.min(pressure)),
        "input_sha256": {path.name: file_sha256(path) for path in files},
    }
    return data, summary


def load_diagnostics(directory: Path, accuracy: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    path = directory / "solution.diagnostics.csv"
    require(path.is_file(), f"Missing conservation diagnostics: {path}")
    diagnostics = np.atleast_1d(np.genfromtxt(path, delimiter=",", names=True))
    require(diagnostics.dtype.names is not None, f"{directory.name}: diagnostics has no header")
    required = {"iteration", "time", *CONSERVATION_FIELDS}
    require(required.issubset(diagnostics.dtype.names),
            f"{directory.name}: conservation diagnostic columns are incomplete")
    require(len(diagnostics) >= 2, f"{directory.name}: incomplete conservation history")
    endpoints = np.column_stack([diagnostics[[0, -1]][field]
                                 for field in ("iteration", "time", *CONSERVATION_FIELDS)])
    require(np.isfinite(endpoints).all(), f"{directory.name}: non-finite conservation diagnostic")
    require(abs(float(diagnostics[0]["time"])) < 1e-14,
            f"{directory.name}: conservation history does not start at t=0")
    close(diagnostics[-1]["time"], accuracy["final"]["time"],
          f"{directory.name}: final diagnostic time", rtol=0.0, atol=1e-12)
    require(int(diagnostics[-1]["iteration"]) == int(accuracy["final"]["iteration"]),
            f"{directory.name}: final diagnostic iteration differs from accuracy.json")
    absolute = {}
    scaled = {}
    initial = {}
    final = {}
    for field in CONSERVATION_FIELDS:
        initial[field] = float(diagnostics[0][field])
        final[field] = float(diagnostics[-1][field])
        absolute[field] = final[field] - initial[field]
        scaled[field] = absolute[field] / max(abs(initial[field]), 1.0)
    return diagnostics, {
        "initial": initial,
        "final": final,
        "absolute_drift": absolute,
        "scaled_drift": scaled,
        "max_abs_scaled_drift": max(abs(value) for value in scaled.values()),
        "input_sha256": file_sha256(path),
    }


def audit_transient(directory: Path, expected_ranks: int, expected_nodes: int) -> tuple[dict[str, Any],
                                                                                       dict[str, np.ndarray]]:
    path = directory / "accuracy.json"
    accuracy = read_json(path)
    validate_configuration(accuracy["configuration"], 3)
    require(accuracy["mpi_ranks"] == expected_ranks,
            f"{directory.name}: expected {expected_ranks} MPI ranks")
    require(abs(float(accuracy["initial"]["time"])) < 1e-14,
            f"{directory.name}: transient run does not start at t=0")
    close(accuracy["final"]["time"], 2.0, f"{directory.name}: final time",
          rtol=0.0, atol=1e-12)
    require(int(accuracy["final"]["iteration"]) > 0,
            f"{directory.name}: transient run has no time steps")
    require(accuracy.get("normalization") == "dual-bounds-half-span (thesis 3-34)",
            f"{directory.name}: unexpected reconstruction normalization")
    validate_stencil(accuracy["initial"], f"{directory.name} initial")
    validate_stencil(accuracy["final"], f"{directory.name} final")
    require(accuracy["initial"]["gauss_points_stored"] > 0 and
            accuracy["initial"]["gauss_points_stored"] == accuracy["final"]["gauss_points_stored"],
            f"{directory.name}: traditional quadrature storage is absent or changed")
    configuration = accuracy["configuration"]
    time = configuration["time"]
    require(time["useCFLTimeStep"] and not time["useLocalTimeStep"],
            f"{directory.name}: expected a global physical CFL step")
    require(time["cfl"] == 0.5, f"{directory.name}: expected CFL=0.5")
    close(time["endTime"], 2.0, f"{directory.name}: configured end time",
          rtol=0.0, atol=1e-14)
    require(time["maximumTimeStep"] >= 1e20 and time["minimumTimeStep"] <= 1e-20,
            f"{directory.name}: CFL step is unexpectedly capped")

    snapshots: dict[str, np.ndarray] = {}
    snapshot_summaries = {}
    for label in ("initial", "final"):
        snapshots[label], snapshot_summaries[label] = load_snapshot(
            directory, label, expected_ranks, expected_nodes, configuration["physics"]["gamma"],
            accuracy[label])
    _, conservation = load_diagnostics(directory, accuracy)
    result = {
        "directory": str(directory),
        "mpi_ranks": int(accuracy["mpi_ranks"]),
        "steps": int(accuracy["final"]["iteration"]),
        "time": float(accuracy["final"]["time"]),
        "cfl": float(time["cfl"]),
        "initial_cfl_step": float(accuracy["initial_cfl_step"]),
        "wall_seconds": float(accuracy["wall_seconds"]),
        "stencil_size": [int(accuracy["final"]["stencil_size_min"]),
                         float(accuracy["final"]["stencil_size_mean"]),
                         int(accuracy["final"]["stencil_size_max"])],
        "gauss_points_stored": int(accuracy["final"]["gauss_points_stored"]),
        "initial": snapshot_summaries["initial"],
        "final": snapshot_summaries["final"],
        "conservation": conservation,
        "configuration": configuration,
        "input_sha256": file_sha256(path),
    }
    return result, snapshots


def write_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    require(rows or fieldnames is not None, f"Empty CSV {path.name} needs explicit field names")
    names = fieldnames or list(rows[0])
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=names, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def compare_iv10_nodes(np1: dict[str, np.ndarray], np2: dict[str, np.ndarray], output: Path) -> dict[str, Any]:
    rows = []
    summary: dict[str, Any] = {}
    maximum_state_difference = 0.0
    maximum_all_field_difference = 0.0
    maximum_node_error_difference = 0.0
    maximum_error_norm_difference = 0.0
    for label in ("initial", "final"):
        first = np1[label]
        second = np2[label]
        require(first.dtype.names == second.dtype.names, f"iv10 {label}: snapshot schemas differ")
        require(np.array_equal(first["original_node"], second["original_node"]),
                f"iv10 {label}: np1/np2 original-node ordering differs")
        summary[label] = {"fields": {}, "error_norms": {}}
        fields = [name for name in first.dtype.names if name != "original_node"]
        first_errors = {
            "rho": first["point_rho"] - first["rho_exact"],
            "pressure": first["p_point"] - first["p_exact"],
        }
        second_errors = {
            "rho": second["point_rho"] - second["rho_exact"],
            "pressure": second["p_point"] - second["p_exact"],
        }
        for index in range(len(first)):
            row: dict[str, Any] = {
                "snapshot": label,
                "original_node": int(first["original_node"][index]),
            }
            for field in fields:
                row[f"delta_{field}"] = float(second[field][index] - first[field][index])
            row["delta_rho_error"] = float(second_errors["rho"][index] - first_errors["rho"][index])
            row["delta_pressure_error"] = float(
                second_errors["pressure"][index] - first_errors["pressure"][index])
            rows.append(row)
        for field in fields:
            difference = second[field] - first[field]
            values = {
                "max_abs": float(np.max(np.abs(difference))),
                "mean_abs": float(np.mean(np.abs(difference))),
                "rms": float(np.sqrt(np.mean(difference * difference))),
            }
            summary[label]["fields"][field] = values
            maximum_all_field_difference = max(maximum_all_field_difference, values["max_abs"])
            if field.startswith(("mean_", "point_")) or field == "p_point":
                maximum_state_difference = max(maximum_state_difference, values["max_abs"])

        for quantity in ("rho", "pressure"):
            error_difference = second_errors[quantity] - first_errors[quantity]
            maximum_node_error_difference = max(
                maximum_node_error_difference, float(np.max(np.abs(error_difference))))
            norm_results = {}
            for name, data, error in (("np1", first, first_errors[quantity]),
                                      ("np2", second, second_errors[quantity])):
                volume = data["partial_volume"]
                total_volume = float(np.sum(volume))
                norm_results[name] = {
                    "L1": float(np.dot(volume, np.abs(error)) / total_volume),
                    "L2": float(np.sqrt(np.dot(volume, error * error) / total_volume)),
                    "Linf": float(np.max(np.abs(error))),
                }
            norm_results["absolute_difference"] = {
                norm: abs(norm_results["np2"][norm] - norm_results["np1"][norm])
                for norm in NORMS
            }
            maximum_error_norm_difference = max(
                maximum_error_norm_difference,
                max(norm_results["absolute_difference"].values()),
            )
            summary[label]["error_norms"][quantity] = norm_results
    write_rows(output / "iv10_np1_np2_node_differences.csv", rows)
    tolerance = 1e-10
    require(maximum_all_field_difference < tolerance,
            f"iv10 np1/np2 node fields differ by {maximum_all_field_difference:.3e}")
    require(maximum_node_error_difference < tolerance,
            f"iv10 np1/np2 point errors differ by {maximum_node_error_difference:.3e}")
    require(maximum_error_norm_difference < tolerance,
            f"iv10 np1/np2 error norms differ by {maximum_error_norm_difference:.3e}")
    return {
        "passed": True,
        "tolerance": tolerance,
        "max_abs_state_difference": maximum_state_difference,
        "max_abs_all_field_difference": maximum_all_field_difference,
        "max_abs_node_error_difference": maximum_node_error_difference,
        "max_abs_error_norm_difference": maximum_error_norm_difference,
        "by_snapshot_and_field": summary,
    }


def format_value(value: float | None, digits: int = 4) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def markdown_static_table(runs: list[dict[str, Any]]) -> str:
    text = ("| 网格 | h | 跳量积分点 L2 | 阶 | Roe 面平均 L2 | 阶 | 中心 RHS L2 | 阶 | "
            "Roe 耗散 RHS L2 | 阶 | 总 RHS L2 | 阶 |\n"
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    labels = ("jump_quadrature", "roe_face", "central_rhs", "roe_rhs", "total_rhs")
    for run in runs:
        cells = [run["mesh"], f'{run["h"]:.8e}']
        for label in labels:
            cells.extend((f'{run["l2"][label]["rho"]:.8e}',
                          format_value(run["orders"][label]["rho"])))
        text += "| " + " | ".join(cells) + " |\n"
    return text


def markdown_transient_table(runs: list[dict[str, Any]], field: str) -> str:
    text = "| 网格 | h | MPI | 步数 | L1 | 阶 | L2 | 阶 | L∞ | 阶 |\n"
    text += "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    for run in runs:
        cells = [run["mesh"], f'{run["h"]:.8e}', str(run["mpi_ranks"]), str(run["steps"])]
        for norm in NORMS:
            cells.extend((f'{run["final"]["errors"][field][norm]:.8e}',
                          format_value(run["orders"][field][norm])))
        text += "| " + " | ".join(cells) + " |\n"
    return text


def markdown_fit_table(fits: dict[str, dict[str, dict[str, Any]]],
                       labels: list[tuple[str, str]]) -> str:
    text = "| 量 | 全网格拟合阶 | R² | 最细三网格拟合阶 | R² |\n"
    text += "|---|---:|---:|---:|---:|\n"
    for key, display in labels:
        all_fit = fits[key]["all_grids"]
        fine_fit = fits[key]["finest_three"]
        text += (f"| {display} | {format_value(all_fit['order'])} | "
                 f"{format_value(all_fit['r_squared'], 5)} | "
                 f"{format_value(fine_fit['order'])} | "
                 f"{format_value(fine_fit['r_squared'], 5)} |\n")
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA,
                        help=f"Input root (default: {DEFAULT_DATA})")
    parser.add_argument("--output", type=Path, required=True,
                        help="New or existing directory for CSV, metrics.json, and report.md")
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="Analyze completed transient grids while later grids are still running")
    args = parser.parse_args()
    data_root = args.data.resolve()
    output = args.output.resolve()
    static_root = data_root / "static"
    transient_root = data_root / "transient"
    require(static_root.is_dir() and transient_root.is_dir(),
            f"Expected static/ and transient/ below {data_root}")
    output.mkdir(parents=True, exist_ok=True)

    static_runs = []
    static_by_n: dict[int, dict[str, Any]] = {}
    for resolution, (name, ranks) in STATIC_RUNS.items():
        run = audit_static(static_root / name, ranks, 3)
        run.update(mesh=f"iv{resolution}", nominal_resolution=resolution, surface_order=3)
        static_runs.append(run)
        static_by_n[resolution] = run

    common_static = comparable_configuration(static_runs[0]["configuration"])
    for run in static_runs[1:]:
        require(comparable_configuration(run["configuration"]) == common_static,
                f"{run['mesh']}: static numerical settings differ from iv10")

    for index, run in enumerate(static_runs):
        run["orders"] = {label: {} for label, _, _ in STATIC_QUANTITIES}
        for label, _, _ in STATIC_QUANTITIES:
            for component in COMPONENTS:
                run["orders"][label][component] = (
                    observed_order(static_runs[index - 1]["h"], run["h"],
                                   static_runs[index - 1]["l2"][label][component],
                                   run["l2"][label][component])
                    if index else None
                )

    static_fits: dict[str, dict[str, dict[str, Any]]] = {}
    fit_rows = []
    for label, _, _ in STATIC_QUANTITIES:
        static_fits[label] = {}
        for component in COMPONENTS:
            values = [run["l2"][label][component] for run in static_runs]
            fits = {
                "all_grids": least_squares_order(static_runs, values),
                "finest_three": least_squares_order(static_runs, values, 3),
            }
            static_fits[label][component] = fits
            for subset, fit in fits.items():
                fit_rows.append({
                    "category": "static_L2",
                    "quantity": label,
                    "component_or_norm": component,
                    "grid_subset": subset,
                    **fit,
                })

    static_rows = []
    for run in static_runs:
        for component in COMPONENTS:
            row: dict[str, Any] = {
                "mesh": run["mesh"],
                "mpi_ranks": run["mpi_ranks"],
                "h": run["h"],
                "component": component,
            }
            for label, _, _ in STATIC_QUANTITIES:
                row[f"{label}_L2"] = run["l2"][label][component]
                row[f"{label}_order"] = run["orders"][label][component]
            static_rows.append(row)
    write_rows(output / "static_convergence.csv", static_rows)

    # The static iv10 MPI comparison supplements the requested per-node transient comparison.
    static_np2 = audit_static(static_root / "iv10_np2", 2, 3)
    require(comparable_configuration(static_np2["configuration"], keep_mesh_file=True) ==
            comparable_configuration(static_by_n[10]["configuration"], keep_mesh_file=True),
            "iv10 np2 static configuration differs from np1")
    static_mpi_max = 0.0
    for label, _, _ in STATIC_QUANTITIES:
        for component in COMPONENTS:
            static_mpi_max = max(
                static_mpi_max,
                abs(static_by_n[10]["l2"][label][component] - static_np2["l2"][label][component]),
            )
    require(static_mpi_max < 1e-10,
            f"iv10 np1/np2 static L2 metrics differ by {static_mpi_max:.3e}")

    higher_surface_runs = []
    surface_rows = []
    missing_surface_checks = []
    for surface_order, checks in HIGHER_SURFACE_RUNS.items():
        for resolution, name in checks.items():
            directory = static_root / name
            if not (directory / "metrics.json").is_file():
                # q5 is the required control; q7 is an optional extra check.
                if surface_order == 5:
                    missing_surface_checks.append(name)
                continue
            baseline = static_by_n[resolution]
            higher = audit_static(directory, baseline["mpi_ranks"], surface_order)
            higher.update(mesh=f"iv{resolution}", nominal_resolution=resolution,
                          surface_order=surface_order)
            candidate = comparable_configuration(higher["configuration"], keep_mesh_file=True)
            candidate["algorithm"]["surfaceQuadratureOrder"] = 3
            require(candidate == comparable_configuration(baseline["configuration"],
                                                          keep_mesh_file=True),
                    f"iv{resolution}: q{surface_order} check changed settings other than surface quadrature")
            for label, _, _ in STATIC_QUANTITIES:
                for component in COMPONENTS:
                    base_value = baseline["l2"][label][component]
                    higher_value = higher["l2"][label][component]
                    relative = higher_value / base_value - 1.0 if base_value > 1e-14 else None
                    surface_rows.append({
                        "mesh": f"iv{resolution}",
                        "component": component,
                        "quantity": label,
                        "baseline_surface_order": 3,
                        "comparison_surface_order": surface_order,
                        "surface_q3_L2": base_value,
                        "comparison_L2": higher_value,
                        "absolute_change": higher_value - base_value,
                        "relative_change": relative,
                    })
            higher_surface_runs.append(higher)
    if missing_surface_checks and not args.allow_incomplete:
        raise RuntimeError("Incomplete required surface-q5 matrix; missing: " +
                           ", ".join(missing_surface_checks))
    require(surface_rows or args.allow_incomplete,
            "No completed higher-order surface quadrature checks were found")
    write_rows(
        output / "surface_quadrature_sensitivity.csv",
        surface_rows,
        ["mesh", "component", "quantity", "baseline_surface_order",
         "comparison_surface_order", "surface_q3_L2", "comparison_L2",
         "absolute_change", "relative_change"],
    )

    transient_runs = []
    transient_snapshots: dict[str, dict[str, np.ndarray]] = {}
    missing_transient = []
    common_transient = None
    for resolution, (name, ranks) in TRANSIENT_RUNS.items():
        directory = transient_root / name
        if not (directory / "accuracy.json").is_file():
            missing_transient.append(name)
            continue
        run, snapshots = audit_transient(directory, ranks, static_by_n[resolution]["nodes"])
        run.update(mesh=f"iv{resolution}", nominal_resolution=resolution,
                   h=static_by_n[resolution]["h"])
        transient_signature = comparable_transient_configuration(run["configuration"])
        spatial_signature = comparable_configuration(run["configuration"], keep_mesh_file=True)
        if common_transient is None:
            common_transient = transient_signature
        else:
            require(transient_signature == common_transient,
                    f"{run['mesh']}: transient numerical settings differ from previous grids")
        require(spatial_signature == comparable_configuration(
                    static_by_n[resolution]["configuration"], keep_mesh_file=True),
                f"{run['mesh']}: static and transient spatial settings differ")
        expected_gauss_points = sum(
            static_by_n[resolution][key]
            for key in ("volume_quadrature_points", "surface_quadrature_points",
                        "boundary_quadrature_points")
        )
        require(run["gauss_points_stored"] == expected_gauss_points,
                f"{run['mesh']}: transient quadrature count does not match the new surface-q3 static geometry")
        run["expected_gauss_points_from_static"] = expected_gauss_points
        transient_runs.append(run)
        transient_snapshots[name] = snapshots

    if missing_transient and not args.allow_incomplete:
        raise RuntimeError("Incomplete transient matrix; missing completed accuracy.json for: " +
                           ", ".join(missing_transient))
    require(transient_runs, "No completed transient runs were found")

    for index, run in enumerate(transient_runs):
        run["orders"] = {field: {} for field in ("rho", "pressure")}
        for field in ("rho", "pressure"):
            for norm in NORMS:
                run["orders"][field][norm] = (
                    observed_order(transient_runs[index - 1]["h"], run["h"],
                                   transient_runs[index - 1]["final"]["errors"][field][norm],
                                   run["final"]["errors"][field][norm])
                    if index else None
                )

    transient_fits: dict[str, dict[str, dict[str, Any]]] = {}
    for field in ("rho", "pressure"):
        transient_fits[field] = {}
        for norm in NORMS:
            values = [run["final"]["errors"][field][norm] for run in transient_runs]
            fits = {
                "all_grids": least_squares_order(transient_runs, values),
                "finest_three": least_squares_order(transient_runs, values, 3)
                if len(transient_runs) >= 3 else
                {"points": len(transient_runs), "order": None, "r_squared": None},
            }
            transient_fits[field][norm] = fits
            for subset, fit in fits.items():
                fit_rows.append({
                    "category": "transient_t2",
                    "quantity": field,
                    "component_or_norm": norm,
                    "grid_subset": subset,
                    **fit,
                })

    transient_rows = []
    conservation_rows = []
    for run in transient_runs:
        for field in ("rho", "pressure"):
            row: dict[str, Any] = {
                "mesh": run["mesh"],
                "mpi_ranks": run["mpi_ranks"],
                "h_from_static": run["h"],
                "time": run["time"],
                "steps": run["steps"],
                "quantity": field,
            }
            for norm in NORMS:
                row[norm] = run["final"]["errors"][field][norm]
                row[f"order_{norm}"] = run["orders"][field][norm]
            transient_rows.append(row)
        conservation_rows.append({
            "mesh": run["mesh"],
            "mpi_ranks": run["mpi_ranks"],
            "steps": run["steps"],
            "mass_scaled_drift": run["conservation"]["scaled_drift"]["mass"],
            "momentum_x_scaled_drift": run["conservation"]["scaled_drift"]["momentum_x"],
            "momentum_y_scaled_drift": run["conservation"]["scaled_drift"]["momentum_y"],
            "momentum_z_scaled_drift": run["conservation"]["scaled_drift"]["momentum_z"],
            "total_energy_scaled_drift": run["conservation"]["scaled_drift"]["total_energy"],
            "max_abs_scaled_drift": run["conservation"]["max_abs_scaled_drift"],
            "min_density_mean": run["final"]["min_density_mean"],
            "min_pressure_of_mean": run["final"]["min_pressure_of_mean"],
            "min_density_point": run["final"]["min_density_point"],
            "min_pressure_point": run["final"]["min_pressure_point"],
        })
    write_rows(output / "transient_convergence.csv", transient_rows)
    write_rows(output / "conservation_and_physicality.csv", conservation_rows)
    write_rows(output / "fitted_orders.csv", fit_rows)

    # iv10_np2 is a reproducibility check, not another point on the convergence curve.
    transient_np2, transient_np2_snapshots = audit_transient(
        transient_root / "iv10_np2", 2, static_by_n[10]["nodes"])
    require(comparable_transient_configuration(transient_np2["configuration"], keep_mesh_file=True) ==
            comparable_transient_configuration(transient_runs[0]["configuration"], keep_mesh_file=True),
            "iv10 np2 transient configuration differs from np1")
    expected_np2_gauss = sum(
        static_np2[key] for key in ("volume_quadrature_points", "surface_quadrature_points",
                                    "boundary_quadrature_points")
    )
    require(transient_np2["gauss_points_stored"] == expected_np2_gauss,
            "iv10 np2 transient quadrature count does not match its new surface-q3 static geometry")
    transient_np2["expected_gauss_points_from_static"] = expected_np2_gauss
    transient_snapshots["iv10_np2"] = transient_np2_snapshots
    node_mpi = compare_iv10_nodes(transient_snapshots["iv10_np1"],
                                  transient_snapshots["iv10_np2"], output)

    complete = not missing_transient and not missing_surface_checks
    result = {
        "schema": "traditional-roe16-accuracy-v1",
        "data_directory": str(data_root),
        "complete": complete,
        "missing_transient_runs": missing_transient,
        "validated_settings": {
            "riemann_solver": "Roe",
            "algorithm_mode": "TraditionalQuadrature",
            "volume_quadrature_order": 4,
            "baseline_surface_quadrature_order": 3,
            "required_surface_sensitivity_order": 5,
            "optional_surface_sensitivity_order": 7,
            "surface_points_per_micro_triangle": SURFACE_POINTS,
            "stencil_size": 16,
            "coefficient_rows": 9,
            "transient_cfl": 0.5,
            "transient_final_time": 2.0,
        },
        "static_runs": static_runs,
        "static_fitted_orders": static_fits,
        "static_iv10_np2": static_np2,
        "static_iv10_np1_np2_max_L2_difference": static_mpi_max,
        "higher_surface_quadrature_runs": higher_surface_runs,
        "surface_quadrature_comparison": surface_rows,
        "missing_required_surface_checks": missing_surface_checks,
        "transient_runs": transient_runs,
        "transient_fitted_orders": transient_fits,
        "transient_iv10_np2": transient_np2,
        "iv10_np1_np2_node_comparison": node_mpi,
    }
    # Remove full resolved configurations from each repeated record while retaining one auditable copy.
    result["common_spatial_configuration"] = static_runs[0]["configuration"]
    for key in ("static_runs", "higher_surface_quadrature_runs", "transient_runs"):
        for run in result[key]:
            run.pop("configuration", None)
    result["static_iv10_np2"].pop("configuration", None)
    result["transient_iv10_np2"].pop("configuration", None)
    (output / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    surface_density = [row for row in surface_rows if row["component"] == "rho" and
                       row["quantity"] in ("jump_quadrature", "roe_face", "roe_rhs", "total_rhs")]
    surface_table = "| 网格 | 对照阶数 | 量 | surface q3 L2 | 对照 L2 | 相对变化 |\n"
    surface_table += "|---|---:|---|---:|---:|---:|\n"
    for row in surface_density:
        relative = "—" if row["relative_change"] is None else f'{100 * row["relative_change"]:.6f}%'
        surface_table += (f'| {row["mesh"]} | q{row["comparison_surface_order"]} | '
                          f'{row["quantity"]} | {row["surface_q3_L2"]:.8e} | '
                          f'{row["comparison_L2"]:.8e} | {relative} |\n')

    static_fit_table = markdown_fit_table(
        {label: static_fits[label]["rho"] for label in
         ("jump_quadrature", "roe_face", "central_rhs", "roe_rhs", "total_rhs")},
        [("jump_quadrature", "density trace jump"),
         ("roe_face", "density Roe face"),
         ("central_rhs", "density central RHS"),
         ("roe_rhs", "density Roe dissipation RHS"),
         ("total_rhs", "density total RHS")],
    )
    transient_fit_table = markdown_fit_table(
        {f"{field}_{norm}": transient_fits[field][norm]
         for field in ("rho", "pressure") for norm in NORMS},
        [(f"{field}_{norm}", f"{field} {norm}")
         for field in ("rho", "pressure") for norm in NORMS],
    )
    surface_relative_max = max(
        (abs(row["relative_change"]) for row in surface_density
         if row["relative_change"] is not None),
        default=0.0,
    )

    conservation_table = "| 网格 | 最大守恒缩放漂移 | min rho(mean) | min p(mean) | min rho(point) | min p(point) |\n"
    conservation_table += "|---|---:|---:|---:|---:|---:|\n"
    for row in conservation_rows:
        conservation_table += (f'| {row["mesh"]} | {row["max_abs_scaled_drift"]:.4e} | '
                               f'{row["min_density_mean"]:.8e} | {row["min_pressure_of_mean"]:.8e} | '
                               f'{row["min_density_point"]:.8e} | {row["min_pressure_point"]:.8e} |\n')

    completion_note = (
        "四套瞬态网格和必需的q5界面积分对照均已完成，以下相邻阶覆盖 "
        "iv10→iv20→iv40→iv80。"
        if complete else
        "这是运行中的临时报告；尚缺 " +
        ", ".join(missing_transient + missing_surface_checks) +
        "，当前瞬态阶数只使用已完成网格。待全部 `accuracy.json` 写完后，不带 "
        "`--allow-incomplete` 重新运行。"
    )
    static_fine = static_runs[-1]["orders"]
    if complete:
        rho_fine_l2 = transient_runs[-1]["orders"]["rho"]["L2"]
        pressure_fine_l2 = transient_runs[-1]["orders"]["pressure"]["L2"]
        rho_all_l2 = transient_fits["rho"]["L2"]["all_grids"]["order"]
        pressure_all_l2 = transient_fits["pressure"]["L2"]["all_grids"]["order"]
        transient_conclusion = (
            f"完整t=2解的细端L2阶为密度 {rho_fine_l2:.4f}、压力 {pressure_fine_l2:.4f}；"
            f"四网格全区间拟合阶为密度 {rho_all_l2:.4f}、压力 {pressure_all_l2:.4f}。"
        )
    else:
        transient_conclusion = (", ".join(missing_transient) +
                                "尚未完成，完整t=2细端阶和四网格拟合阶暂不下结论。")
    conclusion = (
        "**新版传统格式没有达到三阶。** 首次重构的密度界面trace跳量与Roe面平均通量"
        f"在iv40→iv80上分别达到 {static_fine['jump_quadrature']['rho']:.4f} 和 "
        f"{static_fine['roe_face']['rho']:.4f} 阶，已经接近三阶；但映射到节点后，central、"
        f"Roe耗散和total RHS仅为 {static_fine['central_rhs']['rho']:.4f}、"
        f"{static_fine['roe_rhs']['rho']:.4f}、{static_fine['total_rhs']['rho']:.4f} 阶。"
        f"界面积分q3提高到q5/q7时，表中密度关键量最大相对变化仅 "
        f"{100 * surface_relative_max:.6f}%，排除了界面Gauss/Hammer点数不足作为主因。"
        + transient_conclusion +
        " 因此只调整Roe耗散不足以恢复三阶；下一步应优先做exact-face-flux/重构误差分解与"
        "三次多项式patch test，检查QR16两侧共模 O(h^3) 迹误差和离散散度的三阶矩一致性。"
        "现有闭合、法向与守恒审计未发现几何或装配错误。"
    )
    report = f"""# 新版传统 NCFV（Roe + QR16）精度检测

## 数据完整性与配置

{completion_note}

## 结论

{conclusion}

脚本逐项确认主算例使用 `TraditionalQuadrature`、标准 `Roe`、体积分 q4、界面积分 q3、
无黏性且无限制器。全部静态首次重构的模板大小严格为16，二次重构系数严格为9行；
瞬态初始和最终记录的模板大小也严格为16。高阶交叉检查只把界面积分从 q3 改为
q5（可选q7），体积分仍保持 q4。脚本强制检查新版映射：q3/q5/q7分别对应每个微三角面
3/6/12点，分别具有二/四/六次多项式代数精度；因此11:30之前生成、q3仍为6点的旧数据
会被拒绝。每个瞬态算例存储的总积分点数还必须逐网格等于相同MPI静态几何的体/内表面/
边界积分点数之和，防止新静态与旧瞬态混用。静态主算例每个微四面体使用
{static_runs[0]['points_per_micro_tetrahedron']} 个点，每个微三角面使用
{static_runs[0]['points_per_micro_triangle']} 个点。

## 首次重构静态误差

下表给出密度分量。`jump_quadrature` 是左右重构迹在界面积分点的面积加权 L2；
`Roe 面平均` 是先积分后除面积的耗散通量；RHS 范数以对偶体积加权。
网格尺度统一取静态诊断给出的 `h=(400/N_periodic)^(1/3)`，没有假定网格比严格为2。

{markdown_static_table(static_runs)}
全部5个守恒分量和每个相邻网格阶见 `static_convergence.csv`。

密度静态量的对数最小二乘拟合为：

{static_fit_table}

## 界面积分敏感性

iv40/iv80 保持同一重构、Roe格式和体积分q4，仅将界面积分从 q3 提高到 q5；
若存在q7结果也一并列入：

{surface_table}
全部守恒分量见 `surface_quadrature_sensitivity.csv`。相对变化定义为 `(qN-q3)/q3`；
接近机器零的分量不报告相对比值。

## t=2 瞬态解误差

误差由 `points_final.rank*.csv` 的恢复守恒点值独立复算。压力按
`p=(gamma-1)[rhoE-(rhou^2+rhov^2+rhow^2)/(2rho)]` 重算并与文件中的压力交叉核对。
相邻阶使用对应静态网格的 h。

### 密度

{markdown_transient_table(transient_runs, 'rho')}
### 压力

{markdown_transient_table(transient_runs, 'pressure')}

t=2各误差范数的对数最小二乘拟合为：

{transient_fit_table}

## 守恒性与物理性

守恒缩放漂移定义为 `(Q(t=2)-Q(0))/max(|Q(0)|,1)`；物理性同时检查控制体均值和恢复点值。

{conservation_table}
## iv10 串并行逐节点一致性

iv10 的 np1/np2 初始与最终快照均按 `original_node` 排序后逐节点、逐字段比较。
守恒/恢复状态最大绝对差为 **{node_mpi['max_abs_state_difference']:.4e}**，
全部节点字段最大绝对差为 **{node_mpi['max_abs_all_field_difference']:.4e}**，
密度/压力误差范数最大绝对差为 **{node_mpi['max_abs_error_norm_difference']:.4e}**，
均低于判据 {node_mpi['tolerance']:.1e}。静态 jump/face/RHS 的全部 L2 指标最大差为
**{static_mpi_max:.4e}**。每个节点的原始差值在 `iv10_np1_np2_node_differences.csv`，
逐字段 max/mean/RMS 汇总在 `metrics.json`。

## 输出文件

- `static_convergence.csv`：首次重构 jump、Roe界面通量、中心/Roe/总RHS的L2与相邻阶；
- `surface_quadrature_sensitivity.csv`：iv40/iv80界面q3→q5（以及可选q7）变化；
- `transient_convergence.csv`：t=2密度和压力L1/L2/L∞及相邻阶；
- `fitted_orders.csv`：静态与t=2误差的全区间/最细三网格拟合阶和R²；
- `conservation_and_physicality.csv`：守恒漂移和正性下界；
- `iv10_np1_np2_node_differences.csv`：np2减np1的逐节点差；
- `metrics.json`：全部汇总、配置核验、输入校验值，以及 accuracy/diagnostics/各rank点快照SHA-256。
"""
    (output / "report.md").write_text(report)
    print(f"Wrote {'complete' if complete else 'partial'} Roe/QR16 traditional audit to {output}")


if __name__ == "__main__":
    main()
