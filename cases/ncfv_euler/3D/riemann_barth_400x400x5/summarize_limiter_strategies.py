"""Combine controlled limiter experiments, excluding derivative projections.

Historical coordinate-invariance runs remain in diagnostic history and are
never eligible recommendations. The primary table combines discontinuity
control, exact-solution error, conservation, and smooth-wave dissipation.
"""

import argparse
import csv
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/ncfv-limiter-summary-matplotlib")

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np

from run_limiter_strategy_tests import read_output


VARIANTS = ["base", "pressure", "pair", "flux", "flux_pair", "flux_only", "flux_pair_only", "low_order",
            "complete_rings", "complete_rings_flux_only"]
LABELS = {"base": "Barth", "pressure": "Barth + p", "pair": "State pair bounds", "flux": "Flux + Barth",
          "flux_pair": "Flux pair + Barth", "flux_only": "Flux only", "flux_pair_only": "Flux pair only", "low_order": "Low LLF",
          "complete_rings": "Complete rings + Barth", "complete_rings_flux_only": "Complete rings + flux"}
WORKLOADS = [("RP1", 40), ("planar_shock", 40), ("contact", 40), ("smooth", 20), ("smooth", 40)]


def write_csv(path, rows, columns):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--candidates", nargs="+", default=VARIANTS)
    parser.add_argument("--true3d-summary", type=Path,
                        help="Completed independent cubic-mesh smooth/oblique summary")
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    runs, sources = [], []
    keys = set()
    for directory in args.directories:
        path = directory.resolve() / "results.json"
        data = json.loads(path.read_text())
        sources.append({"results": str(path), "status": data["status"],
                        "frozen_inputs": data["frozen_inputs"], "variants": data["variants"],
                        "mpi_ranks": data["mpi_ranks"], "cfl": data["cfl"], "time": data["time"]})
        for record in data["runs"]:
            key = record["problem"], record["cells"], record["variant"]
            if key in keys:
                raise ValueError("Duplicate controlled run: " + str(key))
            keys.add(key)
            record = dict(record, source_results=str(path))
            record["eligible"] = record["variant"] in args.candidates
            record["diagnostic_rejection_reason"] = "" if record["eligible"] else "Direct coordinate-derivative projection is excluded by the user."
            if record["problem"] == "contact" and record["returncode"] == 0:
                average = np.asarray(record["cut_density_yz_average"])
                record["contact_10_90_width_valid"] = bool(average.max() >= 1.9 and average.min() <= 1.1)
            if record["eligible"] and record["returncode"] == 0:
                with h5py.File(record["restart"]) as saved:
                    raw = saved.attrs["configuration"]
                    actual = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
                    if actual["reconstruction"].get("preserveCoordinateInvariance", False):
                        raise ValueError("A selected candidate modifies coordinate derivatives")
                    if abs(float(saved.attrs["simulationTime"]) - .25) > 1e-12 or actual["time"]["cfl"] != .5:
                        raise ValueError("Physical time or CFL differs between comparisons")
            runs.append(record)
    lookup = {(r["problem"], r["cells"], r["variant"]): r for r in runs if r["eligible"] and r["returncode"] == 0}
    missing = [(problem, cells, variant) for variant in args.candidates for problem, cells in WORKLOADS
               if (problem, cells, variant) not in lookup]
    if missing and not args.allow_incomplete:
        raise ValueError("Candidate matrix is incomplete: " + str(missing))
    true3d, true3d_lookup, true3d_runs = None, {}, []
    if args.true3d_summary:
        path = args.true3d_summary.resolve()
        data = json.loads(path.read_text())
        if not data["all_completed"]:
            raise ValueError("Genuine three-dimensional matrix is incomplete")
        true3d_sources = []
        for source in data["source_results"]:
            source_path = Path(source)
            source_data = json.loads(source_path.read_text())
            if source_data["status"] != "complete":
                raise ValueError("A genuine three-dimensional source is incomplete")
            true3d_sources.append({"results": str(source_path), "mpi_ranks": source_data["mpi_ranks"],
                                   "frozen_inputs": source_data["frozen_inputs"]})
            for record in source_data["runs"]:
                if record["returncode"] != 0:
                    raise ValueError("A genuine three-dimensional calculation failed")
                with h5py.File(record["restart"]) as saved:
                    raw = saved.attrs["configuration"]
                    actual = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
                    if actual["reconstruction"].get("preserveCoordinateInvariance", False):
                        raise ValueError("Genuine three-dimensional run modifies coordinate derivatives")
                    if actual["time"]["cfl"] != .5:
                        raise ValueError("Genuine three-dimensional run has a different CFL")
                    if abs(float(saved.attrs["simulationTime"]) - record["time"]) > 1e-12:
                        raise ValueError("Genuine three-dimensional final time differs from its record")
                true3d_runs.append(dict(record, source_results=str(source_path)))
        if len(true3d_runs) != data["run_count"]:
            raise ValueError("Genuine three-dimensional run count differs from summary")
        true3d_lookup = {row["variant"]: row for row in data["summary"]}
        true3d = {"summary_path": str(path), "combined_summary": data,
                  "sources": true3d_sources, "runs": true3d_runs}
    table = []
    for variant in args.candidates:
        row = {"variant": variant, "label": LABELS[variant]}
        group = {(problem, cells): lookup.get((problem, cells, variant)) for problem, cells in WORKLOADS}
        for workload, prefix, fields in [
            (("RP1", 40), "rp1", ["rho_z_span_max", "p_z_span_max", "w_abs_max"]),
            (("planar_shock", 40), "shock", ["rho_overshoot_exact_states", "p_overshoot_away_shocks_3dx",
                                             "p_undershoot_exact_states", "rho_L1_exact_dualmean"]),
            (("contact", 40), "contact", ["rho_L1", "rho_max", "mean_contact_10_90_width", "contact_10_90_width_valid"]),
            (("smooth", 20), "smooth20", ["rho_L1", "fundamental_amplitude_ratio"]),
            (("smooth", 40), "smooth40", ["rho_L1", "fundamental_amplitude_ratio"]),
        ]:
            record = group[workload]
            if record:
                row.update({prefix + "_" + name: record[name] for name in fields})
        if group["smooth", 20] and group["smooth", 40]:
            row["smooth_observed_L1_order"] = math.log(group["smooth", 20]["rho_L1"] / group["smooth", 40]["rho_L1"]) / math.log(2)
        conserved = [record["conservative_mean_error_max"] for record in group.values()
                     if record and "conservative_mean_error_max" in record]
        if conserved:
            row["maximum_periodic_conservative_mean_error"] = max(conserved)
        if variant in true3d_lookup:
            row.update({"true3d_" + name: value for name, value in true3d_lookup[variant].items()
                        if name != "variant"})
        table.append(row)
    summary = {"complete_candidate_matrix": not missing, "missing": missing,
               "successful_eligible_runs": len(lookup), "eligible_expected_runs": len(args.candidates) * len(WORKLOADS),
               "diagnostic_history_runs": sum(not r["eligible"] for r in runs),
               "successful_true3d_runs": len(true3d_runs),
               "successful_valid_experiments": len(lookup) + len(true3d_runs),
               "true3d": true3d,
               "candidate_policy": "Candidates must use general three-dimensional state/flux limiting. Direct coordinate-derivative projection is excluded.",
               "binary_compatibility_note": "The first matrix enables Barth throughout. Later binaries add the missing geometry/point positivity guards only to the previously untested Barth-disabled flux path, and add optional edge-pair bounds. First-matrix baseline states match prior pressure-off states bitwise. Source and executable hashes are retained separately; no cross-binary identity is assumed for new branches.",
               "interpretation_notes": [
                   "The primary five-workload matrix uses 4 MPI ranks, Roe, CFL=0.5, t=0.25 and the same mesh/initialization per workload.",
                   "RP1 has no exact reference here; scalar maxima above their initial values alone are not treated as proof of oscillation.",
                   "Shock error is measured against exact conservative dual-volume means; pressure overrun excludes 3dx around both acoustic shocks.",
                   "Smooth-wave order uses only two x,y resolutions with five fixed z cells; it is not a general 3-D convergence proof.",
                   "Contact width is a grid-count transition-support estimate. It is invalid as a classical 10-to-90 width when the pulse no longer reaches 90 percent amplitude.",
                   "Independent genuine three-dimensional runs are scoped separately: 20x20x20 cells, smooth t=0.25, oblique shock t=0.05, and error measured in the stated interior ROI.",
                   "No tested candidate establishes a general solution to transverse contamination with superior overall accuracy. Reduced RP1 z span alone is insufficient evidence.",
               ], "table": table, "runs": runs, "sources": sources}
    if true3d and "flux_only" in true3d_lookup and "base" in true3d_lookup:
        baseline, candidate = true3d_lookup["base"], true3d_lookup["flux_only"]
        rp1_base = lookup["RP1", 40, "base"]["rho_z_span_max"]
        rp1_candidate = lookup["RP1", 40, "flux_only"]["rho_z_span_max"]
        summary["cross_workload_conclusion"] = {
            "recommendation": "No universally superior replacement for Barth established; retain the experimental strategies as opt-in diagnostics.",
            "flux_only_RP1_density_z_span_reduction_percent": 100 * (1 - rp1_candidate / rp1_base),
            "flux_only_true3d_oblique_density_L1_increase_percent": 100 * (candidate["oblique_roi_rho_L1"] / baseline["oblique_roi_rho_L1"] - 1),
            "flux_only_true3d_oblique_tangential_velocity_max_increase_percent": 100 * (candidate["oblique_roi_tangential_velocity_max"] / baseline["oblique_roi_tangential_velocity_max"] - 1),
            "flux_only_true3d_smooth_density_L1_increase_percent": 100 * (candidate["smooth_rho_L1"] / baseline["smooth_rho_L1"] - 1),
            "limitation": "One genuine three-dimensional mesh resolution is a smoke test, not a convergence-order proof.",
        }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    columns = list(dict.fromkeys(key for row in table for key in row))
    write_csv(out / "strategy_summary.csv", table, columns)
    run_columns = ["problem", "cells", "variant", "eligible", "returncode", "iteration", "rho_z_span_max", "p_z_span_max",
                   "w_abs_max", "rho_min", "rho_max", "p_min", "p_max", "rho_L1", "rho_overshoot_exact_states",
                   "p_overshoot_away_shocks_3dx", "rho_L1_exact_dualmean", "fundamental_amplitude_ratio",
                   "mean_contact_10_90_width", "contact_10_90_width_valid", "conservative_mean_error_max", "source_results"]
    write_csv(out / "all_runs.csv", runs, run_columns)
    complete = [row for row in table if "contact_rho_L1" in row and "smooth40_rho_L1" in row and "shock_rho_L1_exact_dualmean" in row]
    if complete:
        figure, axes = plt.subplots(2, 2, figsize=(15, 10), layout="constrained", facecolor="white")
        for axis, field, title, logarithmic in zip(axes.flat,
             ["shock_rho_L1_exact_dualmean", "shock_p_overshoot_away_shocks_3dx", "contact_rho_L1", "smooth40_rho_L1"],
             ["Planar-shock density L1 error", "Pressure overrun away from shocks", "Periodic contact density L1 error", "Smooth-wave density L1 error (40 cells)"],
             [False, False, False, True]):
            axis.bar(np.arange(len(complete)), [row[field] for row in complete], color=["#547ca5" if row["variant"] != "flux_only" else "#c45c43" for row in complete])
            axis.set_xticks(np.arange(len(complete)), [row["label"] for row in complete], rotation=28, ha="right", fontsize=9)
            axis.set_title(title, fontsize=12)
            if logarithmic:
                axis.set_yscale("log")
            axis.grid(axis="y", alpha=.2)
            axis.set_axisbelow(True)
        figure.suptitle("Limiter strategy tradeoffs: error, nonphysical overshoot, and dissipation\nAll shown strategies avoid direct coordinate-derivative modification", fontsize=15)
        figure.savefig(out / "strategy_tradeoffs.png", dpi=180)
        plt.close(figure)
    true3d_rows = [row for row in table if "true3d_oblique_roi_rho_L1" in row]
    if true3d_rows:
        figure, axes = plt.subplots(2, 2, figsize=(15, 10), layout="constrained", facecolor="white")
        for axis, field, title, logarithmic in zip(axes.flat,
             ["rp1_rho_z_span_max", "true3d_oblique_roi_rho_L1",
              "true3d_oblique_roi_tangential_velocity_max", "true3d_smooth_rho_L1"],
             ["RP1 density z span: 40 x 40 x 5, t=0.25", "3-D oblique shock density L1: 20 cubed, t=0.05",
              "3-D oblique shock: spurious tangential velocity", "3-D smooth wave density L1: 20 cubed, t=0.25"],
             [True, False, False, True]):
            axis.bar(np.arange(len(true3d_rows)), [row[field] for row in true3d_rows],
                     color=["#c45c43" if row["variant"] == "flux_only" else "#547ca5" for row in true3d_rows])
            axis.set_xticks(np.arange(len(true3d_rows)), [row["label"] for row in true3d_rows],
                            rotation=28, ha="right", fontsize=9)
            axis.set_title(title, fontsize=12)
            if logarithmic:
                axis.set_yscale("log")
            axis.grid(axis="y", alpha=.2)
            axis.set_axisbelow(True)
        figure.suptitle("RP1 improvements do not establish general 3-D superiority\nOblique-shock errors use interior ROI [0.2, 0.8] cubed; no coordinate-derivative projection", fontsize=15)
        figure.savefig(out / "rp1_vs_true3d_tradeoffs.png", dpi=180)
        plt.close(figure)
    selected = [name for name in ("base", "pair", "flux_only", "low_order") if ("RP1", 40, name) in lookup]
    if selected:
        fields = [read_output(Path(lookup["RP1", 40, name]["restart"]), 40, False)[1] for name in selected]
        figure, axes = plt.subplots(2, 2, figsize=(11, 10), layout="constrained", facecolor="white")
        norm = Normalize(min(a.min() for a in fields), max(a.max() for a in fields))
        levels = np.linspace(norm.vmin, norm.vmax, 27)[1:-1]
        x = np.linspace(0, 1, 41)
        for axis, name, rho in zip(axes.flat, selected, fields):
            record = lookup["RP1", 40, name]
            image = axis.pcolormesh(x, x, rho[2], cmap="coolwarm", norm=norm, shading="gouraud")
            axis.contour(x, x, rho[2], levels=levels, colors="black", linewidths=.45, alpha=.65)
            axis.set_title(LABELS[name] + f"\nmax z density span={record['rho_z_span_max']:.3g}, max |w|={record['w_abs_max']:.3g}", fontsize=11)
            axis.set(xlabel="x", ylabel="y", xlim=(0, 1), ylim=(0, 1), aspect="equal")
        for axis in list(axes.flat)[len(selected):]:
            axis.set_visible(False)
        figure.colorbar(image, ax=axes, shrink=.86, label="Density (conservative mean)")
        figure.suptitle("RP1 selected strategies at t=0.25, node plane z=0.05\nShared blue-to-red scale and 25 fixed contour levels", fontsize=14)
        figure.savefig(out / "RP1_selected_density.png", dpi=180)
        plt.close(figure)
    print(out / "summary.json")
    print(out / "strategy_summary.csv")


if __name__ == "__main__":
    main()
