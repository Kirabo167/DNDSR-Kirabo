"""Compare the literal InvisFluxRoeHO diagnostic with the preceding reference Roe run."""

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze_taylor_green_self_convergence import FIELDS, LEVELS, load_case, weighted_difference


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("previous", type=Path)
    args = parser.parse_args()
    output, previous = args.output.resolve(), args.previous.resolve()
    checks, entropy, tgv_comparison = [], [], []
    for problem in ("entropy", "tgv", "entropy_dt_half", "tgv_dt_half"):
        for path in sorted((output / problem / "raw").glob("*.json")):
            if path.name.endswith(".manifest.json"):
                continue
            meta = json.loads(path.read_text())
            if not all(meta[k] for k in ("reference_document_flux", "reference_roe_harten",
                                          "traditional_edge_midpoint_flux")):
                raise RuntimeError(f"Incorrect flux selection: {path}")
            if meta["mean_fallback_count"] or meta["nonphysical_point_count"]:
                raise RuntimeError(f"Physicality fallback or invalid final state: {path}")
            if meta["final_rhs_riemann_calls"] != 4 * meta["nodes"]:
                raise RuntimeError(f"Unexpected interface solve count: {path}")
            checks.append({"problem": problem, "file": path.name, "nodes": meta["nodes"],
                           "dt": meta["dt"], "mean_fallback_count": meta["mean_fallback_count"],
                           "final_rhs_riemann_calls": meta["final_rhs_riemann_calls"],
                           "mass_drift": meta["final_conserved"][0] - meta["initial_conserved"][0],
                           "minimum_point_density": meta["minimum_point_density"],
                           "minimum_point_pressure": meta["minimum_point_pressure"]})
    if len(checks) != 10:
        raise RuntimeError(f"Expected ten completed cases, got {len(checks)}")
    coarse = None
    for level in LEVELS:
        name = f"entropy_prism{level}_traditional_p2.json"
        current = json.loads((output / "entropy/raw" / name).read_text())
        old = json.loads((previous / "entropy_reference/raw" / name).read_text())
        row = {"level": level, "nodes": current["nodes"],
               "mean_density_L2": current["mean_conservative"]["L2"][0],
               "point_density_L2": current["point_primitive"]["L2"][0],
               "mean_order": "", "point_order": "",
               "previous_point_density_L2": old["point_primitive"]["L2"][0]}
        row["relative_change_to_previous"] = abs(
            row["point_density_L2"] / row["previous_point_density_L2"] - 1)
        if coarse:
            log_ratio = math.log((row["nodes"] / coarse["nodes"]) ** (1 / 3))
            for prefix in ("mean", "point"):
                row[f"{prefix}_order"] = math.log(
                    coarse[f"{prefix}_density_L2"] / row[f"{prefix}_density_L2"]) / log_ratio
        entropy.append(row)
        coarse = row
        meta, a, ma = load_case(output / "tgv/raw", level, "Traditional", 2, 0.1)
        _, b, mb = load_case(previous / "tgv_reference/raw", level, "Traditional", 2, 0.1)
        keys = sorted(ma)
        if set(keys) != set(mb):
            raise RuntimeError(f"Different coordinates on level {level}")
        ia, ib = [ma[k] for k in keys], [mb[k] for k in keys]
        for field, columns in FIELDS.items():
            tgv_comparison.append({"level": level, "field": field,
                                   "point_difference_L2": weighted_difference(
                                       a, b, ia, ib, a[ia, 8], columns)})
    normal = json.loads((output / "entropy/raw/entropy_prism40_traditional_p2.json").read_text())
    half = json.loads((output / "entropy_dt_half/raw/entropy_prism40_traditional_p2.json").read_text())
    time_rows = [{"problem": "entropy", "field": "density_error_L2",
                  "normal": normal["point_primitive"]["L2"][0],
                  "half": half["point_primitive"]["L2"][0],
                  "difference": abs(normal["point_primitive"]["L2"][0] -
                                    half["point_primitive"]["L2"][0])}]
    _, a, ma = load_case(output / "tgv/raw", 40, "Traditional", 2, 0.1)
    _, b, mb = load_case(output / "tgv_dt_half/raw", 40, "Traditional", 2, 0.1)
    keys = sorted(ma)
    ia, ib = [ma[k] for k in keys], [mb[k] for k in keys]
    for field, columns in FIELDS.items():
        time_rows.append({"problem": "tgv", "field": field, "normal": "", "half": "",
                          "difference": weighted_difference(a, b, ia, ib, a[ia, 8], columns)})
    shared_cases = {n: load_case(output / "tgv/raw", n, "Traditional", 2, 0.1) for n in LEVELS}
    common = sorted(set.intersection(*(set(shared_cases[n][2]) for n in LEVELS)))
    coarse_meta, coarse_table, coarse_map = shared_cases[10]
    fixed_weights = coarse_table[[coarse_map[k] for k in common], 8]
    for field, columns in FIELDS.items():
        time_rows.append({"problem": "tgv", "field": field + "_fixed_common_nodes",
                          "normal": "", "half": "", "difference": weighted_difference(
                              a, b, [ma[k] for k in common], [mb[k] for k in common],
                              fixed_weights, columns)})
    write_csv(output / "checks.csv", checks)
    write_csv(output / "entropy_accuracy.csv", entropy)
    write_csv(output / "tgv_previous_difference.csv", tgv_comparison)
    write_csv(output / "time_step_check.csv", time_rows)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for label, root in (("Surface quadrature (previous)", previous / "entropy_quadrature"),
                        ("Reference Roe midpoint (previous)", previous / "entropy_reference"),
                        ("Literal document method (new)", output / "entropy")):
        data = [json.loads((root / "raw" / f"entropy_prism{n}_traditional_p2.json").read_text())
                for n in LEVELS]
        axes[0].loglog([d["nodes"] ** (-1 / 3) for d in data],
                       [d["point_primitive"]["L2"][0] for d in data], "o-", label=label)
    axes[0].set(xlabel=r"$h=N^{-1/3}$", ylabel=r"Density point $L_2$ error", title="Exact entropy wave")
    axes[0].legend(fontsize=8)
    rows = list(csv.DictReader((output / "tgv/all_four_self_convergence.csv").open()))
    for field in FIELDS:
        row = next(r for r in rows if r["field"] == field)
        axes[1].plot(["10/20/40", "20/40/80"],
                     [float(row["order_10_20_40"]), float(row["order_20_40_80"])], "o-", label=field)
    axes[1].axhline(2, color="gray", linestyle=":")
    axes[1].set(ylabel="Observed self-convergence order", title="Taylor-Green: fixed common nodes")
    axes[1].legend()
    for axis in axes:
        axis.grid(True, which="both", alpha=0.25)
    fig.suptitle("Quadratic variational reconstruction; static periodic prism meshes")
    fig.savefig(output / "comparison.png", dpi=180)
    fig.savefig(output / "comparison.pdf")
    print(json.dumps({"cases": len(checks), "entropy": entropy,
                      "time_step_check": time_rows,
                      "maximum_tgv_difference_to_previous": max(
                          r["point_difference_L2"] for r in tgv_comparison)}, indent=2))


if __name__ == "__main__":
    main()
