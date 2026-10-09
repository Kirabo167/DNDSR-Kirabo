"""Compare fixed-rank prism vortex accuracy with and without LS distance weights."""

import csv
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "docs/reports/ncfv_distance_weight_ablation_20260929"
MODES = ("Traditional", "Efficient")
METHODS = ("LeastSquares", "SVDLeastSquares")
LEVELS = (10, 20, 40)
POWERS = (("weighted", 1.0), ("unweighted", 0.0))


def read_result(group, level, mode, method):
    stem = f"prism_{level}_{mode}_{method}_Roe"
    result = json.loads((OUTPUT / group / f"{stem}.json").read_text())
    settings = result["configuration"]["reconstruction"]
    expected = dict(POWERS)[group]
    if settings["distanceWeightPower"] != expected:
        raise ValueError(f"Wrong weight exponent in {stem}")
    if result["mpi_ranks"] != 4 or result["end_time"] != 1.6:
        raise ValueError(f"Wrong run settings in {stem}")
    return result, OUTPUT / group / stem


def read_field(directory, steps):
    paths = sorted(directory.glob(f"solution_{steps:08d}.nodes.rank*.csv"))
    if len(paths) != 4:
        raise ValueError(f"Expected four owned-node files in {directory}")
    return np.sort(np.concatenate([
        np.genfromtxt(path, delimiter=",", names=True) for path in paths
    ]), order="original_node")


def field_difference(left, right, steps):
    first, second = read_field(left, steps), read_field(right, steps)
    if not np.array_equal(first["original_node"], second["original_node"]):
        raise ValueError("Mismatch in node identity")
    for name in ("x", "y", "z", "partial_volume"):
        if not np.allclose(first[name], second[name], atol=1e-12, rtol=1e-12):
            raise ValueError(f"Mismatch in {name}")
    weights = first["partial_volume"]
    delta = first["rho_point"] - second["rho_point"]
    return math.sqrt(float(np.sum(weights * delta**2) / np.sum(weights)))


def write_rows(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    rows, orders = [], []
    for mode in MODES:
        for method in METHODS:
            previous = None
            for level in LEVELS:
                weighted, weighted_dir = read_result("weighted", level, mode, method)
                unweighted, unweighted_dir = read_result("unweighted", level, mode, method)
                if (weighted["nodes"] != unweighted["nodes"] or
                    weighted["steps"] != unweighted["steps"] or
                    weighted["dt"] != unweighted["dt"]):
                    raise ValueError("Ablation runs have different resolution or time step")
                row = {
                    "mode": mode, "method": method, "level": level,
                    "nodes": weighted["nodes"], "ranks": weighted["mpi_ranks"],
                    "dt": weighted["dt"], "steps": weighted["steps"],
                    "weighted_initial_L2": weighted["initial"]["rho_point_L2V"],
                    "unweighted_initial_L2": unweighted["initial"]["rho_point_L2V"],
                    "weighted_final_L2": weighted["final"]["rho_point_L2V"],
                    "unweighted_final_L2": unweighted["final"]["rho_point_L2V"],
                    "unweighted_over_weighted": (
                        unweighted["final"]["rho_point_L2V"] /
                        weighted["final"]["rho_point_L2V"]),
                    "final_rho_field_L2_difference": field_difference(
                        weighted_dir, unweighted_dir, weighted["steps"]),
                    "weighted_mass_drift": weighted["final"]["mass_drift"],
                    "unweighted_mass_drift": unweighted["final"]["mass_drift"],
                }
                rows.append(row)
                if previous is not None:
                    h_ratio = (weighted["nodes"] / previous["nodes"]) ** (1 / 3)
                    orders.append({
                        "mode": mode, "method": method,
                        "levels": f"{previous['level']}-{level}",
                        "weighted_order": math.log(
                            previous["weighted_final_L2"] / row["weighted_final_L2"]
                        ) / math.log(h_ratio),
                        "unweighted_order": math.log(
                            previous["unweighted_final_L2"] / row["unweighted_final_L2"]
                        ) / math.log(h_ratio),
                    })
                previous = row
    write_rows(OUTPUT / "metrics.csv", rows)
    write_rows(OUTPUT / "orders.csv", orders)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for axis, mode in zip(axes, MODES):
        # SVDLS lies on the LS curves to roundoff; plot each curve only once.
        selected = [row for row in rows if row["mode"] == mode and
                    row["method"] == "LeastSquares"]
        x = [row["nodes"] ** (1 / 3) for row in selected]
        for group, style, color in (("weighted", "-", "C0"),
                                    ("unweighted", "--", "C1")):
            axis.loglog(x, [row[f"{group}_final_L2"] for row in selected],
                        linestyle=style, color=color, marker="o",
                        label=f"{group} (LS = SVDLS)")
        axis.set_title(mode)
        axis.set_xlabel(r"$N^{1/3}$")
        axis.grid(True, which="both", alpha=0.2)
    axes[0].set_ylabel("Volume-weighted density L2 at t=1.6")
    axes[1].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(OUTPUT / "convergence.png", dpi=180)
    figure.savefig(OUTPUT / "convergence.pdf")
    print(f"{len(rows)} pairs, {len(orders)} order comparisons")
    for row in rows:
        print(row["mode"], row["method"], row["level"],
              f"{row['weighted_final_L2']:.9e}",
              f"{row['unweighted_final_L2']:.9e}",
              f"ratio={row['unweighted_over_weighted']:.5f}")
    for row in orders:
        print(row)


if __name__ == "__main__":
    main()
