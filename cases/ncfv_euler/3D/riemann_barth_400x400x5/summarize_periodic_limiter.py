"""Plot the controlled periodic limiter comparisons and export a metric table."""

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/ncfv-periodic-limiter-matplotlib")

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np


LABELS = {
    "baseline": "Baseline Barth",
    "pair_bounds": "Endpoint bounds",
    "LLF_only": "LLF only",
    "face_Roe": "Face Roe only",
    "first_order": "First order",
    "first_order_local": "First order + local boundary flux",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    directory = args.directory.resolve()
    result = json.loads((directory / "results.json").read_text())
    records = {run["variant"]: run for run in result["runs"]}
    comparison = json.loads((directory / "first_order_comparison.json").read_text())
    density_min, density_max = .13, 1.56
    levels = np.linspace(density_min, density_max, 27)[1:-1]
    x = np.linspace(0, 1, 41)
    figure, axes = plt.subplots(2, 3, figsize=(15, 10.8), sharex=True, sharey=True,
                                layout="constrained", facecolor="white")
    normalizer = Normalize(density_min, density_max)
    for label, axis in zip(LABELS, axes.flat):
        record = records[label]
        with h5py.File(record["restart"]) as data:
            values = data["state/father/array/data"][:].reshape(-1, 5)
            ids = data["state/origIndex"][:].astype(np.int64)
        ordered = np.empty_like(values)
        ordered[ids] = values
        density = ordered[:, 0].reshape(5, 41, 41)[2]
        image = axis.pcolormesh(x, x, density, shading="gouraud", cmap="coolwarm", norm=normalizer,
                               rasterized=True)
        axis.contour(x, x, density, levels=levels, colors="black", linewidths=.43, alpha=.65)
        axis.set_title(LABELS[label] + "\n" +
                       rf"$\max_{{x,y}}\Delta_z\rho={record['rho_z_span_max']:.3g}$" + "   " +
                       rf"$\max|w|={record['w_abs_max']:.3g}$", fontsize=11.5, pad=9)
        axis.set_aspect("equal")
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.set_xticks([0, .5, 1])
        axis.set_yticks([0, .5, 1])
        axis.tick_params(labelsize=10)
        axis.set_xlabel("x", fontsize=11)
        axis.set_ylabel("y", fontsize=11)
    colorbar = figure.colorbar(image, ax=axes, location="right", shrink=.87, pad=.02, fraction=.035)
    colorbar.set_label(r"Density $\rho$ (conservative mean)", fontsize=12)
    figure.suptitle("RP1: controlled z-periodic limiter and flux comparisons\n"
                    r"$t=0.25$, $40\times40\times5$ cells, 4 MPI ranks, node plane $z=0.05$"
                    "\nShared blue-to-red density scale; 25 fixed contour levels", fontsize=15)
    path = directory / "RP1_periodic_limiter_comparison.png"
    figure.savefig(path, dpi=180, facecolor="white")
    plt.close(figure)
    columns = ["variant", "cfl", "time", "iteration", "rho_z_span_max", "p_z_span_max", "w_abs_max",
               "rho_z_TV_excess_max", "rho_xy_symmetry_abs_max", "p_xy_symmetry_abs_max",
               "rho_min", "rho_max", "p_min", "p_max", "rejected_steps",
               "rho_cut_y095_TV", "p_cut_y095_TV", "rho_abs_mean_vs_first_order",
               "p_abs_mean_vs_first_order"]
    csv_path = directory / "metrics_summary.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for label in [*LABELS, *(["cfl_half"] if "cfl_half" in records else [])]:
            record = records[label]
            row = {name: record[name] for name in columns if name in record}
            row.update(cfl=record.get("cfl", .5),
                       rho_cut_y095_TV=record["rho_cut_y095_midplane"]["TV"],
                       p_cut_y095_TV=record["p_cut_y095_midplane"]["TV"])
            difference = comparison.get(label + "_vs_first_order")
            row.update(rho_abs_mean_vs_first_order=difference["conservative_abs_mean"][0] if difference else 0,
                       p_abs_mean_vs_first_order=difference["pressure_abs_mean"] if difference else 0)
            writer.writerow(row)
    manifest = {"figure": str(path), "table": str(csv_path), "plane_index": 2, "z": .05,
                "density_limits": [density_min, density_max], "contour_levels": levels.tolist(),
                "note": "Node conservative means. Scalar maxima above their initial values alone do not establish oscillation."}
    (directory / "comparison_figure.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(path)
    print(csv_path)


if __name__ == "__main__":
    main()
