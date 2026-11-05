"""Summarize traditional quadrature versus primal-edge midpoint flux accuracy."""
import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--include-reference", action="store_true")
    args = parser.parse_args()
    root = args.output.resolve()
    rows = []
    schemes = ("quadrature", "midpoint", "reference") if args.include_reference else ("quadrature", "midpoint")
    for scheme in schemes:
        previous = None
        for level in (10, 20, 40, 80):
            path = root / f"entropy_{scheme}/raw/entropy_prism{level}_traditional_p2.json"
            data = json.loads(path.read_text())
            row = {"scheme": scheme, "level": level, "nodes": data["nodes"],
                   "method": data.get("method", "LeastSquares"),
                   "point_L2": data["point_primitive"]["L2"][0],
                   "mean_L2": data["mean_conservative"]["L2"][0],
                   "point_order": "", "mean_order": "",
                   "rhs_riemann_calls": data["final_rhs_riemann_calls"],
                   "mass_drift": data["final_conserved"][0]-data["initial_conserved"][0]}
            if previous:
                denominator = math.log(row["nodes"] / previous["nodes"]) / 3
                for field in ("point", "mean"):
                    row[f"{field}_order"] = math.log(previous[f"{field}_L2"] /
                                                       row[f"{field}_L2"]) / denominator
            rows.append(row)
            previous = row
    with (root / "entropy_comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(rows[0]["method"] + " reconstruction")
    styles = {"quadrature": ("Interface quadrature H2", "#456990"),
              "midpoint": ("Edge midpoint H2", "#d1495b"),
              "reference": ("Edge midpoint reference Roe", "#4a8f50")}
    for scheme in schemes:
        label, color = styles[scheme]
        series = [r for r in rows if r["scheme"] == scheme]
        x = [(r["nodes"] / 512)**(1/3) for r in series]
        axes[0].loglog(x, [r["point_L2"] for r in series], "o-", color=color, label=label)
        with (root / f"tgv_{scheme}/all_four_self_convergence.csv").open() as stream:
            velocity = next(r for r in csv.DictReader(stream) if r["field"] == "velocity")
        axes[1].loglog(x[:3], [float(velocity[f"difference_{p}"]) for p in
                              ("10_20", "20_40", "40_80")], "o-", color=color, label=label)
    for ax, title, labels, locations in zip(
            axes, ("Entropy wave density error", "Taylor–Green velocity self-difference"),
            (["IV10", "IV20", "IV40", "IV80"], ["IV10–20", "IV20–40", "IV40–80"]),
            (x, x[:3])):
        ax.set_title(title)
        ax.set_ylabel("Volume-weighted L2")
        ax.set_xlabel("Prism refinement")
        ax.xaxis.set_major_locator(FixedLocator(locations))
        ax.xaxis.set_major_formatter(FixedFormatter(labels))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(root / "comparison.png", dpi=180)
    fig.savefig(root / "comparison.pdf")


if __name__ == "__main__":
    main()
