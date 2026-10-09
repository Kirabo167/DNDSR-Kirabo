"""Summarize fixed-rank prism LS distance-weight scans through IV80."""

import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "docs/reports/ncfv_distance_weight_scan_iv80_20260929"
POWERS = (("p0", 0.0), ("p0p5", 0.5), ("p1", 1.0), ("p2", 2.0),
          ("p3", 3.0), ("p4", 4.0))
MODES = ("Traditional", "Efficient")
LEVELS = (10, 20, 40, 80)


def read_result(group, power, mode, level):
    stem = f"prism_{level}_{mode}_LeastSquares_Roe"
    path = OUTPUT / group / f"{stem}.json"
    raw = json.loads(path.read_text())
    config = raw["configuration"]
    if config["reconstruction"]["distanceWeightPower"] != power:
        raise ValueError(f"Incorrect distance-weight power in {path}")
    if raw["mpi_ranks"] != 8 or abs(raw["end_time"] - 1.6) > 1e-12:
        raise ValueError(f"Incorrect MPI or final-time setting in {path}")
    if abs(raw["final"]["time"] - 1.6) > 1e-12:
        raise ValueError(f"Incomplete run in {path}")
    return raw


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    rows, orders = [], []
    references = {}
    for group, power in POWERS:
        for mode in MODES:
            previous = None
            for level in LEVELS:
                raw = read_result(group, power, mode, level)
                key = mode, level
                reference = references.get(key)
                settings = raw["configuration"].copy()
                settings["reconstruction"] = settings["reconstruction"].copy()
                settings["io"] = settings["io"].copy()
                del settings["reconstruction"]["distanceWeightPower"]
                del settings["io"]["outputPrefix"]
                if reference is None:
                    references[key] = settings
                elif settings != reference:
                    raise ValueError(f"Other configuration fields differ: {group}, {key}")
                row = {
                    "power": power, "mode": mode, "level": level,
                    "nodes": raw["nodes"], "ranks": raw["mpi_ranks"],
                    "dt": raw["dt"], "steps": raw["steps"],
                    "initial_rho_L2": raw["initial"]["rho_point_L2V"],
                    "final_rho_L1": raw["final"]["rho_point_L1V"],
                    "final_rho_L2": raw["final"]["rho_point_L2V"],
                    "final_rho_Linf": raw["final"]["rho_point_Linf"],
                    "mass_drift": raw["final"]["mass_drift"],
                    "initialization_seconds": raw["initialization_seconds"],
                    "march_seconds": raw["march_seconds"],
                }
                rows.append(row)
                if previous is not None:
                    denominator = math.log(raw["nodes"] / previous["nodes"]) / 3
                    orders.append({
                        "power": power, "mode": mode,
                        "levels": f"{previous['level']}-{level}",
                        "observed_order": math.log(
                            previous["final_rho_L2"] / row["final_rho_L2"]
                        ) / denominator,
                    })
                previous = row
    write_csv(OUTPUT / "metrics.csv", rows)
    write_csv(OUTPUT / "orders.csv", orders)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), sharey=True)
    for axis, mode in zip(axes, MODES):
        for index, (group, power) in enumerate(POWERS):
            selected = [row for row in rows if row["power"] == power and row["mode"] == mode]
            axis.loglog([row["nodes"] ** (1 / 3) for row in selected],
                        [row["final_rho_L2"] for row in selected],
                        color=f"C{index}", marker="o", label=f"p={power:g}")
        axis.set_title(mode)
        axis.set_xlabel(r"$N^{1/3}$")
        axis.grid(True, which="both", alpha=0.2)
    axes[0].set_ylabel("Volume-weighted density L2 at t=1.6")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(OUTPUT / "convergence.png", dpi=180)
    figure.savefig(OUTPUT / "convergence.pdf")

    order_figure, order_axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    power_values = [power for _, power in POWERS]
    for axis, mode in zip(order_axes, MODES):
        for levels, style, marker in (("20-40", "--", "s"), ("40-80", "-", "o")):
            values = [next(row["observed_order"] for row in orders
                           if row["mode"] == mode and row["power"] == power
                           and row["levels"] == levels)
                      for power in power_values]
            axis.plot(power_values, values, linestyle=style, marker=marker,
                      label=f"IV{levels}")
        axis.set_title(mode)
        axis.set_xlabel("Distance-weight exponent p")
        axis.set_xticks(power_values)
        axis.grid(True, alpha=0.25)
    order_axes[0].set_ylabel("Observed density L2 order")
    order_axes[1].legend()
    order_figure.tight_layout()
    order_figure.savefig(OUTPUT / "order_vs_power.png", dpi=180)
    order_figure.savefig(OUTPUT / "order_vs_power.pdf")

    for mode in MODES:
        print(mode)
        for _, power in POWERS:
            selected = [row for row in rows if row["mode"] == mode and row["power"] == power]
            observed = [row["observed_order"] for row in orders
                        if row["mode"] == mode and row["power"] == power]
            print(power, "errors", *(f"{row['final_rho_L2']:.9e}" for row in selected),
                  "orders", *(f"{value:.5f}" for value in observed))


if __name__ == "__main__":
    main()
