"""Summarize the raw Riemann audit; keep old and corrected HLLC distinct."""
import argparse
import csv
import json
import math
from pathlib import Path


def read_cases(directory):
    cases = {}
    for path in directory.glob("*.json"):
        if path.name.endswith(".manifest.json"):
            continue
        raw = json.loads(path.read_text())
        level = int(path.stem.split("_")[1])
        key = (raw["mode"], raw["method"], raw["riemann_solver"], level)
        cases[key] = raw
    return cases


def write_csv(path, rows):
    if rows:
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)


def compare_fields(output):
    import numpy as np
    def read(directory):
        files = sorted(directory.glob("solution_00000016.nodes.rank*.csv"))
        if not files:
            return None
        return np.sort(np.concatenate([np.genfromtxt(path, delimiter=",", names=True)
                                       for path in files]), order="original_node")
    rows = []
    def compare(mode, label, left_path, right_path):
        left, right = read(left_path), read(right_path)
        if left is None or right is None:
            return
        if not np.array_equal(left["original_node"], right["original_node"]):
            raise ValueError("Mismatched node identities in field comparison")
        for name in ["x", "y", "z", "partial_volume"]:
            if not np.allclose(left[name], right[name], rtol=1e-12, atol=1e-12):
                raise ValueError(f"Mismatched {name} in field comparison")
        difference = left["rho_point"] - right["rho_point"]
        weight = left["partial_volume"]
        rows.append(dict(mode=mode, comparison=label,
                         L2_difference=np.sqrt(np.sum(weight * difference**2) / np.sum(weight)),
                         max_difference=np.max(np.abs(difference))))
    for mode in ["Traditional", "Efficient"]:
        for ranks in [1, 8]:
            directory = output / f"partition{ranks}"
            compare(mode, f"LS_vs_SVDLS_np{ranks}",
                    directory / f"prism_20_{mode}_LeastSquares_Roe",
                    directory / f"prism_20_{mode}_SVDLeastSquares_Roe")
        for method in ["LeastSquares", "Variational"]:
            for directory, ranks in [("before", 4), ("partition8", 8)]:
                compare(mode, f"{method}_np1_vs_np{ranks}",
                        output / "partition1" / f"prism_20_{mode}_{method}_Roe",
                        output / directory / f"prism_20_{mode}_{method}_Roe")
    write_csv(output / "field_comparison.csv", rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    output = args.output
    before = read_cases(output / "before")
    fixed = read_cases(output / "fixed_hllc")
    # A missing corrected HLLC result must not silently fall back to the bugged flux.
    combined = {key: value for key, value in before.items() if key[2] != "HLLC"}
    combined.update(fixed)
    rows = []
    for key, raw in sorted(combined.items()):
        mode, method, flux, level = key
        coarse = combined.get((mode, method, flux, level // 2))
        order = ""
        if coarse:
            order = math.log(coarse["final"]["rho_point_L2V"] / raw["final"]["rho_point_L2V"]) / (
                math.log(raw["nodes"] / coarse["nodes"]) / 3)
        rows.append(dict(mode=mode, method=method, flux=flux, level=level,
                         nodes=raw["nodes"], dt=raw["dt"],
                         L2=raw["final"]["rho_point_L2V"], order=order,
                         mass_drift=raw["final"]["mass_drift"]))
    write_csv(output / "comparison.csv", rows)
    changes = []
    budgets = []
    for key, raw in sorted(fixed.items()):
        mode, method, flux, level = key
        if key in before:
            old = before[key]["final"]["rho_point_L2V"]
            new = raw["final"]["rho_point_L2V"]
            changes.append(dict(mode=mode, method=method, level=level,
                                old_L2=old, corrected_L2=new, relative_change=new / old - 1))
    for key, raw in sorted(read_cases(output / "budget").items()):
        mode, method, flux, level = key
        for time in ["initial", "final"]:
            budget = raw.get(time + "_flux_budget")
            if budget:
                for component, jump, correction in zip(budget["components"], budget["trace_jump_area_RMS"],
                                                        budget["correction_rhs_volume_RMS"]):
                    budgets.append(dict(mode=mode, method=method, level=level,
                                        time=raw[time]["time"],
                                        component=component, trace_jump=jump, correction_rhs=correction))
    write_csv(output / "hllc_fix_effect.csv", changes)
    write_csv(output / "flux_budget.csv", budgets)
    partitions = []
    for directory in ["partition1", "before", "partition8"]:
        for (mode, method, flux, level), raw in sorted(read_cases(output / directory).items()):
            if flux == "Roe" and level == 20:
                partitions.append(dict(mode=mode, method=method, ranks=raw["mpi_ranks"],
                                       initial_L2=raw["initial"]["rho_point_L2V"],
                                       final_L2=raw["final"]["rho_point_L2V"]))
    write_csv(output / "partition_comparison.csv", partitions)
    compare_fields(output)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    colors = dict(Roe="C0", HLLC="C1", HLLEP="C2", HLLEP_V1="C3", Roe_M6="C4")
    for axis, mode in zip(axes, ["Traditional", "Efficient"]):
        for method in ["LeastSquares", "Variational"]:
            for flux, color in colors.items():
                selected = [row for row in rows if row["mode"] == mode and row["method"] == method and row["flux"] == flux]
                if selected:
                    axis.loglog([row["nodes"] ** (1/3) for row in selected],
                                [row["L2"] for row in selected],
                                color=color, linestyle="-" if method == "LeastSquares" else "--",
                                marker="o" if method == "LeastSquares" else "s",
                                label=f"{flux}, {'LS' if method == 'LeastSquares' else 'VR'}")
        axis.set_title(mode)
        axis.set_xlabel(r"$N^{1/3}$")
        axis.grid(True, which="both", alpha=0.2)
    axes[0].set_ylabel("Volume-weighted density L2 error")
    axes[1].legend(fontsize=7, ncol=2)
    figure.suptitle("Periodic prism vortex, t=1.6, 4 MPI ranks; corrected HLLC")
    figure.tight_layout()
    figure.savefig(output / "convergence.png", dpi=180)
    figure.savefig(output / "convergence.pdf")
    print(f"{len(rows)} cases, {len(changes)} HLLC before/after pairs, {len(partitions)} partition checks")
    for row in rows:
        if row["level"] == 40:
            print(row["mode"], row["method"], row["flux"], f"{row['L2']:.8e}", f"{row['order']:.5f}")


if __name__ == "__main__":
    main()
