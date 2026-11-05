"""Analyze the complete-reference entropy diagnostics and Taylor--Green fields.

This script reads standalone solver CSV/JSON outputs, not DNDSR Python bindings.
Taylor--Green differences compare recovered point values on shared coordinates;
control-volume means from different meshes are deliberately not compared.
"""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


LEVELS = (10, 20, 40, 80)
FIELDS = {"velocity": slice(4, 7), "density": slice(3, 4), "pressure": slice(7, 8)}


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def load_fields(raw, problem, level, mean=False):
    stem = f"{problem}_iv{level}.json"
    meta = json.loads((raw / stem).read_text())
    suffix = ".mean.rank*.csv" if mean else ".rank*.csv"
    paths = sorted(raw.glob(stem + suffix))
    if len(paths) != meta["ranks"]:
        raise RuntimeError(f"Missing fields for {raw / stem}")
    table = np.concatenate([np.atleast_2d(np.loadtxt(p, delimiter=",", skiprows=1))
                            for p in paths])
    if table.shape != (meta["nodes"], 7 if mean else 9):
        raise RuntimeError(f"Wrong field shape: {table.shape}")
    if not np.isfinite(table).all():
        raise RuntimeError("Nonfinite field")
    volume = table[:, 5 if mean else 8]
    if np.min(volume) <= 0 or not np.isclose(volume.sum(), 400):
        raise RuntimeError("Invalid owned volume")
    keys = [tuple(row) for row in np.rint(table[:, :3] * 1e8).astype(np.int64)]
    mapping = dict(zip(keys, range(len(keys))))
    if len(mapping) != len(keys):
        raise RuntimeError("Duplicate owned point coordinates")
    return meta, table, mapping


def regions(coords):
    z = np.isclose(coords[:, 2], 0) | np.isclose(coords[:, 2], 4)
    xy = np.any(np.isclose(coords[:, :2], 0) | np.isclose(coords[:, :2], 10), axis=1)
    return {"all": np.ones(len(coords), dtype=bool), "z_boundary": z,
            "other_boundary": xy & ~z, "interior": ~(xy | z),
            "z_only": z & ~xy, "z_xy_intersection": z & xy}


def difference(a, b, weights, field, mask):
    delta = a[mask, field] - b[mask, field]
    return float(np.sqrt(np.sum(weights[mask] * np.sum(delta**2, axis=1)) /
                         weights[mask].sum()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = args.output.resolve()
    summary = {}
    rhs_rows, polynomial_rows = [], []
    for level in LEVELS:
        data = json.loads((root / "rhs/raw" / f"entropy_iv{level}.json").read_text())
        for name, result in data["schemes"].items():
            for region, metrics in result.items():
                rhs_rows.append({"level": level, "scheme": name, "region": region,
                                 **metrics})
        for name, result in zip(("x", "y", "z", "xz"), data["polynomial_consistency"]):
            for method, metrics in result.items():
                polynomial_rows.append({"level": level, "polynomial": name,
                                        "method": method, **metrics["all"]})
    write_csv(root / "rhs_norms.csv", rhs_rows)
    write_csv(root / "polynomial_consistency.csv", polynomial_rows)
    summary["rhs"] = rhs_rows
    summary["polynomial_consistency"] = polynomial_rows

    entropy_rows = []
    old = root.parent / "hocfv_complete_port_20261010/main_halo/raw"
    for variant, raw in (("midpoint", old), ("moment_corrected", root / "entropy_corrected/raw")):
        previous = None
        for level in LEVELS:
            meta = json.loads((raw / f"entropy_iv{level}.json").read_text())
            row = {"variant": variant, "level": level, "nodes": meta["nodes"],
                   "dt": meta["dt"], "steps": meta["steps"],
                   "rhs_evaluations": meta["rhs_evaluations"],
                   "last_rhs_riemann_calls": meta["last_rhs_riemann_calls"],
                   "fallback_count": meta["mean_fallback_count"],
                   "mass_drift": meta["mass_drift"]}
            if meta["rhs_evaluations"] != 3 * meta["steps"] or row["fallback_count"] != 0:
                raise RuntimeError("Unexpected march/fallback")
            for norm in ("L1", "L2", "Linf"):
                value = meta[f"density_{norm}"]
                row[norm] = value
                row[f"order_{norm}"] = math.log2(previous[norm] / value) if previous else None
            entropy_rows.append(row)
            previous = row
    write_csv(root / "entropy_convergence.csv", entropy_rows)
    summary["entropy"] = entropy_rows

    region_rows = []
    for variant, raw in (("midpoint", root / "entropy_fields/raw"),
                         ("moment_corrected", root / "entropy_corrected/raw")):
        for level in (40, 80):
            meta, table, _ = load_fields(raw, "entropy", level, mean=True)
            delta, volume = table[:, 3] - table[:, 4], table[:, 5]
            total_squared = np.sum(volume * delta**2)
            computed = np.sqrt(total_squared / volume.sum())
            if not np.isclose(computed, meta["density_L2"], rtol=1e-11, atol=1e-15):
                raise RuntimeError("Mean field norm disagrees with solver output")
            for name, mask in regions(table[:, :3]).items():
                weight, error = volume[mask], delta[mask]
                region_rows.append({"variant": variant, "level": level, "region": name,
                                    "nodes": int(mask.sum()), "volume": float(weight.sum()),
                                    "volume_fraction": float(weight.sum() / volume.sum()),
                                    "L1": float(np.sum(weight * np.abs(error)) / weight.sum()),
                                    "L2": float(np.sqrt(np.sum(weight * error**2) / weight.sum())),
                                    "Linf": float(np.max(np.abs(error))),
                                    "squared_error_share": float(np.sum(weight * error**2) / total_squared)})
    write_csv(root / "entropy_regions.csv", region_rows)
    summary["entropy_regions"] = region_rows

    cases = {n: load_fields(root / "tgv/raw", "tgv", n) for n in LEVELS}
    common = sorted(set.intersection(*(set(cases[n][2]) for n in LEVELS)))
    if len(common) < 100:
        raise RuntimeError("Insufficient shared TGV points")
    tables = [cases[n][1][[cases[n][2][key] for key in common]] for n in LEVELS]
    weights = tables[0][:, 8]
    tgv_rows = []
    for region, mask in regions(tables[0][:, :3]).items():
        for name, field in FIELDS.items():
            ds = [difference(tables[i], tables[i + 1], weights, field, mask) for i in range(3)]
            tgv_rows.append({"region": region, "field": name, "shared_nodes": int(mask.sum()),
                             "difference_10_20": ds[0], "difference_20_40": ds[1],
                             "difference_40_80": ds[2],
                             "order_10_20_40": math.log2(ds[0] / ds[1]),
                             "order_20_40_80": math.log2(ds[1] / ds[2])})
    write_csv(root / "tgv_self_convergence.csv", tgv_rows)
    summary["tgv"] = tgv_rows
    summary["tgv_common_coordinates"] = len(common)
    summary["tgv_common_IV10_volume"] = float(weights.sum())
    summary["tgv_diagnostics"] = [cases[n][0] for n in LEVELS]

    triplet_rows = []
    for triplet in (LEVELS[:3], LEVELS[1:]):
        keys = sorted(set.intersection(*(set(cases[n][2]) for n in triplet)))
        sample = [cases[n][1][[cases[n][2][key] for key in keys]] for n in triplet]
        w = sample[0][:, 8]
        mask = np.ones(len(keys), dtype=bool)
        for name, field in FIELDS.items():
            ds = [difference(sample[i], sample[i + 1], w, field, mask) for i in range(2)]
            triplet_rows.append({"triplet": "-".join(map(str, triplet)), "field": name,
                                 "shared_nodes": len(keys), "coarse_difference": ds[0],
                                 "fine_difference": ds[1], "order": math.log2(ds[0] / ds[1])})
    write_csv(root / "tgv_triplet_self_convergence.csv", triplet_rows)
    summary["tgv_triplets"] = triplet_rows

    _, half, half_map = load_fields(root / "tgv_dt_half/raw", "tgv", 40)
    _, full, full_map = cases[40]
    time_rows = []
    for sample, keys in (("all_IV40", sorted(full_map)), ("four_grid_common", common)):
        a = full[[full_map[key] for key in keys]]
        b = half[[half_map[key] for key in keys]]
        w = a[:, 8] if sample == "all_IV40" else weights
        mask = np.ones(len(keys), dtype=bool)
        for name, field in FIELDS.items():
            value = difference(a, b, w, field, mask)
            fine = next(r["difference_40_80"] for r in tgv_rows if r["region"] == "all" and r["field"] == name)
            time_rows.append({"sample": sample, "field": name, "shared_nodes": len(keys),
                              "dt_half_L2_difference": value,
                              "fraction_of_fine_grid_difference": value / fine})
    write_csv(root / "tgv_time_sensitivity.csv", time_rows)
    summary["tgv_time_sensitivity"] = time_rows

    _, iterated, iterated_map = load_fields(root / "tgv_sweeps60/raw", "tgv", 40)
    iterated = iterated[[iterated_map[key] for key in common]]
    scan_rows = []
    for name, field in FIELDS.items():
        value = difference(tables[2], iterated, weights, field, np.ones(len(common), dtype=bool))
        fine = next(r["difference_40_80"] for r in tgv_rows if r["region"] == "all" and r["field"] == name)
        scan_rows.append({"field": name, "scan_30_60_L2_difference": value,
                          "fraction_of_fine_grid_difference": value / fine})
    write_csv(root / "tgv_scan_sensitivity.csv", scan_rows)
    summary["tgv_scan_sensitivity"] = scan_rows

    base = json.loads((root / "entropy_corrected/raw/entropy_iv40.json").read_text())
    half_meta = json.loads((root / "entropy_corrected_dt_half/raw/entropy_iv40.json").read_text())
    summary["entropy_corrected_dt_half"] = {
        norm: {"baseline": base[f"density_{norm}"], "half_dt": half_meta[f"density_{norm}"],
               "relative_change": half_meta[f"density_{norm}"] / base[f"density_{norm}"] - 1}
        for norm in ("L1", "L2", "Linf")}
    (root / "analysis.json").write_text(json.dumps(summary, indent=2) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for variant, label in (("midpoint", "Edge midpoint"), ("moment_corrected", "Entropy flux moments")):
        values = [r["L2"] for r in entropy_rows if r["variant"] == variant]
        axes[0].loglog(LEVELS, values, "o-", label=label)
    axes[0].set(xlabel="Prism mesh level", ylabel="Density mean L2 error", title="Entropy wave, t=1.6")
    axes[0].legend()
    for name in FIELDS:
        row = next(r for r in tgv_rows if r["region"] == "all" and r["field"] == name)
        axes[1].loglog((10, 20, 40), [row[f"difference_{a}_{b}"] for a, b in zip(LEVELS[:-1], LEVELS[1:])], "o-", label=name)
    axes[1].set(xlabel="Coarser level in each grid pair", ylabel="Shared-point L2 difference", title="Taylor--Green self-convergence, t=0.4")
    axes[1].legend()
    for ax in axes:
        ax.grid(True, which="both", alpha=0.25)
    fig.savefig(root / "convergence.png", dpi=180)
    print(json.dumps({k: summary[k] for k in ("entropy", "tgv_common_coordinates", "tgv_time_sensitivity")}, indent=2))
    print(json.dumps([r for r in tgv_rows if r["region"] == "all"], indent=2))


if __name__ == "__main__":
    main()
