"""Compare surface-quadrature and midpoint Roe fluxes on the same vortex chain."""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


LEVELS = (10, 20, 40, 80)


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def load_means(raw, level, meta):
    paths = sorted(raw.glob(f"vortex_iv{level}.json.mean.rank*.csv"))
    if len(paths) != meta["ranks"]:
        raise RuntimeError("Missing rank fields")
    table = np.concatenate([np.atleast_2d(np.loadtxt(p, delimiter=",", skiprows=1)) for p in paths])
    if table.shape != (meta["nodes"], 7) or not np.isfinite(table).all():
        raise RuntimeError("Invalid owned mean fields")
    volume, delta = table[:, 5], table[:, 3] - table[:, 4]
    if volume.min() <= 0 or not np.isclose(volume.sum(), 400):
        raise RuntimeError("Incorrect volume")
    norms = {"L1": np.sum(volume * np.abs(delta)) / volume.sum(),
             "L2": np.sqrt(np.sum(volume * delta**2) / volume.sum()),
             "Linf": np.max(np.abs(delta))}
    for key, value in norms.items():
        if not np.isclose(value, meta[f"density_{key}"], rtol=1e-10, atol=1e-14):
            raise RuntimeError(f"Owned field/JSON error mismatch: {key}")
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = args.output.resolve()
    cases, rows, checks, tables = {}, [], [], {}
    for mode in ("surface6", "midpoint"):
        previous = None
        for level in LEVELS:
            path = root / mode / "raw" / f"vortex_iv{level}.json"
            meta = json.loads(path.read_text())
            if meta["surface_flux_mode"] != mode or meta["problem"] != "vortex":
                raise RuntimeError("Wrong problem or flux mode")
            if meta["sweeps"] != 30 or meta["variational_weight"] != 1 or meta["ranks"] != 8:
                raise RuntimeError("Wrong reconstruction/partition settings")
            if meta["dt"] != 2 / level or meta["end_time"] != 1.6 or meta["rhs_evaluations"] != 3 * meta["steps"]:
                raise RuntimeError("Wrong time integration settings")
            if meta.get("entropy_moment_correction", False):
                raise RuntimeError("Unexpected entropy correction")
            cases[mode, level] = meta
            tables[mode, level] = load_means(path.parent, level, meta)
            row = {"mode": mode, "level": level, "nodes": meta["nodes"], "dt": meta["dt"]}
            for norm in ("L1", "L2", "Linf"):
                row[norm] = meta[f"density_{norm}"]
                row[f"order_{norm}"] = math.log2(previous[norm] / row[norm]) if previous else None
            rows.append(row)
            previous = row
            checks.append({"mode": mode, "level": level, "volume": meta["total_volume"],
                           "fallbacks": meta["mean_fallback_count"], "mass_drift": meta["mass_drift"],
                           "coupled_residual": meta["last_coupled_residual"],
                           "reference_8_24_L2": meta["reference_quadrature_difference_L2"],
                           "L2_24_point_reference": meta["density_L2_higher_reference"],
                           "last_rhs_riemann_calls": meta["last_rhs_riemann_calls"],
                           "march_seconds": meta["march_seconds"]})
    comparison = []
    for level in LEVELS:
        a, b = cases["surface6", level], cases["midpoint", level]
        ta, tb = tables["surface6", level], tables["midpoint", level]
        if not np.array_equal(ta[:, :3], tb[:, :3]) or not np.array_equal(ta[:, 4:], tb[:, 4:]):
            raise RuntimeError("Different geometry or analytic reference fields")
        for key in ("initial_integrals", "maximum_volume_mismatch", "maximum_moment_mismatch", "maximum_matrix_condition"):
            if a[key] != b[key]:
                raise RuntimeError(f"Paired setup differs: {key}")
        row = {"level": level}
        for norm in ("L1", "L2", "Linf"):
            row[f"surface6_{norm}"] = a[f"density_{norm}"]
            row[f"midpoint_{norm}"] = b[f"density_{norm}"]
            row[f"surface6_over_midpoint_{norm}"] = a[f"density_{norm}"] / b[f"density_{norm}"]
        row["riemann_call_ratio"] = a["last_rhs_riemann_calls"] / b["last_rhs_riemann_calls"]
        comparison.append(row)
    write_csv(root / "convergence.csv", rows)
    write_csv(root / "comparison.csv", comparison)
    write_csv(root / "checks.csv", checks)

    archived = root.parent / "hocfv_complete_port_20261010/main_halo/raw"
    regression = []
    for level in LEVELS:
        old = json.loads((archived / f"vortex_iv{level}.json").read_text())
        new = cases["midpoint", level]
        row = {"level": level}
        for norm in ("L1", "L2", "Linf"):
            delta = new[f"density_{norm}"] - old[f"density_{norm}"]
            row[f"difference_{norm}"] = delta
            if abs(delta) > 1e-10 * old[f"density_{norm}"] + 1e-14:
                raise RuntimeError("Midpoint baseline changed materially")
        regression.append(row)
    write_csv(root / "midpoint_regression.csv", regression)
    summary = {"convergence": rows, "comparison": comparison, "checks": checks, "regression": regression}
    (root / "analysis.json").write_text(json.dumps(summary, indent=2) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    for ax, norm in zip(axes, ("L1", "L2", "Linf")):
        for mode, label in (("surface6", "Surface quadrature"), ("midpoint", "Edge midpoint")):
            selected = [r for r in rows if r["mode"] == mode]
            ax.loglog(LEVELS, [r[norm] for r in selected], "o-", label=label)
        ax.set(xlabel="Prism mesh level", ylabel=f"Density mean {norm} error")
        ax.set_xticks(LEVELS, labels=[str(n) for n in LEVELS])
        ax.grid(True, which="both", alpha=.25)
        ax.legend(fontsize=8)
    fig.suptitle("Isentropic vortex, t=1.6; same quadratic VR and SSPRK3")
    fig.savefig(root / "comparison.png", dpi=180)
    print(json.dumps(rows, indent=2))
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
