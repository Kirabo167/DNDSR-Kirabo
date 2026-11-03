"""Create a compact, qualified comparison of the completed small-grid tests."""
import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/ncfv-efficient-derivation-mpl")
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    directory = args.root / "diagnostics/preflight"
    data = json.loads((directory / "results.json").read_text())
    if data["status"] != "complete" or len(data["runs"]) != 20:
        raise RuntimeError("Only a fully completed 20-run preflight matrix may be summarized")
    rows = []
    names = [v["name"] for v in data["variants"]]
    for name in names:
        runs = [r for r in data["runs"] if r["variant"] == name]
        by_problem = {r["problem"]: r for r in runs if r["problem"] != "smooth"}
        rp, shock, contact = (by_problem[k] for k in ("RP1", "planar_shock", "contact"))
        rows.append({"name": name, "RP1_rho_z_span": rp["rho_z_span_max"], "RP1_p_z_span": rp["p_z_span_max"],
                     "RP1_max_abs_w": rp["w_abs_max"], "RP1_total_seconds": rp["elapsed_seconds"],
                     "planar_shock_rho_overshoot": shock["rho_overshoot_exact_states"],
                     "planar_shock_rho_L1": shock["rho_L1_exact_dualmean"], "contact_rho_L1": contact["rho_L1"],
                     "smooth_observed_order_20_to_40": data["smooth_observed_L1_order_20_to_40"][name],
                     "all_saved_states_physical": all(r["rho_min"] > 0 and r["p_min"] > 0 for r in runs),
                     "total_rejected_steps": sum(r["rejected_steps"] for r in runs)})
    result = {"scope": "Small-grid evidence only, not 400x400 production results",
              "grid": "RP1/shock/contact 40x40x5; smooth 20/40x20/40x5", "target_time": .25,
              "physical_depth": .125, "mpi_ranks": 4, "maximum_time_step": .001,
              "caveats": ["Fixed depth differs from production Lz=0.05; aspect ratio also differs.",
                           "Two-grid convergence at fixed dt and fixed z resolution is not a general accuracy proof.",
                           "Total runtime includes initialization; not directly a per-step limiter cost.",
                           "Complete-flux candidates use first-order LLF open boundaries."], "rows": rows}
    (directory / "small_grid_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    with (directory / "small_grid_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    rp_runs = [r for r in data["runs"] if r["problem"] == "RP1"]
    arrays = []
    for r in rp_runs:
        with h5py.File(r["restart"]) as f:
            u = f["state/father/array/data"][:].reshape(-1, 5)
            ids = f["state/origIndex"][:].astype(int)
        q = np.empty_like(u)
        q[ids] = u
        q = q.reshape(5, 41, 41, 5)
        rho = q[..., 0]
        p = .4 * (q[..., 4] - .5 * np.sum(q[..., 1:4]**2, axis=-1) / rho)
        arrays.append((rho, p))
    for k, field in enumerate(("density", "pressure")):
        lo, hi = min(float(a[k].min()) for a in arrays), max(float(a[k].max()) for a in arrays)
        levels = np.linspace(lo, hi, 22)[1:-1]
        x = np.linspace(0, 1, 41)
        fig, axes = plt.subplots(1, 4, figsize=(18, 4.8), layout="constrained")
        for ax, r, a in zip(axes, rp_runs, arrays):
            im = ax.pcolormesh(x, x, a[k][2], cmap="coolwarm", shading="nearest", vmin=lo, vmax=hi)
            valid = levels[(levels > a[k][2].min()) & (levels < a[k][2].max())]
            ax.contour(x, x, a[k][2], levels=valid, colors="black", linewidths=.45, alpha=.6)
            ax.set(title=r["variant"], xlabel="x", ylabel="y", aspect="equal")
        fig.colorbar(im, ax=axes, shrink=.8, label=field)
        fig.suptitle(f"RP1 SMALL-GRID comparison: 40 x 40 x 5, t=0.25, z=0.05, Lz=0.125\nCommon color limits; 20 contour levels; production runs are separate")
        fig.savefig(directory / f"RP1_{field}_20contours.png", dpi=180)
        plt.close(fig)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
