"""Independently audit the saved NCFV vortex solution and generate report figures."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np


def load_nodes(prefix, step):
    files = sorted(prefix.parent.glob(f"{prefix.name}_{step:08d}.nodes.rank*.csv"))
    if not files:
        raise FileNotFoundError(f"No nodal output for step {step}")
    data = np.concatenate([np.atleast_1d(np.genfromtxt(f, delimiter=",", names=True)) for f in files])
    data.sort(order="original_node")
    if len(np.unique(data["original_node"])) != len(data):
        raise ValueError("Duplicate original node ownership in output")
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prefix", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    diagnostics = np.atleast_1d(np.genfromtxt(str(args.prefix) + ".diagnostics.csv", delimiter=",", names=True))
    first, last = diagnostics[0], diagnostics[-1]
    initial = load_nodes(args.prefix, int(first["iteration"]))
    final = load_nodes(args.prefix, int(last["iteration"]))
    volume = final["partial_volume"].sum()
    error = final["rho_point"] - final["rho_exact"]
    l1 = np.sum(final["partial_volume"] * np.abs(error)) / volume
    l2 = np.sqrt(np.sum(final["partial_volume"] * error**2) / volume)
    linf = np.max(np.abs(error))
    if not np.isclose(l2, last["rho_point_L2"], rtol=1e-12, atol=1e-15):
        raise ValueError("Independent density-error reduction differs from solver diagnostics")
    conserved = ["mass", "momentum_x", "momentum_y", "momentum_z", "total_energy"]
    drift = {key: float(last[key] - first[key]) for key in conserved}
    relative = {key: float(abs(drift[key]) / max(abs(first[key]), 1.0)) for key in conserved}

    coordinates = np.column_stack([final[k] for k in ("x", "y", "z")])
    lengths = np.array([10, 10, 4])
    wrapped = coordinates.copy()
    for d, length in enumerate(lengths):
        wrapped[np.isclose(wrapped[:, d], length, rtol=0, atol=1e-8), d] = 0
    _, groups = np.unique(np.round(wrapped / 1e-8).astype(np.int64), axis=0, return_inverse=True)
    periodic_error = 0.0
    for field in ("rho_mean", "rhou", "rhov", "rhow", "rhoE"):
        minimum = np.full(groups.max() + 1, np.inf)
        maximum = np.full(groups.max() + 1, -np.inf)
        np.minimum.at(minimum, groups, final[field])
        np.maximum.at(maximum, groups, final[field])
        periodic_error = max(periodic_error, float(np.max(maximum - minimum)))

    stats = {
        "steps": int(last["iteration"]), "time": float(last["time"]), "period": 10.0,
        "period_fraction": float(last["time"] / 10), "nodes": len(final),
        "periodic_unknowns": int(groups.max() + 1), "volume": float(volume),
        "density_point_L1": float(l1), "density_point_L2": float(l2), "density_point_Linf": float(linf),
        "initial_density_point_L2": float(first["rho_point_L2"]),
        "density_mean_min": float(final["rho_mean"].min()), "density_mean_max": float(final["rho_mean"].max()),
        "density_point_min": float(final["rho_point"].min()), "density_point_max": float(final["rho_point"].max()),
        "conserved_initial": {key: float(first[key]) for key in conserved},
        "conserved_final": {key: float(last[key]) for key in conserved},
        "conservation_absolute_drift": drift, "conservation_scaled_drift": relative,
        "periodic_state_max_difference": periodic_error, "entropy_L1": float(last["entropy_L1"]),
    }
    (args.output / "metrics.json").write_text(json.dumps(stats, indent=2) + "\n")

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "figure.dpi": 140})
    level = np.unique(final["z"])[np.argmin(np.abs(np.unique(final["z"]) - 2))]
    cut = final[np.isclose(final["z"], level, rtol=0, atol=1e-9)]
    tri = mtri.Triangulation(cut["x"], cut["y"])
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.8), constrained_layout=True)
    for ax, field, title in zip(axes[:2], ("rho_point", "rho_exact"), ("Computed density", "Translated analytic density")):
        plot = ax.tricontourf(tri, cut[field], levels=np.linspace(0.45, 1.01, 29), cmap="viridis", extend="both")
        fig.colorbar(plot, ax=ax, shrink=0.8)
        ax.set_title(title)
    err = cut["rho_point"] - cut["rho_exact"]
    extent = max(float(np.max(np.abs(err))), 1e-12)
    plot = axes[2].tricontourf(tri, err, levels=np.linspace(-extent, extent, 25), cmap="RdBu_r", extend="both")
    fig.colorbar(plot, ax=axes[2], shrink=0.8, format="%.1e")
    axes[2].set_title("Pointwise density error")
    for ax in axes:
        ax.set(xlim=(0, 10), ylim=(0, 10), xlabel="x", ylabel="y", aspect="equal")
    fig.suptitle(f"NCFV / iv40: t={stats['time']:.6g}, z={level:g}")
    fig.savefig(args.output / "density_comparison.png", dpi=180)
    fig.savefig(args.output / "density_comparison.pdf")
    plt.close(fig)

    center = 5.0 + stats["time"]
    line_x = np.linspace(max(0, center - 3.5), min(10, center + 3.5), 500)
    interpolator = mtri.LinearTriInterpolator(tri, cut["rho_point"])
    computed = interpolator(line_x, np.full_like(line_x, center))
    x = line_x - center
    analytic = (1 - 0.4 * 25 / (8 * 1.4 * np.pi**2) * np.exp(1 - x*x))**2.5
    fig, ax = plt.subplots(figsize=(7, 3.6), constrained_layout=True)
    ax.plot(line_x, analytic, "k-", label="Analytic")
    ax.plot(line_x, computed, "--", label="Computed (linear interpolation of node points)")
    ax.set(xlabel="x", ylabel="Density", title=f"Through vortex center: y={center:g}, z={level:g}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    fig.savefig(args.output / "density_profile.png", dpi=180)
    fig.savefig(args.output / "density_profile.pdf")
    plt.close(fig)

    def number(value):
        return f"{value:.6e}"

    macros = {
        "RunSteps": str(stats["steps"]), "RunTime": f"{stats['time']:.12g}",
        "RunLone": number(l1), "RunLtwo": number(l2), "RunLinf": number(linf),
        "RunInitialLtwo": number(stats["initial_density_point_L2"]),
        "RunMassDrift": number(relative["mass"]), "RunEnergyDrift": number(relative["total_energy"]),
        "RunPeriodicMismatch": number(periodic_error), "RunEntropy": number(stats["entropy_L1"]),
        "RunDensityMin": number(stats["density_point_min"]), "RunDensityMax": number(stats["density_point_max"]),
    }
    (args.output / "metrics.tex").write_text("\n".join("\\newcommand{\\" + key + "}{" + value + "}" for key, value in macros.items()) + "\n")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
