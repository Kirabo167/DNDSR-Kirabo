"""Validate planar entropy-wave means and every adjacent refinement order."""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


ORIGINAL = {
    8: {"L1": 6.20795449e-3, "L2": 6.98363869e-3, "Linf": 1.28962523e-2},
    16: {"L1": 6.66745021e-4, "L2": 8.22105537e-4, "Linf": 5.35327007e-3},
    32: {"L1": 2.30679452e-4, "L2": 2.72162841e-4, "Linf": 2.78801545e-3},
}


def norms(delta, volume):
    return {"L1": float(np.sum(volume * np.abs(delta)) / volume.sum()),
            "L2": float(np.sqrt(np.sum(volume * delta**2) / volume.sum())),
            "Linf": float(np.max(np.abs(delta)))}


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def load_case(raw, n):
    stem = f"entropy2d_n{n}.json"
    meta = json.loads((raw / stem).read_text())

    def field(label):
        paths = sorted(raw.glob(stem + f".{label}.rank*.csv"))
        if len(paths) != meta["ranks"]:
            raise RuntimeError(f"Missing owned fields: {raw / stem}")
        a = np.concatenate([np.atleast_2d(np.loadtxt(p, delimiter=",", skiprows=1)) for p in paths])
        a = a[np.argsort(a[:, 0])]
        if a.shape != (meta["nodes"], 11) or not np.array_equal(a[:, 0], np.arange(len(a))):
            raise RuntimeError("Incomplete or duplicate owned fields")
        if not np.isfinite(a).all() or a[:, 4].min() <= 0:
            raise RuntimeError("Invalid conservative field")
        return a

    initial, final = field("initial"), field("final")
    if not np.array_equal(initial[:, :5], final[:, :5]) or not np.allclose(final[:, 5], .5, atol=1e-14, rtol=0):
        raise RuntimeError("Geometry/time mismatch")
    if not np.isclose(final[:, 4].sum(), 4 * math.pi**2 * 4 * math.pi / n):
        raise RuntimeError("Volume mismatch")
    quad_paths = sorted(raw.glob(stem + ".quadrature.rank*.bin"))
    if len(quad_paths) != meta["ranks"]:
        raise RuntimeError("Missing quadrature output")
    density_integrals = np.zeros((len(final), 2))
    weights = np.zeros(len(final))
    for path in quad_paths:
        with path.open("rb") as stream:
            if stream.read(8) != b"HOQv0001":
                raise RuntimeError("Quadrature format mismatch")
        q_all = np.fromfile(path, dtype="<f8", offset=8).reshape(-1, 5)
        for start in range(0, len(q_all), 65536):
            q = q_all[start:start + 65536]
            node = q[:, 0].astype(np.int64)
            weights += np.bincount(node, weights=q[:, 4], minlength=len(final))
            for k, time in enumerate((0, .5)):
                rho = 1 + .2 * np.sin(q[:, 1] + q[:, 2] - 1.5 * time)
                density_integrals[:, k] += np.bincount(node, weights=q[:, 4] * rho, minlength=len(final))
    volume = final[:, 4]
    density_means = density_integrals / volume[:, None]
    exact = np.column_stack((density_means[:, 1], density_means[:, 1], .5 * density_means[:, 1],
                            np.zeros(len(final)), 2.5 + .625 * density_means[:, 1]))
    exact_initial = np.column_stack((density_means[:, 0], density_means[:, 0], .5 * density_means[:, 0],
                                    np.zeros(len(final)), 2.5 + .625 * density_means[:, 0]))
    init_error = float(np.max(np.abs(initial[:, 6:] - exact_initial)))
    volume_error = float(np.max(np.abs(weights / volume - 1)))
    if init_error > 1e-11 or volume_error > 1e-11:
        raise RuntimeError("Initial/quadrature validation failed")
    error = final[:, 6] - density_means[:, 1]
    calculated = norms(error, volume)
    for key, value in calculated.items():
        if not np.isclose(value, meta[f"density_{key}"], rtol=1e-10, atol=1e-14):
            raise RuntimeError(f"Independent mean norm mismatch: {key}")
    velocity = final[:, 7:10] / final[:, 6:7]
    pressure = .4 * (final[:, 10] - .5 * np.sum(final[:, 7:10]**2, axis=1) / final[:, 6])
    if final[:, 6].min() <= 0 or pressure.min() <= 0:
        raise RuntimeError("Nonphysical final mean")
    h = 2 * math.pi / n
    xy = final[:, 1:3]
    strip = np.any((xy <= h + 1e-12) | (xy >= 2 * math.pi - h - 1e-12), axis=1)
    total_squared = np.sum(volume * error**2)
    region_rows = []
    for name, mask in (("periodic_xy_strip", strip), ("outside_strip", ~strip)):
        region_rows.append({"region": name, "volume_fraction": float(volume[mask].sum() / volume.sum()),
                            "squared_error_fraction": float(np.sum(volume[mask] * error[mask]**2) / total_squared),
                            **norms(error[mask], volume[mask])})
    result = {"n": n, "nodes": meta["nodes"], "ranks": meta["ranks"], "dt": meta["dt"], "steps": meta["steps"],
              "sweeps": meta["sweeps"], "density_errors": calculated,
              "conservative_errors": {name: norms(final[:, 6 + k] - exact[:, k], volume)
                                      for k, name in enumerate(("rho", "rhou", "rhov", "rhow", "rhoE"))},
              "maximum_initial_projection_error": init_error, "maximum_quadrature_volume_relative_error": volume_error,
              "maximum_velocity_deviation": float(np.max(np.abs(velocity - np.array([1, .5, 0])))),
              "maximum_pressure_deviation": float(np.max(np.abs(pressure - 1))),
              "minimum_density": float(final[:, 6].min()), "minimum_pressure": float(pressure.min()),
              "fallbacks": meta["mean_fallback_count"], "last_coupled_residual": meta["last_coupled_residual"],
              "higher_reference_L2": meta["density_L2_higher_reference"],
              "reference_8_24_difference_L2": meta["reference_quadrature_difference_L2"],
              "conserved_integral_drift": np.sum(volume[:, None] * (final[:, 6:] - initial[:, 6:]), axis=0).tolist(),
              "regions": region_rows}
    (raw / f"entropy2d_n{n}.metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    return result, final


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = args.output.resolve()
    metrics, fields = {}, {}
    for run, levels in (("main", (8, 16, 32, 64, 80)), ("serial", (8, 16, 32)), ("dt_half", (32,))):
        for n in levels:
            if not (root / run / "raw" / f"entropy2d_n{n}.json").exists():
                if run == "main":
                    raise RuntimeError(f"Missing primary case: {n}")
                continue
            metrics[run, n], fields[run, n] = load_case(root / run / "raw", n)
    rows, regions = [], []
    levels = (8, 16, 32, 64, 80)
    for i, n in enumerate(levels):
        m = metrics["main", n]
        row = {"n": n, "nodes": m["nodes"], "dt": m["dt"], **m["density_errors"]}
        for norm in ("L1", "L2", "Linf"):
            row[f"order_{norm}"] = math.log(metrics["main", levels[i - 1]]["density_errors"][norm] / row[norm]) / math.log(n / levels[i - 1]) if i else None
        rows.append(row)
        regions.extend({"n": n, **r} for r in m["regions"])
    write_csv(root / "convergence.csv", rows)
    write_csv(root / "regions.csv", regions)
    source_rows = []
    for n in (8, 16, 32):
        if ("serial", n) not in metrics:
            continue
        for norm in ("L1", "L2", "Linf"):
            value = metrics["serial", n]["density_errors"][norm]
            source_rows.append({"n": n, "norm": norm, "original_project": ORIGINAL[n][norm],
                                "local_serial": value, "relative_difference": value / ORIGINAL[n][norm] - 1})
    if source_rows:
        write_csv(root / "original_comparison.csv", source_rows)
    controls = []
    for run, levels_control in (("serial", (8, 16, 32)), ("dt_half", (32,))):
        for n in levels_control:
            if (run, n) not in fields:
                continue
            baseline = "serial" if run == "dt_half" else "main"
            if (baseline, n) not in fields:
                raise RuntimeError("Missing same-partition baseline for time control")
            a, b = fields[baseline, n], fields[run, n]
            if not np.array_equal(a[:, :5], b[:, :5]):
                raise RuntimeError("Control geometry mismatch")
            gap = norms(a[:, 6] - b[:, 6], a[:, 4])["L2"]
            controls.append({"control": run, "baseline_run": baseline, "n": n, "density_two_solution_L2": gap,
                             "fraction_of_baseline_density_error": gap / metrics[baseline, n]["density_errors"]["L2"]})
    if controls:
        write_csv(root / "controls.csv", controls)
    (root / "analysis.json").write_text(json.dumps({"convergence": rows, "metrics": list(metrics.values()),
                                                   "original_comparison": source_rows, "controls": controls}, indent=2) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    for name in ("L1", "L2", "Linf"):
        ax.loglog(levels, [row[name] for row in rows], "o-", label=name)
    ax.loglog(levels, rows[1]["L2"] * (16 / np.asarray(levels, dtype=float))**3,
              "k--", linewidth=1, alpha=.5, label="N^-3 guide")
    ax.set_xticks(levels, labels=[str(n) for n in levels])
    ax.set(xlabel="Planar mesh level N", ylabel="Density control-volume mean error",
           title="Entropy wave, midpoint Roe, t=0.5")
    ax.legend()
    ax.grid(True, which="both", alpha=.25)
    fig.savefig(root / "convergence.png", dpi=180)
    print(json.dumps(rows, indent=2))
    print(json.dumps(controls, indent=2))


if __name__ == "__main__":
    main()
