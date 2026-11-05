"""Compare the reproduced 2D HO TGV against projected spectral reference means."""

import argparse
import csv
import importlib.util
import json
import math
from pathlib import Path

import numpy as np


FIELDS = ("rho", "rhou", "rhov", "rhow", "rhoE", "u", "v", "w", "pressure_from_mean")
ORIGINAL_L2 = {
    8: {"rho": 4.05707129e-3, "u": 1.80704524e-2, "v": 1.82218642e-2,
        "w": 0.002927891127136902, "rhoE": 0.11621952967525549, "pressure_from_mean": 0.046328276964951315},
    16: {"rho": 5.35738525e-4, "u": 1.85150587e-3, "v": 1.85148423e-3,
         "w": 0.0007306953059849362, "rhoE": 0.016568244870971572, "pressure_from_mean": 0.006517793559782651},
    32: {"rho": 8.10537424e-5, "u": 1.84009542e-4, "v": 1.84466053e-4,
         "w": 0.0001832212646990316, "rhoE": 0.002061727414661929, "pressure_from_mean": 0.0008213455303593415},
}


def norms(error, volume):
    return {"L1": float(np.sum(volume * np.abs(error)) / volume.sum()),
            "L2": float(np.sqrt(np.sum(volume * error**2) / volume.sum())),
            "Linf": float(np.max(np.abs(error)))}


def fields(state):
    velocity = state[:, 1:4] / state[:, :1]
    pressure = .4 * (state[:, 4] - .5 * np.sum(state[:, 1:4]**2, axis=1) / state[:, 0])
    if np.min(state[:, 0]) <= 0 or np.min(pressure) <= 0:
        raise RuntimeError("Invalid conservative mean")
    return np.column_stack((state, velocity, pressure))


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def analyze(root, run, n, spectral):
    raw, stem = root / run / "raw", f"tgv2d_n{n}.json"
    meta = json.loads((raw / stem).read_text())

    def load(label):
        paths = sorted(raw.glob(stem + f".{label}.rank*.csv"))
        if len(paths) != meta["ranks"]:
            raise RuntimeError(f"Missing rank output: {raw / stem}")
        table = np.concatenate([np.atleast_2d(np.loadtxt(p, delimiter=",", skiprows=1)) for p in paths])
        table = table[np.argsort(table[:, 0])]
        if table.shape != (meta["nodes"], 11) or not np.array_equal(table[:, 0], np.arange(meta["nodes"])):
            raise RuntimeError("Incomplete owned conservative fields")
        if not np.isfinite(table).all():
            raise RuntimeError("Nonfinite field")
        return table

    initial, final = load("initial"), load("final")
    if not np.array_equal(initial[:, :5], final[:, :5]) or not np.allclose(final[:, 5], .2, atol=1e-14, rtol=0):
        raise RuntimeError("Geometry/time mismatch")
    volume = final[:, 4]
    if volume.min() <= 0 or not np.isclose(volume.sum(), 4 * math.pi**2 * 4 * math.pi / n):
        raise RuntimeError("Invalid owned volume")
    quad_paths = sorted(raw.glob(stem + ".quadrature.rank*.bin"))
    if len(quad_paths) != meta["ranks"]:
        raise RuntimeError("Missing quadrature ranks")
    reference_files = (root / "reference/tgv_n128_dt0.00025.npz", root / "reference/tgv_n64_dt0.0005.npz")
    references = [np.load(path)["U"] for path in reference_files]
    integrated = [np.zeros((len(volume), 5)) for _ in range(3)]
    weight_sum = np.zeros(len(volume))
    for path in quad_paths:
        with path.open("rb") as stream:
            if stream.read(8) != b"HOQv0001":
                raise RuntimeError("Wrong quadrature header")
        quad = np.fromfile(path, dtype="<f8", offset=8).reshape(-1, 5)
        for begin in range(0, len(quad), 16384):
            q = quad[begin:begin + 16384]
            owner = q[:, 0].astype(np.int64)
            weight_sum += np.bincount(owner, weights=q[:, 4], minlength=len(volume))
            x, y = q[:, 1], q[:, 2]
            u, v = np.sin(x) * np.cos(y), -np.cos(x) * np.sin(y)
            pressure = 1 / (1.4 * .09) + .25 * (np.cos(2 * x) + np.cos(2 * y))
            values = [np.column_stack((np.ones(len(x)), u, v, np.zeros(len(x)), pressure / .4 + .5 * (u*u + v*v)))]
            for reference in references:
                sampled = spectral.sample(reference, x, y)
                values.append(np.column_stack((sampled[:, :3], np.zeros(len(x)), sampled[:, 3])))
            for target, samples in zip(integrated, values):
                for k in range(5):
                    target[:, k] += np.bincount(owner, weights=q[:, 4] * samples[:, k], minlength=len(volume))
    means = [value / volume[:, None] for value in integrated]
    maximum_initial_error = float(np.max(np.abs(initial[:, 6:] - means[0])))
    volume_error = float(np.max(np.abs(weight_sum / volume - 1)))
    if maximum_initial_error > 1e-11 or volume_error > 1e-11:
        raise RuntimeError("Initial projection/quadrature validation failed")
    numerical, high, low = fields(final[:, 6:]), fields(means[1]), fields(means[2])
    metrics = {"run": run, "n": n, "nodes": meta["nodes"], "ranks": meta["ranks"],
               "dt": meta["dt"], "volume": float(volume.sum()),
               "maximum_initial_projection_error": maximum_initial_error,
               "maximum_quadrature_volume_relative_error": volume_error,
               "errors": {name: norms(numerical[:, k] - high[:, k], volume) for k, name in enumerate(FIELDS)},
               "reference_64_128_gap": {name: norms(low[:, k] - high[:, k], volume) for k, name in enumerate(FIELDS)},
               "conserved_integral_drift": np.sum((final[:, 6:] - initial[:, 6:]) * volume[:, None], axis=0).tolist(),
               "last_coupled_residual": meta["last_coupled_residual"], "fallbacks": meta["mean_fallback_count"]}
    (raw / f"tgv2d_n{n}.metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics, final


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = args.output.resolve()
    spec = importlib.util.spec_from_file_location("cross_spectral", root / "spectral_reference.py")
    spectral = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(spectral)
    metrics, cases = {}, {}
    for run, levels in (("main", (8, 16, 32)), ("mpi8", (32,)), ("dt_half", (32,))):
        for n in levels:
            if not (root / run / "raw" / f"tgv2d_n{n}.json").exists():
                if run == "main":
                    raise RuntimeError(f"Missing primary case: {run} {n}")
                continue
            metrics[run, n], cases[run, n] = analyze(root, run, n, spectral)
    rows = []
    for n in (8, 16, 32):
        result = metrics["main", n]
        for field in FIELDS:
            row = {"n": n, "field": field, **result["errors"][field]}
            for norm in ("L1", "L2", "Linf"):
                row[f"order_{norm}"] = math.log2(metrics["main", n//2]["errors"][field][norm] / row[norm]) if n > 8 else None
            original = ORIGINAL_L2[n].get(field)
            row["original_project_L2"] = original
            row["relative_difference_from_reported_original_L2"] = row["L2"] / original - 1 if original else None
            rows.append(row)
    write_csv(root / "comparison.csv", rows)
    controls = []
    for run in ("mpi8", "dt_half"):
        if (run, 32) not in cases:
            continue
        a, b = cases["main", 32], cases[run, 32]
        if not np.array_equal(a[:, :5], b[:, :5]):
            raise RuntimeError("Control geometry mismatch")
        fa, fb = fields(a[:, 6:]), fields(b[:, 6:])
        for k, field in enumerate(FIELDS):
            delta = norms(fa[:, k] - fb[:, k], a[:, 4])["L2"]
            controls.append({"control": run, "field": field, "two_solution_L2": delta,
                             "fraction_of_main_reference_error": delta / metrics["main", 32]["errors"][field]["L2"]})
    if controls:
        write_csv(root / "controls.csv", controls)
    rng = np.random.default_rng(704)
    x, y = rng.random(256) * 2 * np.pi, rng.random(256) * 2 * np.pi
    high = np.load(root / "reference/tgv_n128_dt0.00025.npz")["U"]
    low = np.load(root / "reference/tgv_n64_dt0.0005.npz")["U"]
    reference_checks = {
        "sample_count": 256,
        "maximum_interpolation_fourier_gap": np.max(np.abs(spectral.sample(high, x, y) - spectral.sample_fourier(high, x, y)), axis=0).tolist(),
        "maximum_64_128_fourier_gap": np.max(np.abs(spectral.sample_fourier(high, x, y) - spectral.sample_fourier(low, x, y)), axis=0).tolist()}
    (root / "analysis.json").write_text(json.dumps({"metrics": list(metrics.values()), "comparison": rows,
                                                   "controls": controls, "reference_checks": reference_checks}, indent=2) + "\n")
    print(json.dumps([r for r in rows if r["field"] in ("rho", "u", "v", "w", "pressure_from_mean")], indent=2))
    print(json.dumps(controls, indent=2))


if __name__ == "__main__":
    main()
