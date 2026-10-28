"""Read-only chain-rule audit of frozen initial reconstruction diagnostics."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


COMPONENTS = ("rho", "rhou", "rhov", "rhow", "rhoE")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(directory, name, gamma):
    paths = sorted(directory.glob(f"{name}.json.initial_nodes.rank*.csv"))
    if not paths:
        raise FileNotFoundError(f"No initial node diagnostics for {name}")
    rows = np.concatenate([np.atleast_1d(np.genfromtxt(p, delimiter=",", names=True)) for p in paths])
    ids = rows["original_node"].astype(np.int64)
    if len(np.unique(ids)) != len(ids):
        raise ValueError("The diagnostic contains duplicate owned nodes")
    means = np.column_stack([rows[f"mean_{c}"] for c in COMPONENTS])
    points = np.column_stack([rows[f"point_{c}"] for c in COMPONENTS])
    dz = np.column_stack([rows[f"d_{c}_dz"] for c in COMPONENTS])
    density = points[:, 0]
    velocity = points[:, 1:4] / density[:, None]
    pressure = (gamma - 1) * (points[:, 4] - .5 * density * np.sum(velocity ** 2, axis=1))
    if np.any(density <= 0) or np.any(pressure <= 0):
        raise ValueError("Recovered states must be physical for this chain-rule audit")
    velocity_dz = (dz[:, 1:4] - velocity * dz[:, 0, None]) / density[:, None]
    pressure_dz = (gamma - 1) * (dz[:, 4] - np.sum(velocity * dz[:, 1:4], axis=1) +
                                .5 * np.sum(velocity ** 2, axis=1) * dz[:, 0])
    xy = np.round(np.column_stack([rows["x"], rows["y"]]) * 1e12).astype(np.int64)
    _, groups = np.unique(xy, axis=0, return_inverse=True)
    mean_spans = np.zeros(5)
    for group in range(groups.max() + 1):
        mean_spans = np.maximum(mean_spans, np.ptp(means[groups == group], axis=0))
    fields = dict(zip(("rho", "u", "v", "w", "p"),
                      (dz[:, 0], *velocity_dz.T, pressure_dz)))
    maxima = {}
    for field, values in fields.items():
        worst = int(np.argmax(np.abs(values)))
        maxima[field] = {
            "max_abs_dz": float(np.abs(values).max()),
            "nodes_above_1e_minus_10": int(np.count_nonzero(np.abs(values) > 1e-10)),
            "worst_node": int(ids[worst]),
            "worst_coordinate": [float(rows[c][worst]) for c in ("x", "y", "z")],
            "signed_dz_at_worst": float(values[worst]),
        }
    return {
        "nodes": len(rows), "mean_column_z_span_by_component": mean_spans.tolist(),
        "primitive_derivative_metrics": maxima,
        "input_sha256": {str(p): digest(p) for p in paths},
    }


def main():
    root = Path(__file__).resolve().parents[4]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=root / "data/out/NCFV/flux_strategy_audit/initial_ring_comparison")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gamma", type=float, default=1.4)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = {
        "gamma": args.gamma,
        "note": "Chain-rule derivatives of the unmodified conservative reconstruction, evaluated at recovered point states; no time integration or derivative correction.",
        "conservative_component_order": COMPONENTS,
        "cases": {name: summarize(args.input.resolve(), name, args.gamma) for name in ("compact", "complete")},
        "script_sha256": digest(Path(__file__).resolve()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({name: data["primitive_derivative_metrics"] for name, data in result["cases"].items()}, indent=2))


if __name__ == "__main__":
    main()
