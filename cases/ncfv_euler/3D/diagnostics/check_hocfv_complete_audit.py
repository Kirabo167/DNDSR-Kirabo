"""Check actual C++ matrices, reconstructions, Roe flux and RK stages against the supplied Python core."""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--runs", default="main")
    args = parser.parse_args()
    root = args.root.resolve()
    spec = importlib.util.spec_from_file_location("reference_core", root / "reference_core.py")
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    derivatives = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1),
                   (2, 0, 0), (1, 1, 0), (0, 2, 0), (0, 1, 1), (0, 0, 2), (1, 0, 1)]
    vec = lambda value: np.asarray(value, dtype=float).reshape(-1)
    checks = []
    for path in sorted((root / args.runs / "raw").glob("*.audit.json")):
        audit = json.loads(path.read_text())
        meta = json.loads(Path(str(path).removesuffix(".audit.json")).read_text())
        node = audit["first_node"]
        matrix = np.zeros((9, 9))
        right = np.zeros((9, 5))
        for sample in node["samples"]:
            length = sample["length"]
            rows = np.array([core.basis(vec(sample["point"]), vec(node["center"]),
                                       vec(node["ref"]), vec(node["moment"]), d)
                             for d in derivatives])
            weights = np.array([1] + [length / 2] * 3 + [length**2 / 4] * 6)
            weights[[5, 7, 9]] *= np.sqrt(2)
            weighted = weights[:, None] * rows
            matrix += weighted.T @ weighted
            right += np.outer(rows[0], vec(sample["neighbor_mean"]) - vec(node["initial_mean"]))
        matrix_error = np.max(np.abs(matrix - np.array(node["A"]))) / np.max(np.abs(matrix))
        sweep_error = np.max(np.abs(np.linalg.solve(matrix, right) -
                                    np.array(node["coefficients_after_first_sweep"])))
        edge = audit["first_edge"]
        states = []
        reconstruction_error = 0
        for side in ("left", "right"):
            expected = core.reconstruct(vec(edge[f"{side}_mean"]),
                                        np.array(edge[f"{side}_coefficients"]).T.tolist(),
                                        vec(edge["point"]), vec(edge[f"{side}_center"]),
                                        vec(edge[f"{side}_ref"]), vec(edge[f"{side}_moment"]))
            expected = core.face_state(expected, vec(edge[f"{side}_mean"]))
            reconstruction_error = max(reconstruction_error,
                                       np.max(np.abs(vec(expected) - vec(edge[f"{side}_state"]))))
            states.append(expected)
        normal = vec(edge["area_vector"])
        normal /= np.linalg.norm(normal)
        flux_error = np.max(np.abs(vec(core.roe_flux(normal, *states)) - vec(edge["flux"])))
        rk_error = 0
        for c, stage in zip((1, 0.25, 2 / 3), audit["rk_stage"]):
            expected = (1 - c) * vec(stage["base"]) + c * (
                vec(stage["before"]) + meta["dt"] * vec(stage["rhs_divided_by_volume"]))
            rk_error = max(rk_error, np.max(np.abs(expected - vec(stage["after"]))))
        check = {"file": path.name, "matrix_relative_max_error": float(matrix_error),
                 "first_scan_coefficient_max_error": float(sweep_error),
                 "reconstruction_max_error": float(reconstruction_error),
                 "roe_max_error": float(flux_error), "rk_max_error": float(rk_error)}
        if matrix_error > 1e-12 or max(sweep_error, reconstruction_error, flux_error, rk_error) > 1e-11:
            raise RuntimeError(f"Independent reference check failed: {check}")
        checks.append(check)
    if not checks:
        raise RuntimeError("No C++ audit snapshots found")
    (root / "independent_checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
