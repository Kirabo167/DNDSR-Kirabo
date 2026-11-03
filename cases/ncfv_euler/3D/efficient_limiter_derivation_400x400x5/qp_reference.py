"""A coupled coefficient QP on a small patch of the actual frozen NCFV operator.

All constraints affected by patch variables are retained. Other coefficients are
fixed to zero, so this is a patch diagnostic, not the global optimum or a CFD run.
Pressure is audited afterwards; the QP contains conservative linear bounds only.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import minimize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    data = json.loads((args.directory / "constraints_rank0.json").read_text())
    scales = dict(data["owned_scales"])
    # A contiguous patch in this single-rank mesh; its external coefficients
    # are fixed to zero and every coupled row is still present below.
    nodes = sorted(scales)[len(scales) // 3:len(scales) // 3 + 80]
    columns = {node: j for j, node in enumerate(nodes)}
    matrices, lower, upper, constraints = [], [], [], []
    for row in data["constraints"]:
        a = np.zeros((5, len(nodes)))
        for entry in row["entries"]:
            if entry["node"] in columns:
                a[:, columns[entry["node"]]] += entry["c"]
        if not np.any(a):
            continue
        matrices.append(a)
        lower.extend(np.array(row["lower"]) - row["mean"])
        upper.extend(np.array(row["upper"]) - row["mean"])
        constraints.append((row, a))
    a = np.concatenate(matrices)
    lo, hi = np.array(lower), np.array(upper)
    nonzero = np.max(np.abs(a), axis=1) > 1e-12
    a, lo, hi = a[nonzero], lo[nonzero], hi[nonzero]
    norm = np.maximum(np.max(np.abs(a), axis=1), 1e-12)
    scaled = a / norm[:, None]
    # SLSQP accepts equalities separately; remove duplicate numerical zero
    # widths only by treating those rows as equalities, never by dropping them.
    equality = np.abs(hi - lo) < 1e-13
    eq = scaled[equality]
    if len(eq):
        # Keep an independent equality basis to avoid a rank-deficient KKT system.
        _, s, vh = np.linalg.svd(eq, full_matrices=False)
        eq = vh[s > 1e-10 * s[0]]
    g = np.concatenate((scaled[~equality], -scaled[~equality]))
    b = np.concatenate((hi[~equality] / norm[~equality], -lo[~equality] / norm[~equality]))
    tests = [{"type": "ineq", "fun": lambda x: b - g @ x, "jac": lambda x: -g}]
    if len(eq):
        tests.append({"type": "eq", "fun": lambda x: eq @ x, "jac": lambda x: eq})
    started = time.monotonic()
    result = minimize(lambda x: .5 * np.sum((x - 1) ** 2), np.zeros(len(nodes)),
                      jac=lambda x: x - 1, bounds=[(0., 1.)] * len(nodes),
                      constraints=tests, method="SLSQP", options={"maxiter": 300, "ftol": 1e-11})
    elapsed = time.monotonic() - started
    theta = result.x
    violation = float(max(0, np.max(a @ theta - hi), np.max(lo - a @ theta)))
    density_min, pressure_min = np.inf, np.inf
    for row, matrix in constraints:
        state = np.array(row["mean"]) + matrix @ theta
        density_min = min(density_min, state[0])
        pressure_min = min(pressure_min, .4 * (state[4] - .5 * np.sum(state[1:4] ** 2) / state[0]))
    budget = np.array([scales[node] for node in nodes])
    audit = {"status": "passed" if result.success and violation < 1e-8 else "failed",
             "scope": "80-node patch; all affected full-support rows; external theta fixed to zero",
             "variables": len(nodes), "linear_rows": len(a), "solver_message": str(result.message),
             "iterations": int(result.nit), "seconds": elapsed, "maximum_bound_violation": violation,
             "qp_mean_theta": float(theta.mean()), "budget_mean_theta": float(budget.mean()),
             "qp_objective": float(result.fun), "budget_objective": float(.5 * np.sum((budget - 1) ** 2)),
             "recovered_density_min": float(density_min), "recovered_pressure_min": float(pressure_min),
             "caution": "No nonlinear pressure constraints in this QP; patch result is not full CFD evidence."}
    (args.directory / "qp_reference.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))
    if audit["status"] != "passed":
        raise RuntimeError("QP reference failed validation")


if __name__ == "__main__":
    main()
