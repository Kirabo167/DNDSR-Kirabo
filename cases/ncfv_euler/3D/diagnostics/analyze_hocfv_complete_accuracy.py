"""Summarize the complete HOCFV port, convergence and implementation checks."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, rows[0].keys())
        writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    levels = (10, 20, 40, 80)
    cases, checks, transport = {}, [], []
    for problem in ("entropy", "vortex"):
        for level in levels:
            name = f"{problem}_iv{level}.json"
            data = json.loads((root / "main_halo/raw" / name).read_text())
            if data["sweeps"] != 30 or data["variational_weight"] != 1 or data["ranks"] != 8:
                raise RuntimeError(f"Incorrect primary algorithm settings: {name}")
            if data["rhs_evaluations"] != 3 * data["steps"] or abs(data["total_volume"] - 400) > 1e-8:
                raise RuntimeError(f"Incorrect RK chain or volume: {name}")
            cases[problem, level] = data
            checks.append({"problem": problem, "level": level, "nodes": data["nodes"],
                           "volume": data["total_volume"], "volume_mismatch": data["maximum_volume_mismatch"],
                           "closure": data["maximum_relative_closure"], "moment_mismatch": data["maximum_moment_mismatch"],
                           "matrix_condition": data["maximum_matrix_condition"], "matrix_solve_error": data["maximum_matrix_solve_error"],
                           "first_coupled_residual": data["first_coupled_residual"],
                           "last_RK_stage_coupled_residual": data["last_coupled_residual"],
                           "fallbacks": data["mean_fallback_count"], "mass_drift": data["mass_drift"],
                           "eight_vs_24_point_reference_L2": data["reference_quadrature_difference_L2"]})
            old_path = root / "main/raw" / name
            if old_path.exists():
                old = json.loads(old_path.read_text())
                fields = ("density_L1", "density_L2", "density_Linf", "mass_drift",
                          "first_coupled_residual", "last_coupled_residual")
                difference = max(abs(data[f] - old[f]) for f in fields)
                if difference != 0:
                    raise RuntimeError(f"Transport optimization changed a checked result: {name}")
                transport.append({"problem": problem, "level": level, "maximum_checked_difference": difference})
    if len(transport) != 6:
        raise RuntimeError("Expected six complete communication comparisons")
    sensitivity = []
    for problem in ("entropy", "vortex"):
        normal = cases[problem, 40]
        for name in ("dt_half", "sweeps60"):
            data = json.loads((root / name / "raw" / f"{problem}_iv40.json").read_text())
            sensitivity.append({"problem": problem, "check": name, "normal_L2": normal["density_L2"],
                                "checked_L2": data["density_L2"],
                                "relative_L2_change": data["density_L2"] / normal["density_L2"] - 1,
                                "last_coupled_residual": data["last_coupled_residual"]})
    write_csv(root / "geometry_and_solver_checks.csv", checks)
    write_csv(root / "transport_equivalence.csv", transport)
    write_csv(root / "sensitivity.csv", sensitivity)
    rows = list(csv.DictReader((root / "main_halo/metrics.csv").open()))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    for axis, problem, title in zip(axes, ("vortex", "entropy"), ("Specified isentropic vortex", "3-D entropy-wave comparison")):
        selected = [r for r in rows if r["problem"] == problem]
        for norm in ("L1", "L2", "Linf"):
            axis.loglog(levels, [float(r[norm]) for r in selected], "o-", label=norm)
        for row in selected[1:]:
            axis.annotate(f"q={float(row['order_L2']):.3f}",
                          (float(row["level"]), float(row["L2"])),
                          xytext=(-3 if int(row["level"]) == 80 else 0, 10),
                          textcoords="offset points", fontsize=8,
                          ha="right" if int(row["level"]) == 80 else "center")
        axis.set(xlabel="Prism mesh level", ylabel="Density mean error", title=title)
        axis.set_xticks(levels, [str(n) for n in levels]); axis.legend(); axis.grid(True, which="both", alpha=.25)
        axis.xaxis.set_minor_formatter(NullFormatter())
    fig.suptitle("Complete static periodic HOCFV chain; quadratic VR; 30 persistent scans; SSPRK3")
    fig.savefig(root / "accuracy.png", dpi=180); fig.savefig(root / "accuracy.pdf")
    paths = [root / "reference_specification.md", root / "reference_core.py", root / "source_full_gather.cpp",
             root / "source_sparse_at_run.cpp",
             Path("cases/ncfv_euler/3D/diagnostics/hocfv_complete_reference_probe.cpp"),
             Path("/tmp/hocfv_complete_reference_20261010"), Path("/tmp/hocfv_complete_reference_halo_20261010"),
             Path("/tmp/hocfv_complete_reference_final_20261010")]
    (root / "provenance.json").write_text(json.dumps({str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                                                    for p in paths}, indent=2) + "\n")
    print(json.dumps({"primary_cases": len(checks), "transport_comparisons": len(transport),
                      "sensitivity": sensitivity, "metrics": rows}, indent=2))


if __name__ == "__main__":
    main()
