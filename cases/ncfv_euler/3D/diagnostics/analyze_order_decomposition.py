"""Compare coupled h/dt orders with orders evaluated at the same dt=0.025."""
import argparse
import csv
import math
from pathlib import Path


PAIRS = ((10, 20), (20, 40), (40, 80))


def rows_by_key(path: Path):
    with path.open(newline="") as stream:
        return {(row["mesh_type"], row["mode"], row["method"], int(row["level"])): row
                for row in csv.DictReader(stream)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("iv10", type=Path)
    parser.add_argument("iv20", type=Path)
    parser.add_argument("iv40", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    baseline = rows_by_key(args.baseline / "metrics.csv")
    fixed = {}
    for level, directory in ((10, args.iv10), (20, args.iv20), (40, args.iv40)):
        fixed[level] = rows_by_key(directory / "dt_0p025" / "metrics.csv")
    fixed[80] = baseline
    result = []
    for mesh in ("tet", "prism", "hex"):
        for mode in ("traditional", "efficient"):
            row = {"mesh_type": mesh, "mode": mode, "method": "ls"}
            for coarse, fine in PAIRS:
                key_coarse = (mesh, mode, "ls", coarse)
                key_fine = (mesh, mode, "ls", fine)
                base_coarse = baseline[key_coarse]
                base_fine = baseline[key_fine]
                fixed_coarse = fixed[coarse][key_coarse]
                fixed_fine = fixed[fine][key_fine]
                log_h = math.log(float(base_coarse["h"]) / float(base_fine["h"]))
                p_coupled = math.log(
                    float(base_coarse["rho_point_L2V_final"]) /
                    float(base_fine["rho_point_L2V_final"])) / log_h
                p_fixed = math.log(
                    float(fixed_coarse["rho_point_L2V_final"]) /
                    float(fixed_fine["rho_point_L2V_final"])) / log_h
                stem = f"iv{coarse}_iv{fine}"
                row[f"coupled_order_{stem}"] = p_coupled
                row[f"fixed_dt_0p025_order_{stem}"] = p_fixed
                row[f"order_change_{stem}"] = p_coupled - p_fixed
            row["initial_order_iv40_iv80"] = float(
                baseline[(mesh, mode, "ls", 80)]["observed_order_t0"])
            result.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=result[0].keys())
        writer.writeheader()
        writer.writerows(result)
    print(args.output)


if __name__ == "__main__":
    main()
