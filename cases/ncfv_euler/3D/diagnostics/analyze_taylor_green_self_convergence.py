"""Measure Taylor--Green self-convergence at nodes shared by three meshes."""

import argparse
import csv
import glob
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import brentq


LEVELS = (10, 20, 40, 80)
MODES = ("Traditional", "Efficient")
FIELDS = {"velocity": slice(4, 7), "density": slice(3, 4),
          "pressure": slice(7, 8)}


def load_case(raw, level, mode, weight, mach):
    stem = f"tgv_prism{level}_{mode.lower()}_p{weight:g}_m{mach:g}.json"
    meta = json.loads((raw / stem).read_text())
    paths = sorted(glob.glob(str(raw / f"{stem}.rank*.csv")))
    if len(paths) != meta["mpi_ranks"]:
        raise RuntimeError(f"Missing rank fields for {stem}")
    table = np.concatenate([np.atleast_2d(np.loadtxt(p, delimiter=",", skiprows=1))
                            for p in paths])
    if len(table) != meta["nodes"] or table.shape[1] != 9:
        raise RuntimeError(f"Unexpected nodal field shape for {stem}: {table.shape}")
    keys = [tuple(row) for row in np.rint(table[:, :3] * 1e8).astype(np.int64)]
    mapping = dict(zip(keys, range(len(keys))))
    if len(mapping) != len(table):
        raise RuntimeError(f"Duplicate owned coordinates on {stem}")
    if not np.isfinite(table).all() or np.min(table[:, 8]) <= 0:
        raise RuntimeError(f"Nonfinite field or invalid dual volume on {stem}")
    return meta, table, mapping


def weighted_difference(a, b, indices_a, indices_b, weights, field):
    difference = a[indices_a, field] - b[indices_b, field]
    return math.sqrt(np.sum(weights * np.sum(difference**2, axis=1)) /
                     np.sum(weights))


def observed_order(coarse_difference, fine_difference, nodes):
    if coarse_difference <= fine_difference:
        return float("nan")
    h = [n ** (-1 / 3) for n in nodes]
    ratio = coarse_difference / fine_difference

    def model(power):
        return (h[0] ** power - h[1] ** power) / (
            h[1] ** power - h[2] ** power) - ratio

    if model(1e-5) * model(12) > 0:
        return float("nan")
    return brentq(model, 1e-5, 12)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--weight", type=float, default=2)
    parser.add_argument("--mach", type=float, default=0.1)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    args = parser.parse_args()
    output = args.output.resolve()
    raw = output / "raw"
    rows = []
    all_four_rows = []
    diagnostics = []
    for mode in args.modes:
        cases = {level: load_case(raw, level, mode, args.weight, args.mach)
                 for level in LEVELS}
        for level, (meta, table, _) in cases.items():
            diagnostics.append({"mode": mode, "level": level,
                                "nodes": meta["nodes"], "dt": meta["dt"],
                                "mass_drift": meta["final_conserved"][0] -
                                meta["initial_conserved"][0],
                                "kinetic_energy": meta["kinetic_energy_point_integral"],
                                "minimum_density": meta["minimum_point_density"],
                                "minimum_pressure": meta["minimum_point_pressure"],
                                "nonphysical_count": meta["nonphysical_point_count"]})
        common_four = sorted(set.intersection(
            *(set(cases[level][2]) for level in LEVELS)))
        if len(common_four) < 100:
            raise RuntimeError(f"Insufficient four-grid shared coordinates: {mode}")
        tables = [cases[level][1] for level in LEVELS]
        indices = [[cases[level][2][key] for key in common_four]
                   for level in LEVELS]
        weights_four = tables[0][indices[0], 8]
        nodes_four = [cases[level][0]["nodes"] for level in LEVELS]
        for name, field in FIELDS.items():
            differences = [weighted_difference(
                tables[i], tables[i + 1], indices[i], indices[i + 1],
                weights_four, field) for i in range(3)]
            all_four_rows.append({
                "mode": mode, "field": name, "shared_nodes": len(common_four),
                "difference_10_20": differences[0],
                "difference_20_40": differences[1],
                "difference_40_80": differences[2],
                "order_10_20_40": observed_order(
                    differences[0], differences[1], nodes_four[:3]),
                "order_20_40_80": observed_order(
                    differences[1], differences[2], nodes_four[1:])})
        for triplet in (LEVELS[:3], LEVELS[1:]):
            (meta0, a, map0), (meta1, b, map1), (meta2, c, map2) = (
                cases[level] for level in triplet)
            common = sorted(map0.keys() & map1.keys() & map2.keys())
            if len(common) < 100:
                raise RuntimeError(f"Insufficient shared coordinates: {mode} {triplet}")
            i0 = [map0[key] for key in common]
            i1 = [map1[key] for key in common]
            i2 = [map2[key] for key in common]
            weights = a[i0, 8]
            for name, field in FIELDS.items():
                d01 = weighted_difference(a, b, i0, i1, weights, field)
                d12 = weighted_difference(b, c, i1, i2, weights, field)
                rows.append({"mode": mode, "field": name,
                             "mesh_triplet": "-".join(map(str, triplet)),
                             "shared_nodes": len(common),
                             "coarse_nodes": meta0["nodes"],
                             "coarse_middle_L2_difference": d01,
                             "middle_fine_L2_difference": d12,
                             "observed_order": observed_order(
                                 d01, d12, [meta0["nodes"],
                                           meta1["nodes"], meta2["nodes"]])})
    for filename, entries in (("self_convergence.csv", rows),
                              ("all_four_self_convergence.csv", all_four_rows),
                              ("diagnostics.csv", diagnostics)):
        with (output / filename).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, entries[0].keys())
            writer.writeheader()
            writer.writerows(entries)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
