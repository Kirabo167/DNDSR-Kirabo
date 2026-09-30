"""Compare fixed-grid NCFV density fields across four physical time steps."""
import argparse
import csv
import json
import math
from pathlib import Path


STEPS = {0.2: ("dt_0p2", 8), 0.1: ("dt_0p1", 16),
         0.05: ("dt_0p05", 32), 0.025: ("dt_0p025", 64)}
MESHES = ("tet", "prism", "hex")
MODES = ("traditional", "efficient")


def read_result(directory: Path, name: str, iteration: int):
    raw = json.loads((directory / f"{name}.json").read_text())
    assert raw["steps"] == iteration and math.isclose(raw["final"]["time"], 1.6)
    nodes = {}
    files = sorted((directory / name).glob(
        f"solution_{iteration:08d}.nodes.rank*.csv"))
    assert len(files) == raw["mpi_ranks"], (name, len(files), raw["mpi_ranks"])
    for file in files:
        with file.open(newline="") as stream:
            for row in csv.DictReader(stream):
                key = int(row["original_node"])
                assert key not in nodes, (name, key)
                nodes[key] = (float(row["rho_point"]),
                              float(row["partial_volume"]),
                              float(row["rho_exact"]))
    assert len(nodes) == raw["nodes"], (name, len(nodes), raw["nodes"])
    volume = sum(value[1] for value in nodes.values())
    error = math.sqrt(sum(value[1] * (value[0] - value[2]) ** 2
                          for value in nodes.values()) / volume)
    assert math.isclose(error, raw["final"]["rho_point_L2V"], rel_tol=1e-11)
    return raw, nodes


def field_difference(a, b):
    assert a.keys() == b.keys()
    weighted = volume = 0.0
    for key, (density, measure, exact) in a.items():
        other_density, other_measure, other_exact = b[key]
        assert math.isclose(measure, other_measure, rel_tol=1e-10, abs_tol=1e-13)
        assert math.isclose(exact, other_exact, rel_tol=1e-12, abs_tol=1e-13)
        weighted += measure * (density - other_density) ** 2
        volume += measure
    return math.sqrt(weighted / volume)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--level", type=int, default=20)
    args = parser.parse_args()
    directory = args.directory.resolve()
    rows = []
    for mesh in MESHES:
        for mode in MODES:
            name = f"{mesh}_iv{args.level}_{mode}_ls"
            raw = {}
            fields = {}
            for dt, (subdirectory, iteration) in STEPS.items():
                raw[dt], fields[dt] = read_result(
                    directory / subdirectory, name, iteration)
            differences = {(a, b): field_difference(fields[a], fields[b])
                           for a, b in ((0.2, 0.1), (0.1, 0.05),
                                        (0.05, 0.025), (0.2, 0.025),
                                        (0.1, 0.025))}
            error = {dt: raw[dt]["final"]["rho_point_L2V"] for dt in STEPS}
            rows.append({
                "mesh_type": mesh, "mode": mode,
                **{f"error_dt_{dt}": error[dt] for dt in STEPS},
                **{f"field_difference_{a}_{b}": value
                   for (a, b), value in differences.items()},
                "temporal_order_0p2_0p1_0p05": math.log(
                    differences[0.2, 0.1] / differences[0.1, 0.05], 2),
                "temporal_order_0p1_0p05_0p025": math.log(
                    differences[0.1, 0.05] / differences[0.05, 0.025], 2),
                "difference_dt_0p1_to_0p025_over_fine_error":
                    differences[0.1, 0.025] / error[0.025],
                "difference_dt_0p2_to_0p025_over_fine_error":
                    differences[0.2, 0.025] / error[0.025],
            })
    output = directory / "temporal_refinement.csv"
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(output)


if __name__ == "__main__":
    main()
