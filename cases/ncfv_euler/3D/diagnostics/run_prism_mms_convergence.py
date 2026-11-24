#!/usr/bin/env python3
"""Refine the jittered prism MMS to t=0.5 and measure dual-mean convergence."""

import argparse
import csv
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

import h5py
import numpy as np


CASE_DIR = Path(__file__).resolve().parents[1]
CASE = CASE_DIR / "NCFV_exponential_mms_prism12.json"
TEMPLATE = CASE_DIR / "prism_n12_j0.1.cgns"
FIELDS = ("rho", "rhou", "rhov", "rhow", "rhoE")


def write_prism_mesh(path, n):
    """Retain the fixture's CGNS metadata and replace its structured-grid arrays."""
    k, j, i = np.indices((n + 1,) * 3)
    coordinates = np.stack((i, j, k), axis=-1) / n
    interior = coordinates[1:-1, 1:-1, 1:-1]
    rng = np.random.default_rng(230501466 + n)
    interior += rng.uniform(-0.1 / n, 0.1 / n, interior.shape)

    # One-based CGNS nodes. Each brick is sliced along the 0--2 bottom diagonal.
    nodes = np.arange(1, (n + 1) ** 3 + 1).reshape((n + 1,) * 3)
    brick = np.stack((
        nodes[:-1, :-1, :-1], nodes[:-1, :-1, 1:],
        nodes[:-1, 1:, 1:], nodes[:-1, 1:, :-1],
        nodes[1:, :-1, :-1], nodes[1:, :-1, 1:],
        nodes[1:, 1:, 1:], nodes[1:, 1:, :-1],
    ), axis=-1)
    prisms = brick[..., [[0, 1, 2, 4, 5, 6], [0, 2, 3, 4, 6, 7]]]
    faces = {
        "bc-2": brick[:, :, 0][..., [0, 4, 7, 3]],
        "bc-2-1": brick[:, :, -1][..., [1, 2, 6, 5]],
        "bc-3": brick[:, 0][..., [0, 1, 5, 4]],
        "bc-3-1": brick[:, -1][..., [3, 7, 6, 2]],
        "bc-4": brick[0][..., [[0, 2, 1], [0, 3, 2]]],
        "bc-4-1": brick[-1][..., [[7, 4, 6], [4, 5, 6]]],
    }

    with tempfile.TemporaryDirectory(dir=path.parent) as temporary:
        staged = Path(temporary) / path.name
        shutil.copyfile(TEMPLATE, staged)
        with h5py.File(staged, "r+") as mesh:
            zone = mesh["Base/Zone"]

            def replace_array(name, values):
                dataset = f"{name}/ data"
                dtype = zone[dataset].dtype
                del zone[dataset]
                zone.create_dataset(dataset, data=values.ravel(), dtype=dtype)

            cell_count = 2 * n ** 3
            zone[" data"][:, 0] = [(n + 1) ** 3, cell_count, 0]
            for axis, name in enumerate("XYZ"):
                replace_array(f"GridCoordinates/Coordinate{name}", coordinates[..., axis])
            replace_array("Elements/ElementConnectivity", prisms)
            zone["Elements/ElementRange/ data"][:] = [1, cell_count]
            last = cell_count
            for name, connectivity in faces.items():
                first = last + 1
                last += connectivity.size // connectivity.shape[-1]
                replace_array(f"{name}/ElementConnectivity", connectivity)
                zone[f"{name}/ElementRange/ data"][:] = [first, last]
                zone[f"ZoneBC/{name}/PointRange/ data"][:, 0] = [first, last]
        staged.replace(path)


def run_level(solver, mesh, n, ranks, mode, log):
    command = ["mpiexec", "-n", str(ranks), str(solver), str(CASE)]
    for key, value in (
        ("mesh/meshFile", mesh), ("physics/boundaryMode", mode),
        ("time/timeStep", 0.05 / n), ("time/iterations", 10 * n),
    ):
        command.extend(["-k", f"/{key}", "-v", str(value)])
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    with log.open("w") as output:
        print(f"mpi_ranks={ranks}", file=output, flush=True)
        subprocess.run(command, cwd=solver.parents[1], env=env, stdout=output,
                       stderr=subprocess.STDOUT, check=True)


def read_errors(log):
    text = log.read_text()
    lines = [line for line in text.splitlines() if "NCFV exponential MMS t=" in line]
    if len(lines) != 1:
        raise ValueError(f"Expected one final MMS diagnostic in {log}")
    measured = dict(re.findall(r"(\w+)=([^, ]+)", lines[0]))
    if not math.isclose(float(measured["t"]), 0.5, abs_tol=1e-12):
        raise ValueError(f"MMS stopped at t={measured['t']} in {log}")
    names = [f"{field}{suffix}" for field in FIELDS
             for suffix in ("_mean_L2_Q6", "_boundary_mean_L2_Q6")]
    names.append("rho_source_mean_L2_Q6")
    errors = {name: float(measured[name]) for name in names}
    if not all(math.isfinite(error) and error > 0 for error in errors.values()):
        raise ValueError(f"Invalid MMS error norm in {log}")
    ranks_match = re.search(r"^mpi_ranks=(\d+)$", text, re.MULTILINE)
    if ranks_match:
        ranks = int(ranks_match[1])
    else:
        # Logs from direct mpiexec runs list the partition cell counts by rank.
        rank_ids = {int(value) for value in re.findall(r"\[(\d+): nCell \d+\]", text)}
        if not rank_ids or rank_ids != set(range(max(rank_ids) + 1)):
            raise ValueError(f"Missing MPI partition ranks in {log}")
        ranks = len(rank_ids)
    return ranks, errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sizes", nargs="+", type=int, default=[12, 16, 20, 24])
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--solver", type=Path, help="Run the solver after generating the meshes")
    action.add_argument("--summarize-only", action="store_true", help="Read completed logs in --output")
    parser.add_argument("--ranks", type=int, default=8)
    parser.add_argument("--mode", choices=["CharacteristicFarField", "MMSExactExterior", "FarField"],
                        nargs="+", default=["CharacteristicFarField", "MMSExactExterior"],
                        help="Boundary mode names used by the solver or recorded in the input log filenames")
    args = parser.parse_args()
    if min(args.sizes) < 2 or args.sizes != sorted(set(args.sizes)) or args.ranks < 1:
        parser.error("Use increasing unique sizes >= 2 and a positive rank count")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if not args.summarize_only:
        for n in args.sizes:
            mesh = output / f"prism_n{n}_j0.1.cgns"
            if mesh.resolve() == TEMPLATE.resolve():
                raise ValueError("Choose a results directory outside the case fixture")
            write_prism_mesh(mesh, n)
    if args.solver is None and not args.summarize_only:
        return

    rows = []
    for mode in args.mode:
        previous = None
        for n in args.sizes:
            log = output / f"{mode}_n{n}.log"
            if args.solver:
                print(f"{mode} n={n}", flush=True)
                mesh = output / f"prism_n{n}_j0.1.cgns"
                run_level(args.solver.resolve(), mesh, n, args.ranks, mode, log)
            ranks, errors = read_errors(log)
            row = {"mode": mode, "n": n, "ranks": ranks, "dt": 0.05 / n}
            for name, value in errors.items():
                row[name] = value
                row[f"{name}_order"] = (math.log(previous[name] / value) / math.log(n / previous["n"])
                                        if previous is not None else "")
            rows.append(row)
            previous = row
    with (output / "prism_mms_characteristic_convergence.csv").open("w", newline="") as summary:
        writer = csv.DictWriter(summary, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
