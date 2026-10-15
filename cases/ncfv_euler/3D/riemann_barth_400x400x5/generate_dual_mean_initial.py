#!/usr/bin/env python3
"""Write exact conservative dual means for the four Cartesian Riemann cases.

Infer dimensions from a uniform Cartesian CGNS mesh and verify x-fast original
node order. Intersect each clipped median dual cell with the four quadrants;
convert each prescribed primitive state to conservative variables before volume
averaging. Neither solver configurations nor existing CSV files are modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[4]
GAMMA = 1.4
INITIAL_STATES = {
    "RP1": [[0.138, 1.206, 1.206, 0, 0.029],
            [0.5323, 0, 1.206, 0, 0.3],
            [0.5323, 1.206, 0, 0, 0.3], [1.5, 0, 0, 0, 1.5]],
    "RP2": [[1.1, 0.8939, 0.8939, 0, 1.1],
            [0.5065, 0, 0.8939, 0, 0.35],
            [0.5065, 0.8939, 0, 0, 0.35], [1.1, 0, 0, 0, 1.1]],
    "RP3": [[0.8, 0.1, 0.1, 0, 0.4],
            [0.5197, 0.1, -0.6259, 0, 0.4],
            [0.5197, -0.6259, 0.1, 0, 0.4], [1, 0.1, 0.1, 0, 1]],
    "RP4": [[0.8, 0.1, 0.1, 0, 1], [1, 0.1, 0.8276, 0, 1],
            [1.0222, -0.6179, 0.1, 0, 1], [0.5313, 0.1, 0.1, 0, 0.4]],
}  # BL, BR, TL, TR; [rho,u,v,w,p], exactly as in initial_states.png.


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_cartesian_axes(mesh_path):
    with h5py.File(mesh_path, "r") as mesh:
        zone = mesh["Base/blk-1"]
        counts = zone[" data"][:].reshape(-1)
        coordinates = [zone[f"GridCoordinates/Coordinate{axis}/ data"][:]
                       for axis in "XYZ"]
    axes = [np.unique(value) for value in coordinates]
    nx, ny, nz = [len(axis) - 1 for axis in axes]
    assert min(nx, ny, nz) >= 1
    np.testing.assert_array_equal(counts, [(nx + 1) * (ny + 1) * (nz + 1),
                                           nx * ny * nz, 0])
    for axis in axes:
        assert axis[0] == 0.0 and axis[-1] > 0
        np.testing.assert_allclose(axis, np.linspace(0, axis[-1], len(axis)),
                                   rtol=0, atol=2e-15)
    assert axes[0][-1] == 1.0 and axes[1][-1] == 1.0
    expected = (np.tile(axes[0], (ny + 1) * (nz + 1)),
                np.tile(np.repeat(axes[1], nx + 1), nz + 1),
                np.repeat(axes[2], (nx + 1) * (ny + 1)))
    for stored, ordered in zip(coordinates, expected):
        np.testing.assert_array_equal(stored, ordered)
    return nx, ny, nz, float(axes[2][-1])


def median_overlap(n):
    """Use index coordinates for exact 1/2 fractions on an aligned discontinuity."""
    node = np.arange(n + 1, dtype=float)
    lower, upper = np.maximum(0, node - 0.5), np.minimum(n, node + 0.5)
    width = upper - lower
    left_fraction = np.clip((n / 2 - lower) / width, 0, 1)
    return left_fraction, width / n


def write_case(name, output, mesh_path, mesh_sha256, dimensions):
    nx, ny, nz, depth = dimensions
    metadata_path = output.with_suffix(".metadata.json")
    if output.exists() or metadata_path.exists():
        raise FileExistsError(f"refusing to overwrite {output} or its metadata")
    primitive = np.asarray(INITIAL_STATES[name], dtype=float)
    conservative = primitive.copy()
    conservative[:, 1:4] *= primitive[:, :1]
    conservative[:, 4] = (primitive[:, 4] / (GAMMA - 1)
                          + 0.5 * primitive[:, 0]
                          * np.sum(primitive[:, 1:4] ** 2, axis=1))
    left, dx = median_overlap(nx)
    bottom, dy = median_overlap(ny)
    weights = np.stack((np.outer(bottom, left), np.outer(bottom, 1 - left),
                        np.outer(1 - bottom, left), np.outer(1 - bottom, 1 - left)),
                       axis=-1).reshape(-1, 4)
    np.testing.assert_array_equal(weights.sum(axis=1), np.ones(len(weights)))
    values = weights @ conservative
    volumes_xy = np.outer(dy, dx).reshape(-1)
    actual_integral = np.sum(values.astype(np.longdouble)
                             * volumes_xy[:, None].astype(np.longdouble),
                             axis=0).astype(float) * depth
    expected_integral = conservative.mean(axis=0) * depth
    np.testing.assert_allclose(actual_integral, expected_integral,
                               rtol=2e-13, atol=1e-16)
    pressure = (GAMMA - 1) * (values[:, 4] - 0.5 * np.sum(
        values[:, 1:4] ** 2, axis=1) / values[:, 0])
    assert np.all(values[:, 0] > 0) and np.all(pressure > 0)
    unique_weights, row_codes = np.unique(weights, axis=0, return_inverse=True)
    formatted = [",".join(format(float(value), ".17g") for value in row)
                 for row in unique_weights @ conservative]
    row_text = [formatted[code] for code in row_codes]
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("x", buffering=1024 * 1024) as stream:
        stream.write("globalNode,rho,rhou,rhov,rhow,rhoE\n")
        for _ in range(nz + 1):
            for text in row_text:
                stream.write(f"{count},{text}\n")
                count += 1
    assert count == (nx + 1) * (ny + 1) * (nz + 1)
    mixture_count = np.count_nonzero(weights, axis=1)
    metadata = {
        "case": name, "mesh": str(mesh_path), "mesh_sha256": mesh_sha256,
        "csv": str(output), "csv_sha256": digest(output),
        "cells_per_axis": [nx, ny, nz], "domain_depth": depth, "node_count": count,
        "variables": "Conservative", "node_file_index_base": 0, "gamma": GAMMA,
        "initialization": "Exact quadrant intersection of clipped median dual volumes",
        "quadrant_order": ["BL", "BR", "TL", "TR"],
        "primitive_states": primitive.tolist(),
        "conservative_states": conservative.tolist(),
        "nodes_mixing_two_quadrants": int(np.sum(mixture_count == 2)) * (nz + 1),
        "nodes_mixing_four_quadrants": int(np.sum(mixture_count == 4)) * (nz + 1),
        "expected_conservative_integral": expected_integral.tolist(),
        "actual_conservative_integral": actual_integral.tolist(),
        "minimum_density": float(values[:, 0].min()),
        "minimum_pressure": float(pressure.min()),
        "size_bytes": output.stat().st_size,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path,
                        default=ROOT / "data/mesh/riemann_barth_400x400x5.cgns")
    parser.add_argument("--cases", nargs="+", choices=list(INITIAL_STATES),
                        default=list(INITIAL_STATES))
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "data/mesh/riemann_barth_400x400x5_initial")
    args = parser.parse_args()
    mesh_path = args.mesh.resolve()
    dimensions, mesh_sha256 = read_cartesian_axes(mesh_path), digest(mesh_path)
    for name in args.cases:
        write_case(name, args.output_dir.resolve() / f"{name}.csv",
                   mesh_path, mesh_sha256, dimensions)


if __name__ == "__main__":
    main()
