#!/usr/bin/env python3
"""Write exact Cartesian dual-cell conservative means for the small RP1 pilot.

This diagnostic is deliberately fixed to the existing 40 x 40 x 4 pilot mesh.
The 400 x 400 x 10 production configurations and outputs are not changed.
The CSV uses zero-based CGNS original-node IDs, with x varying fastest.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[4]
NX, NY, NZ = 40, 40, 4
GAMMA = 1.4
PRIMITIVE = np.array([
    [0.138, 1.206, 1.206, 0.0, 0.029],
    [0.5323, 0.0, 1.206, 0.0, 0.3],
    [0.5323, 1.206, 0.0, 0.0, 0.3],
    [1.5, 0.0, 0.0, 0.0, 1.5],
])  # BL, BR, TL, TR; [rho,u,v,w,p]


def halfspace_weights(coordinate: float, spacing: float):
    """Exact overlap with x<0.5 and x>=0.5 in a clipped median dual interval."""
    lower = max(0.0, coordinate - 0.5 * spacing)
    upper = min(1.0, coordinate + 0.5 * spacing)
    length = upper - lower
    fraction_left = max(0.0, min(upper, 0.5) - lower) / length
    return fraction_left, 1.0 - fraction_left, length


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path,
                        default=ROOT / "data/mesh/riemann_barth_pilot40x40x4.cgns")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "data/out/NCFV/riemann_barth_400/pilots/"
                        "RP1_exact_dual_means_40x40x4.csv")
    args = parser.parse_args()
    mesh_path, output = args.mesh.resolve(), args.output.resolve()
    metadata_path = output.with_suffix(".metadata.json")
    if output.exists() or metadata_path.exists():
        raise FileExistsError(f"refusing to overwrite {output} or its metadata")

    x = np.linspace(0.0, 1.0, NX + 1)
    y = np.linspace(0.0, 1.0, NY + 1)
    with h5py.File(mesh_path, "r") as mesh:
        zone = mesh["Base/blk-1"]
        np.testing.assert_array_equal(zone[" data"][:].reshape(-1),
                                      [(NX + 1) * (NY + 1) * (NZ + 1),
                                       NX * NY * NZ, 0])
        mesh_coords = np.column_stack([
            zone[f"GridCoordinates/Coordinate{axis}/ data"][:] for axis in "XYZ"])
    depth = float(mesh_coords[:, 2].max())
    z = np.linspace(0.0, depth, NZ + 1)
    coords = np.column_stack((np.tile(x, (NY + 1) * (NZ + 1)),
                              np.tile(np.repeat(y, NX + 1), NZ + 1),
                              np.repeat(z, (NX + 1) * (NY + 1))))
    np.testing.assert_array_equal(mesh_coords, coords)

    conservative = PRIMITIVE.copy()
    conservative[:, 1:4] *= PRIMITIVE[:, :1]
    conservative[:, 4] = (PRIMITIVE[:, 4] / (GAMMA - 1.0)
                          + 0.5 * PRIMITIVE[:, 0]
                          * np.sum(PRIMITIVE[:, 1:4] ** 2, axis=1))
    values, volumes, mixtures = [], [], []
    for point in coords:
        left, right, dx = halfspace_weights(point[0], 1.0 / NX)
        bottom, top, dy = halfspace_weights(point[1], 1.0 / NY)
        dz = min(depth, point[2] + depth / NZ / 2) - max(
            0.0, point[2] - depth / NZ / 2)
        weights = np.array([left * bottom, right * bottom,
                            left * top, right * top])
        np.testing.assert_allclose(weights.sum(), 1.0, rtol=0, atol=1e-14)
        values.append(weights @ conservative)
        volumes.append(dx * dy * dz)
        mixtures.append(np.count_nonzero(weights > 1e-13))
    values, volumes = np.array(values), np.array(volumes)
    expected_integral = conservative.mean(axis=0) * depth
    actual_integral = np.sum(values * volumes[:, None], axis=0)
    np.testing.assert_allclose(actual_integral, expected_integral, rtol=2e-13)
    np.testing.assert_allclose(volumes.sum(), depth, rtol=2e-13)
    pressure = (GAMMA - 1) * (values[:, 4] - 0.5 * np.sum(
        values[:, 1:4] ** 2, axis=1) / values[:, 0])
    assert np.all(values[:, 0] > 0) and np.all(pressure > 0)
    assert len(values) == 8405

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["globalNode", "rho", "rhou", "rhov", "rhow", "rhoE"])
        for node_id, value in enumerate(values):
            writer.writerow([node_id, *[format(float(v), ".17g") for v in value]])
    with output.open("rb") as stream:
        sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    metadata = {
        "purpose": "Small-grid feasibility diagnostic; not production-grid results",
        "mesh": str(mesh_path), "csv": str(output), "csv_sha256": sha256,
        "cells_per_axis": [NX, NY, NZ], "node_count": len(values),
        "domain_depth": depth,
        "variables": "Conservative", "node_file_index_base": 0,
        "quadrant_order": ["BL", "BR", "TL", "TR"],
        "primitive_states": PRIMITIVE.tolist(),
        "nodes_mixing_two_quadrants": mixtures.count(2),
        "nodes_mixing_four_quadrants": mixtures.count(4),
        "expected_conservative_integral": expected_integral.tolist(),
        "actual_conservative_integral": actual_integral.tolist(),
        "minimum_density": float(values[:, 0].min()),
        "minimum_pressure": float(pressure.min()),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
