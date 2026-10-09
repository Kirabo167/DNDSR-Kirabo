#!/usr/bin/env python3
"""Compute volume-based size statistics for the NCFV prism/tet/hex mesh matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


IVS = (10, 20, 40, 80)
ELEMENTS = {
    "prism": ("PrismElements", 6),
    "hex": ("HexElements", 8),
    "tet": ("TetElements", 4),
}
HEX_TETS = (
    (0, 1, 2, 6),
    (0, 2, 3, 6),
    (0, 3, 7, 6),
    (0, 7, 4, 6),
    (0, 4, 5, 6),
    (0, 5, 1, 6),
)


def mesh_path(root: Path, family: str, iv: int) -> Path:
    if family == "prism":
        return root / f"cases/ncfv_euler/3D/iv{iv}_3d_1_dnds.cgns"
    return root / f"cases/ncfv_euler/3D/generated_periodic_meshes/periodic_{family}_iv{iv}.cgns"


def find_zone(base: h5py.Group) -> h5py.Group:
    zones = [value for key, value in base.items() if key.strip() != "data" and "GridCoordinates" in value]
    if len(zones) != 1:
        raise RuntimeError(f"expected one zone, found {len(zones)}")
    return zones[0]


def tetra_volume(points: np.ndarray) -> np.ndarray:
    return np.abs(
        np.einsum(
            "ij,ij->i",
            np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]),
            points[:, 3] - points[:, 0],
        )
    ) / 6.0


def cell_volumes(family: str, points: np.ndarray) -> np.ndarray:
    if family == "tet":
        return tetra_volume(points)
    if family == "prism":
        first = points[:, (0, 1, 2, 3), :]
        second = points[:, (1, 2, 4, 3), :]
        third = points[:, (2, 4, 5, 3), :]
        return tetra_volume(first) + tetra_volume(second) + tetra_volume(third)
    return sum(tetra_volume(points[:, tet, :]) for tet in HEX_TETS)


def statistics(path: Path, family: str, iv: int) -> dict[str, object]:
    element_name, nodes_per_cell = ELEMENTS[family]
    with h5py.File(path, "r") as mesh:
        zone = find_zone(mesh["Base"])
        grid = zone["GridCoordinates"]
        coordinates = np.column_stack(
            [grid[f"Coordinate{axis}"][" data"][:] for axis in "XYZ"]
        )
        connectivity = zone[element_name]["ElementConnectivity"][" data"][:]

    if connectivity.size % nodes_per_cell:
        raise RuntimeError(f"invalid connectivity length in {path}")
    connectivity = connectivity.reshape((-1, nodes_per_cell)) - 1

    total_volume = 0.0
    total_equivalent_size = 0.0
    minimum_volume = np.inf
    maximum_volume = 0.0
    chunk_size = 200_000
    for start in range(0, len(connectivity), chunk_size):
        points = coordinates[connectivity[start : start + chunk_size]]
        volumes = cell_volumes(family, points)
        if np.any(volumes <= 0):
            raise RuntimeError(f"non-positive cell volume in {path}")
        total_volume += float(volumes.sum())
        total_equivalent_size += float(np.cbrt(volumes).sum())
        minimum_volume = min(minimum_volume, float(volumes.min()))
        maximum_volume = max(maximum_volume, float(volumes.max()))

    cells = len(connectivity)
    extents = np.ptp(coordinates, axis=0)
    box_volume = float(np.prod(extents))
    if not np.isclose(total_volume, box_volume, rtol=1e-10, atol=1e-10):
        raise RuntimeError(
            f"cell volume {total_volume} does not match bounding-box volume {box_volume} in {path}"
        )

    mean_volume = total_volume / cells
    return {
        "family": family,
        "iv": iv,
        "mesh_file": str(path),
        "domain_extents": extents.tolist(),
        "domain_volume": total_volume,
        "nodes": len(coordinates),
        "volume_cells": cells,
        "mean_cell_volume": mean_volume,
        "equivalent_size_from_mean_volume": mean_volume ** (1.0 / 3.0),
        "mean_volume_equivalent_size": total_equivalent_size / cells,
        "minimum_cell_volume": minimum_volume,
        "maximum_cell_volume": maximum_volume,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[4]
    records = [
        statistics(mesh_path(root, family, iv), family, iv)
        for family in ("prism", "hex", "tet")
        for iv in IVS
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
