#!/usr/bin/env python3
"""Generate the Cartesian thin-layer mesh for the four Barth-limited Riemann cases.

Defaults are 400 x 400 x 5 HEXA_8 cells in [0, 1] x [0, 1] x [0, .0125].
The three spacings are therefore .0025.  The six named boundary zones are
XMIN, XMAX, YMIN, YMAX, ZMIN, ZMAX; the case configurations supply their
transmissive or slip-wall behavior.  No DNDSR Python extension is imported.

Run with the repository venv, from any directory.  Existing meshes are never
overwritten.  A metadata JSON sidecar records the generated mesh and its hash.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import sys

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts"))
from generate_taylor_green_meshes import CG_SIZE, cgns_library, check, pointer


BOUNDARY_NAMES = ("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX")


def make_arrays(nx: int, ny: int, nz: int, depth: float):
    """Use CGNS HEXA_8 ordering and outward-oriented boundary quadrilaterals."""
    sx, sy = nx + 1, ny + 1
    x = np.linspace(0.0, 1.0, sx)
    y = np.linspace(0.0, 1.0, sy)
    z = np.linspace(0.0, depth, nz + 1)
    coords = (
        np.ascontiguousarray(np.tile(x, sy * (nz + 1))),
        np.ascontiguousarray(np.tile(np.repeat(y, sx), nz + 1)),
        np.ascontiguousarray(np.repeat(z, sx * sy)),
    )

    def node(i, j, k):
        return 1 + i + sx * (j + sy * k)

    k, j, i = np.indices((nz, ny, nx), dtype=np.int64)
    cells = np.stack((node(i, j, k), node(i + 1, j, k),
                      node(i + 1, j + 1, k), node(i, j + 1, k),
                      node(i, j, k + 1), node(i + 1, j, k + 1),
                      node(i + 1, j + 1, k + 1), node(i, j + 1, k + 1)),
                     axis=-1).reshape(-1, 8)
    del k, j, i
    k, j = np.indices((nz, ny), dtype=np.int64)
    xmin = np.stack((node(0, j, k), node(0, j, k + 1),
                     node(0, j + 1, k + 1), node(0, j + 1, k)), axis=-1)
    xmax = np.stack((node(nx, j, k), node(nx, j + 1, k),
                     node(nx, j + 1, k + 1), node(nx, j, k + 1)), axis=-1)
    k, i = np.indices((nz, nx), dtype=np.int64)
    ymin = np.stack((node(i, 0, k), node(i + 1, 0, k),
                     node(i + 1, 0, k + 1), node(i, 0, k + 1)), axis=-1)
    ymax = np.stack((node(i, ny, k), node(i, ny, k + 1),
                     node(i + 1, ny, k + 1), node(i + 1, ny, k)), axis=-1)
    j, i = np.indices((ny, nx), dtype=np.int64)
    zmin = np.stack((node(i, j, 0), node(i, j + 1, 0),
                     node(i + 1, j + 1, 0), node(i + 1, j, 0)), axis=-1)
    zmax = np.stack((node(i, j, nz), node(i + 1, j, nz),
                     node(i + 1, j + 1, nz), node(i, j + 1, nz)), axis=-1)
    boundaries = tuple(np.ascontiguousarray(a.reshape(-1, 4))
                       for a in (xmin, xmax, ymin, ymax, zmin, zmax))
    return coords, np.ascontiguousarray(cells), boundaries


def write_mesh(path: Path, nx: int, ny: int, nz: int, depth: float) -> None:
    """Write atomically through the project's CGNS C library."""
    partial = path.with_suffix(".partial.cgns")
    if path.exists() or partial.exists() or path.with_suffix(".metadata.json").exists():
        raise FileExistsError(f"refusing to overwrite {path} or its sidecars")
    path.parent.mkdir(parents=True, exist_ok=True)
    coords, cells, boundaries = make_arrays(nx, ny, nz, depth)
    lib = cgns_library(ROOT)
    file_id = ctypes.c_int()
    check(lib, lib.cg_open(str(partial).encode(), 1, ctypes.byref(file_id)), "cg_open")
    try:
        base, zone, section, coord_id = (ctypes.c_int() for _ in range(4))
        check(lib, lib.cg_base_write(file_id.value, b"Base", 3, 3,
                                     ctypes.byref(base)), "cg_base_write")
        sizes = (CG_SIZE * 3)(len(coords[0]), len(cells), 0)
        check(lib, lib.cg_zone_write(file_id.value, base.value, b"blk-1", sizes,
                                     3, ctypes.byref(zone)), "cg_zone_write")
        for name, values in zip((b"CoordinateX", b"CoordinateY", b"CoordinateZ"), coords):
            check(lib, lib.cg_coord_write(file_id.value, base.value, zone.value,
                                          4, name, pointer(values, ctypes.c_double),
                                          ctypes.byref(coord_id)), name.decode())
        check(lib, lib.cg_section_write(file_id.value, base.value, zone.value,
                                        b"HexElements", 17, 1, len(cells), 0,
                                        pointer(cells, CG_SIZE), ctypes.byref(section)),
              "cg_section_write HexElements")
        first = len(cells) + 1
        for name, faces in zip(BOUNDARY_NAMES, boundaries):
            last = first + len(faces) - 1
            check(lib, lib.cg_section_write(file_id.value, base.value, zone.value,
                                            name.encode(), 7, first, last, 0,
                                            pointer(faces, CG_SIZE), ctypes.byref(section)),
                  f"cg_section_write {name}")
            region = (CG_SIZE * 2)(first, last)
            bc_id = ctypes.c_int()
            check(lib, lib.cg_boco_write(file_id.value, base.value, zone.value,
                                         name.encode(), 0, 4, 2, region,
                                         ctypes.byref(bc_id)), f"cg_boco_write {name}")
            check(lib, lib.cg_boco_gridlocation_write(file_id.value, base.value,
                                                       zone.value, bc_id.value, 4),
                  f"cg_boco_gridlocation_write {name}")
            first = last + 1
        check(lib, lib.cg_close(file_id.value), "cg_close")
    except Exception:
        lib.cg_close(file_id.value)
        partial.unlink(missing_ok=True)
        raise
    partial.replace(path)
    metadata = verify_mesh(path, nx, ny, nz, depth)
    path.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


def verify_mesh(path: Path, nx: int, ny: int, nz: int, depth: float) -> dict:
    """Check on-disk counts, coordinates, all hexahedra, and boundary normals."""
    expected_nodes = (nx + 1) * (ny + 1) * (nz + 1)
    expected_cells = nx * ny * nz
    counts = (ny * nz, ny * nz, nx * nz, nx * nz, nx * ny, nx * ny)
    with h5py.File(path, "r") as mesh:
        zone = mesh["Base/blk-1"]
        assert np.array_equal(zone[" data"][:].reshape(-1),
                              [expected_nodes, expected_cells, 0])
        xyz = np.stack([zone[f"GridCoordinates/Coordinate{axis}/ data"][:]
                        for axis in "XYZ"], axis=-1)
        assert xyz.shape == (expected_nodes, 3)
        assert np.array_equal(xyz.min(axis=0), [0.0, 0.0, 0.0])
        assert np.array_equal(xyz.max(axis=0), [1.0, 1.0, depth])
        connectivity = zone["HexElements/ElementConnectivity/ data"][:].reshape(-1, 8)
        assert connectivity.shape == (expected_cells, 8)
        assert connectivity.min() == 1 and connectivity.max() == expected_nodes
        origin = xyz[connectivity[:, 0] - 1]
        ex = xyz[connectivity[:, 1] - 1] - origin
        ey = xyz[connectivity[:, 3] - 1] - origin
        ez = xyz[connectivity[:, 4] - 1] - origin
        jacobian = np.einsum("ij,ij->i", np.cross(ex, ey), ez)
        np.testing.assert_allclose(jacobian, depth / expected_cells, rtol=1e-12)
        first = expected_cells + 1
        for side, (name, count) in enumerate(zip(BOUNDARY_NAMES, counts)):
            faces = zone[f"{name}/ElementConnectivity/ data"][:].reshape(-1, 4)
            assert faces.shape == (count, 4)
            assert np.array_equal(zone[f"{name}/ElementRange/ data"][:],
                                  [first, first + count - 1])
            assert np.array_equal(zone[f"ZoneBC/{name}/PointRange/ data"][:].reshape(-1),
                                  [first, first + count - 1])
            p = xyz[faces - 1]
            normal = np.cross(p[:, 1] - p[:, 0], p[:, 3] - p[:, 0])
            axis, sign = side // 2, (-1 if side % 2 == 0 else 1)
            assert np.all(sign * normal[:, axis] > 0)
            assert np.allclose(np.delete(normal, axis, axis=1), 0.0)
            first += count
    with path.open("rb") as stream:
        sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    return {
        "file": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
        "cells_per_axis": [nx, ny, nz], "cell_count": expected_cells,
        "node_count": expected_nodes, "element_type": "HEXA_8",
        "bounds": [[0.0, 1.0], [0.0, 1.0], [0.0, depth]],
        "spacing": [1.0 / nx, 1.0 / ny, depth / nz],
        "boundary_faces": dict(zip(BOUNDARY_NAMES, counts)),
        "total_boundary_faces": sum(counts),
        "all_cell_volumes_positive": True, "all_boundary_normals_outward": True,
        "size_bytes": path.stat().st_size, "sha256": sha256,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=400)
    parser.add_argument("--ny", type=int, default=400)
    parser.add_argument("--nz", type=int, default=5)
    parser.add_argument("--depth", type=float, default=0.0125)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "data/mesh/riemann_barth_400x400x5.cgns")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if min(args.nx, args.ny, args.nz) < 1 or not np.isfinite(args.depth) or args.depth <= 0:
        parser.error("cell counts and depth must be positive")
    output = args.output.resolve()
    if args.verify_only:
        print(json.dumps(verify_mesh(output, args.nx, args.ny, args.nz, args.depth), indent=2))
    else:
        write_mesh(output, args.nx, args.ny, args.nz, args.depth)


if __name__ == "__main__":
    main()
