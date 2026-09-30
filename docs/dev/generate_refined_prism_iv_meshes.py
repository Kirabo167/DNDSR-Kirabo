#!/usr/bin/env python3
"""Uniformly refine the original IV80 triangular plane and extrude Prism6 meshes.

Each 2-D triangle is split into four by shared edge midpoints.  Every refinement
also doubles the number of z slabs, so IV160 and IV320 form a genuine 3-D
refinement sequence of the *same* nonuniform IV80 prism mesh.  Existing files
are never overwritten.
"""

from __future__ import annotations

import argparse
import ctypes
import json
from pathlib import Path

import h5py
import numpy as np


CG_SIZE = ctypes.c_long
ZONE_UNSTRUCTURED = 3
REAL_DOUBLE = 4
PENTA_6 = 14
TRI_3 = 5
QUAD_4 = 7
BC_TYPE_NULL = 0
POINT_RANGE = 4
FACE_CENTER = 4
BOUNDARY_NAMES = ("bc-2", "bc-2-1", "bc-3", "bc-3-1", "bc-4", "bc-4-1")


def load_source(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path) as mesh:
        zone = mesh["Base/blk-1"]
        xyz = np.column_stack([
            zone[f"GridCoordinates/Coordinate{axis}/ data"][:].ravel()
            for axis in "XYZ"
        ])
        original = zone["PrismElements/ElementConnectivity/ data"][:].reshape(-1, 6) - 1
    plane_nodes = int(np.count_nonzero(xyz[:, 2] == 0))
    slabs = len(np.unique(xyz[:, 2])) - 1
    if len(xyz) != plane_nodes * (slabs + 1) or len(original) % slabs:
        raise ValueError("source is not a layered prism mesh")
    triangles = original[:len(original) // slabs, :3].astype(np.int64)
    if not np.array_equal(original[:len(triangles), 3:], triangles + plane_nodes):
        raise ValueError("source prism top nodes are not aligned to the bottom layer")
    if np.any(triangles < 0) or np.any(triangles >= plane_nodes):
        raise ValueError("source base triangle indexing is inconsistent")
    xy = xyz[:plane_nodes, :2].copy()
    area2 = ((xy[triangles[:, 1], 0] - xy[triangles[:, 0], 0]) *
             (xy[triangles[:, 2], 1] - xy[triangles[:, 0], 1]) -
             (xy[triangles[:, 2], 0] - xy[triangles[:, 0], 0]) *
             (xy[triangles[:, 1], 1] - xy[triangles[:, 0], 1]))
    if np.min(area2) <= 0:
        raise ValueError("source base triangles are not all counterclockwise")
    if slabs != 32 or plane_nodes != 7660 or len(triangles) != 14998:
        raise ValueError("source is not the audited IV80 prism mesh")
    return xy, triangles


def refine(xy: np.ndarray, triangles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    edges = np.concatenate((triangles[:, [0, 1]], triangles[:, [1, 2]],
                            triangles[:, [2, 0]]))
    unique, inverse = np.unique(np.sort(edges, axis=1), axis=0, return_inverse=True)
    n = len(xy)
    a, b, c = triangles.T
    ab, bc, ca = inverse.reshape(3, -1) + n
    child = np.empty((4 * len(triangles), 3), dtype=np.int64)
    child[0::4] = np.column_stack((a, ab, ca))
    child[1::4] = np.column_stack((ab, b, bc))
    child[2::4] = np.column_stack((ca, bc, c))
    child[3::4] = np.column_stack((ab, bc, ca))
    middle = 0.5 * (xy[unique[:, 0]] + xy[unique[:, 1]])
    return np.vstack((xy, middle)), child


def boundary_edges(xy: np.ndarray, triangles: np.ndarray, size: int) -> dict[str, np.ndarray]:
    directed = np.concatenate((triangles[:, [0, 1]], triangles[:, [1, 2]],
                               triangles[:, [2, 0]]))
    _, first, count = np.unique(np.sort(directed, axis=1), axis=0,
                                return_index=True, return_counts=True)
    boundary = directed[first[count == 1]]
    if len(boundary) != 4 * size:
        raise ValueError(f"expected {4 * size} planar boundary edges, got {len(boundary)}")
    edge_xy = xy[boundary]
    selectors = {
        "bc-2": np.all(np.abs(edge_xy[:, :, 0]) < 1e-10, axis=1),
        "bc-2-1": np.all(np.abs(edge_xy[:, :, 0] - 10) < 1e-10, axis=1),
        "bc-3": np.all(np.abs(edge_xy[:, :, 1]) < 1e-10, axis=1),
        "bc-3-1": np.all(np.abs(edge_xy[:, :, 1] - 10) < 1e-10, axis=1),
    }
    if np.any(np.sum(np.stack(list(selectors.values())), axis=0) != 1):
        raise ValueError("a planar boundary edge was unclassified or multiply classified")
    selected = {name: boundary[mask] for name, mask in selectors.items()}
    if any(len(edges) != size for edges in selected.values()):
        raise ValueError("the four planar boundary sides have inconsistent refinement")
    return selected


def make_cgns_library(repository_root: Path) -> ctypes.CDLL:
    path = repository_root / "external/cfd_externals/install/lib/libcgns.so"
    cgns = ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
    cgns.cg_get_error.restype = ctypes.c_char_p
    cgns.cg_open.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
    cgns.cg_close.argtypes = [ctypes.c_int]
    cgns.cg_base_write.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
                                   ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
    cgns.cg_zone_write.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_char_p,
                                   ctypes.POINTER(CG_SIZE), ctypes.c_int,
                                   ctypes.POINTER(ctypes.c_int)]
    cgns.cg_coord_write.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                    ctypes.c_int, ctypes.c_char_p,
                                    ctypes.POINTER(ctypes.c_double),
                                    ctypes.POINTER(ctypes.c_int)]
    cgns.cg_section_write.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                      ctypes.c_char_p, ctypes.c_int, CG_SIZE, CG_SIZE,
                                      ctypes.c_int, ctypes.POINTER(CG_SIZE),
                                      ctypes.POINTER(ctypes.c_int)]
    cgns.cg_boco_write.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                   ctypes.c_char_p, ctypes.c_int, ctypes.c_int,
                                   CG_SIZE, ctypes.POINTER(CG_SIZE),
                                   ctypes.POINTER(ctypes.c_int)]
    cgns.cg_boco_gridlocation_write.argtypes = [ctypes.c_int, ctypes.c_int,
                                                 ctypes.c_int, ctypes.c_int,
                                                 ctypes.c_int]
    return cgns


def check(cgns: ctypes.CDLL, status: int, operation: str) -> None:
    if status:
        raise RuntimeError(f"{operation}: {cgns.cg_get_error().decode()}")


def size_pointer(values: np.ndarray) -> ctypes.POINTER(CG_SIZE):
    if values.dtype != np.int64 or not values.flags.c_contiguous:
        raise TypeError("CGNS connectivity needs contiguous int64")
    return values.ctypes.data_as(ctypes.POINTER(CG_SIZE))


def write_mesh(path: Path, xy: np.ndarray, triangles: np.ndarray, size: int,
               source: Path, cgns: ctypes.CDLL) -> dict[str, object]:
    manifest_path = path.with_suffix(".manifest.json")
    partial = path.with_suffix(path.suffix + ".partial")
    if path.exists() or manifest_path.exists() or partial.exists():
        raise FileExistsError(f"refusing to overwrite {path} or its manifest/partial")
    n_plane = len(xy)
    n_slabs = 32 * size // 80
    n_cells = len(triangles) * n_slabs
    n_nodes = n_plane * (n_slabs + 1)
    edges = boundary_edges(xy, triangles, size)
    coordinates = (
        np.ascontiguousarray(np.tile(xy[:, 0], n_slabs + 1), dtype=np.float64),
        np.ascontiguousarray(np.tile(xy[:, 1], n_slabs + 1), dtype=np.float64),
        np.ascontiguousarray(np.repeat(np.linspace(0.0, 4.0, n_slabs + 1), n_plane),
                              dtype=np.float64),
    )
    volume = np.empty((n_cells, 6), dtype=np.int64)
    for k in range(n_slabs):
        block = volume[k * len(triangles):(k + 1) * len(triangles)]
        block[:, :3] = triangles + 1 + k * n_plane
        block[:, 3:] = triangles + 1 + (k + 1) * n_plane
    boundary: dict[str, np.ndarray] = {}
    for name, side in edges.items():
        faces = np.empty((len(side) * n_slabs, 4), dtype=np.int64)
        for k in range(n_slabs):
            block = faces[k * len(side):(k + 1) * len(side)]
            block[:, :2] = side + 1 + k * n_plane
            block[:, 2] = side[:, 1] + 1 + (k + 1) * n_plane
            block[:, 3] = side[:, 0] + 1 + (k + 1) * n_plane
        boundary[name] = faces
    boundary["bc-4"] = np.ascontiguousarray(triangles[:, ::-1] + 1)
    boundary["bc-4-1"] = np.ascontiguousarray(triangles + 1 + n_slabs * n_plane)

    file_number = ctypes.c_int()
    check(cgns, cgns.cg_open(str(partial).encode(), 1, ctypes.byref(file_number)), "cg_open")
    try:
        base = ctypes.c_int()
        zone = ctypes.c_int()
        section = ctypes.c_int()
        coordinate_id = ctypes.c_int()
        check(cgns, cgns.cg_base_write(file_number.value, b"Base", 3, 3,
                                       ctypes.byref(base)), "cg_base_write")
        sizes = (CG_SIZE * 3)(n_nodes, n_cells, 0)
        check(cgns, cgns.cg_zone_write(file_number.value, base.value, b"blk-1", sizes,
                                       ZONE_UNSTRUCTURED, ctypes.byref(zone)), "cg_zone_write")
        for name, values in zip((b"CoordinateX", b"CoordinateY", b"CoordinateZ"), coordinates):
            check(cgns, cgns.cg_coord_write(file_number.value, base.value, zone.value,
                                            REAL_DOUBLE, name,
                                            values.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                            ctypes.byref(coordinate_id)), name.decode())
        check(cgns, cgns.cg_section_write(file_number.value, base.value, zone.value,
                                          b"PrismElements", PENTA_6, 1, n_cells, 0,
                                          size_pointer(volume), ctypes.byref(section)),
              "cg_section_write PrismElements")
        element_start = n_cells + 1
        ranges: dict[str, tuple[int, int]] = {}
        for name in BOUNDARY_NAMES:
            faces = boundary[name]
            last = element_start + len(faces) - 1
            kind = TRI_3 if name.startswith("bc-4") else QUAD_4
            check(cgns, cgns.cg_section_write(file_number.value, base.value, zone.value,
                                              name.encode(), kind, element_start, last, 0,
                                              size_pointer(faces), ctypes.byref(section)),
                  f"cg_section_write {name}")
            ranges[name] = (element_start, last)
            element_start = last + 1
        bc_id = ctypes.c_int()
        for name, (first, last) in ranges.items():
            element_range = (CG_SIZE * 2)(first, last)
            check(cgns, cgns.cg_boco_write(file_number.value, base.value, zone.value,
                                           name.encode(), BC_TYPE_NULL, POINT_RANGE, 2,
                                           element_range, ctypes.byref(bc_id)),
                  f"cg_boco_write {name}")
            check(cgns, cgns.cg_boco_gridlocation_write(file_number.value, base.value,
                                                        zone.value, bc_id.value, FACE_CENTER),
                  f"cg_boco_gridlocation_write {name}")
    except Exception:
        cgns.cg_close(file_number.value)
        raise
    check(cgns, cgns.cg_close(file_number.value), "cg_close")
    partial.replace(path)
    manifest = {
        "source_mesh": str(source), "generation": "shared-edge 2D midpoint refinement and z bisection",
        "size": size, "plane_nodes": n_plane, "plane_triangles": len(triangles),
        "z_slabs": n_slabs, "nodes": n_nodes, "prism_cells": n_cells,
        "boundary_faces": {name: len(faces) for name, faces in boundary.items()},
        "domain_lengths": [10.0, 10.0, 4.0],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=root / "cases/NCFV/iv80_3d_1_dnds.cgns")
    parser.add_argument("--output-dir", type=Path, default=root / "cases/NCFV")
    parser.add_argument("--sizes", type=int, nargs="+", default=[160, 320])
    args = parser.parse_args()
    if any(size not in (160, 320) for size in args.sizes):
        parser.error("only IV160 and IV320 uniform refinements of IV80 are supported")
    xy, triangles = load_source(args.source)
    cgns = make_cgns_library(root)
    for size in sorted(set(args.sizes)):
        xy, triangles = refine(xy, triangles)
        if size == 320 and 160 not in args.sizes:
            xy, triangles = refine(xy, triangles)
        output = args.output_dir / f"refined_iv{size}_3d_1_dnds.cgns"
        print(json.dumps(write_mesh(output, xy, triangles, size, args.source, cgns)), flush=True)


if __name__ == "__main__":
    main()
