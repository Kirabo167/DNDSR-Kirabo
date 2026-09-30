#!/usr/bin/env python3
"""Generate triply periodic Cartesian CGNS meshes for the 3-D Taylor–Green vortex.

The domain is [-pi, pi]^3.  Each file contains HEXA_8 volume elements and
six named QUAD_4 FaceCenter boundary zones, ordered as x-/x+, y-/y+, z-/z+.
Run from anywhere with the repository virtual environment's Python.
"""

from __future__ import annotations

import argparse
import ctypes
import math
from pathlib import Path

import numpy as np


CG_SIZE = ctypes.c_long
BOUNDARY_NAMES = ("bc-2", "bc-2-1", "bc-3", "bc-3-1", "bc-4", "bc-4-1")


def cgns_library(root: Path) -> ctypes.CDLL:
    path = root / "external/cfd_externals/install/lib/libcgns.so"
    if not path.exists():
        raise FileNotFoundError(f"CGNS library not found: {path}")
    lib = ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
    lib.cg_get_error.restype = ctypes.c_char_p
    lib.cg_open.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
    lib.cg_close.argtypes = [ctypes.c_int]
    lib.cg_base_write.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
                                  ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
    lib.cg_zone_write.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_char_p,
                                  ctypes.POINTER(CG_SIZE), ctypes.c_int,
                                  ctypes.POINTER(ctypes.c_int)]
    lib.cg_coord_write.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                   ctypes.c_int, ctypes.c_char_p,
                                   ctypes.POINTER(ctypes.c_double),
                                   ctypes.POINTER(ctypes.c_int)]
    lib.cg_section_write.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                     ctypes.c_char_p, ctypes.c_int, CG_SIZE, CG_SIZE,
                                     ctypes.c_int, ctypes.POINTER(CG_SIZE),
                                     ctypes.POINTER(ctypes.c_int)]
    lib.cg_boco_write.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                  ctypes.c_char_p, ctypes.c_int, ctypes.c_int,
                                  CG_SIZE, ctypes.POINTER(CG_SIZE),
                                  ctypes.POINTER(ctypes.c_int)]
    lib.cg_boco_gridlocation_write.argtypes = [ctypes.c_int] * 5
    return lib


def check(lib: ctypes.CDLL, status: int, operation: str) -> None:
    if status:
        raise RuntimeError(f"{operation}: {lib.cg_get_error().decode(errors='replace')}")


def pointer(values: np.ndarray, element: type[ctypes._SimpleCData]):
    return values.ctypes.data_as(ctypes.POINTER(element))


def make_arrays(n: int) -> tuple[tuple[np.ndarray, ...], np.ndarray, tuple[np.ndarray, ...]]:
    side = n + 1
    plane = side * side
    grid = np.arange(side, dtype=np.float64) * (2 * math.pi / n) - math.pi
    coordinates = (
        np.ascontiguousarray(np.tile(grid, side * side)),
        np.ascontiguousarray(np.tile(np.repeat(grid, side), side)),
        np.ascontiguousarray(np.repeat(grid, plane)),
    )

    def node(i: np.ndarray, j: np.ndarray, k: np.ndarray) -> np.ndarray:
        return 1 + i + side * (j + side * k)

    k, j, i = np.indices((n, n, n), dtype=np.int64)
    cells = np.stack((node(i, j, k), node(i + 1, j, k),
                      node(i + 1, j + 1, k), node(i, j + 1, k),
                      node(i, j, k + 1), node(i + 1, j, k + 1),
                      node(i + 1, j + 1, k + 1), node(i, j + 1, k + 1)),
                     axis=-1).reshape(-1, 8)
    del i, j, k

    b, a = np.indices((n, n), dtype=np.int64)
    # Outward orientation follows the project's existing Cartesian CGNS writer.
    boundaries = (
        np.stack((node(0, a, b), node(0, a, b + 1),
                  node(0, a + 1, b + 1), node(0, a + 1, b)), axis=-1),
        np.stack((node(n, a, b), node(n, a + 1, b),
                  node(n, a + 1, b + 1), node(n, a, b + 1)), axis=-1),
        np.stack((node(a, 0, b), node(a + 1, 0, b),
                  node(a + 1, 0, b + 1), node(a, 0, b + 1)), axis=-1),
        np.stack((node(a, n, b), node(a, n, b + 1),
                  node(a + 1, n, b + 1), node(a + 1, n, b)), axis=-1),
        np.stack((node(a, b, 0), node(a, b + 1, 0),
                  node(a + 1, b + 1, 0), node(a + 1, b, 0)), axis=-1),
        np.stack((node(a, b, n), node(a + 1, b, n),
                  node(a + 1, b + 1, n), node(a, b + 1, n)), axis=-1),
    )
    return coordinates, np.ascontiguousarray(cells), tuple(
        np.ascontiguousarray(face.reshape(-1, 4)) for face in boundaries)


def write_mesh(path: Path, n: int, lib: ctypes.CDLL) -> None:
    if n < 2:
        raise ValueError("resolution must be at least 2")
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".partial.cgns")
    if partial.exists():
        raise FileExistsError(f"refusing to overwrite {partial}")
    coords, cells, boundaries = make_arrays(n)
    file_id = ctypes.c_int()
    check(lib, lib.cg_open(str(partial).encode(), 1, ctypes.byref(file_id)), "cg_open")
    try:
        base = ctypes.c_int()
        zone = ctypes.c_int()
        section = ctypes.c_int()
        coord_id = ctypes.c_int()
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
        partial.replace(path)
    except Exception:
        lib.cg_close(file_id.value)
        partial.unlink(missing_ok=True)
        raise


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[8, 16, 32, 64],
                        help="cells along each axis (default: 8 16 32 64)")
    parser.add_argument("--output-dir", type=Path, default=root / "data/mesh/taylor_green")
    args = parser.parse_args()
    if len(set(args.sizes)) != len(args.sizes) or any(n < 2 for n in args.sizes):
        parser.error("sizes must be unique integers of at least 2")
    lib = cgns_library(root)
    for n in args.sizes:
        path = args.output_dir / f"tgv_hex_{n}.cgns"
        write_mesh(path, n, lib)
        print(f"{path}: {n**3} HEXA_8 cells, {(n+1)**3} nodes, {6*n*n} boundary faces")


if __name__ == "__main__":
    main()
