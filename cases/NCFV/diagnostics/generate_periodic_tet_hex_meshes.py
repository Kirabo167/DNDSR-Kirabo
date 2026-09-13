#!/usr/bin/env python3
"""Generate three-directionally periodic pure Tet4 and Hex8 CGNS mesh families.

The meshes cover [0, Lx] x [0, Ly] x [0, Lz], with ``N`` structured cells
per direction.  A Tet4 mesh splits every structured hexahedral block into six
conforming tetrahedra around its 000--111 body diagonal.  Boundary faces retain
the normal external names ``bc-2``, ``bc-2-1``, ..., ``bc-4-1`` rather than
the CGNS-reserved ``PERIODIC_*`` names: NCFV builds its own translational
periodic quotient from the duplicated box-boundary nodes and requires the
primal mesh itself to remain non-periodically connected.

No existing file is overwritten.  Each mesh is first written as ``.partial``
and atomically renamed only after CGNS closes successfully.
"""

from __future__ import annotations

import argparse
from array import array
import ctypes
import json
from pathlib import Path
from typing import Final


CG_MODE_WRITE: Final = 1
UNSTRUCTURED: Final = 3
REAL_DOUBLE: Final = 4
BC_TYPE_NULL: Final = 0
POINT_RANGE: Final = 4
FACE_CENTER: Final = 4
TRI_3: Final = 5
QUAD_4: Final = 7
TETRA_4: Final = 10
HEXA_8: Final = 17

CG_SIZE = ctypes.c_long
BOUNDARY_NAMES: Final = (
    "bc-2", "bc-2-1",  # x-min / x-max
    "bc-3", "bc-3-1",  # y-min / y-max
    "bc-4", "bc-4-1",  # z-min / z-max
)
TETRAHEDRA: Final = (
    (0, 1, 2, 6),
    (0, 2, 3, 6),
    (0, 3, 7, 6),
    (0, 7, 4, 6),
    (0, 4, 5, 6),
    (0, 5, 1, 6),
)
TET_BOUNDARY_FACES: Final = {
    "bc-2": ((0, 7, 3), (0, 4, 7)),
    "bc-2-1": ((1, 2, 6), (5, 1, 6)),
    "bc-3": ((0, 5, 4), (0, 1, 5)),
    "bc-3-1": ((2, 3, 6), (3, 7, 6)),
    "bc-4": ((0, 2, 1), (0, 3, 2)),
    "bc-4-1": ((7, 4, 6), (4, 5, 6)),
}
HEX_BOUNDARY_FACES: Final = {
    "bc-2": ((0, 4, 7, 3),),
    "bc-2-1": ((1, 2, 6, 5),),
    "bc-3": ((0, 1, 5, 4),),
    "bc-3-1": ((3, 7, 6, 2),),
    "bc-4": ((0, 3, 2, 1),),
    "bc-4-1": ((4, 5, 6, 7),),
}


def _load_cgns(repository_root: Path) -> ctypes.CDLL:
    library = repository_root / "external/cfd_externals/install/lib/libcgns.so"
    if not library.is_file():
        raise FileNotFoundError(f"CGNS shared library not found: {library}")
    cgns = ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)
    cgns.cg_get_error.restype = ctypes.c_char_p
    cgns.cg_open.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
    cgns.cg_close.argtypes = [ctypes.c_int]
    cgns.cg_base_write.argtypes = [
        ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int)
    ]
    cgns.cg_zone_write.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.POINTER(CG_SIZE), ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
    ]
    cgns.cg_coord_write.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int),
    ]
    cgns.cg_section_write.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_int, CG_SIZE,
        CG_SIZE, ctypes.c_int, ctypes.POINTER(CG_SIZE), ctypes.POINTER(ctypes.c_int),
    ]
    cgns.cg_boco_write.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_int,
        CG_SIZE, ctypes.POINTER(CG_SIZE), ctypes.POINTER(ctypes.c_int),
    ]
    cgns.cg_boco_gridlocation_write.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ]
    return cgns


def _check(status: int, operation: str, cgns: ctypes.CDLL) -> None:
    if status == 0:
        return
    error = cgns.cg_get_error()
    detail = error.decode(errors="replace") if error else "unknown CGNS error"
    raise RuntimeError(f"{operation} failed: {detail}")


def _cg_size_values(values: tuple[int, ...] | list[int]) -> ctypes.Array[CG_SIZE]:
    return (CG_SIZE * len(values))(*values)


def _cg_size_pointer(values: array) -> ctypes.POINTER[CG_SIZE]:
    if values.typecode != "q" or values.itemsize != ctypes.sizeof(CG_SIZE):
        raise TypeError("CGNS connectivity must be a native 64-bit signed array")
    view = (CG_SIZE * len(values)).from_buffer(values)
    return ctypes.cast(view, ctypes.POINTER(CG_SIZE))


def _double_pointer(values: array) -> ctypes.POINTER[ctypes.c_double]:
    if values.typecode != "d" or values.itemsize != ctypes.sizeof(ctypes.c_double):
        raise TypeError("CGNS coordinates must be native double arrays")
    view = (ctypes.c_double * len(values)).from_buffer(values)
    return ctypes.cast(view, ctypes.POINTER(ctypes.c_double))


def _node_id(i: int, j: int, k: int, nodes_per_direction: int) -> int:
    """Return the one-based CGNS index of a structured-grid node."""
    return (k * nodes_per_direction + j) * nodes_per_direction + i + 1


def _cube_nodes(i: int, j: int, k: int, nodes_per_direction: int) -> tuple[int, ...]:
    node = lambda di, dj, dk: _node_id(i + di, j + dj, k + dk, nodes_per_direction)
    return (
        node(0, 0, 0), node(1, 0, 0), node(1, 1, 0), node(0, 1, 0),
        node(0, 0, 1), node(1, 0, 1), node(1, 1, 1), node(0, 1, 1),
    )


def _coordinates(n: int, lengths: tuple[float, float, float]) -> tuple[array, array, array]:
    n_nodes = n + 1
    x, y, z = array("d"), array("d"), array("d")
    for k in range(n_nodes):
        coordinate_z = lengths[2] * k / n
        for j in range(n_nodes):
            coordinate_y = lengths[1] * j / n
            for i in range(n_nodes):
                x.append(lengths[0] * i / n)
                y.append(coordinate_y)
                z.append(coordinate_z)
    return x, y, z


def _volume_connectivity(topology: str, n: int) -> tuple[array, int, int]:
    nodes_per_direction = n + 1
    if topology == "hex":
        connectivity = array("q")
        for k in range(n):
            for j in range(n):
                for i in range(n):
                    connectivity.extend(_cube_nodes(i, j, k, nodes_per_direction))
        return connectivity, n**3, HEXA_8

    connectivity = array("q")
    for k in range(n):
        for j in range(n):
            for i in range(n):
                cube = _cube_nodes(i, j, k, nodes_per_direction)
                for tetrahedron in TETRAHEDRA:
                    connectivity.extend(cube[vertex] for vertex in tetrahedron)
    return connectivity, 6 * n**3, TETRA_4


def _append_faces(target: array, cube: tuple[int, ...], patterns: tuple[tuple[int, ...], ...]) -> None:
    for pattern in patterns:
        target.extend(cube[vertex] for vertex in pattern)


def _boundary_connectivity(topology: str, n: int) -> tuple[dict[str, array], int]:
    nodes_per_direction = n + 1
    patterns = HEX_BOUNDARY_FACES if topology == "hex" else TET_BOUNDARY_FACES
    boundaries = {name: array("q") for name in BOUNDARY_NAMES}

    for k in range(n):
        for j in range(n):
            _append_faces(boundaries["bc-2"], _cube_nodes(0, j, k, nodes_per_direction),
                          patterns["bc-2"])
            _append_faces(boundaries["bc-2-1"], _cube_nodes(n - 1, j, k, nodes_per_direction),
                          patterns["bc-2-1"])
    for k in range(n):
        for i in range(n):
            _append_faces(boundaries["bc-3"], _cube_nodes(i, 0, k, nodes_per_direction),
                          patterns["bc-3"])
            _append_faces(boundaries["bc-3-1"], _cube_nodes(i, n - 1, k, nodes_per_direction),
                          patterns["bc-3-1"])
    for j in range(n):
        for i in range(n):
            _append_faces(boundaries["bc-4"], _cube_nodes(i, j, 0, nodes_per_direction),
                          patterns["bc-4"])
            _append_faces(boundaries["bc-4-1"], _cube_nodes(i, j, n - 1, nodes_per_direction),
                          patterns["bc-4-1"])

    nodes_per_face = 4 if topology == "hex" else 3
    for name, faces in boundaries.items():
        if len(faces) != n * n * len(patterns[name]) * nodes_per_face:
            raise AssertionError(f"unexpected {name} boundary-connectivity size")
    return boundaries, QUAD_4 if topology == "hex" else TRI_3


def _write_mesh(path: Path, topology: str, n: int, lengths: tuple[float, float, float],
                repository_root: Path) -> dict[str, object]:
    if path.exists() or path.with_suffix(".manifest.json").exists():
        raise FileExistsError(f"refusing to overwrite existing mesh or manifest for {path}")
    partial = path.with_suffix(path.suffix + ".partial")
    if partial.exists():
        raise FileExistsError(f"partial mesh already exists; inspect it before retrying: {partial}")
    path.parent.mkdir(parents=True, exist_ok=True)

    coordinates = _coordinates(n, lengths)
    volume, n_cells, volume_type = _volume_connectivity(topology, n)
    boundaries, boundary_type = _boundary_connectivity(topology, n)
    n_nodes = (n + 1) ** 3
    expected_volume_entries = n_cells * (8 if topology == "hex" else 4)
    if len(volume) != expected_volume_entries:
        raise AssertionError("unexpected volume-connectivity size")

    cgns = _load_cgns(repository_root)
    file_number = ctypes.c_int()
    _check(cgns.cg_open(str(partial).encode(), CG_MODE_WRITE, ctypes.byref(file_number)),
           "cg_open", cgns)
    try:
        base = ctypes.c_int()
        _check(cgns.cg_base_write(file_number.value, b"Base", 3, 3, ctypes.byref(base)),
               "cg_base_write", cgns)
        zone = ctypes.c_int()
        sizes = _cg_size_values([n_nodes, n_cells, 0])
        _check(cgns.cg_zone_write(file_number.value, base.value, b"Zone", sizes, UNSTRUCTURED,
                                  ctypes.byref(zone)), "cg_zone_write", cgns)
        coordinate_id = ctypes.c_int()
        for name, values in zip((b"CoordinateX", b"CoordinateY", b"CoordinateZ"), coordinates):
            _check(cgns.cg_coord_write(file_number.value, base.value, zone.value, REAL_DOUBLE,
                                       name, _double_pointer(values), ctypes.byref(coordinate_id)),
                   f"cg_coord_write {name.decode()}", cgns)

        section = ctypes.c_int()
        _check(cgns.cg_section_write(file_number.value, base.value, zone.value,
                                     f"{topology.capitalize()}Elements".encode(), volume_type,
                                     1, n_cells, 0, _cg_size_pointer(volume), ctypes.byref(section)),
               "cg_section_write volume", cgns)

        element_start = n_cells + 1
        boundary_elements: dict[str, tuple[int, int]] = {}
        nodes_per_boundary_face = 4 if topology == "hex" else 3
        for name in BOUNDARY_NAMES:
            count = len(boundaries[name]) // nodes_per_boundary_face
            element_end = element_start + count - 1
            _check(cgns.cg_section_write(file_number.value, base.value, zone.value, name.encode(),
                                         boundary_type, element_start, element_end, 0,
                                         _cg_size_pointer(boundaries[name]), ctypes.byref(section)),
                   f"cg_section_write {name}", cgns)
            boundary_elements[name] = (element_start, element_end)
            element_start = element_end + 1

        boundary_id = ctypes.c_int()
        for name, (start, end) in boundary_elements.items():
            element_range = _cg_size_values([start, end])
            _check(cgns.cg_boco_write(file_number.value, base.value, zone.value, name.encode(),
                                      BC_TYPE_NULL, POINT_RANGE, 2, element_range,
                                      ctypes.byref(boundary_id)), f"cg_boco_write {name}", cgns)
            _check(cgns.cg_boco_gridlocation_write(file_number.value, base.value, zone.value,
                                                   boundary_id.value, FACE_CENTER),
                   f"cg_boco_gridlocation_write {name}", cgns)
    except Exception:
        cgns.cg_close(file_number.value)
        raise
    _check(cgns.cg_close(file_number.value), "cg_close", cgns)
    partial.replace(path)

    n_boundary_faces = sum((end - start + 1) for start, end in boundary_elements.values())
    return {
        "schema_version": 1,
        "mesh_file": str(path),
        "topology": "Hex8" if topology == "hex" else "Tet4",
        "structured_cells_per_direction": n,
        "domain_lengths": list(lengths),
        "nodes": n_nodes,
        "volume_cells": n_cells,
        "boundary_faces": n_boundary_faces,
        "boundary_element_type": "Quad4" if topology == "hex" else "Tri3",
        "boundary_zones": {
            "bc-2": "x-min periodic side",
            "bc-2-1": "x-max periodic side",
            "bc-3": "y-min periodic side",
            "bc-3-1": "y-max periodic side",
            "bc-4": "z-min periodic side",
            "bc-4-1": "z-max periodic side",
        },
        "periodicity": {
            "kind": "NCFV translational quotient",
            "lengths": list(lengths),
            "directions": ["x", "y", "z"],
        },
    }


def main() -> None:
    repository_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=repository_root / "cases/NCFV/generated_periodic_meshes")
    parser.add_argument("--sizes", type=int, nargs="+", default=[10, 20, 40, 80])
    parser.add_argument("--topologies", choices=("tet", "hex"), nargs="+",
                        default=["tet", "hex"])
    parser.add_argument("--lengths", type=float, nargs=3, default=(10.0, 10.0, 4.0),
                        metavar=("LX", "LY", "LZ"))
    args = parser.parse_args()

    if any(size < 2 for size in args.sizes):
        parser.error("every structured mesh size must be at least 2")
    if len(set(args.sizes)) != len(args.sizes):
        parser.error("mesh sizes must be unique")
    if any(length <= 0 for length in args.lengths):
        parser.error("domain lengths must be positive")

    lengths = tuple(float(length) for length in args.lengths)
    for topology in args.topologies:
        for size in args.sizes:
            output = args.output_dir / f"periodic_{topology}_iv{size}.cgns"
            manifest = _write_mesh(output, topology, size, lengths, repository_root)
            manifest_path = output.with_suffix(".manifest.json")
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
