"""Normalize legacy CGNS ElementList BCs to PointList/FaceCenter for DNDSR.

Coordinates, section ranges, connectivity and BC element IDs are preserved.
Uses the repository CGNS library, including its ADF reader; does not load pybind11.
"""

import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    if args.destination.exists():
        raise FileExistsError(args.destination)
    root = Path(__file__).resolve().parents[3]
    lib = ct.CDLL(str(root / "external/cfd_externals/install/lib/libcgns.so"))
    lib.cg_get_error.restype = ct.c_char_p
    integer, size = ct.c_int, ct.c_int64

    def call(name, *values):
        if getattr(lib, name)(*values):
            raise RuntimeError(f"{name}: {lib.cg_get_error().decode()}")

    src, dst = integer(), integer()
    call("cg_open", str(args.source).encode(), 0, ct.byref(src))
    call("cg_set_file_type", 2)
    call("cg_open", str(args.destination).encode(), 1, ct.byref(dst))
    bases = integer()
    call("cg_nbases", src, ct.byref(bases))
    if bases.value != 1:
        raise ValueError("This conversion requires the supplied one-base mesh")
    name, cell_dim, physical_dim = ct.create_string_buffer(33), integer(), integer()
    call("cg_base_read", src, 1, name, ct.byref(cell_dim), ct.byref(physical_dim))
    base = integer()
    call("cg_base_write", dst, name, cell_dim, physical_dim, ct.byref(base))
    zones = integer()
    call("cg_nzones", src, 1, ct.byref(zones))
    if zones.value != 1:
        raise ValueError("This conversion requires the supplied one-zone mesh")
    shape = (size * 9)()
    call("cg_zone_read", src, 1, 1, name, shape)
    zone = integer()
    call("cg_zone_write", dst, base, name, shape, 3, ct.byref(zone))
    lower, upper = size(1), size(shape[0])
    report = {"source": str(args.source), "nodes": shape[0], "cells": shape[1], "coordinates": {}, "sections": [], "boundaries": []}
    for axis in ("CoordinateX", "CoordinateY", "CoordinateZ"):
        values = np.empty(shape[0], dtype=np.float64)
        call("cg_coord_read", src, 1, 1, axis.encode(), 4, ct.byref(lower), ct.byref(upper), values.ctypes.data_as(ct.c_void_p))
        coord = integer()
        call("cg_coord_write", dst, base, zone, 4, axis.encode(), values.ctypes.data_as(ct.c_void_p), ct.byref(coord))
        report["coordinates"][axis] = {"min": float(values.min()), "max": float(values.max()), "sha256": hashlib.sha256(values.tobytes()).hexdigest()}
    sections = integer()
    call("cg_nsections", src, 1, 1, ct.byref(sections))
    for s in range(1, sections.value + 1):
        kind, start, end, boundary_count, parent_flag = integer(), size(), size(), integer(), integer()
        call("cg_section_read", src, 1, 1, s, name, ct.byref(kind), ct.byref(start), ct.byref(end), ct.byref(boundary_count), ct.byref(parent_flag))
        data_size = size()
        call("cg_ElementDataSize", src, 1, 1, s, ct.byref(data_size))
        connectivity = np.empty(data_size.value, dtype=np.int64)
        call("cg_elements_read", src, 1, 1, s, connectivity.ctypes.data_as(ct.c_void_p), None)
        output_section = integer()
        call("cg_section_write", dst, base, zone, name, kind, start, end, boundary_count, connectivity.ctypes.data_as(ct.c_void_p), ct.byref(output_section))
        report["sections"].append({"name": name.value.decode(), "type": kind.value, "start": start.value, "end": end.value, "sha256": hashlib.sha256(connectivity.tobytes()).hexdigest()})
    boundaries = integer()
    call("cg_nbocos", src, 1, 1, ct.byref(boundaries))
    for b in range(1, boundaries.value + 1):
        kind, point_type, count, normals_size, normal_type, datasets = integer(), integer(), size(), size(), integer(), integer()
        normal_index = (integer * 3)()
        call("cg_boco_info", src, 1, 1, b, name, ct.byref(kind), ct.byref(point_type), ct.byref(count), normal_index, ct.byref(normals_size), ct.byref(normal_type), ct.byref(datasets))
        if point_type.value not in (2, 4, 6, 7):
            raise ValueError(f"Unsupported point type {point_type.value}")
        points = np.empty(count.value, dtype=np.int64)
        call("cg_boco_read", src, 1, 1, b, points.ctypes.data_as(ct.c_void_p), None)
        # PointRange=4, ElementRange=6; PointList=2, ElementList=7.
        normalized = 4 if point_type.value in (4, 6) else 2
        output_bc = integer()
        call("cg_boco_write", dst, base, zone, name, kind, normalized, count, points.ctypes.data_as(ct.c_void_p), ct.byref(output_bc))
        call("cg_boco_gridlocation_write", dst, base, zone, output_bc, 4)
        report["boundaries"].append({"name": name.value.decode(), "old_point_type": point_type.value, "new_point_type": normalized, "count": count.value})
    call("cg_close", src)
    call("cg_close", dst)
    args.destination.with_suffix(".manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
