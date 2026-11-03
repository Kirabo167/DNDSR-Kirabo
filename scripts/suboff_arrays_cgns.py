#!/usr/bin/env python3
"""Validate first-order tetra/prism/pyramid arrays and write ACM-compatible CGNS."""

import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path

import numpy as np


def determinant(a, b, c):
    return np.einsum("ij,ij->i", np.cross(a, b), c)


def jacobians(points, cells):
    p = points[cells]
    width = cells.shape[1]
    if width == 4:
        return determinant(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0], p[:, 3] - p[:, 0])[:, None]
    if width == 6:
        values = []
        for z in (0., 0.5, 1.):
            dr = (1 - z) * (p[:, 1] - p[:, 0]) + z * (p[:, 4] - p[:, 3])
            ds = (1 - z) * (p[:, 2] - p[:, 0]) + z * (p[:, 5] - p[:, 3])
            for r, s in ((0., 0.), (1., 0.), (0., 1.), (1 / 3, 1 / 3)):
                dz = ((1 - r - s) * (p[:, 3] - p[:, 0]) +
                      r * (p[:, 4] - p[:, 1]) + s * (p[:, 5] - p[:, 2]))
                values.append(determinant(dr, ds, dz))
        samples = np.column_stack(values)
        # At a triangular corner the determinant is quadratic through the
        # thickness. Include its interior minimum, not just endpoint samples.
        for corner in range(3):
            j0, jm, j1 = samples[:, corner], samples[:, 4 + corner], samples[:, 8 + corner]
            a = 2 * (j1 + j0 - 2 * jm)
            b = j1 - j0 - a
            zmin = np.divide(-b, 2*a, out=np.zeros_like(a), where=a != 0)
            inside = (a > 0) & (zmin > 0) & (zmin < 1)
            jmin = np.where(inside, a*zmin*zmin + b*zmin + j0, np.minimum(j0, j1))
            values.append(jmin)
        return np.column_stack(values)
    if width == 5:
        values = []
        for r, s in ((0., 0.), (1., 0.), (1., 1.), (0., 1.), (.5, .5)):
            dr = (1 - s) * (p[:, 1] - p[:, 0]) + s * (p[:, 2] - p[:, 3])
            ds = (1 - r) * (p[:, 3] - p[:, 0]) + r * (p[:, 2] - p[:, 1])
            base = ((1-r)*(1-s)*p[:, 0] + r*(1-s)*p[:, 1] +
                    r*s*p[:, 2] + (1-r)*s*p[:, 3])
            values.append(determinant(dr, ds, p[:, 4] - base))
        return np.column_stack(values)
    raise ValueError(f"Unsupported cell width {width}")


def orient_validate(points, cells):
    minimum = np.inf
    for start in range(0, len(cells), 100_000):
        block = cells[start:start + 100_000]
        jac = jacobians(points, block)
        inverted = np.all(jac < 0, axis=1)
        if cells.shape[1] == 4:
            permutation = [0, 2, 1, 3]
        elif cells.shape[1] == 6:
            permutation = [0, 2, 1, 3, 5, 4]
        else:
            permutation = [0, 3, 2, 1, 4]
        block[inverted] = block[inverted][:, permutation]
        jac = jacobians(points, block)
        invalid = np.any(jac <= 0, axis=1)
        if np.any(invalid):
            raise ValueError(f"{np.count_nonzero(invalid)} invalid {cells.shape[1]}-node cells in block {start}")
        minimum = min(minimum, float(jac.min()))
    return minimum


def write_cgns(output, library, points, blocks, boundaries):
    lib = ct.CDLL(str(library))
    integer = ct.c_int
    size = ct.c_int64
    ip = ct.POINTER(integer)
    sp = ct.POINTER(size)
    lib.cg_get_error.restype = ct.c_char_p
    signatures = {
        "cg_open": [ct.c_char_p, integer, ip], "cg_close": [integer],
        "cg_base_write": [integer, ct.c_char_p, integer, integer, ip],
        "cg_zone_write": [integer, integer, ct.c_char_p, sp, integer, ip],
        "cg_coord_write": [integer, integer, integer, integer, ct.c_char_p, ct.c_void_p, ip],
        "cg_section_write": [integer, integer, integer, ct.c_char_p, integer, size, size, integer, sp, ip],
        "cg_boco_write": [integer, integer, integer, ct.c_char_p, integer, integer, size, sp, ip],
        "cg_boco_gridlocation_write": [integer, integer, integer, integer, integer],
    }
    for name, signature in signatures.items():
        getattr(lib, name).argtypes = signature

    def call(name, *args):
        if getattr(lib, name)(*args):
            raise RuntimeError(f"{name}: {lib.cg_get_error().decode()}")

    number, base, zone = integer(), integer(), integer()
    call("cg_set_compress", 4)
    call("cg_open", str(output).encode(), 1, ct.byref(number))
    try:
        call("cg_base_write", number, b"Base", 3, 3, ct.byref(base))
        zone_size = np.array([len(points), sum(len(b) for b in blocks.values()), 0], dtype=np.int64)
        call("cg_zone_write", number, base, b"SUBOFF", zone_size.ctypes.data_as(sp), 3, ct.byref(zone))
        for axis, name in enumerate((b"CoordinateX", b"CoordinateY", b"CoordinateZ")):
            values = np.ascontiguousarray(points[:, axis], dtype=np.float64)
            index = integer()
            call("cg_coord_write", number, base, zone, 4, name, values.ctypes.data_as(ct.c_void_p), ct.byref(index))
        cursor = 1
        for width, block in blocks.items():
            kind, name = {4: (10, b"TETRA_4"), 5: (12, b"PYRA_5"), 6: (14, b"PENTA_6")}[width]
            connectivity = np.ascontiguousarray(block + 1, dtype=np.int64)
            section = integer()
            call("cg_section_write", number, base, zone, name, kind, cursor,
                 cursor + len(block) - 1, 0, connectivity.ctypes.data_as(sp), ct.byref(section))
            cursor += len(block)
        for name, block in boundaries.items():
            connectivity = np.ascontiguousarray(block + 1, dtype=np.int64)
            section, boundary = integer(), integer()
            kind = {3: 5, 4: 7}[block.shape[1]]
            call("cg_section_write", number, base, zone, (name + "_FACES").encode(), kind,
                 cursor, cursor + len(block) - 1, 0, connectivity.ctypes.data_as(sp), ct.byref(section))
            indices = np.arange(cursor, cursor + len(block), dtype=np.int64)
            bc_type = {"WALL": 22, "INLET": 9, "OUTLET": 13, "FAR": 7}[name]
            call("cg_boco_write", number, base, zone, name.encode(), bc_type, 2,
                 len(block), indices.ctypes.data_as(sp), ct.byref(boundary))
            call("cg_boco_gridlocation_write", number, base, zone, boundary, 4)
            cursor += len(block)
    finally:
        call("cg_close", number)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arrays", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--library", type=Path, required=True)
    args = parser.parse_args()
    if args.arrays.suffix == ".vol":
        from netgen.meshing import Mesh
        mesh = Mesh(dim=3)
        mesh.Load(str(args.arrays))
        volumes = mesh.Elements3D().NumPy()
        surfaces = mesh.Elements2D().NumPy()
        external = [i for i, descriptor in enumerate(mesh.FaceDescriptors(), 1)
                    if descriptor.domin == 0 or descriptor.domout == 0]
        selected = np.isin(surfaces["index"], external)
        data = dict(points=np.array([tuple(p.p) for p in mesh.Points()]),
                    volume_nodes=volumes["nodes"], volume_sizes=volumes["np"],
                    surface_nodes=surfaces["nodes"][selected],
                    surface_sizes=surfaces["np"][selected])
    else:
        data = np.load(args.arrays)
    points = data["points"]
    blocks, quality = {}, {}
    for width in np.unique(data["volume_sizes"]):
        block = np.ascontiguousarray(data["volume_nodes"][data["volume_sizes"] == width, :width] - 1, dtype=np.int64)
        print("VALIDATE", width, len(block), flush=True)
        quality[str(width)] = orient_validate(points, block)
        blocks[int(width)] = block
    count = sum(map(len, blocks.values()))
    if not 2_000_000 < count < 10_000_000:
        raise ValueError(f"Volume cell count {count} is outside requested bounds")
    boundaries = {}
    for width in np.unique(data["surface_sizes"]):
        faces = data["surface_nodes"][data["surface_sizes"] == width, :width].astype(np.int64) - 1
        centers = points[faces].mean(axis=1)
        wall = (centers[:, 0] > -.01) & (centers[:, 0] < 4.37) & (np.linalg.norm(centers[:, 1:], axis=1) < .7)
        inlet = np.isclose(centers[:, 0], -13.068, atol=1e-6)
        outlet = np.isclose(centers[:, 0], 30.492, atol=1e-6)
        for name, mask in dict(WALL=wall, INLET=inlet, OUTLET=outlet, FAR=~(wall | inlet | outlet)).items():
            if np.any(mask):
                if name in boundaries:
                    raise ValueError("Mixed triangle/quad boundaries need separate named sections")
                boundaries[name] = faces[mask]
    write_cgns(args.output, args.library, points, blocks, boundaries)
    summary = dict(cells=count, nodes=len(points),
                   cells_by_nodes={str(k): len(v) for k, v in blocks.items()},
                   boundary_faces={k: len(v) for k, v in boundaries.items()},
                   minimum_sampled_jacobian=quality,
                   sha256=hashlib.sha256(args.output.read_bytes()).hexdigest())
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    np.savez(args.output.with_suffix(".validated.npz"), points=points,
             **{f"cells_{k}": v for k, v in blocks.items()},
             **{f"boundary_{k}": v for k, v in boundaries.items()})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
