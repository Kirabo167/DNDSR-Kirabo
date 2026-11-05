"""Reproduce the linked Windows HO project's planar TGV/entropy cases.

The mesh connectivity and node order follow the prepare_cases.py retrieved from
the user-linked thread. This is a standalone C++ driver, not a pybind11 test.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import ctypes as C
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess

import numpy as np


ROOT = Path(__file__).resolve().parents[4]
spec = importlib.util.spec_from_file_location("cgns_writer", ROOT / "scripts/generate_taylor_green_meshes.py")
writer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(writer)


def generate_mesh(destination, n):
    if destination.exists():
        raise RuntimeError(f"Refusing to replace existing mesh: {destination}")
    lib = writer.cgns_library(ROOT)
    length, nz, depth = 2 * math.pi, 2, 4 * math.pi / n
    node = lambda x, y, z: 1 + x + (n + 1) * y + (n + 1)**2 * z
    points = np.array([(length * x / n, length * y / n, depth * z / nz)
                       for z in range(nz + 1) for y in range(n + 1) for x in range(n + 1)])
    triangles = []
    for y in range(n):
        for x in range(n):
            a, b, c, d = (x, y), (x + 1, y), (x + 1, y + 1), (x, y + 1)
            triangles.extend([(a, b, c), (a, c, d)] if (x + y) % 2 == 0 else
                             [(a, b, d), (b, c, d)])
    prisms = [[node(x, y, z) for x, y in triangle] +
              [node(x, y, z + 1) for x, y in triangle]
              for z in range(nz) for triangle in triangles]
    sections = [("PrismElements", 14, prisms)]
    for axis, side, name in [(0, 0, "bc-2"), (0, n, "bc-2-1"),
                             (1, 0, "bc-3"), (1, n, "bc-3-1")]:
        quads = []
        for z in range(nz):
            for j in range(n):
                a = (side, j) if axis == 0 else (j, side)
                b = (side, j + 1) if axis == 0 else (j + 1, side)
                quads.append([node(*a, z), node(*b, z), node(*b, z + 1), node(*a, z + 1)])
        sections.append((name, 7, quads))
    sections.extend([("bc-4", 5, [[node(x, y, 0) for x, y in tri] for tri in triangles]),
                     ("bc-4-1", 5, [[node(x, y, nz) for x, y in tri] for tri in triangles])])
    fn, base, zone = C.c_int(), C.c_int(), C.c_int()
    check = lambda status: writer.check(lib, status, str(destination))
    check(lib.cg_open(str(destination).encode(), 1, C.byref(fn)))
    try:
        check(lib.cg_base_write(fn.value, b"Base", 3, 3, C.byref(base)))
        size = np.array([len(points), len(prisms), 0], dtype=np.int64)
        check(lib.cg_zone_write(fn.value, base.value, b"Zone", writer.pointer(size, writer.CG_SIZE), 3, C.byref(zone)))
        for dim, name in enumerate((b"CoordinateX", b"CoordinateY", b"CoordinateZ")):
            coordinate, idx = np.ascontiguousarray(points[:, dim]), C.c_int()
            check(lib.cg_coord_write(fn.value, base.value, zone.value, 4, name,
                                      writer.pointer(coordinate, C.c_double), C.byref(idx)))
        first = 1
        for name, element, connectivity in sections:
            array, idx = np.ascontiguousarray(connectivity, dtype=np.int64), C.c_int()
            check(lib.cg_section_write(fn.value, base.value, zone.value, name.encode(), element,
                                        first, first + len(connectivity) - 1, 0,
                                        writer.pointer(array, writer.CG_SIZE), C.byref(idx)))
            first += len(connectivity)
    finally:
        check(lib.cg_close(fn.value))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--problem", choices=("tgv2d", "entropy2d"), default="tgv2d")
    parser.add_argument("--levels", nargs="+", type=int, default=[8, 16, 32])
    parser.add_argument("--ranks", type=int, default=1)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--sweeps", type=int, default=30)
    parser.add_argument("--dt-scale", type=float, default=1)
    args = parser.parse_args()
    if min(args.levels + [args.ranks, args.jobs, args.sweeps]) <= 0 or args.dt_scale <= 0:
        parser.error("All inputs must be positive")
    output, binary = args.output.resolve(), args.binary.resolve()
    raw, meshes = output / "raw", output / "meshes"
    raw.mkdir(parents=True, exist_ok=True)
    meshes.mkdir(parents=True, exist_ok=True)
    for n in args.levels:
        mesh = meshes / f"periodic_n{n}_nz2.cgns"
        if not mesh.exists():
            generate_mesh(mesh, n)
    digest = hashlib.sha256(binary.read_bytes()).hexdigest()

    def run(n):
        stem = f"{args.problem}_n{n}"
        mesh, result = meshes / f"periodic_n{n}_nz2.cgns", raw / f"{stem}.json"
        final_time = 0.2 if args.problem == "tgv2d" else 0.5
        dt = (0.04 if args.problem == "tgv2d" else 0.08) / n * args.dt_scale
        depth = 4 * math.pi / n
        env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   HWLOC_COMPONENTS="-gl,-cuda,-nvml", HOCFV_TGV2D_DEPTH=str(depth))
        for key in ("HOCFV_AUDIT_RHS", "HOCFV_WRITE_FIELDS", "HOCFV_ENTROPY_MOMENT_CORRECTION",
                    "HOCFV_SURFACE_QUADRATURE", "HOCFV_UNCACHED_QUADRATURE"):
            env.pop(key, None)
        command = ["mpirun", "--mca", "pml", "ob1", "--mca", "btl", "self,vader", "--bind-to", "none",
                   "-np", str(args.ranks), str(binary), str(mesh), str(result), args.problem,
                   str(dt), str(final_time), str(args.sweeps)]
        manifest = {"command": command, "binary_sha256": digest,
                    "mesh_sha256": hashlib.sha256(mesh.read_bytes()).hexdigest(), "depth": depth,
                    "reference_thread": "codex://threads/01a125a4-7835-7d20-8ea8-1e0dc06330d0"}
        manifest_path = raw / f"{stem}.manifest.json"
        if result.exists():
            if not manifest_path.exists() or json.loads(manifest_path.read_text()) != manifest:
                raise RuntimeError(f"Different existing case: {result}")
        else:
            if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
                raise RuntimeError(f"Different partial case: {result}")
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
            print(f"Start N={n}, ranks={args.ranks}, dt={dt}", flush=True)
            with (raw / f"{stem}.log").open("w") as log:
                subprocess.run(command, cwd=ROOT / "build", env=env, check=True, stdout=log, stderr=subprocess.STDOUT)
        meta = json.loads(result.read_text())
        if meta["problem"] != args.problem or meta["mean_fallback_count"] != 0 or meta["rhs_evaluations"] != 3 * meta["steps"]:
            raise RuntimeError(f"Unexpected case metadata: {result}")
        print(f"Done N={n}, seconds={meta['march_seconds']:.3f}, residual={meta['last_coupled_residual']:.3e}", flush=True)

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        list(pool.map(run, args.levels))


if __name__ == "__main__":
    main()
