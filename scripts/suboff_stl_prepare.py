#!/usr/bin/env python3
"""Prepare a full-domain SUBOFF surface and local sizes from a remeshed STL."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import trimesh


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("surface", type=Path)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--surface-size", type=float, default=0.014)
    parser.add_argument("--simplify", action="store_true", help="Decimate and remesh a dense source STL")
    parser.add_argument("--appendage-size", type=float, help="Use finer surface sizes on appendages and stern")
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    input_surface = args.surface
    if args.simplify:
        import pymeshlab as ml
        mesher = ml.MeshSet()
        mesher.load_new_mesh(str(args.surface))
        mesher.apply_filter("meshing_decimation_quadric_edge_collapse", targetfacenum=100000,
                            preservetopology=True, preservenormal=True, qualitythr=0.5, autoclean=True)
        mesher.apply_filter("meshing_isotropic_explicit_remeshing", iterations=4,
                            targetlen=ml.PureValue(0.012), featuredeg=35, checksurfdist=True,
                            maxsurfdist=ml.PureValue(0.00015), adaptive=False)
        input_surface = args.directory / "simplified_surface.stl"
        mesher.save_current_mesh(str(input_surface))
    body = trimesh.load_mesh(input_surface, process=True)
    if not body.is_watertight or not body.is_winding_consistent:
        raise ValueError("Input surface must be watertight and consistently oriented")
    body.fix_normals()
    length = 4.356
    lower = np.array([-3 * length, -3 * length, -3 * length])
    upper = np.array([7 * length, 3 * length, 3 * length])
    box = trimesh.creation.box(extents=upper - lower)
    box.apply_translation((upper + lower) / 2)
    inner = body.copy()
    inner.invert()
    fluid = trimesh.util.concatenate([box, inner])
    fluid.export(args.directory / "fluid_domain.stl")
    body.export(args.directory / "body_surface.stl")
    # Select one wall sample per 2 cm cube, retaining appendage samples.
    keys = np.floor(body.vertices / 0.02).astype(np.int64)
    _, indices = np.unique(keys, axis=0, return_index=True)
    if args.appendage_size is not None:
        vertices = body.vertices
        fine = (((vertices[:, 1] > .245) & (vertices[:, 0] > .9) & (vertices[:, 0] < 1.32)) |
                ((vertices[:, 0] > 3.7) & (np.linalg.norm(vertices[:, 1:], axis=1) > .07)) |
                (vertices[:, 0] > 4.05))
        fine_indices = np.flatnonzero(fine)
        _, selected = np.unique(np.floor(vertices[fine] / (args.appendage_size/2)).astype(np.int64),
                                axis=0, return_index=True)
        indices = np.unique(np.r_[indices, fine_indices[selected]])
    samples = body.vertices[indices]
    sizes = np.full(len(samples), args.surface_size)
    radius = np.linalg.norm(samples[:, 1:], axis=1)
    sizes[(samples[:, 1] > 0.25) | ((samples[:, 0] > 3.4) & (radius > 0.12))] *= 0.7
    if args.appendage_size is not None:
        fine = (((samples[:, 1] > .245) & (samples[:, 0] > .9) & (samples[:, 0] < 1.32)) |
                ((samples[:, 0] > 3.7) & (radius > .07)) | (samples[:, 0] > 4.05))
        sizes[fine] = args.appendage_size
        caps = ((samples[:, 1] > .45) |
                ((samples[:, 0] > 3.7) & (np.max(np.abs(samples[:, 1:]), axis=1) > .2435)))
        sizes[caps] = min(.0015, args.appendage_size)
    # These downstream samples affect the surface mesher only. They do not
    # constitute a volume-size field for the subsequent TetGen core.
    wake = np.array([(x, y, z)
                     for x in np.arange(length, 3.5 * length, 0.25)
                     for y in np.arange(-0.5, 0.51, 0.25)
                     for z in np.arange(-0.5, 0.51, 0.25)])
    wake_sizes = 0.05 + 0.02 * (wake[:, 0] - length)
    with (args.directory / "local_sizes.txt").open("w") as stream:
        stream.write(f"{len(samples) + len(wake)}\n")
        for point, size in zip(np.vstack([samples, wake]), np.r_[sizes, wake_sizes]):
            stream.write(" ".join(f"{v:.12g}" for v in [*point, size]) + "\n")
        stream.write("0\n")
    summary = {
        "input": str(args.surface.resolve()),
        "sha256": hashlib.sha256(args.surface.read_bytes()).hexdigest(),
        "surface_triangles": len(body.faces), "surface_nodes": len(body.vertices),
        "watertight": bool(body.is_watertight), "volume_m3": float(body.volume),
        "body_bounds_m": body.bounds.tolist(), "domain_bounds_m": [lower.tolist(), upper.tolist()],
        "hull_length_m": length, "surface_size_m": args.surface_size,
        "appendage_surface_size_m": args.appendage_size,
        "coordinate_convention": "Original STL axes: x streamwise, +y sail/up, z transverse",
    }
    (args.directory / "preparation.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
