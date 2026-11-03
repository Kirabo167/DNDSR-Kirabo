#!/usr/bin/env python3
"""Rebuild a high-fidelity computational surface from the supplied SUBOFF STL.

Keep the curved sail roof. Apply bounded local fairing to the appendages,
restore the original planar fin tips, and verify the repaired surface before
it is used as meshing geometry. This does not invent missing CAD features.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pymeshlab as ml
import trimesh
from scipy.sparse import coo_matrix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    directory = args.directory
    directory.mkdir(parents=True, exist_ok=True)
    mesher = ml.MeshSet()
    mesher.load_new_mesh(str(args.source))
    original_faces = mesher.current_mesh().face_number()
    mesher.apply_filter("meshing_remove_duplicate_faces")
    mesher.apply_filter("meshing_remove_duplicate_vertices")
    print("DECIMATE_BEGIN", mesher.current_mesh().face_number(), flush=True)
    mesher.apply_filter("meshing_decimation_quadric_edge_collapse", targetfacenum=400000,
                        preservetopology=True, preservenormal=True, qualitythr=0.5, autoclean=True)
    before = mesher.current_mesh().vertex_matrix().copy()
    faces = mesher.current_mesh().face_matrix().copy()
    np.savez(directory / "decimated_geometry.npz", points=before, faces=faces)
    # Work on fixed NumPy indices: filters that compact a selected mesh can
    # reorder vertex storage and invalidate comparisons to the old matrix.
    temporary = trimesh.Trimesh(before, faces, process=False)
    sharp = temporary.face_adjacency_edges[temporary.face_adjacency_angles > np.deg2rad(35)]
    movable = (((before[:, 1] > .245) & (before[:, 0] > .9) & (before[:, 0] < 1.32)) |
               (before[:, 0] > 3.65))
    movable[np.unique(sharp)] = False
    edges = temporary.edges_unique
    row = np.r_[edges[:, 0], edges[:, 1]]
    col = np.r_[edges[:, 1], edges[:, 0]]
    adjacency = coo_matrix((np.ones(len(row)), (row, col)), shape=(len(before), len(before))).tocsr()
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    points = before.copy()
    for _ in range(4):
        for coefficient in (.15, -.155):
            laplacian = adjacency @ points / degree[:, None] - points
            points[movable] += coefficient * laplacian[movable]
    displacement = points - before
    distance = np.linalg.norm(displacement, axis=1)
    scale = np.minimum(1., 5e-5 / np.maximum(distance, 1e-30))
    points = before + scale[:, None] * displacement
    # Original fin end faces are planar; protect their vertices from fairing.
    p = before[faces]
    normal = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    normal /= np.maximum(np.linalg.norm(normal, axis=1)[:, None], 1e-30)
    centers = p.mean(axis=1)
    tip_vertices = set()
    for axis in (1, 2):
        for sign in (-1., 1.):
            selected = ((centers[:, 0] > 3.7) &
                        (np.abs(centers[:, axis] - sign*.24384) < 2e-5) &
                        (sign*normal[:, axis] > .995))
            vertices = np.unique(faces[selected])
            points[vertices, axis] = sign*.24384
            tip_vertices.update(vertices.tolist())
    repaired = trimesh.Trimesh(points, faces, process=True)
    repaired.fix_normals()
    if not repaired.is_watertight or not repaired.is_winding_consistent:
        raise ValueError("Repair did not preserve a closed, oriented surface")
    checker = ml.MeshSet()
    checker.add_mesh(ml.Mesh(repaired.vertices, repaired.faces))
    checker.apply_filter("compute_selection_by_self_intersections_per_face")
    intersections = checker.current_mesh().selected_face_number()
    if intersections:
        raise ValueError(f"Repaired geometry has {intersections} intersecting faces")
    repaired.export(directory / "SUBOFF_repaired.stl")
    np.savez(directory / "repaired_geometry.npz", points=repaired.vertices, faces=repaired.faces)
    summary = dict(source=str(args.source.resolve()), original_triangles=original_faces,
                   repaired_triangles=len(repaired.faces), repaired_vertices=len(repaired.vertices),
                   watertight=bool(repaired.is_watertight), consistently_oriented=bool(repaired.is_winding_consistent),
                   self_intersecting_faces=intersections,
                   local_fairing_displacement_cap_m=5e-5, protected_fin_tip_vertices=len(tip_vertices),
                   fairing_vertices=int(movable.sum()),
                   maximum_unselected_fairing_displacement_m=float(np.linalg.norm(points-before, axis=1)[~movable].max()),
                   vertex_displacement_after_decimation_quantiles_m=np.quantile(np.linalg.norm(points-before, axis=1), [0,.5,.95,.99,1]).tolist(),
                   volume_m3=float(repaired.volume), body_bounds_m=repaired.bounds.tolist(),
                   sail_roof="Original curved roof retained; no planar replacement")
    (directory / "geometry_repair.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("REPAIR_DONE", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
