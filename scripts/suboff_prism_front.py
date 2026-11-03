#!/usr/bin/env python3
"""Build a limited, closed prism front, then a constrained tetrahedral core.

The displacement is bounded using changes in adjacent vertex normals. Sharp
trailing edges use the interior of the incident-face normal cone rather than
unbounded miter normals. Every prism column must have positive Jacobians;
the offset surface is checked for self intersections before tetrahedralizing.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import cKDTree

from suboff_arrays_cgns import jacobians


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--tetgen-package", type=Path)
    parser.add_argument("--meshlab-package", type=Path)
    parser.add_argument("--limit-mode", choices=("all", "concave"), default="concave")
    args = parser.parse_args()
    directory = args.directory
    data = np.load(directory / "surface_arrays.npz")
    all_points, all_faces = data["points"], data["triangles"]
    center = all_points[all_faces].mean(axis=1)
    wall = (center[:, 0] > -.01) & (center[:, 0] < 4.37) & (np.linalg.norm(center[:, 1:], axis=1) < .7)
    used, inverse = np.unique(all_faces[wall], return_inverse=True)
    points = all_points[used]
    faces = inverse.reshape((-1, 3))
    p = points[faces]
    signed_volume = np.einsum("ij,ij->i", p[:, 0], np.cross(p[:, 1], p[:, 2])).sum() / 6
    if signed_volume < 0:
        faces = faces[:, [0, 2, 1]]
        p = points[faces]
    normal = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    normal /= np.linalg.norm(normal, axis=1)[:, None]
    direction = np.zeros_like(points)
    for i in range(3):
        u = p[:, (i + 1) % 3] - p[:, i]
        v = p[:, (i + 2) % 3] - p[:, i]
        angle = np.arccos(np.clip(np.einsum("ij,ij->i", u, v) /
                                 (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1)), -1, 1))
        np.add.at(direction, faces[:, i], normal * angle[:, None])
    direction /= np.linalg.norm(direction, axis=1)[:, None]
    dots = np.einsum("fij,fj->fi", direction[faces], normal)
    bad = np.unique(faces[dots <= 1e-6])
    print("NORMAL_CONES", len(points), len(faces), "repair", len(bad), flush=True)
    for vertex in bad:
        incident = normal[np.any(faces == vertex, axis=1)]
        result = linprog([0, 0, 0, -1], A_ub=np.c_[-incident, np.ones(len(incident))],
                         b_ub=np.zeros(len(incident)), bounds=[(-1, 1)] * 3 + [(0, None)], method="highs")
        if not result.success or result.x[3] <= 0:
            raise ValueError(f"No outward normal cone at vertex {vertex}")
        direction[vertex] = result.x[:3] / np.linalg.norm(result.x[:3])
    dots = np.einsum("fij,fj->fi", direction[faces], normal)
    print("MIN_NORMAL_PROJECTION", dots.min(), flush=True)
    if dots.min() <= 0:
        raise ValueError("A first wall layer would be inverted")

    first, growth, layers = 1e-5, 1.18, 40
    increments = first * growth**np.arange(layers)
    total = increments.sum()
    edges = np.unique(np.sort(np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1), axis=0)
    length = np.linalg.norm(points[edges[:, 0]] - points[edges[:, 1]], axis=1)
    delta_direction = direction[edges[:, 1]] - direction[edges[:, 0]]
    if args.limit_mode == "concave":
        # Outward normals diverge on convex caps: curvature there does not
        # impose a collision limit. Limit approaching normals only, then
        # verify the complete columns and the offset surface independently.
        edge_vector = points[edges[:, 1]] - points[edges[:, 0]]
        change = np.maximum(-np.einsum("ij,ij->i", edge_vector, delta_direction) / length, 0.)
    else:
        change = np.linalg.norm(delta_direction, axis=1)
    cap = np.full(len(points), total)
    allowed = .25 * length / np.maximum(change, 1e-12)
    np.minimum.at(cap, edges[:, 0], allowed)
    np.minimum.at(cap, edges[:, 1], allowed)
    # Smooth compression into adjacent columns without increasing any limit.
    smoothing_passes = 12 if args.limit_mode == "all" else 6
    neighbor_ratio = 1.2 if args.limit_mode == "all" else 1.5
    for _ in range(smoothing_passes):
        old = cap.copy()
        np.minimum.at(cap, edges[:, 0], neighbor_ratio * old[edges[:, 1]])
        np.minimum.at(cap, edges[:, 1], neighbor_ratio * old[edges[:, 0]])
    if args.meshlab_package:
        sys.path.append(str(args.meshlab_package))
    import pymeshlab
    intersections = 0
    for attempt in range(40):
        top = points + direction * cap[:, None]
        column_points = np.vstack([points, top])
        columns = np.c_[faces, faces + len(points)]
        values = jacobians(column_points, columns)
        invalid = np.any(values <= 0, axis=1)
        print("COLUMN_CHECK", attempt, int(invalid.sum()), float(values.min()), flush=True)
        if not invalid.any():
            # Local Jacobians alone do not catch collisions between remote
            # columns, especially on a newly refined attachment root.
            checker = pymeshlab.MeshSet()
            checker.add_mesh(pymeshlab.Mesh(np.vstack([points, top]),
                                            np.vstack([faces, faces + len(points)])))
            checker.apply_filter("compute_selection_by_self_intersections_per_face")
            intersections = checker.current_mesh().selected_face_number()
            print("FRONT_INTERSECTIONS", attempt, intersections, flush=True)
            if not intersections:
                break
            selected = np.flatnonzero(checker.current_mesh().face_selection_array())
            invalid[np.unique(selected % len(faces))] = True
        cap[np.unique(faces[invalid])] *= .5
        for _ in range(4):
            old = cap.copy()
            np.minimum.at(cap, edges[:, 0], neighbor_ratio * old[edges[:, 1]])
            np.minimum.at(cap, edges[:, 1], neighbor_ratio * old[edges[:, 0]])
    else:
        raise ValueError("Could not obtain positive, non-intersecting prism columns")
    np.savez(directory / "prism_front.npz", points=points, top=top, faces=faces, cap=cap, direction=direction)

    outer_used, outer_inverse = np.unique(all_faces[~wall], return_inverse=True)
    outer_points = all_points[outer_used]
    outer_faces = outer_inverse.reshape((-1, 3))
    core_input = np.vstack([top, outer_points])
    core_faces = np.vstack([faces[:, [0, 2, 1]], outer_faces + len(top)])
    if args.tetgen_package:
        sys.path.append(str(args.tetgen_package))
    import tetgen
    core = tetgen.TetGen(core_input, core_faces)
    core.add_hole([2., 0., 0.])
    print("CORE_BEGIN", len(core_input), len(core_faces), flush=True)
    core_points, tetrahedra, _, _ = core.tetrahedralize(switches="pq1.4YQ")
    # Verify that TetGen kept the entire prism interface triangulation.
    distance, mapping = cKDTree(core_points).query(core_input)
    if distance.max() > 1e-11:
        raise ValueError("TetGen moved a constrained boundary point")
    raw_faces = np.sort(core.trifaces, axis=1)
    expected = np.sort(mapping[core_faces], axis=1)
    face_set = set(map(tuple, raw_faces))
    if any(tuple(face) not in face_set for face in expected):
        raise ValueError("TetGen changed a constrained boundary facet")
    print("CORE_DONE", len(core_points), len(tetrahedra), flush=True)
    # One shared point number on both sides of each prism/tetra interface.
    n = len(points)
    fractions = np.r_[0., np.cumsum(increments) / total]
    layer_points = np.vstack([points + f * direction * cap[:, None] for f in fractions])
    prisms = np.vstack([np.c_[faces + k*n, faces + (k+1)*n] for k in range(layers)])
    point_map = np.full(len(core_points), -1, dtype=np.int64)
    point_map[mapping[:n]] = layers*n + np.arange(n)
    point_map[mapping[n:]] = len(layer_points) + np.arange(len(outer_points))
    additional = np.flatnonzero(point_map < 0)
    point_map[additional] = len(layer_points) + len(outer_points) + np.arange(len(additional))
    combined_points = np.vstack([layer_points, outer_points, core_points[additional]])
    tetrahedra = point_map[tetrahedra]
    volume_nodes = np.zeros((len(prisms) + len(tetrahedra), 6), dtype=np.int64)
    volume_nodes[:len(prisms)] = prisms + 1
    volume_nodes[len(prisms):, :4] = tetrahedra + 1
    boundary_faces = np.vstack([faces, outer_faces + len(layer_points)])
    np.savez(directory / "mesh_arrays.npz", points=combined_points, volume_nodes=volume_nodes,
             volume_sizes=np.r_[np.full(len(prisms), 6), np.full(len(tetrahedra), 4)],
             surface_nodes=boundary_faces + 1, surface_sizes=np.full(len(boundary_faces), 3))
    summary = dict(method="Normal-cone direction, edge-curvature height limit, positive prism columns, constrained TetGen core",
                   limit_mode=args.limit_mode,
                   cells=len(volume_nodes), prisms=len(prisms), tetrahedra=len(tetrahedra), nodes=len(combined_points),
                   first_full_height_m=first, growth=growth, layers=layers, requested_layer_thickness_m=float(total),
                   actual_column_thickness_m=np.quantile(cap, [0, .01, .5, .95, 1]).tolist(),
                   actual_first_displacement_m=np.quantile(cap/total*first, [0, .01, .5, .95, 1]).tolist(),
                   minimum_normal_projection=float(dots.min()), self_intersecting_front_faces=int(intersections),
                   constrained_boundary_facets_preserved=True)
    (directory / "generation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("SUMMARY", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
