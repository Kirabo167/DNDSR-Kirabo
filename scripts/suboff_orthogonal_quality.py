#!/usr/bin/env python3
"""Compute face-based finite-volume orthogonality for this layered hybrid mesh.

Q = min(n_f dot (face_center-cell_center)/|face_center-cell_center|,
        n_f dot (neighbor_center-cell_center)/|neighbor_center-cell_center|).
The second term applies only to internal faces. Normals point out of the
owner cell. Report signed cosines and clip to [0, 1] for the quality score.
This is an explicitly defined orthogonality measure, not Fluent's proprietary
combined/modified orthogonal-quality implementation.
"""

import json
import argparse
from pathlib import Path
import time

import numpy as np


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('directory', type=Path)
parser.add_argument('--arrays', default='SUBOFF_optimized.validated.npz')
parser.add_argument('--output-prefix', default='orthogonal_quality')
parser.add_argument('--skip-jacobian-check', action='store_true',help='Intermediate diagnostics only; not a validation pass')
args = parser.parse_args()
D = args.directory
data = np.load(D / args.arrays)
front = np.load(D / "prism_front.npz")
points = data["points"]
tet = data["cells_4"]
prism = data["cells_6"]
from suboff_arrays_cgns import orient_validate
if not args.skip_jacobian_check:
    print('JACOBIAN_MINIMA', orient_validate(points, tet), orient_validate(points, prism), flush=True)
else:
    print('INTERMEDIATE DIAGNOSTIC: Jacobian validation skipped',flush=True)
nt, np_ = len(tet), len(prism)
nf = len(front["faces"])
nv = len(front["points"])
layers = np_ // nf
assert layers >= 1 and np_ == nf*layers
for layer in range(layers):
    assert np.array_equal(prism[layer*nf:(layer+1)*nf, :3], front["faces"] + layer*nv)
    assert np.array_equal(prism[layer*nf:(layer+1)*nf, 3:], front["faces"] + (layer+1)*nv)
centers = np.empty((nt + np_, 3))
cf = np.ones(nt + np_)
cc = np.ones(nt + np_)
worst = {"cell_face": {"signed_cosine": 1.}, "cell_cell": {"signed_cosine": 1.}}
started = time.time()


def determinant(a, b, c):
    return np.einsum("ij,ij->i", np.cross(a, b), c)


def face_geometry(ids):
    if ids.shape[1] == 4:
        # Choose the same diagonal for either orientation of a shared quad.
        ids = np.take_along_axis(ids, (ids.argmin(axis=1)[:, None] + np.arange(4)) % 4, axis=1)
    q = points[ids]
    cross1 = np.cross(q[:, 1]-q[:, 0], q[:, 2]-q[:, 0])
    if ids.shape[1] == 3:
        return q.mean(axis=1), cross1
    cross2 = np.cross(q[:, 2]-q[:, 0], q[:, 3]-q[:, 0])
    w1 = np.linalg.norm(cross1, axis=1)
    w2 = np.linalg.norm(cross2, axis=1)
    center = (q[:, :3].mean(axis=1)*w1[:, None] +
              q[:, [0, 2, 3]].mean(axis=1)*w2[:, None]) / (w1+w2)[:, None]
    return center, cross1 + cross2


def measure(ids, owners, neighbors=None):
    face_center, normal = face_geometry(ids)
    delta = (face_center - centers[owners]) if neighbors is None else (centers[neighbors] - centers[owners])
    denominator = np.linalg.norm(normal, axis=1)*np.linalg.norm(delta, axis=1)
    assert np.all(denominator > 0)
    cosine = np.einsum("ij,ij->i", normal, delta) / denominator
    kind = "cell_face" if neighbors is None else "cell_cell"
    target = cf if neighbors is None else cc
    np.minimum.at(target, owners, cosine)
    if neighbors is not None:
        np.minimum.at(target, neighbors, cosine)
    j = int(cosine.argmin())
    if cosine[j] < worst[kind]["signed_cosine"]:
        worst[kind] = dict(signed_cosine=float(cosine[j]),
                           owner_global_cgns_element=int(owners[j]+1),
                           owner_type="TETRA_4" if owners[j] < nt else "PENTA_6",
                           owner_center_m=centers[owners[j]].tolist(),
                           face_node_ids_1based=(ids[j]+1).tolist(),
                           face_center_m=face_center[j].tolist())
        if neighbors is not None:
            worst[kind]["neighbor_global_cgns_element"] = int(neighbors[j]+1)


# Tetrahedral volume centroids are vertex averages. Integrate trilinear
# PENTA_6 positions with degree-2 triangle / two-point Gauss quadrature.
for first in range(0, nt, 100_000):
    last = min(first+100_000, nt)
    centers[first:last] = points[tet[first:last]].mean(axis=1)
for first in range(0, np_, 100_000):
    q = points[prism[first:first+100_000]]
    volume = np.zeros(len(q))
    moment = np.zeros((len(q), 3))
    for z in (.5-1/(2*np.sqrt(3)), .5+1/(2*np.sqrt(3))):
        dr = (1-z)*(q[:, 1]-q[:, 0])+z*(q[:, 4]-q[:, 3])
        ds = (1-z)*(q[:, 2]-q[:, 0])+z*(q[:, 5]-q[:, 3])
        for r, s in ((1/6, 1/6), (2/3, 1/6), (1/6, 2/3)):
            bottom = (1-r-s)*q[:, 0]+r*q[:, 1]+s*q[:, 2]
            top = (1-r-s)*q[:, 3]+r*q[:, 4]+s*q[:, 5]
            position = (1-z)*bottom+z*top
            weight = determinant(dr, ds, top-bottom) / 12
            volume += weight
            moment += weight[:, None]*position
    assert np.all(volume > 0)
    centers[nt+first:nt+first+len(q)] = moment / volume[:, None]
print("CENTROIDS_DONE", time.time()-started, flush=True)

tet_faces = ((0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3))
prism_faces = ((0, 2, 1), (3, 4, 5), (0, 1, 4, 3), (1, 2, 5, 4), (2, 0, 3, 5))
for first in range(0, nt, 100_000):
    block = tet[first:first+100_000]
    owners = first+np.arange(len(block))
    for face in tet_faces:
        measure(block[:, face], owners)
for layer in range(layers):
    block = prism[layer*nf:(layer+1)*nf]
    owners = nt+layer*nf+np.arange(nf)
    for face in prism_faces:
        measure(block[:, face], owners)
    if layer+1 < layers:
        measure(block[:, [3, 4, 5]], owners, owners+nf)
print("CELL_FACE_AND_LAYER_INTERFACES_DONE", time.time()-started, flush=True)

# A side quad is shared with the same base-surface neighbor at each layer.
faces = front["faces"]
base_edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
base_owners = np.tile(np.arange(nf), 3)
key = np.sort(base_edges, axis=1)
order = np.lexsort((key[:, 1], key[:, 0]))
assert np.array_equal(key[order[::2]], key[order[1::2]])
edge_a, edge_b = order[::2], order[1::2]
edge = base_edges[edge_a]
owner, neighbor = base_owners[edge_a], base_owners[edge_b]
for layer in range(layers):
    ids = np.c_[edge[:, 0]+layer*nv, edge[:, 1]+layer*nv,
                edge[:, 1]+(layer+1)*nv, edge[:, 0]+(layer+1)*nv]
    measure(ids, nt+layer*nf+owner, nt+layer*nf+neighbor)
print("PRISM_SIDE_INTERFACES_DONE", time.time()-started, flush=True)

# Match tetrahedral triangles and the outermost prism triangles exactly.
triangles = np.vstack([tet[:, f] for f in tet_faces] + [prism[-nf:, 3:6]])
owners = np.r_[np.tile(np.arange(nt), 4), nt+(layers-1)*nf+np.arange(nf)]
key = np.sort(triangles, axis=1)
order = np.lexsort((key[:, 2], key[:, 1], key[:, 0]))
ordered = key[order]
same = np.all(ordered[1:] == ordered[:-1], axis=1)
assert not np.any(same[1:] & same[:-1]), "Non-manifold triangle"
paired = np.flatnonzero(same)
for first in range(0, len(paired), 100_000):
    rows = paired[first:first+100_000]
    a, b = order[rows], order[rows+1]
    measure(triangles[a], owners[a], owners[b])
print("ALL_INTERFACES_DONE", time.time()-started, flush=True)

signed = np.minimum(cf, cc)
quality = np.clip(signed, 0., 1.)
result = dict(
    definition="Minimum signed face-normal cosine against cell-to-face and internal cell-to-cell directions; score clipped to [0,1]",
    jacobian_validation_performed=not args.skip_jacobian_check,
    cell_centers="Exact volume centroids for TETRA_4 and trilinear PENTA_6 (degree-exact quadrature)",
    face_centers="Triangle area centroid; quadrilateral area centroid from a globally consistent diagonal triangulation",
    fluent_combined_orthogonal_quality=False,
    cells=len(quality), minimum=float(quality.min()), raw_signed_minimum=float(signed.min()),
    mean=float(quality.mean()),
    quantiles=dict(zip(["min", "p001", "p01", "p1", "p5", "median", "max"],
                       np.quantile(quality, [0,.00001,.0001,.01,.05,.5,1]).tolist())),
    cells_below={str(t): int((quality < t).sum()) for t in (.01,.05,.1,.2)},
    minimum_by_type={"TETRA_4": float(quality[:nt].min()), "PENTA_6": float(quality[nt:].min())},
    worst_components=worst,
    negative_signed_cells=int((signed < 0).sum()), seconds=time.time()-started,
)
np.save(D / (args.output_prefix+"_values.npy"), quality.astype(np.float32))
(D / (args.output_prefix+".json")).write_text(json.dumps(result, indent=2)+"\n")
print(json.dumps(result, indent=2), flush=True)
