"""Independently rebuild IV10 geometry and one complete RK step in Python."""
import argparse
import importlib.util
import json
from pathlib import Path

import h5py
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("mesh", type=Path)
    parser.add_argument("--runs", default="main")
    parser.add_argument("--problems", nargs="+", choices=("entropy", "vortex"), default=("entropy", "vortex"))
    args = parser.parse_args()
    root = args.root.resolve()
    spec = importlib.util.spec_from_file_location("reference_core", root / "reference_core.py")
    core = importlib.util.module_from_spec(spec); spec.loader.exec_module(core)
    with h5py.File(args.mesh) as mesh:
        zone = mesh["Base/blk-1"]
        x = np.array([zone[f"GridCoordinates/Coordinate{d}/ data"][()] for d in "XYZ"]).T
        cells = zone["PrismElements/ElementConnectivity/ data"][()].reshape(-1, 6) - 1
    n = len(x)
    if n > 2000:
        raise RuntimeError("Independent Python step check is intended for IV10")
    local_faces = ((0, 2, 1), (3, 4, 5), (0, 1, 4, 3), (1, 2, 5, 4), (2, 0, 3, 5))
    faces, face_map, cell_faces = [], {}, []
    for cell in cells:
        ids = []
        for local in local_faces:
            nodes = cell[list(local)]
            key = tuple(sorted(nodes))
            if key not in face_map:
                points = x[nodes]
                weights = np.ones(len(nodes)) if len(nodes) == 3 else (
                    np.sum((np.roll(points, -1, axis=0) - points)**2, axis=1) +
                    np.sum((np.roll(points, 1, axis=0) - points)**2, axis=1))
                face_map[key] = len(faces)
                faces.append({"nodes": nodes, "center": weights @ points / weights.sum(), "cells": []})
            ids.append(face_map[key])
            faces[ids[-1]]["cells"].append(len(cell_faces))
        cell_faces.append(ids)
    edges, edge_map, incidence = [], {}, [[] for _ in range(n)]
    for face in faces:
        nodes = face["nodes"]
        for i, j in zip(nodes, np.roll(nodes, -1)):
            key = tuple(sorted((int(i), int(j))))
            if key not in edge_map:
                edge_map[key] = len(edges)
                edges.append({"ends": key, "mid": (x[key[0]] + x[key[1]]) / 2,
                              "S": np.zeros(3), "pieces": []})
                incidence[key[0]].append((len(edges) - 1, 1))
                incidence[key[1]].append((len(edges) - 1, -1))
    ref = np.array([(np.max(np.vstack((x[i], [edges[e]["mid"] for e, _ in incidence[i]])), axis=0) -
                     np.min(np.vstack((x[i], [edges[e]["mid"] for e, _ in incidence[i]])), axis=0)) / 2
                    for i in range(n)])
    a, b, c, d = .106994014770537, .6790179556883893, .3280585716625127, .0158242850124619
    q = np.array([(a, a, b), (a, b, a), (b, a, a), (a, a, a),
                  (c, c, d), (c, d, c), (d, c, c), (c, c, c)])
    qw = np.array([.75 * .1487451997767295] * 4 + [.75 * .1845881335566038] * 4)
    quadrature, centers = [[] for _ in range(n)], []
    for ids in cell_faces:
        weights = []
        for f in ids:
            face = faces[f]; p = x[face["nodes"]] - face["center"]
            area2 = np.linalg.norm(np.cross(p, np.roll(p, -1, axis=0)), axis=1).sum()
            weights.append(area2**2)
        center = np.array(weights) @ np.array([faces[f]["center"] for f in ids]) / sum(weights)
        centers.append(center)
        for f in ids:
            face = faces[f]; nodes = face["nodes"]
            for i, j in zip(nodes, np.roll(nodes, -1)):
                edge = edges[edge_map[tuple(sorted((int(i), int(j))))]]
                left, right = edge["ends"]; mid = edge["mid"]
                s = np.cross(face["center"] - mid, center - mid) / 2
                if np.dot(s, x[right] - x[left]) < 0: s = -s
                edge["S"] += s
                edge["pieces"].append((np.array([mid, face["center"], center]), s))
                for node in (left, right):
                    measure = abs(np.dot(mid - x[node], s)) / 3
                    points = mid + q @ np.array([face["center"] - mid, center - mid, x[node] - mid])
                    quadrature[node].append((points, qw * measure))
    volume = np.zeros(n)
    for edge in edges:
        i, j = edge["ends"]; v = np.dot(edge["S"], x[j] - x[i]) / 6
        volume[i] += v; volume[j] += v
    coordinates = {tuple(np.rint(p * 1e8).astype(np.int64)): i for i, p in enumerate(x)}
    patches, patch_triangles, boundary_samples = [], [], [[] for _ in range(n)]
    for face in faces:
        if len(face["cells"]) != 1: continue
        nodes = face["nodes"]; p = x[nodes]
        normal = np.cross(p[1] - p[0], p[2] - p[0]); normal /= np.linalg.norm(normal)
        if np.dot(normal, face["center"] - centers[face["cells"][0]]) < 0: normal = -normal
        axis = np.argmax(np.abs(normal)); shift = np.zeros(3)
        shift[axis] = -np.copysign((10, 10, 4)[axis], normal[axis])
        for j, node in enumerate(nodes):
            m1, m2 = (x[node] + p[(j + 1) % len(nodes)]) / 2, (x[node] + p[(j - 1) % len(nodes)]) / 2
            area = (np.linalg.norm(np.cross(face["center"] - x[node], m1 - x[node])) +
                    np.linalg.norm(np.cross(face["center"] - x[node], m2 - x[node]))) / 2
            point = (x[node] + face["center"] + m1 + m2) / 4
            partner = coordinates[tuple(np.rint((x[node] + shift) * 1e8).astype(np.int64))]
            patches.append((node, partner, point, point + shift, area * normal))
            patch_triangles.append((np.array([x[node], face["center"], m1]),
                                    np.array([x[node], face["center"], m2])))
            boundary_samples[node].append((partner, point, point + shift, 2 * np.sqrt(area)))
    points = [np.concatenate([p for p, w in quadrature[i]]) for i in range(n)]
    weights = [np.concatenate([w for p, w in quadrature[i]]) for i in range(n)]
    moment = np.zeros((n, 9))
    for i in range(n):
        xi = (points[i] - x[i]) / ref[i]
        monomials = np.array([np.prod(xi**power, axis=1) for power in core.POWERS])
        moment[i] = monomials @ weights[i] / volume[i]
    derivatives = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1),
                   (2, 0, 0), (1, 1, 0), (0, 2, 0), (0, 1, 1), (0, 0, 2), (1, 0, 1)]
    samples, operators = [], []

    def rows(node, point, length):
        value = np.array([core.basis(point, x[node], ref[node], moment[node], derivative) for derivative in derivatives])
        w = np.array([1] + [length / 2] * 3 + [length**2 / 4] * 6); w[[5, 7, 9]] *= np.sqrt(2)
        return w[:, None] * value

    for node in range(n):
        sample = []
        for e, sign in incidence[node]:
            edge = edges[e]; i, j = edge["ends"]
            sample.append((j if sign > 0 else i, edge["mid"], edge["mid"], np.linalg.norm(x[j] - x[i])))
        sample.extend(boundary_samples[node]); samples.append(sample)
        left = [rows(node, p, length) for neighbor, p, other, length in sample]
        matrix = sum(v.T @ v for v in left)
        operators.append([(neighbor, np.linalg.solve(matrix, l.T @ rows(neighbor, other, length)),
                           np.linalg.solve(matrix, l[0]))
                          for l, (neighbor, p, other, length) in zip(left, sample)])
    output = []
    for problem in args.problems:
        meta = json.loads((root / args.runs / f"raw/{problem}_iv10.json").read_text())
        audit = json.loads((root / args.runs / f"raw/{problem}_iv10.json.audit.json").read_text())
        ranks, dt = meta["ranks"], meta["dt"]
        partition = np.array([next(r for r in range(ranks) if n * r // ranks <= i < n * (r + 1) // ranks) for i in range(n)])

        def exact(point):
            if problem == "entropy":
                rho = 1 + .2 * np.sin(2 * np.pi * (point[0] / 10 + point[1] / 10 + point[2] / 4))
                return core.conservative(rho, (1, 1, 0), 1)
            dx, dy = point[:2] - 5; r2 = dx * dx + dy * dy
            beta = 5 / (2 * np.pi) * np.exp((1 - r2) / 2)
            theta = 1 - .4 * 25 / (8 * 1.4 * np.pi**2) * np.exp(1 - r2)
            rho = theta**2.5
            return core.conservative(rho, (1 - beta * dy, 1 + beta * dx, 0), rho**1.4)

        state = np.array([weights[i] @ np.array([exact(p) for p in points[i]]) / volume[i] for i in range(n)])
        coefficients = np.zeros((n, 9, 5)); base = state.copy()
        max_error = 0
        for stage, c in enumerate((1, .25, 2 / 3)):
            for sweep in range(meta["sweeps"]):
                frozen = coefficients.copy()
                for i in range(n):
                    coefficients[i] = sum(matrix @ (coefficients[j] if partition[i] == partition[j] else frozen[j]) +
                                          np.outer(jump, state[j] - state[i]) for j, matrix, jump in operators[i])
            if stage == 0:
                max_error = max(max_error, np.max(np.abs(coefficients[0] -
                    np.array(audit["first_node"]["coefficients_after_initial_reconstruction"]))))

            def face_state(node, point):
                reconstructed = state[node] + coefficients[node].T @ core.basis(point, x[node], ref[node], moment[node])
                return core.face_state(reconstructed, state[node])

            rhs = np.zeros((n, 5))
            surface_quadrature = meta.get("surface_flux_mode", "midpoint") == "surface6"
            qa, qb, qc, qd = .091576213509771, .8168475729804585, .445948490915965, .108103018168070
            surface_points = np.array([(qa, qa), (qb, qa), (qa, qb), (qc, qc), (qd, qc), (qc, qd)])
            surface_weights = np.array([.109951743655322] * 3 + [.2233815896780115] * 3)
            moment_correction = meta.get("entropy_moment_correction", False)
            triangle_points = np.array([[2 / 3, 1 / 6, 1 / 6], [1 / 6, 2 / 3, 1 / 6],
                                        [1 / 6, 1 / 6, 2 / 3]])
            velocity = np.array([1., 1., 0.])
            entropy_scale = np.array([1., 1., 1., 0., 1.])
            for edge in edges:
                i, j = edge["ends"]; area = np.linalg.norm(edge["S"]); normal = edge["S"] / area
                if surface_quadrature:
                    for triangle, s in edge["pieces"]:
                        area, normal = np.linalg.norm(s), s / np.linalg.norm(s)
                        for (xi, eta), weight in zip(surface_points, surface_weights):
                            point = triangle[0] + xi * (triangle[1] - triangle[0]) + eta * (triangle[2] - triangle[0])
                            flux = np.array(core.roe_flux(normal, face_state(i, point), face_state(j, point)))
                            rhs[i] -= area * weight * (flux - core.physical_flux(normal, state[i]))
                            rhs[j] += area * weight * (flux - core.physical_flux(normal, state[j]))
                    continue
                flux = np.array(core.roe_flux(normal, face_state(i, edge["mid"]), face_state(j, edge["mid"])))
                rhs[i] -= area * (flux - core.physical_flux(normal, state[i]))
                rhs[j] += area * (flux - core.physical_flux(normal, state[j]))
                if moment_correction:
                    # Independent degree-two triangle quadrature of the central
                    # density flux, rather than the C++ analytic moment formula.
                    integral = sum(np.dot(s, velocity) * np.mean([
                        .5 * (face_state(i, point)[0] + face_state(j, point)[0])
                        for point in triangle_points @ triangle])
                        for triangle, s in edge["pieces"])
                    midpoint = .5 * (face_state(i, edge["mid"])[0] + face_state(j, edge["mid"])[0])
                    correction = (integral - np.dot(edge["S"], velocity) * midpoint) * entropy_scale
                    rhs[i] -= correction; rhs[j] += correction
            for patch_id, (i, j, p, other, s) in enumerate(patches):
                area = np.linalg.norm(s); normal = s / area
                if surface_quadrature:
                    for triangle in patch_triangles[patch_id]:
                        for (xi, eta), weight in zip(surface_points, surface_weights):
                            point = triangle[0] + xi * (triangle[1] - triangle[0]) + eta * (triangle[2] - triangle[0])
                            flux = np.array(core.roe_flux(normal, face_state(i, point), face_state(j, point + other - p)))
                            rhs[i] -= .5 * area * weight * (flux - core.physical_flux(normal, state[i]))
                    continue
                flux = np.array(core.roe_flux(normal, face_state(i, p), face_state(j, other)))
                rhs[i] -= area * (flux - core.physical_flux(normal, state[i]))
                if moment_correction:
                    shift = other - p
                    integral = sum(.5 * np.dot(s, velocity) * np.mean([
                        .5 * (face_state(i, point)[0] + face_state(j, point + shift)[0])
                        for point in triangle_points @ triangle])
                        for triangle in patch_triangles[patch_id])
                    midpoint = .5 * (face_state(i, p)[0] + face_state(j, other)[0])
                    rhs[i] -= (integral - np.dot(s, velocity) * midpoint) * entropy_scale
            rhs /= volume[:, None]
            expected_rhs = np.array(audit["rk_stage"][stage]["rhs_divided_by_volume"]).reshape(-1)
            max_error = max(max_error, np.max(np.abs(rhs[0] - expected_rhs)))
            state = (1 - c) * base + c * (state + dt * rhs)
            max_error = max(max_error, np.max(np.abs(state[0] -
                np.array(audit["rk_stage"][stage]["after"]).reshape(-1))))
        if max_error > 1e-10:
            raise RuntimeError(f"Independent full step mismatch: {problem} {max_error}")
        result = {"problem": problem, "nodes": n, "total_volume": float(volume.sum()),
                  "surface_flux_mode": meta.get("surface_flux_mode", "midpoint"),
                  "entropy_moment_correction": meta.get("entropy_moment_correction", False),
                  "maximum_coefficient_rhs_stage_difference": float(max_error)}
        output.append(result); print(result, flush=True)
    (root / "independent_full_step.json").write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
