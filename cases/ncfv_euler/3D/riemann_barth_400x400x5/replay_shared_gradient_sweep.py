"""Audit sequential face limiting using exported production integration weights.

This is an independent serial experiment, not an MPI solver integration.  A face
event evaluates both traces from the currently updated shared gradients, takes
one coefficient, and multiplies each participating node exactly once.  Physical
states are never clipped, and final constraints are always audited again.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np


@dataclass
class Trace:
    anchor: int
    nodes: np.ndarray
    weights: np.ndarray

    def increment(self, gradients: np.ndarray) -> np.ndarray:
        return np.einsum("nd,ndv->v", self.weights, gradients[self.nodes])


@dataclass
class Event:
    traces: list[Trace]
    support: np.ndarray


def coalesce(anchor: int, entries: dict[int, np.ndarray]) -> Trace:
    nodes = np.array(sorted(entries), dtype=np.int64)
    weights = np.array([entries[int(node)] for node in nodes]).reshape((-1, 3))
    return Trace(anchor, nodes, weights)


def pressure(states: np.ndarray, gamma: float) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return (gamma - 1) * (
            states[:, 4] - 0.5 * np.sum(states[:, 1:4] ** 2, axis=1) / states[:, 0]
        )


def load_operators(data: dict) -> tuple:
    nodes = data["nodes"]
    means = np.array([node["mean"] for node in nodes], dtype=float)
    gradients = np.array([node["gradient"] for node in nodes], dtype=float)
    coordinates = np.array([node["coordinate"] for node in nodes], dtype=float)
    ids = np.array([node["original_id"] for node in nodes], dtype=np.int64)
    if not all(np.all(np.isfinite(array)) for array in (means, gradients, coordinates)):
        raise ValueError("Exported states, gradients, and coordinates must be finite")
    recovery = []
    for i, node in enumerate(nodes):
        entries: dict[int, np.ndarray] = {}
        for entry in node["recovery"]:
            j = int(entry["node"])
            entries[j] = entries.get(j, np.zeros(3)) - np.array(entry["weight"])
        recovery.append(coalesce(i, entries))

    minimum, maximum = means.copy(), means.copy()
    ordered_faces = sorted(data["faces"], key=lambda face: tuple(face["key"]))
    for face in ordered_faces:
        a, b = face["anchors"]
        minimum[a] = np.minimum(minimum[a], means[b])
        maximum[a] = np.maximum(maximum[a], means[b])
        minimum[b] = np.minimum(minimum[b], means[a])
        maximum[b] = np.maximum(maximum[b], means[a])

    events = []
    for face in ordered_faces:
        traces, support = [], set()
        for side, i in enumerate(face["anchors"]):
            r = recovery[i]
            entries = {int(j): w.copy() for j, w in zip(r.nodes, r.weights)}
            support.update(int(j) for j, w in zip(r.nodes, r.weights) if np.any(w))
            for entry in face["stencil"]:
                j = int(entry["node"])
                weight = np.array(entry["weights"][side]) / face["measure"]
                entries[j] = entries.get(j, np.zeros(3)) + weight
                if np.any(weight):
                    support.add(j)
            traces.append(coalesce(i, entries))
        events.append(Event(traces, np.array(sorted(support), dtype=np.int64)))
    point_events = [
        Event([trace], trace.nodes[np.any(trace.weights != 0, axis=1)])
        for trace in recovery
    ]

    raw_points = means + np.array([trace.increment(gradients) for trace in recovery])
    exported_points = np.array([node["point"] for node in nodes])
    if not np.all(np.isfinite(exported_points)):
        raise ValueError("Exported point values must be finite")
    recovery_error = float(np.max(np.abs(raw_points - exported_points)))
    if recovery_error > 1e-10:
        raise ValueError(f"Exported point recovery mismatch: {recovery_error}")
    return (
        means, gradients, coordinates, ids, recovery, events, point_events,
        minimum, maximum, recovery_error,
    )


def factor_for_increment(
    anchor: np.ndarray,
    increment: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    tolerance: float,
) -> float:
    value = anchor + increment
    above = (value > upper + tolerance) & (increment > tolerance)
    below = (value < lower - tolerance) & (increment < -tolerance)
    theta = 1.0
    if np.any(above):
        theta = min(theta, float(np.min((upper[above] - anchor[above]) / increment[above])))
    if np.any(below):
        theta = min(theta, float(np.min((lower[below] - anchor[below]) / increment[below])))
    return float(np.clip(theta, 0.0, 1.0))


def audit_values(
    values: np.ndarray, anchors: np.ndarray, lower: np.ndarray, upper: np.ndarray,
    gamma: float, tolerance: float,
) -> dict:
    overshoot = np.maximum(np.maximum(values - upper[anchors], lower[anchors] - values), 0)
    p = pressure(values, gamma)
    return {
        "states": len(values),
        "violating_states": int(np.count_nonzero(np.any(overshoot > tolerance, axis=1))),
        "violations_by_conservative_component": np.count_nonzero(overshoot > tolerance, axis=0).tolist(),
        "max_overshoot_by_conservative_component": np.max(overshoot, axis=0).tolist(),
        "nonphysical_states": int(np.count_nonzero(
            (values[:, 0] <= 0) | (p <= 0) | ~np.isfinite(p)
        )),
        "minimum_density": float(np.min(values[:, 0])),
        "minimum_pressure": float(np.min(p)),
    }


def z_column_span(values: np.ndarray, columns: np.ndarray) -> list:
    result = np.zeros(values.shape[1])
    order = np.argsort(columns, kind="stable")
    split = np.flatnonzero(np.diff(columns[order])) + 1
    for group in np.split(order, split):
        result = np.maximum(result, np.ptp(values[group], axis=0))
    return result.tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operators", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--maximum-sweeps", type=int, default=20)
    parser.add_argument("--order", choices=("forward", "reverse"), default="forward")
    parser.add_argument("--point-events", action="store_true")
    parser.add_argument("--tolerance", type=float, default=1e-10)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.maximum_sweeps < 1:
        parser.error("--maximum-sweeps must be positive")
    if not np.isfinite(args.tolerance) or args.tolerance < 0:
        parser.error("--tolerance must be finite and nonnegative")

    source = json.loads(args.operators.read_text())
    (
        means, raw_gradients, coordinates, ids, recovery, faces, point_events,
        lower, upper, recovery_error,
    ) = load_operators(source)
    gradients = raw_gradients.copy()
    scales = np.ones(len(means))
    gamma = float(source.get("gamma", 1.4))
    _, columns = np.unique(np.round(coordinates[:, :2], 12), axis=0, return_inverse=True)
    face_traces = [trace for event in faces for trace in event.traces]
    face_anchors = np.array([trace.anchor for trace in face_traces], dtype=np.int64)
    point_anchors = np.arange(len(means))

    def audit() -> dict:
        points = means + np.array([trace.increment(gradients) for trace in recovery])
        face_values = np.array([
            means[trace.anchor] + trace.increment(gradients) for trace in face_traces
        ])
        return {
            "faces": audit_values(face_values, face_anchors, lower, upper, gamma, args.tolerance),
            "points": audit_values(points, point_anchors, lower, upper, gamma, args.tolerance),
            "point_z_column_span_by_conservative_component": z_column_span(points, columns),
            "scale_z_column_span": z_column_span(scales[:, None], columns)[0],
            "max_abs_z_gradient_by_conservative_component": np.max(np.abs(gradients[:, 2, :]), axis=0).tolist(),
            "zero_scale_nodes": int(np.count_nonzero(scales == 0)),
            "scale_below_0_1_nodes": int(np.count_nonzero(scales < 0.1)),
            "unchanged_scale_nodes": int(np.count_nonzero(scales == 1)),
            "gradient_l2_retained_fraction": (
                float(np.linalg.norm(gradients) / np.linalg.norm(raw_gradients))
                if np.any(raw_gradients) else 1.0
            ),
        }

    result = {
        "experiment": "Sequential shared-gradient face limiting with real production integration weights",
        "production_time_advanced": False,
        "mpi_ranks": 1,
        "source": str(args.operators.resolve()),
        "source_configuration": source.get("configuration"),
        "nodes": len(means),
        "physical_interfaces": len(faces),
        "conservative_component_order": ["rho", "rhou", "rhov", "rhow", "rhoE"],
        "bounds": "Fixed per-anchor one-ring conservative mean extrema, including the anchor",
        "theta_input": "Current already-updated shared gradients; recomputed immediately at each interface",
        "support": "Union of both interface-side integration stencils and both point-recovery stencils; deduplicated",
        "extra_point_events": args.point_events,
        "order": args.order,
        "order_definition": "Only the canonical face order is reversed; optional point events run first in original-node-ID order on every sweep",
        "tolerance": args.tolerance,
        "maximum_sweeps": args.maximum_sweeps,
        "point_recovery_export_max_difference": recovery_error,
        "initial": audit(),
        "sweeps": [],
    }
    ordered_points = [point_events[int(i)] for i in np.argsort(ids)]
    ordered_faces = faces if args.order == "forward" else list(reversed(faces))
    events = (ordered_points if args.point_events else []) + ordered_faces
    started = time.monotonic()
    for sweep in range(args.maximum_sweeps):
        updates = 0
        zero_events = 0
        homothety_error = 0.0
        immediate_overshoot = 0.0
        for event in events:
            increments = [trace.increment(gradients) for trace in event.traces]
            theta = min(
                factor_for_increment(means[trace.anchor], increment,
                                     lower[trace.anchor], upper[trace.anchor], args.tolerance)
                for trace, increment in zip(event.traces, increments)
            )
            if theta >= 1:
                continue
            updates += 1
            zero_events += theta == 0
            gradients[event.support] *= theta
            scales[event.support] *= theta
            # This checks the actual update, including recovery dependencies.
            for trace, previous in zip(event.traces, increments):
                updated = trace.increment(gradients)
                homothety_error = max(homothety_error, float(np.max(np.abs(
                    updated - theta * previous
                ))))
                value = means[trace.anchor] + updated
                immediate_overshoot = max(immediate_overshoot, float(np.max(np.maximum(
                    value - upper[trace.anchor], lower[trace.anchor] - value
                ))))
        current = audit()
        current.update({
            "sweep": sweep + 1,
            "updated_events": updates,
            "zero_theta_events": int(zero_events),
            "maximum_event_increment_scaling_error": homothety_error,
            "maximum_overshoot_immediately_after_update": immediate_overshoot,
        })
        result["sweeps"].append(current)
        print(json.dumps({"sweep": sweep + 1, "updates": updates,
                          "face_violations": current["faces"]["violating_states"],
                          "point_violations": current["points"]["violating_states"],
                          "elapsed_seconds": time.monotonic() - started}), flush=True)
        if current["faces"]["violating_states"] == 0 and (
            not args.point_events or current["points"]["violating_states"] == 0
        ):
            break
        if updates == 0:
            break
    result["elapsed_seconds"] = time.monotonic() - started
    result["final"] = result["sweeps"][-1]
    result["converged_requested_bounds"] = (
        result["final"]["faces"]["violating_states"] == 0
        and (not args.point_events or result["final"]["points"]["violating_states"] == 0)
    )
    product_error = float(np.max(np.abs(gradients - raw_gradients * scales[:, None, None])))
    result["accumulated_product_identity_max_error"] = product_error
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    np.savez_compressed(args.output.with_suffix(".npz"),
                        original_id=ids, coordinates=coordinates, scale=scales,
                        raw_gradients=raw_gradients, limited_gradients=gradients)
    print(str(args.output.resolve()), flush=True)


if __name__ == "__main__":
    main()
