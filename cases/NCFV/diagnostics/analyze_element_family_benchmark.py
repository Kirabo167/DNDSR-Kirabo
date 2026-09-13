#!/usr/bin/env python3
"""Summarize NCFV family benchmarks into paired timing and memory tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


PHASES = ("pre_initialize", "after_initialize", "after_warmup", "after_measurement")
METRICS = ("rss", "pss", "hwm")
MODE_LABELS = {
    "EfficientDifferential": "efficient",
    "TraditionalQuadrature": "traditional",
}


def read_number(value: Any, name: str, source: Path) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{source}: {name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{source}: invalid {name}={value!r}")
    return result


def infer_family(path: Path, fallback: str | None) -> str:
    if fallback is not None:
        return fallback
    match = re.match(r"(hex|tet)_iv(?:10|20|40|80)_", path.name)
    if not match:
        raise ValueError(f"cannot infer mesh family from {path.name}")
    return match.group(1)


def load_records(root: Path, fallback_family: str | None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(root.glob("*.json")):
        if path.name == "manifest.json":
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or data.get("schema_version") != 1:
            continue
        mode = MODE_LABELS.get(data.get("mode"))
        if mode is None:
            continue
        mesh_match = re.search(r"iv(10|20|40|80)", path.name, re.IGNORECASE)
        if mesh_match is None:
            raise ValueError(f"cannot infer IV size from {path}")
        memory = data.get("memory")
        if not isinstance(memory, dict):
            raise ValueError(f"{path}: missing memory object")
        snapshots: dict[str, dict[str, dict[str, float]]] = {}
        for phase in PHASES:
            phase_data = memory.get(phase)
            if not isinstance(phase_data, dict):
                raise ValueError(f"{path}: missing memory.{phase}")
            snapshots[phase] = {}
            for metric in METRICS:
                metric_data = phase_data.get(metric)
                if not isinstance(metric_data, dict):
                    raise ValueError(f"{path}: missing memory.{phase}.{metric}")
                snapshots[phase][metric] = {
                    "sum_mib": read_number(metric_data.get("sum_mib"),
                                           f"memory.{phase}.{metric}.sum_mib", path),
                    "max_mib": read_number(metric_data.get("max_mib"),
                                           f"memory.{phase}.{metric}.max_mib", path),
                }
        samples = data.get("rhs_seconds_per_call")
        if not isinstance(samples, list) or not samples:
            raise ValueError(f"{path}: missing RHS samples")
        rhs_median = statistics.median(
            read_number(value, "rhs_seconds_per_call", path) for value in samples
        )
        records.append({
            "source": str(path),
            "family": infer_family(path, fallback_family),
            "size": int(mesh_match.group(1)),
            "mode": mode,
            "mpi_ranks": int(data["mpi_ranks"]),
            "repetitions": int(data["repetitions"]),
            "nodes": int(data["nodes"]),
            "cells": int(data["cells"]),
            "edges": int(data["edges"]),
            "mesh_file": str(data["mesh_file"]),
            "rhs_seconds_median": rhs_median,
            "initialize_seconds": read_number(data["initialize_seconds"], "initialize_seconds", path),
            "warmup_seconds": read_number(data["warmup_seconds"], "warmup_seconds", path),
            "residual_spread": abs(
                read_number(data["residual"]["last_rank_max"], "last_rank_max", path) -
                read_number(data["residual"]["last_rank_min"], "last_rank_min", path)
            ),
            "quadrature_total": int(data["quadrature_points"]["total"]),
            "memory": snapshots,
        })
    return records


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--element-root", type=Path, required=True)
    parser.add_argument("--original-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    element_records = load_records(args.element_root, None)
    original_records = load_records(args.original_root, "prism")
    records = element_records + original_records
    if len(element_records) != 32:
        raise ValueError(f"expected 32 Tet4/Hex8 records, found {len(element_records)}")
    if len(original_records) != 16:
        raise ValueError(f"expected 16 original-prism records, found {len(original_records)}")

    groups: dict[tuple[str, int, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        key = record["family"], record["size"], record["mpi_ranks"]
        if record["mode"] in groups[key]:
            raise ValueError(f"duplicate benchmark record for {key} / {record['mode']}")
        groups[key][record["mode"]] = record
    for key, pair in groups.items():
        if set(pair) != {"efficient", "traditional"}:
            raise ValueError(f"incomplete paired result for {key}")
        for mode, record in pair.items():
            if mode == "efficient" and record["quadrature_total"] != 0:
                raise ValueError(f"efficient result unexpectedly has quadrature points: {record['source']}")
            if mode == "traditional" and record["quadrature_total"] <= 0:
                raise ValueError(f"traditional result has no quadrature points: {record['source']}")
            if record["residual_spread"] > 1e-10:
                raise ValueError(f"rank-dependent residual in {record['source']}: {record['residual_spread']}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    phase_rows: list[dict[str, Any]] = []
    for record in sorted(records, key=lambda row: (row["family"], row["size"], row["mpi_ranks"], row["mode"])):
        row = {key: record[key] for key in (
            "family", "size", "mode", "mpi_ranks", "repetitions", "nodes", "cells", "edges",
            "mesh_file", "initialize_seconds", "warmup_seconds", "rhs_seconds_median",
            "residual_spread", "quadrature_total", "source")}
        for phase in PHASES:
            for metric in METRICS:
                for kind in ("sum_mib", "max_mib"):
                    row[f"{phase}_{metric}_{kind}"] = record["memory"][phase][metric][kind]
        phase_rows.append(row)
    phase_fields = list(phase_rows[0])
    write_csv(args.output_dir / "memory_phases.csv", phase_fields, phase_rows)

    pair_rows: list[dict[str, Any]] = []
    for family, size, ranks in sorted(groups):
        efficient = groups[(family, size, ranks)]["efficient"]
        traditional = groups[(family, size, ranks)]["traditional"]
        if (efficient["nodes"], efficient["cells"], efficient["edges"]) != (
            traditional["nodes"], traditional["cells"], traditional["edges"]):
            raise ValueError(f"paired mesh topology mismatch for {(family, size, ranks)}")
        row: dict[str, Any] = {
            "family": family,
            "size": size,
            "mpi_ranks": ranks,
            "nodes": efficient["nodes"],
            "cells": efficient["cells"],
            "edges": efficient["edges"],
            "repetitions": efficient["repetitions"],
            "efficient_rhs_ms": efficient["rhs_seconds_median"] * 1000.0,
            "traditional_rhs_ms": traditional["rhs_seconds_median"] * 1000.0,
            "traditional_to_efficient_rhs_ratio": traditional["rhs_seconds_median"] / efficient["rhs_seconds_median"],
            "efficient_initialize_seconds": efficient["initialize_seconds"],
            "traditional_initialize_seconds": traditional["initialize_seconds"],
            "traditional_to_efficient_initialize_ratio": traditional["initialize_seconds"] / efficient["initialize_seconds"],
            "traditional_quadrature_points": traditional["quadrature_total"],
        }
        for phase in PHASES:
            for metric in METRICS:
                efficient_value = efficient["memory"][phase][metric]["sum_mib"]
                traditional_value = traditional["memory"][phase][metric]["sum_mib"]
                row[f"efficient_{phase}_{metric}_sum_mib"] = efficient_value
                row[f"traditional_{phase}_{metric}_sum_mib"] = traditional_value
        for metric in METRICS:
            efficient_final = efficient["memory"]["after_measurement"][metric]["sum_mib"]
            traditional_final = traditional["memory"]["after_measurement"][metric]["sum_mib"]
            row[f"efficient_final_{metric}_sum_mib"] = efficient_final
            row[f"traditional_final_{metric}_sum_mib"] = traditional_final
            row[f"{metric}_saving_mib"] = traditional_final - efficient_final
            row[f"{metric}_saving_percent"] = 100.0 * (traditional_final - efficient_final) / traditional_final
            row[f"efficient_final_{metric}_max_mib"] = efficient["memory"]["after_measurement"][metric]["max_mib"]
            row[f"traditional_final_{metric}_max_mib"] = traditional["memory"]["after_measurement"][metric]["max_mib"]
        pair_rows.append(row)
    write_csv(args.output_dir / "summary.csv", list(pair_rows[0]), pair_rows)

    validation = {
        "element_records": len(element_records),
        "original_records": len(original_records),
        "paired_groups": len(pair_rows),
        "maximum_residual_rank_spread": max(record["residual_spread"] for record in records),
        "checks": {
            "all_pairs_complete": True,
            "efficient_quadrature_zero": True,
            "traditional_quadrature_positive": True,
            "all_residuals_rank_consistent": True,
        },
    }
    (args.output_dir / "validation.json").write_text(
        json.dumps(validation, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
