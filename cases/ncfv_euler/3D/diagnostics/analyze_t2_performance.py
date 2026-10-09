#!/usr/bin/env python3
"""Summarize paired production NCFV runs that advance the IV mesh to t=2."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


SIZES = (10, 20, 40, 80)
MODES = {"efficient": "EfficientDifferential", "traditional": "TraditionalQuadrature"}


def number(value: object, name: str, source: Path) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{source}: invalid {name}={value!r}")
    return result


def memory(data: dict[str, object], phase: str, metric: str, source: Path) -> float:
    try:
        item = data["memory"][phase][metric]["sum_mib"]  # type: ignore[index]
    except (KeyError, TypeError) as error:
        raise ValueError(f"{source}: missing memory.{phase}.{metric}.sum_mib") from error
    return number(item, f"memory.{phase}.{metric}.sum_mib", source)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")

    records: list[dict[str, object]] = []
    for size in SIZES:
        pair: dict[str, dict[str, object]] = {}
        for label, expected_mode in MODES.items():
            matches = sorted(args.input_root.glob(f"original_iv{size}_{label}_np*.json"))
            if len(matches) != 1:
                raise ValueError(f"IV{size} {label}: expected one result, found {len(matches)}")
            path = matches[0]
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("schema_version") != 1 or data.get("mode") != expected_mode:
                raise ValueError(f"{path}: unexpected benchmark mode")
            if abs(number(data.get("end_time"), "end_time", path) - 2.0) > 1e-12:
                raise ValueError(f"{path}: run did not reach t=2")
            if int(data.get("physical_steps", 0)) <= 0:
                raise ValueError(f"{path}: no physical time steps")
            if "Solver::Run()" not in str(data.get("measurement_scope", "")):
                raise ValueError(f"{path}: not a production time-marching run")
            pair[label] = data

        efficient, traditional = pair["efficient"], pair["traditional"]
        if int(efficient["mpi_ranks"]) != int(traditional["mpi_ranks"]):
            raise ValueError(f"IV{size}: paired MPI rank count differs")
        if (int(efficient["nodes"]), int(efficient["cells"])) != (
            int(traditional["nodes"]), int(traditional["cells"])):
            raise ValueError(f"IV{size}: pair mesh topology differs")
        efficient_march = number(efficient["time_march_seconds"], "time_march_seconds", Path("efficient"))
        traditional_march = number(traditional["time_march_seconds"], "time_march_seconds", Path("traditional"))
        efficient_pss_t2 = memory(efficient, "after_time_2", "pss", Path("efficient"))
        traditional_pss_t2 = memory(traditional, "after_time_2", "pss", Path("traditional"))
        records.append({
            "mesh": f"IV{size}", "size": size, "mpi_ranks": int(efficient["mpi_ranks"]),
            "nodes": int(efficient["nodes"]), "cells": int(efficient["cells"]),
            "efficient_steps": int(efficient["physical_steps"]),
            "traditional_steps": int(traditional["physical_steps"]),
            "efficient_initialize_seconds": number(efficient["initialize_seconds"], "initialize_seconds", Path("efficient")),
            "traditional_initialize_seconds": number(traditional["initialize_seconds"], "initialize_seconds", Path("traditional")),
            "efficient_march_seconds": efficient_march,
            "traditional_march_seconds": traditional_march,
            "traditional_to_efficient_march_ratio": traditional_march / efficient_march,
            "efficient_total_seconds": number(efficient["total_seconds"], "total_seconds", Path("efficient")),
            "traditional_total_seconds": number(traditional["total_seconds"], "total_seconds", Path("traditional")),
            "efficient_initial_pss_mib": memory(efficient, "after_initialize", "pss", Path("efficient")),
            "traditional_initial_pss_mib": memory(traditional, "after_initialize", "pss", Path("traditional")),
            "efficient_t2_pss_mib": efficient_pss_t2,
            "traditional_t2_pss_mib": traditional_pss_t2,
            "pss_saving_mib": traditional_pss_t2 - efficient_pss_t2,
            "pss_saving_percent": 100.0 * (traditional_pss_t2 - efficient_pss_t2) / traditional_pss_t2,
            "efficient_t2_hwm_mib": memory(efficient, "after_time_2", "hwm", Path("efficient")),
            "traditional_t2_hwm_mib": memory(traditional, "after_time_2", "hwm", Path("traditional")),
            "efficient_pss_growth_mib": efficient_pss_t2 - memory(efficient, "after_initialize", "pss", Path("efficient")),
            "traditional_pss_growth_mib": traditional_pss_t2 - memory(traditional, "after_initialize", "pss", Path("traditional")),
        })

    args.output_dir.mkdir(parents=True)
    with (args.output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    validation = {
        "schema_version": 1,
        "mesh_family": "original IV Prism6",
        "mpi_ranks_by_mesh": {record["mesh"]: record["mpi_ranks"] for record in records},
        "pairs": len(records),
        "all_runs_reached_time": 2.0,
        "all_runs_use_production_solver_run": True,
        "output_io_disabled": True,
        "note": "Each rank is single-threaded; PSS/RSS/HWM are sums over all ranks.",
    }
    (args.output_dir / "validation.json").write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
    print(args.output_dir / "summary.csv")


if __name__ == "__main__":
    main()
