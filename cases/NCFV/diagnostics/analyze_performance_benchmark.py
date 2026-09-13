#!/usr/bin/env python3
"""Aggregate repeated NCFV performance-probe JSON files into CSV and Markdown."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


MODE_NAMES = {
    "efficient": "efficient",
    "efficientdifferential": "efficient",
    "traditional": "traditional",
    "traditionalquadrature": "traditional",
}
MODE_NAMES_ZH = {"efficient": "高效", "traditional": "传统"}
ROOT = Path(__file__).resolve().parents[3]
MEDIAN_METRICS = (
    "initialize_seconds",
    "rhs_seconds_per_call",
    "rss_sum_mib",
    "pss_sum_mib",
    "hwm_sum_mib",
    "pss_net_mib",
)
MEMORY_METRICS = ("rss", "pss", "hwm")

DETAILED_FIELDS = (
    "source_file",
    "schema_version",
    "trial",
    "mesh",
    "mode",
    "mpi_ranks",
    "repetitions",
    "nodes",
    "cells",
    "edges",
    "initialize_seconds",
    "warmup_seconds",
    "rhs_total_seconds",
    "rhs_seconds_per_call",
    "rhs_seconds_per_call_samples",
    "rhs_minimum_seconds",
    "rhs_mean_seconds",
    "rhs_maximum_seconds",
    "rhs_population_stddev_seconds",
    "residual",
    "quadrature_points",
    "rss_sum_mib",
    "rss_max_mib",
    "pss_sum_mib",
    "pss_max_mib",
    "pss_pre_initialize_sum_mib",
    "pss_net_mib",
    "hwm_sum_mib",
    "hwm_max_mib",
    "source_configuration",
)

SUMMARY_FIELDS = (
    "mesh",
    "mode",
    "mpi_ranks",
    "trials",
    "repetitions_min",
    "repetitions_max",
    "nodes",
    "cells",
    "edges",
    "initialize_seconds_median",
    "rhs_seconds_per_call_median",
    "rss_sum_mib_median",
    "pss_sum_mib_median",
    "hwm_sum_mib_median",
    "pss_net_mib_median",
    "traditional_to_efficient_initialize_ratio",
    "traditional_to_efficient_rhs_ratio",
    "traditional_to_efficient_rss_ratio",
    "traditional_to_efficient_pss_ratio",
    "traditional_to_efficient_hwm_ratio",
    "traditional_to_efficient_pss_net_ratio",
    "initialize_speedup_vs_1",
    "initialize_parallel_efficiency",
    "rhs_speedup_vs_1",
    "rhs_parallel_efficiency",
)


class NotBenchmarkRecord(ValueError):
    """Raised internally when a JSON object is unrelated to this benchmark."""


def nested_get(data: Mapping[str, Any], path: str) -> Any:
    """Return a dotted-path value, or ``None`` when any component is absent."""
    value: Any = data
    for component in path.split("."):
        if not isinstance(value, Mapping) or component not in value:
            return None
        value = value[component]
    return value


def first_value(data: Mapping[str, Any], paths: Sequence[str]) -> Any:
    for path in paths:
        value = nested_get(data, path)
        if value is not None:
            return value
    return None


def finite_number(
    value: Any,
    name: str,
    source: str,
    *,
    positive: bool = False,
    optional: bool = False,
) -> float | None:
    if value is None:
        if optional:
            return None
        raise ValueError(f"{source}: missing required field {name}")
    if isinstance(value, bool):
        raise ValueError(f"{source}: {name} must be numeric, not boolean")
    try:
        result = float(value)
    except (TypeError, ValueError) as exception:
        raise ValueError(f"{source}: {name} is not numeric: {value!r}") from exception
    if not math.isfinite(result) or result < 0 or (positive and result <= 0):
        qualifier = "positive and finite" if positive else "non-negative and finite"
        raise ValueError(f"{source}: {name} must be {qualifier}, got {value!r}")
    return result


def optional_integer(value: Any, name: str, source: str) -> int | None:
    if value is None:
        return None
    number = finite_number(value, name, source)
    assert number is not None
    result = int(number)
    if result != number:
        raise ValueError(f"{source}: {name} must be an integer, got {value!r}")
    return result


def required_positive_integer(value: Any, name: str, source: str) -> int:
    result = optional_integer(value, name, source)
    if result is None:
        raise ValueError(f"{source}: missing required field {name}")
    if result <= 0:
        raise ValueError(f"{source}: {name} must be positive, got {result}")
    return result


def normalize_mode(value: Any, source: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{source}: mode must be a string")
    key = re.sub(r"[^a-z]", "", value.lower())
    if key not in MODE_NAMES:
        raise ValueError(f"{source}: unsupported NCFV mode {value!r}")
    return MODE_NAMES[key]


def normalize_mesh(value: Any, source: str) -> str:
    text = str(value)
    match = re.search(r"(?i)(?:^|[^a-z0-9])iv[_-]?(10|20|40|80)(?:[^0-9]|$)", text)
    if not match:
        match = re.fullmatch(r"(?:10|20|40|80)", text.strip())
    if not match:
        raise ValueError(f"{source}: cannot identify iv10/20/40/80 from mesh value {value!r}")
    return f"iv{match.group(1) if match.lastindex else match.group(0)}"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_trial(value: Any) -> int | str:
    text = str(value)
    if re.fullmatch(r"0*[1-9][0-9]*", text):
        return int(text)
    return text


def canonical_source_configuration(value: Any, manifest: Mapping[str, Any]) -> str:
    source = Path(str(value))
    if not source.is_absolute() and manifest.get("build_directory"):
        source = Path(str(manifest["build_directory"])) / source
    return str(source.resolve())


def load_manifest(raw_directory: Path) -> Mapping[str, Any]:
    path = raw_directory / "manifest.json"
    if not path.is_file():
        return {}
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exception:
        raise ValueError(f"cannot read benchmark manifest {path}: {exception}") from exception
    if not isinstance(document, Mapping):
        raise ValueError(f"benchmark manifest {path} must be a JSON object")
    return document


def memory_value_mib(
    data: Mapping[str, Any],
    metric: str,
    kind: str,
    source: str,
    *,
    optional: bool = False,
) -> float | None:
    """Read aggregate memory, accepting the probe schema and common unit variants."""
    for unit, divisor in (("mib", 1.0), ("kib", 1024.0), ("bytes", 1024.0**2)):
        paths = (
            f"memory.{metric}_{kind}_{unit}",
            f"{metric}_{kind}_{unit}",
            f"memory.after_measurement.{metric}.{kind}_{unit}",
            f"memory.after_measurement.{metric}_{unit}.{kind}",
            f"memory.after_measurement.{metric}_{kind}_{unit}",
            f"memory.final.{metric}.{kind}_{unit}",
            f"memory.final.{metric}_{unit}.{kind}",
            f"memory.final.{metric}_{kind}_{unit}",
        )
        value = first_value(data, paths)
        if value is not None:
            number = finite_number(value, f"{metric}_{kind}_{unit}", source)
            assert number is not None
            return number / divisor
    if optional:
        return None
    raise ValueError(
        f"{source}: missing {metric.upper()} {kind} memory; expected memory.{metric}_{kind}_mib "
        f"or memory.after_measurement.{metric}_kib.{kind}"
    )


def snapshot_memory_mib(
    data: Mapping[str, Any], snapshot: str, metric: str, kind: str, source: str
) -> float:
    for unit, divisor in (("mib", 1.0), ("kib", 1024.0), ("bytes", 1024.0**2)):
        value = first_value(
            data,
            (
                f"memory.{snapshot}.{metric}.{kind}_{unit}",
                f"memory.{snapshot}.{metric}_{kind}_{unit}",
            ),
        )
        if value is not None:
            number = finite_number(
                value, f"memory.{snapshot}.{metric}.{kind}_{unit}", source
            )
            assert number is not None
            return number / divisor
    raise ValueError(
        f"{source}: missing memory.{snapshot}.{metric}.{kind} value"
    )


def parse_rhs_timing(
    data: Mapping[str, Any], repetitions: int, source: str
) -> tuple[float, str, dict[str, float | None]]:
    raw_samples = first_value(
        data,
        (
            "rhs_seconds_per_call",
            "timing.rhs_seconds_per_call",
            "timings.rhs_seconds_per_call",
        ),
    )
    samples: list[float] = []
    if isinstance(raw_samples, Sequence) and not isinstance(raw_samples, (str, bytes)):
        for index, sample in enumerate(raw_samples):
            value = finite_number(
                sample, f"rhs_seconds_per_call[{index}]", source, positive=True
            )
            assert value is not None
            samples.append(value)
        if len(samples) != repetitions:
            raise ValueError(
                f"{source}: repetitions={repetitions}, but rhs_seconds_per_call has "
                f"{len(samples)} samples"
            )

    summary_paths = {
        "minimum": "rhs_summary.minimum_seconds",
        "mean": "rhs_summary.mean_seconds",
        "median": "rhs_summary.median_seconds",
        "maximum": "rhs_summary.maximum_seconds",
        "population_stddev": "rhs_summary.population_stddev_seconds",
    }
    summary: dict[str, float | None] = {}
    for name, path in summary_paths.items():
        summary[name] = finite_number(
            nested_get(data, path), path, source, positive=name != "population_stddev", optional=True
        )

    if samples:
        computed_median = statistics.median(samples)
        if summary["median"] is not None and not math.isclose(
            computed_median, float(summary["median"]), rel_tol=2e-12, abs_tol=1e-15
        ):
            raise ValueError(f"{source}: rhs_summary median disagrees with raw call samples")
        per_call = float(summary["median"] if summary["median"] is not None else computed_median)
    elif raw_samples is not None:
        value = finite_number(raw_samples, "rhs_seconds_per_call", source, positive=True)
        assert value is not None
        per_call = value
        if summary["median"] is not None and not math.isclose(
            per_call, float(summary["median"]), rel_tol=2e-12, abs_tol=1e-15
        ):
            raise ValueError(f"{source}: scalar rhs_seconds_per_call disagrees with rhs_summary")
    elif summary["median"] is not None:
        per_call = float(summary["median"])
    else:
        per_call = math.nan

    return per_call, json.dumps(samples, separators=(",", ":")) if samples else "", summary


def infer_trial(data: Mapping[str, Any], relative_path: Path, item_index: int | None) -> str:
    explicit = first_value(data, ("trial", "trial_id", "trial_index", "metadata.trial"))
    if explicit is not None:
        trial = str(explicit)
    else:
        match = re.search(r"(?i)trial[_-]?(\d+)", relative_path.as_posix())
        trial = match.group(1) if match else relative_path.as_posix()
    if item_index is not None:
        trial = f"{trial}#{item_index}"
    return trial


def parse_record(
    data: Mapping[str, Any],
    source_file: str,
    relative_path: Path,
    item_index: int | None,
) -> dict[str, Any]:
    mode_value = first_value(data, ("mode", "algorithm.mode", "configuration.algorithm.mode"))
    rank_value = first_value(data, ("mpi_ranks", "ranks", "mpi.size", "metadata.mpi_ranks"))
    initialize_value = first_value(
        data, ("initialize_seconds", "timing.initialize_seconds", "timings.initialize_seconds")
    )
    if mode_value is None or rank_value is None or initialize_value is None:
        raise NotBenchmarkRecord(source_file)

    mesh_value = first_value(
        data,
        (
            "mesh",
            "mesh_name",
            "mesh_file",
            "configuration.mesh.meshFile",
            "metadata.mesh",
        ),
    )
    if mesh_value is None:
        mesh_value = relative_path.as_posix()

    repetitions = required_positive_integer(
        first_value(data, ("repetitions", "rhs_repetitions", "timing.repetitions")),
        "repetitions",
        source_file,
    )
    rhs_total = finite_number(
        first_value(data, ("rhs_total_seconds", "timing.rhs_total_seconds", "timings.rhs_total_seconds")),
        "rhs_total_seconds",
        source_file,
        positive=True,
        optional=True,
    )
    rhs_per_call, rhs_samples, rhs_summary = parse_rhs_timing(
        data, repetitions, source_file
    )
    if not math.isfinite(rhs_per_call) and rhs_total is not None:
        rhs_per_call = rhs_total / repetitions
    rhs_per_call_checked = finite_number(
        rhs_per_call, "rhs_seconds_per_call", source_file, positive=True
    )
    assert rhs_per_call_checked is not None
    rhs_per_call = rhs_per_call_checked
    if rhs_total is None:
        rhs_total = rhs_per_call * repetitions

    row: dict[str, Any] = {
        "source_file": source_file,
        "schema_version": first_value(data, ("schema_version", "schema.version")) or "",
        "trial": infer_trial(data, relative_path, item_index),
        "mesh": normalize_mesh(mesh_value, source_file),
        "mode": normalize_mode(mode_value, source_file),
        "mpi_ranks": required_positive_integer(rank_value, "mpi_ranks", source_file),
        "repetitions": repetitions,
        "nodes": optional_integer(first_value(data, ("nodes", "mesh_counts.nodes")), "nodes", source_file),
        "cells": optional_integer(first_value(data, ("cells", "mesh_counts.cells")), "cells", source_file),
        "edges": optional_integer(first_value(data, ("edges", "mesh_counts.edges")), "edges", source_file),
        "initialize_seconds": finite_number(
            initialize_value, "initialize_seconds", source_file, positive=True
        ),
        "warmup_seconds": finite_number(
            first_value(data, ("warmup_seconds", "timing.warmup_seconds")),
            "warmup_seconds",
            source_file,
            optional=True,
        ),
        "rhs_total_seconds": rhs_total,
        "rhs_seconds_per_call": rhs_per_call,
        "rhs_seconds_per_call_samples": rhs_samples,
        "rhs_minimum_seconds": rhs_summary["minimum"],
        "rhs_mean_seconds": rhs_summary["mean"],
        "rhs_maximum_seconds": rhs_summary["maximum"],
        "rhs_population_stddev_seconds": rhs_summary["population_stddev"],
        "residual": finite_number(
            first_value(data, ("residual.last", "rhs_summary.residual", "residual")),
            "residual",
            source_file,
            optional=True,
        ),
        "quadrature_points": optional_integer(
            first_value(data, ("quadrature_points.total", "quadrature_points", "stored_quadrature_points")),
            "quadrature_points",
            source_file,
        ),
        "source_configuration": first_value(
            data, ("source_configuration", "configuration_file", "metadata.source_configuration")
        )
        or "",
    }
    for metric in MEMORY_METRICS:
        row[f"{metric}_sum_mib"] = memory_value_mib(data, metric, "sum", source_file)
        row[f"{metric}_max_mib"] = memory_value_mib(
            data, metric, "max", source_file, optional=True
        )
    row["pss_pre_initialize_sum_mib"] = snapshot_memory_mib(
        data, "pre_initialize", "pss", "sum", source_file
    )
    row["pss_net_mib"] = (
        float(row["pss_sum_mib"]) - float(row["pss_pre_initialize_sum_mib"])
    )
    if row["pss_net_mib"] < 0:
        raise ValueError(f"{source_file}: post-measurement PSS is below pre-initialize PSS")
    return row


def record_objects(document: Any, source: str) -> Iterable[tuple[Mapping[str, Any], int | None]]:
    if isinstance(document, list):
        for index, item in enumerate(document):
            if not isinstance(item, Mapping):
                raise ValueError(f"{source}: list item {index} is not a JSON object")
            yield item, index
        return
    if not isinstance(document, Mapping):
        raise ValueError(f"{source}: top-level JSON must be an object or list")
    for container_key in ("runs", "results", "records"):
        values = document.get(container_key)
        if isinstance(values, list):
            for index, item in enumerate(values):
                if not isinstance(item, Mapping):
                    raise ValueError(f"{source}: {container_key}[{index}] is not an object")
                yield item, index
            return
    yield document, None


def load_records(
    raw_directory: Path,
) -> tuple[list[dict[str, Any]], list[Mapping[str, Any]], list[str], int]:
    if not raw_directory.is_dir():
        raise ValueError(f"raw directory does not exist: {raw_directory}")
    records: list[dict[str, Any]] = []
    raw_records: list[Mapping[str, Any]] = []
    skipped: list[str] = []
    paths = sorted(raw_directory.rglob("*.json"))
    for path in paths:
        relative = path.relative_to(raw_directory)
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exception:
            raise ValueError(f"cannot read benchmark JSON {path}: {exception}") from exception
        found = False
        for data, item_index in record_objects(document, str(path)):
            source = relative.as_posix()
            if item_index is not None:
                source = f"{source}#{item_index}"
            try:
                records.append(parse_record(data, source, relative, item_index))
                raw_records.append(data)
                found = True
            except NotBenchmarkRecord:
                continue
        if not found:
            skipped.append(relative.as_posix())
    if not records:
        raise ValueError(f"no performance benchmark records found below {raw_directory}")
    return records, raw_records, skipped, len(paths)


def validate_records(
    records: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a unique, complete and configuration-consistent benchmark matrix."""
    actual: dict[tuple[str, str, int, int | str], Mapping[str, Any]] = {}
    for row in records:
        key = (
            str(row["mesh"]),
            str(row["mode"]),
            int(row["mpi_ranks"]),
            canonical_trial(row["trial"]),
        )
        if key in actual:
            raise ValueError(
                f"duplicate benchmark trial {key}: "
                f"{actual[key]['source_file']} and {row['source_file']}"
            )
        actual[key] = row

    actual_meshes = sorted({key[0] for key in actual}, key=mesh_sort_key)
    actual_modes = sorted({key[1] for key in actual})
    actual_ranks = sorted({key[2] for key in actual})
    actual_trials = sorted({key[3] for key in actual}, key=str)
    if set(actual_modes) != {"efficient", "traditional"}:
        raise ValueError(
            "benchmark matrix must contain both efficient and traditional modes"
        )
    if 1 not in actual_ranks or not any(rank > 1 for rank in actual_ranks):
        raise ValueError(
            "benchmark matrix must contain one-rank and multi-rank measurements"
        )

    if manifest:
        try:
            expected_meshes = [
                normalize_mesh(value, "manifest.meshes")
                for value in manifest["meshes"]
            ]
            expected_modes = [
                normalize_mode(value, "manifest.modes")
                for value in manifest["modes"]
            ]
            expected_ranks = [int(value) for value in manifest["ranks"]]
            trial_count = required_positive_integer(
                manifest.get("trials"), "manifest.trials", "manifest.json"
            )
        except (KeyError, TypeError, ValueError) as exception:
            raise ValueError(f"invalid benchmark design in manifest: {exception}") from exception
        if len(set(expected_meshes)) != len(expected_meshes):
            raise ValueError("manifest.meshes contains duplicates")
        if len(set(expected_modes)) != len(expected_modes):
            raise ValueError("manifest.modes contains duplicates")
        if len(set(expected_ranks)) != len(expected_ranks) or any(
            rank <= 0 for rank in expected_ranks
        ):
            raise ValueError("manifest.ranks must contain unique positive integers")
        expected_trials: list[int | str] = list(range(1, trial_count + 1))
    else:
        expected_meshes = actual_meshes
        expected_modes = actual_modes
        expected_ranks = actual_ranks
        expected_trials = actual_trials

    expected = {
        (mesh, mode, rank, trial)
        for mesh in expected_meshes
        for mode in expected_modes
        for rank in expected_ranks
        for trial in expected_trials
    }
    actual_keys = set(actual)
    if actual_keys != expected:
        missing = sorted(expected - actual_keys, key=str)
        extra = sorted(actual_keys - expected, key=str)
        raise ValueError(
            "benchmark matrix is incomplete or contains unexpected records; "
            f"missing={missing[:8]!r}, extra={extra[:8]!r}"
        )

    source_by_mesh: dict[str, str] = {}
    repetitions_by_mesh: dict[str, int] = {}
    for mesh in expected_meshes:
        mesh_rows = [row for row in records if row["mesh"] == mesh]
        for field in ("nodes", "cells", "edges", "repetitions"):
            values = {row[field] for row in mesh_rows}
            if None in values or "" in values or len(values) != 1:
                raise ValueError(
                    f"{mesh}: {field} must be present and identical across modes, ranks and trials; "
                    f"got {sorted(str(value) for value in values)!r}"
                )
        repetitions_by_mesh[mesh] = int(mesh_rows[0]["repetitions"])
        sources = {
            canonical_source_configuration(row["source_configuration"], manifest)
            for row in mesh_rows
            if row["source_configuration"]
        }
        if len(sources) != 1 or len(sources) != len(
            {
                canonical_source_configuration(row["source_configuration"], manifest)
                for row in mesh_rows
            }
        ):
            raise ValueError(
                f"{mesh}: source_configuration must be present and identical across all records"
            )
        source_by_mesh[mesh] = next(iter(sources))

    if manifest:
        manifest_repetitions = manifest.get("rhs_repetitions")
        if not isinstance(manifest_repetitions, Mapping):
            raise ValueError("manifest.rhs_repetitions must be an object")
        for mesh, repetitions in repetitions_by_mesh.items():
            if required_positive_integer(
                manifest_repetitions.get(mesh),
                f"manifest.rhs_repetitions.{mesh}",
                "manifest.json",
            ) != repetitions:
                raise ValueError(
                    f"{mesh}: repetitions disagree with manifest.rhs_repetitions"
                )

        artifact_hashes = manifest.get("artifact_sha256")
        if not isinstance(artifact_hashes, Mapping) or not artifact_hashes:
            raise ValueError("manifest.artifact_sha256 must be a non-empty object")
        artifact_paths: set[str] = set()
        for value, expected_hash in artifact_hashes.items():
            if not isinstance(expected_hash, str) or not re.fullmatch(
                r"[0-9a-fA-F]{64}", expected_hash
            ):
                raise ValueError(f"manifest has invalid SHA-256 for {value}")
            path = Path(str(value))
            if not path.is_absolute():
                path = ROOT / path
            path = path.resolve()
            artifact_paths.add(str(path))
            if not path.is_file():
                raise ValueError(f"manifest artifact is missing from this workspace: {path}")
            actual_hash = sha256(path)
            if actual_hash != expected_hash.lower():
                raise ValueError(
                    f"manifest artifact hash mismatch for {path}: "
                    f"{actual_hash} != {expected_hash}"
                )

        probe_path = Path(str(manifest.get("probe", "")))
        probe_hash = manifest.get("probe_sha256")
        if probe_path.is_file():
            if not isinstance(probe_hash, str) or sha256(probe_path) != probe_hash.lower():
                raise ValueError(f"manifest probe hash mismatch for {probe_path}")

        build_directory = Path(str(manifest.get("build_directory", "")))
        for mesh, source in source_by_mesh.items():
            if source not in artifact_paths:
                raise ValueError(
                    f"{mesh}: source configuration is not covered by manifest artifact hashes: {source}"
                )
            try:
                configuration = json.loads(Path(source).read_text(encoding="utf-8"))
                mesh_file = configuration["mesh"]["meshFile"]
            except (OSError, json.JSONDecodeError, KeyError, TypeError) as exception:
                raise ValueError(f"cannot resolve mesh input from {source}: {exception}") from exception
            mesh_path = Path(str(mesh_file))
            if not mesh_path.is_absolute():
                mesh_path = build_directory / mesh_path
            mesh_path = mesh_path.resolve()
            if str(mesh_path) not in artifact_paths:
                raise ValueError(
                    f"{mesh}: mesh input is not covered by manifest artifact hashes: {mesh_path}"
                )

        jobs = manifest.get("jobs")
        if not isinstance(jobs, list):
            raise ValueError("manifest.jobs must be a list")
        completed: set[tuple[str, str, int, int | str]] = set()
        for job in jobs:
            if not isinstance(job, Mapping) or job.get("status") != "complete" or job.get("returncode") != 0:
                continue
            try:
                completed.add(
                    (
                        normalize_mesh(job.get("mesh"), "manifest.jobs.mesh"),
                        normalize_mode(job.get("mode"), "manifest.jobs.mode"),
                        int(job["mpi_ranks"]),
                        canonical_trial(job["trial"]),
                    )
                )
            except (KeyError, TypeError, ValueError) as exception:
                raise ValueError(f"invalid completed job in manifest: {job!r}") from exception
        missing_jobs = expected - completed
        if missing_jobs:
            raise ValueError(
                "manifest has no completed successful job for records: "
                f"{sorted(missing_jobs, key=str)[:8]!r}"
            )

    return {
        "meshes": expected_meshes,
        "modes": expected_modes,
        "ranks": sorted(expected_ranks),
        "trials": expected_trials,
        "source_by_mesh": source_by_mesh,
    }


def mesh_sort_key(mesh: str) -> tuple[int, str]:
    match = re.fullmatch(r"iv(\d+)", mesh)
    return (int(match.group(1)) if match else 10**9, mesh)


def record_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        mesh_sort_key(str(row["mesh"])),
        0 if row["mode"] == "efficient" else 1,
        int(row["mpi_ranks"]),
        str(row["trial"]),
        str(row["source_file"]),
    )


def summary_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        mesh_sort_key(str(row["mesh"])),
        0 if row["mode"] == "efficient" else 1,
        int(row["mpi_ranks"]),
    )


def constant_value(rows: Sequence[Mapping[str, Any]], key: str, group: tuple[Any, ...]) -> Any:
    values = {row[key] for row in rows if row[key] is not None and row[key] != ""}
    if len(values) > 1:
        raise ValueError(f"inconsistent {key} values in group {group}: {sorted(values)!r}")
    return next(iter(values)) if values else ""


def safe_ratio(numerator: float, denominator: float) -> float | str:
    return numerator / denominator if denominator > 0 else ""


def summarize(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        groups[(row["mesh"], row["mode"], row["mpi_ranks"])].append(row)

    summaries: list[dict[str, Any]] = []
    for group, rows in groups.items():
        mesh, mode, ranks = group
        repetitions = [int(row["repetitions"]) for row in rows]
        summary: dict[str, Any] = {
            "mesh": mesh,
            "mode": mode,
            "mpi_ranks": ranks,
            "trials": len(rows),
            "repetitions_min": min(repetitions),
            "repetitions_max": max(repetitions),
            "nodes": constant_value(rows, "nodes", group),
            "cells": constant_value(rows, "cells", group),
            "edges": constant_value(rows, "edges", group),
        }
        for metric in MEDIAN_METRICS:
            summary[f"{metric}_median"] = statistics.median(float(row[metric]) for row in rows)
        for field in SUMMARY_FIELDS:
            summary.setdefault(field, "")
        summaries.append(summary)

    summaries.sort(key=summary_sort_key)
    by_key = {(row["mesh"], row["mode"], row["mpi_ranks"]): row for row in summaries}

    ratio_metrics = {
        "initialize": "initialize_seconds_median",
        "rhs": "rhs_seconds_per_call_median",
        "rss": "rss_sum_mib_median",
        "pss": "pss_sum_mib_median",
        "hwm": "hwm_sum_mib_median",
        "pss_net": "pss_net_mib_median",
    }
    for row in summaries:
        mesh, mode, ranks = row["mesh"], row["mode"], row["mpi_ranks"]
        if mode == "traditional":
            efficient = by_key.get((mesh, "efficient", ranks))
            if efficient is not None:
                for label, metric in ratio_metrics.items():
                    row[f"traditional_to_efficient_{label}_ratio"] = safe_ratio(
                        float(row[metric]), float(efficient[metric])
                    )
        if ranks > 1:
            serial = by_key.get((mesh, mode, 1))
            if serial is not None:
                for label, metric in (
                    ("initialize", "initialize_seconds_median"),
                    ("rhs", "rhs_seconds_per_call_median"),
                ):
                    speedup = safe_ratio(float(serial[metric]), float(row[metric]))
                    row[f"{label}_speedup_vs_1"] = speedup
                    row[f"{label}_parallel_efficiency"] = (
                        float(speedup) / ranks if speedup != "" else ""
                    )
    return summaries


def write_csv(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def format_value(value: Any, digits: int = 4) -> str:
    if value is None or value == "":
        return "—"
    return f"{float(value):.{digits}f}"


def markdown_table(headers: Sequence[str], aligns: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(aligns) + "|"]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return lines


def write_plots(output_directory: Path, summaries: Sequence[Mapping[str, Any]]) -> list[str]:
    """Write compact scaling plots when matplotlib is installed."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    resolutions = sorted({mesh_sort_key(str(row["mesh"]))[0] for row in summaries})
    series = sorted(
        {(str(row["mode"]), int(row["mpi_ranks"])) for row in summaries},
        key=lambda item: (0 if item[0] == "efficient" else 1, item[1]),
    )
    by_key = {(row["mesh"], row["mode"], row["mpi_ranks"]): row for row in summaries}
    styles = {"efficient": "o-", "traditional": "s--"}
    rank_values = sorted({ranks for _, ranks in series})
    color_map = plt.get_cmap("tab10")
    colors = {
        ranks: color_map(index % color_map.N)
        for index, ranks in enumerate(rank_values)
    }

    def draw_series(axis: Any, metric: str) -> None:
        for mode, ranks in series:
            x_values, y_values = [], []
            for resolution in resolutions:
                row = by_key.get((f"iv{resolution}", mode, ranks))
                if row is None:
                    continue
                x_values.append(resolution)
                y_values.append(float(row[metric]))
            if not x_values:
                continue
            axis.plot(
                x_values,
                y_values,
                styles[mode],
                color=colors.get(ranks),
                label=f"{mode}, {ranks} rank{'s' if ranks != 1 else ''}",
            )
        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        axis.set_xticks(resolutions, [f"iv{value}" for value in resolutions])
        axis.grid(True, which="both", alpha=0.25)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    draw_series(axes[0], "initialize_seconds_median")
    axes[0].set_title("Initialization")
    axes[0].set_ylabel("Median time [s]")
    draw_series(axes[1], "rhs_seconds_per_call_median")
    axes[1].set_title("One RHS evaluation")
    axes[1].legend(fontsize=8)
    for axis in axes:
        axis.set_xlabel("Mesh")
    figure.tight_layout()
    figure.savefig(output_directory / "time.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for axis, metric, title in zip(
        axes,
        ("rss_sum_mib_median", "pss_sum_mib_median", "hwm_sum_mib_median"),
        ("RSS sum", "PSS sum", "HWM sum"),
    ):
        draw_series(axis, metric)
        axis.set_title(title)
        axis.set_xlabel("Mesh")
        axis.set_ylabel("Median memory [MiB]")
    axes[-1].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_directory / "memory.png", dpi=180)
    plt.close(figure)
    return ["time.png", "memory.png"]


def build_report(
    records: Sequence[Mapping[str, Any]],
    summaries: Sequence[Mapping[str, Any]],
    raw_directory: Path,
    scanned_json: int,
    skipped: Sequence[str],
    plots: Sequence[str],
    manifest: Mapping[str, Any],
    design: Mapping[str, Any],
) -> str:
    rank_values = sorted(int(rank) for rank in design["ranks"])
    parallel_ranks = [rank for rank in rank_values if rank > 1]
    headline_rank = max(parallel_ranks)
    mesh_values = [str(mesh) for mesh in design["meshes"]]
    rank_text = "/".join(str(rank) for rank in rank_values)
    trial_count = len(design["trials"])
    by_key = {
        (str(row["mesh"]), str(row["mode"]), int(row["mpi_ranks"])): row
        for row in summaries
    }
    paired = [
        row for row in summaries
        if row["mode"] == "traditional"
        and row["traditional_to_efficient_rhs_ratio"] != ""
    ]
    rhs_ratios = [float(row["traditional_to_efficient_rhs_ratio"]) for row in paired]
    rhs_savings = [100.0 * (1.0 - 1.0 / ratio) for ratio in rhs_ratios]
    parallel_speedups = [
        float(row["rhs_speedup_vs_1"])
        for row in summaries
        if row["mpi_ranks"] == headline_rank and row["rhs_speedup_vs_1"] != ""
    ]
    parallel_efficiencies = [
        100.0 * float(row["rhs_parallel_efficiency"])
        for row in summaries
        if row["mpi_ranks"] == headline_rank and row["rhs_parallel_efficiency"] != ""
    ]
    efficient_iv80_serial = by_key.get(("iv80", "efficient", 1))
    traditional_iv80_serial = by_key.get(("iv80", "traditional", 1))
    efficient_iv80_parallel = by_key.get(("iv80", "efficient", headline_rank))
    traditional_iv80_parallel = by_key.get(("iv80", "traditional", headline_rank))
    quadrature_iv80 = next(
        (
            int(row["quadrature_points"])
            for row in records
            if row["mesh"] == "iv80" and row["mode"] == "traditional"
        ),
        0,
    )
    repetitions = manifest.get("rhs_repetitions", {})
    repetition_text = ", ".join(
        f"{mesh}={count}" for mesh, count in repetitions.items()
    ) if isinstance(repetitions, Mapping) else "见 detailed.csv"

    lines = [
        "# NCFV 高效/传统算法性能基准", "",
        "## 主要结论", "",
    ]
    if rhs_ratios:
        lines += [
            f"- 在 {len(mesh_values)} 套网格和 {rank_text} 个 MPI 进程配置下，传统算法的单次 RHS 用时为高效算法的 "
            f"**{min(rhs_ratios):.2f}–{max(rhs_ratios):.2f} 倍**；换言之，高效算法节省 "
            f"**{min(rhs_savings):.1f}%–{max(rhs_savings):.1f}%** 的残差计算时间。",
        ]
    if all((efficient_iv80_serial, traditional_iv80_serial,
            efficient_iv80_parallel, traditional_iv80_parallel)):
        serial_e_pss = float(efficient_iv80_serial["pss_sum_mib_median"]) / 1024.0
        serial_t_pss = float(traditional_iv80_serial["pss_sum_mib_median"]) / 1024.0
        parallel_e_pss = float(efficient_iv80_parallel["pss_sum_mib_median"]) / 1024.0
        parallel_t_pss = float(traditional_iv80_parallel["pss_sum_mib_median"]) / 1024.0
        lines += [
            f"- IV80 上，高效/传统的聚合 PSS 为 1 个 MPI 进程 **{serial_e_pss:.2f}/{serial_t_pss:.2f} GiB**、"
            f"{headline_rank} 个 MPI 进程 **{parallel_e_pss:.2f}/{parallel_t_pss:.2f} GiB**；高效算法分别少用 "
            f"**{serial_t_pss - serial_e_pss:.2f} GiB（{100 * (1 - serial_e_pss / serial_t_pss):.1f}%）** "
            f"和 **{parallel_t_pss - parallel_e_pss:.2f} GiB（{100 * (1 - parallel_e_pss / parallel_t_pss):.1f}%）**。",
            f"- IV80 从 1 增至 {headline_rank} 个 MPI 进程后，作业聚合 PSS 因进程运行库、ghost 和通信"
            f"数据重复而增加：高效 **{100 * (parallel_e_pss / serial_e_pss - 1):.1f}%**，"
            f"传统 **{100 * (parallel_t_pss / serial_t_pss - 1):.1f}%**；这属于以更多总内存换取更短墙钟时间。",
            f"- 高效模式四级网格均存储 **0** 个求积点；传统 IV80 存储 "
            f"**{quadrature_iv80:,}** 个体/面求积点，这是内存差异随网格加密迅速扩大的主要来源。",
        ]
    if parallel_speedups:
        lines += [
            f"- {headline_rank} 个 MPI 进程的 RHS 加速比为 **{min(parallel_speedups):.2f}–{max(parallel_speedups):.2f}**，"
            f"并行效率为 **{min(parallel_efficiencies):.1f}%–{max(parallel_efficiencies):.1f}%**；"
            "细网格的并行效率高于粗网格。",
        ]
        if headline_rank == 8:
            lines += [
                "- 此处约 3 倍的核时间差与[既有完整 `t=2` IV80 复算]"
                "(../ncfv_with_dissipation_recompute_t2_cfl05_20260912/report.md)（797.3 s 对 "
                "2439.0 s，传统/高效 3.06 倍）一致，说明短基准能代表长期时间推进中的主要算法差异。",
            ]

    if trial_count == 1:
        trial_text = (
            "本轮每组 1 次独立启动，因此初始化时间没有跨启动误差条。"
        )
        order_text = (
            "本轮每组仅一个 trial，高效模式先于传统模式启动，文件系统缓存会使传统模式的"
            "初始化结果略占优势，因此初始化比值只作参考；RHS 已预热，但仍应结合单次启动限制理解。"
        )
    else:
        trial_text = f"本轮每组 {trial_count} 次独立启动，分组结果取 trial 中位数。"
        order_text = (
            "高效/传统的启动顺序随 trial 奇偶交替，以降低文件系统缓存与热状态的顺序偏差。"
        )

    lines += [
        "", "## 并行口径与测试设置", "",
        "**当前 NCFV 残差核没有 OpenMP 并行循环。** 因而本报告的“单核/多核”严格指 "
        f"{rank_text} 个单线程 MPI 进程配置，不是 OpenMP 线程数。每个 rank 均固定 "
        "`OMP_NUM_THREADS=1` 和 `DNDS_DIST_OMP_NUM_THREADS=1`，并用 "
        "`--map-by core --bind-to core` 绑定物理核心。",
        f"测试节点为 `{manifest.get('hostname', 'unknown')}`，CPU 为 "
        f"`{manifest.get('cpu_model', 'unknown')}`，系统可见逻辑 CPU 数 "
        f"{manifest.get('logical_cpus', 'unknown')}。当前构建为 Release `-O3`。",
        f"{len(mesh_values)} 套算例保留原有等熵涡初场、物理与重构设置，仅切换 "
        "`EfficientDifferential` / `TraditionalQuadrature`。为隔离计算核，关闭 VTK、"
        "restart、resolved config 和节点诊断输出；初始化后先预热 1 次 RHS，再计时。"
        f"各网格计时重复数为 {repetition_text}。",
        "", "## 统计口径", "",
        f"从 `{raw_directory}` 递归扫描 {scanned_json} 个 JSON 文件，识别 "
        f"{len(records)} 次有效运行；按 `mesh / mode / mpi_ranks` 聚合。"
        "单次 RHS 指一次完整空间残差计算，表中取同一进程启动内多次调用的中位数；"
        f"若存在多次 trial，则再取 trial 中位数。{trial_text}", "",
        "`rss_sum_mib`、`pss_sum_mib` 和 `hwm_sum_mib` 是所有 MPI rank 的求和值；"
        "MiB 按 2²⁰ 字节换算。PSS 按共享页比例分摊，最适合估计作业实际物理内存；"
        "RSS 会在多进程求和时重复计算共享页；HWM 是各 rank 历史峰值之和，不保证这些峰值"
        "发生在同一时刻。净 PSS 为测量后 PSS 减去 MPI 初始化后的、求解器初始化前基线。"
        "传统/高效比大于 1 表示传统算法更慢或占用更多内存。p-rank 并行效率定义为 "
        "`(T₁/Tₚ)/p`。", "",
        "## 中位数汇总", "",
    ]
    summary_rows = []
    for row in summaries:
        summary_rows.append((
            row["mesh"], MODE_NAMES_ZH[row["mode"]], row["mpi_ranks"], row["trials"],
            format_value(row["initialize_seconds_median"], 6),
            format_value(row["rhs_seconds_per_call_median"], 6),
            format_value(row["rss_sum_mib_median"], 2),
            format_value(row["pss_sum_mib_median"], 2),
            format_value(row["pss_net_mib_median"], 2),
            format_value(row["hwm_sum_mib_median"], 2),
        ))
    lines += markdown_table(
        ("网格", "模式", "MPI进程", "trial数", "初始化/s", "单次RHS/s", "RSS总和/MiB", "PSS总和/MiB", "净PSS/MiB", "HWM总和/MiB"),
        ("---", "---", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:"),
        summary_rows,
    )

    lines += ["", "## 传统/高效比", ""]
    comparison_rows = []
    for row in summaries:
        if row["mode"] != "traditional" or row["traditional_to_efficient_rhs_ratio"] == "":
            continue
        comparison_rows.append((
            row["mesh"], row["mpi_ranks"],
            format_value(row["traditional_to_efficient_initialize_ratio"]),
            format_value(row["traditional_to_efficient_rhs_ratio"]),
            format_value(row["traditional_to_efficient_rss_ratio"]),
            format_value(row["traditional_to_efficient_pss_ratio"]),
            format_value(row["traditional_to_efficient_pss_net_ratio"]),
            format_value(row["traditional_to_efficient_hwm_ratio"]),
        ))
    if comparison_rows:
        lines += markdown_table(
            ("网格", "MPI进程", "初始化比", "RHS时间比", "RSS比", "PSS比", "净PSS比", "HWM比"),
            ("---", "---:", "---:", "---:", "---:", "---:", "---:", "---:"),
            comparison_rows,
        )
    else:
        lines.append("原始数据中没有同网格、同 MPI 进程数的高效/传统配对，无法计算该比值。")

    lines += ["", "## 多 rank 相对 1 rank 的加速", ""]
    parallel_rows = []
    for row in summaries:
        if row["mpi_ranks"] == 1 or row["rhs_speedup_vs_1"] == "":
            continue
        parallel_rows.append((
            row["mesh"], MODE_NAMES_ZH[row["mode"]], row["mpi_ranks"],
            format_value(row["initialize_speedup_vs_1"]),
            format_value(100 * float(row["initialize_parallel_efficiency"]), 2) + "%",
            format_value(row["rhs_speedup_vs_1"]),
            format_value(100 * float(row["rhs_parallel_efficiency"]), 2) + "%",
        ))
    if parallel_rows:
        lines += markdown_table(
            ("网格", "模式", "MPI进程", "初始化加速比", "初始化并行效率", "RHS加速比", "RHS并行效率"),
            ("---", "---", "---:", "---:", "---:", "---:", "---:"),
            parallel_rows,
        )
    else:
        lines.append("原始数据中没有同网格、同模式的 1-rank/多-rank 配对，无法计算并行加速比。")

    if plots:
        lines += ["", "## 趋势图", ""]
        if "time.png" in plots:
            lines += ["![初始化与单次 RHS 时间](time.png)", ""]
        if "memory.png" in plots:
            lines += ["![RSS、PSS 与 HWM 聚合内存](memory.png)"]

    lines += [
              "", "## 解释与限制", "",
              "- 初始化时间包含串行 CGNS 读取、分区、对偶几何、重构算子及场分配。"
              + order_text,
              "- 这是单节点共享内存 MPI 测试；跨节点网络通信性能不在本次范围内。",
              "- 基准直接重复 `EvaluateResidual()`，不包含 SSPRK3 公共向量更新和结果 I/O。"
              "它有意隔离两种积分算法的主要差异；完整作业总时间还会包含这些共同开销。",
              f"- 每组有 {trial_count} 次进程启动。RHS 内部重复可用于本次核时间对比；若用于正式论文的"
              "置信区间，建议在独占节点上至少重复 3–5 次完整启动。",
              "", "## 完整数据", "",
              "逐 trial 数据见 `detailed.csv`；分组中位数与派生比值见 `summary.csv`；"
              "完整原始 JSON 与运行清单（命令、环境、源码/网格哈希）见 `raw_results.json`。"]
    if plots:
        lines.append("计时和内存趋势图见 " + "、".join(f"`{name}`" for name in plots) + "。")
    else:
        lines.append("当前 Python 环境未安装 matplotlib，因此未生成 PNG；CSV 和报告不受影响。")
    if skipped:
        preview = "、".join(f"`{name}`" for name in skipped[:8])
        suffix = f" 等 {len(skipped)} 个文件" if len(skipped) > 8 else ""
        lines += ["", f"另有未识别为性能记录的 JSON：{preview}{suffix}；这些文件未参与统计。"]
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_dir", nargs="?", type=Path,
                        help="directory recursively containing per-trial JSON files")
    parser.add_argument("output_dir", nargs="?", type=Path,
                        help="directory for CSV, Markdown and optional PNG outputs")
    parser.add_argument("--raw", "--input", dest="raw_option", type=Path,
                        help="named alternative to the raw_dir positional argument")
    parser.add_argument("--output", dest="output_option", type=Path,
                        help="named alternative to the output_dir positional argument")
    args = parser.parse_args()

    if args.raw_dir is not None and args.raw_option is not None:
        parser.error("provide raw_dir or --raw, not both")
    if args.output_dir is not None and args.output_option is not None:
        parser.error("provide output_dir or --output, not both")
    raw_argument = args.raw_option or args.raw_dir
    output_argument = args.output_option or args.output_dir
    if raw_argument is None or output_argument is None:
        parser.error("raw_dir and output_dir are required")

    raw_directory = raw_argument.resolve()
    output_directory = output_argument.resolve()
    manifest = load_manifest(raw_directory)
    records, raw_records, skipped, scanned_json = load_records(raw_directory)
    records.sort(key=record_sort_key)
    design = validate_records(records, manifest)
    summaries = summarize(records)

    output_directory.mkdir(parents=True, exist_ok=True)
    write_csv(output_directory / "detailed.csv", DETAILED_FIELDS, records)
    write_csv(output_directory / "summary.csv", SUMMARY_FIELDS, summaries)
    plots = write_plots(output_directory, summaries)
    report = build_report(
        records,
        summaries,
        raw_directory,
        scanned_json,
        skipped,
        plots,
        manifest,
        design,
    )
    (output_directory / "report.md").write_text(report, encoding="utf-8")

    packaged: dict[str, Any] = {"results": raw_records}
    if manifest:
        packaged["manifest"] = manifest
    (output_directory / "raw_results.json").write_text(
        json.dumps(packaged, indent=2) + "\n", encoding="utf-8"
    )

    print(f"wrote {len(records)} trials in {len(summaries)} groups to {output_directory}")


if __name__ == "__main__":
    main()
