#!/usr/bin/env python3
"""Run reproducible NCFV timing and process-memory benchmarks by mesh family.

The C++ performance probe writes RSS, PSS, and HWM snapshots before
initialization, after initialization, after one warm-up RHS, and after the
timed RHS loop.  This runner keeps every MPI rank single-threaded and stores
one JSON result plus its launcher log for every (family, mesh, mode, rank)
condition.  It never overwrites a completed result; use ``--resume`` after an
interrupted matrix run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[3]
BUILD = ROOT / "build"
SIZES = (10, 20, 40, 80)
MODES = ("EfficientDifferential", "TraditionalQuadrature")
FAMILIES = ("original", "hex", "tet")
DEFAULT_REPETITIONS = {10: 200, 20: 80, 40: 20, 80: 3}


def config_path(family: str, size: int) -> Path:
    if family == "original":
        return ROOT / "cases/NCFV" / f"NCFV_iv{size}.json"
    return ROOT / "cases/NCFV" / f"NCFV_periodic_{family}_iv{size}.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def parse_repetitions(value: str) -> dict[int, int]:
    repetitions = DEFAULT_REPETITIONS.copy()
    if not value:
        return repetitions
    for assignment in value.split(","):
        key, raw_count = assignment.split("=", 1)
        size = int(key.removeprefix("iv"))
        count = int(raw_count)
        if size not in SIZES:
            raise argparse.ArgumentTypeError(f"unknown mesh size {key!r}")
        if count <= 0:
            raise argparse.ArgumentTypeError("repetition counts must be positive")
        repetitions[size] = count
    return repetitions


def validate_result(path: Path, *, configuration: str, mode: str, ranks: int,
                    repetitions: int) -> None:
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exception:
        raise ValueError(f"cannot read existing result {path}: {exception}") from exception
    expected = {
        "schema_version": 1,
        "source_configuration": configuration,
        "mode": mode,
        "mpi_ranks": ranks,
        "repetitions": repetitions,
    }
    mismatches = {key: (result.get(key), value) for key, value in expected.items()
                  if result.get(key) != value}
    if mismatches:
        detail = ", ".join(f"{key}={actual!r}, expected {wanted!r}"
                           for key, (actual, wanted) in mismatches.items())
        raise ValueError(f"existing result {path} does not match requested job: {detail}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--families", choices=FAMILIES, nargs="+", default=["hex", "tet"])
    parser.add_argument("--sizes", type=int, choices=SIZES, nargs="+", default=list(SIZES))
    parser.add_argument("--ranks", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--repetitions", default="",
                        help="override defaults, e.g. iv10=200,iv80=2")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.trials <= 0 or any(rank <= 0 for rank in args.ranks):
        parser.error("trials and MPI ranks must be positive")
    if len(set(args.ranks)) != len(args.ranks):
        parser.error("MPI ranks must be unique")
    if len(set(args.families)) != len(args.families):
        parser.error("mesh families must be unique")
    if len(set(args.sizes)) != len(args.sizes):
        parser.error("mesh sizes must be unique")
    repetitions = parse_repetitions(args.repetitions)
    probe = args.probe.resolve()
    output_root = args.output_root.resolve()
    if not probe.is_file():
        parser.error(f"probe does not exist: {probe}")
    if output_root.exists() and not args.resume:
        parser.error(f"output root exists: {output_root}; use --resume to continue it")
    output_root.mkdir(parents=True, exist_ok=True)

    environment = os.environ.copy()
    environment.update({
        "OMP_NUM_THREADS": "1",
        "DNDS_DIST_OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OMP_PROC_BIND": "true",
        "OMP_PLACES": "cores",
    })
    configurations = [config_path(family, size) for family in args.families for size in args.sizes]
    missing = [str(path) for path in configurations if not path.is_file()]
    if missing:
        parser.error("missing configurations: " + ", ".join(missing))
    manifest_path = output_root / "manifest.json"
    manifest = {
        "schema_version": 1,
        "started_unix_seconds": time.time(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "cpu_model": cpu_model(),
        "logical_cpus": os.cpu_count(),
        "probe": str(probe),
        "probe_sha256": sha256(probe),
        "families": args.families,
        "sizes": args.sizes,
        "ranks": args.ranks,
        "trials": args.trials,
        "modes": list(MODES),
        "rhs_repetitions": {f"iv{size}": repetitions[size] for size in args.sizes},
        "thread_environment": {key: environment[key] for key in (
            "OMP_NUM_THREADS", "DNDS_DIST_OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS", "OMP_PROC_BIND", "OMP_PLACES")},
        "config_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in configurations},
        "mesh_sha256": {},
        "jobs": [],
    }
    for path in configurations:
        mesh = json.loads(path.read_text(encoding="utf-8"))["mesh"]["meshFile"]
        mesh_path = (BUILD / mesh).resolve()
        if not mesh_path.is_file():
            parser.error(f"mesh named by {path} does not exist: {mesh_path}")
        manifest["mesh_sha256"][str(mesh_path.relative_to(ROOT))] = sha256(mesh_path)

    if args.resume and manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key in ("schema_version", "probe_sha256", "families", "sizes", "ranks", "trials",
                    "modes", "rhs_repetitions", "thread_environment", "config_sha256", "mesh_sha256"):
            if previous.get(key) != manifest[key]:
                parser.error(f"cannot resume because manifest field {key!r} differs")
        manifest = previous
        manifest.setdefault("resumed_unix_seconds", []).append(time.time())
        manifest.pop("finished_unix_seconds", None)

    for family in args.families:
        for size in args.sizes:
            configuration = config_path(family, size)
            source_configuration = os.path.relpath(configuration, BUILD)
            for trial in range(1, args.trials + 1):
                modes = MODES if trial % 2 else tuple(reversed(MODES))
                for ranks in args.ranks:
                    for mode in modes:
                        stem = f"{family}_iv{size}_{mode}_np{ranks}_trial{trial:02d}"
                        result = output_root / f"{stem}.json"
                        log = output_root / f"{stem}.log"
                        if result.exists():
                            if not args.resume:
                                parser.error(f"result unexpectedly exists: {result}")
                            validate_result(result, configuration=source_configuration, mode=mode,
                                            ranks=ranks, repetitions=repetitions[size])
                            print(f"SKIP {stem}", flush=True)
                            continue
                        command = [
                            "mpirun", "--mca", "pml", "ob1", "--mca", "btl", "self,vader",
                            "--bind-to", "core", "--map-by", "core", "-np", str(ranks),
                            str(probe), source_configuration, mode, str(repetitions[size]), str(result),
                        ]
                        job = {
                            "family": family, "size": size, "mode": mode, "mpi_ranks": ranks,
                            "trial": trial, "result": str(result), "log": str(log), "command": command,
                            "status": "running", "started_unix_seconds": time.time(),
                        }
                        manifest["jobs"].append(job)
                        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
                        print(f"RUN  {stem}", flush=True)
                        started = time.perf_counter()
                        with log.open("w", encoding="utf-8") as stream:
                            stream.write("command: " + shlex.join(command) + "\n")
                            stream.write("environment: " + json.dumps(manifest["thread_environment"], sort_keys=True) + "\n")
                            stream.flush()
                            completed = subprocess.run(command, cwd=BUILD, env=environment,
                                                       stdout=stream, stderr=subprocess.STDOUT,
                                                       check=False)
                        job["launcher_wall_seconds"] = time.perf_counter() - started
                        job["returncode"] = completed.returncode
                        job["status"] = "complete" if completed.returncode == 0 else "failed"
                        job["finished_unix_seconds"] = time.time()
                        if completed.returncode == 0:
                            job["result_sha256"] = sha256(result)
                        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
                        if completed.returncode:
                            print(f"FAIL {stem}; inspect {log}", file=sys.stderr, flush=True)
                            return completed.returncode
                        print(f"DONE {stem} ({job['launcher_wall_seconds']:.3f} s)", flush=True)
    manifest["finished_unix_seconds"] = time.time()
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
