"""Run paired NCFV algorithm performance measurements on the IV mesh family.

The NCFV residual kernel is MPI parallel, not OpenMP parallel.  Consequently
this runner keeps every rank single-threaded and compares one MPI rank with a
fixed multi-rank configuration.  The companion C++ probe disables solver I/O,
warms the residual kernel once, and records phase timings and per-rank memory.
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
MESHES = ("iv10", "iv20", "iv40", "iv80")
MODES = ("EfficientDifferential", "TraditionalQuadrature")
DEFAULT_REPETITIONS = {
    "iv10": 200,
    "iv20": 80,
    "iv40": 20,
    "iv80": 3,
}


def parse_repetitions(value: str) -> dict[str, int]:
    repetitions = DEFAULT_REPETITIONS.copy()
    if not value:
        return repetitions
    for assignment in value.split(","):
        mesh, count = assignment.split("=", 1)
        if mesh not in MESHES:
            raise argparse.ArgumentTypeError(f"unknown mesh {mesh!r}")
        repetitions[mesh] = int(count)
        if repetitions[mesh] <= 0:
            raise argparse.ArgumentTypeError("repetition counts must be positive")
    return repetitions


def cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=ROOT, text=True
    ).strip()


def validate_existing_result(
    path: Path,
    *,
    mesh: str,
    mode: str,
    ranks: int,
    repetitions: int,
    source_configuration: str,
) -> None:
    """Reject stale or unrelated output before treating it as resumed work."""
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exception:
        raise ValueError(f"cannot read existing result {path}: {exception}") from exception
    if not isinstance(document, dict):
        raise ValueError(f"existing result {path} is not a JSON object")
    expected = {
        "schema_version": 1,
        "mesh": mesh,
        "mode": mode,
        "mpi_ranks": ranks,
        "repetitions": repetitions,
        "source_configuration": source_configuration,
    }
    mismatches = {
        key: (document.get(key), value)
        for key, value in expected.items()
        if document.get(key) != value
    }
    if mismatches:
        details = ", ".join(
            f"{key}={actual!r} (expected {wanted!r})"
            for key, (actual, wanted) in mismatches.items()
        )
        raise ValueError(f"existing result {path} does not match this job: {details}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--ranks", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument(
        "--repetitions",
        default="",
        help="comma-separated overrides such as iv10=200,iv80=3",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.trials <= 0 or any(rank <= 0 for rank in args.ranks):
        parser.error("trials and ranks must be positive")
    if len(set(args.ranks)) != len(args.ranks):
        parser.error("ranks must not contain duplicates")
    repetitions = parse_repetitions(args.repetitions)
    probe = args.probe.resolve()
    output_root = args.output_root.resolve()
    if not probe.is_file():
        parser.error(f"probe does not exist: {probe}")
    if output_root.exists() and not args.resume:
        parser.error(f"output root already exists: {output_root}; use --resume")
    output_root.mkdir(parents=True, exist_ok=True)

    environment = os.environ.copy()
    environment.update(
        {
            "OMP_NUM_THREADS": "1",
            "DNDS_DIST_OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OMP_PROC_BIND": "true",
            "OMP_PLACES": "cores",
        }
    )

    fresh_manifest = {
        "schema_version": 1,
        "started_unix_seconds": time.time(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "cpu_model": cpu_model(),
        "logical_cpus": os.cpu_count(),
        "probe": str(probe),
        "probe_sha256": sha256(probe),
        "build_directory": str(BUILD),
        "git_head": git_output("rev-parse", "HEAD"),
        "git_status_short": git_output("status", "--short"),
        "artifact_sha256": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in (
                ROOT / "build" / "src" / "NCFV" / "libncfv.a",
                ROOT / "src" / "NCFV" / "NCFVSolver.cpp",
                ROOT / "src" / "NCFV" / "NCFVSpatial.cpp",
                ROOT / "src" / "NCFV" / "NCFVDualGeometry.cpp",
                ROOT / "src" / "NCFV" / "NCFVReconstruction.cpp",
                *(ROOT / "cases" / "NCFV" / f"NCFV_{mesh}.json" for mesh in MESHES),
                *(ROOT / "cases" / "NCFV" / f"{mesh}_3d_1_dnds.cgns" for mesh in MESHES),
            )
        },
        "ranks": args.ranks,
        "trials": args.trials,
        "rhs_repetitions": repetitions,
        "modes": list(MODES),
        "meshes": list(MESHES),
        "thread_environment": {
            key: environment[key]
            for key in (
                "OMP_NUM_THREADS",
                "DNDS_DIST_OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OMP_PROC_BIND",
                "OMP_PLACES",
            )
        },
        "parallelism_note": (
            "NCFV has no OpenMP-parallel residual loops; multi-core runs use "
            "multiple single-threaded MPI ranks."
        ),
        "jobs": [],
    }
    manifest_path = output_root / "manifest.json"
    if args.resume and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key in (
            "schema_version",
            "hostname",
            "cpu_model",
            "probe_sha256",
            "git_head",
            "artifact_sha256",
            "rhs_repetitions",
            "modes",
            "meshes",
            "trials",
            "thread_environment",
        ):
            if manifest.get(key) != fresh_manifest[key]:
                parser.error(
                    f"cannot resume because {key} differs from the existing manifest"
                )
        try:
            existing_ranks = sorted(int(rank) for rank in manifest.get("ranks", []))
        except (TypeError, ValueError):
            parser.error("cannot resume because the existing manifest has invalid ranks")
        if existing_ranks != sorted(args.ranks):
            parser.error("cannot resume because ranks differ from the existing manifest")
        if not isinstance(manifest.get("jobs"), list):
            parser.error("cannot resume because the existing manifest has no jobs list")
        manifest.setdefault("resumed_unix_seconds", []).append(time.time())
        manifest.pop("finished_unix_seconds", None)
    else:
        manifest = fresh_manifest

    for mesh in MESHES:
        config = ROOT / "cases" / "NCFV" / f"NCFV_{mesh}.json"
        for trial in range(1, args.trials + 1):
            for ranks in args.ranks:
                # Alternate the pair order between trials to reduce thermal/order bias.
                modes = MODES if trial % 2 else tuple(reversed(MODES))
                for mode in modes:
                    stem = f"{mesh}_{mode}_np{ranks}_trial{trial:02d}"
                    result = output_root / f"{stem}.json"
                    log = output_root / f"{stem}.log"
                    if result.exists() and args.resume:
                        source_configuration = os.path.relpath(config, BUILD)
                        try:
                            validate_existing_result(
                                result,
                                mesh=mesh,
                                mode=mode,
                                ranks=ranks,
                                repetitions=repetitions[mesh],
                                source_configuration=source_configuration,
                            )
                        except ValueError as exception:
                            parser.error(str(exception))
                        completed_jobs = [
                            job
                            for job in manifest["jobs"]
                            if job.get("mesh") == mesh
                            and job.get("mode") == mode
                            and job.get("mpi_ranks") == ranks
                            and job.get("trial") == trial
                            and Path(str(job.get("result", ""))).resolve() == result
                            and job.get("status") == "complete"
                            and job.get("returncode") == 0
                        ]
                        if not completed_jobs:
                            parser.error(
                                f"cannot resume {result.name}: no matching completed job in manifest"
                            )
                        result_hash = sha256(result)
                        recorded_hash = completed_jobs[-1].get("result_sha256")
                        if recorded_hash is not None and recorded_hash != result_hash:
                            parser.error(
                                f"cannot resume {result.name}: result hash differs from manifest"
                            )
                        completed_jobs[-1]["result_sha256"] = result_hash
                        print(f"SKIP {result.name}", flush=True)
                        continue
                    command = [
                        "mpirun",
                        "--mca",
                        "pml",
                        "ob1",
                        "--mca",
                        "btl",
                        "self,vader",
                        "--bind-to",
                        "core",
                        "--map-by",
                        "core",
                        "-np",
                        str(ranks),
                        str(probe),
                        os.path.relpath(config, BUILD),
                        mode,
                        str(repetitions[mesh]),
                        str(result),
                    ]
                    job = {
                        "mesh": mesh,
                        "mode": mode,
                        "mpi_ranks": ranks,
                        "trial": trial,
                        "result": str(result),
                        "log": str(log),
                        "command": command,
                        "status": "running",
                        "started_unix_seconds": time.time(),
                    }
                    manifest["jobs"].append(job)
                    manifest_path.write_text(
                        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
                    )
                    print(f"RUN  {stem}", flush=True)
                    started = time.perf_counter()
                    with log.open("w", encoding="utf-8") as stream:
                        stream.write("command: " + shlex.join(command) + "\n")
                        stream.write(
                            "environment: "
                            + json.dumps(manifest["thread_environment"], sort_keys=True)
                            + "\n"
                        )
                        stream.flush()
                        completed = subprocess.run(
                            command,
                            cwd=BUILD,
                            env=environment,
                            stdout=stream,
                            stderr=subprocess.STDOUT,
                            check=False,
                        )
                    job["launcher_wall_seconds"] = time.perf_counter() - started
                    job["returncode"] = completed.returncode
                    job["status"] = "complete" if completed.returncode == 0 else "failed"
                    job["finished_unix_seconds"] = time.time()
                    if completed.returncode == 0:
                        job["result_sha256"] = sha256(result)
                    manifest_path.write_text(
                        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
                    )
                    if completed.returncode != 0:
                        print(f"FAIL {stem}; see {log}", file=sys.stderr, flush=True)
                        return completed.returncode
                    print(
                        f"DONE {stem} ({job['launcher_wall_seconds']:.3f} s)",
                        flush=True,
                    )

    manifest["finished_unix_seconds"] = time.time()
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
