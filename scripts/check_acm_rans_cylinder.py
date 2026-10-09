#!/usr/bin/env python3
"""Run short ACM cylinder smoke tests for all steady RANS and physical BDF2 modes.

The default Re=20 2-D case checks solver integration and field output. It is
not a turbulence-model validation case. Use --case with a suitable 3-D case
and a longer run for physical comparisons.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "Laminar": ([], None),
    "SpalartAllmaras": (["TurbulenceNuTilde"], [1e-6, 1.0]),
    "KOmegaWilcox": (["TurbulenceK", "TurbulenceOmega"], [0.001, 10.0]),
    "KOmegaSST": (["TurbulenceK", "TurbulenceOmega"], [0.001, 10.0]),
    "RealizableKEpsilon": (["TurbulenceK", "TurbulenceEpsilon"], [0.001, 0.0009]),
}
STEP_PATTERN = re.compile(
    r"ACM (?:physical )?step\s+(\d+)(?: time=[\d.eE+-]+ BDF[12])? residual\s+"
    r"([\d.eE+-]+)\s+->\s+([\d.eE+-]+)"
    r"\s+dtauMin=([\d.eE+-]+)\s+turbResidual=([\d.eE+-]+)"
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, default=ROOT / "cases/acm2D/acm2D.json")
    parser.add_argument("--mesh", type=Path, help="Override the case mesh without rewriting its JSON")
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument(
        "--integrator", choices=(
            "ExplicitSSPRK3", "ImplicitEulerBlockJacobi", "ImplicitEulerGMRES", "ImplicitEulerLUSGS",
            "BDF2DualTimeGMRES", "BDF2DualTimeLUSGS",
        ),
        default="ExplicitSSPRK3",
    )
    parser.add_argument("--physical-time-step", type=float)
    parser.add_argument("--max-implicit-iterations", type=int)
    parser.add_argument("--reconstruction", choices=("FirstOrder", "GreenGauss", "Variational"))
    parser.add_argument("--limiter", choices=("LocalExtrema", "WBAP", "CWBAP"))
    parser.add_argument("--order", type=int, choices=(0, 1, 2, 3), help="CFV polynomial degree")
    parser.add_argument("--disable-limiter", action="store_true")
    parser.add_argument("--turbulence-second-order", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--reconstruction-tolerance", type=float)
    parser.add_argument("--reconstruction-use-gmres", action="store_true")
    parser.add_argument("--cfl", type=float, default=0.2)
    parser.add_argument("--viscosity", type=float, help="Override molecular viscosity")
    parser.add_argument("--np", type=int, default=1, help="MPI ranks; 1 runs the executable directly")
    parser.add_argument("--omp-threads", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=1200, help="Seconds per model")
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument(
        "--output", type=Path,
        default=Path("/tmp") / f"acm-rans-cylinder-{datetime.now():%Y%m%d-%H%M%S}",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running")
    args = parser.parse_args()
    if args.steps < 1 or args.np < 1 or args.omp_threads < 1 or args.timeout < 1:
        parser.error("steps, np, omp-threads, and timeout must be positive")
    if not math.isfinite(args.cfl) or args.cfl <= 0:
        parser.error("cfl must be finite and positive")
    if args.viscosity is not None and (not math.isfinite(args.viscosity) or args.viscosity <= 0):
        parser.error("viscosity must be finite and positive")
    if args.physical_time_step is not None and (
        not math.isfinite(args.physical_time_step) or args.physical_time_step <= 0
    ):
        parser.error("physical-time-step must be finite and positive")
    if args.max_implicit_iterations is not None and args.max_implicit_iterations < 1:
        parser.error("max-implicit-iterations must be positive")
    if args.reconstruction_tolerance is not None and (
        not math.isfinite(args.reconstruction_tolerance) or args.reconstruction_tolerance < 0
    ):
        parser.error("reconstruction-tolerance must be finite and non-negative")
    return args


def command(args: argparse.Namespace, model: str, model_dir: Path) -> list[str]:
    executable = (args.build_dir / "app/euler.exe").resolve()
    invocation = [str(executable), str(args.case.resolve())]
    overrides = {
        "/turbulenceSettings/model": model,
        "/timeMarchSettings/integrator": args.integrator,
        "/timeMarchSettings/nSteps": args.steps,
        "/timeMarchSettings/cfl": args.cfl,
        "/outputSettings/interval": args.steps,
        "/outputSettings/writeInitial": False,
        "/outputSettings/directory": str(model_dir),
        "/outputSettings/prefix": model,
    }
    if MODELS[model][1] is not None:
        overrides["/turbulenceSettings/initialValue"] = MODELS[model][1]
        overrides["/turbulenceSettings/farFieldValue"] = MODELS[model][1]
        overrides["/acmSettings/enableViscousFlux"] = True
    if args.mesh is not None:
        overrides["/meshSettings/meshFile"] = str(args.mesh.resolve())
    if args.viscosity is not None:
        overrides["/acmSettings/dynamicViscosity"] = args.viscosity
    if args.physical_time_step is not None:
        overrides["/timeMarchSettings/physicalTimeStep"] = args.physical_time_step
    if args.max_implicit_iterations is not None:
        overrides["/timeMarchSettings/maxImplicitIterations"] = args.max_implicit_iterations
    if args.reconstruction is not None:
        overrides["/reconstructionSettings/type"] = args.reconstruction
    if args.limiter is not None:
        overrides["/reconstructionSettings/limiterType"] = args.limiter
    if args.order is not None:
        overrides["/vfvSettings/maxOrder"] = args.order
    if args.disable_limiter:
        overrides["/reconstructionSettings/enableLimiter"] = False
    if args.turbulence_second_order is not None:
        overrides["/turbulenceSettings/secondOrderReconstruction"] = args.turbulence_second_order
    if args.reconstruction_tolerance is not None:
        overrides["/reconstructionSettings/variationalTolerance"] = args.reconstruction_tolerance
    if args.reconstruction_use_gmres:
        overrides["/reconstructionSettings/variationalUseGMRES"] = True
    for key, value in overrides.items():
        invocation.extend(["-k", key, "-v", json.dumps(value)])
    if args.np > 1:
        return ["mpirun", "-np", str(args.np), *invocation]
    return invocation


def inspect_output(path: Path, expected_fields: list[str]) -> dict[str, object]:
    stats: dict[str, object] = {}
    with h5py.File(path, "r") as handle:
        cell_data = handle["VTKHDF/CellData"]
        required = {"Velocity", "Pressure", *expected_fields}
        actual = set(cell_data.keys())
        if actual != required:
            raise ValueError(f"CellData fields {sorted(actual)} differ from {sorted(required)}")
        for field in sorted(required):
            values = np.asarray(cell_data[field][:], dtype=float)
            if not values.size or not np.isfinite(values).all():
                raise ValueError(f"{field} has empty or non-finite data")
            if field in expected_fields and np.min(values) <= 0:
                raise ValueError(f"{field} has a non-positive transported value")
            stats[field] = {"min": float(np.min(values)), "max": float(np.max(values))}
    return stats


def run_one(args: argparse.Namespace, model: str) -> dict[str, object]:
    model_dir = args.output / model
    model_dir.mkdir(parents=True)
    invocation = command(args, model, model_dir)
    log_path = model_dir / "run.log"
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = str(args.omp_threads)
    # hwloc's GL plugin probes X11 displays and can hang on headless compute nodes.
    environment.setdefault("HWLOC_COMPONENTS", "-gl")
    result: dict[str, object] = {"model": model, "command": invocation, "log": str(log_path)}
    try:
        with log_path.open("w") as log:
            process = subprocess.run(
                invocation, cwd=args.build_dir, env=environment,
                stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout, check=False,
            )
        result["exit_code"] = process.returncode
        log_text = log_path.read_text(errors="replace")
        if process.returncode != 0:
            lines = log_text.splitlines()
            excerpt = " | ".join([*lines[:3], *lines[-3:]])
            raise ValueError(f"solver exited with code {process.returncode}: {excerpt}")
        steps = [match.groups() for match in STEP_PATTERN.finditer(log_text)]
        if not steps or int(steps[-1][0]) != args.steps:
            raise ValueError(f"expected step {args.steps}, found {steps[-1][0] if steps else 'none'}")
        if args.integrator.startswith("BDF2"):
            orders = [int(value) for value in re.findall(
                r"ACM physical step\s+\d+ time=[\d.eE+-]+ BDF([12]) residual", log_text
            )]
            expected_orders = [1, *([2] * (args.steps - 1))]
            if orders != expected_orders:
                raise ValueError(f"BDF order sequence {orders} differs from {expected_orders}")
            result["bdf_orders"] = orders
            convergence = [bool(int(value)) for value in re.findall(
                r"ACM physical step[^\n]*\bconverged=([01])", log_text
            )]
            if len(convergence) != args.steps:
                raise ValueError("BDF physical-step convergence flags are missing")
            result["inner_converged"] = convergence
        last = steps[-1]
        residuals = [float(value) for value in last[1:]]
        if not all(math.isfinite(value) for value in residuals):
            raise ValueError("non-finite residual or pseudo-time step")
        result["last_step"] = int(last[0])
        result["flow_residual_before"] = residuals[0]
        result["flow_residual_after"] = residuals[1]
        result["minimum_pseudo_time_step"] = residuals[2]
        result["turbulence_residual"] = residuals[3]
        output = model_dir / f"{model}_{args.steps:08d}.vtkhdf"
        result["output"] = str(output)
        result["fields"] = inspect_output(output, MODELS[model][0])
        result["status"] = "pass"
    except (OSError, KeyError, ValueError, subprocess.TimeoutExpired) as error:
        result["status"] = "fail"
        result["error"] = str(error)
    return result


def main() -> int:
    args = arguments()
    if args.dry_run:
        for model in args.models:
            print(json.dumps(command(args, model, args.output / model), ensure_ascii=False))
        return 0
    if args.output.exists():
        raise SystemExit(f"Output directory already exists: {args.output}")
    args.output.mkdir(parents=True)
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM):
            pass
    except OSError as error:
        message = f"local sockets are unavailable before MPI_Init: {error}"
        results = [
            {"model": model, "status": "environment_blocked", "error": message}
            for model in args.models
        ]
        report_path = args.output / "report.json"
        report_path.write_text(json.dumps(
            {"case": str(args.case.resolve()), "steps": args.steps, "results": results},
            indent=2, ensure_ascii=False,
        ) + "\n")
        print(f"Environment blocked: {message}\nReport: {report_path}")
        return 2
    results = []
    for model in args.models:
        print(f"Running {model}...", flush=True)
        result = run_one(args, model)
        results.append(result)
        detail = result.get("error", "step and fields verified")
        if result["status"] == "pass" and result.get("inner_converged") is not None:
            if not all(result["inner_converged"]):
                detail += "; physical inner solve did not converge"
        print(f"  {result['status']}: {detail}", flush=True)
    report = {"case": str(args.case.resolve()), "steps": args.steps, "results": results}
    report_path = args.output / "report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"Report: {report_path}")
    return 0 if all(result["status"] == "pass" for result in results) else 1


if __name__ == "__main__":
    sys.exit(main())
