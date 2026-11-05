#!/usr/bin/env python3
"""Compare frozen CPU, copied host and CUDA solver fields on small evolving cases.

Uses numpy/h5py only; does not load the DNDSR Python bindings.
Run on a GPU host after building all_gpu_solvers and gpu_flux_verify.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import xml.etree.ElementTree as ET

import h5py
import numpy as np


def fields(directory, model):
    if model == "acm":
        files = sorted(directory.rglob("*.vtkhdf"))
        if not files:
            raise RuntimeError(f"Missing ACM fields in {directory}")
        with h5py.File(files[-1]) as handle:
            return {name: np.asarray(value) for name, value in handle["VTKHDF/CellData"].items()}
    files = sorted(directory.rglob("*.vtu"))
    if not files:
        raise RuntimeError(f"Missing NCFV fields in {directory}")
    # Compare every written step and MPI partition, including initial/final data.
    result = {}
    for path in files:
        tree = ET.parse(path)
        for data in tree.findall(".//PointData/DataArray"):
            if data.get("format", "ascii") != "ascii":
                raise RuntimeError("NCFV verification expects ASCII VTU")
            result[f"{path.relative_to(directory)}:{data.get('Name')}"] = np.fromstring(data.text or "", sep=" ")
    return result


def compare(reference, actual):
    if reference.keys() != actual.keys() or not reference:
        raise RuntimeError("Output fields or files differ")
    maximum = 0.0
    for name in reference:
        left, right = reference[name], actual[name]
        if left.shape != right.shape or left.size == 0:
            raise RuntimeError(f"Field shape mismatch: {name}")
        if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
            raise RuntimeError(f"Non-finite field: {name}")
        error = float(np.max(np.abs(left - right) / np.maximum(1.0, np.abs(left))))
        maximum = max(maximum, error)
        if error > 5e-10:
            raise RuntimeError(f"Field mismatch {name}: scaled error {error:.3e}")
    return maximum


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", default="build-gpu-solvers")
    parser.add_argument("--device", default="1")
    parser.add_argument("--models", nargs="+", choices=["acm", "ncfv"], default=["acm", "ncfv"])
    parser.add_argument("--modes", nargs="+", help="Limit the validation to named algorithms")
    parser.add_argument("--cpu-snapshot", help="Frozen CPU directory; defaults to frozen/active_cpu_snapshot.txt")
    parser.add_argument("--skip-completed-summary", help="Reuse completed comparisons from the same CPU baseline")
    parser.add_argument("--mpi", action="store_true", help="Also compare two-rank runs using automatic device assignment")
    parser.add_argument("--transport-only", action="store_true", help="Run only the two-GPU halo stress verification")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    build = (root / args.build).resolve()
    active = root / "frozen/active_cpu_snapshot.txt"
    default = active.read_text().strip() if active.exists() else "ACM_NCFV_CPU_20261103"
    snapshot = (root / args.cpu_snapshot).resolve() if args.cpu_snapshot else root / "frozen" / default
    metadata = json.loads((snapshot / "metadata.json").read_text())
    output = root / "data/out/GPU" / f"validation_{time.strftime('%Y%m%d_%H%M%S')}"
    output.mkdir(parents=True, exist_ok=False)
    (output / "baseline.json").write_text(json.dumps({"cpu_snapshot": str(snapshot), "cpu_git_head": metadata["git_head"], "execution_kind": "resident"}, indent=2) + "\n")
    environment = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", HWLOC_COMPONENTS="-opencl")
    environment.update(OMPI_MCA_pml="ob1", OMPI_MCA_btl="self,vader,tcp",
                       OMPI_MCA_btl_tcp_if_include="lo", OMPI_MCA_oob_tcp_if_include="lo")
    environment.pop("DISPLAY", None)
    if args.transport_only:
        subprocess.run(["/usr/bin/mpirun", "-np", "2", str(build / "src/GPU/gpu_resident_halo_verify")],
                       cwd=build, env=environment, check=True, timeout=180)
        return
    summary = []
    completed_cases = set()
    if args.skip_completed_summary:
        previous = Path(args.skip_completed_summary).resolve()
        prior_baseline = json.loads((previous.parent / "baseline.json").read_text())
        if prior_baseline.get("execution_kind") != "resident":
            raise RuntimeError("A hybrid-backend comparison cannot validate the resident GPU solver")
        if prior_baseline["cpu_snapshot"] != str(snapshot) or prior_baseline["cpu_git_head"] != metadata["git_head"]:
            raise RuntimeError("Previous comparisons use a different CPU baseline")
        summary = json.loads(previous.read_text())
        for name in {entry["case"] for entry in summary}:
            if {entry["backend"] for entry in summary if entry["case"] == name} == {"cpu", "host", "cuda"}:
                completed_cases.add(name)
        summary = [entry for entry in summary if entry["case"] in completed_cases]
    checks = [(model, dimension, mode, 1)
              for model in ["acm", "ncfv"] for dimension in [2, 3]
              for mode in (["Roe", "Rusanov"] if model == "acm" else ["EfficientDifferential", "TraditionalQuadrature"])]
    checks += [("ncfv", 3, mode, 1) for mode in
               ["RankAware", "SharedGradientParallel", "SharedGradientBudget", "FluxCorrectionBinary", "FluxCorrectionEntropy"]]
    checks += [("ncfv", dimension, mode, 1) for dimension in [2, 3] for mode in
               ["PressureLimit", "ViscousEfficient", "ViscousTraditional", "VariationalEfficient", "VariationalTraditional"]]
    checks += [("acm", dimension, mode, 1) for dimension in [2, 3] for mode in
               ["GreenGauss", "FirstOrder", "JacobiReconstruction", "Cubic", "ControlledReconstruction", "CFL",
                "NoSlipWall", "SlipWall", "PressureOutlet", "VelocityInlet", "Symmetry"]]
    if args.mpi:
        checks += [(model, dimension, mode, 2) for model, dimension, mode, _ in checks
                   if mode in ["Roe", "EfficientDifferential", "SharedGradientParallel", "SharedGradientBudget",
                               "FluxCorrectionBinary", "FluxCorrectionEntropy", "PressureLimit", "ViscousEfficient",
                               "ViscousTraditional", "VariationalEfficient", "VariationalTraditional"]]
        checks += [("acm", dimension, mode, 2) for dimension in [2, 3] for mode in
                   ["GreenGauss", "JacobiReconstruction", "Cubic", "ControlledReconstruction", "CFL", "NoSlipWall"]]
    checks = [entry for entry in checks if entry[0] in args.models and (not args.modes or entry[2] in args.modes)]
    if not checks:
        raise RuntimeError("No solver comparisons matched the requested filters")
    if not completed_cases:
        subprocess.run([str(build / "app/gpu_flux_verify.exe"), args.device], cwd=build, env=environment, check=True)
    for model, dimension, mode, ranks in checks:
        name = f"{model}{dimension}D_{mode}_np{ranks}"
        if name in completed_cases:
            continue
        gpu = f"acm_gpu{dimension}D" if model == "acm" else f"ncfv_euler_gpu{dimension}D"
        cpu = f"acm{dimension}D" if model == "acm" else f"ncfv_euler{dimension}D"
        case = root / f"cases/{'acm' if model == 'acm' else 'ncfv_euler'}/GPU/{dimension}D/{gpu}.json"
        reference = None
        for backend in ["cpu", "host", "cuda"]:
            target = output / name / backend
            target.mkdir(parents=True)
            exe = snapshot / f"bin/{cpu}.exe" if backend == "cpu" else build / f"app/{gpu}.exe"
            command = [str(exe), str(case)]
            if backend != "cpu":
                command += ["--backend", backend, "--gpu-device", "auto" if ranks > 1 else args.device]
            overrides = {}
            if model == "acm":
                overrides = {"/outputSettings/directory": str(target),
                             "/acmSettings/riemannSolverType": mode if mode in ["Roe", "Rusanov"] else "Roe"}
                if mode in ["GreenGauss", "FirstOrder"]:
                    overrides["/reconstructionSettings/type"] = mode
                if mode == "JacobiReconstruction":
                    overrides["/vfvSettings/SORInstead"] = False
                if mode == "Cubic":
                    overrides["/vfvSettings/maxOrder"] = 3
                    overrides["/vfvSettings/intOrder"] = 6
                if mode == "ControlledReconstruction":
                    overrides["/reconstructionSettings/variationalTolerance"] = 1e-9
                    overrides["/reconstructionSettings/variationalMaxIterations"] = 10000
                if mode == "CFL":
                    overrides.update({"/timeMarchSettings/useCFLTimeStep": True,
                                      "/timeMarchSettings/useLocalTimeStep": False,
                                      "/timeMarchSettings/maximumPseudoTimeStep": .005})
                if mode in ["NoSlipWall", "SlipWall", "PressureOutlet", "VelocityInlet", "Symmetry"]:
                    overrides["/defaultBoundaryType"] = mode
            else:
                overrides = {"/io/outputPrefix": str(target / "flow"), "/io/vtkSeriesName": str(target / "series"),
                             "/io/restartPrefix": str(target / "restart"),
                             "/algorithm/mode": "TraditionalQuadrature" if "Traditional" in mode else "EfficientDifferential"}
                if "Traditional" in mode:
                    overrides.update({"/reconstruction/barthZeroOrderFallback": False, "/reconstruction/limitPointRecovery": False})
                if mode == "PressureLimit":
                    overrides.update({"/reconstruction/barthZeroOrderFallback": False, "/reconstruction/limitPressure": True})
                if mode.startswith("Viscous"):
                    overrides.update({"/reconstruction/barthZeroOrderFallback": False, "/physics/viscous/enabled": True,
                                      "/physics/viscous/model": "Constant", "/physics/viscous/dynamicViscosity": 0.01,
                                      "/time/useCFLTimeStep": True, "/time/useLocalTimeStep": False,
                                      "/time/cfl": 0.1, "/time/maximumTimeStep": 0.005})
                if mode.startswith("Variational"):
                    overrides.update({"/reconstruction/method": "Variational"})
                if mode == "RankAware":
                    overrides["/reconstruction/stencilSelectionStrategy"] = "RankAware"
                if mode in ["SharedGradientParallel", "SharedGradientBudget"]:
                    overrides.update({"/reconstruction/barthZeroOrderFallback": False,
                                      "/reconstruction/sharedGradientFaceSweep": True,
                                      "/reconstruction/sharedGradientFaceSweepParallel": True,
                                      "/reconstruction/sharedGradientBudget": mode == "SharedGradientBudget",
                                      "/reconstruction/limitPressure": mode == "SharedGradientParallel"})
                if mode in ["FluxCorrectionBinary", "FluxCorrectionEntropy"]:
                    overrides.update({"/reconstruction/enableLimiter": False,
                                      "/reconstruction/limitPointRecovery": False,
                                      "/reconstruction/barthZeroOrderFallback": False,
                                      "/reconstruction/limitFluxCorrection": True,
                                      "/reconstruction/fluxCorrectionBinary": mode == "FluxCorrectionBinary",
                                      "/reconstruction/fluxCorrectionEntropy": True,
                                      "/time/useCFLTimeStep": True, "/time/useLocalTimeStep": False,
                                      "/time/cfl": 0.2, "/time/maximumTimeStep": 0.0001})
            for key, value in overrides.items():
                command += ["-k", key, "-v", json.dumps(value)]
            # Use an explicit launcher also at np=1: this avoids Open MPI's
            # singleton-daemon startup path on remote desktop hosts.
            command = ["/usr/bin/mpirun", "-np", str(ranks)] + command
            log = target / "run.log"
            for attempt in range(3):
                with log.open("w") as handle:
                    completed = subprocess.run(command, cwd=build, env=environment, stdout=handle, stderr=subprocess.STDOUT, timeout=180)
                if not completed.returncode:
                    break
                message = log.read_text()
                transient_startup = "An error occurred in MPI_Init_thread" in message or "OPAL ERROR: Unreachable" in message
                if not transient_startup or attempt == 2:
                    break
                log.rename(target / f"mpi_startup_attempt_{attempt + 1}.log")
                time.sleep(0.5)
            if completed.returncode:
                raise RuntimeError(f"{name}/{backend} failed; see {log}")
            if backend == "cuda" and "batches=0," in log.read_text():
                raise RuntimeError(f"No CUDA batches were executed: {log}")
            if backend == "cuda" and "Resident CUDA rank" not in log.read_text():
                raise RuntimeError(f"The solver used a hybrid path instead of resident GPU execution: {log}")
            values = fields(target, model)
            if reference is None:
                reference = values
                error = 0.0
            else:
                error = compare(reference, values)
            summary.append({"case": name, "backend": backend, "maximum_scaled_error": error, "log": str(log)})
            print(f"{name}/{backend}: passed, maximum scaled error={error:.3e}", flush=True)
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Saved comparison results: {output}")


if __name__ == "__main__":
    main()
