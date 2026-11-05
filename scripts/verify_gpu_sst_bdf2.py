#!/usr/bin/env python3
"""Compare actual conservative SST/BDF2 GPU fields against the frozen CPU source."""
import argparse
import hashlib
import json
import os
import signal
from pathlib import Path
import subprocess

import h5py
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/out/GPU/SUBOFF_SST_BDF2_20261104/verification")
    parser.add_argument("--ranks", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--modes", nargs="+", default=["wall", "far", "inlet", "outlet", "symmetry", "rusanov", "first", "green_gauss", "cfl_local", "cfl_global", "rho2", "linear_sweeps", "history_BE"])
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = (root / args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = root / "cases/acm/GPU/3D/suboff/verify_SST_BDF2.json"
    environment = os.environ.copy()
    environment.pop("DISPLAY", None)
    environment.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", HWLOC_COMPONENTS="-opencl",
                       OMPI_MCA_pml="ob1", OMPI_MCA_btl="self,vader,tcp", OMPI_MCA_btl_tcp_if_include="lo",
                       OMPI_MCA_oob_tcp_if_include="lo")
    summary = []
    for ranks in args.ranks:
        for mode in args.modes:
            overrides = {}
            bc = {"far": "BCFar", "inlet": "BCInPsTs", "outlet": "BCOutP", "symmetry": "BCSym"}.get(mode)
            if bc:
                overrides["/boundaryConditions"] = [{"name": "FAR", "type": bc, "value": [.35, .05, 0., .1],
                    "frameOption": 0, "anchorOption": 0, "integrationOption": 0, "specialOption": 0, "rectifyOption": 0, "valueExtra": []}]
            if mode == "rusanov":
                overrides["/acmSettings/riemannSolverType"] = "Rusanov"
            if mode in ["first", "green_gauss"]:
                overrides["/reconstructionSettings/type"] = "FirstOrder" if mode == "first" else "GreenGauss"
            if mode.startswith("cfl_"):
                overrides.update({"/timeMarchSettings/useCFLTimeStep": True,
                                  "/timeMarchSettings/useLocalTimeStep": mode == "cfl_local",
                                  "/timeMarchSettings/maximumPseudoTimeStep": .0002})
            if mode == "rho2":
                overrides["/acmSettings/rho0"] = 2.
            if mode == "linear_sweeps":
                overrides.update({"/timeMarchSettings/lusgsSweeps": 3,
                                  "/timeMarchSettings/linearRelativeTolerance": 1e-13,
                                  "/timeMarchSettings/linearAbsoluteTolerance": 1e-15})
            if mode == "history_BE":
                overrides.update({"/boundaryConditions/0/type": "BCOut",
                                  "/initialState": [0., 0., 0., 0., .001, 1000.],
                                  "/timeMarchSettings/pseudoTimeStep": 5.,
                                  "/timeMarchSettings/physicalTimeStep": 10.,
                                  "/timeMarchSettings/maxImplicitIterations": 30,
                                  "/timeMarchSettings/implicitTolerance": 1e-10})
            cpu = None
            for backend in ["cpu", "host", "cuda"]:
                directory = output / f"{mode}_np{ranks}" / backend
                directory.mkdir(parents=True, exist_ok=True)
                exe = root / ("build/app/acm2EQ3D.exe" if backend == "cpu" else "build-gpu-solvers/app/acm2EQ_gpu3D.exe")
                command = ["mpirun", "-np", str(ranks), str(exe), str(config)]
                if backend != "cpu":
                    command += ["--backend", backend, "--gpu-device", "auto"]
                for key, value in {**overrides, "/outputSettings/directory": str(directory)}.items():
                    command += ["-k", key, "-v", json.dumps(value)]
                for attempt in range(3):
                    with (directory / "run.log").open("w") as handle:
                        process = subprocess.Popen(command, cwd=root / "build-gpu-solvers", env=environment,
                                                   stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
                        try:
                            exit_code = process.wait(timeout=90)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait()
                            exit_code = 124
                    text = (directory / "run.log").read_text()
                    if not exit_code or "Reading complete ACM" in text:
                        break
                    (directory / "run.log").rename(directory / f"mpi_startup_attempt_{attempt + 1}.log")
                if exit_code:
                    raise RuntimeError(f"{mode} np{ranks} {backend} failed: {directory / 'run.log'}")
                log = (directory / "run.log").read_text()
                if backend == "cuda" and ("ACM resident physical step 3" not in log or "Resident CUDA" not in log):
                    raise RuntimeError("CUDA run did not execute the resident physical solver")
                fields = {}
                for file in sorted(directory.glob("*.vtkhdf")):
                    with h5py.File(file) as hf:
                        fields[file.name] = {k: v[()] for k, v in hf["VTKHDF/CellData"].items()}
                if len(fields) != 4:
                    raise RuntimeError(f"Expected initial and three physical outputs: {directory}")
                if cpu is None:
                    cpu = fields
                maximum = 0.
                for name, arrays in fields.items():
                    for key, values in arrays.items():
                        if not np.isfinite(values).all():
                            raise RuntimeError(f"Non-finite {name}/{key}")
                        reference = cpu[name][key]
                        maximum = max(maximum, float(np.max(np.abs(values - reference) / np.maximum(1., np.abs(reference)))))
                entry = dict(mode=mode, ranks=ranks, backend=backend, maximum_scaled_error=maximum,
                             tolerance=5e-8, passed=maximum <= 5e-8, log=str(directory / "run.log"),
                             executable_sha256=hashlib.file_digest(exe.open("rb"), "sha256").hexdigest())
                summary.append(entry)
                (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
                print(json.dumps(entry), flush=True)
                if not entry["passed"]:
                    raise RuntimeError(f"GPU/CPU SST/BDF2 comparison failed: {entry}")


if __name__ == "__main__":
    main()
