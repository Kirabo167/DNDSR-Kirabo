#!/usr/bin/env python3
"""Capture real CUDA activity with Nsight; inspect with check_gpu_residency.py.

This is an external profiler audit, independent of solver-owned byte counters.
It does not import DNDSR Python bindings or rewrite any solver configuration.
"""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", default="build-gpu-solvers")
    parser.add_argument("--nsys", default=shutil.which("nsys") or
                        "/opt/nvidia/hpc_sdk/Linux_x86_64/26.3/compilers/bin/nsys")
    parser.add_argument("--sanitizer", default="/usr/local/cuda-13.2/bin/compute-sanitizer")
    parser.add_argument("--mpi", action="store_true")
    parser.add_argument("--sst", action="store_true", help="Audit conservative SST/BDF2 instead of the explicit solvers")
    parser.add_argument("--output", help="Explicit report directory")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    build = (root / args.build).resolve()
    output = (root / args.output).resolve() if args.output else root / "data/out/GPU" / time.strftime("resident_profile_%Y%m%d_%H%M%S")
    output.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", HWLOC_COMPONENTS="-opencl")
    environment.update(OMPI_MCA_pml="ob1", OMPI_MCA_btl="self,vader,tcp",
                       OMPI_MCA_btl_tcp_if_include="lo", OMPI_MCA_oob_tcp_if_include="lo")
    environment.pop("DISPLAY", None)
    for model in (["acm_sst"] if args.sst else ["acm", "ncfv"]):
        exe = f"{model}_gpu3D" if model == "acm" else "ncfv_euler_gpu3D"
        case = root / f"cases/{'acm' if model == 'acm' else 'ncfv_euler'}/GPU/3D/{exe}.json"
        if model == "acm_sst":
            exe = "acm2EQ_gpu3D"
            case = root / "cases/acm/GPU/3D/suboff/verify_SST_BDF2.json"
        command = [str(build / f"app/{exe}.exe"), str(case), "--backend", "cuda", "--gpu-device", "auto"]
        overrides = ({"/outputSettings/interval": "0", "/timeMarchSettings/nSteps": "4"}
                     if model in ["acm", "acm_sst"] else {"/io/writeVTK": "false", "/io/writeFinalRestart": "false", "/time/iterations": "4"})
        for key, value in overrides.items():
            command += ["-k", key, "-v", value]
        for ranks in ([1, 2] if args.mpi else [1]):
            label = f"{model}_np{ranks}"
            report = str(output / (label + "_rank%q{OMPI_COMM_WORLD_RANK}"))
            profile = ["/usr/bin/mpirun", "-np", str(ranks), args.nsys, "profile",
                       "--trace=cuda,nvtx", "--sample=none", "--cpuctxsw=none", "--export=sqlite",
                       "--output", report] + command
            with (output / f"{label}_profile.log").open("w") as handle:
                subprocess.run(profile, cwd=build, env=environment, stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT,
                               check=True, timeout=240)
            print(f"Captured {label}: {output}", flush=True)
        sanitizer = ["/usr/bin/mpirun", "-np", "1", args.sanitizer, "--tool", "memcheck",
                     "--error-exitcode", "99"] + command
        with (output / f"{model}_memcheck.log").open("w") as handle:
            subprocess.run(sanitizer, cwd=build, env=environment, stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT,
                           check=True, timeout=240)
        print(f"Memory check passed: {model}", flush=True)
    print(output)


if __name__ == "__main__":
    main()
