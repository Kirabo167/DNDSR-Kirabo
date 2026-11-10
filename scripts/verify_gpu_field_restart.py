#!/usr/bin/env python3
"""Verify explicit BE restart, changed dt, absolute clocks and SST field import on one/two GPUs."""
import argparse
import json
import os
from pathlib import Path
import re
import signal
import shutil
import subprocess

import h5py
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    executable = root / "build-gpu-solvers/app/acm2EQ_gpu3D.exe"
    case = root / "cases/acm/GPU/3D/suboff/verify_SST_BDF2.json"
    environment = os.environ.copy()
    environment.pop("DISPLAY", None)
    environment.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", HWLOC_COMPONENTS="-opencl",
                       OMPI_MCA_pml="ob1", OMPI_MCA_btl="self,vader,tcp",
                       OMPI_MCA_btl_tcp_if_include="lo", OMPI_MCA_oob_tcp_if_include="lo")
    results = []

    def run(ranks, backend, directory, overrides, reject=False,
            rejection_message="resetBDFHistory", runtime_reject=False):
        directory.mkdir(parents=True, exist_ok=True)
        command = ["mpirun", "--bind-to", "none", "-np", str(ranks), str(executable), str(case),
                   "--backend", backend, "--gpu-device", "auto"]
        for key, value in {"/outputSettings/directory": str(directory), **overrides}.items():
            command += ["-k", key, "-v", json.dumps(value)]
        if reject and not runtime_reject:
            command.append("--check-config")
        for attempt in range(3):
            with (directory / "run.log").open("w") as log:
                process = subprocess.Popen(command, cwd=root / "build-gpu-solvers", env=environment,
                                           stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                           start_new_session=True)
                try:
                    code = process.wait(timeout=120)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                    code = 124
            text = (directory / "run.log").read_text()
            if code == 0 or "Reading complete ACM" in text:
                break
            (directory / "run.log").rename(directory / f"startup_{attempt}.log")
        if reject:
            if code == 0 or rejection_message not in text:
                raise RuntimeError(f"Missing-history restart was not rejected: {directory}")
        elif code:
            raise RuntimeError(f"Restart verification failed: {directory}")
        return text

    def arrays(path):
        with h5py.File(path) as hf:
            return {key: dataset[()] for key, dataset in hf["VTKHDF/CellData"].items()}

    for ranks in (1, 2):
        seed = output / f"np{ranks}" / "seed"
        run(ranks, "cuda", seed, {})
        seed_files = sorted(seed.glob("*.vtkhdf"))
        source = seed_files[-1]
        imported = arrays(source)
        overrides = {"/restartSettings/flowFile": str(source), "/restartSettings/completedSteps": 3,
                     "/restartSettings/writerRanks": ranks, "/restartSettings/requireTurbulence": True,
                     "/restartSettings/physicalTime": .03, "/restartSettings/resetBDFHistory": True,
                     "/timeMarchSettings/physicalTimeStep": .02}
        run(ranks, "host", output / f"np{ranks}" / "reject_history",
            {**overrides, "/restartSettings/resetBDFHistory": False}, reject=True)
        reference = None
        for backend in ("host", "cuda"):
            directory = output / f"np{ranks}" / backend
            log = run(ranks, backend, directory, overrides)
            steps = re.findall(r"(?:resident )?physical step\s+(\d+) time=(\S+) BDF(\d)", log)
            if [int(step[2]) for step in steps] != [1, 2, 2]:
                raise RuntimeError(f"BE/BDF2 restart sequence is wrong: {steps}")
            series = json.loads(next(directory.glob("*.series")).read_text())["files"]
            series.sort(key=lambda entry: entry["time"])
            if len(series) != 4 or not np.allclose([entry["time"] for entry in series], [.03, .05, .07, .09], rtol=0, atol=1e-14):
                raise RuntimeError(f"Restart output clocks are wrong: {series}")
            fields = [arrays(directory / entry["name"]) for entry in series]
            for key, value in imported.items():
                if not np.array_equal(fields[0][key], value):
                    raise RuntimeError(f"Initial restart import changed {key}")
            if reference is None:
                reference = fields
            maximum = 0.
            for field, expected in zip(fields, reference):
                for key, value in field.items():
                    if not np.isfinite(value).all() or ("Turbulence" in key and np.min(value) <= 0):
                        raise RuntimeError(f"Invalid resumed field {key}")
                    maximum = max(maximum, float(np.max(np.abs(value - expected[key]) / np.maximum(1., np.abs(expected[key])))))
            if maximum > 5e-8:
                raise RuntimeError(f"Resumed host/CUDA mismatch: {maximum}")
            results.append(dict(ranks=ranks, backend=backend, maximum_scaled_error=maximum,
                                initial_fields_identical=True, output_times=[entry["time"] for entry in series],
                                orders=[1, 2, 2], missing_history_rejected=True, passed=True))
            (output / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
            print(json.dumps(results[-1]), flush=True)

        legacy_seed = output / f"np{ranks}" / "legacy_seed"
        run(ranks, "cuda", legacy_seed, {"/turbulenceSettings/includeIsotropicStress": False})
        legacy_source = sorted(legacy_seed.glob("*.vtkhdf"))[-1]
        expected = arrays(legacy_source)
        expected["Pressure"] = expected["Pressure"] - (2. / 3) * expected["rho_TurbulenceK"]
        conversion = {**overrides, "/restartSettings/flowFile": str(legacy_source),
                      "/restartSettings/pressureIsModified": True,
                      "/turbulenceSettings/includeIsotropicStress": True,
                      "/timeMarchSettings/nSteps": 0}
        for backend in ("host", "cuda"):
            directory = output / f"np{ranks}" / f"legacy_to_static_{backend}"
            run(ranks, backend, directory, conversion)
            fields = arrays(next(directory.glob("*.vtkhdf")))
            for key, value in expected.items():
                if key == "Pressure":
                    np.testing.assert_allclose(fields[key], value, rtol=1e-12, atol=1e-14)
                else:
                    np.testing.assert_array_equal(fields[key], value,
                                                  err_msg=f"Legacy conversion changed {key}")
            results.append(dict(ranks=ranks, backend=backend, case="legacy_modified_to_static",
                                passed=True, all_nonpressure_fields_identical=True))
        missing = output / f"np{ranks}" / "legacy_without_turbulence.vtkhdf"
        shutil.copy2(legacy_source, missing)
        with h5py.File(missing, "r+") as handle:
            for name in ("TurbulenceK", "TurbulenceOmega", "rho_TurbulenceK", "rho_TurbulenceOmega"):
                del handle["VTKHDF/CellData/" + name]
        run(ranks, "cuda", output / f"np{ranks}" / "reject_missing_k",
            {**conversion, "/restartSettings/flowFile": str(missing),
             "/restartSettings/requireTurbulence": False}, reject=True, runtime_reject=True,
            rejection_message="Modified-pressure restart conversion requires loaded turbulence k")
        results.append(dict(ranks=ranks, case="conversion_rejects_missing_k", passed=True))
        (output / "summary.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
