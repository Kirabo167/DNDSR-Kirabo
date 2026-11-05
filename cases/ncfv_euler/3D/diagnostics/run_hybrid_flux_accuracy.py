"""Run four native flux crossovers and the native-efficient initialization control.

The C++ probe must first be built against the current ncfv library. This driver
does not rewrite any JSON case. It never overwrites an existing output directory.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[4]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parallel-fine-grids", action="store_true",
                        help="run IV40(np16) and IV80(np32) together, total 48 ranks")
    args = parser.parse_args()
    executable = args.binary.resolve(strict=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    environment = dict(os.environ, HWLOC_COMPONENTS="-gl,-cuda,-nvml",
                       OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")

    def run(level, ranks, suffix="", verify=False):
        name = f"iv{level}{suffix}"
        command = ["mpirun", "--bind-to", "none", "-np", str(ranks), str(executable),
                   str(ROOT / f"cases/ncfv_euler/3D/NCFV_iv{level}.json"),
                   str(ROOT / f"cases/ncfv_euler/3D/iv{level}_3d_1_dnds.cgns"),
                   str(output / f"{name}.json"), str(2.0 / level), "1.6"]
        if verify:
            command.append("verify-production")
        with (output / f"{name}.log").open("w") as log:
            subprocess.run(command, cwd=ROOT / "build", env=environment,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        print(f"Completed {name}, MPI ranks={ranks}", flush=True)

    run(10, 1, "_np1", True)
    run(10, 2, "_np2", True)
    run(20, 8)
    if args.parallel_fine_grids:
        assert 16 + 32 <= 64
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(run, 40, 16), pool.submit(run, 80, 32)]
            for future in futures:
                future.result()
    else:
        run(40, 16)
        run(80, 32)


if __name__ == "__main__":
    main()
