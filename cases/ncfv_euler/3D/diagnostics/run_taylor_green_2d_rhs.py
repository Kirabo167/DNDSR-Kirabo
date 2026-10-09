#!/usr/bin/env python3
"""Run the exact 2-D Taylor--Green momentum-RHS probe on periodic 3-D meshes."""

from __future__ import annotations

import argparse
import csv
import math
import os
import signal
import subprocess
import time
from pathlib import Path


MODES = ("Efficient", "Traditional")
TIMES = (0.0, 8.0)


def available_memory_gib() -> float:
    with Path("/proc/meminfo").open(encoding="ascii") as stream:
        for line in stream:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1048576.0
    raise RuntimeError("/proc/meminfo does not report MemAvailable")


def run_probe(command: list[str], cwd: Path, environment: dict[str, str],
              log: Path, timeout_seconds: float,
              minimum_available_gib: float) -> tuple[str, float, int]:
    started = time.monotonic()
    with log.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(command, cwd=cwd, env=environment,
                                   stdout=stream, stderr=subprocess.STDOUT,
                                   text=True, start_new_session=True)
        reason = ""
        try:
            while process.poll() is None:
                if time.monotonic() - started > timeout_seconds:
                    reason = f"timeout after {timeout_seconds:g} seconds"
                    break
                if minimum_available_gib > 0:
                    available = available_memory_gib()
                    if available < minimum_available_gib:
                        reason = (f"memory guard: {available:.1f} GiB available "
                                  f"< {minimum_available_gib:.1f} GiB")
                        break
                time.sleep(2)
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
            raise
        if reason:
            os.killpg(process.pid, signal.SIGINT)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=15)
            raise RuntimeError(f"{reason}; stopped {log}")
        return log.read_text(encoding="utf-8"), time.monotonic() - started, process.returncode


def write_plot(rows: list[dict[str, float | int | str]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator

    times = sorted({float(row["time"]) for row in rows})
    figure, axes = plt.subplots(1, len(times), figsize=(5.3 * len(times), 4.2),
                                squeeze=False, sharey=True)
    for axis, physical_time in zip(axes[0], times):
        for mode, color, marker in (("Efficient", "#147d86", "o"),
                                    ("Traditional", "#bf5b31", "s")):
            subset = sorted((row for row in rows
                             if float(row["time"]) == physical_time and row["mode"] == mode),
                            key=lambda row: int(row["n"]))
            sizes = [int(row["n"]) for row in subset]
            errors = [float(row["momentum_rhs_l2"]) for row in subset]
            axis.loglog(sizes, errors, color=color, marker=marker, linewidth=1.8,
                        markersize=6, label=mode)
        sample = sorted((row for row in rows if float(row["time"]) == physical_time
                         and row["mode"] == "Traditional"), key=lambda row: int(row["n"]))
        ticks = sorted({int(row["n"]) for row in rows
                        if float(row["time"]) == physical_time})
        if len(sample) >= 2:
            n0 = int(sample[0]["n"])
            e0 = float(sample[0]["momentum_rhs_l2"])
            axis.loglog(ticks,
                        [e0 * (n0 / value) ** 3 for value in ticks],
                        color="#666666", linestyle="--", label=r"$N^{-3}$")
        axis.set_title(f"Exact 2-D field at t={physical_time:g}")
        axis.set_xlabel("Hex cells per direction N")
        axis.xaxis.set_major_locator(FixedLocator(ticks))
        axis.xaxis.set_major_formatter(FixedFormatter([str(value) for value in ticks]))
        axis.xaxis.set_minor_locator(NullLocator())
        axis.grid(True, which="both", alpha=0.25)
        axis.legend(frameon=False)
    axes[0, 0].set_ylabel(r"Momentum RHS error $L_2$")
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    root = Path(__file__).resolve().parents[4]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[8, 16, 32, 64, 128])
    parser.add_argument("--times", type=float, nargs="+", default=TIMES)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=MODES)
    parser.add_argument("--ranks", type=int,
                        help="override the MPI rank count selected for each grid size")
    parser.add_argument("--max-rings", type=int, default=4,
                        help="maximum reconstruction adjacency rings (default: 4)")
    parser.add_argument("--allow-large", action="store_true",
                        help="permit N>=256 after confirming memory capacity")
    parser.add_argument("--min-available-gib", type=float,
                        help="stop a run when host available memory falls below this value; "
                             "defaults to 128 GiB for N>=256")
    parser.add_argument("--output", type=Path,
                        default=root / "docs/reports/taylor_green_2d_rhs_20260929/metrics.csv")
    parser.add_argument("--append", action="store_true",
                        help="extend an existing metrics CSV with new grid sizes")
    args = parser.parse_args()
    if args.ranks is not None and args.ranks < 1:
        parser.error("--ranks must be positive")
    if not 1 <= args.max_rings <= 8:
        parser.error("--max-rings must be between 1 and 8")
    if args.min_available_gib is not None and args.min_available_gib < 0:
        parser.error("--min-available-gib must be nonnegative")
    if any(n >= 256 for n in args.sizes) and not args.allow_large:
        parser.error("N>=256 exceeded available memory on this host; "
                     "pass --allow-large only after changing resources or implementation")
    if args.output.exists() and not args.append:
        parser.error(f"refusing to overwrite {args.output}")
    executable = root / "build/src/NCFV/ncfv_taylor_green_2d_rhs_probe"
    if not executable.is_file():
        parser.error(f"build the probe target first: {executable}")
    logs = root / "data/out/NCFV/taylor_green_2d_rhs"
    logs.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update({"OMP_NUM_THREADS": "1", "HWLOC_COMPONENTS": "-gl,-cuda,-nvml",
                        "OMPI_MCA_mpi_yield_when_idle": "1"})
    rows: list[dict[str, float | int | str]] = []
    if args.output.exists():
        with args.output.open(newline="", encoding="utf-8") as stream:
            rows.extend(csv.DictReader(stream))
    for row in rows:
        row.setdefault("maximum_rings", 4)
    existing = {(int(row["n"]), float(row["time"]), str(row["mode"]))
                for row in rows}
    for n in args.sizes:
        mesh = root / f"data/mesh/taylor_green/tgv_hex_{n}.cgns"
        if not mesh.is_file():
            parser.error(f"missing mesh: {mesh}")
        ranks = args.ranks or (1 if n <= 16 else 4 if n <= 32 else
                               8 if n <= 64 else 32 if n <= 128 else 64)
        for physical_time in args.times:
            for mode in args.modes:
                if (n, physical_time, mode) in existing:
                    parser.error(f"result already exists: N={n}, t={physical_time}, mode={mode}")
                launcher = (["mpirun"] + (["--use-hwthread-cpus"] if ranks > 32 else []) +
                            ["-np", str(ranks)]) if ranks > 1 else []
                command = launcher + [
                    str(executable), str(mesh), mode,
                    format(physical_time, ".17g"), str(args.max_rings)]
                print(f"N={n}, t={physical_time:g}, mode={mode}, ranks={ranks}, "
                      f"max_rings={args.max_rings}", flush=True)
                log = logs / (f"hex{n}_{mode.lower()}_t{physical_time:g}"
                              f"_rings{args.max_rings}.log")
                minimum_available = (args.min_available_gib
                                     if args.min_available_gib is not None else
                                     128.0 if n >= 256 else 0.0)
                output, elapsed, returncode = run_probe(
                    command, root / "build", environment, log,
                    7200 if n >= 256 else 3600 if n >= 128 else 900,
                    minimum_available)
                if returncode:
                    raise RuntimeError(f"probe failed (exit {returncode}): {log}")
                matches = [line.split() for line in output.splitlines()
                           if line.startswith("TGV2D_RHS_RESULT ")]
                if len(matches) != 1 or len(matches[0]) != 9:
                    raise RuntimeError(f"expected one complete result line in {log}")
                _, printed_mode, printed_time, cells, l2, relative_l2, l1, linf, mass_l2 = matches[0]
                if printed_mode != mode or abs(float(printed_time) - physical_time) > 1e-12:
                    raise RuntimeError(f"result label mismatch in {log}")
                rows.append({
                    "n": n, "time": physical_time, "mode": mode, "cells": int(cells),
                    "momentum_rhs_l2": float(l2),
                    "momentum_rhs_relative_l2": float(relative_l2),
                    "momentum_rhs_l1": float(l1),
                    "momentum_rhs_linf": float(linf),
                    "mass_rhs_l2": float(mass_l2),
                    "wall_seconds": elapsed,
                    "order_vs_previous": "",
                    "maximum_rings": args.max_rings,
                })
    rows.sort(key=lambda row: (int(row["n"]), float(row["time"]),
                               MODES.index(str(row["mode"]))))
    for row in rows:
        row["order_vs_previous"] = ""
    for physical_time in sorted({float(row["time"]) for row in rows}):
        for mode in MODES:
            subset = [row for row in rows
                      if float(row["time"]) == physical_time and row["mode"] == mode]
            subset.sort(key=lambda row: int(row["n"]))
            for previous, current in zip(subset, subset[1:]):
                if int(current["n"]) == 2 * int(previous["n"]):
                    current["order_vs_previous"] = math.log(
                        float(previous["momentum_rhs_l2"]) /
                        float(current["momentum_rhs_l2"]), 2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_suffix(".partial.csv")
    with partial.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    partial.replace(args.output)
    os.environ.setdefault("MPLCONFIGDIR", str(logs / "matplotlib-cache"))
    write_plot(rows, args.output.with_name("convergence.svg"))
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
