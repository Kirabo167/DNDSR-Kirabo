#!/usr/bin/env python3
"""Monitor ACM inner-residual reduction from an existing solver log."""

import argparse
import csv
from datetime import datetime
import json
import math
import os
from pathlib import Path
import re
import time


GMRES = re.compile(
    r"ACM GMRES restart=\d+ residual=(?P<linear>[0-9.eE+-]+) "
    r"initial=(?P<initial>[0-9.eE+-]+)"
)
STEP = re.compile(
    r"ACM step\s+(?P<step>\d+) residual (?P<initial>[0-9.eE+-]+) "
    r"-> (?P<final>[0-9.eE+-]+).* inner=(?P<inner>\d+) "
    r"converged=(?P<converged>[01]).* innerTarget=(?P<target>[0-9.eE+-]+) "
    r"steadyResidual=(?P<steady>[0-9.eE+-]+).* recSweeps=(?P<rec_sweeps>\d+) "
    r"recDefect=(?P<rec_defect>[0-9.eE+-]+) recConverged=(?P<rec_converged>[01])"
)


def process_exists(pid_file):
    try:
        pid = int(pid_file.read_text().strip())
        os.kill(pid, 0)
        return True
    except (OSError, ValueError):
        return False


def parse_log(path):
    pending = []
    records = []
    for line in path.read_text(errors="replace").splitlines():
        match = GMRES.search(line)
        if match:
            pending.append(float(match["initial"]))
            continue
        match = STEP.search(line)
        if not match:
            continue
        inner = int(match["inner"])
        values = pending[-inner:] if inner else []
        ratios = [values[i] / values[i - 1] for i in range(1, len(values))
                  if values[i - 1] != 0]
        initial = float(match["initial"])
        final = float(match["final"])
        outer_ratio = final / initial if initial != 0 else math.nan
        records.append({
            "step": int(match["step"]),
            "outer_initial": initial,
            "outer_final": final,
            "outer_ratio": outer_ratio,
            "average_inner_factor": outer_ratio ** (1.0 / inner)
            if inner > 0 and outer_ratio >= 0 else math.nan,
            "inner_iterations": inner,
            "inner_target": float(match["target"]),
            "converged": int(match["converged"]),
            "linear_proxy_initials": values,
            "linear_proxy_ratios": ratios,
            "linear_proxy_monotonic": all(ratio < 1 for ratio in ratios),
            "steady_residual": float(match["steady"]),
            "reconstruction_corrections": int(match["rec_sweeps"]),
            "reconstruction_defect": float(match["rec_defect"]),
            "reconstruction_converged": int(match["rec_converged"]),
        })
        pending = []
    return records


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def write_csv(path, records):
    temporary = path.with_suffix(path.suffix + ".tmp")
    fields = [
        "step", "outer_initial", "outer_final", "outer_ratio",
        "average_inner_factor", "inner_iterations", "inner_target",
        "converged", "linear_proxy_initials", "linear_proxy_ratios",
        "linear_proxy_monotonic", "steady_residual",
        "reconstruction_corrections", "reconstruction_defect",
        "reconstruction_converged",
    ]
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row = dict(record)
            row["linear_proxy_initials"] = ";".join(
                f"{value:.12e}" for value in row["linear_proxy_initials"])
            row["linear_proxy_ratios"] = ";".join(
                f"{value:.12e}" for value in row["linear_proxy_ratios"])
            writer.writerow(row)
    temporary.replace(path)


def status(records, solver_running):
    latest = records[-1] if records else None
    alerts = []
    if not solver_running:
        alerts.append("solver_process_not_running")
    if latest:
        if not latest["converged"]:
            alerts.append("inner_iteration_not_converged")
        if latest["outer_final"] > latest["inner_target"]:
            alerts.append("inner_target_not_met")
        if not latest["linear_proxy_monotonic"]:
            alerts.append("linear_proxy_not_monotonic")
        if any(value >= 0.5 for value in latest["linear_proxy_ratios"]):
            alerts.append("linear_proxy_reduction_slower_than_0.5")
        if not latest["reconstruction_converged"]:
            alerts.append("reconstruction_not_converged")
    recent = records[-5:]
    return {
        "updated_at": datetime.now().astimezone().isoformat(),
        "solver_process_running": solver_running,
        "latest": latest,
        "recent_5_mean_outer_ratio": (
            sum(item["outer_ratio"] for item in recent) / len(recent)
            if recent else None
        ),
        "recent_5_mean_inner_factor": (
            sum(item["average_inner_factor"] for item in recent) / len(recent)
            if recent else None
        ),
        "alert": bool(alerts),
        "alert_reasons": alerts,
        "note": (
            "linear_proxy_initials are the preconditioned flow-GMRES RHS norms "
            "logged once per nonlinear inner iteration; outer_initial/final are "
            "the true nonlinear pseudo-time defect norms."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--mpi-pid", type=Path, required=True)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=15)
    args = parser.parse_args()
    if args.poll_seconds <= 0:
        parser.error("poll-seconds must be positive")

    previous_step = None
    while True:
        records = parse_log(args.log)
        running = process_exists(args.mpi_pid)
        latest_step = records[-1]["step"] if records else None
        if latest_step != previous_step or not args.status.exists() or not running:
            write_csv(args.csv, records)
            atomic_json(args.status, status(records, running))
            previous_step = latest_step
        if not args.watch or not running:
            break
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
