#!/usr/bin/env python3
"""Externally retain every 50th ACM field and six other completed fields.

Completion is established by ACM's existing post-close log message. Never delete
an unconfirmed/in-progress file or edit the solver's own VTK series. Each cleanup
pass finishes synchronously; a separate retained series lists only existing data.
"""

import argparse
import fcntl
import json
from pathlib import Path
import re
import sys
import time


def atomic_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


class CompletionLog:
    """Read only complete appended log lines, validating their output paths."""

    pattern = re.compile(r"ACM flow field written at step (\d+) to (.+\.vtkhdf)$")

    def __init__(self, path, directory, prefix, solver_cwd):
        self.stream = path.open("rb")
        self.directory = directory.resolve()
        self.prefix = prefix
        self.solver_cwd = solver_cwd.resolve()
        self.pending = b""
        self.completed = set()

    def poll(self):
        self.pending += self.stream.read()
        lines = self.pending.split(b"\n")
        self.pending = lines.pop()
        for line in lines:
            match = self.pattern.search(line.decode("utf-8", errors="replace").strip())
            if match is None:
                continue
            step = int(match[1])
            actual = Path(match[2])
            if not actual.is_absolute():
                actual = self.solver_cwd / actual
            expected = self.directory / f"{self.prefix}_{step:08d}.vtkhdf"
            if actual.resolve() == expected:
                self.completed.add(step)
        return self.completed

    def close(self):
        self.stream.close()


def prune(directory, prefix, window, every, completed, dry_run=False):
    """Keep all periodic fields plus the latest window non-periodic fields."""
    if not completed:
        return None
    step = max(completed)
    newest = directory / f"{prefix}_{step:08d}.vtkhdf"
    if newest.is_symlink() or not newest.is_file() or newest.stat().st_size == 0:
        raise ValueError(f"Latest confirmed field is missing or empty: {newest}")
    pattern = re.compile(re.escape(prefix) + r"_(\d{8,})\.vtkhdf$")
    fields = {}
    for path in directory.iterdir():
        match = pattern.fullmatch(path.name)
        if match and not path.is_symlink() and path.is_file():
            file_step = int(match[1])
            if file_step in completed:
                fields[file_step] = path
    recent = sorted(s for s in fields if s % every != 0)
    expired_steps = recent[:-window]
    retained = sorted(set(fields) - set(expired_steps))
    expired = [fields[s] for s in expired_steps]
    removed_bytes = sum(path.stat().st_size for path in expired)
    result = {
        "completed_step": step, "window_nonperiodic": window,
        "permanent_interval": every, "removed_files": [p.name for p in expired],
        "removed_bytes": removed_bytes, "retained_field_steps": retained,
        "retained_nonperiodic_steps": [s for s in retained if s % every != 0],
        "dry_run": dry_run,
    }
    if not dry_run:
        # A deletion failure aborts this pass instead of silently growing a backlog.
        for path in expired:
            path.unlink()
        atomic_json(directory / f"{prefix}.retained.vtkhdf.series", {
            "file-series-version": "1.0",
            "files": [{"name": fields[s].name, "time": s} for s in retained],
        })
        atomic_json(directory / f"{prefix}.retention_status.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--solver-cwd", type=Path, required=True)
    parser.add_argument("--window", type=int, default=6)
    parser.add_argument("--every", type=int, default=50)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=0.25)
    parser.add_argument("--status-file", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.window < 1 or args.every < 1 or args.poll_seconds <= 0:
        parser.error("window, every and poll-seconds must be positive")
    if not args.prefix or Path(args.prefix).name != args.prefix or args.prefix in (".", ".."):
        parser.error("prefix must be a filename component")
    directory = args.directory.resolve(strict=True)
    with (directory / f"{args.prefix}.retention.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        tracker = CompletionLog(args.log, directory, args.prefix, args.solver_cwd)
        try:
            last_step = -1
            while True:
                completed = tracker.poll()
                if completed and max(completed) != last_step:
                    result = prune(directory, args.prefix, args.window, args.every, completed, args.dry_run)
                    print(json.dumps(result), flush=True)
                    last_step = max(completed)
                if not args.watch:
                    break
                if args.status_file and args.status_file.exists() and args.status_file.read_text().strip() != "running":
                    result = prune(directory, args.prefix, args.window, args.every, tracker.poll(), args.dry_run)
                    if result is not None:
                        print(json.dumps(result), flush=True)
                    break
                time.sleep(args.poll_seconds)
        finally:
            tracker.close()


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError) as error:
        print(f"ACM retention failed: {error}", file=sys.stderr, flush=True)
        sys.exit(1)
