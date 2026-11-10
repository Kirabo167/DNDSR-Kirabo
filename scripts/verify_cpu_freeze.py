#!/usr/bin/env python3
"""Check the frozen CPU artifacts and the CPU sources they were made from."""
import hashlib
import argparse
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", help="Snapshot directory, relative to the project or absolute")
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--artifacts-only", action="store_true", help="Check an older snapshot without comparing current sources")
    scope.add_argument("--sources-only", action="store_true", help="Check frozen CPU sources without local-only binaries and archives")
    args = parser.parse_args()
    active = root / "frozen/active_cpu_snapshot.txt"
    default = active.read_text().strip() if active.exists() else "ACM_NCFV_CPU_20261103"
    snapshot = root / args.snapshot if args.snapshot else root / "frozen" / default
    checked = 0
    manifests = []
    if not args.sources_only:
        manifests.append((snapshot / "artifacts.sha256", snapshot))
    if not args.artifacts_only:
        manifests.append((snapshot / "cpu_sources.sha256", root))
    for manifest, base in manifests:
        for line in manifest.read_text().splitlines():
            expected, name = line.split("  ", 1)
            path = base / name
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != expected:
                raise SystemExit(f"CPU freeze mismatch: {path}")
            checked += 1
    print(f"CPU freeze verified: {snapshot.name}, {checked} artifact/source hashes match.")


if __name__ == "__main__":
    main()
