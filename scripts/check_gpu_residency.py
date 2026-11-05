#!/usr/bin/env python3
"""Audit actual CUDA copies/allocations inside NVTX numerical-step ranges.

Consumes Nsight Systems SQLite exports from profile_gpu_residency.py. Outside
explicit MPI halo transport, only 4/8/32/48-byte control scalars may cross the
host/device boundary during a step. Halo staging is measured separately.
"""
import argparse
import json
from pathlib import Path
import sqlite3


def inspect(path):
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
        ranges = connection.execute(
            "SELECT start,end,text,globalTid FROM NVTX_EVENTS WHERE text IN "
            "('ACM numerical step','NCFV numerical step') ORDER BY start").fetchall()
        if not ranges:
            raise RuntimeError(f"No numerical steps were captured: {path}")
        halo_ranges = connection.execute(
            "SELECT start,end FROM NVTX_EVENTS WHERE text='MPI halo transport'").fetchall()
        kernels = scalar_bytes = device_bytes = allocations = forbidden_bytes = halo_bytes = 0
        for start, end, _, tid in ranges:
            # Inspect GPU activity directly. NCCL may issue driver APIs from
            # proxy threads, without a matching runtime-API correlation record.
            copies = connection.execute(
                "SELECT copyKind,bytes,start,end FROM CUPTI_ACTIVITY_KIND_MEMCPY "
                "WHERE start>=? AND end<=?", (start, end)).fetchall()
            for kind, size, copy_start, copy_end in copies:
                if kind in [1, 2]:  # CUPTI HtoD and DtoH.
                    if any(a <= copy_start and copy_end <= b for a, b in halo_ranges):
                        halo_bytes += size
                    elif size not in [4, 8, 32, 48]:
                        forbidden_bytes += size
                    else:
                        scalar_bytes += size
                elif kind in [8, 10]:  # DtoD and peer-to-peer.
                    device_bytes += size
                else:
                    raise RuntimeError(f"Unexpected CUDA copy kind {kind}: {path}")
            kernels += connection.execute(
                "SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL "
                "WHERE start>=? AND end<=?", (start, end)).fetchone()[0]
            allocations += connection.execute(
                "SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON r.nameId=s.id "
                "WHERE r.start>=? AND r.end<=? "
                "AND (s.value LIKE 'cudaMalloc%' OR s.value LIKE 'cudaFree%')", (start, end)).fetchone()[0]
        if forbidden_bytes or allocations or kernels == 0:
            raise RuntimeError(f"Residency audit failed: {path}, field transfers={forbidden_bytes}, "
                               f"allocations={allocations}, kernels={kernels}")
        return {"profile": str(path.resolve()), "steps": len(ranges), "kernels": kernels,
                "field_transfer_scope": "numerical fields excluding explicitly marked MPI halo transport",
                "control_scalar_bytes": scalar_bytes, "device_copy_bytes": device_bytes,
                "MPI_halo_staging_bytes": halo_bytes,
                "step_host_field_transfer_bytes": forbidden_bytes, "step_device_allocations": allocations}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    files = sorted(args.directory.glob("*.sqlite"))
    if not files:
        raise RuntimeError("No Nsight SQLite reports found")
    results = [inspect(path) for path in files]
    destination = args.directory / "residency_audit.json"
    destination.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
    print(f"Saved external profiler audit: {destination}")


if __name__ == "__main__":
    main()
