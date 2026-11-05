"""Run controlled initial-RHS and polynomial consistency checks on prism meshes."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[4]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--levels", nargs="+", type=int, default=[10, 20, 40, 80])
    parser.add_argument("--problem", choices=["entropy", "entropy_xy"], default="entropy")
    parser.add_argument("--ranks", type=int, default=8)
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()
    raw = args.output.resolve() / "raw"; raw.mkdir(parents=True, exist_ok=True)
    binary = args.binary.resolve(); digest = hashlib.sha256(binary.read_bytes()).hexdigest()
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               HWLOC_COMPONENTS="-gl,-cuda,-nvml", HOCFV_AUDIT_RHS="1")

    def run(level):
        mesh = ROOT / f"cases/ncfv_euler/3D/iv{level}_3d_1_dnds.cgns"
        path = raw / f"{args.problem}_iv{level}.json"
        manifest_path = path.with_suffix(".manifest.json")
        command = ["mpirun", "--mca", "pml", "ob1", "--mca", "btl", "self,vader", "--bind-to", "none",
                   "-np", str(args.ranks), str(binary), str(mesh), str(path), args.problem, "0.01", "0.01", "30"]
        manifest = {"command": command, "binary_sha256": digest, "rhs_audit": True,
                    "mesh_sha256": hashlib.sha256(mesh.read_bytes()).hexdigest()}
        if path.exists():
            if not manifest_path.exists() or json.loads(manifest_path.read_text()) != manifest:
                raise RuntimeError(f"Different existing audit: {path}")
        else:
            if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
                raise RuntimeError(f"Different pending audit: {path}")
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
            print(f"Start initial RHS {args.problem} IV{level}", flush=True)
            with path.with_suffix(".log").open("w") as log:
                subprocess.run(command, cwd=ROOT / "build", env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        data = json.loads(path.read_text())
        print(f"Done IV{level}: midpoint={data['schemes']['vr_midpoint']['all']['L2']:.7e}, "
              f"exact surface={data['schemes']['exact_surface6']['all']['L2']:.7e}", flush=True)
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        list(pool.map(run, args.levels))


if __name__ == "__main__":
    main()
