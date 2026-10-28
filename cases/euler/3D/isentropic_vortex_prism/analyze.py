"""Read CFV's reconstructed-density IV error from completed Prism6 runs."""

import argparse
import base64
import json
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np


ROOT = Path(__file__).resolve().parents[4]
ANSI = re.compile(r"\x1b\[[0-9;]*m")
STEP = re.compile(r"=== Step\s*\[\s*(\d+)\].*?t,dT,dTaumin,CFL,nFix\s*\[([\d.eE+-]+)")
ERROR = re.compile(r"=== Mean Error IV: \[([\d.eE+-]+),\s*([\d.eE+-]+),\s*([\d.eE+-]+)\]")


def check_extrusion(n, time):
    vtk = ROOT / f"data/out/CFV/iv_prism{n}/solution__t_{time:.6f}.vtu"
    piece = ET.parse(vtk).getroot().find(".//Piece")

    def binary_float(node):
        raw = base64.b64decode(node.text.strip())
        count = int.from_bytes(raw[:8], "little")
        return np.frombuffer(raw[8:8 + count], dtype="<f8")

    points = binary_float(piece.find("Points/DataArray")).reshape(-1, 3)
    connectivity = np.fromstring(piece.find("Cells/DataArray[@Name='connectivity']").text,
                                 sep=" ", dtype=np.int64).reshape(-1, 6)
    xy = np.round(points[connectivity].mean(axis=1)[:, :2], 10)
    density = binary_float(piece.find("CellData/DataArray[@Name='R']"))
    velocity = binary_float(piece.find("CellData/DataArray[@Name='Velo']")).reshape(-1, 3)
    _, groups = np.unique(xy, axis=0, return_inverse=True)
    low = np.full(groups.max() + 1, np.inf)
    high = np.full(groups.max() + 1, -np.inf)
    np.minimum.at(low, groups, density)
    np.maximum.at(high, groups, density)
    variation = float(np.max(high - low))
    vertical_velocity = float(np.max(np.abs(velocity[:, 2])))
    if variation > 1e-11 or vertical_velocity > 1e-11:
        raise RuntimeError(f"iv{n}: expected z invariance, got density spread {variation}, |w| {vertical_velocity}")
    return variation, vertical_velocity


def read_run(n):
    case = Path(__file__).with_name(f"iv{n}.json")
    config = json.loads(case.read_text())
    manifest = json.loads((ROOT / f"cases/ncfv_euler/3D/iv{n}_3d_1_dnds.manifest.json").read_text())
    log = ROOT / f"data/out/CFV/iv_prism{n}/run.logstr.txt"
    contents = ANSI.sub("", log.read_text())
    steps = list(STEP.finditer(contents))
    errors = list(ERROR.finditer(contents))
    if not steps or not errors:
        raise RuntimeError(f"iv{n}: missing final time or density error in {log}")
    step, time = int(steps[-1][1]), float(steps[-1][2])
    expected_steps = config["timeMarchControl"]["nTimeStep"]
    expected_time = config["timeMarchControl"]["tEnd"]
    if step != expected_steps or not math.isclose(time, expected_time, rel_tol=0, abs_tol=1e-12):
        raise RuntimeError(f"iv{n}: incomplete run: step={step}, time={time}")
    integrated, mean, maximum = map(float, errors[-1].groups())
    if not math.isclose(integrated / 400.0, mean, rel_tol=2e-7):
        raise RuntimeError(f"iv{n}: error normalization does not match domain volume 400")
    z_density_spread, max_z_velocity = check_extrusion(n, time)
    return {"mesh": f"iv{n}", "N": n, "cells": manifest["cells"], "h_nominal": 10 / n,
            "steps": step, "t": time, "rho_L1_mean": mean, "rho_L1_integrated": integrated,
            "rho_Linf": maximum, "z_density_spread": z_density_spread,
            "max_z_velocity": max_z_velocity, "log": str(log)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meshes", type=int, nargs="+", default=[10, 20, 40])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = [read_run(n) for n in args.meshes]
    for before, after in zip(rows, rows[1:]):
        ratio = after["N"] / before["N"]
        after["observed_order_L1"] = math.log(before["rho_L1_mean"] / after["rho_L1_mean"]) / math.log(ratio)
        after["observed_order_Linf"] = math.log(before["rho_Linf"] / after["rho_Linf"]) / math.log(ratio)
    print("mesh     cells      h_nominal     rho_L1_mean     p_L1       rho_Linf      p_Linf      z_spread       max_|w|")
    for row in rows:
        print(f"{row['mesh']:<8} {row['cells']:>6d} {row['h_nominal']:>12.6f} {row['rho_L1_mean']:>15.8e} "
              f"{row.get('observed_order_L1', float('nan')):>8.3f} "
              f"{row['rho_Linf']:>15.8e} {row.get('observed_order_Linf', float('nan')):>8.3f} "
              f"{row['z_density_spread']:>13.3e} {row['max_z_velocity']:>13.3e}")
    if args.output:
        args.output.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
