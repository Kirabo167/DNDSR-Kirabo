#!/usr/bin/env python3
"""Compare completed NCFV Taylor–Green runs with Re=1600 DNS statistics.

Input directories are <results>/hex<N>/<mode>/solution_*.pvtu.  Only a VTK
series whose final recorded time is exactly the requested time is accepted.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy


MODES = ("efficient", "traditional")
NU = 1.0 / 1600.0


def final_output(folder: Path, target_time: float) -> Path:
    series_path = folder / "solution.pvtu.series"
    series = json.loads(series_path.read_text(encoding="utf-8"))
    entries = series["files"]
    if not entries or abs(float(entries[-1]["time"]) - target_time) > 1e-10:
        raise ValueError(f"{series_path}: final time is not {target_time}")
    path = folder / entries[-1]["name"]
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def periodic_fields(path: Path, n: int) -> tuple[np.ndarray, np.ndarray]:
    reader = vtk.vtkXMLPUnstructuredGridReader()
    reader.SetFileName(str(path))
    reader.Update()
    mesh = reader.GetOutput()
    coordinates = vtk_to_numpy(mesh.GetPoints().GetData())
    density = vtk_to_numpy(mesh.GetPointData().GetArray("Density"))
    velocity = vtk_to_numpy(mesh.GetPointData().GetArray("Velocity"))
    h = 2.0 * math.pi / n
    integer = np.rint((coordinates + math.pi) / h).astype(np.int64)
    if not np.allclose(coordinates, -math.pi + integer * h, atol=1e-9):
        raise ValueError(f"{path}: output points do not match the Cartesian grid")
    integer %= n
    linear = integer[:, 0] + n * (integer[:, 1] + n * integer[:, 2])
    counts = np.bincount(linear, minlength=n**3)
    if np.any(counts == 0):
        raise ValueError(f"{path}: missing periodic grid points")
    rho = np.bincount(linear, weights=density, minlength=n**3) / counts
    u = np.column_stack([
        np.bincount(linear, weights=velocity[:, component], minlength=n**3) / counts
        for component in range(3)
    ])
    return rho.reshape((n, n, n)), u.reshape((n, n, n, 3))


def statistics(rho: np.ndarray, velocity: np.ndarray) -> dict[str, float]:
    n = rho.shape[0]
    speed2 = np.sum(velocity * velocity, axis=-1)
    wave = np.fft.fftfreq(n, d=1.0 / n)
    transformed = np.fft.fftn(velocity, axes=(0, 1, 2))
    derivatives = [
        np.fft.ifftn(1j * wave.reshape(
            [n if i == axis else 1 for i in range(3)] + [1]) * transformed,
            axes=(0, 1, 2)).real
        for axis in range(3)
    ]
    curl = np.stack((derivatives[1][..., 2] - derivatives[2][..., 1],
                     derivatives[2][..., 0] - derivatives[0][..., 2],
                     derivatives[0][..., 1] - derivatives[1][..., 0]), axis=-1)
    enstrophy = float(0.5 * np.mean(np.sum(curl * curl, axis=-1)))
    return {
        "kinetic_energy": float(0.5 * np.mean(speed2)),
        "density_weighted_kinetic_energy": float(0.5 * np.mean(rho * speed2)),
        "enstrophy": enstrophy,
        "enstrophy_dissipation": 2.0 * NU * enstrophy,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=root / "data/out/NCFV/taylor_green/t8")
    parser.add_argument("--reference", type=Path,
                        default=root / "data/reference/taylor_green/TGV_Re1600.dat")
    parser.add_argument("--time", type=float, default=8.0)
    parser.add_argument("--sizes", type=int, nargs="+", default=[8, 16, 32, 64])
    args = parser.parse_args()
    reference = np.loadtxt(args.reference)
    row = reference[np.argmin(np.abs(reference[:, 0] - args.time))]
    if abs(row[0] - args.time) > 1e-10:
        parser.error("DNS reference does not contain the requested time")
    output: list[dict[str, float | str | int]] = []
    for n in args.sizes:
        fields: dict[str, np.ndarray] = {}
        for mode in MODES:
            path = final_output(args.results / f"hex{n}" / mode, args.time)
            rho, velocity = periodic_fields(path, n)
            values = statistics(rho, velocity)
            values.update({
                "n": n, "mode": mode, "time": args.time,
                "energy_reference": float(row[1]),
                "enstrophy_reference": float(row[4]),
                "dissipation_reference": float(row[2]),
                "energy_abs_error": abs(values["kinetic_energy"] - row[1]),
                "energy_rel_error": abs(values["kinetic_energy"] - row[1]) / abs(row[1]),
                "enstrophy_rel_error": abs(values["enstrophy"] - row[4]) / abs(row[4]),
            })
            output.append(values)
            fields[mode] = velocity
        delta = fields["efficient"] - fields["traditional"]
        rms = float(np.sqrt(np.mean(np.sum(delta * delta, axis=-1))))
        for values in output[-2:]:
            values["mode_velocity_rms_difference"] = rms
    args.results.mkdir(parents=True, exist_ok=True)
    csv_path = args.results / "accuracy.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    print(f"DNS at t={args.time}: E={row[1]:.12g}, enstrophy={row[4]:.12g}, "
          f"epsilon={row[2]:.12g}")
    for values in output:
        print(f"{values['n']:>3} {values['mode']:<11} "
              f"E={values['kinetic_energy']:.9g} "
              f"E_rel={values['energy_rel_error']:.3%} "
              f"Z={values['enstrophy']:.9g} "
              f"Z_rel={values['enstrophy_rel_error']:.3%}")
    print(csv_path)


if __name__ == "__main__":
    main()
