"""Recover a separate H5 restart from an intact saved RP1 volume snapshot.

Keep density and total conservative energy directly from the double precision
VTK arrays; recover momentum as density times velocity (roundoff only). Preserve
the native restart layout and original-node mapping from a compatible checkpoint.
Never clip a state or overwrite existing solver output.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import h5py
import numpy as np
import pyvista as pv


ROOT = Path(__file__).resolve().parents[4]
DIRECTORY = ROOT / "data/out/NCFV/riemann_barth_400x400x5/RP1"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", type=int, default=1290)
    parser.add_argument("--template", type=Path, default=DIRECTORY / "restart_00001000.dnds.h5")
    parser.add_argument("--output", type=Path, default=DIRECTORY / "recovery_1290/recovered_00001290.dnds.h5")
    args = parser.parse_args()
    master = DIRECTORY / f"solution_{args.step:08d}.pvtu"
    entry = next(e for e in json.loads((DIRECTORY / "series.pvtu.series").read_text())["files"]
                 if e["name"] == master.name)
    pieces = [master.parent / p.attrib["Source"] for p in ET.parse(master).getroot().findall("./PUnstructuredGrid/Piece")]
    if len(pieces) != 64 or args.output.exists():
        raise RuntimeError("Expected 64 intact pieces and a new output file")
    values = np.empty((964806, 5))
    seen = np.zeros(len(values), dtype=bool)
    record = {"step": args.step, "simulation_time": entry["time"],
              "source_pvtu": str(master), "template": str(args.template),
              "template_sha256": digest(args.template), "source_pieces": [],
              "momentum_recovery": "rho * Velocity; double precision roundoff only; no clipping"}
    minimum_pressure = np.inf
    maximum_pressure_error = 0.
    for piece in pieces:
        with piece.open("rb") as stream:
            stream.seek(max(0, piece.stat().st_size - 96))
            assert b"</VTKFile>" in stream.read()
        reader = pv.get_reader(piece)
        reader.disable_all_point_arrays()
        reader.disable_all_cell_arrays()
        for name in ("Density", "Pressure", "TotalEnergy", "Velocity"):
            reader.enable_point_array(name)
        mesh = reader.read()
        logical = np.rint(mesh.points * 400).astype(np.int64)
        np.testing.assert_allclose(mesh.points, logical / 400, rtol=0, atol=1e-12)
        assert np.all(logical >= 0) and np.all(logical <= [400, 400, 5])
        ids = logical[:, 0] + 401 * (logical[:, 1] + 401 * logical[:, 2])
        state = np.column_stack((mesh["Density"], mesh["Density"][:, None] * mesh["Velocity"], mesh["TotalEnergy"]))
        assert state.dtype == np.float64 and np.isfinite(state).all() and np.all(state[:, 0] > 1e-12)
        pressure = .4 * (state[:, 4] - .5 * np.sum(state[:, 1:4]**2, axis=1) / state[:, 0])
        assert np.all(pressure > 1e-12)
        np.testing.assert_allclose(pressure, mesh["Pressure"], rtol=2e-13, atol=2e-13)
        minimum_pressure = min(minimum_pressure, float(pressure.min()))
        maximum_pressure_error = max(maximum_pressure_error, float(np.max(np.abs(pressure - mesh["Pressure"]))))
        duplicate = seen[ids]
        if duplicate.any():
            np.testing.assert_allclose(values[ids[duplicate]], state[duplicate], rtol=2e-13, atol=2e-13)
        values[ids] = state
        seen[ids] = True
        record["source_pieces"].append({"path": str(piece), "sha256": digest(piece)})
    assert seen.all()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_suffix(".partial.h5")
    if partial.exists():
        raise FileExistsError(partial)
    shutil.copy2(args.template, partial)
    with h5py.File(partial, "r+") as data:
        assert data.attrs["dimension"] == 3 and data.attrs["gamma"] == 1.4
        order = data["state/origIndex"][:]
        assert len(order) == len(values) and np.array_equal(np.sort(order), np.arange(len(values)))
        data["state/father/array/data"][:] = values[order].reshape(-1)
        data.attrs.modify("iteration", args.step)
        data.attrs.modify("simulationTime", float(entry["time"]))
        data.attrs["recoverySource"] = str(master)
        data.attrs["recoveryMomentum"] = record["momentum_recovery"]
    with h5py.File(partial) as data:
        np.testing.assert_array_equal(data["state/father/array/data"][:].reshape(-1, 5), values[data["state/origIndex"][:]])
        assert data.attrs["simulationTime"] == entry["time"] and data.attrs["iteration"] == args.step
    partial.replace(args.output)
    record.update(output=str(args.output), output_sha256=digest(args.output), verified_nodes=int(seen.sum()),
                  minimum_density=float(values[:, 0].min()), minimum_pressure=minimum_pressure,
                  maximum_pressure_roundtrip_error=maximum_pressure_error)
    args.output.with_suffix(".recovery.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({k: v for k, v in record.items() if k != "source_pieces"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
