"""Verify the full production t=0 state against exact conservative dual means."""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pyvista as pv

from generate_dual_mean_initial import INITIAL_STATES, GAMMA, median_overlap


ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "data/out/NCFV/riemann_barth_400x400x5"


def verify(name):
    directory = OUTPUT / name
    master = directory / "solution_00000000.pvtu"
    pieces = [master.parent / p.attrib["Source"] for p in
              ET.parse(master).getroot().findall("./PUnstructuredGrid/Piece")]
    if len(pieces) != 64:
        raise RuntimeError(f"Expected 64 pieces, found {len(pieces)}")
    primitive = np.array(INITIAL_STATES[name])
    conservative = primitive.copy()
    conservative[:, 1:4] *= primitive[:, :1]
    conservative[:, 4] = primitive[:, 4] / (GAMMA - 1) + .5 * primitive[:, 0] * np.sum(primitive[:, 1:4]**2, axis=1)
    left, _ = median_overlap(400)
    seen_nodes = np.zeros(964806, dtype=bool)
    seen_cells = np.zeros(800000, dtype=bool)
    errors = np.zeros(5)
    for piece in pieces:
        with piece.open("rb") as stream:
            stream.seek(max(0, piece.stat().st_size - 96))
            if b"</VTKFile>" not in stream.read():
                raise RuntimeError(f"Incomplete VTU: {piece}")
        reader = pv.get_reader(piece)
        reader.disable_all_point_arrays()
        reader.disable_all_cell_arrays()
        for field in ("Density", "Pressure", "Velocity"):
            reader.enable_point_array(field)
        mesh = reader.read()
        logical = np.rint(mesh.points * 400).astype(np.int64)
        np.testing.assert_allclose(mesh.points, logical / 400, atol=1e-12, rtol=0)
        assert np.all(logical >= 0) and np.all(logical <= [400, 400, 5])
        node_ids = logical[:, 0] + 401 * (logical[:, 1] + 401 * logical[:, 2])
        seen_nodes[node_ids] = True
        fx, fy = left[logical[:, 0]], left[logical[:, 1]]
        weights = np.column_stack((fx*fy, (1-fx)*fy, fx*(1-fy), (1-fx)*(1-fy)))
        average = weights @ conservative
        expected = average.copy()
        expected[:, 1:4] /= average[:, :1]
        expected[:, 4] = (GAMMA - 1) * (average[:, 4] - .5 * np.sum(average[:, 1:4]**2, axis=1) / average[:, 0])
        actual = np.column_stack((mesh["Density"], mesh["Velocity"], mesh["Pressure"]))
        assert np.isfinite(actual).all() and np.all(actual[:, [0, 4]] > 0)
        errors = np.maximum(errors, np.max(np.abs(actual - expected), axis=0))
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
        assert np.all(mesh.celltypes == 12)
        corners = logical[mesh.cells_dict[12]]
        lo, hi = corners.min(axis=1), corners.max(axis=1)
        assert np.all(hi - lo == [1, 1, 1])
        ids = lo[:, 0] + 400 * (lo[:, 1] + 400 * lo[:, 2])
        assert len(np.unique(ids)) == len(ids) and not seen_cells[ids].any()
        seen_cells[ids] = True
    assert seen_nodes.all() and seen_cells.all()
    result = {"case": name, "verified_nodes": int(seen_nodes.sum()),
              "verified_cells": int(seen_cells.sum()), "pieces": len(pieces),
              "maximum_absolute_error_rho_u_v_w_p": errors.tolist(),
              "exact_conservative_dual_mean_initialization_verified": True}
    (directory / "initial_validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", choices=list(INITIAL_STATES), default=list(INITIAL_STATES))
    for name in parser.parse_args().cases:
        verify(name)
