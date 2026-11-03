"""Exact conservative dual-volume means for Kurganov--Tadmor Configuration 13.

Source: Numerical Methods for Partial Differential Equations 18 (2002), p.601,
Configuration 13; reference figure 3.13 at T=0.3 on p.602.
"""
import argparse
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
PRIMITIVE_BL_BR_TL_TR = [
    [1.0625, 0.0, 0.8145, 0.0, 0.4],
    [0.5313, 0.0, 0.4276, 0.0, 0.4],
    [2.0, 0.0, 0.3, 0.0, 1.0],
    [1.0, 0.0, -0.3, 0.0, 1.0],
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source = ROOT / "cases/ncfv_euler/3D/riemann_barth_400x400x5/generate_dual_mean_initial.py"
    spec = importlib.util.spec_from_file_location("conservative_dual_initial", source)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    helper.INITIAL_STATES["KT13"] = PRIMITIVE_BL_BR_TL_TR
    mesh = args.mesh.resolve()
    helper.write_case("KT13", args.output_dir.resolve() / "KT13.csv", mesh,
                      helper.digest(mesh), helper.read_cartesian_axes(mesh))


if __name__ == "__main__":
    main()
