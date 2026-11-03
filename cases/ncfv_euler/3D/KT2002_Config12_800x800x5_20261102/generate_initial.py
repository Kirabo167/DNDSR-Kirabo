"""Exact conservative dual means for KT2002 Configuration 12, Fig.3.12, T=.25."""
import argparse
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
# BL, BR, TL, TR; rho,u,v,w,p (paper p.600).
PRIMITIVE_BL_BR_TL_TR = [
    [0.8, 0., 0., 0., 1.],
    [1., 0., 0.7276, 0., 1.],
    [1., 0.7276, 0., 0., 1.],
    [0.5313, 0., 0., 0., 0.4],
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mesh', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    source = ROOT/'cases/ncfv_euler/3D/riemann_barth_400x400x5/generate_dual_mean_initial.py'
    spec = importlib.util.spec_from_file_location('conservative_dual_initial', source)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    helper.INITIAL_STATES['KT12'] = PRIMITIVE_BL_BR_TL_TR
    mesh = args.mesh.resolve()
    helper.write_case('KT12', args.output_dir.resolve()/'KT12.csv', mesh,
                      helper.digest(mesh), helper.read_cartesian_axes(mesh))


if __name__ == '__main__':
    main()
