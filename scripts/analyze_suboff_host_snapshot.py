#!/usr/bin/env python3
"""Combine read-only per-rank SST audits and attribute residual concentrations."""
import argparse
import json
from pathlib import Path

import numpy as np


def closure_terms(cell, viscosity):
    k, omega = cell['state'][4:]
    distance = cell['wall_distance']
    gf = np.asarray(cell['flow_gradient'])
    gq = np.asarray(cell['turbulence_gradient'])
    cross = float(gq[:, 0] @ gq[:, 1])
    cd = max(2 * .856 * cross / omega, 1e-20)
    near = np.sqrt(k) / (.09 * omega * distance)
    viscous = 500 * viscosity / (distance * distance * omega)
    arg1 = min(10., min(max(near, viscous), 4 * .856 * k / (cd * distance * distance)))
    arg2 = min(10., max(2 * near, viscous))
    f1, f2 = np.tanh(arg1 ** 4), np.tanh(arg2 ** 2)
    divergence = float(np.trace(gf))
    strain = .5 * (gf + gf.T) - np.eye(3) * divergence / 3
    rotation = .5 * (gf - gf.T)
    invariant = 2 * float(np.sum(strain * strain))
    eddy = min(.31 * k / max(.31 * omega, np.sqrt(2 * np.sum(rotation * rotation)) * f2), 1e5 * viscosity)
    beta = .075 * f1 + .0828 * (1 - f1)
    gamma1 = .075 / .09 - .5 * .41 ** 2 / np.sqrt(.09)
    gamma2 = .0828 / .09 - .856 * .41 ** 2 / np.sqrt(.09)
    gamma = gamma1 * f1 + gamma2 * (1 - f1)
    terms = dict(k_production=min(max(0., eddy * invariant - 2 / 3 * k * divergence), 20 * .09 * k * omega),
        k_destruction=-.09 * k * omega, omega_production=gamma * invariant,
        omega_destruction=-beta * omega * omega, omega_cross=2 * (1 - f1) * .856 * cross / omega)
    source = [terms['k_production'] + terms['k_destruction'],
              terms['omega_production'] + terms['omega_destruction'] + terms['omega_cross']]
    dx = np.asarray(cell['gradient_stencil_displacement'])
    weight = np.asarray(cell['gradient_stencil_weight'])
    scale = np.max(np.abs(dx), axis=0)
    design = dx / scale
    design /= np.linalg.norm(design, axis=1)[:, None]
    return dict(f1=float(f1), f2=float(f2), cross=cross, divergence=divergence,
        eddy_viscosity=float(eddy), **{k: float(v) for k, v in terms.items()},
        reconstructed_source=list(map(float, source)),
        source_scaled_error=float(np.max(np.abs(np.asarray(source) - cell['source']) /
                                        np.maximum(1., np.abs(cell['source'])))),
        scaled_stencil_condition=float(np.linalg.cond(design)),
        affine_gradient_error=float(np.max(np.abs(weight.T @ dx - np.eye(3)))),
        gradient_weight_max=float(np.max(np.abs(weight))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prefix', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--viscosity', type=float, default=3.630083418e-7)
    args = parser.parse_args()
    paths = sorted(args.prefix.parent.glob(args.prefix.name + '.rank*.json'))
    reports = [json.loads(path.read_text()) for path in paths]
    if not reports or len(reports) != reports[0]['ranks']:
        raise ValueError('Incomplete set of rank audit reports')
    cells = sum(r['cells'] for r in reports)
    square = np.sum([r['equation_square_sum'] for r in reports], axis=0)
    rms = np.sqrt(square / cells)
    for report in reports:
        if report['global_cells'] != cells or not np.allclose(report['equation_rms'], rms, rtol=1e-13):
            raise ValueError('Inconsistent distributed audit norms')
    result = dict(cells=cells, ranks=len(reports), equation_rms=rms.tolist(),
        rank_audits=list(map(str, paths)), read_only=True,
        floor_cells=np.sum([r['floor_cells'] for r in reports], axis=0).tolist(),
        floor_negative_rhs_cells=np.sum([r['floor_negative_rhs_cells'] for r in reports], axis=0).tolist())
    floor_square = np.sum([r['floor_negative_rhs_square_sum'] for r in reports], axis=0)
    result['floor_negative_rhs_rms'] = np.sqrt(floor_square / cells).tolist()
    result['floor_negative_rhs_square_fraction'] = (floor_square / square[4:]).tolist()
    result['worst_cells'], result['residual_concentration'] = [], []
    for equation in range(6):
        candidates = [c for r in reports for c in r['worst_cells'][equation]]
        candidates.sort(key=lambda c: abs(c['rhs'][equation]), reverse=True)
        worst = candidates[:10]
        for cell in worst:
            cell['source_analysis'] = closure_terms(cell, args.viscosity)
            cell['equation_square_fraction'] = cell['rhs'][equation] ** 2 / square[equation]
        result['worst_cells'].append(worst)
        result['residual_concentration'].append(dict(equation=['u', 'v', 'w', 'p', 'k', 'omega'][equation],
            top1_square_fraction=worst[0]['equation_square_fraction'],
            top10_square_fraction=sum(c['equation_square_fraction'] for c in worst)))
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'worst_cells'}, indent=2))


if __name__ == '__main__':
    main()
