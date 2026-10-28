"""Compare completed midpoint RP1 with the earlier full-resolution result."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/ncfv-midpoint-comparison-matplotlib')
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from run_midpoint_barth import ROOT, digest, save


def read(path):
    with h5py.File(path) as data:
        u = data['state/father/array/data'][:].reshape(-1, 5)
        ids = data['state/origIndex'][:].astype(np.int64)
        actual_time = float(data.attrs['simulationTime'])
        iteration = int(data.attrs['iteration'])
        raw = data.attrs['configuration']
        config = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
    if not np.array_equal(np.sort(ids), np.arange(964806)):
        raise ValueError('Expected complete 400x400x5 output')
    ordered = np.empty_like(u)
    ordered[ids] = u
    cube = ordered.reshape(6, 401, 401, 5)
    rho = cube[..., 0]
    p = .4 * (cube[..., 4] - .5 * np.sum(cube[..., 1:4]**2, axis=-1) / rho)
    w = cube[..., 3] / rho
    if not np.isfinite(cube).all() or rho.min() <= 0 or p.min() <= 0:
        raise ValueError('Nonphysical state')
    metrics = {'time': actual_time, 'iteration': iteration,
        'rho_min': float(rho.min()), 'rho_max': float(rho.max()),
        'p_min': float(p.min()), 'p_max': float(p.max()),
        'rho_z_span_max': float(np.ptp(rho, axis=0).max()),
        'p_z_span_max': float(np.ptp(p, axis=0).max()),
        'w_abs_max': float(np.abs(w).max()),
        'rho_z_TV_excess_max': float(np.maximum(0, np.abs(np.diff(rho, axis=0)).sum(axis=0)-np.ptp(rho, axis=0)).max()),
        'restart': str(path), 'restart_sha256': digest(path)}
    return metrics, (rho, p, w), config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    status = json.loads((out/'status.json').read_text())
    if status['status'] != 'complete':
        raise ValueError('The full midpoint calculation is not complete')
    baseline = ROOT/'data/out/NCFV/riemann_barth_400x400x5_positive_tol1e10'
    old_status = json.loads((baseline/'pipeline_status.json').read_text())
    old, old_fields, old_config = read(Path(old_status['restart']))
    new, new_fields, new_config = read(Path(status['full_grid']['restart']))
    if old['time'] != .25 or new['time'] != .25:
        raise ValueError('Comparison requires t=.25')
    for section in ('physics', 'time'):
        for key in (('gamma', 'riemannSolver', 'boundaryZones') if section == 'physics' else
                    ('cfl', 'maximumTimeStep', 'endTime', 'rejectNonPhysicalSteps')):
            if old_config[section][key] != new_config[section][key]:
                raise ValueError(f'Mismatched {section}/{key}')
    inputs = json.loads((out/'input_manifest.json').read_text())
    for ending, key in [('riemann_barth_400x400x5.cgns', 'mesh_sha256'), ('RP1.csv', 'initial_sha256')]:
        item = next(value for path, value in inputs.items() if path.endswith(ending))
        if item['sha256'] != old_status[key]:
            raise ValueError('Different mesh or initial condition')
    result = {'original_efficient': old, 'quadratic_midpoint_barth': new,
        'comparison': 'Same mesh, initial means, boundary conditions, Roe solver, CFL, SSPRK3 and end time. Reconstruction/integration and boundary sampling follow their respective methods.',
        'relative_change': {key: new[key]/old[key]-1 for key in ('rho_z_span_max', 'p_z_span_max', 'w_abs_max', 'rho_z_TV_excess_max')}}
    save(out/'comparison.json', result)
    fig, axes = plt.subplots(2, 4, figsize=(21, 10), constrained_layout=True)
    titles = ['Original efficient integration + Barth', 'Quadratic edge midpoint + Barth']
    for row, (fields, title) in enumerate(zip((old_fields, new_fields), titles)):
        rho, p, w = fields
        for col, (field, name, lower, upper) in enumerate(zip(
                (rho[3], p[3], w[3], np.ptp(rho, axis=0)),
                ('Density', 'Pressure', 'Spanwise velocity w', 'Density span over z'),
                (.1, .02, -max(np.abs(old_fields[2]).max(), np.abs(new_fields[2]).max()), 0),
                (max(old_fields[0].max(),new_fields[0].max()), max(old_fields[1].max(),new_fields[1].max()),
                 max(np.abs(old_fields[2]).max(),np.abs(new_fields[2]).max()),
                 max(np.ptp(old_fields[0],axis=0).max(),np.ptp(new_fields[0],axis=0).max())))):
            ax = axes[row, col]
            im = ax.imshow(field, origin='lower', extent=[0,1,0,1], cmap='coolwarm' if col<3 else 'magma', vmin=lower, vmax=upper)
            if col<2:
                ax.contour(np.linspace(0,1,401),np.linspace(0,1,401),field,levels=np.linspace(lower,upper,27)[1:-1],colors='k',linewidths=.25)
            ax.set_title(f'{title}\n{name}');fig.colorbar(im,ax=ax,shrink=.8)
    fig.suptitle('RP1, 400 x 400 x 5, t=0.25; common color scales; slice at z=0.0075')
    fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
