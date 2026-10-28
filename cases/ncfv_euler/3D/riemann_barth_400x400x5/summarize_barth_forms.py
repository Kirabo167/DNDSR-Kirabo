"""Summarize isolated Barth experiments and plot matching RP1 density slices."""
import argparse
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from test_barth_forms import metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    out = parser.parse_args().directory.resolve()
    report = json.loads((out/'results.json').read_text())
    if (out/'extra_results.json').exists():
        report['runs'] += json.loads((out/'extra_results.json').read_text())
    for row in report['runs']:
        d = out/row['problem']/row['variant']
        paths = sorted(d.glob('restart_*.dnds.h5'))
        if row['returncode'] == 0 and paths:
            row.update(metrics(paths[-1], row['problem']=='contact'))
    if (out/'uniform_results.json').exists():
        report['uniform_tests'] = json.loads((out/'uniform_results.json').read_text())
    if (out/'contact_initial_audit.json').exists():
        audit = json.loads((out/'contact_initial_audit.json').read_text())
        report['initial_contact_audit'] = {k:audit[k] for k in (
            'contact_transverse_gradient_y_z_theta_z_max',
            'protected_density_outside_neighbor_means',
            'interface_density_outside_neighbor_means')}
    if (out/'full_grid_comparison.json').exists():
        report['previous_full_grid_comparison'] = json.loads((out/'full_grid_comparison.json').read_text())
    (out/'analysis.json').write_text(json.dumps(report,indent=2)+'\n')
    names = ['baseline','pair_bounds','local_boundary','face_roe_local',
             'face_llf_local','first_order_roe']
    titles = ['Current Barth + integrated flux','Pair bounds + integrated flux',
              'Current Barth + local boundary','Barth + face Roe + local boundary',
              'Barth + face LLF + local boundary','First order Roe + local boundary']
    fig, axes = plt.subplots(2,3,figsize=(12,8),layout='constrained')
    for ax,name,title in zip(axes.flat,names,titles):
        path = sorted((out/'RP1'/name).glob('restart_*.dnds.h5'))[-1]
        with h5py.File(path) as f:
            u = f['state/father/array/data'][:].reshape(-1,5)
            ids = f['state/origIndex'][:].astype(int)
        rho = np.empty(len(ids));rho[ids]=u[:,0];rho=rho.reshape(6,41,41)
        x = np.linspace(0,1,41)
        image=ax.pcolormesh(x,x,rho[3],cmap='coolwarm',vmin=.1,vmax=1.6,shading='auto')
        ax.contour(x,x,rho[3],levels=np.linspace(.1,1.6,26)[1:-1],colors='k',linewidths=.35)
        ax.set(title=title+'\nmax z span = '+format(np.ptp(rho,axis=0).max(),'.4g'),
               xlabel='x',ylabel='y',aspect='equal')
    fig.colorbar(image,ax=list(axes.flat),label='Density',shrink=.85)
    fig.suptitle('Diagnostic RP1 tests: 40 x 40 x 5, t = 0.25, z = 0.075')
    fig.savefig(out/'RP1_forms_comparison.png',dpi=170)
    plt.close(fig)
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
