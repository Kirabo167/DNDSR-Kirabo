"""Collect completed checkpoint diagnostics while a single midpoint run proceeds."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import matplotlib.pyplot as plt

from run_midpoint_barth import save
from summarize_midpoint_barth import read


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    history_path = out/'checkpoint_metrics.json'
    history = json.loads(history_path.read_text()) if history_path.exists() else []
    known = {row['iteration'] for row in history}
    while True:
        status = json.loads((out/'status.json').read_text())
        changed = False
        for path in sorted((out/'RP1_400_np64').glob('restart_*.dnds.h5')):
            iteration = int(path.name.split('_')[1].split('.')[0])
            if iteration in known:
                continue
            try:
                metrics, _, _ = read(path)
            except (OSError, KeyError, ValueError):
                # The solver creates the final filename before closing HDF5.
                continue
            history.append(metrics)
            known.add(iteration)
            changed = True
            print(json.dumps(metrics), flush=True)
        if changed:
            history.sort(key=lambda row: row['iteration'])
            save(history_path, history)
            fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
            for ax, key in zip(axes.flat, ('rho_z_span_max', 'p_z_span_max', 'w_abs_max', 'rho_z_TV_excess_max')):
                ax.plot([row['time'] for row in history], [row[key] for row in history], '.-')
                ax.set_xlabel('Physical time');ax.set_ylabel(key);ax.grid(alpha=.3)
            fig.suptitle('RP1, 400 x 400 x 5: edge-midpoint Roe + Barth')
            fig.savefig(out/'oscillation_history.png', dpi=150);plt.close(fig)
        if status['status'] == 'complete':
            subprocess.run([sys.executable, str(Path(__file__).with_name('summarize_midpoint_barth.py')), str(out)], check=True)
            break
        if status['status'] == 'failed':
            raise RuntimeError(status.get('error', 'Solver failed'))
        time.sleep(20)


if __name__ == '__main__':
    main()
