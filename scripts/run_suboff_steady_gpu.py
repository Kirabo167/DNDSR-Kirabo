#!/usr/bin/env python3
"""Run resident steady SST on one GPU; completion requires verified convergence."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = re.compile(r'ACM resident steady reference eqInitial=(\[[^\n]+?\]) eqTarget=(\[[^\n]+?\]) window=(\d+)')
ITERATION = re.compile(r'ACM resident steady iteration (\d+) residual (\S+) -> (\S+) dtauMin=(\S+) turbResidual=(\S+) cfl=(\S+) nextCFL=(\S+) relaxation=(\S+) accepted=(\d+) stable=(\d+) converged=(\d+) eqResidual=(\[[^\n]+\])')


def stamp():
    return datetime.now(timezone.utc).isoformat()


def audit_field(path):
    fields = {}
    with h5py.File(path) as handle:
        cells = len(handle['VTKHDF/CellData/Pressure'])
        if cells != 3914184:
            raise ValueError(f'Unexpected SUBOFF mesh size: {cells}')
        for name, data in handle['VTKHDF/CellData'].items():
            low, high = float('inf'), -float('inf')
            for begin in range(0, len(data), 100000):
                values = data[begin:begin+100000]
                if not np.isfinite(values).all() or ('Turbulence' in name and np.any(values <= 0)):
                    raise ValueError(f'Invalid saved field: {name}')
                low, high = min(low, float(values.min())), max(high, float(values.max()))
            fields[name] = dict(minimum=low, maximum=high, all_finite=True)
    return dict(file=str(path), cells=cells, fields=fields)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu-device', default='1')
    parser.add_argument('--detach', action='store_true')
    args = parser.parse_args()
    configuration, directory = args.config.resolve(), args.output.resolve()
    config = json.loads(configuration.read_text())
    settings = config['timeMarchSettings']
    if not (settings['integrator'] == 'ImplicitEulerLUSGS' and settings['steadyUntilConverged']
            and settings['useCFLTimeStep'] and settings['useLocalTimeStep']
            and config['turbulenceSettings']['model'] == 'KOmegaSST'
            and config['restartSettings']['writerRanks'] == 1):
        raise ValueError('Expected steady SST with convergence stopping, local steps and a single-rank restart')
    if (ROOT/'build-gpu-solvers'/config['outputSettings']['directory']).resolve() != directory/'fields':
        raise ValueError('The output directory does not match the solver configuration')
    directory.mkdir(parents=True, exist_ok=True)
    if (directory/'status.json').exists() or any((directory/'fields').glob('*.vtkhdf')):
        raise FileExistsError('Choose a fresh steady output directory; existing results are preserved')
    if args.detach:
        with (directory/'worker.log').open('a') as log:
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--config', str(configuration),
                       '--output', str(directory), '--gpu-device', args.gpu_device], cwd=ROOT,
                       stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps(dict(worker_pid=process.pid, directory=str(directory))))
        return
    executable = ROOT/'build-gpu-solvers/app/acm2EQ_gpu3D.exe'
    command = ['mpirun','--bind-to','none','-np','1',str(executable),str(configuration),
               '--backend','cuda','--gpu-device',args.gpu_device]
    restart = config['restartSettings']
    state = dict(mode='steady', state='starting', worker_pid=os.getpid(), created_utc=stamp(),
                 config=str(configuration), executable=str(executable),
                 executable_sha256=hashlib.file_digest(executable.open('rb'),'sha256').hexdigest(),
                 ranks=1, gpu_device=args.gpu_device, until_converged=True, iteration_limit=None,
                 initial_iteration=restart['completedSteps'], completed_iteration=restart['completedSteps'],
                 completed_steady_iterations=0, physical_time=restart['physicalTime'],
                 physical_time_advancing=False, converged=False,
                 relative_tolerance=settings['steadyConvergenceRelativeTolerance'],
                 absolute_tolerances=settings['steadyConvergenceAbsoluteTolerances'],
                 convergence_window=settings['steadyConvergenceWindow'],
                 line_search_window=settings.get('steadyLineSearchWindow',20),
                 local_time_step=True, communication='one rank; no halo exchange', command=command,
                 log=str(directory/'run.log'), residual_history=str(directory/'residual_history.csv'))
    start = time.monotonic()

    def persist():
        state.update(updated_utc=stamp(), elapsed_seconds=time.monotonic()-start)
        temporary = directory/'status.json.tmp'
        temporary.write_text(json.dumps(state,indent=2,allow_nan=False)+'\n')
        temporary.replace(directory/'status.json')

    persist()
    environment = dict(os.environ, OMP_NUM_THREADS='8', OPENBLAS_NUM_THREADS='1', HWLOC_COMPONENTS='-opencl',
                       OMPI_MCA_pml='ob1', OMPI_MCA_btl='self,vader,tcp', OMPI_MCA_btl_tcp_if_include='lo',
                       OMPI_MCA_oob_tcp_if_include='lo')
    environment.pop('DISPLAY', None)
    position, pending, count, last_observed = 0, '', 0, None
    final_window = []
    exit_seen = False
    try:
        with (directory/'run.log').open('w') as log, (directory/'residual_history.csv').open('w') as history:
            writer = csv.writer(history)
            writer.writerow(['iteration','elapsed_seconds','flow_residual','turbulence_residual','cfl','next_cfl',
                             'accepted_relaxation','accepted','stable','converged','u','v','w','p','k','omega'])
            process = subprocess.Popen(command,cwd=ROOT/'build-gpu-solvers',env=environment,
                       stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            state['solver_pid'] = process.pid
            persist()
            while True:
                previous_iteration = state['completed_iteration']
                with (directory/'run.log').open() as reader:
                    reader.seek(position)
                    chunk = reader.read()
                    position = reader.tell()
                pending += chunk
                lines = pending.split('\n')
                pending = lines.pop()
                for line in lines:
                    reference = REFERENCE.search(line)
                    if reference:
                        state.update(initial_equation_residuals=json.loads(reference[1]),
                                     target_equation_residuals=json.loads(reference[2]))
                    match = ITERATION.search(line)
                    if not match:
                        if line and state['state']=='starting': state['state']='initializing'
                        continue
                    step = int(match[1]); norms = json.loads(match[12]); count += 1
                    now = time.monotonic()
                    converged = bool(int(match[11]))
                    state.update(state='saving_converged_result' if converged else 'running',
                         completed_iteration=step, completed_steady_iterations=step-restart['completedSteps'],
                         flow_residual=float(match[3]), turbulence_residual=float(match[5]),
                         minimum_local_time_step=float(match[4]), cfl=float(match[6]), next_cfl=float(match[7]),
                         accepted_relaxation=float(match[8]), update_accepted=bool(int(match[9])),
                         stable_checks=int(match[10]), converged=converged, equation_residuals=norms)
                    final_window.append(norms)
                    final_window = final_window[-settings['steadyConvergenceWindow']:]
                    writer.writerow([step,now-start,float(match[3]),float(match[5]),float(match[6]),float(match[7]),
                                     float(match[8]),int(match[9]),int(match[10]),int(match[11]),*norms])
                history.flush()
                if state['completed_iteration'] > previous_iteration:
                    now = time.monotonic()
                    if last_observed is not None:
                        state['last_observed_seconds_per_iteration'] = (now-last_observed)/(state['completed_iteration']-previous_iteration)
                    last_observed = now
                persist()
                if process.poll() is not None:
                    if exit_seen: break
                    exit_seen = True
                    continue
                time.sleep(2)
            state['exit_code'] = process.wait()
        if state['exit_code'] != 0 or not state['converged']:
            raise RuntimeError('Steady solver exited without verified spatial convergence')
        targets = np.asarray(state['target_equation_residuals'])
        if (len(final_window) != settings['steadyConvergenceWindow'] or
                not np.all(np.asarray(final_window) <= targets)):
            raise RuntimeError('The complete consecutive convergence window was not verified')
        final = directory/'fields'/f"{config['outputSettings']['prefix']}_{state['completed_iteration']:08d}.vtkhdf"
        audit = audit_field(final)
        audit.update(steady_convergence_verified=True, target_equation_residuals=targets.tolist(),
                     final_equation_residuals=state['equation_residuals'],
                     consecutive_checks_verified=len(final_window), accuracy_verified=False)
        (directory/'final_field_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
        state.update(state='completed', final_field_audit=audit, completion_reason='all six spatial residual targets held for the full convergence window')
    except Exception as error:
        state.update(state='failed',error=str(error),converged=False)
        raise
    finally:
        persist()


if __name__ == '__main__':
    main()
