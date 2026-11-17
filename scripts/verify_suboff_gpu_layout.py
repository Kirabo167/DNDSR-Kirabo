#!/usr/bin/env python3
"""Verify exact repartitioned steady SST restarts with an asymmetric GPU rank map."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np

from remap_acm_vtkhdf_restart import remap


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
        HWLOC_COMPONENTS='-opencl,-gl,-cuda,-nvml', OMPI_MCA_pml='ob1',
        OMPI_MCA_btl='self,vader,tcp', OMPI_MCA_btl_tcp_if_include='lo',
        OMPI_MCA_oob_tcp_if_include='lo', DNDS_SUBOFF_RANK_DEVICES='0,1,1')
    environment.pop('DISPLAY', None)
    executable = root / 'build-gpu-solvers/app/acm2EQ_gpu3D.exe'
    case = root / 'cases/acm/GPU/3D/suboff/verify_SST_BDF2.json'
    overrides = {'/timeMarchSettings/integrator': 'ImplicitEulerLUSGS',
        '/timeMarchSettings/nSteps': 2, '/timeMarchSettings/steadyUntilConverged': False,
        '/timeMarchSettings/dualTimeConvergenceCriterion': 'Absolute',
        '/timeMarchSettings/dualTimeFlowRelativeTolerance': 0.,
        '/timeMarchSettings/dualTimeTurbulenceRelativeTolerance': 0.,
        '/timeMarchSettings/dualTimeRelativeTolerances': [0.] * 6,
        '/timeMarchSettings/useCFLTimeStep': True, '/timeMarchSettings/useLocalTimeStep': True,
        '/timeMarchSettings/cfl': .5, '/timeMarchSettings/implicitRelaxation': .3,
        '/timeMarchSettings/maxImplicitIterations': 1,
        '/timeMarchSettings/lusgsSweeps': 3, '/timeMarchSettings/steadyLineSearch': False,
        '/outputSettings/interval': 1, '/outputSettings/writeInitial': True,
        '/outputSettings/prefix': 'verify'}

    def run(name, ranks, cuda, extra):
        directory = output / name
        directory.mkdir(exist_ok=False)
        command = ['mpirun', '--bind-to', 'none', '-np', str(ranks)]
        command += ([sys.executable, str(root / 'scripts/run_acm_gpu_rank.py'),
                     str(executable), str(case)] if cuda else
                    [str(executable), str(case), '--backend', 'host'])
        for key, value in {**overrides, '/outputSettings/directory': str(directory), **extra}.items():
            command += ['-k', key, '-v', json.dumps(value)]
        with (directory / 'run.log').open('w') as log:
            process = subprocess.Popen(command, cwd=root / 'build-gpu-solvers', env=environment,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = process.wait(timeout=120)
            except subprocess.TimeoutExpired:
                import signal
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                raise
        if code:
            raise RuntimeError(str(directory / 'run.log'))
        return directory

    seed = run('seed_np1', 1, False, {})
    template = run('template_np3', 3, False, {'/timeMarchSettings/nSteps': 0})
    mapped = output / 'restart_np3.vtkhdf'
    remap(seed / 'verify_00000002.vtkhdf', template / 'verify_00000000.vtkhdf', mapped)
    restart = {'/restartSettings/flowFile': str(mapped), '/restartSettings/writerRanks': 3,
               '/restartSettings/completedSteps': 2, '/restartSettings/requireTurbulence': True}
    host = run('resume_host_np3', 3, False, restart)
    cuda = run('resume_cuda_np3', 3, True, restart)
    maximum, checks = 0., 0
    files = sorted(host.glob('*.vtkhdf'))
    if len(files) != 3:
        raise RuntimeError('Expected imported state and two steady iterations')
    for field in files:
        with h5py.File(field) as a, h5py.File(cuda / field.name) as b:
            for key in a['VTKHDF/CellData']:
                x = a['VTKHDF/CellData'][key][()]
                y = b['VTKHDF/CellData'][key][()]
                if not np.isfinite(y).all() or ('Turbulence' in key and np.any(y <= 0)):
                    raise RuntimeError('Invalid field: ' + key)
                maximum = max(maximum, float(np.max(np.abs(x - y) / np.maximum(1., np.abs(x)))))
                checks += 1
    with h5py.File(mapped) as a, h5py.File(cuda / 'verify_00000002.vtkhdf') as b:
        for key in a['VTKHDF/CellData']:
            if not np.array_equal(a['VTKHDF/CellData'][key][()], b['VTKHDF/CellData'][key][()]):
                raise RuntimeError('Import changed ' + key)
    log = (cuda / 'run.log').read_text()
    if 'Resident CUDA' not in log or 'device 0' not in log or log.count('device 1') < 2:
        raise RuntimeError('GPU rank mapping was not established')
    summary = dict(passed=maximum < 5e-8, rank_devices=[0, 1, 1], steady_local_time_step=True,
        restart_np1_to_np3_exact=True, initial_import_exact=True,
        compared_field_arrays=checks, maximum_scaled_host_cuda_error=maximum)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary), flush=True)
    if not summary['passed']:
        raise RuntimeError(summary)


if __name__ == '__main__':
    main()
