#!/usr/bin/env python3
"""Validate resident steady SST against host updates and BDF2 regression fields.

Only invokes solver binaries and reads HDF5; no DNDSR Python bindings are used.
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess

import h5py
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference-executable', type=Path,
                        help='Optional prior executable with the same spatial/implicit operator')
    parser.add_argument('--gpu-device', default='1')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    executable = root/'build-gpu-solvers/app/acm2EQ_gpu3D.exe'
    case = root/'cases/acm/GPU/3D/suboff/verify_SST_BDF2.json'
    environment = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', HWLOC_COMPONENTS='-opencl',
                       OMPI_MCA_pml='ob1', OMPI_MCA_btl='self,vader,tcp', OMPI_MCA_btl_tcp_if_include='lo',
                       OMPI_MCA_oob_tcp_if_include='lo')
    environment.pop('DISPLAY', None)
    summary = []

    def run(name, backend, overrides, exe=executable):
        directory = output/name
        directory.mkdir()
        command = ['mpirun','--bind-to','none','-np','1',str(exe),str(case),
                   '--backend',backend,'--gpu-device',args.gpu_device]
        for key, value in {'/outputSettings/directory':str(directory),**overrides}.items():
            command += ['-k',key,'-v',json.dumps(value)]
        with (directory/'run.log').open('w') as log:
            subprocess.run(command,cwd=root/'build-gpu-solvers',env=environment,stdout=log,
                           stderr=subprocess.STDOUT,check=True,timeout=180)
        series=json.loads(next(directory.glob('*.series')).read_text())['files']
        fields=[]
        for entry in series:
            with h5py.File(directory/entry['name']) as handle:
                values={key:dataset[()] for key,dataset in handle['VTKHDF/CellData'].items()}
                for key,value in values.items():
                    if not np.isfinite(value).all() or ('Turbulence' in key and np.any(value<=0)):
                        raise ValueError(f'Invalid {key} in {name}')
                fields.append(values)
        return fields,(directory/'run.log').read_text()

    def compare(name, reference, actual):
        if len(reference)!=len(actual): raise ValueError('Saved state count differs')
        maximum=0.
        for expected,field in zip(reference,actual):
            if expected.keys()!=field.keys(): raise ValueError('Field names differ')
            for key,values in field.items():
                maximum=max(maximum,float(np.max(np.abs(values-expected[key])/np.maximum(1.,np.abs(expected[key])))))
        if maximum>5e-8: raise ValueError(f'{name}: scaled error {maximum}')
        summary.append(dict(case=name,maximum_scaled_error=maximum,passed=True))

    if args.reference_executable:
        old,_=run('prior_BDF2','cuda',{},args.reference_executable.resolve())
    current,_=run('current_BDF2','cuda',{})
    if args.reference_executable:
        compare('BDF2_unchanged',old,current)
    bdf_host,_=run('current_BDF2_host','host',{})
    compare('BDF2_host_cuda_updates',bdf_host,current)
    steady={'/timeMarchSettings/integrator':'ImplicitEulerLUSGS',
            '/timeMarchSettings/useCFLTimeStep':True,'/timeMarchSettings/useLocalTimeStep':True,
            '/timeMarchSettings/cfl':.5,'/timeMarchSettings/maxImplicitIterations':1,
            '/timeMarchSettings/lusgsSweeps':1,'/timeMarchSettings/steadyUntilConverged':False,
            '/timeMarchSettings/steadyLineSearch':False,
            '/timeMarchSettings/steadyConvergenceRelativeTolerance':0.,
            '/timeMarchSettings/dualTimeConvergenceCriterion':'Absolute',
            '/timeMarchSettings/dualTimeFlowRelativeTolerance':0.,
            '/timeMarchSettings/dualTimeTurbulenceRelativeTolerance':0.,
            '/timeMarchSettings/dualTimeRelativeTolerances':[0.]*6,
            '/timeMarchSettings/dualTimeRequireConvergence':False}
    host,_=run('steady_host','host',steady)
    cuda,log=run('steady_cuda','cuda',steady)
    compare('steady_host_cuda_updates',host,cuda)
    changed,_=run('steady_changed_unused_physical_dt','cuda',{**steady,'/timeMarchSettings/physicalTimeStep':.7})
    compare('no_physical_dt_dependency',cuda,changed)
    if len(re.findall('ACM resident steady iteration',log))!=3 or 'BDF1' in log or 'BDF2 residual' in log:
        raise ValueError('Steady execution incorrectly uses physical BDF stepping')
    if 'converged=1' in log: raise ValueError('A strict finite iteration budget was reported as converged')
    _,stop=run('loose_target_stop_check','cuda',{**steady,'/timeMarchSettings/nSteps':1,
             '/timeMarchSettings/steadyUntilConverged':True,'/timeMarchSettings/steadyConvergenceWindow':3,
             '/timeMarchSettings/steadyConvergenceAbsoluteTolerances':[1e10]*6})
    if len(re.findall('ACM resident steady iteration',stop))!=3 or 'stable=3 converged=1' not in stop:
        raise ValueError('Until-converged iteration/window control failed')
    summary.append(dict(case='until_converged_ignores_budget_and_honors_window',passed=True))
    _,search=run('steady_line_search','cuda',{**steady,'/timeMarchSettings/nSteps':30,
                  '/timeMarchSettings/cfl':2.,'/timeMarchSettings/steadyLineSearch':True})
    initial=json.loads(re.search(r'eqInitial=(\[[^\n]+?\]) eqTarget=',search)[1])
    scales=np.maximum(initial,[1e-8,1e-8,1e-8,1e-8,1e-10,1e-3])
    merits=[float(np.sqrt(np.mean((np.asarray(initial)/scales)**2)))]
    for line in search.splitlines():
        if not line.startswith('ACM resident steady iteration'): continue
        values=np.asarray(json.loads(line.split('eqResidual=')[1]))
        merit=float(np.sqrt(np.mean((values/scales)**2)))
        if merit>max(merits)*(1+2e-8): raise ValueError('Accepted steady merit exceeded its history envelope')
        merits.append(merit)
        merits=merits[-20:]
    summary.append(dict(case='fixed_reference_nonmonotone_spatial_line_search',passed=True))
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
