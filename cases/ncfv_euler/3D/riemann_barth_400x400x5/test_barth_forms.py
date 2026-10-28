"""Isolated Barth/flux experiments; never modifies production C++ or old results.

Builds private source copies with diagnostic switches, then compares RP1 and a
moving contact on 40x40x5. Flags are present only in the diagnostic executable.
"""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
CASE = Path(__file__).resolve().parent


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"Expected one patch location: {old[:80]}")
    return source.replace(old, new, 1)


def build(out):
    """Inject experiments into private copies and link ahead of static archives."""
    commands = json.loads((ROOT / "build/compile_commands.json").read_text())
    objects = []
    for name in ("NCFVSpatial", "NCFVReconstruction"):
        src = ROOT / f"src/NCFV/{name}.cpp"
        text = src.read_text()
        helper = ('#include <cstdlib>\n#include <cstring>\n'
                  'namespace { bool AuditFlag(const char *name) { '
                  'const char *v = std::getenv(name); return v && std::strcmp(v, "1") == 0; } }\n')
        if name == "NCFVSpatial":
            text = replace_once(text, '        Reconstruct(means);', '''        Reconstruct(means);
        if (AuditFlag("BJ_FIRST_ORDER"))
            for (index i = 0; i < _pointValues.Size(); i++)
            {
                _pointValues[i] = means[i];
                _limiterFactors[i](0, 0) = 0;
            }''')
            text = replace_once(text, '        State flux = State::Zero();\n        SpatialVector gridVelocity', '''        if (AuditFlag("BJ_LLF"))
        {
            const auto speed = [&](const State &u)
            {
                return std::abs(u.template segment<dimension>(1).dot(unitNormal) / u(0)) +
                       std::sqrt(_physics.gamma * Pressure(u) / u(0));
            };
            return .5 * (PhysicalFlux(left, unitNormal) + PhysicalFlux(right, unitNormal)) -
                   .5 * std::max(speed(left), speed(right)) * (right - left);
        }
        State flux = State::Zero();
        SpatialVector gridVelocity''')
            text = replace_once(text, '        State integral = 0.5 * (leftIntegral + rightIntegral) +\n                         dissipativeCorrection;', '''        State integral = 0.5 * (leftIntegral + rightIntegral) +
                         dissipativeCorrection;
        if (AuditFlag("BJ_FACE_ONLY"))
            integral = surface.measure * numerical;''')
            text = replace_once(text, '            total += inviscidIntegral;', '''            if (AuditFlag("BJ_LOCAL_BOUNDARY"))
                inviscidIntegral = piece.measure * BoundaryNumericalFlux(
                    State(_pointValues[iNode]), unitNormal, boundary);
            total += inviscidIntegral;''')
        else:
            text = replace_once(text, '''                    limit(candidate, minimum, maximum);''', '''                    if (AuditFlag("BJ_PAIR_BOUNDS") && _settings.limitPointRecovery)
                        limit(candidate, means[iNode].cwiseMin(means[neighbor]),
                              means[iNode].cwiseMax(means[neighbor]));
                    else
                        limit(candidate, minimum, maximum);''')
        patched = out / (name + ".cpp")
        patched.write_text(helper + text)
        obj = out / (name + ".o")
        record = next(c for c in commands if c["file"] == str(src) and
                      (name != "NCFVSpatial" or "INSTANTIATION_DIM=3" in c["command"]))
        cmd = shlex.split(record["command"])
        cmd[cmd.index('-o') + 1] = str(obj)
        cmd[cmd.index('-c') + 1] = str(patched)
        cmd += ['-I' + str(ROOT / 'src/NCFV')]
        with (out / (name + '.build.log')).open('w') as log:
            subprocess.run(cmd, cwd=ROOT / 'build', stdout=log, stderr=subprocess.STDOUT, check=True)
        objects.append(str(obj))
    raw = subprocess.check_output(['ninja', '-C', 'build', '-t', 'commands',
                                   'app/ncfv_euler3D.exe'], cwd=ROOT, text=True).splitlines()[-1]
    cmd = shlex.split(raw.removeprefix(': && ').removesuffix(' && :'))
    cmd[cmd.index('-o') + 1] = str(out / 'diagnostic.exe')
    cmd[1:1] = objects
    subprocess.run(cmd, cwd=ROOT / 'build', check=True)
    return out / 'diagnostic.exe'


def metrics(path, contact=False):
    with h5py.File(path) as f:
        u = f['state/father/array/data'][:].reshape(-1, 5)
        ids = f['state/origIndex'][:].astype(int)
        time = float(f.attrs['simulationTime'])
        iteration = int(f.attrs['iteration'])
    ordered = np.empty_like(u)
    ordered[ids] = u
    rho = ordered[:, 0].reshape(6, 41, 41)
    p = (.4 * (ordered[:, 4] - .5 * np.sum(ordered[:, 1:4]**2, axis=1) /
               ordered[:, 0])).reshape(6, 41, 41)
    result = {'time': time, 'iteration': iteration,
              'rho_min': float(rho.min()), 'rho_max': float(rho.max()),
              'w_abs_max': float(np.abs(ordered[:, 3] / ordered[:, 0]).max()),
              'p_min': float(p.min()), 'p_max': float(p.max()),
              'rho_z_span_max': float(np.ptp(rho, axis=0).max()),
              'rho_z_span_rms': float(np.sqrt(np.mean(np.ptp(rho, axis=0)**2))),
              'rho_z_TV_excess_max': float(np.maximum(0, np.abs(np.diff(rho, axis=0)).sum(axis=0) -
                                                     np.ptp(rho, axis=0)).max()),
              'p_z_span_max': float(np.ptp(p, axis=0).max())}
    if contact:
        # Exact advected contact: rho=1 to the left, 2 to the right, u=.5, p=1.
        dx = .025
        x = np.arange(41) * dx
        lo, hi = np.maximum(0, x - dx/2), np.minimum(1, x + dx/2)
        fraction = np.clip((hi - (.5 + .5*time)) / (hi-lo), 0, 1)
        exact = 1 + fraction
        result.update(rho_overshoot=max(0., float(rho.max()-2)),
                      rho_undershoot=max(0., float(1-rho.min())),
                      pressure_error_max=float(np.abs(p-1).max()),
                      rho_L1=float(np.mean(np.abs(rho-exact[None, None, :]))),
                      rho_midline_TV=float(np.abs(np.diff(rho[3,20])).sum()))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--ranks', type=int, default=4)
    args = parser.parse_args()
    out = (args.output or ROOT / 'data/out/NCFV/barth_forms_tests' /
           datetime.now().strftime('%Y%m%d_%H%M%S')).resolve()
    out.mkdir(parents=True, exist_ok=False)
    print('Output:', out, flush=True)
    exe = build(out)
    audit = ROOT / 'data/out/NCFV/riemann_barth_400x400x5/limiter_audit'
    contact_file = out / 'contact.csv'
    ids = np.arange(41*41*6)
    x = (ids % 41) / 40
    lo, hi = np.maximum(0, x-.0125), np.minimum(1, x+.0125)
    rho = 1 + np.clip((hi-.5)/(hi-lo), 0, 1)
    np.savetxt(contact_file, np.column_stack([ids,rho,.5*rho,0*rho,0*rho,2.5+.125*rho]),
               delimiter=',', header='globalNode,rho,rhou,rhov,rhow,rhoE', comments='',
               fmt=['%d']+['%.17g']*5)
    variants = {
        'baseline': {},
        'pair_bounds': {'BJ_PAIR_BOUNDS':'1'},
        'local_boundary': {'BJ_LOCAL_BOUNDARY':'1'},
        'face_roe': {'BJ_FACE_ONLY':'1'},
        'face_roe_local': {'BJ_FACE_ONLY':'1','BJ_LOCAL_BOUNDARY':'1'},
        'face_llf_local': {'BJ_FACE_ONLY':'1','BJ_LOCAL_BOUNDARY':'1','BJ_LLF':'1'},
        'first_order_roe': {'BJ_FIRST_ORDER':'1','BJ_LOCAL_BOUNDARY':'1'},
    }
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               HWLOC_COMPONENTS='-gl,-cuda,-nvml', OMPI_MCA_mpi_yield_when_idle='1')
    for key in list(env):
        if key.startswith(('BJ_', 'NCFV_')):
            del env[key]
    results = {'diagnostic_only':True, 'threshold':1e-10,
               'binary_sha256':hashlib.sha256(exe.read_bytes()).hexdigest(), 'runs':[]}
    for problem in ('contact', 'RP1'):
        for name, flags in variants.items():
            directory = out / problem / name
            directory.mkdir(parents=True)
            cmd = ['mpirun','--mca','pml','ob1','--mca','btl','self,vader','--bind-to','none',
                   '-np',str(args.ranks),str(exe),str(CASE/'RP1_positive_tol1e10.json')]
            overrides = {'/mesh/meshFile':str(audit/'mesh40.cgns'),
                         '/initialField/nodeFile':str(contact_file if problem=='contact' else audit/'initial40/RP1.csv'),
                         '/time/maximumTimeStep':.001, '/io/writeVTK':False,
                         '/io/restartInterval':0, '/io/outputPrefix':str(directory/'solution'),
                         '/io/vtkSeriesName':str(directory/'series'),
                         '/io/restartPrefix':str(directory/'restart')}
            for key, value in overrides.items():
                cmd += ['-k',key,'-v',json.dumps(value)]
            (directory/'command.json').write_text(json.dumps({'command':cmd,'flags':flags},indent=2)+'\n')
            with (directory/'solver.log').open('w') as log:
                proc = subprocess.run(cmd,cwd=ROOT/'build',env=dict(env,**flags),
                                      stdout=log,stderr=subprocess.STDOUT)
            result = {'problem':problem,'variant':name,'returncode':proc.returncode}
            if proc.returncode==0:
                restart = sorted(directory.glob('restart_*.dnds.h5'))[-1]
                result.update(metrics(restart,problem=='contact'))
                result['rejected_steps']=(directory/'solver.log').read_text().count('NCFV rejected iteration=')
            results['runs'].append(result)
            (out/'results.json').write_text(json.dumps(results,indent=2)+'\n')
            print(json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
