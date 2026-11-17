#!/usr/bin/env python3
"""Queue a partitioned GPU steady SUBOFF restart and audit its saved output."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid as uuid_module

import h5py
import numpy as np

from run_suboff_steady_gpu import ROOT, REFERENCE, ITERATION, audit_field, stamp


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def host_memory_status():
    """Respect both system availability and this account's cgroup memory cap."""
    account = Path('/sys/fs/cgroup/user.slice') / f'user-{os.getuid()}.slice'
    limit_text = (account / 'memory.max').read_text().strip()
    current = int((account / 'memory.current').read_text())
    limit = None if limit_text == 'max' else int(limit_text)
    meminfo = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    system_available = int(meminfo['MemAvailable'].split()[0]) * 1024
    available = system_available if limit is None else min(system_available, max(0, limit - current))
    return dict(host_available_bytes=available, account_memory_limit_bytes=limit,
                account_memory_current_bytes=current, system_memory_available_bytes=system_available)


def resolve_uuid_devices(state, available):
    """Resolve the requested physical cards after device indices or visibility change."""
    by_uuid = {item['uuid']: item for item in available}
    requested = state['rank_device_uuids']
    missing = sorted(set(requested) - by_uuid.keys())
    state['missing_gpu_uuids'] = missing
    if missing:
        state.update(rank_devices=None, minimum_free_by_device_mib=None,
                     state='waiting_for_gpu_access')
        return False
    state['rank_devices'] = [by_uuid[value]['device'] for value in requested]
    state['gpu_candidates'] = sorted(set(state['rank_devices']))
    state['minimum_free_by_device_mib'] = {
        by_uuid[value]['device']: required
        for value, required in state['minimum_free_by_uuid_mib'].items()}
    return True


def audit_import(source, imported, pressure_is_modified=True):
    """Check legacy pressure conversion or an unchanged static-pressure restart."""
    maximum_error = 0.0
    with h5py.File(source) as old, h5py.File(imported) as new:
        a, b = old['VTKHDF/CellData'], new['VTKHDF/CellData']
        for name in a:
            if a[name].shape != b[name].shape:
                raise ValueError(f'Restart shape changed: {name}')
            for begin in range(0, len(a[name]), 100000):
                section = slice(begin, begin + 100000)
                expected = a[name][section]
                if name == 'Pressure' and pressure_is_modified:
                    kinetic_pressure = (2.0 / 3.0) * a['Density'][section] * a['TurbulenceK'][section]
                    expected = expected - kinetic_pressure
                    maximum_error = max(maximum_error, float(np.max(np.abs(b[name][section] - expected))))
                    if not np.allclose(b[name][section], expected, rtol=1e-13, atol=1e-14):
                        raise ValueError('Imported static pressure conversion failed')
                    if not np.allclose(b['TurbulentPressure'][section], kinetic_pressure, rtol=1e-13, atol=1e-14):
                        raise ValueError('Imported turbulent pressure mismatch')
                elif not np.array_equal(expected, b[name][section]):
                    raise ValueError(f'Restart altered initial field: {name}')
        for name in ('Points', 'Connectivity', 'Offsets', 'Types'):
            if old['VTKHDF'][name].shape != new['VTKHDF'][name].shape:
                raise ValueError(f'Restart altered mesh shape: {name}')
            for begin in range(0, len(old['VTKHDF'][name]), 100000):
                section = slice(begin, begin + 100000)
                if not np.array_equal(old['VTKHDF'][name][section], new['VTKHDF'][name][section]):
                    raise ValueError(f'Restart altered mesh: {name}')
    return dict(static_pressure_conversion_verified=pressure_is_modified,
                legacy_pressure_converted=pressure_is_modified,
                static_pressure_import_verified=True, other_initial_fields_identical=True,
                mesh_identical=True, maximum_pressure_conversion_error=maximum_error)


def worker(directory):
    state = json.loads((directory / 'status.json').read_text())
    config_path = directory / 'launch_config.json'
    config = json.loads(config_path.read_text())
    settings, restart = config['timeMarchSettings'], config['restartSettings']
    executable = directory / 'acm2EQ_gpu3D.exe'
    source = (ROOT / 'build-gpu-solvers' / restart['flowFile']).resolve()
    started = time.monotonic()
    state.update(worker_pid=os.getpid(), state='validating_checkpoint')

    def persist():
        state.update(updated_utc=stamp(), elapsed_seconds=time.monotonic() - started)
        save_json(directory / 'status.json', state)

    process = None
    persist()
    try:
        state['source_checkpoint_audit'] = audit_field(source)
        state['source_checkpoint_sha256'] = digest(source)
        state['state'] = 'waiting_for_gpu'
        persist()
        while True:
            result = subprocess.run(['nvidia-smi',
                '--query-gpu=index,uuid,memory.free,utilization.gpu', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, check=True, timeout=30)
            available = []
            for line in result.stdout.strip().splitlines():
                device, uuid, free, utilization = [value.strip() for value in line.split(',')]
                if (uuid in (state.get('rank_device_uuids') or []) or
                        (not state.get('rank_device_uuids') and int(device) in state['gpu_candidates'])):
                    available.append(dict(device=int(device), uuid=uuid, free_mib=int(free),
                                          utilization_percent=int(utilization)))
            state['gpu_candidates_status'] = available
            if state.get('rank_device_uuids') and not resolve_uuid_devices(state, available):
                persist()
                time.sleep(30)
                continue
            if not available:
                raise RuntimeError('None of the requested GPU devices is visible')
            selected = max(available, key=lambda item: item['free_mib'])
            state.update(gpu_candidates_status=available, gpu_free_mib=selected['free_mib'])
            if state.get('minimum_host_free_bytes', 0):
                state.update(host_memory_status())
                if state['host_available_bytes'] < state['minimum_host_free_bytes']:
                    state['state'] = 'waiting_for_host_memory'
                    persist()
                    time.sleep(30)
                    continue
            state['state'] = 'waiting_for_gpu'
            persist()
            if state.get('rank_devices'):
                free_by_device = {item['device']: item['free_mib'] for item in available}
                if all(free_by_device.get(int(device), -1) >= required
                       for device, required in state['minimum_free_by_device_mib'].items()):
                    break
                time.sleep(30)
                continue
            if selected['free_mib'] >= state['minimum_free_mib']:
                state.update(gpu_device=selected['device'], gpu_uuid=selected['uuid'],
                             gpu_utilization_percent=selected['utilization_percent'])
                break
            time.sleep(30)
        if digest(executable) != state['executable_sha256'] or digest(config_path) != state['config_sha256']:
            raise ValueError('Submitted executable or configuration changed while queued')
        if state.get('rank_devices') and digest(directory / 'run_acm_gpu_rank.py') != state['rank_launcher_sha256']:
            raise ValueError('Submitted MPI rank launcher changed while queued')
        if digest(source) != state['source_checkpoint_sha256']:
            raise ValueError('Source checkpoint changed while queued')
        environment = dict(os.environ, OMP_NUM_THREADS='8', OPENBLAS_NUM_THREADS='1',
            HWLOC_COMPONENTS='-opencl,-gl,-cuda,-nvml', OMPI_MCA_pml='ob1', OMPI_MCA_btl='self,vader,tcp',
            OMPI_MCA_btl_tcp_if_include='lo', OMPI_MCA_oob_tcp_if_include='lo')
        environment.pop('DISPLAY', None)
        command = ['mpirun', '--bind-to', 'none', '-np', str(state['ranks'])]
        if state.get('rank_devices'):
            environment['DNDS_SUBOFF_RANK_DEVICES'] = ','.join(map(str, state['rank_devices']))
            command += [sys.executable, str(directory / 'run_acm_gpu_rank.py'),
                        str(executable), str(config_path)]
        else:
            command += [str(executable), str(config_path), '--backend', 'cuda',
                        '--gpu-device', str(state['gpu_device'])]
        state.update(state='initializing', command=command, solver_started_utc=stamp())
        persist()
        position, pending, final_window = 0, '', []
        first_iteration_time = None
        first_iteration = None
        run_started = time.monotonic()
        with (directory / 'run.log').open('w') as log, (directory / 'residual_history.csv').open('w') as history:
            writer = csv.writer(history)
            writer.writerow(['iteration', 'elapsed_solver_seconds', 'flow_residual', 'turbulence_residual',
                             'dtau_min', 'cfl', 'relaxation', 'accepted', 'stable', 'converged',
                             'u', 'v', 'w', 'p', 'k', 'omega'])
            process = subprocess.Popen(command, cwd=ROOT / 'build-gpu-solvers', env=environment,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            state['solver_pid'] = process.pid
            exit_seen = False
            while True:
                with (directory / 'run.log').open() as reader:
                    reader.seek(position)
                    pending += reader.read()
                    position = reader.tell()
                lines = pending.split('\n')
                pending = lines.pop()
                for line in lines:
                    reference = REFERENCE.search(line)
                    if reference:
                        state.update(initial_equation_residuals=json.loads(reference[1]),
                                     target_equation_residuals=json.loads(reference[2]))
                    match = ITERATION.search(line)
                    if not match:
                        continue
                    step, norms = int(match[1]), json.loads(match[12])
                    if not all(math.isfinite(value) for value in norms):
                        raise ValueError('Non-finite spatial residual')
                    now = time.monotonic()
                    if first_iteration_time is None:
                        first_iteration_time, first_iteration = now, step
                    if step > first_iteration:
                        state['mean_observed_seconds_per_iteration'] = (now - first_iteration_time) / (step - first_iteration)
                    state.update(state='running', completed_iteration=step,
                        completed_steady_iterations=step - restart['completedSteps'],
                        equation_residuals=norms, flow_residual=float(match[3]), turbulence_residual=float(match[5]),
                        minimum_local_time_step=float(match[4]), cfl=float(match[6]),
                        accepted_relaxation=float(match[8]), update_accepted=bool(int(match[9])),
                        stable_checks=int(match[10]), converged=bool(int(match[11])))
                    final_window.append(norms)
                    final_window = final_window[-settings['steadyConvergenceWindow']:]
                    writer.writerow([step, now - run_started, float(match[3]), float(match[5]), float(match[4]),
                        float(match[6]), float(match[8]), int(match[9]), int(match[10]), int(match[11]), *norms])
                history.flush()
                if state['completed_steady_iterations'] > 0 and 'initial_import_audit' not in state:
                    # The first iteration log follows completion of the initial file write.
                    initial = directory / 'fields' / f'{config["outputSettings"]["prefix"]}_{restart["completedSteps"]:08d}.vtkhdf'
                    state['initial_import_audit'] = audit_import(source, initial, restart.get('pressureIsModified', False))
                    save_json(directory / 'initial_import_audit.json', state['initial_import_audit'])
                persist()
                if process.poll() is not None:
                    if exit_seen:
                        break
                    exit_seen = True
                    continue
                time.sleep(2)
        state['exit_code'] = process.wait()
        if state['exit_code'] != 0:
            raise RuntimeError(f'Solver exited with code {state["exit_code"]}; see run.log')
        if not state['converged'] and (settings['steadyUntilConverged'] or
                state['completed_steady_iterations'] != settings['nSteps']):
            raise RuntimeError('Solver did not reach the finite test budget or convergence')
        state['state'] = 'auditing_results'
        persist()
        prefix = config['outputSettings']['prefix']
        initial = directory / 'fields' / f'{prefix}_{restart["completedSteps"]:08d}.vtkhdf'
        final = directory / 'fields' / f'{prefix}_{state["completed_iteration"]:08d}.vtkhdf'
        imported = state.get('initial_import_audit')
        if imported is None:
            imported = audit_import(source, initial, restart.get('pressureIsModified', False))
        audit = audit_field(final)
        targets = np.asarray(state['target_equation_residuals'])
        verified = (len(final_window) == settings['steadyConvergenceWindow'] and
                    bool(np.all(np.asarray(final_window) <= targets)))
        if state['converged'] != verified:
            raise ValueError('Convergence flag disagrees with the consecutive residual checks')
        audit.update(import_audit=imported, spatial_convergence_verified=verified,
            final_equation_residuals=state['equation_residuals'], target_equation_residuals=targets.tolist(),
            final_to_initial_residual_ratios=(np.asarray(state['equation_residuals']) /
                np.maximum(np.asarray(state['initial_equation_residuals']), np.finfo(float).tiny)).tolist(),
            physical_time_advancing=False, accuracy_verified=False)
        save_json(directory / 'final_field_audit.json', audit)
        state.update(state='completed' if settings['steadyUntilConverged'] else 'test_completed',
                     converged=verified, final_result=str(final),
                     completion_reason='spatial convergence' if verified else 'finite test budget completed; not converged')
    except Exception as error:
        # Terminate only this worker's child process group if monitoring itself fails.
        if process is not None and process.poll() is None:
            import signal
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        state.update(state='failed', error=str(error))
        raise
    finally:
        persist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu-device', default='1', help='Physical device index, or auto to select GPU 0 or 1')
    parser.add_argument('--min-free-mib', type=int, default=76800)
    parser.add_argument('--minimum-host-free-gib', type=float, default=0,
                        help='Require this much account/system host-memory headroom before GPU initialization')
    parser.add_argument('--foreground', action='store_true',
                        help='Run the queue worker in this process, suitable for a protected systemd service')
    parser.add_argument('--rank-devices', nargs='+', type=int,
                        help='Physical GPU for each MPI rank, e.g. 0 1 1 for a one-third/two-thirds split')
    parser.add_argument('--rank-device-uuids', nargs='+',
                        help='Stable GPU UUID per rank; exclusive with --rank-devices')
    parser.add_argument('--device-min-free-mib', nargs='+', type=int,
                        help='Free MiB per distinct device: sorted index order, or first appearance for UUID mapping')
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    directory = args.output.resolve()
    if args.worker:
        worker(directory)
        return
    if args.config is None:
        parser.error('--config is required when submitting a test')
    try:
        gpu_candidates = [0, 1] if args.gpu_device == 'auto' else [int(args.gpu_device)]
    except ValueError:
        parser.error('--gpu-device must be a non-negative index or auto')
    if any(device < 0 for device in gpu_candidates) or args.min_free_mib <= 0:
        parser.error('GPU indices must be non-negative and the free-memory threshold must be positive')
    if not math.isfinite(args.minimum_host_free_gib) or args.minimum_host_free_gib < 0:
        parser.error('--minimum-host-free-gib must be finite and non-negative')
    ranks = 1
    minimum_free_by_device = None
    minimum_free_by_uuid = None
    if args.rank_device_uuids:
        if args.rank_devices:
            parser.error('Choose either rank indices or rank UUIDs')
        for value in args.rank_device_uuids:
            try:
                if not value.startswith('GPU-'):
                    raise ValueError('Missing GPU prefix')
                uuid_module.UUID(value[4:])
            except ValueError:
                parser.error('Expected full NVIDIA GPU UUIDs')
        distinct = list(dict.fromkeys(args.rank_device_uuids))
        if (len(distinct) < 2 or not args.device_min_free_mib or
                len(args.device_min_free_mib) != len(distinct) or min(args.device_min_free_mib) <= 0):
            parser.error('Specify two or more distinct UUIDs and a positive threshold per distinct card')
        minimum_free_by_uuid = dict(zip(distinct, args.device_min_free_mib))
        gpu_candidates = []
        ranks = len(args.rank_device_uuids)
    elif args.rank_devices:
        gpu_candidates = sorted(set(args.rank_devices))
        ranks = len(args.rank_devices)
        if min(gpu_candidates) < 0 or len(gpu_candidates) < 2:
            parser.error('--rank-devices must select at least two non-negative physical GPU indices')
        if (not args.device_min_free_mib or len(args.device_min_free_mib) != len(gpu_candidates)
                or min(args.device_min_free_mib) <= 0):
            parser.error('Specify one positive --device-min-free-mib value per distinct GPU')
        minimum_free_by_device = dict(zip(gpu_candidates, args.device_min_free_mib))
    elif args.device_min_free_mib:
        parser.error('--device-min-free-mib requires rank indices or UUIDs')
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    settings, restart = config['timeMarchSettings'], config['restartSettings']
    if not (settings['integrator'] == 'ImplicitEulerLUSGS'
            and (settings['steadyUntilConverged'] or settings['nSteps'] > 0)
            and settings['useLocalTimeStep'] and settings['useCFLTimeStep']
            and config['turbulenceSettings']['model'] == 'KOmegaSST'
            and config['turbulenceSettings']['includeIsotropicStress']
            and restart['writerRanks'] == ranks and isinstance(restart.get('pressureIsModified', False), bool)
            and config['outputSettings']['writeInitial'] and config['outputSettings']['interval'] > 0):
        raise ValueError('Expected steady local-CFL SST run with a restart matching the MPI rank count')
    if (ROOT / 'build-gpu-solvers' / config['outputSettings']['directory']).resolve() != directory / 'fields':
        raise ValueError('Output directory does not match the configuration')
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / 'status.json').exists() or any((directory / 'fields').glob('*.vtkhdf')):
        raise FileExistsError('Use a new test directory; existing results are preserved')
    executable = ROOT / 'build-gpu-solvers/app/acm2EQ_gpu3D.exe'
    shutil.copy2(config_path, directory / 'launch_config.json')
    shutil.copy2(executable, directory / executable.name)
    if args.rank_devices or args.rank_device_uuids:
        shutil.copy2(Path(__file__).with_name('run_acm_gpu_rank.py'), directory / 'run_acm_gpu_rank.py')
    state = dict(state='submitted', created_utc=stamp(), config=str(config_path),
        submitted_config=str(directory / 'launch_config.json'), config_sha256=digest(config_path),
        executable=str(directory / executable.name), executable_sha256=digest(executable),
        source_git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        gpu_device=gpu_candidates[0] if len(gpu_candidates) == 1 else None,
        gpu_candidates=gpu_candidates, minimum_free_mib=args.min_free_mib, ranks=ranks,
        minimum_host_free_bytes=int(args.minimum_host_free_gib * 1024 ** 3),
        rank_devices=args.rank_devices, minimum_free_by_device_mib=minimum_free_by_device,
        rank_device_uuids=args.rank_device_uuids, minimum_free_by_uuid_mib=minimum_free_by_uuid,
        initial_iteration=restart['completedSteps'], completed_iteration=restart['completedSteps'],
        completed_steady_iterations=0, until_converged=settings['steadyUntilConverged'],
        iteration_limit=None if settings['steadyUntilConverged'] else settings['nSteps'],
        target_iteration=None if settings['steadyUntilConverged'] else restart['completedSteps'] + settings['nSteps'],
        convergence_window=settings['steadyConvergenceWindow'],
        absolute_tolerances=settings['steadyConvergenceAbsoluteTolerances'],
        relative_tolerance=settings['steadyConvergenceRelativeTolerance'], converged=False,
        local_time_step=True, physical_time=restart['physicalTime'], physical_time_advancing=False,
        log=str(directory / 'run.log'), residual_history=str(directory / 'residual_history.csv'))
    if args.rank_devices or args.rank_device_uuids:
        state['rank_launcher_sha256'] = digest(directory / 'run_acm_gpu_rank.py')
    save_json(directory / 'status.json', state)
    if args.foreground:
        worker(directory)
        return
    with (directory / 'worker.log').open('a') as log:
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--worker', '--output', str(directory)],
            cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    print(json.dumps(dict(worker_pid=process.pid, directory=str(directory))))


if __name__ == '__main__':
    main()
