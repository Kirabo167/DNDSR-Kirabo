"""Launch and check single-layer cases on a Linux Open MPI host."""
from pathlib import Path
import hashlib
import json
import os
import re
import resource
import shutil
import subprocess
import time
import xml.etree.ElementTree as ET

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'DNDS_DIST_OMP_NUM_THREADS'):
    os.environ[key] = '1'

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
EXE = ROOT / 'build/single_layer_barth/ncfv_single_period_3D.exe'
CONFIG = HERE / 'RP1.json'


def save(path, data):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')
    tmp.replace(path)


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def stamp():
    return time.strftime('%Y-%m-%d %H:%M:%S %z')


def latest_progress(path, pattern, tail_bytes=65536):
    """Bound polling memory and I/O independently of the solver log size."""
    with path.open('rb') as stream:
        size = stream.seek(0, os.SEEK_END)
        offset = max(0, size-tail_bytes)
        stream.seek(offset)
        if offset:
            stream.readline(tail_bytes)  # Ignore a potentially partial first record.
        text = stream.read(tail_bytes).decode('utf-8', errors='replace')
    last = None
    for match in pattern.finditer(text):
        last = match
    return last


def select_cores(n):
    topology = {}
    for row in subprocess.check_output(['lscpu', '-p=CPU,CORE,SOCKET'], text=True).splitlines():
        if not row.startswith('#'):
            cpu, core, socket = map(int, row.split(','))
            topology[cpu] = (socket, core)
    used = set()
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            if path.stat().st_uid != os.getuid():
                continue
            if b'OMPI_COMM_WORLD_RANK=' in (path/'environ').read_bytes():
                occupied = os.sched_getaffinity(int(path.name))
                used.update(topology[c] for c in occupied)
        except OSError:
            pass
    selected = []
    for cpu in sorted(os.sched_getaffinity(0)):
        if topology[cpu] not in used:
            selected.append(cpu)
            used.add(topology[cpu])
        if len(selected) == n:
            return topology, selected
    raise RuntimeError(f'Need {n} unoccupied physical cores, found {len(selected)}')


def affinity(exe, topology, cpus):
    rows = []
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            if path.stat().st_uid != os.getuid() or (path/'exe').resolve() != exe.resolve():
                continue
            env = (path/'environ').read_bytes().split(b'\0')
            rank = int(next(e.split(b'=')[1] for e in env if e.startswith(b'OMPI_COMM_WORLD_RANK=')))
            rows.append(dict(rank=rank, pid=int(path.name), cpus=sorted(os.sched_getaffinity(int(path.name)))))
        except (OSError, ValueError, StopIteration):
            pass
    observed = [r['cpus'][0] for r in rows if len(r['cpus']) == 1]
    return dict(verified=len(rows) == len(cpus) == len(observed) and set(observed) == set(cpus)
                and len({topology[c] for c in observed}) == len(cpus), ranks=rows)


def run(destination, ranks, overrides=None, extra_env=None, poll_seconds=.25):
    settings = {'/io/outputPrefix': str(destination/'solution'), '/io/restartPrefix': str(destination/'restart'),
                '/io/vtkSeriesName': str(destination/'series'), **(overrides or {})}
    manifest = dict(executable_sha256=digest(EXE), configuration_sha256=digest(CONFIG),
                    overrides=settings, ranks=ranks, extra_env=extra_env or {})
    if (destination/'run_status.json').exists():
        previous = json.loads((destination/'run_status.json').read_text())
        assert previous['status'] == 'completed' and previous['inputs'] == manifest
        return destination
    topology, cpus = select_cores(ranks)
    destination.mkdir(parents=True, exist_ok=False)
    inputs = destination/'inputs'
    inputs.mkdir()
    exe, config = inputs/EXE.name, inputs/CONFIG.name
    shutil.copy2(EXE, exe)
    shutil.copy2(CONFIG, config)
    env = {k: v for k, v in os.environ.items() if not k.startswith(('NCFV_', 'BJ_')) and k not in ('DISPLAY', 'WAYLAND_DISPLAY')}
    env.update(HWLOC_COMPONENTS='-gl,-cuda,-nvml', NCFV_CELL_AUDIT_FILE=str(destination/'cell_audit.jsonl'),
               NCFV_CELL_AUDIT_INTERVAL='1')
    env.update(extra_env or {})
    tree = ET.fromstring(subprocess.check_output(['lstopo-no-graphics', '--of', 'xml', '-'], env=env, text=True))
    pus = [int(o.attrib['os_index']) for o in tree.iter('object') if o.attrib.get('type') == 'PU']
    mapping = {cpu: i for i, cpu in enumerate(pus)}
    args = ['mpirun', '--mca', 'pml', 'ob1', '--mca', 'btl', 'self,vader', '--use-hwthread-cpus',
            '-np', str(ranks), '--cpu-list', ','.join(str(mapping[c]) for c in cpus),
            '--bind-to', 'cpu-list:ordered', '--report-bindings', str(exe), str(config)]
    for key, value in settings.items():
        args.extend(['-k', key, '-v', json.dumps(value)])
    _, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(65536, hard), hard))
    check = subprocess.run(args[args.index(str(exe)):] + ['--check-config'], cwd=destination,
                           env=env, capture_output=True, text=True, timeout=60)
    (destination/'config_check.log').write_text(check.stdout+check.stderr)
    assert check.returncode == 0
    state = dict(status='initializing', controller_pid=os.getpid(), inputs=manifest, command=args,
                 linux_cpus=cpus, physical_cores=[topology[c] for c in cpus], physical_binding_verified=False,
                 created_at=stamp(), target_time=settings.get('/time/endTime', .3))
    save(destination/'run_status.json', state)
    pattern = re.compile(r'NCFV iteration=(\d+).*?, time=([0-9.eE+-]+), target=([0-9.eE+-]+), step wall\[s\]=([0-9.eE+-]+)')
    started = time.monotonic()
    print(stamp(), 'LAUNCH', destination, ranks, flush=True)
    with (destination/'solver.log').open('x') as log:
        child = subprocess.Popen(args, cwd=destination, env=env, stdout=log, stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL, start_new_session=True)
        state['mpi_pid'] = child.pid
        while child.poll() is None:
            if not state['physical_binding_verified']:
                proof = affinity(exe, topology, cpus)
                if proof['verified']:
                    state['physical_binding_verified'] = True
                    save(destination/'cpu_affinity.json', proof)
            last = latest_progress(destination/'solver.log', pattern)
            if last is not None:
                state.update(status='running', iteration=int(last[1]), physical_time=float(last[2]),
                             progress_fraction=float(last[2])/float(last[3]), last_step_seconds=float(last[4]))
            state.update(updated_at=stamp(), elapsed_seconds=time.monotonic()-started)
            save(destination/'run_status.json', state)
            time.sleep(poll_seconds)
        state.update(status='completed' if child.returncode == 0 else 'failed', returncode=child.returncode,
                     elapsed_seconds=time.monotonic()-started, updated_at=stamp())
        save(destination/'run_status.json', state)
    if child.returncode:
        raise RuntimeError(f'Solver failed: {destination}/solver.log')
    print(stamp(), 'FINISHED', destination, flush=True)
    return destination


def read_checkpoint(path):
    with h5py.File(path) as stream:
        u = stream['state/father/array/data'][:].reshape(-1, 5)
        ids = stream['state/origIndex'][:].astype(np.int64)
        t, step = float(stream.attrs['simulationTime']), int(stream.attrs['iteration'])
    assert np.array_equal(np.sort(ids), np.arange(len(ids)))
    ordered = np.empty_like(u)
    ordered[ids] = u
    p = .4*(ordered[:, 4]-.5*np.sum(ordered[:, 1:4]**2, axis=1)/ordered[:, 0])
    assert np.isfinite(ordered).all() and ordered[:, 0].min() > 0 and p.min() > 0
    return ordered, p, t, step
