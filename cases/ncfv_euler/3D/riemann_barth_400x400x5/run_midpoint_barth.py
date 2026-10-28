"""Frozen quadratic edge-midpoint Roe/Barth RP1 experiment.

Builds private NCFV sources, leaving production files and earlier results intact.
No DNDSR Python extension is imported. Run with the project Python environment.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import time

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
CASE = Path(__file__).resolve().parent


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def replace_function(source, start, end, replacement):
    a = source.index(start)
    b = source.index(end, a)
    return source[:a] + replacement + source[b:]


EDGE = '''    template <int dimension>
    typename SpatialOperator<dimension>::State
    SpatialOperator<dimension>::EvaluateOwnedEdgeFlux(index iEdge, const NodeStatePair &means)
    {
        const auto &surface = _geometry.EdgeSurface(iEdge);
        const SpatialVector normal = surface.vectorMeasure.template head<dimension>().normalized();
        const State left = EvaluateTraditionalState(surface.nodes[0], surface.edgePoint);
        const State right = EvaluateTraditionalState(surface.nodes[1], surface.edgePoint);
        return surface.measure * NumericalFlux(left, right, normal);
    }

'''

BOUNDARY = '''    template <int dimension>
    typename SpatialOperator<dimension>::State
    SpatialOperator<dimension>::EvaluateOwnedBoundaryFlux(index iNode, const NodeStatePair &means) const
    {
        State total = State::Zero();
        for (const auto &piece : _geometry.NodeVolume(iNode).boundaryPieces)
        {
            const auto &boundary = _boundaries.Get(piece.zone);
            if (boundary.mode == BoundaryMode::Periodic)
                continue;
            Vector3 point = Vector3::Zero();
            for (int vertex = 0; vertex < piece.nPoints; vertex++)
                point += piece.points[vertex].coordinate / piece.nPoints;
            const SpatialVector normal = piece.vectorMeasure.template head<dimension>() / piece.measure;
            total += piece.measure * BoundaryNumericalFlux(EvaluateTraditionalState(iNode, point), normal, boundary);
        }
        return total;
    }

'''

LIMITER = '''    std::vector<real> Reconstruction::ComputeLimiterFactors(
        const NodeStatePair &means, const NodeStatePair &pointValues,
        const NodeMatrixPair &gradients, const NodeMatrixPair &coefficients) const
    {
        DNDS_check_throw_info(_mode == IntegrationMode::TraditionalQuadrature,
                              "Private midpoint executable requires full quadratic reconstruction");
        const int dimension = _mesh->getDim();
        std::vector<real> factors(static_cast<std::size_t>(_mesh->NumNode()), 1.0);
        for (index node = 0; node < _mesh->NumNode(); node++)
        {
            const Eigen::VectorXd mean = means[node];
            Eigen::VectorXd minimum = mean, maximum = mean;
            for (index neighbor : _nodeGraph[static_cast<std::size_t>(node)])
            {
                minimum = minimum.cwiseMin(means[neighbor]);
                maximum = maximum.cwiseMax(means[neighbor]);
            }
            const auto &op = _operators[static_cast<std::size_t>(node)];
            std::vector<Eigen::VectorXd> candidates{Eigen::VectorXd(pointValues[node])};
            for (index neighbor : _nodeGraph[static_cast<std::size_t>(node)])
                candidates.push_back(pointValues[node] + coefficients[node].transpose() *
                    EvaluateBasis(.5 * _nodeHalo.Displacement(node, neighbor), op.referenceLengths, dimension));
            for (const auto &piece : _geometry.NodeVolume(node).boundaryPieces)
            {
                Vector3 point = Vector3::Zero();
                for (int vertex = 0; vertex < piece.nPoints; vertex++)
                    point += piece.points[vertex].coordinate / piece.nPoints;
                candidates.push_back(pointValues[node] + coefficients[node].transpose() *
                    EvaluateBasis(_nodeHalo.DisplacementToPoint(node, point), op.referenceLengths, dimension));
            }
            real factor = 1;
            for (const auto &candidate : candidates)
                for (int component = 0; component < mean.size(); component++)
                {
                    const real delta = candidate(component) - mean(component);
                    // Match the referenced experiment's absolute roundoff threshold.
                    if (delta > 1e-10)
                        factor = std::min(factor, (maximum(component) - mean(component)) / delta);
                    else if (delta < -1e-10)
                        factor = std::min(factor, (minimum(component) - mean(component)) / delta);
                }
            factors[static_cast<std::size_t>(node)] = std::clamp(factor, 0.0, 1.0);
        }
        return factors;
    }

'''


def build(out):
    sources = out / 'sources'
    shutil.copytree(ROOT / 'src/NCFV', sources / 'NCFV')
    shutil.copy2(ROOT / 'app/NCFV/ncfv_euler3D.cpp', sources / 'main.cpp')
    spatial = sources / 'NCFV/NCFVSpatial.cpp'
    text = spatial.read_text()
    text = replace_function(text,
        '    template <int dimension>\n    typename SpatialOperator<dimension>::State\n    SpatialOperator<dimension>::EvaluateOwnedEdgeFlux',
        '    template <int dimension>\n    real SpatialOperator<dimension>::EvaluateOwnedEdgeSpectralRadius', EDGE)
    text = replace_function(text,
        '    template <int dimension>\n    typename SpatialOperator<dimension>::State\n    SpatialOperator<dimension>::EvaluateOwnedBoundaryFlux',
        '    template <int dimension>\n    real SpatialOperator<dimension>::EvaluateOwnedBoundarySpectralRadius', BOUNDARY)
    # A second common scalar enforces positive pressure for all samples used by
    # this flux. Multiplying coefficients preserves the evolved volume mean.
    marker = '''                _reconstruction.ApplyLimiter(
                    factors, _stateGradients, _coefficients);'''
    positivity = '''                for (index node = 0; node < _mesh->NumNode(); node++)
                {
                    const State mean = means[node];
                    const State rawPoint = _pointValues[node];
                    const Vector3 lengths = _reconstruction.ReferenceLengths(node);
                    real &factor = factors[static_cast<std::size_t>(node)];
                    const auto limit = [&](const State &candidate)
                    {
                        if (!IsPhysical(mean + factor * (candidate - mean)))
                        {
                            real lower = 0, upper = factor;
                            for (int iteration = 0; iteration < 48; iteration++)
                            {
                                const real middle = .5 * (lower + upper);
                                if (IsPhysical(mean + middle * (candidate - mean))) lower = middle;
                                else upper = middle;
                            }
                            factor = .95 * lower;
                        }
                    };
                    limit(rawPoint);
                    for (index neighbor : _reconstruction.DirectNeighbors(node))
                    {
                        limit(rawPoint + _coefficients[node].transpose() * Reconstruction::EvaluateBasis(
                            .5 * _nodeHalo.Displacement(node, neighbor), lengths, dimension));
                    }
                    for (const auto &piece : _geometry.NodeVolume(node).boundaryPieces)
                    {
                        Vector3 point = Vector3::Zero();
                        for (int vertex = 0; vertex < piece.nPoints; vertex++)
                            point += piece.points[vertex].coordinate / piece.nPoints;
                        limit(rawPoint + _coefficients[node].transpose() * Reconstruction::EvaluateBasis(
                            _nodeHalo.DisplacementToPoint(node, point), lengths, dimension));
                    }
                }
'''
    if text.count(marker) != 1:
        raise ValueError('Unexpected traditional limiter call')
    spatial.write_text(text.replace(marker, positivity + marker))
    reconstruction = sources / 'NCFV/NCFVReconstruction.cpp'
    reconstruction.write_text(replace_function(reconstruction.read_text(),
        '    std::vector<real> Reconstruction::ComputeLimiterFactors(',
        '    void Reconstruction::ApplyLimiter(', LIMITER))
    commands = json.loads((ROOT / 'build/compile_commands.json').read_text())
    selected = [r for r in commands if '/src/NCFV/' in r['file'] and
                ('INSTANTIATION_DIM=' not in r['command'] or 'INSTANTIATION_DIM=3' in r['command'])]
    selected += [next(r for r in commands if r['file'].endswith('/app/NCFV/ncfv_euler3D.cpp'))]
    def compile_one(record):
        original = Path(record['file'])
        source = sources / 'NCFV' / original.name if '/src/NCFV/' in record['file'] else sources / 'main.cpp'
        obj = sources / (original.stem + '.o')
        command = shlex.split(record['command'])
        command[command.index('-o') + 1] = str(obj)
        command[command.index('-c') + 1] = str(source)
        command[1:1] = ['-I' + str(sources)]
        with (sources / (original.stem + '.build.log')).open('x') as log:
            subprocess.run(command, cwd=ROOT / 'build', stdout=log, stderr=subprocess.STDOUT, check=True)
        return str(obj)
    with ThreadPoolExecutor(max_workers=4) as pool:
        objects = list(pool.map(compile_one, selected))
    raw = subprocess.check_output(['ninja', '-C', 'build', '-t', 'commands', 'app/ncfv_euler3D.exe'], cwd=ROOT, text=True).splitlines()[-1]
    command = shlex.split(raw.removeprefix(': && ').removesuffix(' && :'))
    command = [item for item in command if item != 'CMakeFiles/ncfv_euler3D.dir/app/NCFV/ncfv_euler3D.cpp.o' and
               item not in ('src/NCFV/libncfv_3D.a', 'src/NCFV/libncfv_core.a')]
    exe = out / 'midpoint_barth.exe'
    command[command.index('-o') + 1] = str(exe)
    command[1:1] = objects
    subprocess.run(command, cwd=ROOT / 'build', check=True)
    save(out / 'build_manifest.json', {'binary': str(exe), 'sha256': digest(exe),
        'sources': {str(p.relative_to(out)): digest(p) for p in sources.rglob('*') if p.suffix in ('.cpp', '.hpp')},
        'scheme': 'Mean-preserving quadratic SVD reconstruction; common conservative Barth coefficient; edge-midpoint Roe flux; boundary-piece centroid flux; sample positivity; SSPRK3'})
    return exe


def analyze(path, cells, directory):
    with h5py.File(path) as data:
        states = data['state/father/array/data'][:].reshape(-1, 5)
        ids = data['state/origIndex'][:].astype(np.int64)
        actual_time = float(data.attrs['simulationTime'])
        iteration = int(data.attrs['iteration'])
    if not np.array_equal(np.sort(ids), np.arange(6 * (cells + 1)**2)):
        raise ValueError('Incomplete node field')
    ordered = np.empty_like(states)
    ordered[ids] = states
    u = ordered.reshape(6, cells + 1, cells + 1, 5)
    rho = u[..., 0]
    pressure = .4 * (u[..., 4] - .5 * np.sum(u[..., 1:4]**2, axis=-1) / rho)
    velocity = u[..., 3] / rho
    if not np.isfinite(u).all() or rho.min() <= 0 or pressure.min() <= 0:
        raise ValueError('Nonphysical final field')
    result = {'time': actual_time, 'iteration': iteration, 'nodes': len(states),
        'rho_min': float(rho.min()), 'rho_max': float(rho.max()), 'p_min': float(pressure.min()), 'p_max': float(pressure.max()),
        'rho_z_span_max': float(np.ptp(rho, axis=0).max()), 'p_z_span_max': float(np.ptp(pressure, axis=0).max()),
        'w_abs_max': float(np.abs(velocity).max()),
        'rho_z_TV_excess_max': float(np.maximum(0, np.abs(np.diff(rho, axis=0)).sum(axis=0) - np.ptp(rho, axis=0)).max()),
        'restart': str(path), 'restart_sha256': digest(path)}
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/ncfv-midpoint-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)
    for ax, field, title in zip(axes[0], [rho[3], pressure[3], velocity[3]], ['Density (z=0.6 Lz)', 'Pressure (z=0.6 Lz)', 'Spanwise velocity w']):
        im = ax.imshow(field, origin='lower', extent=[0, 1, 0, 1], cmap='coolwarm')
        if np.ptp(field) > 1e-10:
            ax.contour(np.linspace(0, 1, cells+1), np.linspace(0, 1, cells+1), field, levels=25, colors='k', linewidths=.3)
        fig.colorbar(im, ax=ax)
        ax.set_title(title)
    for ax, field, title in zip(axes[1, :2], [rho, pressure], ['Density cuts', 'Pressure cuts']):
        for fraction in (.25, .5, .75):
            ax.plot(np.linspace(0, 1, cells+1), field[3, round(fraction*cells)], label=f'y={fraction}')
        ax.legend(); ax.set_title(title)
    im = axes[1, 2].imshow(np.ptp(rho, axis=0), origin='lower', extent=[0, 1, 0, 1], cmap='magma')
    fig.colorbar(im, ax=axes[1, 2]); axes[1, 2].set_title('Density max(z)-min(z)')
    fig.suptitle(f'RP1: quadratic edge-midpoint Roe + Barth, {cells}x{cells}x5, t={actual_time:.6g}')
    fig.savefig(directory / 'diagnostics.png', dpi=170)
    plt.close(fig)
    save(directory / 'metrics.json', result)
    np.savez_compressed(directory / 'final_fields.npz', conservative=u, pressure=pressure)
    return result, ordered


def run(exe, out, cells, ranks, end_time=.25, uniform=False):
    label = f'RP1_{cells}_np{ranks}' if not uniform else 'uniform40'
    directory = out / label
    directory.mkdir()
    overrides = {'/algorithm/mode': 'TraditionalQuadrature', '/reconstruction/limitPointRecovery': False,
        '/reconstruction/limitPressure': False,
        '/reconstruction/directionalPairBounds': False, '/reconstruction/preserveCoordinateInvariance': False,
        '/reconstruction/limitFluxCorrection': False, '/time/endTime': end_time,
        '/time/maximumTimeStep': 1e-4 if cells == 400 else .001,
        '/io/writeVTK': False, '/io/restartInterval': 100 if cells == 400 else 0,
        '/io/outputPrefix': str(directory/'solution'), '/io/vtkSeriesName': str(directory/'series'),
        '/io/restartPrefix': str(directory/'restart')}
    if cells == 40:
        audit = ROOT/'data/out/NCFV/riemann_barth_400x400x5/limiter_audit'
        overrides.update({'/mesh/meshFile': str(audit/'mesh40.cgns'), '/initialField/nodeFile': str(audit/'initial40/RP1.csv')})
    if uniform:
        overrides.update({'/initialField/nodeFile': '', '/initialField/requireCompleteNodeFile': False})
    command = ['mpirun', '--mca', 'pml', 'ob1', '--mca', 'btl', 'self,vader', '--bind-to', 'none', '-np', str(ranks), str(exe), str(out/'base_case.json')]
    for pointer, value in overrides.items():
        command += ['-k', pointer, '-v', json.dumps(value)]
    environment = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', DNDS_DIST_OMP_NUM_THREADS='1', HWLOC_COMPONENTS='-gl,-cuda,-nvml', OMPI_MCA_mpi_yield_when_idle='1')
    for key in list(environment):
        if key.startswith(('BJ_', 'NCFV_')):
            del environment[key]
    save(directory/'command.json', {'command': command, 'overrides': overrides})
    started = time.monotonic()
    with (directory/'solver.log').open('x') as log:
        process = subprocess.Popen(command, cwd=ROOT/'build', env=environment, stdout=log, stderr=subprocess.STDOUT)
        while process.poll() is None:
            save(out/'status.json', {'status': 'computing', 'run': label, 'pid': process.pid, 'elapsed_seconds': time.monotonic()-started, 'target_time': end_time, 'log': str(directory/'solver.log')})
            time.sleep(15)
    if process.returncode:
        raise RuntimeError(f'{label} failed, see {directory}/solver.log')
    restart = sorted(directory.glob('restart_*.dnds.h5'))[-1]
    metrics, field = analyze(restart, cells, directory)
    if abs(metrics['time'] - end_time) > 1e-12:
        raise ValueError('Incomplete physical end time')
    metrics.update(elapsed_seconds=time.monotonic()-started, ranks=ranks,
        rejected_steps=(directory/'solver.log').read_text().count('NCFV rejected iteration='))
    if uniform:
        metrics['max_uniform_error'] = float(np.abs(field - [1.5, 0, 0, 0, 3.75]).max())
        if metrics['max_uniform_error'] > 1e-11:
            raise ValueError('Uniform flow not preserved')
    save(directory/'metrics.json', metrics)
    print(json.dumps(metrics), flush=True)
    return metrics, field


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ranks', type=int, default=64)
    parser.add_argument('--build-from', type=Path, help='Reuse a frozen build from an earlier attempt')
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, out/'runner.py')
    shutil.copy2(CASE/'RP1_positive_tol1e10.json', out/'base_case.json')
    save(out/'status.json', {'status': 'building', 'started_utc': datetime.now(timezone.utc).isoformat()})
    try:
        if args.build_from:
            previous = args.build_from.resolve()
            shutil.copytree(previous/'sources', out/'sources')
            exe = out/'midpoint_barth.exe'
            shutil.copy2(previous/'midpoint_barth.exe', exe)
            manifest = json.loads((previous/'build_manifest.json').read_text())
            if digest(exe) != manifest['sha256']:
                raise ValueError('Reused executable checksum mismatch')
            manifest.update(binary=str(exe), reused_from=str(previous))
            save(out/'build_manifest.json', manifest)
        else:
            exe = build(out)
        report = {'uniform': run(exe, out, 40, 2, .001, True)[0]}
        a, ua = run(exe, out, 40, 1, .001)
        b, ub = run(exe, out, 40, 2, .001)
        report['mpi_short_check'] = {'np1': a, 'np2': b, 'max_state_difference': float(np.abs(ua-ub).max())}
        if report['mpi_short_check']['max_state_difference'] > 1e-8:
            raise ValueError('MPI consistency check failed')
        report['small_full_time'] = run(exe, out, 40, 4)[0]
        save(out/'validation.json', report)
        report['full_grid'] = run(exe, out, 400, args.ranks)[0]
        save(out/'validation.json', report)
        save(out/'status.json', {'status': 'complete', 'finished_utc': datetime.now(timezone.utc).isoformat(), 'full_grid': report['full_grid']})
    except Exception as error:
        save(out/'status.json', {'status': 'failed', 'error': str(error)})
        raise


if __name__ == '__main__':
    main()
