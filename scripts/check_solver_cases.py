#!/usr/bin/env python3
"""Validate JSONC case selection, schema links and native solver configurations.

This tool launches C++ executables directly and does not load Python bindings.
Run from any directory: python scripts/check_solver_cases.py --build-dir build.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    'NS': 'euler', 'NS_2D': 'euler2D', 'NS_3D': 'euler3D',
    'NS_SA': 'eulerSA', 'NS_SA_3D': 'eulerSA3D',
    'NS_2EQ': 'euler2EQ', 'NS_2EQ_3D': 'euler2EQ3D',
    'NS_EX': 'eulerEX', 'NS_EX_3D': 'eulerEX3D',
    'ConstantDensity2D': 'acm2D', 'ConstantDensity3D': 'acm3D',
    'VariableDensity2D': 'acmVariable2D', 'VariableDensity3D': 'acmVariable3D',
}

def read_jsonc(path):
    pattern = r'"(?:\\.|[^"\\])*"|//[^\n]*|/\*[\s\S]*?\*/'
    text = re.sub(pattern, lambda m: m[0] if m[0].startswith('"') else '', path.read_text())
    return json.loads(text)

def executable_for(document):
    selector = document['solver']
    if selector['type'] == 'ncfv_euler':
        if selector['discretization'] != 'NCFV' or selector['model'] != 'IdealGas':
            raise ValueError('Invalid ncfv_euler selector')
        return f"ncfv_euler{document['dimension']}D"
    if selector['discretization'] != 'CFV':
        raise ValueError('Non-NCFV solver requires CFV')
    return MODELS[selector['model']]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir', type=Path, default=ROOT / 'build')
    parser.add_argument('--report', type=Path)
    parser.add_argument('--jobs', type=int, default=4)
    parser.add_argument('--layout-only', action='store_true')
    parser.add_argument('--negative-only', action='store_true')
    parser.add_argument('--reactive-build-dir', type=Path,
                        help='Cantera-enabled build for cases with reactiveFlow.enabled=true')
    args = parser.parse_args()
    build = args.build_dir.resolve()
    cases = []
    for path in sorted((ROOT / 'cases').rglob('*.json')):
        document = read_jsonc(path)
        if isinstance(document, dict) and 'solver' in document:
            cases.append((path, document))

    def check(item):
        path, document = item
        result = {'case': str(path.relative_to(ROOT))}
        try:
            exe = executable_for(document)
            result['executable'] = exe
            expected_family = 'acm' if exe.startswith('acm') else 'ncfv_euler' if exe.startswith('ncfv') else 'euler'
            if path.relative_to(ROOT / 'cases').parts[0] != expected_family:
                raise ValueError('Case is stored under the wrong solver family')
            schema = (path.parent / document['$schema']).resolve()
            if not schema.is_file():
                raise ValueError('Missing schema: ' + str(schema))
            if not args.layout_only:
                selected_build = build
                if document.get('eulerSettings', {}).get('reactiveFlow', {}).get('enabled', False):
                    if args.reactive_build_dir:
                        selected_build = args.reactive_build_dir.resolve()
                    else:
                        # Query the compiled capability; configuration schemas reflect Cantera support.
                        capability = subprocess.run(
                            [str(build / 'app' / (exe + '.exe')), '--emit-schema'],
                            cwd=build, capture_output=True, text=True, timeout=30, check=True)
                        native_schema = json.loads(capability.stdout[capability.stdout.index('{'):])
                        enabled = native_schema['properties']['eulerSettings']['properties']['reactiveFlow']['properties']['enabled']
                        if enabled.get('const') is False:
                            return {**result, 'passed': True, 'skipped': 'Requires a Cantera-enabled build'}
                process = subprocess.run(
                    [str(selected_build / 'app' / (exe + '.exe')), str(path), '--check-config'],
                    cwd=selected_build, capture_output=True, text=True, timeout=90,
                    env={**os.environ, 'OMP_NUM_THREADS': '1'})
                if process.returncode:
                    raise ValueError((process.stderr + process.stdout)[-4000:])
            result['passed'] = True
        except Exception as error:
            result.update(passed=False, error=str(error))
        return result

    if args.negative_only:
        results = []
        # Each executable must reject a selector that belongs to another model.
        for exe in ('euler', 'euler2D', 'euler3D', 'acm2D', 'acm3D',
                    'acmVariable2D', 'acmVariable3D', 'ncfv_euler2D', 'ncfv_euler3D'):
            path = next(path for path, doc in cases if executable_for(doc) == exe)
            process = subprocess.run(
                [str(build / 'app' / (exe + '.exe')), str(path), '--check-config',
                 '-k', '/solver/model', '-v', 'InvalidModel'],
                cwd=build, capture_output=True, text=True, timeout=90)
            results.append({'case': str(path.relative_to(ROOT)), 'executable': exe,
                            'passed': process.returncode != 0 and
                            'solver metadata does not match' in process.stderr,
                            'error': process.stderr[-2000:]})
        for dimension in (2, 3):
            exe = f'ncfv_euler{dimension}D'
            path = next(path for path, doc in cases if executable_for(doc) == exe)
            process = subprocess.run(
                [str(build / 'app' / (exe + '.exe')), str(path), '--check-config',
                 '-k', '/dimension', '-v', str(5 - dimension)],
                cwd=build, capture_output=True, text=True, timeout=90)
            results.append({'case': str(path.relative_to(ROOT)), 'executable': exe,
                            'passed': process.returncode != 0,
                            'error': process.stderr[-2000:]})
    else:
        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            results = list(executor.map(check, cases))
    failures = [r for r in results if not r['passed']]
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(results, indent=2, ensure_ascii=False) + '\n')
    skipped = [r for r in results if r.get('skipped')]
    print(f'{len(results) - len(failures) - len(skipped)}/{len(results)} configurations passed; {len(skipped)} require another build')
    for result in skipped:
        print(f"{result['case']}: {result['skipped']}")
    for failure in failures:
        print(f"{failure['case']}: {failure['error']}")
    return int(bool(failures))

if __name__ == '__main__':
    raise SystemExit(main())
