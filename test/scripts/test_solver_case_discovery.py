"""Case checks must not parse untracked numerical output during CTest."""

import importlib.util
from pathlib import Path
import subprocess

import pytest


@pytest.fixture
def checker(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[2] / 'scripts/check_solver_cases.py'
    spec = importlib.util.spec_from_file_location('solver_case_checker', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    cases = tmp_path / 'cases'
    cases.mkdir()
    return module, cases


def test_tracked_cases_exclude_local_output_and_keep_spaced_paths(checker):
    module, cases = checker
    subprocess.run(['git', 'init', '-q', str(module.ROOT)], check=True)
    tracked = cases / 'flow example.json'
    tracked.write_text('{"solver": {}}')
    subprocess.run(['git', 'add', '--', str(tracked)], cwd=module.ROOT, check=True)
    output = cases / 'constraints_rank0.json'
    output.write_text('not a configuration and deliberately invalid JSON')
    assert module.case_paths(tracked_only=True) == [tracked]
    assert module.case_paths() == sorted([tracked, output])


def test_source_archive_falls_back_to_its_packaged_cases(checker, monkeypatch):
    module, cases = checker
    packaged = cases / 'flow.json'
    packaged.write_text('{"solver": {}}')

    def no_git(*args, **kwargs):
        raise FileNotFoundError('Git is unavailable in this source archive')

    monkeypatch.setattr(module.subprocess, 'run', no_git)
    assert module.case_paths(tracked_only=True) == [packaged]
