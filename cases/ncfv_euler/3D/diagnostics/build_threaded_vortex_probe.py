"""Build an isolated production-flux diagnostic with parallel independent edge evaluations."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
binary = args.output.resolve()
spatial_path = ROOT / "src/NCFV/NCFVSpatial.cpp"
benchmark_path = Path(__file__).with_name("benchmark_reconstruction.cpp")
spatial = spatial_path.read_text()
old_loop = """        for (index iEdge = 0; iEdge < _topology.NumEdge(); iEdge++)
        {
            _edgeFlux[iEdge] = EvaluateOwnedEdgeFlux(iEdge);"""
new_loop = """        const char *diagnosticEdgeThreadsValue = std::getenv("NCFV_EDGE_OMP_THREADS");
        const int diagnosticEdgeThreads = diagnosticEdgeThreadsValue
                                              ? std::max(1, std::atoi(diagnosticEdgeThreadsValue))
                                              : 1;
#pragma omp parallel for schedule(static) num_threads(diagnosticEdgeThreads)
        for (index iEdge = 0; iEdge < _topology.NumEdge(); iEdge++)
        {
            _edgeFlux[iEdge] = EvaluateOwnedEdgeFlux(iEdge);"""
old_counter = "        ++_riemannSolverCallCount;"
new_counter = "#pragma omp atomic update\n        ++_riemannSolverCallCount;"
assert spatial.count(old_loop) == 1 and spatial.count(old_counter) == 1
assert spatial.count('#include "NCFVSpatial.hpp"') == 1
spatial = spatial.replace(old_loop, new_loop).replace(old_counter, new_counter)
spatial = spatial.replace('#include "NCFVSpatial.hpp"',
                          f'#include "{ROOT / "src/NCFV/NCFVSpatial.hpp"}"')
source = binary.with_suffix(".cpp")
source.write_text("#define DNDS_NCFV_INSTANTIATION_DIM 3\n#include <cstdlib>\n" + spatial +
                  f'\n#include "{benchmark_path.resolve()}"\n')
subprocess.run([sys.executable, str(Path(__file__).with_name("compile_reconstruction_probe.py")),
                "--source", str(source), "--output", str(binary)], check=True)
metadata = {"spatial_source": str(spatial_path),
            "spatial_source_sha256": hashlib.sha256(spatial_path.read_bytes()).hexdigest(),
            "benchmark_source_sha256": hashlib.sha256(benchmark_path.read_bytes()).hexdigest(),
            "generated_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
            "changes": ["Parallel independent edge evaluations only",
                        "Atomic Riemann call counter"],
            "constraints": "Default inviscid LS/SVDLS flux; detailed timing and reference-document branches disabled"}
binary.with_suffix(".build.json").write_text(json.dumps(metadata, indent=2) + "\n")
