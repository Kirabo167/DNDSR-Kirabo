#!/usr/bin/env python3
"""Check production CUDA pressure helpers against analytic static-pressure states."""
import argparse
import ctypes
import json
from pathlib import Path
import subprocess

import numpy as np

from audit_gpu_sst_model import extract_function


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu-device', type=int, default=1)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True, exist_ok=False)
    source = (root/'src/GPU/ResidentACM.cu').read_text()
    harness = r'''
#include <cuda_runtime.h>
#include <cstring>
#include "GPU/FluxMath.cuh"
using namespace DNDS::GPU;
using I = long long;
struct Settings {
    FluxInput flow;
    bool sst=true, includeIsotropicStress=true, turbulenceSecondOrder=false;
    int dimension=3;
    double turbulenceMinimum[2]{1e-14,1e-10}, turbulenceMaximum[2]{1e6,1e12};
    double turbulenceFar[2]{.05,1};
};
struct ResidentACMFace { I left=0, right=1; int type=2; double prescribed[4]{}; };
struct ResidentACMPoint { double normal[3]{1,0,0}, delta[2][3]{}; };
struct View {
    Settings s;
    double *q, *qGradient, *qLimiter;
};
'''
    for name in ['Boundary', 'KineticValue', 'KineticPressures', 'StaticBoundary', 'EffectiveFaceStates']:
        harness += extract_function(source, name)+'\n'
    harness += r'''
__global__ void Probe(int mode, double *out) {
    View v; v.s.flow.rho0=2; v.s.flow.beta2=4; v.s.flow.alpha=.5;
    double q[4]{.3,4,.6,2}, gradient[12]{}, limiter[6]{.5,1,1,1,1,1};
    v.q=q; v.qGradient=gradient; v.qLimiter=limiter;
    ResidentACMFace face; ResidentACMPoint point;
    FluxInput input=v.s.flow; input.left[3]=.1; input.right[3]=-.3;
    double dl=0, dr=0;
    if(mode==0) { dl=.03; dr=-.06; }
    if(mode==1) face.right=-1;
    if(mode==3) v.s.includeIsotropicStress=false;
    if(mode==4) {
        v.s.turbulenceSecondOrder=true; gradient[0]=.2; point.delta[0][0]=.25; dl=.03;
    }
    if(mode==5) v.s.sst=false;
    KineticPressures(v,0,face.right,face.type,point.normal,input.left,
                    point.delta[0],point.delta[1],dl,dr,out);
    double ghost[4];
    StaticBoundary(v,0,2,face.prescribed,point.normal,input.left,ghost);
    out[3]=ghost[3];
    for(int d=0;d<3;++d) input.normal[d]=point.normal[d];
    EffectiveFaceStates(v,face,point,input,dl,dr);
    out[4]=input.left[3]; out[5]=input.right[3];
    FluxOutput flux; Detail::ACMFlux(input,flux);
    for(int k=0;k<4;++k) out[6+k]=flux.value[k];
}
extern "C" int probe(int device,int mode,double *output) {
    auto error=cudaSetDevice(device); if(error) return int(error);
    double *b=nullptr; error=cudaMallocManaged(&b,10*sizeof(double)); if(error) return int(error);
    Probe<<<1,1>>>(mode,b); error=cudaDeviceSynchronize();
    if(!error) std::memcpy(output,b,10*sizeof(double));
    cudaFree(b); return int(error);
}
'''
    src, lib = args.output/'stress_probe.cu', args.output/'stress_probe.so'
    src.write_text(harness)
    subprocess.run(['/usr/local/cuda-13.2/bin/nvcc', '-shared', '-Xcompiler', '-fPIC',
                    '-arch=sm_80', '--fmad=false', '-I'+str(root/'src'), str(src), '-o', str(lib)], check=True)
    dll = ctypes.CDLL(str(lib.resolve()))
    dll.probe.argtypes = [ctypes.c_int, ctypes.c_int,
                         np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')]
    cases = [
        ('conservative_k_perturbations', 0, {0:.42, 1:.76, 2:.59, 4:.52, 5:.46}),
        ('wall_total_stress_extrapolation', 1, {0:.4, 1:-.4, 2:0., 3:.9, 4:.5, 5:.5,
                                              6:.25, 7:0., 8:0., 9:0.}),
        ('static_pressure_compensates_k_jump', 2, {0:.4, 1:.8, 4:.5, 5:.5,
                                                 6:.25, 7:0., 8:0., 9:0.}),
        ('legacy_pressure_option', 3, {0:0., 1:0., 2:0., 3:.1, 4:.1, 5:-.3}),
        ('limited_reconstruction_and_conservative_increment', 4,
         {0:(2./3)*2*.325+.02, 1:.8, 4:.1+(2./3)*2*.325+.02}),
        ('laminar_has_no_kinetic_pressure', 5, {0:0., 1:0., 2:0., 3:.1, 4:.1, 5:-.3}),
    ]
    results = []
    for name, mode, expected in cases:
        out = np.zeros(10)
        status = dll.probe(args.gpu_device, mode, out)
        if status:
            raise RuntimeError(f'CUDA stress probe failed: {status}')
        for index, value in expected.items():
            np.testing.assert_allclose(out[index], value, rtol=1e-12, atol=1e-13, err_msg=name)
        results.append(dict(case=name, passed=True, output=out.tolist()))
    (args.output/'summary.json').write_text(json.dumps(results, indent=2)+'\n')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
