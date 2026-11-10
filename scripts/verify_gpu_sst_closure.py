#!/usr/bin/env python3
"""Probe the actual device SST closure with independent analytic test states.

The small CUDA harness extracts the functions verbatim from the production
header, so no second implementation can hide a GPU transcription error.
Expected values below are hand-derived homogeneous-flow limits of SST-1994.
"""
import argparse
import ctypes
import json
import os
from pathlib import Path
import subprocess

import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gpu-device',type=int,default=0)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True,exist_ok=False)
    source=(root/'src/GPU/ResidentACMSST.cuh').read_text()

    def function(name):
        marker=source.index(f' {name}(')
        begin=source.rfind('    __device__',0,marker)
        brace=source.index('{',marker)
        depth=1;end=brace+1
        while depth:
            depth+=(source[end]=='{')-(source[end]=='}');end+=1
        return source[begin:end]

    harness='''#include <cuda_runtime.h>
#include <cmath>
using I = long long;
struct Cell { double wallDistance; };
struct Settings {
    struct { double rho0=1; } flow;
    double turbulenceMinimum[2]{1e-14,1e-10}, turbulenceMaximum[2]{1e6,1e12};
    double turbulenceFar[2]{1e-6,1}, viscosity=.001, wallOmegaCoefficient=800;
    double minimumWallDistance=1e-10, eddyRatioMaximum=1e5;
    bool turbulenceSource=true;
};
struct View { Settings s; const Cell *cells; };
'''
    harness+='\n'.join(function(name) for name in ['QClamp','Positive','TurbulenceBoundary','Blend','Eddy','Diffusivity','SSTSource','CorrectGradient'])
    harness+='''
__global__ void Probe(const double *input, double *output) {
    View v; Cell cell{input[20]}; v.cells=&cell;
    const double *q=input, *gf=input+2, *gq=input+14;
    Blend(v,q,gq,cell.wallDistance,output[0],output[1]);
    output[2]=Eddy(v,q,gf,gq,cell.wallDistance);
    Diffusivity(v,q,gq,cell.wallDistance,output[2],output+3);
    SSTSource(v,q,gf,gq,cell.wallDistance,output+5);
    TurbulenceBoundary(v,0,2,q,output+7);
    const double left[1]{1e-14},right[1]{1+1e-14},gradient[3]{0,2,0};
    const double displacement[3]{1,1,0},normal[3]{1,0,0};
    CorrectGradient(left,right,gradient,gradient,displacement,normal,1,output+9);
}
extern "C" int probe(int device, const double *input, double *output) {
    cudaError_t error=cudaSetDevice(device); if(error) return int(error);
    double *a=nullptr,*b=nullptr;
    error=cudaMalloc(&a,21*sizeof(double));if(error) return int(error);
    error=cudaMalloc(&b,12*sizeof(double));if(error){cudaFree(a);return int(error);}
    error=cudaMemcpy(a,input,21*sizeof(double),cudaMemcpyHostToDevice);
    if(!error){Probe<<<1,1>>>(a,b);error=cudaMemcpy(output,b,12*sizeof(double),cudaMemcpyDeviceToHost);}
    cudaFree(a);cudaFree(b);return int(error);
}
'''
    src=args.output/'closure_probe.cu';lib=args.output/'closure_probe.so'
    src.write_text(harness)
    compiler=Path('/usr/local/cuda-13.2/bin/nvcc')
    subprocess.run([str(compiler),'-shared','-Xcompiler','-fPIC','-arch=sm_80','--fmad=false',str(src),'-o',str(lib)],check=True)
    dll=ctypes.CDLL(str(lib.resolve()))
    dll.probe.argtypes=[ctypes.c_int,np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS'),np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')]
    cases=[]

    def check(name,gradient,distance,expected,cross=0):
        value=np.zeros(21);value[:2]=[.2,4.];value[2:14]=np.asarray(gradient).ravel();value[20]=distance
        if cross: value[14]=1.;value[17]=cross
        output=np.zeros(12)
        status=dll.probe(args.gpu_device,value,output)
        if status: raise RuntimeError(f'CUDA probe failed: {status}')
        errors={}
        for index,target in expected.items():
            error=abs(output[index]-target)/max(abs(target),1.)
            if error>1e-12: raise ValueError(f'{name}: component {index} got {output[index]}, expected {target}')
            errors[str(index)]=float(error)
        cases.append(dict(case=name,passed=True,output=output.tolist(),scaled_errors=errors))

    zero=np.zeros((4,3))
    check('far_field_homogeneous_decay',zero,1e10,{0:0.,1:0.,2:.05,3:.051,4:.0438,5:-.072,6:-1.3248})
    shear=zero.copy();shear[0,1]=2.
    check('near_wall_homogeneous_shear',shear,.01,{0:1.,1:1.,2:.031,3:.02735,4:.0165,5:.052,6:1.0126666666666666,7:-.2,8:15996.})
    rotation=zero.copy();rotation[0,1]=-2.;rotation[1,0]=2.
    check('rigid_rotation_has_no_strain_production',rotation,.01,{2:.0155,5:-.072,6:-1.2})
    check('positive_cross_diffusion',zero,1e10,{6:-.4688},cross=2.)
    check('negative_cross_diffusion',zero,1e10,{6:-2.1808},cross=-2.)
    normal_derivative=cases[-1]['output'][9]
    if abs(normal_derivative+1)>1e-12:
        raise ValueError('Nonorthogonal diffusion probe did not reproduce the current operator')
    cases.append(dict(case='nonorthogonal_diffusion_counterexample_reproduced',passed=True,
                      satisfies_discrete_maximum_principle=False,normal_derivative=normal_derivative,
                      explanation='Left state is a strict minimum. The corrected normal gradient is negative, so diffusion can decrease it further.'))
    (args.output/'summary.json').write_text(json.dumps(cases,indent=2)+'\n')
    print(json.dumps([dict(case=x['case'],passed=x['passed']) for x in cases],indent=2))


if __name__=='__main__':
    main()
