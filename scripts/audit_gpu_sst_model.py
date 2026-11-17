#!/usr/bin/env python3
"""Check model-version limits and corrected production CUDA SST spatial fluxes.

Builds a small isolated harness from production device functions. It does not
change the solver, load the SUBOFF mesh, or advance a saved calculation.
"""
import argparse
import ctypes
import json
from pathlib import Path
import subprocess

import numpy as np


def extract_function(source, name, qualifier='__device__'):
    marker = source.index(f' {name}(')
    begin = source.rfind(f'    {qualifier}', 0, marker)
    brace = source.index('{', marker)
    depth, end = 1, brace + 1
    while depth:
        depth += (source[end] == '{') - (source[end] == '}')
        end += 1
    return source[begin:end]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu-device', type=int, default=1)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    sst = (root/'src/GPU/ResidentACMSST.cuh').read_text()
    flow = (root/'src/GPU/ResidentACM.cu').read_text()
    harness = r'''
#include <cuda_runtime.h>
#include <cmath>
#include <cstring>
#include "GPU/SSTSpatial.hpp"
using namespace DNDS::GPU;
using I = long long;
struct Cell { double wallDistance=1; };
struct Settings
{
    struct { double rho0=1, alpha=.5, beta2=4; } flow;
    int dimension=3;
    bool turbulenceSecondOrder=false, turbulenceSource=true, sst=false, includeIsotropicStress=true;
    double turbulenceMinimum[2]{1e-14,1e-10}, turbulenceMaximum[2]{1e6,1e12};
    double turbulenceFar[2]{1e-6,1}, viscosity=.001, wallOmegaCoefficient=800;
    double minimumWallDistance=1e-10, eddyRatioMaximum=1e5;
};
struct ResidentACMFace
{
    I left=0, right=-1, begin=0, end=1;
    int type=1;
    double wallDistance=1, normal[3]{1,0,0}, prescribed[4]{-1,0,0,0};
    double rotation[9]{1,0,0,0,1,0,0,0,1};
};
struct ResidentACMPoint
{
    double normal[3]{1,0,0}, displacement[3]{2,0,0}, measure=1, delta[2][3]{};
};
struct View
{
    Settings s;
    I faceCount=1;
    Cell *cells=nullptr;
    ResidentACMFace *faces=nullptr;
    ResidentACMPoint *points=nullptr;
    double *q=nullptr, *u=nullptr, *qGradient=nullptr, *flowGradient=nullptr;
    double *qLimiter=nullptr, *eddy=nullptr, *volumeFlux=nullptr, *qFlux=nullptr, *qRate=nullptr;
    double *qExtrema=nullptr;
};
__device__ I Thread() { return I(blockIdx.x)*blockDim.x+threadIdx.x; }
'''
    harness += '\n'.join(extract_function(flow, name) for name in [
        'Rotate', 'Boundary', 'KineticValue', 'KineticPressures', 'StaticBoundary'])
    harness += '\n'.join(extract_function(sst, name) for name in [
        'QClamp', 'Positive', 'TurbulenceBoundary', 'Blend', 'Eddy', 'Diffusivity',
        'SSTSource', 'CorrectGradient', 'SSTPointData', 'SSTAdvectiveValue'])
    harness += '\n'.join(extract_function(sst, name, '__global__') for name in ['SSTEddy', 'SSTFlux', 'SSTImplicitFaces'])
    harness += r'''
__global__ void Closure(const double *a, double *b)
{
    View v;
    v.s.flow.rho0=a[21]; v.s.viscosity=a[22];
    Cell cell; cell.wallDistance=a[20]; v.cells=&cell;
    Blend(v,a,a+14,a[20],b[0],b[1]);
    b[2]=Eddy(v,a,a+2,a+14,a[20]);
    Diffusivity(v,a,a+14,a[20],b[2],b+3);
    SSTSource(v,a,a+2,a+14,a[20],b+5);
    TurbulenceBoundary(v,0,2,a,b+7);
}
extern "C" int closure_probe(int device, const double *input, double *output)
{
    auto error=cudaSetDevice(device); if(error) return int(error);
    double *a=nullptr, *b=nullptr;
    error=cudaMallocManaged(&a,23*sizeof(double)); if(error) return int(error);
    error=cudaMallocManaged(&b,9*sizeof(double)); if(error) { cudaFree(a); return int(error); }
    std::memcpy(a,input,23*sizeof(double));
    Closure<<<1,1>>>(a,b);
    error=cudaDeviceSynchronize();
    if(!error) std::memcpy(output,b,9*sizeof(double));
    cudaFree(a); cudaFree(b); return int(error);
}
struct Storage
{
    View v;
    Cell cell[1]; ResidentACMFace face[1]; ResidentACMPoint point[1];
    double q[2]{.01,5}, u[4]{-1,0,0,0}, qGradient[6]{}, flowGradient[12]{};
    double limiter[3]{1,1,1}, extrema[2]{}, eddy[1]{}, flux[1]{-1}, qFlux[2]{}, rate[2]{}, output[14]{};
    double blocks[4]{}, advected[2]{};
};
__global__ void ReadPoint(View v, double *out)
{
    double ql[2],qr[2],gq[6],gf[12],mean[2],diff[2];
    SSTPointData(v,v.faces[0],v.points[0],ql,qr,gq,gf);
    for(int k=0;k<2;++k) mean[k]=QClamp(v,.5*(ql[k]+qr[k]),k);
    Diffusivity(v,mean,gq,1,v.eddy[0],diff);
    for(int k=0;k<2;++k)
    {
        out[k]=ql[k]; out[2+k]=qr[k]; out[4+k]=v.qFlux[k];
        out[6+k]=-v.volumeFlux[0]*v.s.turbulenceFar[k]+diff[k]*gq[k*3];
        out[8+k]=v.volumeFlux[0]*SSTAdvectiveValue(v,v.faces[0],v.volumeFlux[0],ql,qr,k);
    }
}
extern "C" int boundary_probe(int device, int type, double *output)
{
    auto error=cudaSetDevice(device); if(error) return int(error);
    Storage *s=nullptr;
    error=cudaMallocManaged(&s,sizeof(Storage)); if(error) return int(error);
    *s=Storage{};
    auto &v=s->v;
    v.s.viscosity=1e-12;
    v.s.eddyRatioMaximum=0; // Isolate the exact boundary-advection and diffusion derivatives.
    v.s.turbulenceFar[0]=.001; v.s.turbulenceFar[1]=1;
    s->face[0].type=type%100;
    v.s.turbulenceSecondOrder=(type/100)%2;
    if(type>=200 && type<400) s->flux[0]=1;
    if(type>=400) s->u[0]=1; // Roe inflow can differ from the sign of reconstructed velocity.
    if(v.s.turbulenceSecondOrder)
    {
        // A zero limiter is permitted by the actual local-extrema limiter.
        s->limiter[0]=s->limiter[1]=s->limiter[2]=0;
        s->qGradient[0]=-.0045; s->qGradient[3]=-2;
        s->point[0].delta[0][0]=1;
    }
    v.cells=s->cell; v.faces=s->face; v.points=s->point; v.q=s->q; v.u=s->u;
    v.qGradient=s->qGradient; v.flowGradient=s->flowGradient; v.qLimiter=s->limiter;
    v.qExtrema=s->extrema;
    v.eddy=s->eddy; v.volumeFlux=s->flux; v.qFlux=s->qFlux; v.qRate=s->rate;
    SSTEddy<<<1,1>>>(v);
    SSTFlux<<<1,1>>>(v);
    SSTImplicitFaces<<<1,1>>>(v,s->blocks,s->advected);
    ReadPoint<<<1,1>>>(v,s->output);
    error=cudaDeviceSynchronize();
    if(!error)
    {
        std::memcpy(output,s->output,10*sizeof(double));
        std::memcpy(output+10,s->blocks,2*sizeof(double));
        std::memcpy(output+12,s->advected,2*sizeof(double));
    }
    cudaFree(s); return int(error);
}
extern "C" int diffusion_probe(int device, int mode, double *output)
{
    auto error=cudaSetDevice(device); if(error) return int(error);
    Storage *s=nullptr;
    error=cudaMallocManaged(&s,sizeof(Storage)); if(error) return int(error);
    *s=Storage{};
    // Local-minimum and local-maximum counterexamples, and an unconstrained affine stencil.
    s->q[0]=mode==0?.01:3; s->q[1]=mode==0?1:5;
    s->qGradient[0]=mode==1?-.5:0; s->qGradient[1]=mode==1?-.5:mode==0?2:-2;
    s->qGradient[3]=s->qGradient[0]; s->qGradient[4]=s->qGradient[1];
    s->extrema[0]=s->extrema[1]=mode==0?1:mode==2?2:0;
    // Use a Dirichlet ghost to supply a controlled difference; zero advection.
    s->v.s.turbulenceFar[0]=s->q[0]+(mode==0?.5:-.5);
    s->v.s.turbulenceFar[1]=s->q[1]+(mode==0?.5:-.5);
    s->v.s.viscosity=1; s->v.s.eddyRatioMaximum=0;
    s->face[0].type=7; s->point[0].displacement[0]=s->point[0].displacement[1]=1;
    s->flux[0]=0;
    auto &v=s->v;
    v.cells=s->cell; v.faces=s->face; v.points=s->point; v.q=s->q; v.u=s->u;
    v.qGradient=s->qGradient; v.flowGradient=s->flowGradient; v.qLimiter=s->limiter;
    v.qExtrema=s->extrema; v.eddy=s->eddy; v.volumeFlux=s->flux; v.qFlux=s->qFlux; v.qRate=s->rate;
    SSTEddy<<<1,1>>>(v); SSTFlux<<<1,1>>>(v);
    error=cudaDeviceSynchronize();
    if(!error) std::memcpy(output,s->qFlux,2*sizeof(double));
    cudaFree(s); return int(error);
}
__global__ void ReadBoundedFace(Storage *s)
{
    double ql[2],qr[2],gq[6],gf[12],c[2],d[2];
    SSTPointData(s->v,s->face[0],s->point[0],ql,qr,gq,gf,c,d);
    for(int k=0;k<2;++k)
    {
        s->output[k]=s->qFlux[k];
        s->output[2+k]=d[k];
        s->output[4+k]=c[k];
        s->output[6+k]=s->blocks[k];
        s->output[8+k]=qr[k]-s->q[k];
    }
}
extern "C" int bounded_face_probe(int device, int mode, double *output)
{
    auto error=cudaSetDevice(device); if(error) return int(error);
    Storage *s=nullptr;
    error=cudaMallocManaged(&s,sizeof(Storage)); if(error) return int(error);
    *s=Storage{};
    s->q[0]=1e-14; s->q[1]=1e-10;
    if(mode==1) { s->q[0]=1.0000000000005337e-14; s->q[1]=1.0000000000005337e-10; }
    s->v.s.turbulenceFar[0]=mode==1?.5*(s->q[0]+1.0000000000000247e-14):s->q[0];
    s->v.s.turbulenceFar[1]=mode==1?.5*(s->q[1]+1.0000000000000247e-10):s->q[1];
    if(mode==2 || mode==3)
    {
        s->q[0]=3; s->q[1]=5;
        s->v.s.turbulenceFar[0]=2.5; s->v.s.turbulenceFar[1]=4.5;
    }
    for(int k=0;k<2;++k)
    { s->qGradient[3*k]=mode==3?1e18:-1e18; s->qGradient[3*k+1]=mode==3?-3e18:3e18; }
    s->v.s.viscosity=1; s->v.s.eddyRatioMaximum=0;
    s->face[0].type=7; s->flux[0]=0;
    s->point[0].displacement[0]=s->point[0].displacement[1]=1;
    if(mode==4) { s->point[0].normal[0]=.6; s->point[0].normal[1]=.8; }
    auto &v=s->v;
    v.cells=s->cell; v.faces=s->face; v.points=s->point; v.q=s->q; v.u=s->u;
    v.qGradient=s->qGradient; v.flowGradient=s->flowGradient; v.qLimiter=s->limiter;
    v.qExtrema=s->extrema; v.eddy=s->eddy; v.volumeFlux=s->flux; v.qFlux=s->qFlux; v.qRate=s->rate;
    SSTEddy<<<1,1>>>(v); SSTFlux<<<1,1>>>(v);
    SSTImplicitFaces<<<1,1>>>(v,s->blocks,s->advected);
    ReadBoundedFace<<<1,1>>>(s);
    error=cudaDeviceSynchronize();
    if(!error) std::memcpy(output,s->output,10*sizeof(double));
    cudaFree(s); return int(error);
}
'''
    src, lib = args.output/'model_probe.cu', args.output/'model_probe.so'
    src.write_text(harness)
    subprocess.run(['/usr/local/cuda-13.2/bin/nvcc', '-shared', '-Xcompiler', '-fPIC',
                    '-arch=sm_80', '--fmad=false', '-I'+str(root/'src'), str(src), '-o', str(lib)], check=True)
    dll = ctypes.CDLL(str(lib.resolve()))
    pointer = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
    dll.closure_probe.argtypes = [ctypes.c_int, pointer, pointer]
    dll.boundary_probe.argtypes = [ctypes.c_int, ctypes.c_int, pointer]
    dll.diffusion_probe.argtypes = [ctypes.c_int, ctypes.c_int, pointer]
    dll.bounded_face_probe.argtypes = [ctypes.c_int, ctypes.c_int, pointer]

    def closure(q, gradient, distance, rho=1., mu=.001):
        a, b = np.zeros(23), np.zeros(9)
        a[:2] = q
        a[2:11] = np.asarray(gradient).ravel()
        a[20:] = [distance, rho, mu]
        error = dll.closure_probe(args.gpu_device, a, b)
        if error:
            raise RuntimeError(f'CUDA closure probe: {error}')
        return b

    shear = np.zeros((3,3)); shear[0,1] = 100
    high = closure([.2,4], shear, .01)
    gamma1 = .075/.09-.5*.41**2/np.sqrt(.09)
    expected94 = [20*.09*.2*4-.09*.2*4, gamma1*10000-.075*16]
    expected03 = [10*.09*.2*4-.09*.2*4, (5/9)*(10*.09*.2*4)/.00062-.075*16]
    np.testing.assert_allclose(high[5:7],expected94,rtol=1e-12,atol=1e-12)
    ext = np.diag([50.,-50.,0.])
    extension = closure([.2,4], ext, .01)
    np.testing.assert_allclose(extension[2],.05,rtol=1e-12)
    simple = shear/50
    normal = closure([.2,4],simple,.01)
    scaled = closure([.2,4],simple,.01,rho=5,mu=.005)
    np.testing.assert_allclose(scaled[[0,1,3,4,5,6,7,8]],normal[[0,1,3,4,5,6,7,8]],rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(scaled[2],5*normal[2],rtol=1e-12)
    mu = 3.630083418e-7
    transition = closure([1e-14,.2],simple,.03,mu=mu)
    arg94 = max(np.sqrt(1e-14)/(.09*.2*.03),500*mu/(.03**2*.2))
    f194 = float(np.tanh(arg94**4))
    np.testing.assert_allclose(transition[0],f194,rtol=1e-12,atol=1e-12)
    report = dict(
        source_functions='Verbatim production CUDA functions; full SSTEddy/SSTFlux kernels execute in the inflow test.',
        high_shear=dict(actual_source=high[5:7].tolist(),SST1994_source=expected94,
                        SST2003m_source=expected03,matches_1994=True),
        incompressible_extension=dict(actual_mu_t=float(extension[2]),SST1994_mu_t=.05,SST2003_mu_t=.00062),
        density_scaling=dict(rho_factor=5,kinematic_diffusion_and_sources_unchanged=True),
        blending_floor=dict(state=[1e-14,.2],distance=.03,viscosity=mu,
                                    actual_F1=float(transition[0]),SST1994_F1=f194,
                                    actual_CD_floor=1e-20,SST1994_CD_floor=1e-20,matches_1994=True),
        boundary_fluxes=[])
    boundary_cases = [(kind, name, second_order)
                      for second_order in [False, True]
                      for kind, name in [(1,'BCFar'),(7,'BCIn'),(8,'BCInPsTs'),(10,'BCSpecial')]]
    for kind, name, second_order in boundary_cases:
        out = np.zeros(14)
        error = dll.boundary_probe(args.gpu_device,kind+100*int(second_order),out)
        if error:
            raise RuntimeError(f'CUDA boundary probe: {error}')
        np.testing.assert_allclose(out[4:6],out[6:8],rtol=1e-12,atol=1e-14)
        np.testing.assert_allclose(out[8:10],[-.001,-1],rtol=1e-12,atol=1e-14)
        np.testing.assert_allclose(out[10:12],[1e-12,1e-12],rtol=1e-10,atol=1e-14)
        np.testing.assert_allclose(out[12:14],[.001,1],rtol=1e-12,atol=1e-14)
        if not np.all(out[2:4]<0):
            raise ValueError('Dirichlet diffusion ghost was incorrectly clamped')
        report['boundary_fluxes'].append(dict(type=name,second_order=second_order,
            limiter=0 if second_order else None,interior=[.01,5],prescribed=[.001,1],normal_velocity=-1,
            actual_exterior_state=out[2:4].tolist(),actual_face_rhs=out[4:6].tolist(),
            expected_face_rhs_using_prescribed_advective_state=out[6:8].tolist(),
            actual_advective_flux=out[8:10].tolist(),implicit_diagonal=out[10:12].tolist(),
            implicit_advected_state=out[12:14].tolist(),implementation_defect_reproduced=False,passed=True))
    report['advection_direction_checks']=[]
    for mode,label in [(200,'outgoing_flux_uses_owner'),(400,'incoming_flux_overrides_outward_velocity')]:
        for kind,name,second_order in boundary_cases:
            out=np.zeros(14)
            error=dll.boundary_probe(args.gpu_device,mode+kind+100*int(second_order),out)
            if error: raise RuntimeError(f'CUDA directional boundary probe: {error}')
            expected=[.01,5] if mode==200 else [.001,1]
            np.testing.assert_allclose(out[12:14],expected,rtol=1e-12,atol=1e-14)
            np.testing.assert_allclose(out[8:10],np.asarray(expected)*(1 if mode==200 else -1),rtol=1e-12,atol=1e-14)
            if mode==200: np.testing.assert_allclose(out[10:12],[1,1],rtol=1e-10,atol=1e-14)
            else: np.testing.assert_allclose(out[4:6],out[6:8],rtol=1e-12,atol=1e-14)
            report['advection_direction_checks'].append(dict(case=label,type=name,second_order=second_order,passed=True))
    report['bounded_diffusion']=[]
    for mode,name,expected in [(0,'minimum_cannot_diffuse_down',[0,0]),
                               (2,'maximum_cannot_diffuse_up',[0,0]),
                               (1,'admissible_affine_nonorthogonal_gradient_retained',[-.5,-.5])]:
        out=np.zeros(2)
        error=dll.diffusion_probe(args.gpu_device,mode,out)
        if error: raise RuntimeError(f'CUDA diffusion probe: {error}')
        np.testing.assert_allclose(out,expected,rtol=1e-12,atol=1e-12)
        report['bounded_diffusion'].append(dict(case=name,face_diffusion_rhs=out.tolist(),passed=True))
    report['plateau_flux_and_frozen_jacobian']=[]
    for mode,name in enumerate(['equal_states','near_floor_platform','large_aligned_correction',
                                'reversed_correction','oblique_normal_cancellation']):
        out=np.zeros(10)
        error=dll.bounded_face_probe(args.gpu_device,mode,out)
        if error: raise RuntimeError(f'CUDA bounded face probe: {error}')
        if not np.isfinite(out).all(): raise ValueError('Nonfinite limited flux')
        np.testing.assert_allclose(out[:2],out[2:4],rtol=1e-14,atol=0)
        np.testing.assert_allclose(out[:2],out[4:6]*out[8:10],rtol=1e-14,atol=0)
        np.testing.assert_allclose(out[6:8],2*out[4:6],rtol=1e-14,atol=0)
        if np.any(out[4:6]<0) or np.any(out[4:6]>2): raise ValueError('Invalid face conductance')
        if mode in (0,4): np.testing.assert_array_equal(out[:2],[0,0])
        report['plateau_flux_and_frozen_jacobian'].append(dict(case=name,passed=True,
            flux=out[:2].tolist(),conductance=out[4:6].tolist(),implicit_diagonal=out[6:8].tolist()))
    (args.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
