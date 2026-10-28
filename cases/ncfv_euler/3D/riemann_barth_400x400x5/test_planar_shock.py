"""Check a planar RP1 shock against its exact 1-D solution with z periodicity."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[4]
CASE = Path(__file__).resolve().parent
GAMMA = 1.4
LEFT = (.5323, 1.206, .3)
RIGHT = (1.5, 0., 1.5)


def exact_shock(time):
    def wave(p, state):
        rho, velocity, pressure = state
        sound = math.sqrt(GAMMA*pressure/rho)
        if p > pressure:
            return (p-pressure)*math.sqrt(2/((GAMMA+1)*rho)/(p+(GAMMA-1)/(GAMMA+1)*pressure))
        return 2*sound/(GAMMA-1)*((p/pressure)**((GAMMA-1)/(2*GAMMA))-1)
    pstar = brentq(lambda p: wave(p, LEFT)+wave(p, RIGHT)+RIGHT[1]-LEFT[1], .001, 10)
    ustar = .5*(LEFT[1]+RIGHT[1]+wave(pstar, RIGHT)-wave(pstar, LEFT))
    beta = (GAMMA-1)/(GAMMA+1)
    stars, speeds = [], []
    for sign,state in [(-1,LEFT),(1,RIGHT)]:
        rho,u,p = state
        assert pstar > p  # These rounded RP1 states produce two shocks, one very weak.
        stars.append(rho*(pstar/p+beta)/(beta*pstar/p+1))
        speeds.append(u+sign*math.sqrt(GAMMA*p/rho)*math.sqrt(
            (GAMMA+1)/(2*GAMMA)*pstar/p+(GAMMA-1)/(2*GAMMA)))
    boundaries = [.5+time*speeds[0], .5+time*ustar, .5+time*speeds[1]]
    states = [(LEFT[0],LEFT[1],LEFT[2]),(stars[0],ustar,pstar),
              (stars[1],ustar,pstar),(RIGHT[0],RIGHT[1],RIGHT[2])]
    return boundaries, states


def conservative(state):
    rho,u,p = state
    return np.array([rho,rho*u,0,0,p/(GAMMA-1)+.5*rho*u*u])


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    out = args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    exe = args.executable.resolve()
    x = np.linspace(0,1,41);dx=.025
    lo,hi = np.maximum(0,x-dx/2),np.minimum(1,x+dx/2)
    frac = np.clip((hi-.5)/(hi-lo),0,1)
    initial = (1-frac[:,None])*conservative(LEFT)+frac[:,None]*conservative(RIGHT)
    all_values = np.tile(initial,(41*6,1))
    node_file = out/'initial.csv'
    np.savetxt(node_file,np.column_stack([np.arange(len(all_values)),all_values]),
               delimiter=',',header='globalNode,rho,rhou,rhov,rhow,rhoE',comments='',
               fmt=['%d']+['%.17g']*5)
    boundaries,states = exact_shock(.25)
    # Exact dual-volume means: integrate the piecewise constant Riemann states.
    reference = np.zeros((41,5))
    edges = [0,*boundaries,1]
    for i in range(4):
        overlap = np.maximum(0,np.minimum(hi,edges[i+1])-np.maximum(lo,edges[i]))/(hi-lo)
        reference += overlap[:,None]*conservative(states[i])
    exact_rho = reference[:,0]
    exact_p = (GAMMA-1)*(reference[:,4]-.5*reference[:,1]**2/reference[:,0])
    summary = {'binary_sha256':hashlib.sha256(exe.read_bytes()).hexdigest(),
               'exact_wave_locations':boundaries,'exact_states_rho_u_p':states,'runs':[]}
    fig,axes = plt.subplots(1,2,figsize=(12,4),layout='constrained')
    axes[0].plot(x,exact_rho,'k--',label='Exact dual mean')
    axes[1].plot(x,exact_p,'k--',label='Pressure from exact dual mean')
    for name,flags in [('Roe',{}),('LLF',{'BJ_LLF':'1'})]:
        d=out/name;d.mkdir()
        cmd=['mpirun','--mca','pml','ob1','--mca','btl','self,vader','--bind-to','none','-np','4',
             str(exe),str(CASE/'RP1_zperiodic_tol1e10.json')]
        audit=ROOT/'data/out/NCFV/riemann_barth_400x400x5/limiter_audit'
        overrides={'/mesh/meshFile':str(audit/'mesh40.cgns'),'/mesh/periodicLengths':[0,0,.125],
                   '/initialField/nodeFile':str(node_file),'/time/maximumTimeStep':.001,
                   '/io/writeVTK':False,'/io/restartInterval':0,'/io/outputPrefix':str(d/'solution'),
                   '/io/vtkSeriesName':str(d/'series'),'/io/restartPrefix':str(d/'restart')}
        for k,v in overrides.items():cmd+=['-k',k,'-v',json.dumps(v)]
        env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',HWLOC_COMPONENTS='-gl,-cuda,-nvml',OMPI_MCA_mpi_yield_when_idle='1')
        env={k:v for k,v in env.items() if not k.startswith(('BJ_','NCFV_'))};env.update(flags)
        (d/'command.json').write_text(json.dumps({'command':cmd,'flags':flags},indent=2)+'\n')
        with (d/'solver.log').open('w') as log:
            p=subprocess.run(cmd,cwd=ROOT/'build',env=env,stdout=log,stderr=subprocess.STDOUT)
        assert p.returncode==0,(name,p.returncode)
        with h5py.File(sorted(d.glob('restart_*.dnds.h5'))[-1]) as f:
            u=f['state/father/array/data'][:].reshape(-1,5);ids=f['state/origIndex'][:].astype(int)
            assert float(f.attrs['simulationTime'])==.25
        a=np.empty_like(u);a[ids]=u;a=a.reshape(5,41,41,5)
        rho=a[:,:,:,0];pressure=(GAMMA-1)*(a[:,:,:,4]-.5*np.sum(a[:,:,:,1:4]**2,axis=3)/rho)
        record={'variant':name,'rho_min':float(rho.min()),'rho_max':float(rho.max()),
                'p_min':float(pressure.min()),'p_max':float(pressure.max()),
                'rho_overshoot':max(0.,float(rho.max()-max(s[0] for s in states))),
                'p_overshoot_away_shock':max(0.,float(pressure[:,:,np.abs(x-boundaries[0])>3*dx].max()-max(s[2] for s in states))),
                'rho_L1':float(np.mean(np.abs(rho-exact_rho[None,None,:]))),
                'rho_z_span':float(np.ptp(rho,axis=0).max()),'w_abs_max':float(np.abs(a[:,:,:,3]/rho).max()),
                'rho_midline_TV':float(np.abs(np.diff(rho[2,20])).sum()),
                'p_midline_TV':float(np.abs(np.diff(pressure[2,20])).sum())}
        summary['runs'].append(record)
        (out/'results.json').write_text(json.dumps(summary,indent=2)+'\n')
        print(json.dumps(record),flush=True)
        axes[0].plot(x,rho[2,20],'.-',label=name)
        axes[1].plot(x,pressure[2,20],'.-',label=name)
    for ax,title in zip(axes,['Density','Pressure']):
        ax.set(xlabel='x',ylabel=title,title=title+' at y=0.5, middle z layer')
        ax.legend();ax.grid(alpha=.2)
    fig.suptitle('Planar shock: same limiter/integrated flux/boundaries; only Riemann solver changes')
    fig.savefig(out/'shock_comparison.png',dpi=170)
    plt.close(fig)


if __name__=='__main__':
    run()
