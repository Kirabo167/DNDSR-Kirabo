#!/usr/bin/env python3
"""Continue small certified appendage expansions, recording every stage."""
import json
import argparse
import os
from pathlib import Path
import subprocess
import sys
import numpy as np

root = Path(__file__).resolve().parents[1]
ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,default=root/'data/mesh/SUBOFF_BL_restart_20261102/local40');ap.add_argument('--start',type=int,default=2);ap.add_argument('--end',type=int,default=8);args=ap.parse_args()
out = Path('/tmp/SUBOFF_local_continue_20261102')
out.mkdir(exist_ok=True)
source = args.source
env = dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='12',PYTHONPATH='/tmp/suboff_quality_cuda',JAX_PLATFORMS='cuda',XLA_PYTHON_CLIENT_PREALLOCATE='false')
reference=root/'data/mesh/SUBOFF_BL_smooth_Q02_20261102/final/SUBOFF_optimized.validated.npz'
def run(script, args, log):
    with log.open('w') as stream:
        subprocess.run([sys.executable,str(root/'scripts'/script),*map(str,args)],cwd=root,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
def certify(D):
    q=json.loads((D/'final_quality.json').read_text())
    if q['minimum']<.2: raise RuntimeError(f'Full quality failed: {q["minimum"]}')
    a=np.load(D/'optimized_coordinates.npz');f=np.load(D/'prism_front.npz')['faces'];nv=int(f.max())+1;nl=len(a['cells_6'])//len(f)
    lp=a['points'][:(nl+1)*nv].reshape(nl+1,nv,3);dp=np.diff(lp,axis=0)
    normal_heights=[]
    for k in range(nl):
        tri=lp[k,f];n=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);n/=np.linalg.norm(n,axis=1)[:,None]
        normal_heights.append(np.einsum('fij,fj->f',dp[k,f],n)/3)
    nh=np.array(normal_heights);ng=float((nh[1:]/nh[:-1]).max());length=np.linalg.norm(dp,axis=2);eg=float((length[1:]/length[:-1]).max())
    ref=np.load(reference)['points'][:(nl+1)*nv].reshape(nl+1,nv,3);rh=[]
    for k in range(nl):
        tri=ref[k,f];n=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);n/=np.linalg.norm(n,axis=1)[:,None]
        rh.append(np.einsum('fij,fj->f',(ref[k+1]-ref[k])[f],n)/3)
    rh=np.array(rh);allowed=np.maximum(1.25,rh[1:]/rh[:-1]+.0005)
    excess=float((nh[1:]/nh[:-1]-allowed).max())
    direction=lp[1]-lp[0];direction/=np.linalg.norm(direction,axis=1)[:,None]
    hp=np.einsum('kij,ij->ki',dp,direction);pg=float((hp[1:]/hp[:-1]).max())
    if nh.min()<=0 or max(pg,eg)>1.25 or excess>0:raise RuntimeError(f'Growth failed: normal_excess={excess} projected={pg} euclidean={eg}')
    sys.path.insert(0,'/tmp/suboff_regen_deps')
    import pymeshlab
    check=pymeshlab.MeshSet();check.add_mesh(pymeshlab.Mesh(np.vstack([lp[0],lp[-1]]),np.vstack([f,f+nv])))
    check.apply_filter('compute_selection_by_self_intersections_per_face')
    intersections=int(check.current_mesh().selected_face_number())
    if intersections:raise RuntimeError(f'Outer front/wall intersects: {intersections}')
    return dict(directory=str(D),minimum_quality=q['minimum'],maximum_normal_growth=ng,maximum_euclidean_growth=eg,maximum_projected_growth=pg,normal_growth_excess=excess)

record=certify(source)
records=json.loads((out/'progress.json').read_text()) if (out/'progress.json').exists() else []
if not records or records[-1]['directory']!=str(source):records.append(record)
(out/'progress.json').write_text(json.dumps(records,indent=2)+'\n')
for step in range(args.start,args.end+1):
    D=out/f'stage{step:02d}'
    raw=out/f'stage{step:02d}_raw'
    print('BEGIN_STAGE',step,'SOURCE',source,flush=True)
    run('suboff_local_expand.py',[source,raw,'--arrays','optimized_coordinates.npz','--target-mm',30],out/f'stage{step:02d}_generation.log')
    run('suboff_redistribute_local.py',[raw,D],out/f'stage{step:02d}_redistribution.log')
    run('suboff_orthogonal_quality.py',[D],D/'quality_initial.log')
    run('suboff_optimize_bl_constraints.py',[D,'--iterations',1800,'--goal-weight',100000,'--growth-weight',100000,'--jacobian-weight',100000,'--jacobian-floor',.05,'--core-scale-floor',.2,'--tangential-scale-floor',.2,'--coordinate-bound',4,'--jax-path','/tmp/suboff_quality_cuda','--normal-growth-reference',reference],D/'optimization.log')
    run('suboff_orthogonal_quality.py',[D,'--arrays','optimized_coordinates.npz','--output-prefix','final_quality'],D/'quality_final.log')
    record=certify(D);records.append(record)
    (out/'progress.json').write_text(json.dumps(records,indent=2)+'\n')
    print('CERTIFIED',json.dumps(record),flush=True)
    source=D
print('DONE',source,flush=True)
