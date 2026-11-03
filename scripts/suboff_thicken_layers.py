#!/usr/bin/env python3
"""Add outer layers and raise thin columns while keeping the wall/prefix fixed."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from suboff_arrays_cgns import jacobians

ap=argparse.ArgumentParser();ap.add_argument('source',type=Path);ap.add_argument('output',type=Path)
ap.add_argument('--layers',type=int,default=64);ap.add_argument('--freeze',type=int,default=16)
ap.add_argument('--thickness',type=float,default=.030)
args=ap.parse_args();S=args.source;D=args.output;D.mkdir(parents=True,exist_ok=True)
a=np.load(S/'SUBOFF_optimized.validated.npz');p=a['points'];f=np.load(S/'prism_front.npz')['faces']
nv=int(f.max())+1;nf=len(f);old=p[:41*nv].reshape(41,nv,3);base=old[0];freeze=args.freeze
normal=np.cross(base[f[:,1]]-base[f[:,0]],base[f[:,2]]-base[f[:,0]])
normal/=np.linalg.norm(normal,axis=1)[:,None]
outward=np.zeros_like(base)
for i in range(3):
    u=base[f[:,(i+1)%3]]-base[f[:,i]];v=base[f[:,(i+2)%3]]-base[f[:,i]]
    angle=np.arctan2(np.linalg.norm(np.cross(u,v),axis=1),np.einsum('ij,ij->i',u,v))
    np.add.at(outward,f[:,i],normal*angle[:,None])
outward/=np.linalg.norm(outward,axis=1)[:,None]
direction=old[1]-base;direction/=np.linalg.norm(direction,axis=1)[:,None]
projection=np.einsum('kij,ij->ki',old-base,direction)
previous=projection[freeze]-projection[freeze-1]
physical=np.einsum('ij,ij->i',old[-1]-base,outward)
displacement=outward*np.maximum(0.,args.thickness-physical)[:,None]
edges=np.unique(np.sort(np.vstack([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]),axis=1),axis=0)
factor=np.ones(nv);count=args.layers-freeze
localcells=np.arange(6*nf).reshape(nf,6)

def build(scale):
    shift=displacement*scale[:,None]
    remaining=projection[-1]-projection[freeze]
    assert count==2*(40-freeze),'Use two children for each original outer prism'
    result=np.empty((args.layers+1,nv,3));result[:freeze+1]=old[:freeze+1]
    for k in range(count):
        index=freeze+(k+1)//2;frac=((k+1)%2)/2
        reference=old[index] if frac==0 else (old[index]+old[index+1])/2
        h=np.einsum('ij,ij->i',reference-old[freeze],direction)
        s=h/remaining
        result[freeze+k+1]=reference+s[:,None]**2*shift
    result[-1]=old[-1]+shift
    heights=np.einsum('kij,ij->ki',np.diff(result,axis=0),direction)
    growth=(heights[1:]/heights[:-1]).max(axis=0)
    return result,growth

sys.path.insert(0,'/tmp/suboff_regen_deps');import pymeshlab
for attempt in range(35):
    result,growth=build(factor);bad=np.zeros(nf,dtype=bool);minimum=float('inf')
    for k in range(freeze,args.layers):
        q=np.concatenate([result[k,f],result[k+1,f]],axis=1)
        j=jacobians(q.reshape(-1,3),localcells);bad|=(j<=0).any(axis=1);minimum=min(minimum,float(j.min()))
    print('ATTEMPT',attempt,'INVALID_COLUMNS',int(bad.sum()),'MIN_JAC',minimum,'FACTOR',np.quantile(factor,[0,.01,.5,.99,1]),flush=True)
    if not bad.any():
        check=pymeshlab.MeshSet();check.add_mesh(pymeshlab.Mesh(np.vstack([base,result[-1]]),np.vstack([f,f+nv])))
        check.apply_filter('compute_selection_by_self_intersections_per_face')
        selected=check.current_mesh().face_selection_array();bad=selected[:nf]|selected[nf:]
        print('TOP_INTERSECTIONS',int(bad.sum()),flush=True)
        if not bad.any():break
    affected=np.unique(f[bad]);factor[affected]*=.65
    # A bound on neighboring factors prevents abrupt transitions in the
    # permitted expansion. Convex feature edges remain coupled.
    for _ in range(5):
        before=factor.copy()
        np.minimum.at(factor,edges[:,0],before[edges[:,1]]+.10)
        np.minimum.at(factor,edges[:,1],before[edges[:,0]]+.10)
else:raise ValueError('Could not find a positive, nonintersecting layer expansion')
np.savez(D/'optimized_layers.npz',points=result,faces=f,direction=direction,growth=growth)
np.savez(D/'prism_front.npz',points=base,faces=f,top=result[-1])
np.savez(D/'thickening_targets.npz',source_top=old[-1],target_top=old[-1]+displacement,accepted_top=result[-1],factor=factor)
summary=dict(source=str(S),layers=args.layers,old_layers=40,fixed_layers=freeze,target_thickness_m=args.thickness,
             factor_quantiles=np.quantile(factor,[0,.01,.05,.5,.95,1]).tolist(),minimum_jacobian=minimum,
             wall_and_prefix_unchanged=bool(np.array_equal(result[:freeze+1],old[:freeze+1])),
             scalar_growth_quantiles=np.quantile(growth,[0,.01,.5,.99,1]).tolist(),
             nodes=nv,faces=nf,prisms=args.layers*nf)
(D/'thickening_plan.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)
