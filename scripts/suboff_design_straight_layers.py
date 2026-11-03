#!/usr/bin/env python3
"""Rebuild smooth extrusion directions while preserving first face heights."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsmr
from scipy.sparse.csgraph import dijkstra
from suboff_arrays_cgns import jacobians
from suboff_optimize_layers import centroids,geometry,cosine

ap=argparse.ArgumentParser();ap.add_argument('reference',type=Path);ap.add_argument('field',type=Path);ap.add_argument('output',type=Path)
ap.add_argument('--layers',type=int,default=64);args=ap.parse_args();R=args.reference;F=args.field;D=args.output;D.mkdir(exist_ok=True)
ref=np.load(R/'SUBOFF_optimized.validated.npz');f=np.load(R/'prism_front.npz')['faces'];nv=int(f.max())+1;nf=len(f)
base=ref['points'][:nv];old_step=ref['points'][nv:2*nv]-base;q=base[f]
normal=np.cross(q[:,1]-q[:,0],q[:,2]-q[:,0]);area2=np.linalg.norm(normal,axis=1);normal/=area2[:,None]
mean_normal=np.zeros_like(base);weight_sum=np.zeros(nv)
for i in range(3):
    u=q[:,(i+1)%3]-q[:,i];v=q[:,(i+2)%3]-q[:,i]
    weight=np.arctan2(np.linalg.norm(np.cross(u,v),axis=1),np.einsum('ij,ij->i',u,v))
    np.add.at(mean_normal,f[:,i],weight[:,None]*normal);np.add.at(weight_sum,f[:,i],weight)
mean_normal/=weight_sum[:,None]
top=np.load(F/'prism_front.npz')['top'];height=np.einsum('ij,ij->i',top-base,mean_normal)
offset=(top-base)/height[:,None];target_step=offset*1e-5
row=np.repeat(np.arange(nf),9);col=(f[:,:,None]*3+np.arange(3)).ravel()
values=np.repeat(normal[:,None,:]/3,3,axis=1).ravel();A=coo_matrix((values,(row,col)),shape=(nf,3*nv)).tocsr()
reference_height=np.einsum('fij,fj->f',old_step[f],normal)/3
edges=np.vstack([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]);owners=np.tile(np.arange(nf),3)
key=np.sort(edges,axis=1);order=np.lexsort((key[:,1],key[:,0]));ia,ib=order[::2],order[1::2]
edge=edges[ia];owner=owners[ia];neighbor=owners[ib]
cells=np.arange(6*nf).reshape(nf,6)

def first_quality(step):
    pp=np.concatenate([base[f],base[f]+step[f]],axis=1);center,_=centroids(pp);value=np.ones(nf)
    ids=np.c_[f,f+nv]
    for face in ((0,2,1),(3,4,5),(0,1,4,3),(1,2,5,4),(2,0,3,5)):
        fc,n=geometry(pp[:,face],ids[:,face]);value=np.minimum(value,cosine(n,fc-center))
    quad=np.stack([base[edge[:,0]],base[edge[:,1]],base[edge[:,1]]+step[edge[:,1]],base[edge[:,0]]+step[edge[:,0]]],axis=1)
    _,n=geometry(quad);cc=cosine(n,center[neighbor]-center[owner]);np.minimum.at(value,owner,cc);np.minimum.at(value,neighbor,cc)
    j=jacobians(pp.reshape(-1,3),cells);bad=(value<.200001)|(j<=0).any(axis=1)
    return value,bad

step=target_step.copy()
if (D/'first_design_iteration.npz').exists():step=np.load(D/'first_design_iteration.npz')['displacement']
for attempt in range(40):
    residual=reference_height-A@step.ravel()
    solve=lsmr(A,residual,atol=1e-9,btol=1e-9,conlim=1e8,maxiter=3000)
    step+=solve[0].reshape(nv,3)
    error=float(np.abs(A@step.ravel()-reference_height).max())
    quality,bad=first_quality(step)
    np.savez(D/'first_design_iteration.npz',displacement=step)
    print('FIRST_DESIGN',attempt,'Q',float(quality.min()),'BAD',int(bad.sum()),'HEIGHT_ERROR',error,'LSMR',solve[1:3],flush=True)
    if not bad.any() and error<2e-9:break
    ids=np.unique(f[bad])
    for _ in range(1 if attempt<6 else 2):ids=np.unique(f[np.isin(f,ids).any(axis=1)])
    step[ids]=.4*step[ids]+.6*old_step[ids]
else:raise ValueError('Could not retain first-layer quality and normal heights')
direction=step/1e-5
np.savez(D/'first_layer_design.npz',displacement=step,reference_displacement=old_step,
         face_normal_height_before=reference_height,face_normal_height_after=A@step.ravel())
unique_edges=np.unique(key,axis=0);wall_length=np.linalg.norm(base[unique_edges[:,0]]-base[unique_edges[:,1]],axis=1)
dots=np.einsum('fij,fj->fi',direction[f],normal)
projection_min=np.full(nv,np.inf)
for i in range(3):np.minimum.at(projection_min,f[:,i],dots[:,i])
norm2=np.sum(direction*direction,axis=1);tangent=np.sqrt(np.maximum(norm2-projection_min**2,0))
allow=projection_min/(.3*np.sqrt(norm2));slope=np.where(allow>1,(-tangent+np.sqrt(tangent*tangent+norm2*np.maximum(allow*allow-1,0)))/norm2,0)
slope=np.minimum(slope,.8);cost=wall_length*np.minimum(slope[unique_edges[:,0]],slope[unique_edges[:,1]])
lo=np.zeros(nf);hi=np.full(nf,.06)
for _ in range(18):
    h=(lo+hi)/2;pp=np.concatenate([base[f],base[f]+direction[f]*h[:,None,None]],axis=1)
    good=(jacobians(pp.reshape(-1,3),cells)>0).all(axis=1);lo=np.where(good,h,lo);hi=np.where(good,hi,h)
cap=np.full(nv,.03);np.minimum.at(cap,f.ravel(),np.repeat(.65*lo,3))
rows=np.r_[unique_edges[:,0],unique_edges[:,1],np.full(nv,nv)]
cols=np.r_[unique_edges[:,1],unique_edges[:,0],np.arange(nv)]
sys.path.insert(0,'/tmp/suboff_regen_deps');import pymeshlab
for geometry_attempt in range(25):
    graph=coo_matrix((np.r_[cost,cost,cap],(rows,cols)),shape=(nv+1,nv+1)).tocsr()
    total=dijkstra(graph,directed=True,indices=nv)[:nv]
    if total.min()<2e-5:raise ValueError('A straight column cannot retain the first height here')
    low=np.full(nv,.5);high=np.full(nv,1.245)
    for _ in range(50):
        growth=(low+high)/2;h=1e-5*np.sum(growth[None,:]**np.arange(args.layers)[:,None],axis=0)
        good=h<total;low=np.where(good,growth,low);high=np.where(good,high,growth)
    growth=(low+high)/2
    heights=np.vstack([np.zeros(nv),1e-5*np.cumsum(growth[None,:]**np.arange(args.layers)[:,None],axis=0)])
    lp=base[None,:,:]+heights[:,:,None]*direction[None,:,:]
    bad_geometry=np.zeros(nf,dtype=bool);minimum_jacobian=float('inf')
    for k in range(args.layers):
        block=np.concatenate([lp[k,f],lp[k+1,f]],axis=1);j=jacobians(block.reshape(-1,3),cells)
        bad_geometry|=(j<=0).any(axis=1);minimum_jacobian=min(minimum_jacobian,float(j.min()))
    print('STRAIGHT_GEOMETRY',geometry_attempt,'BAD',int(bad_geometry.sum()),'H',np.quantile(total,[0,.01,.5,.99,1]),flush=True)
    if not bad_geometry.any():
        check=pymeshlab.MeshSet();check.add_mesh(pymeshlab.Mesh(np.vstack([base,lp[-1]]),np.vstack([f,f+nv])))
        check.apply_filter('compute_selection_by_self_intersections_per_face');selected=check.current_mesh().face_selection_array()
        bad_geometry=selected[:nf]|selected[nf:];print('TOP_INTERSECTIONS',int(bad_geometry.sum()),flush=True)
        if not bad_geometry.any():break
    vertices=np.unique(f[bad_geometry]);cap[vertices]=np.minimum(cap[vertices],.8*total[vertices])
else:raise ValueError('Invalid straight geometric layers')
np.savez(D/'optimized_layers.npz',points=lp,faces=f,direction=direction,growth=growth)
np.savez(D/'prism_front.npz',points=base,faces=f,top=lp[-1])
plan=dict(source=str(R),field_source=str(F),layers=args.layers,old_layers=40,fixed_layers=1,
          preserve_first_face_normal_height=True,first_face_normal_height_max_change_m=error,
          first_layer_minimum_quality=float(quality.min()),target_thickness_m=.03,
          minimum_jacobian=minimum_jacobian,
          total_scalar_height_quantiles_m=np.quantile(total,[0,.01,.5,.99,1]).tolist(),
          growth_quantiles=np.quantile(growth,[0,.01,.5,.99,1]).tolist())
(D/'thickening_plan.json').write_text(json.dumps(plan,indent=2)+'\n');print(json.dumps(plan),flush=True)
