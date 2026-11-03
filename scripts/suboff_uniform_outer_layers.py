#!/usr/bin/env python3
"""Build smooth geometric layers on a Lipschitz-limited thickness envelope."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.sparse import diags
from scipy.sparse.linalg import LinearOperator,cg
from suboff_arrays_cgns import jacobians

ap=argparse.ArgumentParser();ap.add_argument('source',type=Path);ap.add_argument('output',type=Path)
ap.add_argument('--arrays',default='optimized_coordinates.npz')
ap.add_argument('--layers',type=int)
ap.add_argument('--slope',type=float,default=.8);ap.add_argument('--freeze',type=int,default=4)
ap.add_argument('--offset-regularization',type=float,default=0.)
ap.add_argument('--tangent-radius',type=float,default=0.)
ap.add_argument('--turn-distance',type=float,default=0.)
ap.add_argument('--quality-aware-slope',type=float,default=0.)
ap.add_argument('--feature-variance',type=float,default=0.)
args=ap.parse_args();S=args.source;D=args.output;D.mkdir(exist_ok=True)
a=np.load(S/args.arrays);p=a['points'];f=np.load(S/'prism_front.npz')['faces'];nv=int(f.max())+1;nf=len(f)
input_layers=len(a['cells_6'])//nf;layers=args.layers or input_layers
old=p[:(input_layers+1)*nv].reshape(input_layers+1,nv,3);freeze=args.freeze
base=old[0];first=old[1]-base;length=np.linalg.norm(first,axis=1);direction=first/length[:,None]
start=old[freeze];initial=old[freeze]-old[freeze-1];previous=np.einsum('ij,ij->i',initial,direction)
initial/=previous[:,None]
total=np.einsum('ij,ij->i',old[-1]-base,direction);equivalent=total/length*1e-5
prefix=np.einsum('ij,ij->i',start-base,direction)
edges=np.unique(np.sort(np.vstack([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]),axis=1),axis=0)
wall_length=np.linalg.norm(base[edges[:,0]]-base[edges[:,1]],axis=1)
rows=np.r_[edges[:,0],edges[:,1],np.full(nv,nv)];cols=np.r_[edges[:,1],edges[:,0],np.arange(nv)]
edge_cost=args.slope*wall_length
cap=equivalent.copy();count=layers-freeze;cells=np.arange(6*nf).reshape(nf,6)
offset=None
if args.offset_regularization:
    q=base[f];normal=np.cross(q[:,1]-q[:,0],q[:,2]-q[:,0]);normal/=np.linalg.norm(normal,axis=1)[:,None]
    matrix=np.zeros((nv,3,3));rhs=np.zeros_like(base);weight_sum=np.zeros(nv)
    for i in range(3):
        u=q[:,(i+1)%3]-q[:,i];v=q[:,(i+2)%3]-q[:,i]
        weight=np.arctan2(np.linalg.norm(np.cross(u,v),axis=1),np.einsum('ij,ij->i',u,v))
        np.add.at(matrix,f[:,i],weight[:,None,None]*normal[:,:,None]*normal[:,None,:])
        np.add.at(rhs,f[:,i],weight[:,None]*normal);np.add.at(weight_sum,f[:,i],weight)
    original_matrix=matrix.copy()
    matrix+=args.offset_regularization*weight_sum[:,None,None]*np.eye(3)[None,:,:]
    offset=np.linalg.solve(matrix,rhs[:,:,None])[:,:,0]
    offset/=((rhs*offset).sum(axis=1)/weight_sum)[:,None]
    if args.tangent_radius:
        if args.feature_variance:
            mean_normal=rhs/np.linalg.norm(rhs,axis=1)[:,None]
            tangent_projector=np.eye(3)[None,:,:]-mean_normal[:,:,None]*mean_normal[:,None,:]
            covariance=np.einsum('nij,njk,nkl->nil',tangent_projector,original_matrix,tangent_projector)/weight_sum[:,None,None]
            values,vectors=np.linalg.eigh(covariance)
            constrained=values>args.feature_variance
            projector=tangent_projector-np.einsum('nik,nk,njk->nij',vectors,constrained.astype(float),vectors)
        else:
            values,vectors=np.linalg.eigh(original_matrix)
            constrained=values>values[:,-1,None]*1e-3
            projector=np.eye(3)[None,:,:]-np.einsum('nik,nk,njk->nij',vectors,constrained.astype(float),vectors)
        area2=np.linalg.norm(np.cross(q[:,1]-q[:,0],q[:,2]-q[:,0]),axis=1)
        mass=np.zeros(nv);ee=[];ww=[]
        for i in range(3):
            np.add.at(mass,f[:,i],area2/6)
            u=q[:,(i+1)%3]-q[:,i];v=q[:,(i+2)%3]-q[:,i]
            cot=np.einsum('ij,ij->i',u,v)/area2
            ee.append(f[:,[(i+1)%3,(i+2)%3]]);ww.append(np.clip(.5*cot,.02,10.))
        ee=np.vstack(ee);ww=np.concatenate(ww)
        adjacency=coo_matrix((np.r_[ww,ww],(np.r_[ee[:,0],ee[:,1]],np.r_[ee[:,1],ee[:,0]])),shape=(nv,nv)).tocsr()
        lap=diags(np.asarray(adjacency.sum(axis=1)).ravel())-adjacency
        mu=mass/args.tangent_radius**2
        def project(x):return np.einsum('nij,nj->ni',projector,x)
        def matvec(x):
            z=x.reshape(nv,3)
            return (project(lap@project(z))+mu[:,None]*z).ravel()
        diagonal=lap.diagonal()[:,None]*np.maximum(np.diagonal(projector,axis1=1,axis2=2),0)+mu[:,None]
        operator=LinearOperator((3*nv,3*nv),matvec=matvec,dtype=float)
        preconditioner=LinearOperator(operator.shape,matvec=lambda x:x/diagonal.ravel(),dtype=float)
        iterations=[0]
        def callback(x):iterations[0]+=1
        z,info=cg(operator,-project(lap@offset).ravel(),M=preconditioner,rtol=1e-8,maxiter=3000,callback=callback)
        offset+=project(z.reshape(nv,3))
        offset/=((rhs*offset).sum(axis=1)/weight_sum)[:,None]
        print('TANGENT_DIFFUSION',info,'ITERATIONS',iterations[0],'NORM',np.quantile(np.linalg.norm(offset,axis=1),[0,.5,.99,1]),flush=True)
    dots=np.einsum('fij,fj->fi',offset[f],normal)
    print('REGULARIZED_OFFSET_NORMAL_PROJECTIONS',np.quantile(dots,[0,.001,.5,.999,1]),flush=True)
    if dots.min()<=0:raise ValueError('Regularized direction leaves the outward normal cone')
    if args.quality_aware_slope:
        # A strongly tangential miter direction tolerates a much smaller
        # height gradient than an almost normal direction. Bound the latter
        # using the cosine for a locally planar variable offset surface.
        projection_min=np.full(nv,np.inf)
        for i in range(3):np.minimum.at(projection_min,f[:,i],dots[:,i])
        norm2=np.sum(offset*offset,axis=1);tangent=np.sqrt(np.maximum(norm2-projection_min**2,0))
        allowed=projection_min/(args.quality_aware_slope*np.sqrt(norm2))
        slope=np.where(allowed>1,(-tangent+np.sqrt(tangent*tangent+norm2*np.maximum(allowed*allowed-1,0)))/norm2,0)
        slope=np.minimum(slope,args.slope)
        edge_cost=wall_length*np.minimum(slope[edges[:,0]],slope[edges[:,1]])
        print('QUALITY_AWARE_SLOPES',np.quantile(slope,[0,.01,.5,.99,1]),flush=True)
    lo=np.zeros(nf);hi=np.full(nf,.06)
    for _ in range(18):
        height=(lo+hi)/2;end=base[f]+offset[f]*height[:,None,None]
        block=np.concatenate([start[f],end],axis=1)
        good=(jacobians(block.reshape(-1,3),cells)>0).all(axis=1)
        lo=np.where(good,height,lo);hi=np.where(good,hi,height)
    prefix_height=np.einsum('fij,fj->fi',start[f]-base[f],normal)/dots
    safe=prefix_height.max(axis=1)+.65*(lo-prefix_height.max(axis=1))
    cap=np.full(nv,.03);np.minimum.at(cap,f.ravel(),np.repeat(safe,3))
    print('INITIAL_OFFSET_CAP',np.quantile(cap,[0,.01,.5,.99,1]),flush=True)

def build(limit):
    graph=coo_matrix((np.r_[edge_cost,edge_cost,limit],(rows,cols)),shape=(nv+1,nv+1)).tocsr()
    h=dijkstra(graph,directed=True,indices=nv)[:nv]
    target=base+(old[-1]-base)*(h/equivalent)[:,None] if offset is None else base+offset*h[:,None]
    remaining=np.einsum('ij,ij->i',target-start,direction)
    if np.any(remaining<=0):raise ValueError('Envelope thinner than fixed prefix')
    lo=np.full(nv,.5);hi=np.full(nv,1.245)
    for _ in range(45):
        growth=(lo+hi)/2;size=previous*np.sum(growth[None,:]**np.arange(1,count+1)[:,None],axis=0)
        ok=size<remaining;lo=np.where(ok,growth,lo);hi=np.where(ok,hi,growth)
    growth=(lo+hi)/2
    height=previous[None,:]*np.cumsum(growth[None,:]**np.arange(1,count+1)[:,None],axis=0)
    s=height/remaining[None,:]
    # Match the preceding layer's direction, then gradually turn toward the
    # limited outer envelope. Projection along the original first direction
    # follows an exact geometric progression.
    result=np.empty((layers+1,nv,3));result[:freeze+1]=old[:freeze+1]
    if args.turn_distance:
        final_direction=(target-start)/remaining[:,None]
        turn=np.minimum(args.turn_distance,.1*remaining)
        end_weight=-np.expm1(-remaining/turn)
        correction=(turn[None,:]*(-np.expm1(-height/turn[None,:]))-s*turn[None,:]*end_weight[None,:])/(1-turn/remaining*end_weight)[None,:]
        result[freeze+1:]=start[None,:,:]+height[:,:,None]*final_direction[None,:,:]+correction[:,:,None]*(initial-final_direction)[None,:,:]
    else:
        tangent=(target-start)-initial*remaining[:,None]
        result[freeze+1:]=start[None,:,:]+height[:,:,None]*initial[None,:,:]+s[:,:,None]**2*tangent[None,:,:]
    result[-1]=target
    return result,h,growth

sys.path.insert(0,'/tmp/suboff_regen_deps');import pymeshlab
for attempt in range(35):
    result,h,growth=build(cap);bad=np.zeros(nf,dtype=bool);minimum=float('inf')
    # Full exact prism checks; only 131048 cells reside in memory per pass.
    for k in range(freeze,layers):
        q=np.concatenate([result[k,f],result[k+1,f]],axis=1)
        j=jacobians(q.reshape(-1,3),cells);bad|=(j<=0).any(axis=1);minimum=min(minimum,float(j.min()))
    print('ATTEMPT',attempt,'BAD_COLUMNS',int(bad.sum()),'MINJ',minimum,'H',np.quantile(h,[0,.01,.5,.99,1]),flush=True)
    if not bad.any():
        check=pymeshlab.MeshSet();check.add_mesh(pymeshlab.Mesh(np.vstack([base,result[-1]]),np.vstack([f,f+nv])))
        check.apply_filter('compute_selection_by_self_intersections_per_face');sel=check.current_mesh().face_selection_array()
        bad=sel[:nf]|sel[nf:];print('TOP_INTERSECTIONS',int(bad.sum()),flush=True)
        if not bad.any():break
    vertices=np.unique(f[bad]);cap[vertices]=np.minimum(cap[vertices],.8*h[vertices])
else:raise ValueError('Could not build a valid limited envelope')
np.savez(D/'optimized_layers.npz',points=result,faces=f,direction=direction,growth=growth)
np.savez(D/'prism_front.npz',points=base,faces=f,top=result[-1])
plan=dict(source='data/mesh/SUBOFF_BL_smooth_Q02_20261102/final',geometry_input=str(S),layers=layers,old_layers=40,
          fixed_layers=freeze,target_thickness_m=.030,spatial_slope_limit=args.slope,
          offset_regularization=args.offset_regularization,
          tangent_diffusion_radius_m=args.tangent_radius,
          direction_turn_distance_m=args.turn_distance,
          quality_aware_slope_goal=args.quality_aware_slope,
          feature_normal_variance_threshold=args.feature_variance,
          equivalent_thickness_quantiles_m=np.quantile(h,[0,.01,.5,.99,1]).tolist(),
          minimum_jacobian=minimum,geometric_growth_quantiles=np.quantile(growth,[0,.01,.5,.99,1]).tolist(),
          wall_and_prefix_unchanged=bool(np.array_equal(result[:freeze+1],old[:freeze+1])))
(D/'thickening_plan.json').write_text(json.dumps(plan,indent=2)+'\n');print(json.dumps(plan),flush=True)
