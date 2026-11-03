#!/usr/bin/env python3
"""Smooth the layer envelope and redistribute outer layers of an existing mesh.

The original wall and first 16 layers are fixed. A screened surface diffusion
reduces unnecessary thickness gradients without increasing local extrusion.
The tetrahedral core follows the moved interface by a harmonic deformation.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from scipy.sparse import coo_matrix,diags
from scipy.sparse.linalg import cg
from scipy.optimize import minimize
from suboff_arrays_cgns import jacobians


def main():
    ap=argparse.ArgumentParser();ap.add_argument('source',type=Path);ap.add_argument('output',type=Path)
    ap.add_argument('--radius',type=float,default=.025)
    ap.add_argument('--maximum-reduction',type=float,default=.2)
    ap.add_argument('--freeze-layer',type=int,default=16)
    args=ap.parse_args();S=args.source;D=args.output;D.mkdir(parents=True,exist_ok=True)
    src=np.load(S/'SUBOFF_optimized.validated.npz');p=src['points'];t=src['cells_4'];pr=src['cells_6']
    f=np.load(S/'prism_front.npz')['faces'];nv=int(f.max()+1);layers=p[:41*nv].reshape(41,nv,3)
    first=layers[1]-layers[0];length=np.linalg.norm(first,axis=1);direction=first/length[:,None]
    increments=np.einsum('kij,ij->ki',np.diff(layers,axis=0),direction)
    total=increments.sum(axis=0);equivalent=total/length*1e-5
    base=layers[0];q=base[f];cross=np.cross(q[:,1]-q[:,0],q[:,2]-q[:,0]);area=np.linalg.norm(cross,axis=1)/2
    mass=np.zeros(nv)
    for k in range(3):np.add.at(mass,f[:,k],area/3)
    edges=[];weights=[]
    for k in range(3):
        a=q[:,(k+1)%3]-q[:,k];b=q[:,(k+2)%3]-q[:,k]
        cot=np.einsum('ij,ij->i',a,b)/(2*area)
        edges.append(f[:,[(k+1)%3,(k+2)%3]]);weights.append(np.clip(.5*cot,.02,10.))
    edges=np.vstack(edges);w=np.concatenate(weights)
    coherence=np.clip(np.einsum('ij,ij->i',direction[edges[:,0]],direction[edges[:,1]]),0,1)
    w*=coherence**8
    rows=np.r_[edges[:,0],edges[:,1]];cols=np.r_[edges[:,1],edges[:,0]];w=np.r_[w,w]
    adjacency=coo_matrix((w,(rows,cols)),shape=(nv,nv)).tocsr()
    lap=diags(np.asarray(adjacency.sum(axis=1)).ravel())-adjacency
    A=(diags(mass)+args.radius**2*lap).tocsr();s0=np.log(equivalent)
    # Normalize the objective so L-BFGS sees comparable curvature on small
    # appendage triangles and the larger hull triangles.
    norm=mass.mean()
    def obj(s):
        ds=s-s0
        grad=(mass*ds+args.radius**2*(lap@s))/norm
        value=(.5*np.dot(mass*ds,ds)+.5*args.radius**2*np.dot(s,lap@s))/norm
        return value,grad
    lower=s0+np.log(1-args.maximum_reduction)
    smooth=minimize(obj,s0,jac=True,method='L-BFGS-B',bounds=list(zip(lower,s0)),
                    options=dict(maxiter=1200,ftol=1e-13,gtol=1e-7,maxcor=20))
    target=np.exp(smooth.x);factor=target/equivalent
    desired_top=base+(layers[-1]-base)*factor[:,None]
    print('ENVELOPE',smooth.success,smooth.nit,'factor',np.quantile(factor,[0,.01,.5,.99,1]),flush=True)
    candidate=p.copy();freeze=args.freeze_layer
    start=layers[freeze];projected_remaining=np.einsum('ij,ij->i',desired_top-start,direction)
    previous=increments[freeze-1];count=40-freeze
    low=np.full(nv,.7);high=np.full(nv,1.25)
    for _ in range(60):
        g=(low+high)/2;target_sum=previous*np.sum(g[:,None]**np.arange(1,count+1),axis=1)
        ok=target_sum<projected_remaining;low=np.where(ok,g,low);high=np.where(ok,high,g)
    growth=(low+high)/2
    height=previous[None,:]*np.cumsum(growth[None,:]**np.arange(1,count+1)[:,None],axis=0)
    fraction=height/projected_remaining[None,:]
    candidate_layers=candidate[:41*nv].reshape(41,nv,3)
    for k in range(count):candidate_layers[freeze+k+1]=start+fraction[k,:,None]*(desired_top-start)
    candidate_layers[-1]=desired_top
    # Harmonic displacement of the tetrahedral part, including all reused
    # coarse far-field cells, with fixed external boundary coordinates.
    used=np.unique(t);remap=np.full(len(p),-1,dtype=np.int64);remap[used]=np.arange(len(used));tt=remap[t]
    ee=np.vstack([tt[:,[i,j]] for i in range(4) for j in range(i)])
    ee=np.unique(np.sort(ee,axis=1),axis=0)
    dist=np.linalg.norm(p[used[ee[:,0]]]-p[used[ee[:,1]]],axis=1)
    ew=1/np.maximum(dist,1e-10)
    adj=coo_matrix((np.r_[ew,ew],(np.r_[ee[:,0],ee[:,1]],np.r_[ee[:,1],ee[:,0]])),shape=(len(used),len(used))).tocsr()
    L=diags(np.asarray(adj.sum(axis=1)).ravel())-adj
    external=np.unique(np.r_[src['boundary_INLET'].ravel(),src['boundary_OUTLET'].ravel(),src['boundary_FAR'].ravel()])
    boundary=np.r_[40*nv+np.arange(nv),external];fixed=np.unique(remap[boundary]);free=np.setdiff1d(np.arange(len(used)),fixed)
    displacement=candidate[used]-p[used];B=L[free][:,free].tocsr();rhs=-(L[free][:,fixed]@displacement[fixed])
    pre=diags(1/B.diagonal())
    for axis in range(3):
        value,info=cg(B,rhs[:,axis],M=pre,rtol=1e-9,atol=1e-13,maxiter=3000)
        if info:raise ValueError(f'Harmonic core solve failed {info}')
        displacement[free,axis]=value
    candidate[used]=p[used]+displacement
    # A valid starting deformation is required by the following quality
    # optimization. Back off the envelope movement if necessary.
    delta=candidate-p
    accepted=None
    for alpha in (1.,.8,.6,.4,.2,.1):
        test=p+alpha*delta;minimum={};valid=True
        for name,block in (('TETRA_4',t),('PENTA_6',pr)):
            value=np.inf
            for i in range(0,len(block),100000):
                jj=jacobians(test,block[i:i+100000]);value=min(value,float(jj.min()))
                if value<=0:break
            minimum[name]=value;valid=valid and value>0
        print('DEFORMATION',alpha,minimum,flush=True)
        if valid:accepted=(test,alpha,minimum);break
    if accepted is None:raise ValueError('No valid envelope deformation')
    points,alpha,minimum=accepted
    boundaries={name:src['boundary_'+name] for name in ('WALL','INLET','OUTLET','FAR')}
    np.savez(D/'SUBOFF_optimized.validated.npz',points=points,cells_4=t,cells_6=pr,**{f'boundary_{k}':v for k,v in boundaries.items()})
    np.savez(D/'prism_front.npz',points=points[:nv],faces=f,top=points[40*nv:41*nv])
    np.savez(D/'boundary_arrays.npz',**boundaries)
    np.savez(D/'fixed_boundary.npz',nodes=external,wall_nodes=np.arange((freeze+1)*nv))
    info=json.loads((S/'mesh_validation.json').read_text())['domain']
    (D/'domain_extension.json').write_text(json.dumps(info,indent=2)+'\n')
    np.savez(D/'smoothing_targets.npz',source_points=p,desired_points=candidate,normal_direction=direction,
             original_equivalent_thickness=equivalent,target_equivalent_thickness=target,edges=edges)
    def stats(h):
        e=np.unique(np.sort(edges,axis=1),axis=0);co=np.einsum('ij,ij->i',direction[e[:,0]],direction[e[:,1]])
        d=np.linalg.norm(base[e[:,0]]-base[e[:,1]],axis=1);v=np.abs(h[e[:,0]]-h[e[:,1]])/d
        return np.quantile(v[co>.98],[.5,.95,.99,.999,1]).tolist()
    report=dict(source=str(S),radius_m=args.radius,maximum_reduction=args.maximum_reduction,
                fixed_layers=freeze,deformation_fraction=alpha,minimum_jacobian=minimum,
                envelope_slope_quantiles_before=stats(equivalent),envelope_slope_quantiles_target=stats(target),
                planned_growth_quantiles=np.quantile(growth,[0,.01,.5,.99,1]).tolist())
    (D/'smoothing_plan.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
