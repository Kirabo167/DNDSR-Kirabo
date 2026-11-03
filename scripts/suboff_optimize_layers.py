#!/usr/bin/env python3
"""Optimize prism directions and spatial growth while retaining the wall mesh."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from suboff_arrays_cgns import jacobians


def centroids(q):
    v=np.zeros(len(q)); m=np.zeros((len(q),3))
    for z in (.5-1/(2*np.sqrt(3)),.5+1/(2*np.sqrt(3))):
        dr=(1-z)*(q[:,1]-q[:,0])+z*(q[:,4]-q[:,3])
        ds=(1-z)*(q[:,2]-q[:,0])+z*(q[:,5]-q[:,3])
        cross=np.cross(dr,ds)
        for r,s in ((1/6,1/6),(2/3,1/6),(1/6,2/3)):
            bot=(1-r-s)*q[:,0]+r*q[:,1]+s*q[:,2]
            top=(1-r-s)*q[:,3]+r*q[:,4]+s*q[:,5]
            w=np.einsum('ij,ij->i',cross,top-bot)/12
            v+=w;m+=w[:,None]*((1-z)*bot+z*top)
    return m/v[:,None],v


def geometry(x,ids=None):
    if x.shape[1]==4 and ids is not None:
        order=(ids.argmin(axis=1)[:,None]+np.arange(4))%4
        x=np.take_along_axis(x,order[:,:,None],axis=1)
    n=np.cross(x[:,1]-x[:,0],x[:,2]-x[:,0]);fc=x[:,:3].mean(axis=1)
    if x.shape[1]==4:
        nn=np.cross(x[:,2]-x[:,0],x[:,3]-x[:,0]);a=np.linalg.norm(n,axis=1);b=np.linalg.norm(nn,axis=1)
        fc=(fc*a[:,None]+x[:,[0,2,3]].mean(axis=1)*b[:,None])/(a+b)[:,None];n+=nn
    return fc,n


def cosine(n,d):
    return np.einsum('ij,ij->i',n,d)/np.linalg.norm(n,axis=1)/np.linalg.norm(d,axis=1)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('source',type=Path);ap.add_argument('output',type=Path)
    ap.add_argument('--slope',type=float,default=.0005)
    ap.add_argument('--safety',type=float,default=.6)
    ap.add_argument('--first',type=float,default=1e-5)
    ap.add_argument('--maximum-growth',type=float,default=1.18)
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    old=np.load(args.source/'prism_front.npz');p=old['points'];f=old['faces'];nv=len(p);nf=len(f)
    qq=p[f];normal=np.cross(qq[:,1]-qq[:,0],qq[:,2]-qq[:,0]);normal/=np.linalg.norm(normal,axis=1)[:,None]
    A=np.zeros((nv,3,3));b=np.zeros_like(p)
    for i in range(3):
        np.add.at(A,f[:,i],normal[:,:,None]*normal[:,None,:]);np.add.at(b,f[:,i],normal)
    direction=np.einsum('nij,nj->ni',np.linalg.pinv(A,rcond=1e-5),b)
    edges=np.vstack([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]);own=np.tile(np.arange(nf),3)
    key=np.sort(edges,axis=1);order=np.lexsort((key[:,1],key[:,0]));ea,eb=order[::2],order[1::2]
    assert np.array_equal(key[ea],key[eb]);edge=edges[ea];owner,neighbor=own[ea],own[eb]
    unique=key[ea]
    # Bound offset by the first positive-Jacobian failure for each column.
    lo=np.zeros(nf);hi=np.full(nf,.06)
    for it in range(18):
        h=(lo+hi)/2
        q=np.concatenate([p[f],p[f]+direction[f]*h[:,None,None]],axis=1)
        good=(jacobians(q.reshape((-1,3)),np.arange(6*nf).reshape((-1,6)))>0).all(axis=1)
        lo=np.where(good,h,lo);hi=np.where(good,hi,h)
    cap=np.full(nv,.06)
    np.minimum.at(cap,f.ravel(),np.repeat(lo*args.safety,3))
    first=args.first;layers=40
    lower=np.full(nv,.8);upper=np.full(nv,args.maximum_growth)
    def thickness(g):return first*np.sum(g[:,None]**np.arange(layers),axis=1)
    for it in range(40):
        mid=(lower+upper)/2;ok=thickness(mid)<=cap
        lower=np.where(ok,mid,lower);upper=np.where(ok,upper,mid)
    growth=lower
    # Multi-source shortest-path lower envelope bounds growth differences
    # on every surface edge; all source weights are nonnegative.
    rows=np.r_[unique[:,0],unique[:,1],np.full(nv,nv)]
    cols=np.r_[unique[:,1],unique[:,0],np.arange(nv)]
    weights=np.r_[np.full(2*len(unique),args.slope),growth]
    graph=coo_matrix((weights,(rows,cols)),shape=(nv+1,nv+1)).tocsr()
    growth=dijkstra(graph,directed=True,indices=nv)[:nv]
    heights=np.vstack([np.zeros(nv),first*np.cumsum(growth[None,:]**np.arange(layers)[:,None],axis=0)])
    layers_points=p[None,:,:]+heights[:,:,None]*direction[None,:,:]
    quality=[];previous=None;worst=[];minimum_jacobian=float('inf')
    face_templates=((0,2,1),(3,4,5),(0,1,4,3),(1,2,5,4),(2,0,3,5))
    for layer in range(layers):
        q=np.concatenate([layers_points[layer,f],layers_points[layer+1,f]],axis=1)
        c,vol=centroids(q);val=np.ones(nf)
        local_ids=np.c_[f+layer*nv,f+(layer+1)*nv]
        for ff in face_templates:
            fc,n=geometry(q[:,ff],local_ids[:,ff]);val=np.minimum(val,cosine(n,fc-c))
        x=np.stack([layers_points[layer,edge[:,0]],layers_points[layer,edge[:,1]],
                    layers_points[layer+1,edge[:,1]],layers_points[layer+1,edge[:,0]]],axis=1)
        _,n=geometry(x);cc=cosine(n,c[neighbor]-c[owner]);np.minimum.at(val,owner,cc);np.minimum.at(val,neighbor,cc)
        if previous is not None:
            _,n=geometry(layers_points[layer,f]);cc=cosine(n,c-previous)
            val=np.minimum(val,cc);quality[-1]=np.minimum(quality[-1],cc)
        quality.append(val);previous=c
        jac=jacobians(q.reshape((-1,3)),np.arange(6*nf).reshape((-1,6)))
        minimum_jacobian=min(minimum_jacobian,float(jac.min()))
        bad=np.flatnonzero(val<.1)
        worst.append(dict(layer=layer+1,minimum=float(val.min()),below_01=len(bad)))
        print('LAYER',worst[-1],flush=True)
    quality=np.array(quality)
    summary=dict(first_full_normal_height_m=first,layers=layers,growth_quantiles=np.quantile(growth,[0,.01,.5,.99,1]).tolist(),
                 height_quantiles_m=np.quantile(heights[-1],[0,.01,.5,.99,1]).tolist(),
                 minimum=float(quality.min()),below_01=int((quality<.1).sum()),minimum_jacobian=minimum_jacobian,
                 maximum_neighbor_growth_difference=args.slope,safety=args.safety,by_layer=worst)
    np.savez(args.output/'optimized_layers.npz',points=layers_points,faces=f,direction=direction,growth=growth,quality=quality)
    np.savez(args.output/'prism_front.npz',points=p,faces=f,top=layers_points[-1],direction=direction,cap=heights[-1])
    (args.output/'layer_optimization.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
