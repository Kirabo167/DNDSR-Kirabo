#!/usr/bin/env python3
"""Redistribute appendage layer spacing along existing curved columns."""
import argparse
import json
from pathlib import Path
import numpy as np
from suboff_arrays_cgns import jacobians

ap=argparse.ArgumentParser();ap.add_argument('source',type=Path);ap.add_argument('output',type=Path);args=ap.parse_args();S=args.source;D=args.output;D.mkdir(exist_ok=False)
a=np.load(S/'SUBOFF_optimized.validated.npz');p=a['points'].copy();pr=a['cells_6'];f=np.load(S/'prism_front.npz')['faces'];nv=int(f.max())+1;nl=len(pr)//len(f);freeze=16
old=p[:(nl+1)*nv].reshape(nl+1,nv,3).copy();b=old[0];direction=old[1]-old[0];direction/=np.linalg.norm(direction,axis=1)[:,None]
patch=((b[:,0]>.85)&(b[:,0]<1.4)&(b[:,1]>.30))|((b[:,0]>3.5)&(np.linalg.norm(b[:,1:],axis=1)>.17))
ids=np.flatnonzero(patch);height=np.einsum('kij,ij->ki',old-old[freeze],direction);previous=height[freeze]-height[freeze-1];count=nl-freeze
lo=np.full(nv,.3);hi=np.full(nv,1.244)
capacity=previous*sum(1.244**k for k in range(1,count+1));assert np.all(height[-1,ids]<capacity[ids]),'Layer count insufficient for some requested columns'
for _ in range(55):
    g=(lo+hi)/2;total=previous*np.sum(g[None,:]**np.arange(1,count+1)[:,None],axis=0);small=total<height[-1];lo=np.where(small,g,lo);hi=np.where(small,hi,g)
g=(lo+hi)/2;desired=previous[None,:]*np.cumsum(g[None,:]**np.arange(1,count+1)[:,None],axis=0)
new=old.copy()
for k in range(1,count):
    h=desired[k-1,ids];index=np.maximum(freeze,np.minimum(nl-1,np.sum(height[:,ids]<=h[None,:],axis=0)-1))
    fraction=(h-height[index,ids])/(height[index+1,ids]-height[index,ids])
    new[freeze+k,ids]=old[index,ids]+fraction[:,None]*(old[index+1,ids]-old[index,ids])
change=new-old;factor=np.ones(nv)
edges=np.unique(np.sort(np.vstack([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]),axis=1),axis=0)
for attempt in range(35):
    test=old+change*factor[None,:,None];bad=np.zeros(len(f),dtype=bool);minimum=float('inf')
    for start in range(freeze*len(f),len(pr),100000):
        j=jacobians(test.reshape(-1,3),pr[start:start+100000]);minimum=min(minimum,float(j.min()));bad[(start+np.flatnonzero((j<=0).any(axis=1)))%len(f)]=True
    print('BACKOFF',attempt,'BAD',int(bad.sum()),'MIN_JAC',minimum,flush=True)
    if not bad.any():break
    factor[np.unique(f[bad])]*=.6
    for _ in range(3):
        before=factor.copy();np.minimum.at(factor,edges[:,0],before[edges[:,1]]+.1);np.minimum.at(factor,edges[:,1],before[edges[:,0]]+.1)
else:raise RuntimeError('Could not redistribute with positive Jacobians')
p[:(nl+1)*nv]=test.reshape(-1,3);assert np.array_equal(test[-1],old[-1]) and np.array_equal(test[:freeze+1],old[:freeze+1])
np.savez(D/'SUBOFF_optimized.validated.npz',**{k:p if k=='points' else a[k] for k in a.files})
for name in ['prism_front.npz','fixed_boundary.npz','boundary_arrays.npz','local_targets.npz','core_generation.json']:
    if (S/name).exists():(D/name).write_bytes((S/name).read_bytes())
plan=json.loads((S/'thickening_plan.json').read_text());plan['redistribution_source']=str(S);plan['source']='data/mesh/SUBOFF_BL_smooth_Q02_20261102/final';plan['geometric_growth_quantiles']=np.quantile(g[ids],[0,.05,.5,.95,1]).tolist();plan['spacing_backoff_quantiles']=np.quantile(factor[ids],[0,.05,.5,.95,1]).tolist()
(D/'thickening_plan.json').write_text(json.dumps(plan,indent=2)+'\n');print(json.dumps(plan),flush=True)
