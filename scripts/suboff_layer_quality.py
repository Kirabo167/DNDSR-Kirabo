#!/usr/bin/env python3
"""Exact prism-only quality audit before generating the tetrahedral core."""
import argparse
import json
from pathlib import Path
import numpy as np
from suboff_optimize_layers import centroids,geometry,cosine

ap=argparse.ArgumentParser();ap.add_argument('directory',type=Path);args=ap.parse_args();D=args.directory
a=np.load(D/'optimized_layers.npz');lp=a['points'];f=a['faces'];nf=len(f);nv=len(lp[0]);layers=len(lp)-1
edges=np.vstack([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]);own=np.tile(np.arange(nf),3)
key=np.sort(edges,axis=1);order=np.lexsort((key[:,1],key[:,0]));ia,ib=order[::2],order[1::2]
assert np.array_equal(key[ia],key[ib]);edge=edges[ia];owner=own[ia];neighbor=own[ib]
quality=[];previous=None;bylayer=[]
for k in range(layers):
    q=np.concatenate([lp[k,f],lp[k+1,f]],axis=1);c,_=centroids(q);value=np.ones(nf);ids=np.c_[f+k*nv,f+(k+1)*nv]
    for ff in ((0,2,1),(3,4,5),(0,1,4,3),(1,2,5,4),(2,0,3,5)):
        fc,n=geometry(q[:,ff],ids[:,ff]);value=np.minimum(value,cosine(n,fc-c))
    x=np.stack([lp[k,edge[:,0]],lp[k,edge[:,1]],lp[k+1,edge[:,1]],lp[k+1,edge[:,0]]],axis=1)
    _,n=geometry(x);cc=cosine(n,c[neighbor]-c[owner]);np.minimum.at(value,owner,cc);np.minimum.at(value,neighbor,cc)
    if previous is not None:
        _,n=geometry(lp[k,f]);cc=cosine(n,c-previous);value=np.minimum(value,cc);quality[-1]=np.minimum(quality[-1],cc)
    quality.append(value);previous=c
    print('LAYER',k+1,'Q',float(value.min()),'BAD',int((value<.2).sum()),flush=True)
quality=np.array(quality);np.save(D/'orthogonal_quality_values.npy',quality.ravel())
di=lp[1]-lp[0];di/=np.linalg.norm(di,axis=1)[:,None]
dp=np.diff(lp,axis=0);h=np.einsum('kij,ij->ki',dp,di);length=np.linalg.norm(dp,axis=2)
summary=dict(minimum=float(quality.min()),cells_below_02=int((quality<.2).sum()),layers=layers,
             max_projected_growth=float((h[1:]/h[:-1]).max()),max_euclidean_growth=float((length[1:]/length[:-1]).max()),
             by_layer=[dict(layer=k+1,minimum=float(q.min()),bad_cells=int((q<.2).sum())) for k,q in enumerate(quality)])
(D/'layer_quality.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)
