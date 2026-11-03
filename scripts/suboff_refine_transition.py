#!/usr/bin/env python3
"""Conformingly subdivide selected interface tetrahedra at interior centroids."""
import argparse
import json
from pathlib import Path
import numpy as np
from suboff_arrays_cgns import orient_validate

ap=argparse.ArgumentParser();ap.add_argument('source',type=Path);ap.add_argument('output',type=Path)
ap.add_argument('--ratio',type=float,default=6.);args=ap.parse_args();S=args.source;D=args.output;D.mkdir(parents=True,exist_ok=True)
raw=np.load(S/'SUBOFF_optimized.validated.npz');a=np.load(S/'optimized_coordinates.npz')
p=a['points'];t=a['cells_4'];pr=a['cells_6'];front=np.load(S/'prism_front.npz');f=front['faces'];nv=int(f.max())+1;nl=len(pr)//len(f)
def keys(x):return np.ascontiguousarray(np.sort(x,axis=1),dtype=np.int64).view(np.dtype([('a','<i8'),('b','<i8'),('c','<i8')])).ravel()
templates=((0,2,1),(0,1,3),(1,2,3),(2,0,3))
k=keys(np.vstack([t[:,ids] for ids in templates]));order=np.argsort(k);owner=np.tile(np.arange(len(t)),4)
target=keys(f+nl*nv);where=np.searchsorted(k[order],target);assert np.array_equal(k[order[where]],target)
adjacent=owner[order[where]];top=p[f+nl*nv];prev=p[f+(nl-1)*nv];fc=top.mean(axis=1)
n=np.cross(top[:,1]-top[:,0],top[:,2]-top[:,0]);n/=np.linalg.norm(n,axis=1)[:,None]
opposite=p[t[adjacent]].sum(axis=1)-top.sum(axis=1)
height=np.einsum('ij,ij->i',opposite-fc,n);last=np.einsum('ij,ij->i',(top-prev).mean(axis=1),n)
wall=p[f].mean(axis=1)
patch=((wall[:,0]>.85)&(wall[:,0]<1.4)&(wall[:,1]>.30))|((wall[:,0]>3.5)&(np.linalg.norm(wall[:,1:],axis=1)>.17))
selected=np.unique(adjacent[patch&(height>args.ratio*last)&(height>.002)])
keep=np.ones(len(t),dtype=bool);keep[selected]=False;ids=len(p)+np.arange(len(selected))
points=np.vstack([p,p[t[selected]].mean(axis=1)])
tet=np.vstack([t[keep],*[np.c_[t[selected][:,face],ids] for face in templates]])
minimum=orient_validate(points,tet)
np.savez(D/'SUBOFF_optimized.validated.npz',points=points,cells_4=tet,cells_6=pr,**{k:raw[k] for k in raw.files if k.startswith('boundary_')})
np.savez(D/'prism_front.npz',points=p[:nv],faces=f,top=p[nl*nv:(nl+1)*nv])
for name in ['fixed_boundary.npz','boundary_arrays.npz']:
    (D/name).write_bytes((S/name).read_bytes())
plan=json.loads((S/'thickening_plan.json').read_text());plan['source']='data/mesh/SUBOFF_BL_smooth_Q02_20261102/final';plan['incremental_source']=str(S)
(D/'thickening_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
report=dict(source=str(S),method='Tetrahedral 1-to-4 subdivision at interior centroid; original tetrahedron faces and prism interface preserved',selected_tetrahedra=len(selected),added_tetrahedra=3*len(selected),added_nodes=len(selected),cells=len(tet)+len(pr),minimum_tetra_jacobian=minimum,interface_ratio_threshold=args.ratio)
(D/'core_generation.json').write_text(json.dumps(report,indent=2)+'\n')
np.savez(D/'refinement_provenance.npz',source_tetrahedra=selected,new_nodes=ids,source_interface_ratio=height/last)
print(json.dumps(report),flush=True)
