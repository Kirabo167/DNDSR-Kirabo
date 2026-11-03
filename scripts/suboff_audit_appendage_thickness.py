#!/usr/bin/env python3
"""Measure physical normal-projected layer thickness and interface cell sizes."""
import argparse
import json
from pathlib import Path
import numpy as np

ap=argparse.ArgumentParser();ap.add_argument('directory',type=Path);args=ap.parse_args();D=args.directory
a=np.load(D/'SUBOFF_optimized.validated.npz');p=a['points'];t=a['cells_4']
f=np.load(D/'prism_front.npz')['faces'];nv=int(f.max())+1;nf=len(f)
q=p[f];c=q.mean(axis=1);cross=np.cross(q[:,1]-q[:,0],q[:,2]-q[:,0])
area=np.linalg.norm(cross,axis=1)/2;normal=cross/(2*area[:,None])
layers=len(a['cells_6'])//nf
top=p[f+layers*nv];prev=p[f+(layers-1)*nv]
thickness=np.einsum('ij,ij->i',(top-q).mean(axis=1),normal)
first=np.einsum('ij,ij->i',(p[f+nv]-q).mean(axis=1),normal)
last=np.einsum('ij,ij->i',(top-prev).mean(axis=1),normal)
radial=np.linalg.norm(c[:,1:],axis=1)
roof=(c[:,1]>.45)&(normal[:,1]>.5)
tip=(c[:,0]>3.7)&(((abs(abs(c[:,1])-.24384)<2e-5)&(abs(normal[:,1])>.995))|
                                ((abs(abs(c[:,2])-.24384)<2e-5)&(abs(normal[:,2])>.995)))
outer_fin=(c[:,0]>3.5)&(radial>.17)
regions={'whole_wall':np.ones(nf,dtype=bool),'sail_roof':roof,'outer_fin_surface_excluding_tip':outer_fin&~tip,'fin_tip_all':tip}
for axis in (1,2):
    for sign in (-1,1):regions[f'fin_tip_axis{axis}_sign{sign}']=tip&(sign*c[:,axis]>.24)
for lo,hi in ((.92,1.),(1.,1.13),(1.13,1.23),(1.23,1.30)):
    regions[f'roof_x_{lo}_{hi}']=roof&(c[:,0]>=lo)&(c[:,0]<hi)

# Match each prism-top face to its adjacent tetrahedron using exact node IDs.
def keys(v):
    v=np.ascontiguousarray(np.sort(v,axis=1),dtype=np.int64)
    return v.view(np.dtype([('a','<i8'),('b','<i8'),('c','<i8')])).ravel()
tf=((0,2,1),(0,1,3),(1,2,3),(2,0,3))
faces=np.vstack([t[:,ids] for ids in tf]);owner=np.tile(np.arange(len(t)),4)
k=keys(faces);order=np.argsort(k);target=keys(f+layers*nv);where=np.searchsorted(k[order],target)
assert np.array_equal(k[order[where]],target)
adjacent=owner[order[where]];del faces,k,order,owner
tc=p[t[adjacent]].mean(axis=1)
topn=np.cross(top[:,1]-top[:,0],top[:,2]-top[:,0]);topn/=np.linalg.norm(topn,axis=1)[:,None]
tet_height=4*np.einsum('ij,ij->i',tc-top.mean(axis=1),topn)
prism_height=np.einsum('ij,ij->i',(top-prev).mean(axis=1),topn)
assert tet_height.min()>0 and prism_height.min()>0
transition=tet_height/prism_height
quality=np.load(D/'orthogonal_quality_values.npy');prq=quality[len(t):].reshape(layers,nf)

def weighted(values,mask):
    ids=np.flatnonzero(mask);order=np.argsort(values[ids]);v=values[ids][order];w=area[ids][order]
    cdf=(np.cumsum(w)-.5*w)/w.sum()
    return dict(min=float(v.min()),p05=float(np.interp(.05,cdf,v)),median=float(np.interp(.5,cdf,v)),
                p95=float(np.interp(.95,cdf,v)),max=float(v.max()))
result={}
for name,mask in regions.items():
    if not mask.any():continue
    result[name]=dict(wall_triangles=int(mask.sum()),wall_area_m2=float(area[mask].sum()),
                     total_normal_projected_thickness_mm=weighted(1000*thickness,mask),
                     first_normal_projected_height_um=weighted(1e6*first,mask),
                     last_normal_projected_height_mm=weighted(1000*last,mask),
                     adjacent_tetra_height_mm=weighted(1000*tet_height,mask),
                     tetra_to_last_prism_normal_height_ratio=weighted(transition,mask),
                     min_prism_quality=float(prq[:,mask].min()),
                     min_adjacent_tetra_quality=float(quality[adjacent[mask]].min()),
                     area_fraction_total_thickness_below_5mm=float(area[mask&(thickness<.005)].sum()/area[mask].sum()),
                     area_fraction_total_thickness_25_to_40mm=float(area[mask&(thickness>=.025)&(thickness<=.040)].sum()/area[mask].sum()))
report=dict(layers=layers,definition='Full-layer corresponding top-to-wall displacement projected on each wall triangle normal, averaged over its three vertices; face-area weighted percentiles.',
            region_selection=dict(sail_roof='wall center y>0.45 m and outward normal ny>0.5',
                                  outer_fin_surface='wall center x>3.5 m, radial distance from X axis >0.17 m; excludes end faces',
                                  fin_tip='wall center x>3.7 m and y or z at +/-0.24384 m within 20 micrometers; radial normal component >0.995'),
            interface_ratio_note='Adjacent tetrahedron altitude / last prism mean normal height, measured along the prism-top triangle normal. This is a shape-dependent size comparison, not the prism-to-prism growth ratio.',
            regions=result,actual_flow_yplus_verified=False)
(D/'appendage_thickness_audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
