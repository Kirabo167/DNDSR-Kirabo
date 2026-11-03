#!/usr/bin/env python3
"""Remesh appendage cavities while retaining the rest of a validated core."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from scipy.spatial import cKDTree
from suboff_arrays_cgns import orient_validate

ap=argparse.ArgumentParser();ap.add_argument('source',type=Path);ap.add_argument('layers',type=Path);ap.add_argument('output',type=Path);ap.add_argument('--protect-nodes',type=Path);ap.add_argument('--retain-fixed-front',action='store_true');args=ap.parse_args();S,L,D=args.source,args.layers,args.output;D.mkdir(exist_ok=False)
raw=np.load(S/'SUBOFF_optimized.validated.npz');a=np.load(S/'optimized_coordinates.npz');p=a['points'];t=a['cells_4'];pr=a['cells_6'];front=np.load(S/'prism_front.npz');f=front['faces'];nv=int(f.max())+1;nl=len(pr)//len(f);nlayer=(nl+1)*nv
desired=np.load(L/'SUBOFF_optimized.validated.npz')['points'][:nlayer].reshape(nl+1,nv,3)
old=p[:nlayer].reshape(nl+1,nv,3);moved=np.flatnonzero(np.linalg.norm(desired[-1]-old[-1],axis=1)>1e-12)
target=p.copy();target[:nlayer]=desired.reshape(-1,3)
distance,_=cKDTree(old[-1,moved]).query(p[t].mean(axis=1))
selected=(distance<.04)|np.isin(t,nl*nv+moved).any(axis=1)
# One node ring keeps the artificial cavity boundary away from moved nodes.
selected |= np.isin(t,np.unique(t[selected])).any(axis=1)
fixed=np.load(L/'fixed_boundary.npz');isfixed=np.zeros(len(p),dtype=bool);isfixed[fixed['wall_nodes']]=True
selected &= ~isfixed[t].all(axis=1)
if args.retain_fixed_front:
    selected &= ~isfixed[t].any(axis=1)
    qt=target[t];jj=np.einsum('ij,ij->i',np.cross(qt[:,1]-qt[:,0],qt[:,2]-qt[:,0]),qt[:,3]-qt[:,0])
    selected |= jj<=0
if args.protect_nodes:
    selected &= ~np.isin(t,np.load(args.protect_nodes)).any(axis=1)
qt=target[t[~selected]]
assert np.all(np.einsum('ij,ij->i',np.cross(qt[:,1]-qt[:,0],qt[:,2]-qt[:,0]),qt[:,3]-qt[:,0])>0)
templates=((0,2,1),(0,1,3),(1,2,3),(2,0,3))
faces=np.vstack([t[selected][:,ids] for ids in templates]);key=np.sort(faces,axis=1);order=np.lexsort((key[:,2],key[:,1],key[:,0]));sorted_key=key[order]
same=(sorted_key[1:]==sorted_key[:-1]).all(axis=1);interior=np.zeros(len(faces),dtype=bool);interior[order[:-1][same]]=True;interior[order[1:][same]]=True
boundary=faces[~interior];used,inv=np.unique(boundary,return_inverse=True);cf=inv.reshape(-1,3)
cp=target[used]
print('CAVITY',int(selected.sum()),'KEPT_TETRAHEDRA',int((~selected).sum()),'BOUNDARY',len(boundary),flush=True)
sys.path.insert(0,'/tmp/suboff_tetgen_pkg');import tetgen
core=tetgen.TetGen(cp,cf);pp,tt,_,_=core.tetrahedralize(switches='pq1.3/20YQO8/7S500000')
minimum=orient_validate(pp,tt)
dist,mapping=cKDTree(pp).query(cp);assert dist.max()<1e-10
available=set(map(tuple,np.sort(core.trifaces,axis=1)));assert all(tuple(x) in available for x in np.sort(mapping[cf],axis=1))
tri=cp[cf];expected=float(np.einsum('ij,ij->i',tri[:,0],np.cross(tri[:,1],tri[:,2])).sum()/6)
qt=pp[tt];volume=float(np.einsum('ij,ij->i',np.cross(qt[:,1]-qt[:,0],qt[:,2]-qt[:,0]),qt[:,3]-qt[:,0]).sum()/6)
assert abs(volume-expected)<1e-8*max(expected,1e-6),(volume,expected)
kept=t[~selected];old_used=np.unique(np.r_[kept.ravel(),used]);old_core=old_used[old_used>=nlayer]
remap=np.full(len(p),-1,dtype=np.int64);remap[:nlayer]=np.arange(nlayer);remap[old_core]=nlayer+np.arange(len(old_core))
pmap=np.full(len(pp),-1,dtype=np.int64);pmap[mapping]=remap[used];new=np.flatnonzero(pmap<0);pmap[new]=nlayer+len(old_core)+np.arange(len(new))
points=np.vstack([desired.reshape(-1,3),p[old_core],pp[new]]);tet=np.vstack([remap[kept],pmap[tt]])
boundaries={k.removeprefix('boundary_'):remap[raw[k]] for k in raw.files if k.startswith('boundary_')}
assert min(x.min() for x in boundaries.values())>=0
np.savez(D/'SUBOFF_optimized.validated.npz',points=points,cells_4=tet,cells_6=pr,**{'boundary_'+k:v for k,v in boundaries.items()})
np.savez(D/'prism_front.npz',points=points[:nv],faces=f,top=desired[-1]);np.savez(D/'boundary_arrays.npz',**boundaries)
outside=np.unique(np.r_[boundaries['INLET'].ravel(),boundaries['OUTLET'].ravel(),boundaries['FAR'].ravel()])
np.savez(D/'fixed_boundary.npz',nodes=outside,wall_nodes=fixed['wall_nodes'])
plan=json.loads((L/'thickening_plan.json').read_text());plan['method']='local appendage expansion, geometric layer redistribution, and local cavity remeshing';plan['source']='data/mesh/SUBOFF_BL_smooth_Q02_20261102/final';plan['local_core_source']=str(S)
(D/'thickening_plan.json').write_text(json.dumps(plan,indent=2)+'\n');(D/'local_targets.npz').write_bytes((L/'local_targets.npz').read_bytes())
report=dict(source=str(S),local_cavity=True,original_tetrahedra_kept=int((~selected).sum()),original_tetrahedra_replaced=int(selected.sum()),new_cavity_tetrahedra=len(tt),tetrahedra=len(tet),prisms=len(pr),cells=len(tet)+len(pr),minimum_new_tetra_jacobian=minimum,constrained_facets_preserved=True,cavity_volume_m3=volume,cavity_expected_volume_m3=expected)
fixed_points=np.zeros(len(points),dtype=bool);fixed_points[np.r_[outside,fixed['wall_nodes']]]=True
immovable=np.flatnonzero(fixed_points[tet].all(axis=1));qt=points[tet[immovable]];center=qt.mean(axis=1);cfmin=np.ones(len(qt))
for face in templates:
    tri=qt[:,face];nn=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);dd=tri.mean(axis=1)-center
    cfmin=np.minimum(cfmin,np.einsum('ij,ij->i',nn,dd)/(np.linalg.norm(nn,axis=1)*np.linalg.norm(dd,axis=1)))
bad_nodes=np.unique(tet[immovable[cfmin<.2]])
assert (bad_nodes<nlayer).all()
if args.protect_nodes:bad_nodes=np.union1d(bad_nodes,np.load(args.protect_nodes))
np.save(D/'protect_nodes.npy',bad_nodes)
report['immovable_bad_cells']=int((cfmin<.2).sum())
(D/'core_generation.json').write_text(json.dumps(report,indent=2)+'\n');np.savez(D/'cavity_provenance.npz',replaced_source_tetrahedra=np.flatnonzero(selected),kept_source_tetrahedra=np.flatnonzero(~selected),old_node_remap=remap)
print(json.dumps(report),flush=True)
