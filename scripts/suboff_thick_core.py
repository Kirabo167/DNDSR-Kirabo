#!/usr/bin/env python3
"""Regenerate the outer tetrahedral core with a local sizing background."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from scipy.spatial import cKDTree
from suboff_arrays_cgns import orient_validate

ap=argparse.ArgumentParser();ap.add_argument('source',type=Path);ap.add_argument('directory',type=Path)
ap.add_argument('--switches',default='pq1.3/15YQO5/7S2000000')
ap.add_argument('--sizing',action='store_true')
ap.add_argument('--refine-interface',action='store_true');args=ap.parse_args();S=args.source;D=args.directory
lp=np.load(D/'optimized_layers.npz')['points'];f=np.load(D/'prism_front.npz')['faces'];nv=len(lp[0]);layers=len(lp)-1
source=np.load(S/'SUBOFF_optimized.validated.npz');old=source['points'];ot=source['cells_4']
boundaries={name:source['boundary_'+name] for name in ('INLET','OUTLET','FAR')}
outerfaces=np.vstack(list(boundaries.values()));used,inverse=np.unique(outerfaces,return_inverse=True)
outer=old[used];of=inverse.reshape(-1,3);cp=np.vstack([lp[-1],outer]);cf=np.vstack([f[:,[0,2,1]],of+nv])
sys.path.insert(0,'/tmp/suboff_tetgen_pkg');import tetgen
sites=np.empty((0,3))
if args.refine_interface:
    previous=np.load(D/'SUBOFF_optimized.validated.npz');bp=previous['points'];bt=previous['cells_4']
    def keys(a):
        return np.ascontiguousarray(np.sort(a,axis=1),dtype=np.int64).view(np.dtype([('a','<i8'),('b','<i8'),('c','<i8')])).ravel()
    templates=((0,2,1),(0,1,3),(1,2,3),(2,0,3))
    k=keys(np.vstack([bt[:,ids] for ids in templates]));order=np.argsort(k)
    owner=np.tile(np.arange(len(bt)),4);topfaces=f+layers*nv;target=keys(topfaces)
    where=np.searchsorted(k[order],target);assert np.array_equal(k[order[where]],target)
    adjacent=owner[order[where]];tri=bp[topfaces];fc=tri.mean(axis=1)
    n=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);n/=np.linalg.norm(n,axis=1)[:,None]
    opposite=bp[bt[adjacent]].sum(axis=1)-tri.sum(axis=1)
    height=np.einsum('ij,ij->i',opposite-fc,n)
    last=np.einsum('ij,ij->i',(lp[-1,f]-lp[-2,f]).mean(axis=1),n)
    select=height>4*last
    alpha=np.minimum(.55,np.maximum(.12,2*last/height))
    sites=(fc+alpha[:,None]*(opposite-fc))[select]
    # Each site is strictly inside an existing tetrahedron. Avoid inserting
    # coincident sites from multiple boundary faces of the same tetrahedron.
    _,take=np.unique(np.round(sites/1e-7).astype(np.int64),axis=0,return_index=True);sites=sites[take]
    tree=cKDTree(sites);keep=np.ones(len(sites),dtype=bool)
    for i in range(len(sites)):
        if keep[i]:
            neighbors=tree.query_ball_point(sites[i],.0005)
            keep[[j for j in neighbors if j>i]]=False
    sites=sites[keep]
    np.savez(D/'interface_refinement_sites.npz',points=sites,selected_faces=np.flatnonzero(select),
             old_normal_height_ratio=height/last)
    print('INTERFACE_REFINEMENT_POINTS',len(sites),flush=True)
core=tetgen.TetGen(np.vstack([cp,sites]),cf);core.add_hole([2.,0.,0.])
kw={};switches=args.switches
if args.sizing:
    import pyvista as pv
    # Preserve the coarse far field while requesting smaller tetrahedra
    # beside short outer prisms. The field grows with distance to the front.
    hlast=np.linalg.norm(lp[-1]-lp[-2],axis=1)
    edge=np.vstack([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]);elen=np.linalg.norm(lp[-1,edge[:,0]]-lp[-1,edge[:,1]],axis=1)
    es=np.zeros(nv)
    for j in (0,1):np.maximum.at(es,edge[:,j],elen)
    local=np.maximum(2*hlast,1.2*es)
    u,inv=np.unique(ot,return_inverse=True);bt=inv.reshape(-1,4);bp=old[u]
    distance,near=cKDTree(lp[-1]).query(bp)
    size=np.maximum(.0015,local[near]+.6*distance)
    bg=pv.UnstructuredGrid({pv.CellType.TETRA:bt},bp);bg.point_data['target_size']=size
    kw['bgmesh']=bg;switches+='m'
    print('SIZING',np.quantile(size,[0,.01,.5,.99,1]),flush=True)
print('TETGEN_BEGIN',switches,len(cp),len(cf),flush=True)
pp,tt,_,_=core.tetrahedralize(switches=switches,**kw)
distance,mapping=cKDTree(pp).query(cp);assert distance.max()<1e-10
raw=set(map(tuple,np.sort(core.trifaces,axis=1)))
assert all(tuple(x) in raw for x in np.sort(mapping[cf],axis=1))
nlayer=(layers+1)*nv;pmap=np.full(len(pp),-1,dtype=np.int64)
pmap[mapping[:nv]]=layers*nv+np.arange(nv)
pmap[mapping[nv:]]=nlayer+np.arange(len(outer));additional=np.flatnonzero(pmap<0)
pmap[additional]=nlayer+len(outer)+np.arange(len(additional))
points=np.vstack([lp.reshape(-1,3),outer,pp[additional]]);t=pmap[tt]
pr=np.vstack([np.c_[f+k*nv,f+(k+1)*nv] for k in range(layers)])
jmin=orient_validate(points,t)
remap=np.full(len(old),-1,dtype=np.int64);remap[used]=nlayer+np.arange(len(outer))
boundaries={name:remap[face] for name,face in boundaries.items()};boundaries['WALL']=f
np.savez(D/'SUBOFF_optimized.validated.npz',points=points,cells_4=t,cells_6=pr,
         **{'boundary_'+name:face for name,face in boundaries.items()})
np.savez(D/'boundary_arrays.npz',**boundaries)
fixed_layers=json.loads((D/'thickening_plan.json').read_text())['fixed_layers']
np.savez(D/'fixed_boundary.npz',nodes=nlayer+np.arange(len(outer)),wall_nodes=np.arange((fixed_layers+1)*nv))
summary=dict(source=str(S),layers=layers,cells=len(t)+len(pr),tetrahedra=len(t),prisms=len(pr),
             nodes=len(points),switches=switches,local_sizing=args.sizing,minimum_tetra_jacobian=jmin,
             constrained_facets_preserved=True,fixed_layers_at_generation=fixed_layers,interface_refinement_points=len(sites))
(D/'core_generation.json').write_text(json.dumps(summary,indent=2)+'\n')
print('CORE_DONE',json.dumps(summary),flush=True)
assert len(t)+len(pr)<10000000,'Over cell budget; revise sizing before export'
