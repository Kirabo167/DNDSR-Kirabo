#!/usr/bin/env python3
"""Add a coarse tetrahedral shell without replacing the existing hybrid mesh."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from scipy.spatial import cKDTree
from suboff_arrays_cgns import orient_validate


def box_surface(lo,hi,spacing):
    counts=np.maximum(1,np.ceil((hi-lo)/spacing).astype(int))
    nodes={};points=[];faces=[]
    def node(key):
        key=tuple(key)
        if key not in nodes:
            nodes[key]=len(points);points.append(lo+(hi-lo)*np.array(key)/counts)
        return nodes[key]
    for axis in range(3):
        other=[i for i in range(3) if i!=axis]
        for side in (0,1):
            for i in range(counts[other[0]]):
                for j in range(counts[other[1]]):
                    ids=[]
                    for a,b in ((i,j),(i+1,j),(i+1,j+1),(i,j+1)):
                        key=[0,0,0];key[axis]=side*counts[axis];key[other[0]]=a;key[other[1]]=b;ids.append(node(key))
                    pair=((0,1,2),(0,2,3)) if (i+j)%2==0 else ((0,1,3),(1,2,3))
                    for tri in pair:
                        f=[ids[k] for k in tri];p=np.array([points[k] for k in f]);normal=np.cross(p[1]-p[0],p[2]-p[0])
                        if normal[axis]*(2*side-1)<0:f=[f[0],f[2],f[1]]
                        faces.append(f)
    return np.array(points),np.array(faces,dtype=np.int64)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('source',type=Path);ap.add_argument('output',type=Path)
    ap.add_argument('--outer-spacing',type=float,default=10.)
    ap.add_argument('--switches',default='pq1.3/20YQO5/7')
    args=ap.parse_args();S=args.source;D=args.output;D.mkdir(parents=True,exist_ok=True)
    data=np.load(S/'SUBOFF_optimized.validated.npz');old=data['points'];t=data['cells_4'];pr=data['cells_6']
    inner_faces=np.vstack([data['boundary_'+name] for name in ('INLET','OUTLET','FAR')])
    used,inverse=np.unique(inner_faces,return_inverse=True);inner=old[used];fi=inverse.reshape((-1,3))
    L=4.356;lo=np.array([-29/3,-10,-10])*L;hi=np.array([61/3,10,10])*L
    op,of=box_surface(lo,hi,args.outer_spacing)
    cp=np.vstack([inner,op]);cf=np.vstack([fi[:,[0,2,1]],of+len(inner)])
    sys.path.insert(0,'/tmp/suboff_tetgen_pkg');import tetgen
    generator=tetgen.TetGen(cp,cf);generator.add_hole((inner.min(axis=0)+inner.max(axis=0))/2)
    print('SHELL_BEGIN',len(cp),len(cf),args.switches,flush=True)
    pp,tt,_,_=generator.tetrahedralize(switches=args.switches)
    distance,mapping=cKDTree(pp).query(cp)
    assert distance.max()<1e-10
    fs=set(map(tuple,np.sort(generator.trifaces,axis=1)))
    assert all(tuple(x) in fs for x in np.sort(mapping[cf],axis=1))
    remap=np.full(len(pp),-1,dtype=np.int64);remap[mapping[:len(inner)]]=used
    additional=np.flatnonzero(remap<0);remap[additional]=len(old)+np.arange(len(additional))
    points=np.vstack([old,pp[additional]]);shell=remap[tt];orient_validate(points,shell)
    tet=np.vstack([t,shell]);new_outer=remap[mapping[len(inner):]];boundary= new_outer[of]
    centers=points[boundary].mean(axis=1);inlet=np.isclose(centers[:,0],lo[0],atol=1e-9);outlet=np.isclose(centers[:,0],hi[0],atol=1e-9)
    boundaries=dict(WALL=data['boundary_WALL'],INLET=boundary[inlet],OUTLET=boundary[outlet],FAR=boundary[~(inlet|outlet)])
    np.savez(D/'SUBOFF_optimized.validated.npz',points=points,cells_4=tet,cells_6=pr,**{f'boundary_{k}':v for k,v in boundaries.items()})
    front=np.load(S/'prism_front.npz');np.savez(D/'prism_front.npz',**{k:front[k] for k in front.files})
    np.savez(D/'fixed_boundary.npz',nodes=np.unique(boundary),wall_nodes=np.unique(boundaries['WALL']))
    np.savez(D/'boundary_arrays.npz',**boundaries)
    summary=dict(source=str(S),source_cgns_sha256=hashlib.sha256((S/'SUBOFF_optimized.cgns').read_bytes()).hexdigest(),
                 source_cells=len(t)+len(pr),cells=len(tet)+len(pr),added_tetrahedra=len(shell),
                 increase_percent=100*len(shell)/(len(t)+len(pr)),tetrahedra=len(tet),prisms=len(pr),nodes=len(points),
                 length_m=L,bounds_m=[lo.tolist(),hi.tolist()],dimensions_L=[30,20,20],dimensions_m=(hi-lo).tolist(),
                 upstream_clearance_L=29/3,downstream_clearance_L=58/3,
                 old_mesh_nodes_and_cells_preserved_before_quality_optimization=True,
                 conformal_old_outer_interface=True,outer_surface_faces=len(of),switches=args.switches)
    (D/'domain_extension.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
