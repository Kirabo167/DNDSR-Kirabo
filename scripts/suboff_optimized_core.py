#!/usr/bin/env python3
"""Generate a constrained optimized tetrahedral core for optimized prism layers."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from scipy.spatial import cKDTree


def main():
    ap=argparse.ArgumentParser();ap.add_argument('source',type=Path);ap.add_argument('layers',type=Path);ap.add_argument('output',type=Path)
    ap.add_argument('--switches',default='pq1.2/15YQO5/7')
    ap.add_argument('--save-refined-core-only', action='store_true')
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    sys.path[:0]=['/tmp/suboff_tetgen_pkg','/tmp/suboff_regen_deps']
    import pymeshlab
    import tetgen
    data=np.load(args.layers/'optimized_layers.npz');lp=data['points'];f=data['faces'];nv=lp.shape[1];nf=len(f);layers=len(lp)-1
    check=pymeshlab.MeshSet();check.add_mesh(pymeshlab.Mesh(np.vstack([lp[0],lp[-1]]),np.vstack([f,f+nv])))
    check.apply_filter('compute_selection_by_self_intersections_per_face')
    intersections=check.current_mesh().selected_face_number()
    print('FRONT_SELF_INTERSECTIONS',intersections,flush=True)
    if intersections:raise ValueError('Intersecting wall/front surface')
    surface=np.load(args.source/'surface_arrays.npz');sp=surface['points'];sf=surface['triangles'];c=sp[sf].mean(axis=1)
    wall=(c[:,0]>-.01)&(c[:,0]<4.37)&(np.linalg.norm(c[:,1:],axis=1)<.7)
    used,inverse=np.unique(sf[~wall],return_inverse=True);outer=sp[used];of=inverse.reshape((-1,3))
    cp=np.vstack([lp[-1],outer]);cf=np.vstack([f[:,[0,2,1]],of+nv])
    core=tetgen.TetGen(cp,cf);core.add_hole([2.,0.,0.])
    print('TETGEN_BEGIN',args.switches,flush=True)
    p,t,_,_=core.tetrahedralize(switches=args.switches)
    if args.save_refined_core_only:
        np.savez(args.output/'refined_core.npz',points=p,tetrahedra=t,core_input=cp,core_faces=cf,
                 trifaces=core.trifaces,markers=core.triface_markers)
        print('REFINED_CORE_SAVED',len(p),len(t),flush=True)
        return
    distance,mapping=cKDTree(p).query(cp)
    if distance.max()>1e-11:raise ValueError('Moved constrained point')
    raw=np.sort(core.trifaces,axis=1);expected=np.sort(mapping[cf],axis=1)
    face_set=set(map(tuple,raw))
    if any(tuple(x) not in face_set for x in expected):raise ValueError('Changed constrained face')
    np.savez(args.output/'core.npz',points=p,tetrahedra=t,input_mapping=mapping,core_input=cp,core_faces=cf)
    nlayer=(layers+1)*nv
    pmap=np.full(len(p),-1,dtype=np.int64);pmap[mapping[:nv]]=layers*nv+np.arange(nv)
    pmap[mapping[nv:]]=nlayer+np.arange(len(outer));additional=np.flatnonzero(pmap<0)
    pmap[additional]=nlayer+len(outer)+np.arange(len(additional))
    points=np.vstack([lp.reshape((-1,3)),outer,p[additional]])
    t=pmap[t];prisms=np.vstack([np.c_[f+k*nv,f+(k+1)*nv] for k in range(layers)])
    nodes=np.zeros((len(t)+len(prisms),6),dtype=np.int64);nodes[:len(t),:4]=t+1;nodes[len(t):]=prisms+1
    np.savez(args.output/'mesh_arrays.npz',points=points,volume_nodes=nodes,
             volume_sizes=np.r_[np.full(len(t),4),np.full(len(prisms),6)],
             surface_nodes=np.vstack([f,of+nlayer])+1,surface_sizes=np.full(nf+len(of),3))
    # Save a direct array view for quality diagnostics before CGNS export.
    np.savez(args.output/'SUBOFF_optimized.validated.npz',points=points,cells_4=t,cells_6=prisms)
    np.savez(args.output/'prism_front.npz',points=lp[0],faces=f,top=lp[-1],direction=data['direction'])
    summary=dict(cells=len(t)+len(prisms),tetrahedra=len(t),prisms=len(prisms),nodes=len(points),
                 constrained_boundary_facets_preserved=True,front_self_intersections=int(intersections),switches=args.switches)
    (args.output/'core_generation.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
