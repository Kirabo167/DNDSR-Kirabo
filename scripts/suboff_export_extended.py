#!/usr/bin/env python3
"""Validate and export the coupled-optimization result without changing the wall."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from suboff_arrays_cgns import orient_validate,write_cgns
from suboff_optimize_layers import centroids


def main():
    ap=argparse.ArgumentParser();ap.add_argument('directory',type=Path);ap.add_argument('output',type=Path)
    ap.add_argument('--library',type=Path,required=True);args=ap.parse_args();D=args.directory;out=args.output;out.mkdir(parents=True,exist_ok=True)
    original=np.load(D/'SUBOFF_optimized.validated.npz');optimized=np.load(D/'optimized_coordinates.npz')
    p=optimized['points'];t=optimized['cells_4'].copy();pr=optimized['cells_6'].copy()
    jt=orient_validate(p,t);jp=orient_validate(p,pr)
    bd=np.load(D/'boundary_arrays.npz');boundaries={k:bd[k] for k in bd.files}
    fixed=np.unique(boundaries['WALL']);assert np.array_equal(p[fixed],original['points'][fixed])
    nv=len(np.load(D/'prism_front.npz')['points'])
    exterior=np.load(D/'fixed_boundary.npz')['nodes']
    assert np.array_equal(p[exterior],original['points'][exterior])
    reference=original['points'];direction=reference[nv:2*nv]-reference[:nv]
    direction/=np.linalg.norm(direction,axis=1)[:,None]
    first_change=np.einsum('ij,ij->i',p[nv:2*nv]-reference[nv:2*nv],direction)
    assert np.abs(first_change).max()<1e-12
    volume=0.;minimum_volume=float('inf')
    for i in range(0,len(t),100000):
        q=p[t[i:i+100000]];v=np.einsum('ij,ij->i',np.cross(q[:,1]-q[:,0],q[:,2]-q[:,0]),q[:,3]-q[:,0])/6
        volume+=v.sum();minimum_volume=min(minimum_volume,float(v.min()))
    for i in range(0,len(pr),100000):
        _,v=centroids(p[pr[i:i+100000]]);volume+=v.sum();minimum_volume=min(minimum_volume,float(v.min()))
    wallp=p[boundaries['WALL']];body_volume=np.einsum('ij,ij->i',wallp[:,0],np.cross(wallp[:,1],wallp[:,2])).sum()/6
    # The source STL box coordinates have float32 rounding. Use the actual
    # fixed box, not nominal decimal dimensions, for the closure identity.
    expected=np.prod(p.max(axis=0)-p.min(axis=0))-body_volume
    relative_error=abs(volume-expected)/expected
    assert relative_error<1e-10
    file=out/'SUBOFF_30Lx20Lx20L_Q02.cgns';write_cgns(file,args.library,p,{4:t,6:pr},boundaries)
    np.savez(out/'SUBOFF_optimized.validated.npz',points=p,cells_4=t,cells_6=pr,**{f'boundary_{k}':v for k,v in boundaries.items()})
    front=np.load(D/'prism_front.npz');np.savez(out/'prism_front.npz',points=p[:nv],faces=front['faces'],top=p[40*nv:41*nv])
    # Layer-1 normal displacement at each triangle vertex.
    f=front['faces'];q=p[f];n=np.cross(q[:,1]-q[:,0],q[:,2]-q[:,0]);n/=np.linalg.norm(n,axis=1)[:,None]
    normal_height=np.einsum('fij,fj->fi',(p[nv:2*nv]-p[:nv])[f],n)
    layer_points=p[:41*nv].reshape((41,nv,3))
    first_direction=layer_points[1]-layer_points[0]
    first_direction/=np.linalg.norm(first_direction,axis=1)[:,None]
    projected=np.einsum('kij,ij->ki',layer_points[1:]-layer_points[:-1],first_direction)
    growth=projected[1:]/projected[:-1]
    summary=dict(cells=len(t)+len(pr),tetrahedra=len(t),prisms=len(pr),nodes=len(p),
                 minimum_jacobian={'TETRA_4':jt,'PENTA_6':jp},minimum_volume_m3=minimum_volume,
                 volume_m3=float(volume),expected_volume_m3=float(expected),relative_volume_closure_error=float(relative_error),
                 wall_coordinates_unchanged=True,first_layer_projected_height_preserved=True,
                 first_layer_projection_max_change_m=float(np.abs(first_change).max()),layers=40,
                 domain=json.loads((D/'domain_extension.json').read_text()),
                 first_layer_normal_height_m_quantiles=np.quantile(normal_height,[0,.01,.5,.99,1]).tolist(),
                 projected_increment_growth_quantiles=np.quantile(growth,[0,.001,.01,.5,.99,.999,1]).tolist(),
                 projected_growth_above_1_3=int((growth>1.3).sum()),
                 column_displacement_m_quantiles=np.quantile(np.linalg.norm(layer_points[-1]-layer_points[0],axis=1),[0,.01,.5,.99,1]).tolist(),
                 outer_layers_locally_redistributed=True,
                 boundary_faces={k:len(v) for k,v in boundaries.items()},
                 sha256=hashlib.sha256(file.read_bytes()).hexdigest(),flow_yplus_verified=False)
    (out/'mesh_validation.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
