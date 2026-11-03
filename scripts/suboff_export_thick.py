#!/usr/bin/env python3
"""Export a fully audited thickened hybrid mesh and preservation statistics."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from suboff_arrays_cgns import orient_validate,write_cgns
from suboff_optimize_layers import centroids

ap=argparse.ArgumentParser();ap.add_argument('directory',type=Path);ap.add_argument('output',type=Path)
ap.add_argument('--library',type=Path,required=True)
ap.add_argument('--cgns-name',default=None)
ap.add_argument('--preserve-existing-normal-growth',action='store_true')
args=ap.parse_args();D=args.directory;F=args.output;F.mkdir(exist_ok=True)
plan=json.loads((D/'thickening_plan.json').read_text());S=Path(plan['source'])
a=np.load(D/'optimized_coordinates.npz');p=a['points'];t=a['cells_4'].copy();pr=a['cells_6'].copy()
f=np.load(D/'prism_front.npz')['faces'];nv=int(f.max())+1;layers=len(pr)//len(f)
source=np.load(S/'SUBOFF_optimized.validated.npz');bd=np.load(D/'boundary_arrays.npz');boundaries={k:bd[k] for k in bd.files}
fixed_boundary=np.load(D/'fixed_boundary.npz')
fixed_wall_nodes=fixed_boundary['wall_nodes']
optimization_fixed_layers=plan['fixed_layers']
fixed_layers=plan['fixed_layers']
assert np.array_equal(p[:nv],source['points'][:nv])
outer=np.unique(np.vstack([boundaries[k] for k in ('INLET','OUTLET','FAR')]))
baseline=np.load(D/'SUBOFF_optimized.validated.npz');assert np.array_equal(p[outer],baseline['points'][outer])
assert np.array_equal(p[:(optimization_fixed_layers+1)*nv],baseline['points'][:(optimization_fixed_layers+1)*nv])
assert np.array_equal(p[fixed_wall_nodes],baseline['points'][fixed_wall_nodes])
jmin=dict(TETRA_4=orient_validate(p,t),PENTA_6=orient_validate(p,pr))
quality=json.loads((D/'final_quality.json').read_text());assert quality['minimum']>=.2
assert 2000000<len(t)+len(pr)<10000000
volume=0.;minimum_volume=float('inf')
for start in range(0,len(t),100000):
    q=p[t[start:start+100000]];v=np.einsum('ij,ij->i',np.cross(q[:,1]-q[:,0],q[:,2]-q[:,0]),q[:,3]-q[:,0])/6
    volume+=v.sum();minimum_volume=min(minimum_volume,float(v.min()))
for start in range(0,len(pr),100000):
    _,v=centroids(p[pr[start:start+100000]]);volume+=v.sum();minimum_volume=min(minimum_volume,float(v.min()))
wall=p[f];body_volume=np.einsum('ij,ij->i',wall[:,0],np.cross(wall[:,1],wall[:,2])).sum()/6
expected=np.prod(p.max(axis=0)-p.min(axis=0))-body_volume;error=abs(volume-expected)/expected;assert error<1e-10
lp=p[:(layers+1)*nv].reshape(layers+1,nv,3)
direction=lp[1]-lp[0];direction/=np.linalg.norm(direction,axis=1)[:,None]
increment=np.diff(lp,axis=0);projected=np.einsum('kij,ij->ki',increment,direction)
euclidean=np.linalg.norm(increment,axis=2);valid=(projected[1:]>0)&(projected[:-1]>0)
gp=projected[1:][valid]/projected[:-1][valid];ge=euclidean[1:]/euclidean[:-1]
normal_heights=[]
for k in range(layers):
    current=lp[k,f];cross=np.cross(current[:,1]-current[:,0],current[:,2]-current[:,0]);nn=cross/np.linalg.norm(cross,axis=1)[:,None]
    normal_heights.append(np.einsum('fij,fj->f',increment[k,f],nn)/3)
normal_heights=np.array(normal_heights);gn=normal_heights[1:]/normal_heights[:-1]
assert max(gp.max(),ge.max())<=1.25
normal_allowance=np.full_like(gn,1.25)
if args.preserve_existing_normal_growth:
    old_layers=len(source['cells_6'])//len(f)
    assert old_layers==layers
    old_lp=source['points'][:(layers+1)*nv].reshape(layers+1,nv,3)
    assert np.array_equal(lp[:fixed_layers+1],old_lp[:fixed_layers+1])
    base=lp[0]
    appendage_patch=((base[:,0]>.85)&(base[:,0]<1.4)&(base[:,1]>.30))|((base[:,0]>3.5)&(np.linalg.norm(base[:,1:],axis=1)>.17))
    assert np.array_equal(lp[-1,~appendage_patch],old_lp[-1,~appendage_patch])
    old_h=[]
    for k in range(layers):
        tri=old_lp[k,f];nn=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);nn/=np.linalg.norm(nn,axis=1)[:,None]
        old_h.append(np.einsum('fij,fj->f',(old_lp[k+1]-old_lp[k])[f],nn)/3)
    old_h=np.array(old_h)
    normal_allowance=np.maximum(normal_allowance,old_h[1:]/old_h[:-1]+.0005)
assert (gn<=normal_allowance).all()
normals=np.cross(wall[:,1]-wall[:,0],wall[:,2]-wall[:,0]);normals/=np.linalg.norm(normals,axis=1)[:,None]
total=np.einsum('fij,fj->fi',(lp[-1]-lp[0])[f],normals).mean(axis=1)
first=np.einsum('fij,fj->fi',(lp[1]-lp[0])[f],normals)
old_first=np.einsum('fij,fj->fi',(source['points'][nv:2*nv]-source['points'][:nv])[f],normals)
first_mean_error=float(np.abs(first.mean(axis=1)-old_first.mean(axis=1)).max())
assert first_mean_error<2.1e-9
file=F/(args.cgns_name or f'SUBOFF_30Lx20Lx20L_Q02_BL{layers}.cgns');write_cgns(file,args.library,p,{4:t,6:pr},boundaries)
np.savez(F/'SUBOFF_optimized.validated.npz',points=p,cells_4=t,cells_6=pr,**{'boundary_'+k:v for k,v in boundaries.items()})
np.savez(F/'prism_front.npz',points=lp[0],faces=f,top=lp[-1])
domain_provenance=json.loads((S/'mesh_validation.json').read_text())['domain']
domain={k:domain_provenance[k] for k in ['length_m','bounds_m','dimensions_L','dimensions_m','upstream_clearance_L','downstream_clearance_L'] if k in domain_provenance}
summary=dict(cells=len(t)+len(pr),tetrahedra=len(t),prisms=len(pr),nodes=len(p),layers=layers,
             minimum_jacobian=jmin,minimum_volume_m3=minimum_volume,volume_m3=float(volume),
             expected_volume_m3=float(expected),relative_volume_closure_error=float(error),
             domain=domain,domain_extension_provenance=domain_provenance,wall_coordinates_unchanged=True,fixed_layers=fixed_layers,optimization_fixed_layers=optimization_fixed_layers,
             first_layer_coordinates_unchanged=bool(np.array_equal(lp[1],source['points'][nv:2*nv])),
             first_layer_mean_normal_height_preserved=True,first_layer_mean_normal_height_max_change_m=first_mean_error,
             first_mean_normal_height_m_quantiles=np.quantile(first.mean(axis=1),[0,.01,.5,.99,1]).tolist(),
             first_layer_normal_height_m_quantiles=np.quantile(first,[0,.01,.5,.99,1]).tolist(),
             total_face_normal_thickness_m_quantiles=np.quantile(total,[0,.01,.5,.99,1]).tolist(),
             projected_increment_growth_quantiles=np.quantile(gp,[0,.001,.01,.5,.99,.999,1]).tolist(),
             normal_increment_growth_quantiles=np.quantile(gn,[0,.001,.01,.5,.99,.999,1]).tolist(),
             euclidean_increment_growth_quantiles=np.quantile(ge,[0,.001,.01,.5,.99,.999,1]).tolist(),
             maximum_projected_growth=float(gp.max()),maximum_euclidean_growth=float(ge.max()),maximum_normal_growth=float(gn.max()),
             normal_growth_existing_reference_preserved=args.preserve_existing_normal_growth,
             normal_growth_maximum_excess_over_allowance=float((gn-normal_allowance).max()),
             initial_direction_projection_note=('Initial-column projected growth and displacement-length growth are capped at 1.25; local face-normal growth is capped per face/layer at max(1.25, source growth + 0.0005).' if args.preserve_existing_normal_growth else 'Initial-column projected, local layer-face normal, and displacement-length growth are capped at 1.25.'),
             nonpositive_initial_direction_increments=int((projected<=0).sum()),
             boundary_faces={k:len(v) for k,v in boundaries.items()},thickening_plan=plan,
             sha256=hashlib.sha256(file.read_bytes()).hexdigest(),actual_flow_yplus_verified=False)
(F/'mesh_validation.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
