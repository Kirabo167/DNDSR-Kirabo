#!/usr/bin/env python3
"""Refine only independent interface tetrahedra whose children pass exact Q."""
import argparse
import json
from pathlib import Path
import numpy as np
from suboff_arrays_cgns import orient_validate
from suboff_optimize_layers import centroids

ap=argparse.ArgumentParser();ap.add_argument('source',type=Path);ap.add_argument('output',type=Path);ap.add_argument('--ratio',type=float,default=4.);ap.add_argument('--allow-adjacent',action='store_true');args=ap.parse_args();S,D=args.source,args.output;D.mkdir(exist_ok=False)
raw=np.load(S/'SUBOFF_optimized.validated.npz');a=np.load(S/'optimized_coordinates.npz');p=a['points'];t=a['cells_4'];pr=a['cells_6'];f=np.load(S/'prism_front.npz')['faces'];nv=int(f.max())+1;nl=len(pr)//len(f);nt=len(t);nf=len(f)
templates=((0,2,1),(0,1,3),(1,2,3),(2,0,3))
faces=np.vstack([t[:,ids] for ids in templates]+[f+nl*nv]);owners=np.r_[np.tile(np.arange(nt),4),nt+np.arange(nf)]
key=np.sort(faces,axis=1);order=np.lexsort((key[:,2],key[:,1],key[:,0]));same=(key[order[1:]]==key[order[:-1]]).all(axis=1);rows=np.flatnonzero(same);left=order[rows];right=order[rows+1]
neighbors=np.full(len(faces),-1,dtype=np.int64);neighbors[left]=owners[right];neighbors[right]=owners[left]
assert np.all(neighbors[-nf:]>=0)
adjacent=neighbors[-nf:];top=p[f+nl*nv];prev=p[f+(nl-1)*nv];fc=top.mean(axis=1);n=np.cross(top[:,1]-top[:,0],top[:,2]-top[:,0]);n/=np.linalg.norm(n,axis=1)[:,None]
opposite=p[t[adjacent]].sum(axis=1)-top.sum(axis=1);height=np.einsum('ij,ij->i',opposite-fc,n);last=np.einsum('ij,ij->i',(top-prev).mean(axis=1),n);ratio=height/last
wall=p[f].mean(axis=1);patch=((wall[:,0]>.85)&(wall[:,0]<1.4)&(wall[:,1]>.30))|((wall[:,0]>3.5)&(np.linalg.norm(wall[:,1:],axis=1)>.17))
candidates=np.unique(adjacent[patch&(ratio>args.ratio)&(height>.002)])
centers=np.vstack([p[t].mean(axis=1),centroids(p[pr[-nf:]])[0]])
score=np.full(len(candidates),np.inf)
q=p[t[candidates]];inside=q.mean(axis=1)
child_centers=[];child_points=[]
def cosine(nn,dd):return np.einsum('ij,ij->i',nn,dd)/(np.linalg.norm(nn,axis=1)*np.linalg.norm(dd,axis=1))
for i,face in enumerate(templates):
    # Reverse the parent face so the child has positive orientation.
    cq=np.concatenate([q[:,[face[1],face[0],face[2]]],inside[:,None,:]],axis=1);child_points.append(cq);cc=cq.mean(axis=1);child_centers.append(cc)
    for ids in templates:
        tri=cq[:,ids];nn=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);score=np.minimum(score,cosine(nn,tri.mean(axis=1)-cc))
    neighbor=neighbors[i*nt+candidates];exists=neighbor>=0;tri=q[:,face];nn=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);values=np.ones(len(candidates));values[exists]=cosine(nn[exists],centers[neighbor[exists]]-cc[exists]);score=np.minimum(score,values)
for i in range(4):
    for j in range(i):
        shared=sorted(set(templates[i])&set(templates[j]));tri=np.stack([q[:,shared[0]],q[:,shared[1]],inside],axis=1);nn=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);score=np.minimum(score,np.abs(cosine(nn,child_centers[i]-child_centers[j])))
safe=candidates[score>=.201];safe_score=score[score>=.201]
# Never refine face-neighboring parents in the same pass: each external
# child-face certificate above used the original neighbor centroid.
chosen=[];blocked=np.zeros(nt+nf,dtype=bool)
for cell in safe[np.argsort(-safe_score)]:
    if blocked[cell]:continue
    chosen.append(cell);blocked[cell]=True
    nn=neighbors[np.arange(4)*nt+cell];blocked[nn[nn>=0]]=True
chosen=np.array(chosen,dtype=np.int64);assert len(chosen)>0
coupled_minimum=None
if args.allow_adjacent:
    chosen=safe
    selected_parent=np.zeros(nt+nf,dtype=bool);selected_parent[chosen]=True
    both=selected_parent[owners[left]]&selected_parent[owners[right]]
    tri=p[faces[left[both]]];fc=tri.mean(axis=1);nn=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
    # Across a doubly refined face, each child center is .75*face_center
    # + .25*parent_center. Their difference is .25 of the original, so
    # the original normal cosine is retained. Check it explicitly too.
    ca=.75*fc+.25*centers[owners[left[both]]];cb=.75*fc+.25*centers[owners[right[both]]]
    coupled_minimum=float(np.abs(cosine(nn,cb-ca)).min()) if both.any() else None
    assert coupled_minimum is None or coupled_minimum>=.2
keep=np.ones(nt,dtype=bool);keep[chosen]=False;new_ids=len(p)+np.arange(len(chosen));points=np.vstack([p,p[t[chosen]].mean(axis=1)])
tet=np.vstack([t[keep],*[np.c_[t[chosen][:,[face[1],face[0],face[2]]],new_ids] for face in templates]])
minimum=orient_validate(points,tet)
np.savez(D/'SUBOFF_optimized.validated.npz',points=points,cells_4=tet,cells_6=pr,**{k:raw[k] for k in raw.files if k.startswith('boundary_')})
np.savez(D/'optimized_coordinates.npz',points=points,cells_4=tet,cells_6=pr)
for name in ['prism_front.npz','boundary_arrays.npz','local_targets.npz']:
    (D/name).write_bytes((S/name).read_bytes())
fb=np.load(S/'fixed_boundary.npz');np.savez(D/'fixed_boundary.npz',nodes=fb['nodes'],wall_nodes=np.arange((nl+1)*nv))
plan=json.loads((S/'thickening_plan.json').read_text());plan['incremental_source']=str(S);plan['source']='data/mesh/SUBOFF_BL_smooth_Q02_20261102/final';plan['refinement_method']='precertified tetrahedral centroid subdivision; adjacent refined faces preserve original center-line direction' if args.allow_adjacent else 'precertified nonadjacent tetrahedral centroid subdivision'
(D/'thickening_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
report=dict(source=str(S),method=plan['refinement_method'],candidate_parents=len(candidates),individually_safe_parents=len(safe),selected_tetrahedra=len(chosen),added_tetrahedra=3*len(chosen),cells=len(tet)+len(pr),minimum_certified_child_face_quality=float(score[np.isin(candidates,chosen)].min()),minimum_tetra_jacobian=minimum,all_prism_coordinates_unchanged=True,interface_ratio_threshold=args.ratio)
report['minimum_double_refined_interface_quality']=coupled_minimum
(D/'core_generation.json').write_text(json.dumps(report,indent=2)+'\n');np.savez(D/'refinement_provenance.npz',source_tetrahedra=chosen,source_candidates=candidates,candidate_minimum_quality=score,new_nodes=new_ids)
print(json.dumps(report),flush=True)
