#!/usr/bin/env python3
"""Locally expand thin appendage columns; preserve the hull and first 16 layers."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import cg, spsolve
from suboff_arrays_cgns import jacobians

ap = argparse.ArgumentParser()
ap.add_argument('source', type=Path)
ap.add_argument('output', type=Path)
ap.add_argument('--maximum-increase', type=float, default=.8)
ap.add_argument('--target-mm', type=float, default=12.)
ap.add_argument('--arrays', default='SUBOFF_optimized.validated.npz')
ap.add_argument('--remesh-below',type=float,default=.5,help='Regenerate the core when retaining it would require stronger global cutback')
args = ap.parse_args()
S, D = args.source, args.output
D.mkdir(parents=True, exist_ok=True)
a = np.load(S/'SUBOFF_optimized.validated.npz')
p, t, pr = np.load(S/args.arrays)['points'], a['cells_4'], a['cells_6']
f = np.load(S/'prism_front.npz')['faces']
nv = int(f.max())+1
nl = len(pr)//len(f)
lp = p[:(nl+1)*nv].reshape(nl+1, nv, 3)
q = lp[0, f]
cross = np.cross(q[:, 1]-q[:, 0], q[:, 2]-q[:, 0])
area = np.linalg.norm(cross, axis=1)/2
n = cross/(2*area[:, None])
depth = np.einsum('ij,ij->i', (lp[-1, f]-q).mean(axis=1), n)
mass = np.zeros(nv)
total = np.zeros(nv)
for j in range(3):
    np.add.at(mass, f[:, j], area/3)
    np.add.at(total, f[:, j], area*depth/3)
h = total/mass
b = lp[0]
patch = ((b[:, 0]>.85)&(b[:, 0]<1.4)&(b[:, 1]>.30)) | ((b[:, 0]>3.5)&(np.linalg.norm(b[:, 1:],axis=1)>.17))
edges = np.unique(np.sort(np.vstack([f[:, [0,1]], f[:, [1,2]], f[:, [2,0]]]), axis=1),axis=0)
length = np.linalg.norm(b[edges[:,0]]-b[edges[:,1]],axis=1)
adj = coo_matrix((np.ones(2*len(edges)), (np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])),shape=(nv,nv)).tocsr()
L = diags(np.asarray(adj.sum(axis=1)).ravel())-adj
target = np.clip(args.target_mm*.001/h-1,0,args.maximum_increase)
target[~patch] = 0
# Smooth the relative expansion inside the appendage patch, with zero data
# on its boundary. The hull's external layer envelope is exactly retained.
ids = np.flatnonzero(patch)
A = diags(mass)+.012**2*L
relative = np.zeros(nv)
relative[ids] = spsolve(A[ids][:,ids],mass[ids]*target[ids])
relative = np.clip(relative,0,args.maximum_increase)
fd=lp[1]-lp[0];fd/=np.linalg.norm(fd,axis=1)[:,None]
prefix=np.einsum('ij,ij->i',lp[16]-lp[0],fd)
previous=np.einsum('ij,ij->i',lp[16]-lp[15],fd)
current=np.einsum('ij,ij->i',lp[-1]-lp[0],fd)
capacity=prefix+previous*sum(1.244**k for k in range(1,nl-16+1))
relative=np.minimum(relative,np.maximum(.98*capacity/current-1,0))
candidate = p.copy()
cl = candidate[:(nl+1)*nv].reshape(nl+1,nv,3)
direction = lp[1]-lp[0]
direction /= np.linalg.norm(direction,axis=1)[:,None]
freeze = 16
top_delta = (lp[-1]-lp[0])*relative[:,None]
projected = np.einsum('kij,ij->ki',lp-lp[freeze],direction)
for k in range(freeze+1,nl+1):
    fraction = projected[k]/projected[-1]
    cl[k] += fraction[:,None]**2*top_delta
# Back off only columns incident to a folding prism. A global cutback
# otherwise lets one tight corner suppress useful expansion of every fin.
layer_delta = cl-lp
local_factor = np.ones(nv)
for attempt in range(30):
    cl[:] = lp+layer_delta*local_factor[None,:,None]
    bad_faces = np.zeros(len(f),dtype=bool)
    minimum = float('inf')
    flat = cl.reshape(-1,3)
    for start in range(freeze*len(f),len(pr),100000):
        j = jacobians(flat,pr[start:start+100000])
        minimum = min(minimum,float(j.min()))
        bad = np.flatnonzero((j<=0).any(axis=1))
        bad_faces[(start+bad)%len(f)] = True
    print('LOCAL_PRISM_BACKOFF',attempt,'BAD_FACES',int(bad_faces.sum()),'MIN_JAC',minimum,flush=True)
    if not bad_faces.any():break
    local_factor[np.unique(f[bad_faces])] *= .6
    for _ in range(3):
        old_factor=local_factor.copy()
        np.minimum.at(local_factor,edges[:,0],old_factor[edges[:,1]]+.1)
        np.minimum.at(local_factor,edges[:,1],old_factor[edges[:,0]]+.1)
else:raise RuntimeError('Local prism backoff did not produce positive cells')
# Harmonic movement of core interior, with the outer domain fixed.
used, inv = np.unique(t,return_inverse=True)
tt = inv.reshape(-1,4)
ee = np.unique(np.sort(np.vstack([tt[:,[i,j]] for i in range(4) for j in range(i)]),axis=1),axis=0)
dist = np.linalg.norm(p[used[ee[:,0]]]-p[used[ee[:,1]]],axis=1)
w = 1/np.maximum(dist,1e-10)
adj = coo_matrix((np.r_[w,w],(np.r_[ee[:,0],ee[:,1]],np.r_[ee[:,1],ee[:,0]])),shape=(len(used),len(used))).tocsr()
L = diags(np.asarray(adj.sum(axis=1)).ravel())-adj
remap = np.full(len(p),-1,dtype=np.int64)
remap[used] = np.arange(len(used))
outside = np.unique(np.r_[a['boundary_INLET'].ravel(),a['boundary_OUTLET'].ravel(),a['boundary_FAR'].ravel()])
fixed = np.unique(remap[np.r_[nl*nv+np.arange(nv),outside]])
free = np.setdiff1d(np.arange(len(used)),fixed)
delta = candidate[used]-p[used]
B = L[free][:,free].tocsr()
rhs = -(L[free][:,fixed]@delta[fixed])
for axis in range(3):
    sol, info = cg(B,rhs[:,axis],M=diags(1/B.diagonal()),rtol=1e-9,atol=1e-13,maxiter=3000)
    if info: raise RuntimeError(f'Core harmonic solve failed: {info}')
    delta[free,axis] = sol
candidate[used] = p[used]+delta
change = candidate-p
for alpha in (1.,.75,.5,.25,.125,.0625):
    test = p+alpha*change
    minima = {}
    for name,cells in [('tet',t),('prism',pr)]:
        minima[name] = min(float(jacobians(test,cells[i:i+100000]).min()) for i in range(0,len(cells),100000))
    print('DEFORMATION',alpha,'JACOBIANS',minima,flush=True)
    if min(minima.values())>0: break
else: raise RuntimeError('No positive candidate')
if alpha<args.remesh_below:
    # The prism stack is already positive after local backoff. Regenerate
    # the core rather than let an old interior tetrahedron limit all fins.
    np.savez(D/'optimized_layers.npz',points=cl.copy(),faces=f)
    np.savez(D/'prism_front.npz',points=b,faces=f,top=cl[-1])
    plan=dict(source=str(S),layers=nl,fixed_layers=freeze,method='local appendage expansion; regenerated tetrahedral core',target_thickness_m=args.target_mm*.001,maximum_requested_relative_increase=args.maximum_increase,accepted_deformation_fraction=1.,harmonic_core_cutback_rejected=alpha,patch_nodes=int(patch.sum()),first_coordinates_unchanged=bool(np.array_equal(cl[:2],lp[:2])),body_envelope_unchanged=bool(np.array_equal(cl[-1,~patch],lp[-1,~patch])))
    (D/'thickening_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    np.savez(D/'local_targets.npz',patch=patch,relative_expansion=relative,local_factor=local_factor,source_top=lp[-1],target_top=lp[-1]+top_delta)
    subprocess.run([sys.executable,str(Path(__file__).with_name('suboff_thick_core.py')),str(S),str(D),'--switches','pq1.3/20YQO8/7S1000000'],check=True)
    core_boundary=np.load(D/'fixed_boundary.npz')
    fixed_nodes=np.r_[np.arange((freeze+1)*nv),(np.arange(freeze+1,nl+1)[:,None]*nv+np.flatnonzero(~patch)[None,:]).ravel()]
    np.savez(D/'fixed_boundary.npz',nodes=core_boundary['nodes'],wall_nodes=fixed_nodes)
    print('REGENERATED_CORE_PLAN',json.dumps(plan),flush=True)
    sys.exit(0)
np.savez(D/'SUBOFF_optimized.validated.npz',points=test,cells_4=t,cells_6=pr,**{k:a[k] for k in a.files if k.startswith('boundary_')})
np.savez(D/'prism_front.npz',points=b,faces=f,top=test[nl*nv:(nl+1)*nv])
np.savez(D/'boundary_arrays.npz',**{k.removeprefix('boundary_'):a[k] for k in a.files if k.startswith('boundary_')})
fixed_nodes = np.r_[np.arange((freeze+1)*nv), (np.arange(freeze+1,nl+1)[:,None]*nv+np.flatnonzero(~patch)[None,:]).ravel()]
np.savez(D/'fixed_boundary.npz',nodes=outside,wall_nodes=fixed_nodes)
np.savez(D/'local_targets.npz',patch=patch,relative_expansion=relative,local_factor=local_factor,source_top=lp[-1],target_top=lp[-1]+top_delta)
plan = dict(source=str(S),layers=nl,fixed_layers=freeze,method='local appendage expansion with harmonic core displacement',target_thickness_m=args.target_mm*.001,maximum_requested_relative_increase=args.maximum_increase,accepted_deformation_fraction=alpha,patch_nodes=int(patch.sum()),minimum_jacobian=minima,first_coordinates_unchanged=bool(np.array_equal(test[:2*nv],p[:2*nv])),body_envelope_unchanged=bool(np.array_equal(test[nl*nv:(nl+1)*nv][~patch],lp[-1,~patch])))
(D/'thickening_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
print(json.dumps(plan),flush=True)
