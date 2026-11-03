#!/usr/bin/env python3
"""Advance certified prism layers from an unchanged existing first layer."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
from scipy.sparse import coo_matrix
from scipy.optimize import minimize
from suboff_arrays_cgns import jacobians
from suboff_optimize_layers import centroids

ap=argparse.ArgumentParser();ap.add_argument('source',type=Path);ap.add_argument('field',type=Path);ap.add_argument('output',type=Path)
ap.add_argument('--layers',type=int,default=64);ap.add_argument('--depth',type=float,default=.03)
ap.add_argument('--iterations',type=int,default=350)
ap.add_argument('--jacobian-floor',type=float,default=.005)
ap.add_argument('--stop-layer',type=int)
ap.add_argument('--resume',action='store_true');args=ap.parse_args();S=args.source;D=args.output;D.mkdir(exist_ok=True)
sys.path.insert(0,'/tmp/suboff_quality_cuda');import jax
jax.config.update('jax_enable_x64',True);import jax.numpy as jnp
sys.path.insert(0,'/tmp/suboff_regen_deps');import pymeshlab
source=np.load(S/'SUBOFF_optimized.validated.npz');f=np.load(S/'prism_front.npz')['faces'];nv=int(f.max())+1;nf=len(f)
base=source['points'][:nv];first=source['points'][nv:2*nv];layers=[base.copy(),first.copy()]
edges=np.vstack([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]);own=np.tile(np.arange(nf),3)
key=np.sort(edges,axis=1);order=np.lexsort((key[:,1],key[:,0]));ia,ib=order[::2],order[1::2]
edge=edges[ia];owner=own[ia];neighbor=own[ib];unique=key[ia]
cell_ids=np.c_[f,f+nv]
tri_ids=np.vstack([cell_ids[:,[0,2,1]],cell_ids[:,[3,4,5]]]);tri_owner=np.tile(np.arange(nf),2)
quad_ids=np.vstack([cell_ids[:,ff] for ff in ((0,1,4,3),(1,2,5,4),(2,0,3,5))])
quad_ids=np.take_along_axis(quad_ids,(quad_ids.argmin(axis=1)[:,None]+np.arange(4))%4,axis=1)
quad_owner=np.tile(np.arange(nf),3)
side_ids=np.c_[edge[:,0],edge[:,1],edge[:,1]+nv,edge[:,0]+nv]
fj=jnp.asarray(f);tj=jnp.asarray(tri_ids);qj=jnp.asarray(quad_ids);sj=jnp.asarray(side_ids)
to=jnp.asarray(tri_owner);qo=jnp.asarray(quad_owner);oi=jnp.asarray(owner);ni=jnp.asarray(neighbor)

def numpy_normals(points):
    q=points[f];cross=np.cross(q[:,1]-q[:,0],q[:,2]-q[:,0]);area=np.linalg.norm(cross,axis=1)
    return q,cross/area[:,None],area

_,walln,_=numpy_normals(base)
_,initial_front_normal,_=numpy_normals(first)
reference_angle=np.arccos(np.clip(np.einsum('ij,ij->i',initial_front_normal[owner],initial_front_normal[neighbor]),-1,1))
angle_floor=jnp.asarray(np.cos(np.minimum(reference_angle+np.deg2rad(2.),np.pi-.001)))
wall_normal_j=jnp.asarray(walln)
prev_height=np.einsum('fij,fj->f',(first-base)[f],walln)/3
previous_center,_=centroids(np.concatenate([base[f],first[f]],axis=1))
oldq=np.load(S/'orthogonal_quality_values.npy');nt=len(source['cells_4']);qualities=[oldq[nt:nt+nf].copy()]
field=np.load(args.field/'prism_front.npz')['top']-base
q,n,_=numpy_normals(base);rhs=np.zeros_like(base);ws=np.zeros(nv)
for i in range(3):
    u=q[:,(i+1)%3]-q[:,i];v=q[:,(i+2)%3]-q[:,i]
    w=np.arctan2(np.linalg.norm(np.cross(u,v),axis=1),np.einsum('ij,ij->i',u,v))
    np.add.at(rhs,f[:,i],n*w[:,None]);np.add.at(ws,f[:,i],w)
field/=((rhs/ws[:,None]*field).sum(axis=1))[:,None]
reports=[];cumulative=float(np.average(prev_height,weights=np.linalg.norm(np.cross(q[:,1]-q[:,0],q[:,2]-q[:,0]),axis=1)))
previous_nominal=cumulative;started=time.time()
if args.resume:
    checkpoint=np.load(D/'march_checkpoint.npz');layers=list(checkpoint['points'])
    reports=json.loads((D/'march_progress.json').read_text())
    assert len(reports)==len(layers)-2
    current=layers[-1];previous=layers[-2];_,previous_n,_=numpy_normals(previous)
    prev_height=np.einsum('fij,fj->f',(current-previous)[f],previous_n)/3
    previous_center,_=centroids(np.concatenate([previous[f],current[f]],axis=1))
    cumulative=reports[-1]['cumulative_normal_depth_m'];previous_nominal=reports[-1]['mean_normal_height_m']
    field=(current-previous)/previous_nominal
    print('RESUME_AFTER_LAYER',len(layers)-1,flush=True)

def dot(a,b):return jnp.sum(a*b,axis=-1)
def norm(x):return jnp.sqrt(dot(x,x)+1e-60)
def facegeom(x):
    n=jnp.cross(x[:,1]-x[:,0],x[:,2]-x[:,0]);fc=jnp.mean(x[:,:3],axis=1)
    if x.shape[1]==4:
        nn=jnp.cross(x[:,2]-x[:,0],x[:,3]-x[:,0]);aa=norm(n);bb=norm(nn)
        fc=(fc*aa[:,None]+jnp.mean(x[:,jnp.array([0,2,3])],axis=1)*bb[:,None])/(aa+bb)[:,None];n=n+nn
    return fc,n
def cos(n,d):return dot(n,d)/(norm(n)*norm(d))

def objective(x,current,previous_centers,prediction,vertex_normal,ns,ts,area2,height,previous_heights,previous_lengths):
    x=x.reshape(nv,3);normal_part=dot(x,vertex_normal)[:,None]*vertex_normal
    top=prediction+ns*normal_part+ts[:,None]*(x-normal_part)
    pp=jnp.concatenate([current,top]);qp=jnp.concatenate([current[fj],top[fj]],axis=1)
    volume=jnp.zeros(nf);moment=jnp.zeros((nf,3))
    for z in (.5-1/(2*np.sqrt(3)),.5+1/(2*np.sqrt(3))):
        dr=(1-z)*(qp[:,1]-qp[:,0])+z*(qp[:,4]-qp[:,3]);ds=(1-z)*(qp[:,2]-qp[:,0])+z*(qp[:,5]-qp[:,3]);cross=jnp.cross(dr,ds)
        for r,s in ((1/6,1/6),(2/3,1/6),(1/6,2/3)):
            bot=(1-r-s)*qp[:,0]+r*qp[:,1]+s*qp[:,2];end=(1-r-s)*qp[:,3]+r*qp[:,4]+s*qp[:,5]
            w=dot(cross,end-bot)/12;volume=volume+w;moment=moment+w[:,None]*((1-z)*bot+z*end)
    centers=moment/volume[:,None]
    fc,n=facegeom(pp[tj]);v1=cos(n,fc-centers[to])
    fc,n=facegeom(pp[qj]);v2=cos(n,fc-centers[qo])
    _,n=facegeom(pp[sj]);v3=cos(n,centers[ni]-centers[oi])
    bottom_normal=jnp.cross(current[fj[:,1]]-current[fj[:,0]],current[fj[:,2]]-current[fj[:,0]])
    v4=cos(bottom_normal,centers-previous_centers)
    values=jnp.concatenate([v1,v2,v3,v4]);deficit=jnp.maximum(.201-values,0.)
    loss=jnp.sum(jnp.maximum(.215-values,0.)**2)+1000000*jnp.sum(deficit**2)+100000*jnp.sum(deficit**4)
    jj=[]
    for z in (0.,.5,1.):
        dr=(1-z)*(qp[:,1]-qp[:,0])+z*(qp[:,4]-qp[:,3]);ds=(1-z)*(qp[:,2]-qp[:,0])+z*(qp[:,5]-qp[:,3]);cross=jnp.cross(dr,ds)
        for k in range(3):jj.append(dot(cross,qp[:,k+3]-qp[:,k]))
    j0,jm,j1=jnp.array(jj[:3]),jnp.array(jj[3:6]),jnp.array(jj[6:9]);aa=2*(j1+j0-2*jm);bb=j1-j0-aa
    zz=-bb/(2*jnp.where(aa>1e-30,aa,1.));critical=jnp.where((aa>0)&(zz>0)&(zz<1),aa*zz*zz+bb*zz+j0,jnp.minimum(j0,j1))
    ratio=jnp.concatenate([jnp.array(jj),critical],axis=0)/(area2*height)[None,:]
    loss+=1000000*jnp.sum(jnp.maximum(args.jacobian_floor-ratio,0.)**2)
    local_height=jnp.mean(dot(top[fj]-current[fj],bottom_normal[:,None,:]/area2[:,None,None]),axis=1)
    relative=local_height/height
    loss+=1000*jnp.sum(jnp.maximum(jnp.abs(relative-1)-.2,0.)**2)
    physical_growth=local_height/previous_heights
    lengths=norm(top-current);euclidean_growth=lengths/previous_lengths
    loss+=1000000*(jnp.sum(jnp.maximum(physical_growth-1.245,0.)**2)+jnp.sum(jnp.maximum(euclidean_growth-1.245,0.)**2))
    top_normal=jnp.cross(top[fj[:,1]]-top[fj[:,0]],top[fj[:,2]]-top[fj[:,0]])
    alignment=cos(top_normal,bottom_normal)
    edge_alignment=cos(top_normal[oi],top_normal[ni])
    loss+=10000*jnp.sum(jnp.maximum(angle_floor-edge_alignment,0.)**2)
    loss+=1000*jnp.sum(jnp.maximum(.85-cos(top_normal,wall_normal_j),0.)**2)
    loss+=jnp.sum(jnp.maximum(.97-alignment,0.)**2)+1e-5*jnp.sum(x*x)
    stats=(jnp.min(values),jnp.sum(values<.2),jnp.min(ratio),jnp.max(physical_growth),jnp.max(euclidean_growth),jnp.max(jnp.abs(relative-1)))
    return loss,stats

fun=jax.jit(jax.value_and_grad(objective,has_aux=True))
def accepted(stats):return stats[0]>=.200005 and stats[1]==0 and stats[2]>.0001 and max(stats[3:5])<=1.2499 and stats[5]<=.5

last_layer=min(args.layers,args.stop_layer or args.layers)
for layer in range(len(layers),last_layer+1):
    current=layers[-1];previous_step=current-layers[-2];previous_lengths=np.linalg.norm(previous_step,axis=1)
    q,n,area2=numpy_normals(current);matrix=np.zeros((nv,3,3));rhs=np.zeros_like(current);ws=np.zeros(nv)
    edge_list=[];edge_weights=[]
    for i in range(3):
        u=q[:,(i+1)%3]-q[:,i];v=q[:,(i+2)%3]-q[:,i]
        w=np.arctan2(np.linalg.norm(np.cross(u,v),axis=1),np.einsum('ij,ij->i',u,v))
        np.add.at(matrix,f[:,i],w[:,None,None]*n[:,:,None]*n[:,None,:]);np.add.at(rhs,f[:,i],w[:,None]*n);np.add.at(ws,f[:,i],w)
        edge_list.append(f[:,[(i+1)%3,(i+2)%3]]);edge_weights.append(np.clip(.5*np.einsum('ij,ij->i',u,v)/area2,.02,10.))
    mean=rhs/np.linalg.norm(rhs,axis=1)[:,None]
    safe=previous_step/np.maximum(np.einsum('ij,ij->i',previous_step,rhs/ws[:,None]),1e-20)[:,None]
    d=np.linalg.solve(matrix+.01*ws[:,None,None]*np.eye(3)[None,:,:],(rhs+.01*ws[:,None]*field)[:,:,None])[:,:,0]
    tangent_projector=np.eye(3)[None,:,:]-mean[:,:,None]*mean[:,None,:]
    covariance=np.einsum('nij,njk,nkl->nil',tangent_projector,matrix,tangent_projector)/ws[:,None,None]
    ev,vec=np.linalg.eigh(covariance);P=tangent_projector-np.einsum('nik,nk,njk->nij',vec,(ev>.02).astype(float),vec)
    ee=np.vstack(edge_list);ww=np.concatenate(edge_weights)
    adjacency=coo_matrix((np.r_[ww,ww],(np.r_[ee[:,0],ee[:,1]],np.r_[ee[:,1],ee[:,0]])),shape=(nv,nv)).tocsr();degree=np.asarray(adjacency.sum(axis=1)).ravel()
    constrained=d.copy()
    for _ in range(25):d=constrained+np.einsum('nij,nj->ni',P,adjacency@d/degree[:,None]-constrained)
    d/=np.maximum(np.einsum('ij,ij->i',d,rhs/ws[:,None]),1e-20)[:,None]
    for _ in range(15):
        bad_vertices=np.unique(f[(np.einsum('fij,fj->fi',d[f],n)<=1e-8).any(axis=1)])
        if not len(bad_vertices):break
        d[bad_vertices]=.5*(d[bad_vertices]+safe[bad_vertices])
    remaining=args.layers-layer+1;wanted=args.depth-cumulative
    low=.7;high=1.2
    for _ in range(45):
        r=(low+high)/2;total=previous_nominal*np.sum(r**np.arange(1,remaining+1))
        if total<wanted:low=r
        else:high=r
    nominal=previous_nominal*(low+high)/2
    solved=False;warm_top=None;warm_h=None
    for attempt in range(8):
        h=nominal*.8**attempt
        dn=np.einsum('ij,ij->i',d,mean);dt=d-dn[:,None]*mean
        limit=1.24*previous_lengths/h;dn=np.minimum(dn,.99*limit)
        tlength=np.linalg.norm(dt,axis=1);allowed=np.sqrt(np.maximum(limit*limit-dn*dn,0))
        clipped=dn[:,None]*mean+dt*np.minimum(1,allowed/np.maximum(tlength,1e-20))[:,None]
        pred=current+h*clipped
        elen=np.linalg.norm(current[unique[:,0]]-current[unique[:,1]],axis=1);total_length=np.zeros(nv);counts=np.zeros(nv)
        for j in (0,1):np.add.at(total_length,unique[:,j],elen);np.add.at(counts,unique[:,j],1)
        ts=np.maximum(h,.05*total_length/np.maximum(counts,1))
        data=tuple(jnp.asarray(v) for v in (current,previous_center,pred,mean,np.array(h),ts,area2,np.array(h),prev_height,previous_lengths))
        last=[None];calls=[0]
        def scipyfun(x):
            (loss,stats),grad=fun(x,*data);ss=[float(v) for v in stats];last[0]=ss;calls[0]+=1
            if calls[0]%50==1:print('OPT',layer,attempt,calls[0],float(loss),ss,flush=True)
            return float(loss),np.asarray(grad)
        def callback(x):
            if last[0] is not None and accepted(last[0]):raise StopIteration
        x=np.zeros(3*nv)
        if warm_top is not None:
            delta=current+(warm_top-current)*(h/warm_h)-pred
            normal_delta=np.einsum('ij,ij->i',delta,mean)[:,None]*mean
            x=np.clip(normal_delta/h+(delta-normal_delta)/ts[:,None],-2.,2.).ravel()
        scipyfun(x)
        if not accepted(last[0]):
            result=minimize(scipyfun,x,jac=True,method='L-BFGS-B',bounds=[(-2.,2.)]*len(x),callback=callback,
                            options=dict(maxiter=args.iterations,maxcor=15,maxls=25,ftol=1e-13,gtol=1e-8))
            x=result.x;scipyfun(x)
        xx=x.reshape(nv,3);nn=np.einsum('ij,ij->i',xx,mean)[:,None]*mean
        top=pred+h*nn+ts[:,None]*(xx-nn)
        if not accepted(last[0]):
            warm_top=top;warm_h=h
            print('CUTBACK',layer,attempt,last[0],flush=True);continue
        block=np.concatenate([current[f],top[f]],axis=1);minimum=float(jacobians(block.reshape(-1,3),np.arange(6*nf).reshape(nf,6)).min())
        if minimum<=0:raise ValueError('GPU/CPU Jacobian mismatch')
        check=pymeshlab.MeshSet();check.add_mesh(pymeshlab.Mesh(np.vstack([base,top]),np.vstack([f,f+nv])))
        check.apply_filter('compute_selection_by_self_intersections_per_face');intersections=int(check.current_mesh().selected_face_number())
        if intersections:
            print('CUTBACK_INTERSECTIONS',layer,intersections,flush=True);continue
        solved=True;break
    if not solved:raise ValueError(f'Cannot certify layer {layer}')
    local_height=np.einsum('fij,fj->f',(top-current)[f],n)/3
    measured=float(np.average(local_height,weights=area2));cumulative+=measured;previous_nominal=measured
    previous_center,_=centroids(block);prev_height=local_height;field=(top-current)/h
    layers.append(top)
    reports.append(dict(layer=layer,minimum_quality=last[0][0],minimum_jacobian=minimum,maximum_normal_growth=last[0][3],
                        maximum_euclidean_growth=last[0][4],mean_normal_height_m=measured,cumulative_normal_depth_m=cumulative,
                        cutback=attempt,self_intersections=intersections))
    print('ACCEPT',json.dumps(reports[-1]),'SECONDS',time.time()-started,flush=True)
    if True:
        np.savez(D/'march_checkpoint.npz',points=np.array(layers),faces=f)
        (D/'march_progress.json').write_text(json.dumps(reports,indent=2)+'\n')
lp=np.array(layers);np.savez(D/'optimized_layers.npz',points=lp,faces=f)
np.savez(D/'prism_front.npz',points=base,faces=f,top=lp[-1])
plan=dict(source=str(S),field_source=str(args.field),layers=len(lp)-1,old_layers=40,fixed_layers=1,
          method='quality-controlled advancing layers',first_coordinates_unchanged=True,target_thickness_m=cumulative,
          initial_scheduling_target_m=args.depth,initial_scheduling_layers=args.layers,
          cumulative_mean_local_normal_depth_m=cumulative,minimum_step_quality=min(r['minimum_quality'] for r in reports),
          maximum_local_normal_growth=max(r['maximum_normal_growth'] for r in reports),maximum_euclidean_growth=max(r['maximum_euclidean_growth'] for r in reports))
(D/'thickening_plan.json').write_text(json.dumps(plan,indent=2)+'\n');print('COMPLETE',json.dumps(plan),flush=True)
