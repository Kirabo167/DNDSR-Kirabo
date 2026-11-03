#!/usr/bin/env python3
"""Optimize core nodes and tapered outer prism layers with exact face cosines.

The wall and first twenty layers are fixed. Automatic differentiation supplies
derivatives of the same volume-centroid metric used by the quality report.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
from scipy.optimize import minimize
from suboff_arrays_cgns import orient_validate, jacobians
from suboff_optimize_layers import centroids, geometry


def main():
    ap=argparse.ArgumentParser();ap.add_argument('directory',type=Path)
    ap.add_argument('--iterations',type=int,default=300)
    ap.add_argument('--target',type=float,default=.14)
    ap.add_argument('--select',type=float,default=.16)
    ap.add_argument('--start-layer',type=int,default=20)
    ap.add_argument('--resume',action='store_true')
    args=ap.parse_args();D=args.directory
    sys.path.insert(0,'/tmp/suboff_quality_autodiff')
    import jax
    jax.config.update('jax_enable_x64',True)
    import jax.numpy as jnp
    raw=np.load(D/'SUBOFF_optimized.validated.npz');p=raw['points'];t=raw['cells_4'].copy();pr=raw['cells_6'].copy()
    orient_validate(p,t);orient_validate(p,pr)
    f=np.load(D/'prism_front.npz')['faces'];nf=len(f);nv=int(pr.max()+1)//41;nt=len(t);layers=40
    quality=np.load(D/'orthogonal_quality_values.npy')
    active=np.unique(t[quality[:nt]<args.select]);active=active[active>=40*nv]
    # External box nodes are fixed.
    core_info=np.load(D/'core.npz')
    nouter=len(core_info['core_input'])-nv
    active=active[(active<41*nv)|(active>=41*nv+nouter)]
    print('ACTIVE_PARAMETERS',len(active),'TOP',int((active<41*nv).sum()),flush=True)
    param=np.full(len(p),len(active),dtype=np.int32);weight=np.zeros(len(p));param[active]=np.arange(len(active));weight[active]=1.
    top=active[active<41*nv];top_param=param[top];base=top-40*nv
    for k in range(args.start_layer+1,40):
        ids=base+k*nv;param[ids]=top_param;weight[ids]=(k-args.start_layer)/(40-args.start_layer)
    affected_t=(param[t]<len(active)).any(axis=1);affected_pr=(param[pr]<len(active)).any(axis=1)
    affected=np.r_[affected_t,affected_pr]
    # Shared triangular interfaces (tet/tet and tet/prism).
    tf=((0,2,1),(0,1,3),(1,2,3),(2,0,3));pf=((0,2,1),(3,4,5),(0,1,4,3),(1,2,5,4),(2,0,3,5))
    ff=np.vstack([t[:,x] for x in tf]+[pr[-nf:,3:6]])
    own=np.r_[np.tile(np.arange(nt),4),nt+39*nf+np.arange(nf)]
    key=np.sort(ff,axis=1);order=np.lexsort((key[:,2],key[:,1],key[:,0]));same=np.flatnonzero((key[order[1:]]==key[order[:-1]]).all(axis=1))
    a,b=order[same],order[same+1];select=affected[own[a]]|affected[own[b]];a,b=a[select],b[select]
    tri_ids=[ff[a]];tri_owner=[own[a]];tri_neighbor=[own[b]]
    for k in range(39):
        o=nt+k*nf+np.arange(nf);n=o+nf;s=affected[o]|affected[n]
        tri_ids.append(pr[k*nf:(k+1)*nf,3:6][s]);tri_owner.append(o[s]);tri_neighbor.append(n[s])
    tri_ids=np.vstack(tri_ids);tri_owner=np.concatenate(tri_owner);tri_neighbor=np.concatenate(tri_neighbor)
    edges=np.vstack([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]);eo=np.tile(np.arange(nf),3);key=np.sort(edges,axis=1)
    order=np.lexsort((key[:,1],key[:,0]));a,b=order[::2],order[1::2];edge=edges[a]
    quad_ids=[];quad_owner=[];quad_neighbor=[]
    for k in range(40):
        o=nt+k*nf+eo[a];n=nt+k*nf+eo[b];s=affected[o]|affected[n]
        e=edge[s];quad_ids.append(np.c_[e[:,0]+k*nv,e[:,1]+k*nv,e[:,1]+(k+1)*nv,e[:,0]+(k+1)*nv])
        quad_owner.append(o[s]);quad_neighbor.append(n[s])
    quad_ids=np.vstack(quad_ids);quad_owner=np.concatenate(quad_owner);quad_neighbor=np.concatenate(quad_neighbor)
    ct=np.flatnonzero(affected_t);cp=np.flatnonzero(affected_pr)
    needed=np.unique(np.r_[ct,nt+cp,tri_owner,tri_neighbor,quad_owner,quad_neighbor])
    used_t=needed[needed<nt];used_p=needed[needed>=nt]-nt
    cellmap=np.full(nt+len(pr),-1,dtype=np.int32);cellmap[used_t]=np.arange(len(used_t));cellmap[nt+used_p]=len(used_t)+np.arange(len(used_p))
    usedpoints=np.unique(np.r_[t[used_t].ravel(),pr[used_p].ravel()]);pointmap=np.full(len(p),-1,dtype=np.int32);pointmap[usedpoints]=np.arange(len(usedpoints))
    local_t=pointmap[t[used_t]];local_p=pointmap[pr[used_p]]
    cftri=np.vstack([pointmap[t[ct][:,x]] for x in tf]+[pointmap[pr[cp][:,x]] for x in pf[:2]])
    cftowner=np.r_[np.tile(cellmap[ct],4),np.tile(cellmap[nt+cp],2)]
    cfquad=np.vstack([pointmap[pr[cp][:,x]] for x in pf[2:]])
    cfqowner=np.tile(cellmap[nt+cp],3)
    def reorder(ids):return np.take_along_axis(ids,(usedpoints[ids].argmin(axis=1)[:,None]+np.arange(4))%4,axis=1)
    cfquad=reorder(cfquad);quad_ids=reorder(pointmap[quad_ids]);tri_ids=pointmap[tri_ids]
    tri_owner=cellmap[tri_owner];tri_neighbor=cellmap[tri_neighbor];quad_owner=cellmap[quad_owner];quad_neighbor=cellmap[quad_neighbor]
    # Characteristic edge length preconditions the coordinate variables.
    scale=np.full(len(active),np.inf)
    for i in range(4):
        for j in range(i):
            length=np.linalg.norm(p[t[:,i]]-p[t[:,j]],axis=1)
            for k in (i,j):np.minimum.at(scale,param[t[:,k]][param[t[:,k]]<len(active)],length[param[t[:,k]]<len(active)])
    scale=np.where(active<41*nv,np.minimum(scale,.03),scale)
    scale=np.maximum(scale,1e-5)
    local_scale=np.r_[scale,0.][param[usedpoints]]*weight[usedpoints]
    p0=jnp.asarray(p[usedpoints]);par=jnp.asarray(param[usedpoints]);ss=jnp.asarray(local_scale)
    local_t=jnp.asarray(local_t);local_p=jnp.asarray(local_p)
    reference_jac=[]
    q0=p[pr[used_p]]
    for z in (0.,.5,1.):
        dr=(1-z)*(q0[:,1]-q0[:,0])+z*(q0[:,4]-q0[:,3]);ds=(1-z)*(q0[:,2]-q0[:,0])+z*(q0[:,5]-q0[:,3])
        for k in range(3):reference_jac.append(np.einsum('ij,ij->i',np.cross(dr,ds),q0[:,k+3]-q0[:,k]))
    reference_jac=jnp.asarray(np.array(reference_jac))
    def dot(a,b):return jnp.sum(a*b,axis=-1)
    def norm(a):return jnp.sqrt(dot(a,a)+1e-60)
    def geom(x):
        n=jnp.cross(x[:,1]-x[:,0],x[:,2]-x[:,0]);fc=jnp.mean(x[:,:3],axis=1)
        if x.shape[1]==4:
            nn=jnp.cross(x[:,2]-x[:,0],x[:,3]-x[:,0]);a=norm(n);b=norm(nn)
            fc=(fc*a[:,None]+jnp.mean(x[:,jnp.array([0,2,3])],axis=1)*b[:,None])/(a+b)[:,None];n=n+nn
        return fc,n
    def cos(n,d):return dot(n,d)/(norm(n)*norm(d))
    def evaluate(x):
        xx=jnp.concatenate([x.reshape((-1,3)),jnp.zeros((1,3))],axis=0)
        pp=p0+xx[par]*ss[:,None]
        qt=pp[local_t];qp=pp[local_p]
        volume=jnp.zeros(len(used_p));moment=jnp.zeros((len(used_p),3))
        for z in (.5-1/(2*np.sqrt(3)),.5+1/(2*np.sqrt(3))):
            dr=(1-z)*(qp[:,1]-qp[:,0])+z*(qp[:,4]-qp[:,3]);ds=(1-z)*(qp[:,2]-qp[:,0])+z*(qp[:,5]-qp[:,3]);cross=jnp.cross(dr,ds)
            for r,s in ((1/6,1/6),(2/3,1/6),(1/6,2/3)):
                bot=(1-r-s)*qp[:,0]+r*qp[:,1]+s*qp[:,2];top=(1-r-s)*qp[:,3]+r*qp[:,4]+s*qp[:,5]
                w=dot(cross,top-bot)/12;volume=volume+w;moment=moment+w[:,None]*((1-z)*bot+z*top)
        centers=jnp.concatenate([jnp.mean(qt,axis=1),moment/volume[:,None]],axis=0)
        fc,n=geom(pp[cftri]);v1=cos(n,fc-centers[cftowner])
        fc,n=geom(pp[cfquad]);v2=cos(n,fc-centers[cfqowner])
        _,n=geom(pp[tri_ids]);v3=cos(n,centers[tri_neighbor]-centers[tri_owner])
        _,n=geom(pp[quad_ids]);v4=cos(n,centers[quad_neighbor]-centers[quad_owner])
        values=jnp.concatenate([v1,v2,v3,v4])
        loss=jnp.sum(jnp.maximum(args.target-values,0.)**2)
        jj=[]
        for z in (0.,.5,1.):
            dr=(1-z)*(qp[:,1]-qp[:,0])+z*(qp[:,4]-qp[:,3]);ds=(1-z)*(qp[:,2]-qp[:,0])+z*(qp[:,5]-qp[:,3]);cross=jnp.cross(dr,ds)
            for k in range(3):jj.append(dot(cross,qp[:,k+3]-qp[:,k]))
        ratio=jnp.array(jj)/reference_jac
        loss=loss+10*jnp.sum(jnp.maximum(.2-ratio,0.)**2)+1e-6*jnp.sum(x*x)
        return loss,(jnp.min(values),jnp.sum(values<.1),jnp.min(ratio))
    fun=jax.jit(jax.value_and_grad(evaluate,has_aux=True))
    print('LOCAL_MESH',len(used_t),len(used_p),len(usedpoints),'FACES',len(cftri)+len(cfquad)+len(tri_ids)+len(quad_ids),flush=True)
    started=time.time();counter=[0];last=[None]
    def scipyfun(x):
        (val,stats),grad=fun(x);counter[0]+=1;last[0]=(x.copy(),[float(v) for v in stats])
        if counter[0]%10==1:print('EVAL',counter[0],'LOSS',float(val),'MIN_BAD_JAC',last[0][1],'SECONDS',time.time()-started,flush=True)
        return float(val),np.asarray(grad)
    x=np.zeros(len(active)*3)
    if args.resume and (D/'optimization_parameters.npz').exists():
        saved=np.load(D/'optimization_parameters.npz')
        assert np.array_equal(saved['active'],active)
        x=(saved['x'].reshape((-1,3))*saved['scale'][:,None]/scale[:,None]).ravel()
    callbacks=[0]
    def save(x):
        np.savez(D/'optimization_parameters.tmp.npz',x=x,active=active,scale=scale)
        (D/'optimization_parameters.tmp.npz').replace(D/'optimization_parameters.npz')
        pp=p.copy();valid=param<len(active);pp[valid]+=x.reshape((-1,3))[param[valid]]*(np.r_[scale,0.][param[valid]]*weight[valid])[:,None]
        np.savez(D/'optimized_coordinates.tmp.npz',points=pp,cells_4=t,cells_6=pr)
        (D/'optimized_coordinates.tmp.npz').replace(D/'optimized_coordinates.npz')
    def callback(x):
        callbacks[0]+=1
        if callbacks[0]%25==0:save(x)
        if last[0] is not None and last[0][1][0]>=.12 and last[0][1][1]==0 and last[0][1][2]>.15:
            save(x)
            raise StopIteration
    result=minimize(scipyfun,x,jac=True,method='L-BFGS-B',bounds=[(-.75,.75)]*len(x),callback=callback,
                    options=dict(maxiter=args.iterations,ftol=1e-14,gtol=1e-7,maxls=30,maxcor=15))
    save(result.x)
    value,stats=evaluate(result.x)
    report=dict(success=bool(result.success),message=str(result.message),iterations=int(result.nit),evaluations=counter[0],
                loss=float(value),minimum=float(stats[0]),bad_faces=int(stats[1]),minimum_jacobian_ratio=float(stats[2]),seconds=time.time()-started,
                wall_fixed=True,first_fixed_layers=args.start_layer,metric_target=args.target,
                quality_goal_achieved=bool(float(stats[0])>=.1 and int(stats[1])==0 and float(stats[2])>0.))
    (D/'coupled_optimization.json').write_text(json.dumps(report,indent=2)+'\n');print('RESULT',json.dumps(report),flush=True)


if __name__=='__main__':main()
