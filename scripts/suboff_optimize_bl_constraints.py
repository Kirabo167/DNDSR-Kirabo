#!/usr/bin/env python3
"""Optimize selected nodes of a hybrid mesh using exact face orthogonality.

Wall, external box, and the prepared prefix of near-wall layers are fixed.
Outer-layer nodes obey both projected-thickness and displacement-length bounds.
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
    ap.add_argument('--maxcor',type=int,default=15)
    ap.add_argument('--adam-steps',type=int,default=0)
    ap.add_argument('--adam-rate',type=float,default=.01)
    ap.add_argument('--target',type=float,default=.215)
    ap.add_argument('--select',type=float,default=.23)
    ap.add_argument('--required-minimum',type=float,default=.2)
    ap.add_argument('--margin',type=float,default=.0005)
    ap.add_argument('--coordinate-bound',type=float,default=2.)
    ap.add_argument('--core-scale-floor',type=float,default=0.)
    ap.add_argument('--tangential-scale-floor',type=float,default=0.)
    ap.add_argument('--goal-weight',type=float,default=1000.)
    ap.add_argument('--growth-cap',type=float,default=1.25)
    ap.add_argument('--growth-target',type=float,default=1.245)
    ap.add_argument('--growth-floor',type=float,default=.90)
    ap.add_argument('--growth-weight',type=float,default=1000.)
    ap.add_argument('--jacobian-weight',type=float,default=10.)
    ap.add_argument('--jacobian-floor',type=float,default=.2)
    ap.add_argument('--freeze-layer',type=int,default=16)
    ap.add_argument('--resume',action='store_true')
    ap.add_argument('--jax-path',default='/tmp/suboff_quality_autodiff')
    ap.add_argument('--normal-growth-reference',type=Path)
    args=ap.parse_args();D=args.directory
    sys.path.insert(0,args.jax_path)
    import jax
    jax.config.update('jax_enable_x64',True)
    import jax.numpy as jnp
    raw=np.load(D/'SUBOFF_optimized.validated.npz');p=raw['points'];t=raw['cells_4'].copy();pr=raw['cells_6'].copy()
    orient_validate(p,t);orient_validate(p,pr)
    f=np.load(D/'prism_front.npz')['faces'];nf=len(f);layers=len(pr)//nf;nv=int(f.max())+1;nt=len(t)
    quality=np.load(D/'orthogonal_quality_values.npy')
    active=np.unique(np.r_[t[quality[:nt]<args.select].ravel(),pr[quality[nt:]<args.select].ravel()])
    column=p[:(layers+1)*nv].reshape(layers+1,nv,3)
    column_direction=column[1]-column[0]
    column_direction/=np.linalg.norm(column_direction,axis=1)[:,None]
    delta=np.diff(column,axis=0)
    projected=np.einsum('kij,ij->ki',delta,column_direction)
    distance=np.linalg.norm(delta,axis=2)
    gp=projected[1:]/projected[:-1];gn=distance[1:]/distance[:-1]
    lower_slice=slice(args.freeze_layer-1,None)
    bad_columns=np.flatnonzero((gp>args.growth_target).any(axis=0)|(gn>args.growth_target).any(axis=0)|
                              (gp[lower_slice]<args.growth_floor).any(axis=0)|
                              (gn[lower_slice]<args.growth_floor).any(axis=0))
    extra=(np.arange(args.freeze_layer+1,layers+1)[:,None]*nv+bad_columns[None,:]).ravel()
    active=np.unique(np.r_[active,extra])
    normal_reference=None
    if args.normal_growth_reference:
        reference_points=np.load(args.normal_growth_reference)['points'][:(layers+1)*nv].reshape(layers+1,nv,3)
        def normal_heights(lp):
            tri=lp[:-1,f];nn=np.cross(tri[:,:,1]-tri[:,:,0],tri[:,:,2]-tri[:,:,0]);nn/=np.linalg.norm(nn,axis=2)[:,:,None]
            return np.einsum('kfij,kfj->kf',np.diff(lp,axis=0)[:,f],nn)/3
        normal_reference=normal_heights(reference_points)
        normal_limit=np.maximum(normal_reference[1:]/normal_reference[:-1]+1e-5,args.growth_target)
        current_heights=normal_heights(column)
        bad_faces=np.flatnonzero((current_heights[1:]/current_heights[:-1]>normal_limit).any(axis=0))
        bad_vertices=np.unique(f[bad_faces])
        active=np.unique(np.r_[active,(np.arange(args.freeze_layer+1,layers+1)[:,None]*nv+bad_vertices[None,:]).ravel()])
    if (D/'additional_active_nodes.npy').exists():
        active=np.unique(np.r_[active,np.load(D/'additional_active_nodes.npy')])
    fixed=np.load(D/'fixed_boundary.npz');active=np.setdiff1d(active,np.r_[fixed['nodes'],fixed['wall_nodes']])
    print('ACTIVE_PARAMETERS',len(active),'PRISM_LAYER_NODES',int((active<(layers+1)*nv).sum()),flush=True)
    param=np.full(len(p),len(active),dtype=np.int32);weight=np.zeros(len(p));param[active]=np.arange(len(active));weight[active]=1.
    affected_t=(param[t]<len(active)).any(axis=1);affected_pr=(param[pr]<len(active)).any(axis=1)
    affected=np.r_[affected_t,affected_pr]
    # Shared triangular interfaces (tet/tet and tet/prism).
    tf=((0,2,1),(0,1,3),(1,2,3),(2,0,3));pf=((0,2,1),(3,4,5),(0,1,4,3),(1,2,5,4),(2,0,3,5))
    ff=np.vstack([t[:,x] for x in tf]+[pr[-nf:,3:6]])
    own=np.r_[np.tile(np.arange(nt),4),nt+(layers-1)*nf+np.arange(nf)]
    key=np.sort(ff,axis=1);order=np.lexsort((key[:,2],key[:,1],key[:,0]));same=np.flatnonzero((key[order[1:]]==key[order[:-1]]).all(axis=1))
    a,b=order[same],order[same+1];select=affected[own[a]]|affected[own[b]];a,b=a[select],b[select]
    tri_ids=[ff[a]];tri_owner=[own[a]];tri_neighbor=[own[b]]
    for k in range(layers-1):
        o=nt+k*nf+np.arange(nf);n=o+nf;s=affected[o]|affected[n]
        tri_ids.append(pr[k*nf:(k+1)*nf,3:6][s]);tri_owner.append(o[s]);tri_neighbor.append(n[s])
    tri_ids=np.vstack(tri_ids);tri_owner=np.concatenate(tri_owner);tri_neighbor=np.concatenate(tri_neighbor)
    edges=np.vstack([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]);eo=np.tile(np.arange(nf),3);key=np.sort(edges,axis=1)
    order=np.lexsort((key[:,1],key[:,0]));a,b=order[::2],order[1::2];edge=edges[a]
    quad_ids=[];quad_owner=[];quad_neighbor=[]
    for k in range(layers):
        o=nt+k*nf+eo[a];n=nt+k*nf+eo[b];s=affected[o]|affected[n]
        e=edge[s];quad_ids.append(np.c_[e[:,0]+k*nv,e[:,1]+k*nv,e[:,1]+(k+1)*nv,e[:,0]+(k+1)*nv])
        quad_owner.append(o[s]);quad_neighbor.append(n[s])
    quad_ids=np.vstack(quad_ids);quad_owner=np.concatenate(quad_owner);quad_neighbor=np.concatenate(quad_neighbor)
    ct=np.flatnonzero(affected_t);cp=np.flatnonzero(affected_pr)
    needed=np.unique(np.r_[ct,nt+cp,tri_owner,tri_neighbor,quad_owner,quad_neighbor])
    used_t=needed[needed<nt];used_p=needed[needed>=nt]-nt
    cellmap=np.full(nt+len(pr),-1,dtype=np.int32);cellmap[used_t]=np.arange(len(used_t));cellmap[nt+used_p]=len(used_t)+np.arange(len(used_p))
    growth_cols=np.unique(active[active<(layers+1)*nv]%nv)
    if not len(growth_cols):
        # A fixed diagnostic column keeps growth reductions defined during
        # core-only optimization; it contributes no coordinate variables.
        growth_cols=np.array([0],dtype=np.int64)
    normal_faces=np.flatnonzero(np.isin(f,growth_cols).any(axis=1)) if normal_reference is not None else np.empty(0,dtype=int)
    normal_global=np.arange(layers+1)[:,None,None]*nv+f[normal_faces][None,:,:]
    growth_global=np.arange(layers+1)[:,None]*nv+growth_cols[None,:]
    usedpoints=np.unique(np.r_[t[used_t].ravel(),pr[used_p].ravel(),growth_global.ravel(),normal_global.ravel()]);pointmap=np.full(len(p),-1,dtype=np.int32);pointmap[usedpoints]=np.arange(len(usedpoints))
    normal_ids=jnp.asarray(pointmap[normal_global])
    if normal_reference is not None:
        normal_targets=jnp.asarray(normal_limit[:,normal_faces])
        normal_allowance=jnp.asarray(np.maximum(normal_reference[1:,normal_faces]/normal_reference[:-1,normal_faces]+.0005,args.growth_cap))
        normal_scale=jnp.asarray(np.sqrt(normal_reference[1:,normal_faces]*normal_reference[:-1,normal_faces]))
    growth_ids=jnp.asarray(pointmap[growth_global])
    gd=jnp.asarray(column_direction[growth_cols])
    growth_reference=jnp.asarray(projected[:,growth_cols])
    top_ids=jnp.asarray(pointmap[layers*nv+growth_cols])
    top_reference=jnp.asarray(p[layers*nv+growth_cols])
    total_reference=jnp.asarray(projected[:,growth_cols].sum(axis=0))
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
    for i,j in ((0,1),(1,2),(2,0),(3,4),(4,5),(5,3),(0,3),(1,4),(2,5)):
        length=np.linalg.norm(p[pr[:,i]]-p[pr[:,j]],axis=1)
        for k in (i,j):
            pi=param[pr[:,k]];valid=pi<len(active)
            np.minimum.at(scale,pi[valid],length[valid])
    if args.core_scale_floor:
        total=np.zeros(len(active));count=np.zeros(len(active))
        for i in range(4):
            for j in range(i):
                length=np.linalg.norm(p[t[:,i]]-p[t[:,j]],axis=1)
                for k in (i,j):
                    pi=param[t[:,k]];valid=pi<len(active)
                    np.add.at(total,pi[valid],length[valid]);np.add.at(count,pi[valid],1)
        mean=total/np.maximum(count,1)
        core=active>=(layers+1)*nv
        scale[core]=np.maximum(scale[core],args.core_scale_floor*mean[core])
    scale=np.maximum(scale,1e-7)
    anisotropic_normal=np.zeros((len(active)+1,3));tangent_scale=scale.copy()
    prism_active=active<(layers+1)*nv
    anisotropic_normal[np.flatnonzero(prism_active)]=column_direction[active[prism_active]%nv]
    if args.tangential_scale_floor:
        length=np.linalg.norm(p[edges[:,0]]-p[edges[:,1]],axis=1)
        wall_sum=np.zeros(nv);wall_count=np.zeros(nv)
        for j in (0,1):
            np.add.at(wall_sum,edges[:,j],length);np.add.at(wall_count,edges[:,j],1)
        wall_scale=wall_sum/np.maximum(wall_count,1)
        tangent_scale[prism_active]=np.maximum(scale[prism_active],args.tangential_scale_floor*wall_scale[active[prism_active]%nv])
    an=jnp.asarray(anisotropic_normal);tr=jnp.asarray(np.r_[tangent_scale/scale,1.])
    projection_normal=np.zeros((len(active)+1,3))
    first=(active>=nv)&(active<2*nv)
    direction=p[active[first]]-p[active[first]-nv]
    direction/=np.linalg.norm(direction,axis=1)[:,None]
    projection_normal[np.flatnonzero(first)]=direction
    pn=jnp.asarray(projection_normal)
    local_scale=np.r_[scale,0.][param[usedpoints]]*weight[usedpoints]
    p0=jnp.asarray(p[usedpoints]);par=jnp.asarray(param[usedpoints]);ss=jnp.asarray(local_scale)
    local_t=jnp.asarray(local_t);local_p=jnp.asarray(local_p)
    reference_jac=[]
    q0=p[pr[used_p]]
    for z in (0.,.5,1.):
        dr=(1-z)*(q0[:,1]-q0[:,0])+z*(q0[:,4]-q0[:,3]);ds=(1-z)*(q0[:,2]-q0[:,0])+z*(q0[:,5]-q0[:,3])
        for k in range(3):reference_jac.append(np.einsum('ij,ij->i',np.cross(dr,ds),q0[:,k+3]-q0[:,k]))
    reference_jac=jnp.asarray(np.array(reference_jac))
    tq0=p[t[used_t]]
    reference_tet=jnp.asarray(np.einsum('ij,ij->i',np.cross(tq0[:,1]-tq0[:,0],tq0[:,2]-tq0[:,0]),tq0[:,3]-tq0[:,0]))
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
        xx=xx-jnp.sum(xx*pn,axis=1)[:,None]*pn
        xx=tr[:,None]*xx+(1-tr)[:,None]*jnp.sum(xx*an,axis=1)[:,None]*an
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
        loss=loss+args.goal_weight*jnp.sum(jnp.maximum(args.required_minimum+2*args.margin-values,0.)**2)
        jj=[]
        for z in (0.,.5,1.):
            dr=(1-z)*(qp[:,1]-qp[:,0])+z*(qp[:,4]-qp[:,3]);ds=(1-z)*(qp[:,2]-qp[:,0])+z*(qp[:,5]-qp[:,3]);cross=jnp.cross(dr,ds)
            for k in range(3):jj.append(dot(cross,qp[:,k+3]-qp[:,k]))
        ratio=jnp.array(jj)/reference_jac
        # Exact quadratic-in-z corner minima, in addition to the z samples.
        j0,jm,j1=jnp.array(jj[:3]),jnp.array(jj[3:6]),jnp.array(jj[6:9])
        aa=2*(j1+j0-2*jm);bb=j1-j0-aa
        zz=-bb/(2*jnp.where(aa>1e-30,aa,1.))
        critical=jnp.where((aa>0)&(zz>0)&(zz<1),aa*zz*zz+bb*zz+j0,jnp.minimum(j0,j1))
        r0,rm,r1=reference_jac[:3],reference_jac[3:6],reference_jac[6:9]
        ra=2*(r1+r0-2*rm);rb=r1-r0-ra;rz=-rb/(2*jnp.where(ra>1e-30,ra,1.))
        rc=jnp.where((ra>0)&(rz>0)&(rz<1),ra*rz*rz+rb*rz+r0,jnp.minimum(r0,r1))
        cratio=critical/rc
        tet_ratio=dot(jnp.cross(qt[:,1]-qt[:,0],qt[:,2]-qt[:,0]),qt[:,3]-qt[:,0])/reference_tet
        loss=loss+args.jacobian_weight*(jnp.sum(jnp.maximum(args.jacobian_floor-ratio,0.)**2)+jnp.sum(jnp.maximum(args.jacobian_floor-cratio,0.)**2)+
                                       jnp.sum(jnp.maximum(args.jacobian_floor-tet_ratio,0.)**2))
        dp=pp[growth_ids[1:]]-pp[growth_ids[:-1]]
        hp=jnp.sum(dp*gd[None,:,:],axis=2);hn=jnp.sqrt(jnp.sum(dp*dp,axis=2)+1e-60)
        denom=jnp.sqrt(growth_reference[1:]*growth_reference[:-1])
        violation_p=jnp.maximum((hp[1:]-args.growth_target*hp[:-1])/denom,0.)
        violation_n=jnp.maximum((hn[1:]-args.growth_target*hn[:-1])/denom,0.)
        floor_p=jnp.maximum((args.growth_floor*hp[:-1]-hp[1:])/denom,0.)[args.freeze_layer-1:]
        floor_n=jnp.maximum((args.growth_floor*hn[:-1]-hn[1:])/denom,0.)[args.freeze_layer-1:]
        loss=loss+args.growth_weight*(jnp.sum(violation_p**2)+jnp.sum(violation_n**2)+
                                    jnp.sum(floor_p**2)+jnp.sum(floor_n**2)+
                                    jnp.sum(jnp.maximum((.1*growth_reference-hp)/growth_reference,0.)**2))
        normal_change=jnp.sum((pp[top_ids]-top_reference)*gd,axis=1)/(.05*total_reference)
        loss=loss+.5*jnp.sum(normal_change**2)+1e-6*jnp.sum(x*x)
        face_growth_excess=jnp.array(-1.)
        face_growth_max=jnp.array(0.)
        if normal_reference is not None:
            tri=pp[normal_ids]
            nn=jnp.cross(tri[:-1,:,1]-tri[:-1,:,0],tri[:-1,:,2]-tri[:-1,:,0])
            nn=nn/norm(nn)[:,:,None]
            hh=jnp.mean(dot(tri[1:]-tri[:-1],nn[:,:,None,:]),axis=2)
            face_growth=hh[1:]/hh[:-1]
            loss+=args.growth_weight*jnp.sum(jnp.maximum((hh[1:]-normal_targets*hh[:-1])/normal_scale,0.)**2)
            face_growth_excess=jnp.max(face_growth-normal_allowance)
            face_growth_max=jnp.max(face_growth)
        minimum_tet_ratio=jnp.min(tet_ratio) if len(used_t) else jnp.inf
        return loss,(jnp.min(values),jnp.sum(values<args.required_minimum),jnp.minimum(jnp.minimum(jnp.min(ratio),jnp.min(cratio)),minimum_tet_ratio),
                     jnp.max(hp[1:]/hp[:-1]),jnp.max(hn[1:]/hn[:-1]),
                     jnp.min((hp[1:]/hp[:-1])[args.freeze_layer-1:]),jnp.min((hn[1:]/hn[:-1])[args.freeze_layer-1:]),face_growth_excess,face_growth_max)
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
        previous=saved['active'];positions=np.searchsorted(active,previous)
        assert np.array_equal(active[positions],previous)
        mapped=np.zeros((len(active),3));previous_x=saved['x'].reshape((-1,3))
        previous_tangent=saved['tangent_scale'] if 'tangent_scale' in saved.files else saved['scale']
        normals=anisotropic_normal[positions]
        delta=previous_x*previous_tangent[:,None]+np.sum(previous_x*normals,axis=1)[:,None]*normals*(saved['scale']-previous_tangent)[:,None]
        dn=np.sum(delta*normals,axis=1)[:,None]*normals
        mapped[positions]=(delta-dn)/tangent_scale[positions,None]+dn/scale[positions,None]
        x=mapped.ravel()
    callbacks=[0]
    def save(x):
        np.savez(D/'optimization_parameters.tmp.npz',x=x,active=active,scale=scale,tangent_scale=tangent_scale)
        (D/'optimization_parameters.tmp.npz').replace(D/'optimization_parameters.npz')
        xx=x.reshape((-1,3)).copy();nn=projection_normal[:-1]
        xx-=np.sum(xx*nn,axis=1)[:,None]*nn
        rr=tangent_scale/scale;nn=anisotropic_normal[:-1]
        xx=rr[:,None]*xx+(1-rr)[:,None]*np.sum(xx*nn,axis=1)[:,None]*nn
        pp=p.copy();valid=param<len(active);pp[valid]+=xx[param[valid]]*(np.r_[scale,0.][param[valid]]*weight[valid])[:,None]
        np.savez(D/'optimized_coordinates.tmp.npz',points=pp,cells_4=t,cells_6=pr)
        (D/'optimized_coordinates.tmp.npz').replace(D/'optimized_coordinates.npz')
    def callback(x):
        callbacks[0]+=1
        if callbacks[0]%25==0:save(x)
        if (last[0] is not None and last[0][1][0]>=args.required_minimum+args.margin and last[0][1][1]==0 and last[0][1][2]>.5*args.jacobian_floor
            and max(last[0][1][3:5])<=args.growth_cap-.0001
            and min(last[0][1][5:7])>=args.growth_floor-.001 and last[0][1][7]<=0):
            save(x)
            raise StopIteration
    if args.adam_steps:
        # Per-coordinate normalization avoids a few tiny interface elements
        # setting the step size for every other independent mesh region.
        moment=np.zeros_like(x);variance=np.zeros_like(x)
        for step in range(1,args.adam_steps+1):
            value,gradient=scipyfun(x)
            moment=.9*moment+.1*gradient;variance=.999*variance+.001*gradient*gradient
            rate=args.adam_rate*min(1.,20./max(20.,step)**.5)
            x=np.clip(x-rate*(moment/(1-.9**step))/(np.sqrt(variance/(1-.999**step))+1e-8),
                      -args.coordinate_bound,args.coordinate_bound)
            if step%25==0:save(x)
        save(x)
    result=minimize(scipyfun,x,jac=True,method='L-BFGS-B',bounds=[(-args.coordinate_bound,args.coordinate_bound)]*len(x),callback=callback,
                    options=dict(maxiter=args.iterations,ftol=1e-14,gtol=1e-7,maxls=30,maxcor=args.maxcor))
    save(result.x)
    value,stats=evaluate(result.x)
    report=dict(success=bool(result.success),message=str(result.message),iterations=int(result.nit),evaluations=counter[0],
                loss=float(value),minimum=float(stats[0]),bad_faces=int(stats[1]),minimum_jacobian_ratio=float(stats[2]),seconds=time.time()-started,
                wall_fixed=True,first_layer_projected_height_fixed=True,metric_target=args.target,
                required_minimum=args.required_minimum,
                fixed_layers=args.freeze_layer,
                maximum_projected_growth=float(stats[3]),maximum_euclidean_growth=float(stats[4]),
                minimum_outer_projected_growth=float(stats[5]),minimum_outer_euclidean_growth=float(stats[6]),
                maximum_face_normal_growth=float(stats[8]),maximum_face_normal_growth_excess=float(stats[7]),
                quality_goal_achieved=bool(float(stats[0])>=args.required_minimum and int(stats[1])==0 and float(stats[2])>0.),
                growth_goal_achieved=bool(max(float(stats[3]),float(stats[4]))<=args.growth_cap))
    (D/'coupled_optimization.json').write_text(json.dumps(report,indent=2)+'\n');print('RESULT',json.dumps(report),flush=True)


if __name__=='__main__':main()
