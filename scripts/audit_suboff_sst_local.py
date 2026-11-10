#!/usr/bin/env python3
"""Read-only local SST source audit from saved VTKHDF geometry.

This independent geometric reconstruction is approximate on warped faces. The
resident snapshot diagnostic is authoritative for the exact discrete operator.
No DNDSR Python extension is used.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pyvista as pv
import vtk


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('field', type=Path)
    parser.add_argument('--cell', type=int, default=2601986)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    mesh = pv.read(args.field)
    mesh.BuildLinks()
    surface = mesh.extract_surface(algorithm='dataset_surface')
    locator = vtk.vtkStaticCellLocator()
    locator.SetDataSet(surface)
    locator.BuildLocator()
    mu = 3.630083418e-7

    def center(cell):
        return np.array([cell.GetPoints().GetPoint(i) for i in range(cell.GetNumberOfPoints())]).mean(axis=0)

    def distance(cell):
        vectors = []
        for i in range(cell.GetNumberOfPoints()):
            point = np.array(cell.GetPoints().GetPoint(i))
            closest = [0., 0., 0.]
            cid, subid, square = vtk.reference(0), vtk.reference(0), vtk.reference(0.)
            locator.FindClosestPoint(point, closest, cid, subid, square)
            vectors.append(point-np.asarray(closest))
        return max(float(np.linalg.norm(np.mean(vectors, axis=0))), 1e-10)

    cid = args.cell
    cell = vtk.vtkGenericCell()
    mesh.GetCell(cid, cell)
    cc = center(cell)
    volume = float(mesh.extract_cells([cid]).compute_cell_sizes(length=False, area=False)['Volume'][0])
    flow = np.r_[mesh.cell_data['Velocity'][cid], mesh.cell_data['Pressure'][cid]]
    q = np.array([mesh.cell_data['TurbulenceK'][cid], mesh.cell_data['TurbulenceOmega'][cid]])
    d = distance(cell)
    gf, gq, consistency = np.zeros((4, 3)), np.zeros((2, 3)), np.zeros((3, 3))
    neighbors = []
    for i in range(cell.GetNumberOfFaces()):
        face = cell.GetFace(i)
        points = np.array([face.GetPoints().GetPoint(j) for j in range(face.GetNumberOfPoints())])
        fc = points.mean(axis=0)
        area_vector = sum((np.cross(points[j]-points[0], points[j+1]-points[0])/2 for j in range(1,len(points)-1)))
        if np.dot(area_vector, fc-cc)<0: area_vector *= -1
        area = np.linalg.norm(area_vector)
        normal = area_vector/area
        ids = vtk.vtkIdList()
        mesh.GetCellNeighbors(cid, face.GetPointIds(), ids)
        if ids.GetNumberOfIds():
            nid = ids.GetId(0)
            neighbor = vtk.vtkGenericCell()
            mesh.GetCell(nid, neighbor)
            nc = center(neighbor)
            other_flow = np.r_[mesh.cell_data['Velocity'][nid],mesh.cell_data['Pressure'][nid]]
            other_q = np.array([mesh.cell_data['TurbulenceK'][nid],mesh.cell_data['TurbulenceOmega'][nid]])
            interpolation = np.linalg.norm(fc-cc)/(np.linalg.norm(fc-cc)+np.linalg.norm(nc-fc))
            displacement = nc-cc
        else:
            nid = -1
            # The suspect cells are near the hull/appendage, not the outer domain.
            if np.linalg.norm(cc)>7: raise ValueError('Wall assumption invalid at outer boundary')
            other_flow = np.r_[-flow[:3],flow[3]]
            other_q = np.array([-q[0],2*800*mu/d**2-q[1]])
            interpolation = .5
            displacement = 2*normal*np.dot(fc-cc,normal)
        weight = interpolation*area_vector/volume
        gf += np.outer(other_flow-flow,weight)
        gq += np.outer(other_q-q,weight)
        consistency += np.outer(weight,displacement)
        neighbors.append(dict(cell=nid,face_points=points.tolist(),q=other_q.tolist(),
                              normal=normal.tolist(),area=float(area),displacement=displacement.tolist()))

    def neighbor_q_gradient(nid):
        owner=vtk.vtkGenericCell();mesh.GetCell(nid,owner)
        own_center=center(owner)
        own_q=np.array([mesh.cell_data['TurbulenceK'][nid],mesh.cell_data['TurbulenceOmega'][nid]])
        own_volume=float(mesh.extract_cells([nid]).compute_cell_sizes(length=False,area=False)['Volume'][0])
        gradient=np.zeros((2,3))
        for j in range(owner.GetNumberOfFaces()):
            face=owner.GetFace(j)
            points=np.array([face.GetPoints().GetPoint(i) for i in range(face.GetNumberOfPoints())])
            fc=points.mean(axis=0)
            vector=sum((np.cross(points[i]-points[0],points[i+1]-points[0])/2 for i in range(1,len(points)-1)))
            if np.dot(vector,fc-own_center)<0: vector*=-1
            ids=vtk.vtkIdList();mesh.GetCellNeighbors(nid,face.GetPointIds(),ids)
            if ids.GetNumberOfIds():
                other=ids.GetId(0);other_cell=vtk.vtkGenericCell();mesh.GetCell(other,other_cell)
                other_q=np.array([mesh.cell_data['TurbulenceK'][other],mesh.cell_data['TurbulenceOmega'][other]])
                alpha=np.linalg.norm(fc-own_center)/(np.linalg.norm(fc-own_center)+np.linalg.norm(center(other_cell)-fc))
            else:
                other_q=np.array([-own_q[0],2*800*mu/distance(owner)**2-own_q[1]])
                alpha=.5
            gradient+=np.outer(other_q-own_q,alpha*vector/own_volume)
        return gradient

    molecular=np.zeros(2);orthogonal=np.zeros(2)
    for entry in neighbors:
        if entry['cell']<0: raise ValueError('Local diffusion audit currently expects an interior suspect cell')
        gr=neighbor_q_gradient(entry['cell'])
        average=.5*(gq+gr)
        dx=np.asarray(entry['displacement']);normal=np.asarray(entry['normal'])
        dn=float(np.dot(dx,normal));jump=np.asarray(entry['q'])-q
        correction=(jump-average@dx)/dn
        derivative=average@normal+correction
        contribution=mu*entry['area']/volume*derivative
        entry['molecular_diffusion']=contribution.tolist()
        entry['nonorthogonality_degrees']=float(np.degrees(np.arccos(np.clip(dn/np.linalg.norm(dx),-1,1))))
        molecular+=contribution
        orthogonal+=mu*entry['area']/volume*jump/dn

    def source(state):
        k,w=state
        cross=float(np.dot(gq[0],gq[1]))
        cd=max(2*.856*cross/w,1e-10)
        near=np.sqrt(k)/(.09*w*d)
        visc=500*mu/(d*d*w)
        f1=np.tanh(min(10,min(max(near,visc),4*.856*k/(cd*d*d)))**4)
        f2=np.tanh(min(10,max(2*near,visc))**2)
        rotation=(gf[:3]-gf[:3].T)/2
        omega=np.sqrt(2*np.sum(rotation**2))
        eddy=min(.31*k/max(.31*w,omega*f2),1e5*mu)
        div=np.trace(gf[:3]); strain=(gf[:3]+gf[:3].T)/2-np.eye(3)*div/3
        invariant=2*np.sum(strain**2)
        production=max(0,eddy*invariant-2/3*k*div)
        gamma1=.075/.09-.5*.41**2/np.sqrt(.09)
        gamma2=.0828/.09-.856*.41**2/np.sqrt(.09)
        terms=dict(k_production=min(production,20*.09*k*w),k_destruction=-.09*k*w,
                   omega_production=(gamma1*f1+gamma2*(1-f1))*invariant,
                   omega_destruction=-(.075*f1+.0828*(1-f1))*w*w,
                   omega_cross=2*(1-f1)*.856*cross/w)
        return np.array([terms['k_production']+terms['k_destruction'],sum(terms[t] for t in ['omega_production','omega_destruction','omega_cross'])]), dict(f1=float(f1),f2=float(f2),eddy=float(eddy),cross=cross,**terms)

    src,terms=source(q)
    jac=np.zeros((2,2))
    for k in range(2):
        epsilon=np.sqrt(np.finfo(float).eps)*max(q[k],[1e-14,1e-10][k])
        plus=q.copy();minus=q.copy();plus[k]+=epsilon;minus[k]=max(q[k]-epsilon,[1e-14,1e-10][k])
        jac[:,k]=(source(plus)[0]-source(minus)[0])/(plus[k]-minus[k])
    report=dict(approximate_geometry=True,field=str(args.field.resolve()),cell=cid,center=cc.tolist(),
                volume=volume,wall_distance=d,q=q.tolist(),flow_gradient=gf.tolist(),
                q_gradient=gq.tolist(),linear_gradient_consistency=consistency.tolist(),
                source=src.tolist(),source_terms=terms,source_jacobian=jac.tolist(),neighbors=neighbors,
                molecular_diffusion=molecular.tolist(),orthogonal_diffusion=orthogonal.tolist())
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['neighbors','flow_gradient','q_gradient']},indent=2),flush=True)


if __name__=='__main__':
    main()
