#!/usr/bin/env python3
"""Export true polygonal cuts of the final hybrid SUBOFF volume mesh."""
import argparse
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection, LineCollection
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pyvista as pv


def polygons(section, axes):
    points=section.points[:,axes]
    return [points[ids] for ids in section.irregular_faces]


def wall_segments(section,axes):
    ids=section.lines;segments=[];i=0
    while i<len(ids):
        count=int(ids[i]);line=section.points[ids[i+1:i+1+count]][:,axes]
        segments.extend(np.stack([line[:-1],line[1:]],axis=1));i+=count+1
    return np.array(segments)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('directory',type=Path)
    ap.add_argument('--cgns-name',default='SUBOFF_30Lx20Lx20L_Q02.cgns')
    args=ap.parse_args();D=args.directory;out=D/'sections';out.mkdir(exist_ok=True)
    started=time.time();data=np.load(D/'SUBOFF_optimized.validated.npz')
    points=data['points'];tet=data['cells_4'];pr=data['cells_6'];nt=len(tet)
    layers=len(pr)//len(data['boundary_WALL'])
    # Connectivity comes from the exact CGNS-export arrays; no remeshing.
    grid=pv.UnstructuredGrid({pv.CellType.TETRA:tet,pv.CellType.WEDGE:pr},points)
    assert np.all(grid.celltypes[:nt]==pv.CellType.TETRA)
    assert np.all(grid.celltypes[nt:]==pv.CellType.WEDGE)
    grid.cell_data['region']=np.r_[np.zeros(nt,dtype=np.uint8),np.ones(len(pr),dtype=np.uint8)]
    grid.cell_data['CGNS_volume_element_id']=np.arange(1,grid.n_cells+1,dtype=np.int64)
    grid.cell_data['orthogonal_quality']=np.load(D/'orthogonal_quality_values.npy')
    wall_faces=data['boundary_WALL']
    wall=pv.PolyData(points,np.c_[np.full(len(wall_faces),3),wall_faces].ravel())
    planes=[('z0',(0,0,1),(0,0,0),(0,1),'Z = 0 m','X (m)','Y (m)'),
            ('y0',(0,1,0),(0,0,0),(0,2),'Y = 0 m','X (m)','Z (m)'),
            ('x1p10',(1,0,0),(1.10,0,0),(2,1),'X = 1.10 m','Z (m)','Y (m)'),
            ('x3p95',(1,0,0),(3.95,0,0),(2,1),'X = 3.95 m','Z (m)','Y (m)')]
    sections={};manifest=dict(source_cgns=args.cgns_name,cells=grid.n_cells,
                             source_sha256=json.loads((D/'mesh_validation.json').read_text())['sha256'],
                             minimum_orthogonal_quality=json.loads((D/'orthogonal_quality.json').read_text())['minimum'],
                             slicing='Volume-plane cuts of original cells; generate_triangles=False',planes={})
    for name,n,o,axes,label,xlab,ylab in planes:
        section=grid.slice(normal=n,origin=o,generate_triangles=False)
        if section.n_cells==0:raise ValueError(f'Empty section {name}')
        section.save(out/(name+'.vtp'))
        ws=wall.slice(normal=n,origin=o,generate_triangles=False)
        poly=polygons(section,list(axes));bounds=np.array([[v.min(axis=0),v.max(axis=0)] for v in poly])
        sections[name]=(poly,bounds,np.asarray(section['region']),wall_segments(ws,list(axes)),label,xlab,ylab)
        manifest['planes'][name]=dict(normal=n,origin_m=o,polygons=section.n_cells,
                                    polygon_vertex_counts=dict(zip(*[x.tolist() for x in np.unique([len(x) for x in poly],return_counts=True)])))
        print('SLICE',name,section.n_cells,'SECONDS',time.time()-started,flush=True)
    del grid
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.titlesize':13,
                         'axes.labelsize':11,'savefig.facecolor':'white'})
    def draw(ax,name,limits,title=None,detail=False):
        poly,bounds,region,segments,label,xlab,ylab=sections[name]
        x0,x1,y0,y1=limits
        selected=(bounds[:,1,0]>=x0)&(bounds[:,0,0]<=x1)&(bounds[:,1,1]>=y0)&(bounds[:,0,1]<=y1)
        for kind,face,edge,lw in ((0,'#edf3f8','#506879',.28 if detail else .20),
                                  (1,'#fff0d4','#9c642e',.44 if detail else .28)):
            ids=np.flatnonzero(selected&(region==kind))
            ax.add_collection(PolyCollection([poly[i] for i in ids],facecolors=face,edgecolors=edge,
                                             linewidths=lw,antialiased=True,rasterized=True))
        if len(segments):ax.add_collection(LineCollection(segments,colors='#152536',linewidths=1.0,rasterized=True))
        ax.set(xlim=(x0,x1),ylim=(y0,y1),xlabel=xlab,ylabel=ylab)
        ax.set_aspect('equal',adjustable='box');ax.set_facecolor('white')
        if title:ax.set_title(title,loc='left',pad=11,fontweight='bold')
        ax.tick_params(direction='out',length=3,color='#586b7b')
        for spine in ax.spines.values():spine.set_color('#7b8b98')
        ax.ticklabel_format(axis='both',style='plain',useOffset=False)
    legend=[Patch(facecolor='#edf3f8',edgecolor='#506879',label='Tetrahedral core / far field'),
            Patch(facecolor='#fff0d4',edgecolor='#9c642e',label=f'{layers}-layer prismatic boundary layer')]
    def save(fig,name):
        fig.savefig(out/name,dpi=240,bbox_inches='tight',pad_inches=.16);plt.close(fig)
        print('FIGURE',name,flush=True)
    fig,ax=plt.subplots(figsize=(12,9))
    draw(ax,'z0',(-42.108,88.572,-43.56,43.56),'Expanded domain: 30L x 20L x 20L | Z = 0 m')
    ax.add_patch(Rectangle((-13.068,-13.068),43.56,26.136,fill=False,edgecolor='#c04554',linestyle='--',linewidth=1.3))
    ax.text(-12.2,14.2,'Original domain extent (now internal)',color='#aa3444',fontsize=10)
    fig.suptitle('SUBOFF volume-mesh section',fontsize=19,fontweight='bold',y=.98)
    fig.text(.5,.015,f'L = 4.356 m | {len(tet)+len(pr):,} cells | {layers} prismatic layers',ha='center',fontsize=11)
    fig.tight_layout(rect=(0,.09,1,.95));save(fig,'01_full_domain_z0.png')

    fig,axs=plt.subplots(2,1,figsize=(17,8.4))
    draw(axs[0],'z0',(-.15,4.52,-.39,.61),'(a) Longitudinal center plane Z = 0 m: sail and vertical stern fin')
    draw(axs[1],'y0',(-.15,4.52,-.50,.50),'(b) Longitudinal center plane Y = 0 m: horizontal stern fins')
    fig.legend(handles=legend,loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(.5,.005))
    fig.suptitle('Near-body mesh: true cell intersections',fontsize=18,fontweight='bold',y=.99)
    fig.tight_layout(rect=(0,.06,1,.95),h_pad=2.5);save(fig,'02_longitudinal_sections.png')

    fig,axs=plt.subplots(2,2,figsize=(15,8.4))
    roof_window=(.99,1.32,.450,.550)
    fin_window=(3.90,4.10,.215,.310)
    draw(axs[0,0],'z0',roof_window,'(a) Sail roof: layer-to-core transition',True)
    draw(axs[0,1],'z0',fin_window,'(b) Stern-fin tip: layer-to-core transition',True)
    draw(axs[1,0],'z0',(1.090,1.110,.4752,.4812),'(c) Sail roof: near-wall layer detail',True)
    draw(axs[1,1],'z0',(3.960,3.980,.2428,.2488),'(d) Stern-fin tip: near-wall layer detail',True)
    fig.legend(handles=legend,loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(.5,.008))
    fig.suptitle('Boundary-layer details | Z = 0 m',fontsize=18,fontweight='bold',y=.98)
    fig.text(.5,.045,f'{layers} layers in the mesh; close-ups show the near-wall subset. Nominal first normal height: 0.01 mm.',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.075,1,.94),h_pad=3,w_pad=2);save(fig,'03_boundary_layer_details.png')

    fig,axs=plt.subplots(1,2,figsize=(14,8))
    draw(axs[0],'x1p10',(-.39,.39,-.39,.59),'(a) Transverse section X = 1.10 m: sail')
    draw(axs[1],'x3p95',(-.34,.34,-.34,.34),'(b) Transverse section X = 3.95 m: stern fins')
    fig.legend(handles=legend,loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(.5,.02))
    fig.suptitle('Transverse mesh sections',fontsize=18,fontweight='bold',y=.96)
    fig.tight_layout(rect=(0,.07,1,.91),w_pad=2);save(fig,'04_transverse_sections.png')
    manifest['figures']=['01_full_domain_z0.png','02_longitudinal_sections.png','03_boundary_layer_details.png','04_transverse_sections.png']
    manifest['seconds']=time.time()-started
    (out/'sections.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('DONE',manifest['seconds'],flush=True)


if __name__=='__main__':main()
