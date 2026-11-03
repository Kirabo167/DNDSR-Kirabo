#!/usr/bin/env python3
"""Compare actual polygonal Z=0 cuts before/after layer smoothing."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import numpy as np
import pyvista as pv

ap=argparse.ArgumentParser()
ap.add_argument('before',type=Path)
ap.add_argument('after',type=Path)
ap.add_argument('--fixed-layers',type=int,default=16)
ap.add_argument('--preserved-normal-height',action='store_true')
args=ap.parse_args()
fig,axs=plt.subplots(2,2,figsize=(16,8.5))
windows=[(.99,1.30,.450,.530),(3.90,4.07,.215,.295)]
for col,(directory,label) in enumerate(((args.before,'Before'),(args.after,'After'))):
    cut=pv.read(directory/'sections/z0.vtp')
    poly=[cut.points[ids,:2] for ids in cut.irregular_faces]
    bounds=np.array([[p.min(axis=0),p.max(axis=0)] for p in poly])
    region=np.asarray(cut['region'])
    for row,(x0,x1,y0,y1) in enumerate(windows):
        ax=axs[row,col]
        selected=(bounds[:,1,0]>=x0)&(bounds[:,0,0]<=x1)&(bounds[:,1,1]>=y0)&(bounds[:,0,1]<=y1)
        for kind,face,edge in ((0,'#edf3f8','#506879'),(1,'#fff0d4','#9c642e')):
            ids=np.flatnonzero(selected&(region==kind))
            ax.add_collection(PolyCollection([poly[i] for i in ids],facecolors=face,edgecolors=edge,linewidths=.32,rasterized=True))
        ax.set(xlim=(x0,x1),ylim=(y0,y1),xlabel='X (m)',ylabel='Y (m)',
               title=f"{label}: {'sail roof' if row==0 else 'stern-fin tip'}")
        ax.set_aspect('equal',adjustable='box')
        ax.ticklabel_format(style='plain',useOffset=False)
fig.suptitle('Actual mesh sections before / after local boundary-layer optimization | Z = 0 m',fontsize=17)
preserved='wall unchanged; first mean normal height retained' if args.preserved_normal_height else f'wall and first {args.fixed_layers} layers fixed'
fig.text(.5,.02,f'Orange: prismatic layers | Blue: tetrahedra | Same view windows; {preserved}',ha='center')
fig.tight_layout(rect=(0,.05,1,.96))
fig.savefig(args.after/'sections/05_before_after_comparison.png',dpi=220,bbox_inches='tight')
plt.close(fig)
