#!/usr/bin/env python3
"""Check each complete layer front and wall for self/inter-surface crossings."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ap=argparse.ArgumentParser();ap.add_argument('directory',type=Path)
ap.add_argument('--arrays',default='SUBOFF_optimized.validated.npz')
args=ap.parse_args();D=args.directory
sys.path.insert(0,'/tmp/suboff_regen_deps')
import pymeshlab
data=np.load(D/args.arrays);p=data['points'];f=np.load(D/'prism_front.npz')['faces'];nv=int(f.max())+1
result={}
layers=len(data['cells_6'])//len(f)
for layer in range(1,layers+1):
    checker=pymeshlab.MeshSet()
    checker.add_mesh(pymeshlab.Mesh(np.vstack([p[:nv],p[layer*nv:(layer+1)*nv]]),np.vstack([f,f+nv])))
    checker.apply_filter('compute_selection_by_self_intersections_per_face')
    count=int(checker.current_mesh().selected_face_number());result[str(layer)]=count
    print('LAYER',layer,'INTERSECTING_FACES',count,flush=True)
    del checker
(D/'layer_intersections.json').write_text(json.dumps(dict(by_layer=result,all_clear=not any(result.values())),indent=2)+'\n')
if any(result.values()):raise ValueError('Intersecting layer fronts')
