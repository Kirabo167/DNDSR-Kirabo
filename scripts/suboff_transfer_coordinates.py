#!/usr/bin/env python3
"""Transfer displacement at exactly matching mesh nodes, keeping positivity."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from suboff_arrays_cgns import jacobians

ap=argparse.ArgumentParser();ap.add_argument('previous',type=Path);ap.add_argument('source',type=Path);ap.add_argument('output',type=Path);args=ap.parse_args();O,S,D=args.previous,args.source,args.output;D.mkdir(exist_ok=False)
old=np.load(O/'SUBOFF_optimized.validated.npz')['points'];moved=np.load(O/'optimized_coordinates.npz')['points'];a=np.load(S/'SUBOFF_optimized.validated.npz');p=a['points'];distance,near=cKDTree(old).query(p);matched=distance<1e-10;delta=np.zeros_like(p);delta[matched]=moved[near[matched]]-old[near[matched]]
moving=(delta!=0).any(axis=1)
affected={name:a[name][moving[a[name]].any(axis=1)] for name in ['cells_4','cells_6']}
factor=np.ones(len(p))
for attempt in range(40):
    pp=p+factor[:,None]*delta;minimum={};bad_nodes=[]
    for name in ['cells_4','cells_6']:
        cells=affected[name];minimum[name]=float('inf')
        for k in range(0,len(cells),100000):
            block=cells[k:k+100000];j=jacobians(pp,block);minimum[name]=min(minimum[name],float(j.min()));bad=(j<=0).any(axis=1);bad_nodes.extend(block[bad].ravel())
    print('TRANSFER',attempt,minimum,'BAD_NODE_OCCURRENCES',len(bad_nodes),flush=True)
    if not bad_nodes:break
    factor[np.unique(bad_nodes)]*=.5
else:raise RuntimeError('No positive transferred state')
np.savez(D/'SUBOFF_optimized.validated.npz',**{k:pp if k=='points' else a[k] for k in a.files})
for name in ['prism_front.npz','fixed_boundary.npz','boundary_arrays.npz','local_targets.npz','core_generation.json','cavity_provenance.npz','protect_nodes.npy']:
    if (S/name).exists():(D/name).write_bytes((S/name).read_bytes())
plan=json.loads((S/'thickening_plan.json').read_text());plan['warm_source']=str(O);plan['warm_fraction_quantiles']=np.quantile(factor[moving],[0,.05,.5,.95,1]).tolist();plan['warm_matched_nodes']=int(matched.sum());(D/'thickening_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
