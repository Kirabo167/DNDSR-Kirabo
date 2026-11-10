#!/usr/bin/env python3
"""Summarize the version-1, 49-column resident SST snapshot diagnostic."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    header=np.fromfile(args.input,dtype=np.uint64,count=3)
    if len(header)!=3 or header[0]!=1 or header[2]!=49:
        raise ValueError('Unsupported SST audit header')
    count=int(header[1])
    if args.input.stat().st_size!=24+count*49*8:
        raise ValueError('Truncated or mismatched SST audit')
    data=np.memmap(args.input,dtype=np.float64,mode='r',offset=24,shape=(count,49))
    names=['rhs_u','rhs_v','rhs_w','rhs_continuity','rhs_k','rhs_omega',
           'source_k','source_omega','diagonal_kk','diagonal_kw','diagonal_wk','diagonal_ww',
           'k','omega',*[f'grad_u{i}_x{j}' for i in range(3) for j in range(3)],
           *[f'grad_{q}_x{j}' for q in ['k','omega'] for j in range(3)],
           'local_dt','wall_distance','volume','cell_eddy_viscosity','transport_rate_k','transport_rate_omega',
           'correction_k','correction_omega','relaxed_correction_k','relaxed_correction_omega','wall_faces',
           *[f'GG_affine_response_{i}{j}' for i in range(3) for j in range(3)]]
    report=dict(version=1,cells=count,input=str(args.input.resolve()),column_names=names)
    square=np.zeros(6);finite=True;floor_negative=[0,0];floor_negative_square=np.zeros(2)
    negative_diagonal=0;nonpositive_det=0;blocked=[0,0]
    for begin in range(0,count,100000):
        a=np.asarray(data[begin:begin+100000])
        finite=finite and bool(np.isfinite(a).all())
        square+=np.sum(a[:,:6]**2,axis=0)
        negative_diagonal+=int(np.sum((a[:,8]<=0)|(a[:,11]<=0)))
        nonpositive_det+=int(np.sum(a[:,8]*a[:,11]-a[:,9]*a[:,10]<=0))
        for q in range(2):
            floor=a[:,12+q]<=1.01*[1e-14,1e-10][q]
            mask=floor & (a[:,4+q]<0)
            floor_negative[q]+=int(mask.sum())
            floor_negative_square[q]+=np.sum(a[mask,4+q]**2)
            blocked[q]+=int(np.sum(floor & (a[:,37+q]<0)))
    report.update(all_finite=finite,equation_rms=np.sqrt(square/count).tolist(),
                  nonpositive_turbulence_diagonal_cells=negative_diagonal,
                  nonpositive_turbulence_determinant_cells=nonpositive_det,
                  floor_negative_rhs_cells=floor_negative,
                  floor_negative_rhs_rms=np.sqrt(floor_negative_square/count).tolist(),
                  negative_correction_at_floor_cells=blocked)
    report['worst_cells']=[]
    for k in range(6):
        indices=np.argsort(np.abs(data[:,k]))[-10:][::-1]
        report['worst_cells'].append([dict(cell=int(i),**dict(zip(names,map(float,data[i])))) for i in indices])
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['worst_cells','column_names']},indent=2))


if __name__=='__main__':
    main()
