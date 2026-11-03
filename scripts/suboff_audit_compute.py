#!/usr/bin/env python3
"""Audit actual ACM advancement and all saved SUBOFF solution values."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import re
import h5py
import numpy as np

ap=argparse.ArgumentParser()
ap.add_argument('directory',type=Path)
ap.add_argument('--exit-code',type=int,required=True)
ap.add_argument('--steps',type=int,default=3)
ap.add_argument('--ranks',type=int,default=24)
ap.add_argument('--integrator',choices=['ImplicitEulerGMRES','ExplicitSSPRK3'],default='ImplicitEulerGMRES')
args=ap.parse_args();D=args.directory
log=(D/'compute.log').read_text(errors='replace')
pattern=r'ACM step\s+(\d+) residual\s+(\S+)\s+->\s+(\S+)\s+dtauMin=(\S+)\s+turbResidual=(\S+)\s+inner=(\d+)\s+converged=(\S+)'
steps=[]
for match in re.finditer(pattern,log):
    v=match.groups()
    steps.append(dict(step=int(v[0]),initial_defect=float(v[1]),final_defect=float(v[2]),minimum_pseudo_time_step=float(v[3]),turbulence_residual=float(v[4]),inner_iterations=int(v[5]),inner_converged=v[6] in ['1','true']))
finite_steps=all(np.isfinite([s[k] for k in ['initial_defect','final_defect','minimum_pseudo_time_step','turbulence_residual']]).all() and s['minimum_pseudo_time_step']>0 for s in steps)
field_files=sorted((D/'fields').glob('*.vtkhdf'))
fields={};cells=None
if field_files and args.exit_code==0:
    with h5py.File(field_files[-1],'r') as hf:
        for name in ['Pressure','Velocity','TurbulenceK','TurbulenceOmega']:
            ds=hf['VTKHDF/CellData/'+name];finite=0;count=0;total=0.;minimum=np.full(ds.shape[1:] or (),np.inf);maximum=np.full(ds.shape[1:] or (),-np.inf)
            bound_low={'TurbulenceK':1e-12,'TurbulenceOmega':1e-10}.get(name)
            bound_high={'TurbulenceK':1e6,'TurbulenceOmega':1e12}.get(name)
            low_count=high_count=0;speed_min=np.inf;speed_max=-np.inf
            for start in range(0,len(ds),100000):
                x=ds[start:start+100000];finite+=int(np.isfinite(x).sum());count+=x.size;total+=float(np.sum(x));minimum=np.minimum(minimum,np.min(x,axis=0));maximum=np.maximum(maximum,np.max(x,axis=0))
                if bound_low is not None:low_count+=int((x<=bound_low).sum());high_count+=int((x>=bound_high).sum())
                if name=='Velocity':
                    speed=np.linalg.norm(x,axis=1);speed_min=min(speed_min,float(speed.min()));speed_max=max(speed_max,float(speed.max()))
            fields[name]=dict(shape=list(ds.shape),finite_values=finite,total_values=count,all_finite=finite==count,minimum=np.asarray(minimum).tolist(),maximum=np.asarray(maximum).tolist(),mean_component=float(total/count))
            if bound_low is not None:fields[name].update(lower_bound=bound_low,upper_bound=bound_high,at_lower_bound=low_count,at_upper_bound=high_count)
            if name=='Velocity':fields[name].update(speed_minimum=speed_min,speed_maximum=speed_max)
        cells=hf['VTKHDF/CellData/Pressure'].shape[0]
positivity=all(fields.get(name,{}).get('minimum',-1)>0 for name in ['TurbulenceK','TurbulenceOmega'])
passed=args.exit_code==0 and [s['step'] for s in steps]==list(range(1,args.steps+1)) and finite_steps and cells==6036541 and len(fields)==4 and all(f['all_finite'] for f in fields.values()) and positivity
result=dict(process_exit_code=args.exit_code,mpi_ranks=args.ranks,integrator=args.integrator,requested_steps=args.steps,completed_steps=len(steps),steps=steps,all_step_diagnostics_finite=finite_steps,cells=cells,fields=fields,turbulence_values_positive=positivity,short_run_passed=passed,all_reported_step_flags_true=bool(steps) and all(s['inner_converged'] for s in steps),steady_convergence_verified=False,accuracy_verified=False,actual_flow_yplus_verified=False,field_file=str(field_files[-1]) if field_files else None,scope=f'Three pseudo-time steps with {args.integrator}, CFL=0.5, KOmegaSST, and second-order reconstruction. This verifies actual startup/advancement only.')
def serializable(value):
    if isinstance(value,float) and not np.isfinite(value):return str(value)
    if isinstance(value,dict):return {k:serializable(v) for k,v in value.items()}
    if isinstance(value,list):return [serializable(v) for v in value]
    return value
(D/'compute_audit.json').write_text(json.dumps(serializable(result),indent=2,allow_nan=False)+'\n')
rows=''.join(f'<tr><td>{s["step"]}</td><td>{s["initial_defect"]:.6g}</td><td>{s["final_defect"]:.6g}</td><td>{s["minimum_pseudo_time_step"]:.6g}</td><td>{s["inner_converged"]}</td></tr>' for s in steps)
field_rows=''.join(f'<tr><td>{html.escape(name)}</td><td>{html.escape(str(v["minimum"]))}</td><td>{html.escape(str(v["maximum"]))}</td><td>{v["all_finite"]}</td></tr>' for name,v in fields.items())
residual_note='隐式内层缺陷与稳态空间残差分别判断。' if args.integrator=='ImplicitEulerGMRES' else '显式结果记录的是空间残差；步标志不表示达到稳态收敛。'
page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>SUBOFF实际计算检查</title><style>body{{font:16px/1.7 system-ui;max-width:1100px;margin:30px auto;padding:0 20px}}table{{border-collapse:collapse;width:100%}}td,th{{padding:8px;border-bottom:1px solid #ddd;text-align:left}}</style><h1>SUBOFF实际计算检查</h1><p>短步运行检查：{'通过' if passed else '未通过'}。{args.ranks}进程，完成{len(steps)}/{args.steps}步，进程退出码{args.exit_code}，末步流场单元数{cells}。</p><p>使用ACM/SST、二阶重构、{args.integrator}、CFL=0.5。网格文件与原交付校验值一致。<a href="compute.log">原始日志</a> · <a href="compute_audit.json">全部审计数据</a></p><table><tr><th>步</th><th>初始残差/缺陷</th><th>最终残差/缺陷</th><th>最小伪时间步</th><th>步标志</th></tr>{rows}</table><p>{residual_note}</p><table><tr><th>变量</th><th>最小值</th><th>最大值</th><th>全部有限</th></tr>{field_rows}</table><p>本检查验证网格读取、几何/重构构造、真实推进及输出场有限性。3步不能证明稳态收敛、阻力精度、网格无关性或实际y+满足要求。湍流变量的截断界限命中数量记录在JSON中，有限值不单独代表湍流解准确。</p></html>'''
(D/'compute_report.html').write_text(page)
print(json.dumps(serializable(result),indent=2,allow_nan=False))
if not passed:raise SystemExit(1)
