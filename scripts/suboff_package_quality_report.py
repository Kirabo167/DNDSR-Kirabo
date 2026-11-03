#!/usr/bin/env python3
"""Package measured mesh quality and completed solver initialization checks."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import re

ap=argparse.ArgumentParser();ap.add_argument('directory',type=Path);ap.add_argument('original',type=Path)
ap.add_argument('--acm-exit',type=int,required=True);ap.add_argument('--cgns-exit',type=int,required=True)
args=ap.parse_args();D=args.directory
read=lambda path:json.loads(path.read_text())
q=read(D/'orthogonal_quality.json');old=read(args.original/'orthogonal_quality.json')
v=read(D/'mesh_validation.json');intersections=read(D/'layer_intersections.json')
log=(D/'acm_init.log').read_text()
assert args.acm_exit==0 and args.cgns_exit==0
assert 'ACM mesh/reconstruction initialized:' in log
assert q['minimum']>=.1 and q['cells_below']['0.1']==0
assert intersections['all_clear']
if 'first_20_layers_unchanged' in v:
    v['first_20_layers_unchanged_from_miter_candidate']=v.pop('first_20_layers_unchanged')
    (D/'mesh_validation.json').write_text(json.dumps(v,indent=2)+'\n')
cond=re.search(r'A cond Max:\s*\[\s*([^\]]+)',log)
report=dict(quality_goal=.1,quality_goal_achieved=True,orthogonal_quality=q,mesh=v,
            layer_intersections=intersections,cgnscheck_exit_code=args.cgns_exit,
            cgnscheck_metadata_warnings=4,acm_init_exit_code=args.acm_exit,
            mpi_ranks=12,acm_executable='acm2EQ3D.exe',turbulence_model='KOmegaSST',time_steps=0,
            reconstruction_order=2,reconstruction_condition_max_estimate=float(cond.group(1)) if cond else None,
            actual_flow_yplus_verified=False,
            boundary_layer_note='40 layers, nominal 10 micrometre first normal height; smooth spatial growth followed by local outer-layer redistribution. Growth is not uniformly 1.18; projected adjacent-increment ratio reaches 2.3652428 locally.',
            metric_note='Same face-normal cosine definition as the original mesh audit; not Fluent combined orthogonal quality.',
            original_mesh=str(args.original),case='cases/acm/3D/acm3D_suboff_orthogonal01_Re12000000_SST_init.json')
(D/'validation_summary.json').write_text(json.dumps(report,indent=2)+'\n')
rows=[('总体最低正交质量',f"{old['minimum']:.9f}",f"{q['minimum']:.9f}"),
      ('低于 0.1 的单元数',f"{old['cells_below']['0.1']:,}",f"{q['cells_below']['0.1']:,}"),
      ('四面体最低正交质量',f"{old['minimum_by_type']['TETRA_4']:.9f}",f"{q['minimum_by_type']['TETRA_4']:.9f}"),
      ('三棱柱最低正交质量',f"{old['minimum_by_type']['PENTA_6']:.9f}",f"{q['minimum_by_type']['PENTA_6']:.9f}"),
      ('总体单元数',f"{old['cells']:,}",f"{q['cells']:,}")]
table=''.join('<tr>'+''.join(f'<td>{html.escape(x)}</td>' for x in row)+'</tr>' for row in rows)
text=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>SUBOFF 网格正交质量优化</title>
<style>body{{font:16px/1.7 system-ui,sans-serif;max-width:1000px;margin:48px auto;padding:0 24px;color:#182331}}h1{{font-size:28px}}table{{border-collapse:collapse;width:100%}}td,th{{padding:10px 14px;text-align:left;border-bottom:1px solid #dce3e9}}.result{{padding:20px;background:#eaf6ef;border-radius:10px;font-size:22px}}code{{background:#f3f5f7;padding:2px 5px}}a{{color:#135ca5}}</style>
<h1>SUBOFF 网格正交质量优化</h1><p class="result">最低正交质量 {q['minimum']:.6f} ≥ 0.1；低于 0.1 的单元为 0。</p>
<p><a href="SUBOFF_optimized.cgns">下载 CGNS 网格（约 330 MiB）</a> · <a href="validation_summary.json">完整验证数据</a> · <a href="orthogonal_quality.json">正交质量数据</a></p>
<table><thead><tr><th>指标</th><th>原网格</th><th>优化后</th></tr></thead><tbody>{table}</tbody></table>
<h2>网格与边界层</h2><p>包含 {v['prisms']:,} 个三棱柱、{v['tetrahedra']:,} 个四面体和 {v['nodes']:,} 个节点。壁面坐标保持不变，围壳顶面与尾鳍保留完整的 40 层边界层。修正偏移方向后，首层标称法向厚度为 0.01 mm；全部壁面三角形顶点的实际投影范围为 {v['first_layer_normal_height_m_quantiles'][0]*1e3:.6f}–{v['first_layer_normal_height_m_quantiles'][-1]*1e3:.6f} mm。</p>
<p>增长率先作空间平滑，再对外侧 20 层作局部节点重分布，最终增长率不再统一为 1.18。沿初始列方向投影的层间厚度比中位数为 {v['projected_increment_growth_quantiles'][3]:.4f}，99.9% 分位为 {v['projected_increment_growth_quantiles'][-2]:.4f}，局部最大为 {v['projected_increment_growth_quantiles'][-1]:.4f}。这些比值的 {v['projected_growth_above_1_3']:,} 个样本超过 1.3。此处仅完成网格与零步初始化验证，没有收敛流场或实测 y+。</p>
<h2>验证</h2><p>全部体单元 Jacobian 为正；最小体积 {v['minimum_volume_m3']:.6g} m³；体积闭合相对误差 {v['relative_volume_closure_error']:.3g}。40 个层面分别与壁面检查，均无自交。CGNS 检查退出码 0（4 条 family/dataclass 元数据提示）；12 进程 ACM/SST 零步网格与二阶重构初始化退出码 0。</p>
<h2>指标定义与复算</h2><p>采用与原网格审计相同的指标：面法向与“单元体积中心至面心”及“相邻单元体积中心连线”夹角余弦的最小值，截断至 [0,1]。三棱柱采用三线性映射的体积中心，四边形采用全局一致对角线的面积加权面心。这不是 Fluent 内置的组合正交质量，不能直接当作 Fluent 的数值。</p>
<p><a href="REPRODUCE.sh">复算流程</a>依赖原几何修复目录中的表面数据；<a href="scripts/suboff_orthogonal_quality.py">质量计算脚本</a>和所有优化脚本已保存。质量数组 <code>orthogonal_quality_values.npy</code> 与 CGNS 单元顺序一致：先四面体、后三棱柱。</p></html>'''
(D/'quality_report.html').write_text(text)
files=['SUBOFF_optimized.cgns','SUBOFF_optimized.validated.npz','orthogonal_quality.json','orthogonal_quality_values.npy','mesh_validation.json','layer_intersections.json','validation_summary.json','quality_report.html']
(D/'CHECKSUMS.sha256').write_text(''.join(hashlib.sha256((D/f).read_bytes()).hexdigest()+'  '+f+'\n' for f in files))
print(json.dumps(dict(minimum=q['minimum'],cells=q['cells'],checks_passed=True,report=str(D/'quality_report.html'))))
