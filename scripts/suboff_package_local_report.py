#!/usr/bin/env python3
"""Package evidence for local SUBOFF appendage-layer optimization."""
import argparse
import hashlib
import html
import json
from pathlib import Path

ap=argparse.ArgumentParser()
ap.add_argument('directory',type=Path)
ap.add_argument('source',type=Path)
ap.add_argument('--baseline-audit',type=Path,required=True)
args=ap.parse_args();D=args.directory;S=args.source
read=lambda p:json.loads(p.read_text())
v=read(D/'mesh_validation.json');q=read(D/'orthogonal_quality.json')
a=read(D/'appendage_thickness_audit.json');b=read(args.baseline_audit/'appendage_thickness_audit.json')
intersections=read(D/'layer_intersections.json')
checks=read(D/'external_checks.json')
assert q['minimum']>=.2 and q['cells_below']['0.2']==0
assert v['first_layer_coordinates_unchanged'] and v['normal_growth_maximum_excess_over_allowance']<=0
assert v['maximum_projected_growth']<=1.25 and v['maximum_euclidean_growth']<=1.25
assert intersections['all_clear'] and checks['cgns_exit']==0 and checks['acm_exit']==0
assert 'ACM mesh/reconstruction initialized:' in (D/'acm_init.log').read_text()
source=S/'SUBOFF_30Lx20Lx20L_Q02_BLsmooth.cgns'
source_hash=hashlib.sha256(source.read_bytes()).hexdigest()
assert source_hash==read(S/'mesh_validation.json')['sha256']
mesh=next(D.glob('*.cgns'))
core=read(D/'core_generation.json') if (D/'core_generation.json').exists() else None
summary=dict(mesh=v,quality=q,thickness=a,intersections=intersections,core_generation=core,external_checks=checks,source_sha256=source_hash,source_preserved=True,uniform_total_thickness_achieved=False,remaining_issue='Sail trailing edge and fin tips remain thinner than the main hull; thicker candidates did not meet the full quality constraints.',actual_flow_yplus_verified=False)
(D/'validation_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
rows=[]
for key,name in [('whole_wall','全壁面'),('sail_roof','顶盖'),('roof_x_1.23_1.3','顶盖后缘'),('outer_fin_surface_excluding_tip','尾翼外露翼面'),('fin_tip_all','尾翼端面')]:
    old=b['regions'][key]['total_normal_projected_thickness_mm'];new=a['regions'][key]['total_normal_projected_thickness_mm']
    rows.append(f'<tr><td>{name}</td><td>{old["median"]:.3f}</td><td>{new["median"]:.3f}</td><td>{new["p05"]:.3f}–{new["p95"]:.3f}</td><td>{new["min"]:.3f}–{new["max"]:.3f}</td></tr>')
oldratio=b['regions']['fin_tip_all']['tetra_to_last_prism_normal_height_ratio']['median']
newratio=a['regions']['fin_tip_all']['tetra_to_last_prism_normal_height_ratio']['median']
page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>SUBOFF局部边界层优化</title>
<style>body{{font:16px/1.7 system-ui;max-width:1150px;margin:30px auto;padding:0 22px;color:#182331}}table{{border-collapse:collapse;width:100%}}td,th{{padding:9px;border-bottom:1px solid #ddd;text-align:left}}img{{max-width:100%}}</style>
<h1>SUBOFF局部边界层优化</h1>
<p>{v['cells']:,}个单元，{v['layers']}层棱柱；最低正交质量{q['minimum']:.9f}。计算域30L×20L×20L。</p>
<p><a href="{html.escape(mesh.name)}">CGNS网格</a> · <a href="validation_summary.json">完整验证记录</a> · <a href="appendage_thickness_audit.json">厚度审计</a></p>
<table><tr><th>区域</th><th>原中位厚度/mm</th><th>新中位厚度/mm</th><th>新P05–P95/mm</th><th>新最小–最大/mm</th></tr>{''.join(rows)}</table>
<p>厚度采用对应外缘到壁面位移在壁面三角形法向上的投影，分位数按壁面面积加权；极小值和极大值不代表整个区域。</p>
<p>保留原壁面及前{v['fixed_layers']}层坐标、主体边界层外包络和外域坐标。逐步加厚附体薄区，采用局部回退、核心网格调整和质量约束修复。全艇总厚度相近的目标尚未实现：顶盖后缘和尾翼端面仍比主体薄。更大增厚幅度的候选未通过完整质量约束，未交付；凹角处仍有压缩。</p>
<p>初始柱方向投影层高比最大{v['maximum_projected_growth']:.6f}，节点层距比最大{v['maximum_euclidean_growth']:.6f}，均≤1.25。按每层局部面法向计算的高度比最大{v['maximum_normal_growth']:.6f}，包含原网格已有拐角值；逐面约束为不超过1.25或原始值+0.0005，取较大者。此指标与初始柱投影增长率分别统计。</p>
<p>尾翼端面的相邻四面体法向高度/末层棱柱法向高度中位数由{oldratio:.3f}变为{newratio:.3f}；这是界面尺寸比较，不是棱柱增长率。核心是否新增单元详见生成记录。</p>
<h2>真实网格截面</h2><img src="sections/05_before_after_comparison.png"><p><a href="sections/02_longitudinal_sections.png">完整艇体截面</a> · <a href="sections/01_full_domain_z0.png">完整计算域截面</a></p>
<h2>验收</h2><p>完整混合网格质量与正Jacobian检查通过，体积闭合相对误差{v['relative_volume_closure_error']:.3g}，全部{v['layers']}个层面无自交；CGNS检查、{checks['mpi_ranks']}进程ACM/SST二阶重构零步初始化通过。首层坐标未改变，标称厚度10μm；实际局部厚度分布见JSON。</p>
<p>正交质量使用单元体积中心至面心和相邻单元体积中心连线与面法向的余弦最小值，截断到[0,1]；不等同于Fluent组合指标。尚未计算收敛流场或验证实际y+。原CGNS校验值保持{source_hash}。</p></html>'''
(D/'quality_report.html').write_text(page)
files=[mesh.name,'SUBOFF_optimized.validated.npz','prism_front.npz','mesh_validation.json','orthogonal_quality.json','orthogonal_quality_values.npy','appendage_thickness_audit.json','layer_intersections.json','external_checks.json','validation_summary.json','quality_report.html']
(D/'CHECKSUMS.sha256').write_text(''.join(hashlib.sha256((D/f).read_bytes()).hexdigest()+'  '+f+'\n' for f in files))
print(json.dumps(dict(cells=v['cells'],quality=q['minimum'],all_checks_passed=True)))
