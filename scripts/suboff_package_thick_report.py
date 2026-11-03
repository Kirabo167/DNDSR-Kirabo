#!/usr/bin/env python3
"""Package validated layer thickening and transition-refinement evidence."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import shutil

ap=argparse.ArgumentParser();ap.add_argument('directory',type=Path);ap.add_argument('source',type=Path)
ap.add_argument('--acm-exit',type=int,required=True);ap.add_argument('--cgns-exit',type=int,required=True)
args=ap.parse_args();D=args.directory;S=args.source
read=lambda p:json.loads(p.read_text())
q=read(D/'orthogonal_quality.json');v=read(D/'mesh_validation.json');audit=read(D/'appendage_thickness_audit.json')
old=read(S/'mesh_validation.json');before=read(S/'appendage_thickness_audit.json')
intersections=read(D/'layer_intersections.json');core=read(D.parent/'core_generation.json')
assert q['minimum']>=.2 and q['cells_below']['0.2']==0
assert max(v['maximum_normal_growth'],v['maximum_euclidean_growth'])<=1.25
assert v['cells']<10000000 and v['first_layer_mean_normal_height_preserved'] and intersections['all_clear']
assert args.acm_exit==0 and args.cgns_exit==0
assert 'ACM mesh/reconstruction initialized:' in (D/'acm_init.log').read_text()
source_file=S/'SUBOFF_30Lx20Lx20L_Q02_BLsmooth.cgns'
source_hash=hashlib.sha256(source_file.read_bytes()).hexdigest();assert source_hash==old['sha256']
summary=dict(mesh=v,orthogonal_quality=q,appendage_thickness=audit,core_generation=core,
             layer_intersections=intersections,source_file=str(source_file),source_sha256=source_hash,
             source_preserved=True,acm_init_exit_code=args.acm_exit,cgnscheck_exit_code=args.cgns_exit,
             cgnscheck_warnings=[line for line in (D/'cgnscheck.log').read_text().splitlines() if line.startswith('WARNING:')],
             mpi_ranks=12,turbulence_model='KOmegaSST',reconstruction_order=2,time_steps=0,actual_flow_yplus_verified=False)
(D/'validation_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
rows=[]
for key,name in [('sail_roof','顶盖顶面'),('roof_x_1.23_1.3','顶盖后缘'),('outer_fin_surface_excluding_tip','尾翼外露翼面'),('fin_tip_all','尾翼端面')]:
    a=before['regions'][key]['total_normal_projected_thickness_mm'];b=audit['regions'][key]['total_normal_projected_thickness_mm']
    rows.append((name,f"{a['median']:.3f}",f"{b['median']:.3f}",f"{b['min']:.3f}–{b['max']:.3f}"))
table=''.join('<tr>'+''.join('<td>'+html.escape(x)+'</td>' for x in row)+'</tr>' for row in rows)
whole=audit['regions']['whole_wall'];tips=audit['regions']['fin_tip_all'];oldtips=before['regions']['fin_tip_all']
shutil.copyfile(S/'sections/03_boundary_layer_details.png',D/'sections/00_before_boundary_layer_details.png')
page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>SUBOFF 边界层加厚与过渡区加密</title>
<style>body{{font:16px/1.7 system-ui;max-width:1150px;margin:36px auto;padding:0 24px;color:#182331}}table{{border-collapse:collapse;width:100%}}td,th{{padding:10px;border-bottom:1px solid #ddd;text-align:left}}img{{max-width:100%}}a{{color:#1262a1}}</style>
<h1>SUBOFF 边界层加厚与过渡区加密</h1>
<p>{v['cells']:,} 个单元；{v['layers']} 层；最低正交质量 {q['minimum']:.9f}；最大局部法向层高比 {v['maximum_normal_growth']:.6f}，最大节点层距比 {v['maximum_euclidean_growth']:.6f}。</p>
<p><a href="SUBOFF_30Lx20Lx20L_Q02_BL64.cgns">CGNS 网格</a> · <a href="validation_summary.json">完整验证</a> · <a href="appendage_thickness_audit.json">厚度与界面尺寸审计</a></p>
<table><tr><th>区域</th><th>原中位厚度/mm</th><th>新中位厚度/mm</th><th>新最小–最大/mm</th></tr>{table}</table>
<p>总厚度按对应外缘与壁面点位移的局部壁面法向投影统计，中位数按壁面面积加权。尾翼外露翼面统计不含翼根。全艇壁面有 {100*whole['area_fraction_total_thickness_25_to_40mm']:.3f}% 的面积处于 25–40 mm；凹角和狭窄连接区保留必要压缩，未强行达到等厚。详细区域筛选见审计 JSON。</p>
<h2>网格调整</h2><p>壁面与首层原坐标完全保留，首层标称厚度仍为 10 μm，相对原网格的最大平均法向厚度变化为 {v['first_layer_mean_normal_height_max_change_m']*1e9:.3f} nm。采用相邻层面法向拟合、切向平滑和逐层质量修复构造外层，每一层通过正交质量、精确 Jacobian、层距及自交检查后再推进。计算域保持 30L×20L×20L（130.68×87.12×87.12 m）。</p>
<p>增长率按相邻层面局部法向上的平均层高之比统计，同时约束节点间实际层距比；弯曲列在固定初始方向上的投影仅作为诊断保留于 JSON。</p>
<p>针对外缘四面体法向高度超过最外层棱柱 4 倍的位置，插入 {core['interface_refinement_points']:,} 个内部控制点重新剖分，随后联合优化棱柱和四面体。尾翼端面的界面法向高度比中位数由 {oldtips['tetra_to_last_prism_normal_height_ratio']['median']:.3f} 变为 {tips['tetra_to_last_prism_normal_height_ratio']['median']:.3f}。该比值是形状相关的尺寸比较，不是相邻棱柱层增长率。</p>
<h2>优化前后截面</h2><img src="sections/05_before_after_comparison.png">
<p><a href="sections/02_longitudinal_sections.png">完整纵向截面</a> · <a href="sections/01_full_domain_z0.png">完整计算域截面</a></p>
<h2>验证</h2><p>全体单元 Jacobian 为正；体积闭合相对误差 {v['relative_volume_closure_error']:.3g}；{v['layers']} 个完整层面分别与壁面检查均无自交；CGNS 检查和 12 进程 ACM/SST 二阶重构零步初始化通过。CGNS 如有元数据提示，均记录于验证 JSON 和检查日志。</p>
<p>正交质量沿用面法向与单元体积中心至面心、相邻单元体积中心连线夹角余弦的最小值，截断到 [0,1]；不等同于 Fluent 组合正交质量。未计算收敛流场或验证实际 y+。原网格文件保持不变，SHA-256：<code>{source_hash}</code>。</p></html>'''
(D/'quality_report.html').write_text(page)
files=['SUBOFF_30Lx20Lx20L_Q02_BL64.cgns','SUBOFF_optimized.validated.npz','mesh_validation.json',
       'orthogonal_quality.json','orthogonal_quality_values.npy','appendage_thickness_audit.json',
       'layer_intersections.json','validation_summary.json','quality_report.html']
(D/'CHECKSUMS.sha256').write_text(''.join(hashlib.sha256((D/f).read_bytes()).hexdigest()+'  '+f+'\n' for f in files))
print(json.dumps(dict(cells=v['cells'],minimum_quality=q['minimum'],wall_fraction_25_to_40mm=whole['area_fraction_total_thickness_25_to_40mm'],all_checks_passed=True)))
