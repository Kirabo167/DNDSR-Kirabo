#!/usr/bin/env python3
"""Package measured enlarged-domain quality, provenance and solver checks."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import re

ap=argparse.ArgumentParser();ap.add_argument('directory',type=Path)
ap.add_argument('--acm-exit',type=int,required=True);ap.add_argument('--cgns-exit',type=int,required=True)
args=ap.parse_args();D=args.directory;W=D.parent
read=lambda path:json.loads(path.read_text())
q=read(D/'orthogonal_quality.json');v=read(D/'mesh_validation.json');info=v['domain']
old=read(Path(info['source'])/'orthogonal_quality.json')
intersections=read(D/'layer_intersections.json');log=(D/'acm_init.log').read_text()
assert args.acm_exit==0 and args.cgns_exit==0 and intersections['all_clear']
assert 'ACM mesh/reconstruction initialized:' in log
assert q['minimum']>=.2 and q['cells_below']['0.2']==0
source_file=Path(info['source'])/'SUBOFF_optimized.cgns'
source_hash=hashlib.sha256(source_file.read_bytes()).hexdigest()
assert source_hash==info['source_cgns_sha256']
cond=re.search(r'A cond Max:\s*\[\s*([^\]]+)',log)
report=dict(quality_goal=.2,quality_goal_achieved=True,orthogonal_quality=q,mesh=v,
            source_original_file_preserved=True,source_file=str(source_file),source_sha256=source_hash,
            layer_intersections=intersections,cgnscheck_exit_code=args.cgns_exit,acm_init_exit_code=args.acm_exit,
            mpi_ranks=12,acm_executable='acm2EQ3D.exe',turbulence_model='KOmegaSST',time_steps=0,
            reconstruction_order=2,reconstruction_condition_max_estimate=float(cond.group(1)) if cond else None,
            actual_flow_yplus_verified=False,
            metric_note='Same exact-volume-centroid face-normal cosine definition as previous audits; not Fluent combined orthogonal quality.',
            case='cases/acm/3D/acm3D_suboff_domain30x20x20_Q02_Re12000000_SST_init.json')
(D/'validation_summary.json').write_text(json.dumps(report,indent=2)+'\n')
rows=[('计算域尺寸 / L','10 × 6 × 6','30 × 20 × 20'),
      ('总体单元数',f"{old['cells']:,}",f"{q['cells']:,}"),
      ('最低正交质量',f"{old['minimum']:.9f}",f"{q['minimum']:.9f}"),
      ('低于 0.2 的单元数',str(old['cells_below']['0.2']),str(q['cells_below']['0.2'])),
      ('四面体最低质量',f"{old['minimum_by_type']['TETRA_4']:.9f}",f"{q['minimum_by_type']['TETRA_4']:.9f}"),
      ('三棱柱最低质量',f"{old['minimum_by_type']['PENTA_6']:.9f}",f"{q['minimum_by_type']['PENTA_6']:.9f}")]
table=''.join('<tr>'+''.join(f'<td>{html.escape(x)}</td>' for x in row)+'</tr>' for row in rows)
lo,hi=info['bounds_m']
text=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>SUBOFF 30L × 20L × 20L 网格</title>
<style>body{{font:16px/1.7 system-ui,sans-serif;max-width:1000px;margin:44px auto;padding:0 24px;color:#182331}}h1{{font-size:28px}}table{{border-collapse:collapse;width:100%}}td,th{{padding:10px 14px;text-align:left;border-bottom:1px solid #dce3e9}}.result{{padding:20px;background:#eaf6ef;border-radius:10px;font-size:22px}}a{{color:#135ca5}}code{{background:#f3f5f7;padding:2px 5px}}</style>
<h1>SUBOFF 扩大计算域与正交质量优化</h1><p class="result">最低正交质量 {q['minimum']:.9f} ≥ 0.2；低于 0.2 的单元数为 0。</p>
<p><a href="SUBOFF_30Lx20Lx20L_Q02.cgns">CGNS 网格</a> · <a href="validation_summary.json">完整验证数据</a> · <a href="orthogonal_quality.json">正交质量统计</a></p>
<table><tr><th>指标</th><th>原网格</th><th>新网格</th></tr>{table}</table>
<h2>计算域与外围网格</h2><p>艇长 L = 4.356 m；新域为 130.68 m × 87.12 m × 87.12 m。X 范围 {lo[0]:.3f}–{hi[0]:.3f} m，Y、Z 均为 −43.56–43.56 m。入口距艇首 {info['upstream_clearance_L']:.6f}L，出口距艇尾 {info['downstream_clearance_L']:.6f}L；两侧及上下边界距中心轴各 10L。</p>
<p>原域外表面作为内部共形接口，在外围新增 {info['added_tetrahedra']:,} 个逐渐变粗的四面体，总量仅增加 {info['increase_percent']:.4f}%。原有体单元连接保持，随后对不满足 0.2 的单元及其邻域作局部节点优化。新网格包含 {v['tetrahedra']:,} 个四面体、{v['prisms']:,} 个三棱柱和 {v['nodes']:,} 个节点。</p>
<h2>原文件与边界层</h2><p>原 CGNS 文件保留且 SHA-256 校验未改变：<code>{source_hash}</code>。新网格壁面坐标固定，围壳顶面和尾鳍保留 40 层边界层；靠壁第一层节点的位移约束为垂直于初始列方向，原首层沿该方向的投影厚度保持不变，标称法向厚度约 0.01 mm。</p>
<p>按各壁面三角形法向投影统计，新网格首层实际厚度范围为 {v['first_layer_normal_height_m_quantiles'][0]*1e3:.6f}–{v['first_layer_normal_height_m_quantiles'][-1]*1e3:.6f} mm。增长率仍为空间变化量：相邻层厚度沿初始列方向投影之比的中位数 {v['projected_increment_growth_quantiles'][3]:.4f}，99.9% 分位 {v['projected_increment_growth_quantiles'][-2]:.4f}，最大 {v['projected_increment_growth_quantiles'][-1]:.4f}。没有收敛流场或实际 y+ 验证。</p>
<h2>验证</h2><p>全部体单元 Jacobian 为正，最小体积 {v['minimum_volume_m3']:.6g} m³，体积闭合相对误差 {v['relative_volume_closure_error']:.3g}；40 个完整层面分别与壁面检查均无自交；CGNS 检查退出码 0；12 进程 ACM/SST 网格与二阶重构零步初始化退出码 0。</p>
<h2>指标与复算</h2><p>指标沿用前述审计定义：面法向与“单元体积中心至面心”及内部面“相邻单元体积中心连线”夹角余弦的最小值，截断至 [0,1]。三棱柱采用三线性映射体积中心，四边形采用全局一致对角线的面积加权面心。这不是 Fluent 内置的组合正交质量。</p>
<p><a href="REPRODUCE.sh">复算流程</a>使用保留的原网格作为输入。质量数组 <code>orthogonal_quality_values.npy</code> 按 CGNS 体单元编号排列：先四面体、后三棱柱。</p></html>'''
(D/'quality_report.html').write_text(text)
files=['SUBOFF_30Lx20Lx20L_Q02.cgns','SUBOFF_optimized.validated.npz','orthogonal_quality.json','orthogonal_quality_values.npy','mesh_validation.json','layer_intersections.json','validation_summary.json','quality_report.html']
(D/'CHECKSUMS.sha256').write_text(''.join(hashlib.sha256((D/f).read_bytes()).hexdigest()+'  '+f+'\n' for f in files))
print(json.dumps(dict(minimum=q['minimum'],cells=q['cells'],added_cells=info['added_tetrahedra'],checks_passed=True)))
