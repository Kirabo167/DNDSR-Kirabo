#!/usr/bin/env python3
"""Package measured boundary-layer smoothing results and completed checks."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import shutil
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('directory', type=Path)
ap.add_argument('source', type=Path)
ap.add_argument('--acm-exit', type=int, required=True)
ap.add_argument('--cgns-exit', type=int, required=True)
args = ap.parse_args()
D, S = args.directory, args.source
read = lambda p: json.loads(p.read_text())
q = read(D/'orthogonal_quality.json')
v = read(D/'mesh_validation.json')
b = read(D/'boundary_layer_smoothing.json')
i = read(D/'layer_intersections.json')
old = read(S/'mesh_validation.json')
oldq = read(S/'orthogonal_quality.json')
source_arrays=np.load(S/'SUBOFF_optimized.validated.npz')
nv=len(np.load(S/'prism_front.npz')['points'])
increments=np.linalg.norm(np.diff(source_arrays['points'][:41*nv].reshape(41,nv,3),axis=0),axis=2)
old_euclidean_max=float((increments[1:]/increments[:-1]).max())
source_file = S/'SUBOFF_30Lx20Lx20L_Q02.cgns'
source_hash = hashlib.sha256(source_file.read_bytes()).hexdigest()
assert source_hash == '011ecb428dd242488b94e5998955000139550038fecb82afc87d3e7c9cb7fb81'
assert q['minimum'] >= .2 and q['cells_below']['0.2'] == 0
assert b['maximum_projected_growth'] <= 1.25 and b['maximum_euclidean_growth'] <= 1.25
assert b['all_frozen_coordinates_unchanged'] and i['all_clear']
assert args.acm_exit == 0 and args.cgns_exit == 0
assert 'ACM mesh/reconstruction initialized:' in (D/'acm_init.log').read_text()
assert v['cells'] == old['cells']
report = dict(mesh=v, orthogonal_quality=q, boundary_layer_smoothing=b,
              layer_intersections=i, source_file=str(source_file), source_sha256=source_hash,
              source_preserved=True, acm_init_exit_code=args.acm_exit,
              cgnscheck_exit_code=args.cgns_exit, mpi_ranks=12,
              cgnscheck_warnings=[line for line in (D/'cgnscheck.log').read_text().splitlines() if line.startswith('WARNING:')],
              acm_executable='acm2EQ3D.exe', turbulence_model='KOmegaSST', time_steps=0,
              actual_flow_yplus_verified=False,
              metric_note='Exact-volume-centroid face-normal cosine; not Fluent combined orthogonal quality.')
(D/'validation_summary.json').write_text(json.dumps(report, indent=2)+'\n')
rows = [
    ('体单元数', f"{old['cells']:,}", f"{v['cells']:,}"),
    ('最低正交质量', f"{oldq['minimum']:.9f}", f"{q['minimum']:.9f}"),
    ('最大投影层厚比', f"{old['projected_increment_growth_quantiles'][-1]:.6f}", f"{b['maximum_projected_growth']:.6f}"),
    ('最大位移长度比', f"{old_euclidean_max:.6f}", f"{b['maximum_euclidean_growth']:.6f}"),
    ('总厚度空间斜率 P95', f"{b['envelope_slope_quantiles_before'][1]:.6f}", f"{b['envelope_slope_quantiles_after'][1]:.6f}"),
    ('总厚度空间斜率 P99', f"{b['envelope_slope_quantiles_before'][2]:.6f}", f"{b['envelope_slope_quantiles_after'][2]:.6f}"),
]
table = ''.join('<tr>'+''.join('<td>'+html.escape(x)+'</td>' for x in row)+'</tr>' for row in rows)
shutil.copyfile(S/'sections/03_boundary_layer_details.png', D/'sections/00_before_boundary_layer_details.png')
page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>SUBOFF 边界层平滑优化</title><style>body{{font:16px/1.7 system-ui;max-width:1100px;margin:35px auto;padding:0 24px;color:#182331}}table{{border-collapse:collapse;width:100%}}td,th{{padding:10px;border-bottom:1px solid #ddd;text-align:left}}img{{max-width:100%}}a{{color:#1262a1}}</style>
<h1>SUBOFF 边界层平滑优化与实测验证</h1>
<p>最低正交质量 {q['minimum']:.9f}；两种层厚定义下的最大相邻层增长率均小于 1.25。</p>
<p><a href="SUBOFF_30Lx20Lx20L_Q02_BLsmooth.cgns">下载 CGNS</a> · <a href="validation_summary.json">完整验证数据</a> · <a href="boundary_layer_smoothing.json">边界层统计</a></p>
<table><tr><th>指标</th><th>优化前</th><th>优化后</th></tr>{table}</table>
<h2>调整方法与保留约束</h2><p>在壁面上对等效总厚度的对数作带约束扩散，保留凹角处必要的压缩；固定壁面及前 {b['frozen_layers']} 层全部节点，重分布外层并联合优化四面体过渡区。总厚度仍随位置变化，不能将所有拐角强制挤出为等厚层。网格保持 40 层、{v['prisms']:,} 个三棱柱和 {v['tetrahedra']:,} 个四面体，计算域保持 30L×20L×20L（130.68×87.12×87.12 m）。标称首层法向厚度 10 μm，实际首层分辨率未改变。</p>
<p>空间斜率统计采用初始列方向夹角余弦大于 0.98 的相邻壁面节点，对等效总厚度差除以壁面边长；等效厚度将列方向总投影按原首层位移归一到 10 μm。这是同一定义下的前后比较，实际数值及围壳顶面子集见 JSON。层增长率分别按初始列方向投影厚度比和节点位移长度比统计。</p>
<h2>优化前</h2><img src="sections/00_before_boundary_layer_details.png">
<h2>优化后</h2><img src="sections/03_boundary_layer_details.png">
<h2>同视窗前后对比</h2><img src="sections/05_before_after_comparison.png">
<p><a href="sections/02_longitudinal_sections.png">纵向截面</a> · <a href="sections/04_transverse_sections.png">横向截面</a></p>
<h2>验证范围</h2><p>全体单元精确 Jacobian 为正，体积闭合相对误差 {v['relative_volume_closure_error']:.3g}。40 个完整层面分别与壁面检查均无自交；CGNS 检查及 12 进程 ACM/SST 二阶重构零步初始化均通过。未进行流场收敛计算，尚无实际 y+ 验证。</p>
<p>CGNS 检查无错误，有 4 项元数据提示：区域未声明 FamilyName、三个坐标未声明 DataClass，具体记录见检查日志。</p>
<p>正交质量沿用面法向与单元体积中心至面心、相邻单元体积中心连线夹角余弦的最小值，截断到 [0,1]；这与 Fluent 的组合正交质量并不等价。原网格 SHA-256 校验保持不变：<code>{source_hash}</code>。</p></html>'''
(D/'quality_report.html').write_text(page)
files = ['SUBOFF_30Lx20Lx20L_Q02_BLsmooth.cgns', 'SUBOFF_optimized.validated.npz',
         'orthogonal_quality.json', 'orthogonal_quality_values.npy', 'mesh_validation.json',
         'boundary_layer_smoothing.json', 'layer_intersections.json', 'validation_summary.json', 'quality_report.html']
(D/'CHECKSUMS.sha256').write_text(''.join(hashlib.sha256((D/f).read_bytes()).hexdigest()+'  '+f+'\n' for f in files))
print(json.dumps(dict(minimum=q['minimum'], maximum_growth=b['maximum_euclidean_growth'], cells=v['cells'], checks_passed=True)))
