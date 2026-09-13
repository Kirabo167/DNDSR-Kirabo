#!/usr/bin/env python3
"""Create a Word-ready HTML report from t=2 NCFV benchmark summary data."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def f(row: dict[str, str], field: str) -> float:
    return float(row[field])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--html", type=Path, required=True)
    args = parser.parse_args()
    if args.html.exists():
        raise FileExistsError(f"refusing to overwrite {args.html}")
    with args.summary.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 4:
        raise ValueError("expected exactly IV10/20/40/80")

    timing = "".join(
        "<tr>"
        f"<td>{r['mesh']}</td><td>{r['mpi_ranks']}</td><td>{int(r['nodes']):,}</td><td>{int(r['cells']):,}</td>"
        f"<td>{r['efficient_steps']}</td><td>{r['traditional_steps']}</td>"
        f"<td>{f(r, 'efficient_march_seconds'):.3f}</td><td>{f(r, 'traditional_march_seconds'):.3f}</td>"
        f"<td class='accent'>{f(r, 'traditional_to_efficient_march_ratio'):.2f}×</td>"
        "</tr>" for r in rows)
    initialization = "".join(
        "<tr>"
        f"<td>{r['mesh']}</td><td>{f(r, 'efficient_initialize_seconds'):.3f}</td>"
        f"<td>{f(r, 'traditional_initialize_seconds'):.3f}</td>"
        f"<td>{f(r, 'efficient_total_seconds'):.3f}</td><td>{f(r, 'traditional_total_seconds'):.3f}</td>"
        "</tr>" for r in rows)
    memory = "".join(
        "<tr>"
        f"<td>{r['mesh']}</td><td>{f(r, 'efficient_initial_pss_mib') / 1024:.3f}</td>"
        f"<td>{f(r, 'traditional_initial_pss_mib') / 1024:.3f}</td>"
        f"<td>{f(r, 'efficient_t2_pss_mib') / 1024:.3f}</td>"
        f"<td>{f(r, 'traditional_t2_pss_mib') / 1024:.3f}</td>"
        f"<td>{f(r, 'pss_saving_mib') / 1024:.3f}</td><td class='accent'>{f(r, 'pss_saving_percent'):.2f}%</td>"
        "</tr>" for r in rows)
    phase = "".join(
        "<tr>"
        f"<td>{r['mesh']}</td><td>{f(r, 'efficient_pss_growth_mib'):.3f}</td>"
        f"<td>{f(r, 'traditional_pss_growth_mib'):.3f}</td>"
        f"<td>{f(r, 'efficient_t2_hwm_mib') / 1024:.3f}</td>"
        f"<td>{f(r, 'traditional_t2_hwm_mib') / 1024:.3f}</td>"
        "</tr>" for r in rows)
    document = f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>
<title>NCFV 推进至 t=2 的时间与内存对比</title><style>
@page {{ size:A4 landscape; margin:1.25cm 1.4cm; }}
body {{ font-family:'Noto Sans CJK SC','Microsoft YaHei','SimSun',sans-serif; color:#1f2937; font-size:9.5pt; line-height:1.18; }}
h1 {{ text-align:center; color:#17365d; font-size:18pt; margin:0 0 4pt; }} h2 {{ color:#17365d; font-size:13pt; border-bottom:1.2pt solid #2f75b5; padding-bottom:2pt; margin:13pt 0 4pt; }}
p {{ margin:3pt 0; }} .subtitle {{ text-align:center; color:#4b5563; margin-bottom:9pt; }} .note {{ background:#f3f7fb; border-left:3pt solid #2f75b5; padding:5pt 7pt; margin:6pt 0; font-size:8.7pt; }}
table {{ border-collapse:collapse; width:100%; table-layout:fixed; margin:3pt 0 8pt; }} th,td {{ border:.6pt solid #9aa7b6; padding:3pt 4pt; text-align:center; vertical-align:middle; font-size:8.5pt; }} th {{ background:#17365d; color:white; }} td.accent {{ color:#1b5e20; font-weight:bold; }} .source {{ color:#6b7280; font-size:7.7pt; }}
</style></head><body>
<h1>NCFV 传统算法与高效算法：推进至 t=2 的计算时间与内存占用对比</h1>
<p class='subtitle'>三维等熵涡 · 原始 IV 三棱柱网格（Prism6）· IV10 / IV20 / IV40 / IV80</p>
<div class='note'><b>测试口径：</b>每个 MPI rank 固定为单线程（OMP/BLAS=1）。计时使用生产求解器 <code>Solver::Run()</code>，从 t=0 以 CFL=0.5 的 SSPRK3 真正推进至 t=2；每一物理步包含 3 次 RHS、状态更新和 CFL 步长选择。为排除文件系统影响，VTK、重启和诊断文件输出均关闭。下表“推进时间”不含初始化；PSS、RSS、HWM 均为该行所有 rank 的聚合值。IV10–IV40 使用 8 rank；为使 IV80 完整实测不超额订阅，IV80 使用 32 rank；每一行中两种算法的 MPI 数完全相同。</div>
<h2>1. 实际推进计算时间</h2><table><thead><tr><th>网格</th><th>MPI<br>rank</th><th>原始节点</th><th>体单元</th><th>高效法<br>物理步数</th><th>传统法<br>物理步数</th><th>高效法<br>0→2 时间（s）</th><th>传统法<br>0→2 时间（s）</th><th>传统÷高效</th></tr></thead><tbody>{timing}</tbody></table>
<h2>2. 初始化与端到端时间</h2><table><thead><tr><th>网格</th><th>高效法<br>初始化（s）</th><th>传统法<br>初始化（s）</th><th>高效法<br>初始化+推进（s）</th><th>传统法<br>初始化+推进（s）</th></tr></thead><tbody>{initialization}</tbody></table>
<h2>3. 内存占用（聚合 PSS）</h2><table><thead><tr><th>网格</th><th>高效法：初始化后<br>PSS（GiB）</th><th>传统法：初始化后<br>PSS（GiB）</th><th>高效法：t=2<br>PSS（GiB）</th><th>传统法：t=2<br>PSS（GiB）</th><th>t=2 PSS 节省<br>（GiB）</th><th>PSS 节省</th></tr></thead><tbody>{memory}</tbody></table>
<h2>4. 推进过程内存监控</h2><table><thead><tr><th>网格</th><th>高效法：初始化后→t=2<br>PSS 增量（MiB）</th><th>传统法：初始化后→t=2<br>PSS 增量（MiB）</th><th>高效法：t=2<br>聚合 HWM（GiB）</th><th>传统法：t=2<br>聚合 HWM（GiB）</th></tr></thead><tbody>{phase}</tbody></table>
<div class='note'><b>说明：</b>高效法与传统法依据各自空间离散的 CFL 谱半径选择步长，故个别网格的物理步数可相差 1；两者均精确到达 t=2。HWM 是每个 rank 在整个进程生命周期的峰值 RSS 之和；PSS 是结束时实际分摊物理内存，适合比较共享库页后的真实占用。</div>
<p class='source'>原始数据：/tmp/ncfv-t2-original-np8-20260913/*.json；生成日期：2026-09-13。</p></body></html>"""
    args.html.write_text(document, encoding="utf-8")
    print(args.html)


if __name__ == '__main__':
    main()
