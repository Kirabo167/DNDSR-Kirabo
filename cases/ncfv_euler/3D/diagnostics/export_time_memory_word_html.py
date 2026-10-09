#!/usr/bin/env python3
"""Render the paired NCFV timing/memory CSV into a Word-ready HTML document."""

from __future__ import annotations

import argparse
import csv
import html
from pathlib import Path


FAMILY_ORDER = ("prism", "hex", "tet")
FAMILY_NAMES = {
    "prism": "原 IV 三棱柱网格（Prism6）",
    "hex": "纯六面体网格（Hex8）",
    "tet": "纯四面体网格（Tet4）",
}


def value(row: dict[str, str], name: str) -> float:
    return float(row[name])


def timing_table(rows: list[dict[str, str]]) -> str:
    body = []
    for row in rows:
        body.append(
            "<tr>"
            f"<td>IV{row['size']}</td><td>{row['mpi_ranks']}</td>"
            f"<td>{int(row['nodes']):,}</td><td>{int(row['cells']):,}</td>"
            f"<td>{value(row, 'efficient_rhs_ms'):.3f}</td>"
            f"<td>{value(row, 'traditional_rhs_ms'):.3f}</td>"
            f"<td class=\"accent\">{value(row, 'traditional_to_efficient_rhs_ratio'):.2f}×</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr>"
        "<th>网格</th><th>MPI<br>进程</th><th>原始节点</th><th>体单元</th>"
        "<th>高效算法<br>RHS（ms）</th><th>传统算法<br>RHS（ms）</th><th>传统÷高效</th>"
        "</tr></thead><tbody>" + "".join(body) + "</tbody></table>"
    )


def memory_table(rows: list[dict[str, str]]) -> str:
    body = []
    for row in rows:
        efficient = value(row, "efficient_final_pss_sum_mib") / 1024.0
        traditional = value(row, "traditional_final_pss_sum_mib") / 1024.0
        saving = value(row, "pss_saving_mib") / 1024.0
        body.append(
            "<tr>"
            f"<td>IV{row['size']}</td><td>{row['mpi_ranks']}</td>"
            f"<td>{efficient:.3f}</td><td>{traditional:.3f}</td>"
            f"<td>{saving:.3f}</td><td class=\"accent\">{value(row, 'pss_saving_percent'):.2f}%</td>"
            f"<td>{value(row, 'traditional_quadrature_points'):,}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr>"
        "<th>网格</th><th>MPI<br>进程</th><th>高效法<br>PSS（GiB）</th>"
        "<th>传统法<br>PSS（GiB）</th><th>PSS 节省<br>（GiB）</th>"
        "<th>PSS 节省</th><th>传统法高斯点总数</th>"
        "</tr></thead><tbody>" + "".join(body) + "</tbody></table>"
    )


def phase_table(rows: list[dict[str, str]]) -> str:
    body = []
    for row in rows:
        body.append(
            "<tr>"
            f"<td>{html.escape(FAMILY_NAMES[row['family']])}</td><td>{row['mpi_ranks']}</td>"
            f"<td>{value(row, 'efficient_after_initialize_pss_sum_mib') / 1024.0:.3f}</td>"
            f"<td>{value(row, 'efficient_after_measurement_pss_sum_mib') / 1024.0:.3f}</td>"
            f"<td>{value(row, 'traditional_after_initialize_pss_sum_mib') / 1024.0:.3f}</td>"
            f"<td>{value(row, 'traditional_after_measurement_pss_sum_mib') / 1024.0:.3f}</td>"
            f"<td>{value(row, 'traditional_final_hwm_sum_mib') / 1024.0:.3f}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr>"
        "<th>IV80 网格类型</th><th>MPI<br>进程</th>"
        "<th>高效法：初始化后<br>PSS（GiB）</th><th>高效法：测量后<br>PSS（GiB）</th>"
        "<th>传统法：初始化后<br>PSS（GiB）</th><th>传统法：测量后<br>PSS（GiB）</th>"
        "<th>传统法聚合<br>HWM（GiB）</th>"
        "</tr></thead><tbody>" + "".join(body) + "</tbody></table>"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--html", type=Path, required=True)
    args = parser.parse_args()
    if args.html.exists():
        raise FileExistsError(f"refusing to overwrite {args.html}")
    with args.summary.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 24:
        raise ValueError(f"expected 24 paired records, found {len(rows)}")

    grouped = {
        family: sorted((row for row in rows if row["family"] == family),
                       key=lambda row: (int(row["size"]), int(row["mpi_ranks"])))
        for family in FAMILY_ORDER
    }
    if any(len(grouped[family]) != 8 for family in FAMILY_ORDER):
        raise ValueError("each topology must contain IV10/20/40/80 at 1 and 8 MPI ranks")
    iv80 = [row for row in rows if row["size"] == "80"]
    iv80.sort(key=lambda row: (FAMILY_ORDER.index(row["family"]), int(row["mpi_ranks"])))

    family_sections = []
    for family in FAMILY_ORDER:
        family_sections.append(
            f"<h2>{html.escape(FAMILY_NAMES[family])}</h2>"
            "<h3>计算时间对比</h3>" + timing_table(grouped[family]) +
            "<h3>最终内存占用对比</h3>" + memory_table(grouped[family])
        )

    document = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="author" content="DNDSR NCFV benchmark">
<title>NCFV 计算时间与内存占用对比</title>
<style>
@page { size: A4 landscape; margin: 1.15cm 1.35cm; }
body { font-family: "Noto Sans CJK SC", "Microsoft YaHei", "SimSun", sans-serif; color: #1f2937; font-size: 9.2pt; line-height: 1.15; }
h1 { color: #17365d; font-size: 18pt; text-align: center; margin: 0 0 4pt 0; }
h2 { color: #17365d; font-size: 13pt; margin: 12pt 0 3pt 0; border-bottom: 1.2pt solid #2f75b5; padding-bottom: 2pt; }
h3 { color: #365f91; font-size: 10.5pt; margin: 7pt 0 2pt 0; }
p { margin: 2pt 0; }
.subtitle { text-align: center; color: #4b5563; font-size: 9pt; margin-bottom: 9pt; }
.note { background: #f3f7fb; border-left: 3pt solid #2f75b5; padding: 5pt 7pt; margin: 6pt 0; font-size: 8.5pt; }
table { border-collapse: collapse; width: 100%; table-layout: fixed; margin: 0 0 6pt 0; }
th, td { border: 0.6pt solid #9aa7b6; padding: 2pt 3pt; text-align: center; vertical-align: middle; line-height: 1.05; font-size: 8.2pt; }
th { background: #17365d; color: white; font-weight: bold; }
td.accent { color: #1b5e20; font-weight: bold; }
.pagebreak { page-break-before: always; }
.source { color: #6b7280; font-size: 7.5pt; margin-top: 7pt; }
</style></head><body>
<h1>NCFV 传统算法与高效算法：计算时间与内存占用对比</h1>
<p class="subtitle">三维等熵涡周期算例 · IV10 / IV20 / IV40 / IV80 · 2026-09-12 至 2026-09-13</p>
<div class="note"><b>测试口径：</b>时间为预热后一次空间离散 RHS 的中位耗时，MPI 条件报告各 rank 的最大耗时；PSS 为全部 rank 的聚合实际物理内存。构建为 Release（-O3，NDEBUG），所有 rank 的 OMP/BLAS 线程数均固定为 1。Tet4 每个规则六面体块一致剖分为 6 个四面体，因此与 Hex8 同 IV 编号时具有相同的周期未知量数 N³、但有 6 倍体单元。</div>
""" + "".join(family_sections) + """
<div class="pagebreak"></div>
<h2>IV80 的计算过程内存监控</h2>
<p>每项测试均在初始化前、初始化后、一次预热 RHS 后和全部计时 RHS 后采集 RSS、PSS 与 HWM。全部 48 个条件中，初始化后至测量结束的聚合 PSS 最大增量为 4.184 MiB，未观察到重复 RHS 调用导致的持续内存增长。</p>
""" + phase_table(iv80) + """
<div class="note"><b>验证结果：</b>48 个结果全部成功；rank 间最终残差差异为 0；高效法高斯积分点总数为 0，传统法积分点总数均为正。最大传统法聚合 HWM 出现在 Tet4-IV80、8 MPI 进程，为 65.847 GiB；其最终聚合 PSS 为 64.787 GiB。</div>
<p class="source">数据来源：/tmp/ncfv-tet-hex-benchmark-20260913/analysis/summary.csv；原 IV 三棱柱数据来自 /tmp/ncfv-revalidation-WghkB3/raw。</p>
</body></html>
"""
    args.html.parent.mkdir(parents=True, exist_ok=True)
    args.html.write_text(document, encoding="utf-8")
    print(args.html)


if __name__ == "__main__":
    main()
