#!/usr/bin/env python3
"""Validate and report single-thread NCFV integration-initialization timings."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import html
import json
from pathlib import Path


FAMILIES = {
    "prism": ("三棱柱", "Prism6"),
    "hex": ("六面体", "Hex8"),
    "tet": ("四面体", "Tet4"),
}
IVS = (10, 20, 40, 80)
MODES = {
    "efficient": "EfficientDifferential",
    "traditional": "TraditionalQuadrature",
}


def load_matrix(directory: Path) -> list[dict]:
    matrix: list[dict] = []
    for family, (_, topology) in FAMILIES.items():
        for iv in IVS:
            pair: dict[str, dict] = {}
            for short_mode, full_mode in MODES.items():
                path = directory / f"{family}_iv{iv}_{short_mode}_np1.json"
                record = json.loads(path.read_text())
                if record["mode"] != full_mode:
                    raise RuntimeError(f"wrong mode in {path}")
                if record["mpi_ranks"] != 1 or record["omp_threads_per_rank"] != 1:
                    raise RuntimeError(f"not a single-thread run: {path}")
                if record["quadrature_order"] != 4:
                    raise RuntimeError(f"wrong quadrature order in {path}")
                timing = record["timing"]
                component_sum = sum(
                    timing[name]["corrected_seconds"]
                    for name in (
                        "volume",
                        "internal_surface",
                        "boundary_surface",
                        "volume_weight_normalization",
                    )
                )
                total = timing["total"]["corrected_seconds"]
                if abs(component_sum - total) > 1e-8:
                    raise RuntimeError(f"timing sum mismatch in {path}")
                stored = record["stored_data"]
                quadrature = sum(
                    stored[name]
                    for name in (
                        "volume_quadrature_points",
                        "internal_surface_quadrature_points",
                        "boundary_surface_quadrature_points",
                    )
                )
                weights = sum(
                    stored[name]
                    for name in (
                        "volume_recovery_weight_entries",
                        "internal_surface_state_weight_entries",
                        "internal_surface_flux_weight_entries",
                        "internal_surface_gradient_weight_entries",
                        "boundary_gradient_weight_entries",
                    )
                )
                if short_mode == "efficient" and (quadrature != 0 or weights <= 0):
                    raise RuntimeError(f"invalid efficient storage in {path}")
                if short_mode == "traditional" and (quadrature <= 0 or weights != 0):
                    raise RuntimeError(f"invalid traditional storage in {path}")
                record["total_quadrature_points"] = quadrature
                record["total_weight_entries"] = weights
                pair[short_mode] = record

            efficient = pair["efficient"]
            traditional = pair["traditional"]
            for field in ("nodes", "cells", "edges"):
                if efficient[field] != traditional[field]:
                    raise RuntimeError(f"paired {field} mismatch for {family} IV{iv}")
            matrix.append(
                {
                    "family": family,
                    "topology": topology,
                    "iv": iv,
                    "efficient": efficient,
                    "traditional": traditional,
                    "cost_ratio": efficient["timing"]["total"]["corrected_seconds"]
                    / traditional["timing"]["total"]["corrected_seconds"],
                }
            )
    return matrix


def seconds(value: float) -> str:
    if value < 0.1:
        return f"{value:.5f}"
    if value < 10:
        return f"{value:.4f}"
    return f"{value:.3f}"


def write_csv(path: Path, matrix: list[dict]) -> None:
    fields = [
        "family",
        "topology",
        "iv",
        "nodes",
        "volume_cells",
        "edges",
        "mpi_ranks",
        "omp_threads_per_rank",
        "efficient_volume_seconds",
        "efficient_internal_surface_seconds",
        "efficient_boundary_surface_seconds",
        "efficient_normalization_seconds",
        "efficient_total_seconds",
        "traditional_volume_seconds",
        "traditional_internal_surface_seconds",
        "traditional_boundary_surface_seconds",
        "traditional_total_seconds",
        "efficient_over_traditional_initialization_cost",
        "efficient_weight_entries",
        "traditional_quadrature_points",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in matrix:
            efficient = row["efficient"]
            traditional = row["traditional"]
            et = efficient["timing"]
            tt = traditional["timing"]
            writer.writerow(
                {
                    "family": FAMILIES[row["family"]][0],
                    "topology": row["topology"],
                    "iv": row["iv"],
                    "nodes": efficient["nodes"],
                    "volume_cells": efficient["cells"],
                    "edges": efficient["edges"],
                    "mpi_ranks": 1,
                    "omp_threads_per_rank": 1,
                    "efficient_volume_seconds": et["volume"]["corrected_seconds"],
                    "efficient_internal_surface_seconds": et["internal_surface"]["corrected_seconds"],
                    "efficient_boundary_surface_seconds": et["boundary_surface"]["corrected_seconds"],
                    "efficient_normalization_seconds": et["volume_weight_normalization"]["corrected_seconds"],
                    "efficient_total_seconds": et["total"]["corrected_seconds"],
                    "traditional_volume_seconds": tt["volume"]["corrected_seconds"],
                    "traditional_internal_surface_seconds": tt["internal_surface"]["corrected_seconds"],
                    "traditional_boundary_surface_seconds": tt["boundary_surface"]["corrected_seconds"],
                    "traditional_total_seconds": tt["total"]["corrected_seconds"],
                    "efficient_over_traditional_initialization_cost": row["cost_ratio"],
                    "efficient_weight_entries": efficient["total_weight_entries"],
                    "traditional_quadrature_points": traditional["total_quadrature_points"],
                }
            )


def h(value: object) -> str:
    return html.escape(str(value))


def render_html(matrix: list[dict]) -> str:
    efficient_time_rows: list[str] = []
    traditional_time_rows: list[str] = []
    count_rows: list[str] = []
    ratios: list[float] = []
    for row in matrix:
        efficient = row["efficient"]
        traditional = row["traditional"]
        et = efficient["timing"]
        tt = traditional["timing"]
        ratios.append(row["cost_ratio"])
        efficient_time_rows.append(
            "<tr>"
            f"<td>{h(FAMILIES[row['family']][0])}</td><td>{h(row['topology'])}</td>"
            f"<td>IV{row['iv']}</td><td>{efficient['nodes']:,}</td><td>{efficient['cells']:,}</td>"
            f"<td>{seconds(et['volume']['corrected_seconds'])}</td>"
            f"<td>{seconds(et['internal_surface']['corrected_seconds'])}</td>"
            f"<td>{seconds(et['boundary_surface']['corrected_seconds'])}</td>"
            f"<td>{seconds(et['volume_weight_normalization']['corrected_seconds'])}</td>"
            f"<td class='accent'>{seconds(et['total']['corrected_seconds'])}</td>"
            "</tr>"
        )
        traditional_time_rows.append(
            "<tr>"
            f"<td>{h(FAMILIES[row['family']][0])}</td><td>{h(row['topology'])}</td>"
            f"<td>IV{row['iv']}</td><td>{traditional['nodes']:,}</td><td>{traditional['cells']:,}</td>"
            f"<td>{seconds(tt['volume']['corrected_seconds'])}</td>"
            f"<td>{seconds(tt['internal_surface']['corrected_seconds'])}</td>"
            f"<td>{seconds(tt['boundary_surface']['corrected_seconds'])}</td>"
            f"<td class='accent'>{seconds(tt['total']['corrected_seconds'])}</td>"
            f"<td>{row['cost_ratio']:.2f}</td>"
            "</tr>"
        )
        es = efficient["stored_data"]
        ts = traditional["stored_data"]
        count_rows.append(
            "<tr>"
            f"<td>{h(row['topology'])}</td><td>IV{row['iv']}</td>"
            f"<td>{es['volume_recovery_weight_entries']:,}</td>"
            f"<td>{es['internal_surface_state_weight_entries'] + es['internal_surface_flux_weight_entries'] + es['internal_surface_gradient_weight_entries']:,}</td>"
            f"<td>{es['boundary_gradient_weight_entries']:,}</td>"
            f"<td>{efficient['total_weight_entries']:,}</td>"
            f"<td>{ts['volume_quadrature_points']:,}</td>"
            f"<td>{ts['internal_surface_quadrature_points'] + ts['boundary_surface_quadrature_points']:,}</td>"
            f"<td>{traditional['total_quadrature_points']:,}</td>"
            "</tr>"
        )

    generated = dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>NCFV 修改后积分初始化阶段单线程计时</title>
<style>
@page {{ size: A4 landscape; margin: 12mm; }}
body {{ font-family: "Noto Sans CJK SC", "Microsoft YaHei", sans-serif; color:#1f2937; font-size:8.5pt; line-height:1.35; }}
h1 {{ text-align:center; color:#17365d; font-size:18pt; margin:0 0 5pt; }}
h2 {{ color:#17365d; font-size:12pt; border-bottom:1px solid #9fbad0; padding-bottom:2pt; margin:12pt 0 5pt; }}
p {{ margin:4pt 0; }}
.meta {{ text-align:center; color:#52606d; margin-bottom:9pt; }}
.note {{ background:#eef5fb; border-left:4px solid #3478a8; padding:6pt 9pt; margin:7pt 0; }}
table {{ width:100%; border-collapse:collapse; table-layout:fixed; margin:5pt 0 9pt; font-size:7.1pt; }}
th,td {{ border:1px solid #aab7c4; padding:2.2pt; text-align:center; vertical-align:middle; overflow-wrap:anywhere; }}
th {{ background:#dbeaf5; color:#17365d; }}
tr:nth-child(even) td {{ background:#f7fafc; }}
.accent {{ color:#165c35; font-weight:bold; }}
.small {{ color:#52606d; font-size:7.5pt; }}
</style>
</head>
<body>
<h1>NCFV 修改后不同网格积分初始化阶段单线程计时</h1>
<div class="meta">Prism6 / Hex8 / Tet4；IV10 / 20 / 40 / 80；生成时间 {h(generated)}</div>
<div class="note"><b>资源与计时口径：</b>全部使用 1 个 MPI 秩，每秩 OMP/BLAS 线程数均为 1。高效方法仅统计微体恢复权、融合后的内部微面节点权、边界积分权以及恢复权归一化。传统方法仅统计参考求积规则创建、物理高斯点与物理积分权生成和存储。计时器自身开销按本进程实测值扣除。</div>
<p><b>明确排除：</b>网格读取、拓扑构造、两种方法共有的几何量、传统体积分矩累积、重构矩阵、字段分配、流场初始化以及任何时间推进。</p>
<h2>1. 高效方法初始化权时间（秒）</h2>
<table>
<thead><tr><th>网格族</th><th>单元</th><th>网格</th><th>节点</th><th>体单元</th><th>微体恢复权</th><th>内部微面权</th><th>边界权</th><th>恢复权归一化</th><th>合计</th></tr></thead>
<tbody>{''.join(efficient_time_rows)}</tbody>
</table>
<h2>2. 传统方法高斯点与积分权时间（秒）</h2>
<table>
<thead><tr><th>网格族</th><th>单元</th><th>网格</th><th>节点</th><th>体单元</th><th>体高斯点/权</th><th>内部面高斯点/权</th><th>边界面高斯点/权</th><th>合计</th><th>高效/传统初始化成本</th></tr></thead>
<tbody>{''.join(traditional_time_rows)}</tbody>
</table>
<h2>3. 初始化数据规模</h2>
<table>
<thead><tr><th>单元</th><th>网格</th><th>高效：微体恢复权</th><th>高效：内部面全部权</th><th>高效：边界权</th><th>高效：权合计</th><th>传统：体高斯点</th><th>传统：全部面高斯点</th><th>传统：高斯点合计</th></tr></thead>
<tbody>{''.join(count_rows)}</tbody>
</table>
<h2>4. 结论</h2>
<ol>
<li>在 12 个网格上，高效方法的权初始化成本为传统高斯点/积分权初始化的 {min(ratios):.2f}–{max(ratios):.2f} 倍。这里比较的只是一次性预计算，不是时间推进成本。</li>
<li>修改后的高效方法把同一节点的左右状态权、左右通量矩阵权和梯度积分权融合进一个有序短向量条目，使用二分查找原位累加，不再为每次合并构造多个 std::map。</li>
<li>初始化付出的权构造成本在后续每个时间步重复使用；传统方法则在时间推进的每次通量/体积分过程中持续遍历大量高斯点。因此不能用本表推断高效方法的整体求解速度更慢。</li>
</ol>
<p class="small">每个组合独立运行 1 次。时间采用方法专属代码块的累计墙钟，并扣除 100 万次空计时对测得的单次计时器开销。</p>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-html", type=Path, required=True)
    args = parser.parse_args()
    matrix = load_matrix(args.results_dir)
    write_csv(args.output_csv, matrix)
    args.output_html.write_text(render_html(matrix), encoding="utf-8")
    print(args.output_csv)
    print(args.output_html)


if __name__ == "__main__":
    main()
