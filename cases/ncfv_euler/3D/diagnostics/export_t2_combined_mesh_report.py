#!/usr/bin/env python3
"""Validate and report the mixed 1/8/32-worker NCFV t=2 benchmark matrix."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import html
import json
import math
from pathlib import Path


IVS = (10, 20, 40, 80)
FAMILIES = {
    "prism": ("三棱柱", "Prism6"),
    "hex": ("六面体", "Hex8"),
    "tet": ("四面体", "Tet4"),
}
MODES = {
    "efficient": "EfficientDifferential",
    "traditional": "TraditionalQuadrature",
}
RANKS = {10: 1, 20: 1, 40: 8, 80: 32}


def result_path(
    family: str,
    iv: int,
    mode: str,
    serial_dir: Path,
    prism_parallel_dir: Path,
    iv40_tethex_dir: Path,
    iv80_tethex_dir: Path,
) -> Path:
    if iv <= 20:
        return serial_dir / f"{family}_iv{iv}_{mode}_np1.json"
    if family == "prism":
        return prism_parallel_dir / f"original_iv{iv}_{mode}_np{RANKS[iv]}.json"
    if iv == 40:
        return iv40_tethex_dir / f"{family}_iv40_{mode}_np8.json"
    return iv80_tethex_dir / f"{family}_iv80_{mode}_np32.json"


def seconds(value: float) -> str:
    if value < 10:
        return f"{value:.3f}"
    return f"{value:.2f}"


def keep(value: object) -> str:
    return "&#8288;".join(html.escape(str(value)))


def execution_label(iv: int) -> str:
    return "串行" if RANKS[iv] == 1 else f"{RANKS[iv]} 核 MPI 并行"


def load_matrix(args: argparse.Namespace) -> tuple[list[dict], dict[tuple[str, int], dict]]:
    mesh_records = json.loads(args.mesh_statistics.read_text())
    mesh_stats = {(record["family"], int(record["iv"])): record for record in mesh_records}
    if len(mesh_stats) != 12:
        raise RuntimeError(f"expected 12 mesh records, found {len(mesh_stats)}")

    matrix: list[dict] = []
    for family in FAMILIES:
        for iv in IVS:
            paired: dict[str, dict] = {}
            for mode, expected_mode in MODES.items():
                path = result_path(
                    family,
                    iv,
                    mode,
                    args.serial_dir,
                    args.prism_parallel_dir,
                    args.iv40_tethex_dir,
                    args.iv80_tethex_dir,
                )
                record = json.loads(path.read_text())
                mesh = mesh_stats[(family, iv)]
                checks = {
                    "mode": record["mode"] == expected_mode,
                    "mpi_ranks": record["mpi_ranks"] == RANKS[iv],
                    "nodes": record["nodes"] == mesh["nodes"],
                    "cells": record["cells"] == mesh["volume_cells"],
                    "end_time": math.isclose(record["end_time"], 2.0, abs_tol=1e-12),
                    "physical_steps": record["physical_steps"] > 0,
                    "no_checkpoint": record.get("checkpoint_prefix", "") == "",
                    "scope": "restart I/O disabled" in record["measurement_scope"],
                }
                failed = [name for name, passed in checks.items() if not passed]
                if failed:
                    raise RuntimeError(f"{path} failed validation: {failed}")
                paired[mode] = record

            efficient = paired["efficient"]
            traditional = paired["traditional"]
            eff_pss_gib = efficient["memory"]["after_time_2"]["pss"]["sum_mib"] / 1024.0
            trad_pss_gib = traditional["memory"]["after_time_2"]["pss"]["sum_mib"] / 1024.0
            matrix.append(
                {
                    "family": family,
                    "iv": iv,
                    "ranks": RANKS[iv],
                    "execution": execution_label(iv),
                    "efficient": efficient,
                    "traditional": traditional,
                    "algorithm_speedup": traditional["time_march_seconds"] / efficient["time_march_seconds"],
                    "eff_pss_gib": eff_pss_gib,
                    "trad_pss_gib": trad_pss_gib,
                    "pss_saved_gib": trad_pss_gib - eff_pss_gib,
                    "pss_ratio": trad_pss_gib / eff_pss_gib,
                    "pss_reduction_percent": (trad_pss_gib - eff_pss_gib) / trad_pss_gib * 100.0,
                }
            )
    return matrix, mesh_stats


def write_csv(path: Path, matrix: list[dict], mesh_stats: dict[tuple[str, int], dict]) -> None:
    fields = [
        "family", "topology", "iv", "execution", "mpi_ranks", "omp_threads_per_rank",
        "nodes", "volume_cells", "domain_volume", "mean_cell_volume", "mean_volume_equivalent_size",
        "efficient_steps", "traditional_steps", "efficient_initialize_seconds",
        "traditional_initialize_seconds", "efficient_march_seconds", "traditional_march_seconds",
        "algorithm_speedup", "efficient_total_seconds", "traditional_total_seconds",
        "efficient_pss_gib", "traditional_pss_gib", "pss_saved_gib", "pss_ratio",
        "pss_reduction_percent", "efficient_max_rank_hwm_mib", "traditional_max_rank_hwm_mib",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in matrix:
            mesh = mesh_stats[(row["family"], row["iv"])]
            efficient = row["efficient"]
            traditional = row["traditional"]
            writer.writerow(
                {
                    "family": FAMILIES[row["family"]][0],
                    "topology": FAMILIES[row["family"]][1],
                    "iv": row["iv"],
                    "execution": row["execution"],
                    "mpi_ranks": row["ranks"],
                    "omp_threads_per_rank": 1,
                    "nodes": mesh["nodes"],
                    "volume_cells": mesh["volume_cells"],
                    "domain_volume": mesh["domain_volume"],
                    "mean_cell_volume": mesh["mean_cell_volume"],
                    "mean_volume_equivalent_size": mesh["mean_volume_equivalent_size"],
                    "efficient_steps": efficient["physical_steps"],
                    "traditional_steps": traditional["physical_steps"],
                    "efficient_initialize_seconds": efficient["initialize_seconds"],
                    "traditional_initialize_seconds": traditional["initialize_seconds"],
                    "efficient_march_seconds": efficient["time_march_seconds"],
                    "traditional_march_seconds": traditional["time_march_seconds"],
                    "algorithm_speedup": row["algorithm_speedup"],
                    "efficient_total_seconds": efficient["total_seconds"],
                    "traditional_total_seconds": traditional["total_seconds"],
                    "efficient_pss_gib": row["eff_pss_gib"],
                    "traditional_pss_gib": row["trad_pss_gib"],
                    "pss_saved_gib": row["pss_saved_gib"],
                    "pss_ratio": row["pss_ratio"],
                    "pss_reduction_percent": row["pss_reduction_percent"],
                    "efficient_max_rank_hwm_mib": efficient["memory"]["after_time_2"]["hwm"]["max_mib"],
                    "traditional_max_rank_hwm_mib": traditional["memory"]["after_time_2"]["hwm"]["max_mib"],
                }
            )


def render_html(matrix: list[dict], mesh_stats: dict[tuple[str, int], dict]) -> str:
    mesh_rows: list[str] = []
    time_rows: list[str] = []
    memory_rows: list[str] = []
    speedups: dict[str, list[float]] = {family: [] for family in FAMILIES}

    for row in matrix:
        family = row["family"]
        family_name, topology = FAMILIES[family]
        iv = row["iv"]
        mesh = mesh_stats[(family, iv)]
        efficient = row["efficient"]
        traditional = row["traditional"]
        speedups[family].append(row["algorithm_speedup"])

        mesh_rows.append(
            "<tr>"
            f"<td>{keep(family_name)}</td><td>{keep(topology)}</td><td>{keep(f'IV{iv}')}</td>"
            f"<td>{keep(row['execution'])}</td><td>{keep('10×10×4')}</td>"
            f"<td>{keep(mesh['nodes'])}</td><td>{keep(mesh['volume_cells'])}</td>"
            f"<td>{keep(format(mesh['mean_cell_volume'], '.6g'))}</td>"
            f"<td>{keep(format(mesh['mean_volume_equivalent_size'], '.6f'))}</td>"
            "</tr>"
        )

        time_rows.append(
            "<tr>"
            f"<td>{keep(topology)}</td><td>{keep(f'IV{iv}')}</td><td>{keep(row['execution'])}</td>"
            f"<td>{keep(str(efficient['physical_steps']) + '/' + str(traditional['physical_steps']))}</td>"
            f"<td>{keep(seconds(efficient['initialize_seconds']))}</td>"
            f"<td>{keep(seconds(traditional['initialize_seconds']))}</td>"
            f"<td>{keep(seconds(efficient['time_march_seconds']))}</td>"
            f"<td>{keep(seconds(traditional['time_march_seconds']))}</td>"
            f"<td>{keep(format(row['algorithm_speedup'], '.2f'))}</td>"
            f"<td>{keep(seconds(efficient['total_seconds']))}</td>"
            f"<td>{keep(seconds(traditional['total_seconds']))}</td>"
            "</tr>"
        )

        memory_rows.append(
            "<tr>"
            f"<td>{keep(topology)}</td><td>{keep(f'IV{iv}')}</td><td>{keep(row['execution'])}</td>"
            f"<td>{keep(format(row['eff_pss_gib'], '.3f'))}</td>"
            f"<td>{keep(format(row['trad_pss_gib'], '.3f'))}</td>"
            f"<td>{keep(format(row['pss_saved_gib'], '.3f'))}</td>"
            f"<td>{keep(format(row['pss_ratio'], '.2f'))}</td>"
            f"<td>{keep(format(row['pss_reduction_percent'], '.1f') + '%')}</td>"
            f"<td>{keep(format(efficient['memory']['after_time_2']['hwm']['max_mib'], '.1f'))}</td>"
            f"<td>{keep(format(traditional['memory']['after_time_2']['hwm']['max_mib'], '.1f'))}</td>"
            "</tr>"
        )

    now = dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
    family_summary = "；".join(
        f"{FAMILIES[family][0]} {min(values):.2f}–{max(values):.2f} 倍"
        for family, values in speedups.items()
    )
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>NCFV 串并行三类网格综合性能报告</title>
<style>
@page {{ size: A4 landscape; margin: 14mm; }}
body {{ font-family: "Noto Sans CJK SC", "Microsoft YaHei", sans-serif; color: #1f2937; font-size: 9.5pt; line-height: 1.42; }}
h1 {{ text-align: center; color: #17365d; font-size: 19pt; margin: 0 0 6pt; }}
h2 {{ color: #17365d; font-size: 13pt; margin: 14pt 0 6pt; border-bottom: 1px solid #9fbad0; padding-bottom: 2pt; }}
p {{ margin: 4pt 0; }}
.meta {{ text-align: center; color: #52606d; margin-bottom: 12pt; }}
.note {{ background: #eef5fb; border-left: 4px solid #3478a8; padding: 7pt 10pt; margin: 8pt 0; }}
.warning {{ background: #fff8e6; border-left: 4px solid #d49b21; padding: 7pt 10pt; margin: 8pt 0; }}
table {{ width: 100%; border-collapse: collapse; margin: 6pt 0 10pt; page-break-inside: avoid; font-size: 8.3pt; }}
th, td {{ border: 1px solid #aab7c4; padding: 3pt; text-align: center; vertical-align: middle; }}
th {{ background: #dbeaf5; color: #17365d; font-weight: bold; }}
tr:nth-child(even) td {{ background: #f7fafc; }}
.small {{ font-size: 8.5pt; color: #52606d; }}
</style>
</head>
<body>
<h1>NCFV 串行/并行三类网格性能、内存与网格统计</h1>
<div class="meta">三维等熵涡；物理时间 t=0 → t=2；生成时间 {html.escape(now)}</div>

<div class="note"><b>计算资源口径：</b>IV10、IV20 使用 1 个 MPI 秩（串行）；IV40 使用 8 个 MPI 秩；IV80 使用 32 个 MPI 秩。每个 MPI 秩的 OMP/BLAS 线程数固定为 1。当前 NCFV 残差主循环未采用 OpenMP 并行，因此“8/32 线程”按 8/32 个单线程 MPI CPU 工作单元实现。</div>

<h2>1. 数据来源与统计定义</h2>
<p>每个网格均比较高效微分积分方法（EfficientDifferential）与传统高斯求积方法（TraditionalQuadrature），从 t=0 直接推进至 t=2；关闭 VTK、常规输出和重启 I/O。IV10/20 使用本轮单进程新结果；IV40 的 Hex8/Tet4 使用 8 MPI 新结果，Prism6 使用本次独立复测的 8 MPI 结果；IV80 复用已有 32 MPI 完整结果。</p>
<p>所有网格物理域均为 10×10×4，总体积为 400。平均网格尺寸定义为体积等效尺寸的算术平均：<b>h̄ = (1/N) Σ Vᵢ^(1/3)</b>。同时列出节点数和体单元数；“网格数量”在本报告中指体单元数。</p>

<h2>2. 网格数量与平均尺寸</h2>
<table>
<thead><tr><th>网格族</th><th>单元</th><th>网格</th><th>计算模式</th><th>物理域</th><th>节点数</th><th>体单元数</th><th>平均单元体积</th><th>平均尺寸 h̄</th></tr></thead>
<tbody>{''.join(mesh_rows)}</tbody>
</table>

<h2>3. 从 t=0 推进至 t=2 的计算时间</h2>
<table>
<thead><tr><th>单元</th><th>网格</th><th>计算模式</th><th>步数<br>高效/传统</th><th>初始化(s)<br>高效</th><th>初始化(s)<br>传统</th><th>推进(s)<br>高效</th><th>推进(s)<br>传统</th><th>算法加速比<br>传统/高效</th><th>总时间(s)<br>高效</th><th>总时间(s)<br>传统</th></tr></thead>
<tbody>{''.join(time_rows)}</tbody>
</table>

<h2>4. 内存需求</h2>
<table>
<thead><tr><th>单元</th><th>网格</th><th>计算模式</th><th>t=2 PSS(GiB)<br>高效</th><th>t=2 PSS(GiB)<br>传统</th><th>节省(GiB)</th><th>传统/高效</th><th>降低比例</th><th>单秩最大 VmHWM(MiB)<br>高效</th><th>单秩最大 VmHWM(MiB)<br>传统</th></tr></thead>
<tbody>{''.join(memory_rows)}</tbody>
</table>
<div class="warning"><b>内存口径：</b>PSS 按共享页比例分摊，是多进程聚合内存的主指标；VmHWM 是单进程历史 RSS 峰值。不同 IV 使用的进程数不同，因此跨 IV 的聚合内存同时受到网格规模和 MPI 进程基线开销影响。</div>

<h2>5. 结论与解释</h2>
<ol>
<li>在全部 12 个“同网格、同资源”配对中，算法推进加速比范围为：{family_summary}。</li>
<li>同一 IV 编号并不代表三类网格具有相同体单元数：Tet4 是对应 Hex8 的 6 倍；Prism6 使用原始非均匀三棱柱网格。因此跨单元类型的绝对时间应结合体单元数和 h̄ 一起解读。</li>
<li>本表的“算法加速比”比较同一网格上的传统/高效方法；由于 IV10/20 与 IV40/80 使用不同资源，不用跨 IV 的绝对时间计算并行加速效率。</li>
</ol>
<p class="small">每个新组合独立执行 1 次；复用组合均为生产 Solver::Run() 路径、t=0→2 的完整结果。报告保留初始化时间、时间推进时间和总时间三个口径。</p>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial-dir", type=Path, required=True)
    parser.add_argument("--prism-parallel-dir", type=Path, required=True)
    parser.add_argument("--iv40-tethex-dir", type=Path, required=True)
    parser.add_argument("--iv80-tethex-dir", type=Path, required=True)
    parser.add_argument("--mesh-statistics", type=Path, required=True)
    parser.add_argument("--output-html", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()
    matrix, mesh_stats = load_matrix(args)
    write_csv(args.output_csv, matrix, mesh_stats)
    args.output_html.write_text(render_html(matrix, mesh_stats), encoding="utf-8")
    print(args.output_html)
    print(args.output_csv)


if __name__ == "__main__":
    main()
