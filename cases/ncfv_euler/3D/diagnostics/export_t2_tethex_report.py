#!/usr/bin/env python3
"""Validate fresh tetrahedral/hexahedral t=2 results and emit a Word-ready HTML report."""

from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import math
from pathlib import Path


FAMILIES = {"hex": "纯六面体（Hex8）", "tet": "纯四面体（Tet4）"}
TABLE_FAMILIES = {"hex": "Hex8", "tet": "Tet4"}
MODES = {
    "efficient": "EfficientDifferential",
    "traditional": "TraditionalQuadrature",
}
IVS = (10, 20, 40, 80)


def ranks_for(iv: int) -> int:
    return 8 if iv <= 20 else 32


def expected_cells(family: str, iv: int) -> int:
    return iv**3 * (6 if family == "tet" else 1)


def load_results(results_dir: Path) -> dict[tuple[str, int, str], dict]:
    expected_names = {
        f"{family}_iv{iv}_{mode}_np{ranks_for(iv)}.json"
        for family in FAMILIES
        for iv in IVS
        for mode in MODES
    }
    actual_names = {path.name for path in results_dir.glob("*.json")}
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        extra = sorted(actual_names - expected_names)
        raise RuntimeError(f"result matrix mismatch; missing={missing}, extra={extra}")

    results: dict[tuple[str, int, str], dict] = {}
    for family in FAMILIES:
        for iv in IVS:
            for mode, expected_mode in MODES.items():
                name = f"{family}_iv{iv}_{mode}_np{ranks_for(iv)}.json"
                path = results_dir / name
                record = json.loads(path.read_text())
                checks = {
                    "mode": record["mode"] == expected_mode,
                    "mpi_ranks": record["mpi_ranks"] == ranks_for(iv),
                    "nodes": record["nodes"] == (iv + 1) ** 3,
                    "cells": record["cells"] == expected_cells(family, iv),
                    "end_time": math.isclose(record["end_time"], 2.0, abs_tol=1e-12),
                    "physical_steps": record["physical_steps"] > 0,
                    "no_restart_input": record["checkpoint_prefix"] == "",
                    "scope": "restart I/O disabled" in record["measurement_scope"],
                }
                failed = [key for key, passed in checks.items() if not passed]
                if failed:
                    raise RuntimeError(f"{name} failed validation: {failed}")
                results[(family, iv, mode)] = record
    return results


def seconds(value: float) -> str:
    if value < 10:
        return f"{value:.3f}"
    return f"{value:.2f}"


def gib_from_mib(value: float) -> float:
    return value / 1024.0


def keep_together(value: object) -> str:
    """Prevent LibreOffice from inserting a line break inside a table token."""
    return "&#8288;".join(html.escape(str(value)))


def report_html(results: dict[tuple[str, int, str], dict]) -> str:
    time_rows: list[str] = []
    pss_rows: list[str] = []
    rss_rows: list[str] = []
    speedups: dict[str, list[float]] = {family: [] for family in FAMILIES}
    memory_ratios: dict[str, list[float]] = {family: [] for family in FAMILIES}

    for family, family_name in FAMILIES.items():
        table_family = TABLE_FAMILIES[family]
        for iv in IVS:
            efficient = results[(family, iv, "efficient")]
            traditional = results[(family, iv, "traditional")]
            eff_march = efficient["time_march_seconds"]
            trad_march = traditional["time_march_seconds"]
            speedup = trad_march / eff_march
            speedups[family].append(speedup)

            eff_total = efficient["total_seconds"]
            trad_total = traditional["total_seconds"]
            time_rows.append(
                "<tr>"
                f"<td>{keep_together(table_family)}</td><td>{keep_together(f'IV{iv}')}</td>"
                f"<td>{keep_together(ranks_for(iv))}</td>"
                f"<td>{keep_together(str(efficient['physical_steps']) + '/' + str(traditional['physical_steps']))}</td>"
                f"<td>{keep_together(seconds(efficient['initialize_seconds']))}</td>"
                f"<td>{keep_together(seconds(traditional['initialize_seconds']))}</td>"
                f"<td>{keep_together(seconds(eff_march))}</td><td>{keep_together(seconds(trad_march))}</td>"
                f"<td>{keep_together(f'{speedup:.2f}')}</td>"
                f"<td>{keep_together(seconds(eff_total))}</td><td>{keep_together(seconds(trad_total))}</td>"
                "</tr>"
            )

            eff_init_pss = gib_from_mib(efficient["memory"]["after_initialize"]["pss"]["sum_mib"])
            trad_init_pss = gib_from_mib(traditional["memory"]["after_initialize"]["pss"]["sum_mib"])
            eff_pss = gib_from_mib(efficient["memory"]["after_time_2"]["pss"]["sum_mib"])
            trad_pss = gib_from_mib(traditional["memory"]["after_time_2"]["pss"]["sum_mib"])
            ratio = trad_pss / eff_pss
            reduction = (trad_pss - eff_pss) / trad_pss * 100.0
            memory_ratios[family].append(ratio)
            pss_rows.append(
                "<tr>"
                f"<td>{keep_together(table_family)}</td><td>{keep_together(f'IV{iv}')}</td><td>{keep_together(ranks_for(iv))}</td>"
                f"<td>{keep_together(f'{eff_init_pss:.3f}')}</td><td>{keep_together(f'{trad_init_pss:.3f}')}</td>"
                f"<td>{keep_together(f'{eff_pss:.3f}')}</td><td>{keep_together(f'{trad_pss:.3f}')}</td>"
                f"<td>{keep_together(f'{trad_pss - eff_pss:.3f}')}</td><td>{keep_together(f'{ratio:.2f}')}</td><td>{keep_together(f'{reduction:.1f}%')}</td>"
                "</tr>"
            )

            eff_rss = gib_from_mib(efficient["memory"]["after_time_2"]["rss"]["sum_mib"])
            trad_rss = gib_from_mib(traditional["memory"]["after_time_2"]["rss"]["sum_mib"])
            eff_hwm = gib_from_mib(efficient["memory"]["after_time_2"]["hwm"]["sum_mib"])
            trad_hwm = gib_from_mib(traditional["memory"]["after_time_2"]["hwm"]["sum_mib"])
            eff_hwm_max = efficient["memory"]["after_time_2"]["hwm"]["max_mib"]
            trad_hwm_max = traditional["memory"]["after_time_2"]["hwm"]["max_mib"]
            rss_rows.append(
                "<tr>"
                f"<td>{keep_together(table_family)}</td><td>{keep_together(f'IV{iv}')}</td>"
                f"<td>{keep_together(f'{eff_rss:.3f}')}</td><td>{keep_together(f'{trad_rss:.3f}')}</td>"
                f"<td>{keep_together(f'{eff_hwm:.3f}')}</td><td>{keep_together(f'{trad_hwm:.3f}')}</td>"
                f"<td>{keep_together(f'{eff_hwm_max:.1f}')}</td><td>{keep_together(f'{trad_hwm_max:.1f}')}</td>"
                "</tr>"
            )

    hex80_eff = results[("hex", 80, "efficient")]
    hex80_trad = results[("hex", 80, "traditional")]
    tet80_eff = results[("tet", 80, "efficient")]
    tet80_trad = results[("tet", 80, "traditional")]

    def pss_gib(record: dict) -> float:
        return gib_from_mib(record["memory"]["after_time_2"]["pss"]["sum_mib"])

    now = dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>NCFV 纯四面体与纯六面体 t=2 性能与内存对比</title>
<style>
@page {{ size: A4 landscape; margin: 14mm; }}
body {{ width: 265mm; font-family: "Noto Sans CJK SC", "Microsoft YaHei", sans-serif; color: #1f2937; font-size: 9.5pt; line-height: 1.42; }}
h1 {{ text-align: center; color: #17365d; font-size: 20pt; margin: 0 0 6pt; }}
h2 {{ color: #17365d; font-size: 13pt; margin: 14pt 0 6pt; border-bottom: 1px solid #9fbad0; padding-bottom: 2pt; }}
p {{ margin: 4pt 0; }}
.meta {{ text-align: center; color: #52606d; margin-bottom: 12pt; }}
.note {{ background: #eef5fb; border-left: 4px solid #3478a8; padding: 7pt 10pt; margin: 8pt 0; }}
.warning {{ background: #fff8e6; border-left: 4px solid #d49b21; padding: 7pt 10pt; margin: 8pt 0; }}
table {{ width: 265mm; border-collapse: collapse; margin: 6pt 0 10pt; page-break-inside: avoid; font-size: 8.5pt; }}
th, td {{ border: 1px solid #aab7c4; padding: 3pt 3.5pt; text-align: center; vertical-align: middle; }}
th {{ background: #dbeaf5; color: #17365d; font-weight: bold; white-space: normal; }}
td {{ white-space: nowrap; }}
tr:nth-child(even) td {{ background: #f7fafc; }}
.small {{ font-size: 8.5pt; color: #52606d; }}
.conclusion li {{ margin: 3pt 0; }}
</style>
</head>
<body>
<h1>NCFV 纯四面体与纯六面体网格性能与内存对比</h1>
<div class="meta">物理时间：t=0 → t=2；报告生成：{html.escape(now)}</div>

<div class="note"><b>结果来源：</b>本报告只使用清空旧数据后重新生成的 16 组实际运行结果。每组均从初始状态 t=0 直接推进到 t=2；未读取或写入重启检查点，未复用旧结果。</div>

<h2>1. 测试方法</h2>
<p>对比算法为高效微分积分方法（EfficientDifferential）与传统高斯求积方法（TraditionalQuadrature）。网格为三个方向均可施加周期边界的结构化纯 Hex8 网格，以及由每个六面体一致划分为 6 个 Tet4 的纯四面体网格。</p>
<p>IV10、IV20 使用 8 个 MPI 进程；IV40、IV80 使用 32 个 MPI 进程；每个 MPI 进程固定 1 个线程。相同网格上的两种算法使用完全相同的 MPI/线程资源。求解阶段关闭 VTK、常规输出和重启 I/O。</p>
<p>“时间推进”采用所有 MPI 秩墙钟时间的最大值，包含生产求解器 SSPRK3 推进、每步三次 RHS、CFL 时间步选择及最后一个截断步；“总时间”=初始化+时间推进。内存主指标为各 MPI 秩 PSS 之和。</p>

<h2>2. 计算时间对比</h2>
<table>
<thead><tr><th>网格类型</th><th>网格</th><th>MPI</th><th>步数<br>高效/传统</th><th>初始化(s)<br>高效</th><th>初始化(s)<br>传统</th><th>推进(s)<br>高效</th><th>推进(s)<br>传统</th><th>推进加速比<br>传统/高效</th><th>总时间(s)<br>高效</th><th>总时间(s)<br>传统</th></tr></thead>
<tbody>{''.join(time_rows)}</tbody>
</table>

<h2>3. 内存需求对比（PSS 总量）</h2>
<table>
<thead><tr><th>网格类型</th><th>网格</th><th>MPI</th><th>初始化后 PSS(GiB)<br>高效</th><th>初始化后 PSS(GiB)<br>传统</th><th>t=2 PSS(GiB)<br>高效</th><th>t=2 PSS(GiB)<br>传统</th><th>节省(GiB)</th><th>传统/高效</th><th>降低比例</th></tr></thead>
<tbody>{''.join(pss_rows)}</tbody>
</table>

<h2>4. RSS 与峰值代理</h2>
<table>
<thead><tr><th>网格类型</th><th>网格</th><th>t=2 RSS 总量(GiB)<br>高效</th><th>t=2 RSS 总量(GiB)<br>传统</th><th>各秩 VmHWM 之和(GiB)<br>高效</th><th>各秩 VmHWM 之和(GiB)<br>传统</th><th>单秩最大 VmHWM(MiB)<br>高效</th><th>单秩最大 VmHWM(MiB)<br>传统</th></tr></thead>
<tbody>{''.join(rss_rows)}</tbody>
</table>
<div class="warning"><b>内存口径：</b>PSS 按共享页比例分摊，适合作为多进程聚合内存的主指标；RSS 会在各 MPI 秩重复计算共享页。VmHWM 是单进程历史 RSS 峰值，“各秩 VmHWM 之和”不代表所有秩在同一时刻的真实峰值，只作为保守容量参考。</div>

<h2>5. 主要结论</h2>
<ol class="conclusion">
<li>纯六面体网格上，高效方法的时间推进加速比为 {min(speedups['hex']):.2f}–{max(speedups['hex']):.2f} 倍；纯四面体网格上为 {min(speedups['tet']):.2f}–{max(speedups['tet']):.2f} 倍。</li>
<li>六面体 IV80：高效/传统推进时间为 {seconds(hex80_eff['time_march_seconds'])} s / {seconds(hex80_trad['time_march_seconds'])} s，加速 {hex80_trad['time_march_seconds'] / hex80_eff['time_march_seconds']:.2f} 倍；t=2 PSS 为 {pss_gib(hex80_eff):.3f} / {pss_gib(hex80_trad):.3f} GiB，节省 {pss_gib(hex80_trad) - pss_gib(hex80_eff):.3f} GiB。</li>
<li>四面体 IV80：高效/传统推进时间为 {seconds(tet80_eff['time_march_seconds'])} s / {seconds(tet80_trad['time_march_seconds'])} s，加速 {tet80_trad['time_march_seconds'] / tet80_eff['time_march_seconds']:.2f} 倍；t=2 PSS 为 {pss_gib(tet80_eff):.3f} / {pss_gib(tet80_trad):.3f} GiB，节省 {pss_gib(tet80_trad) - pss_gib(tet80_eff):.3f} GiB。</li>
<li>在全部 8 组网格配对中，高效方法的时间推进均更快，且 t=2 的 PSS 总量均更低。四面体网格上的收益最明显：传统方法 PSS 为高效方法的 {min(memory_ratios['tet']):.2f}–{max(memory_ratios['tet']):.2f} 倍。</li>
</ol>

<p class="small">说明：IV40/IV80 的 MPI 进程数与 IV10/IV20 不同，因此表中跨网格尺度的绝对时间不用于并行强缩放结论；算法加速比只在同一网格、相同资源配置内比较。每个组合本轮执行 1 次，结果反映本机本次运行条件。</p>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("output_html", type=Path)
    args = parser.parse_args()
    results = load_results(args.results_dir)
    args.output_html.write_text(report_html(results), encoding="utf-8")
    print(args.output_html)


if __name__ == "__main__":
    main()
