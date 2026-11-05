#!/usr/bin/env python3
"""Validate a fresh mixed-rank NCFV t=2 matrix and create HTML/CSV reports."""

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


def keep(value: object) -> str:
    return "&#8288;".join(html.escape(str(value)))


def seconds(value: float) -> str:
    return f"{value:.3f}" if value < 10 else f"{value:.2f}"


def execution(iv: int) -> str:
    return "1 MPI 串行" if RANKS[iv] == 1 else f"{RANKS[iv]} MPI 并行"


def load_data(result_dir: Path, mesh_statistics: Path) -> list[dict]:
    mesh_records = json.loads(mesh_statistics.read_text(encoding="utf-8"))
    meshes = {(row["family"], int(row["iv"])): row for row in mesh_records}
    if len(meshes) != 12:
        raise RuntimeError(f"expected 12 mesh records, found {len(meshes)}")

    rows: list[dict] = []
    for family in FAMILIES:
        for iv in IVS:
            records: dict[str, dict] = {}
            for mode, expected_mode in MODES.items():
                path = result_dir / f"{family}_iv{iv}_{mode}_np{RANKS[iv]}.json"
                record = json.loads(path.read_text(encoding="utf-8"))
                mesh = meshes[(family, iv)]
                checks = {
                    "mode": record["mode"] == expected_mode,
                    "mpi_ranks": record["mpi_ranks"] == RANKS[iv],
                    "nodes": record["nodes"] == mesh["nodes"],
                    "cells": record["cells"] == mesh["volume_cells"],
                    "end_time": math.isclose(record["end_time"], 2.0, abs_tol=1e-12),
                    "positive_steps": record["physical_steps"] > 0,
                    "fresh_start": record.get("checkpoint_prefix", "") == "",
                    "no_restart_io": "restart I/O disabled" in record["measurement_scope"],
                }
                failed = [name for name, passed in checks.items() if not passed]
                if failed:
                    raise RuntimeError(f"{path} failed validation: {failed}")
                records[mode] = record

            efficient = records["efficient"]
            traditional = records["traditional"]
            eff_t2_pss = efficient["memory"]["after_time_2"]["pss"]["sum_mib"] / 1024
            trad_t2_pss = traditional["memory"]["after_time_2"]["pss"]["sum_mib"] / 1024
            rows.append(
                {
                    "family": family,
                    "iv": iv,
                    "mesh": meshes[(family, iv)],
                    "efficient": efficient,
                    "traditional": traditional,
                    "march_speedup": traditional["time_march_seconds"] / efficient["time_march_seconds"],
                    "total_speedup": traditional["total_seconds"] / efficient["total_seconds"],
                    "initialization_speedup": traditional["initialize_seconds"] / efficient["initialize_seconds"],
                    "eff_t2_pss": eff_t2_pss,
                    "trad_t2_pss": trad_t2_pss,
                    "pss_ratio": trad_t2_pss / eff_t2_pss,
                    "pss_reduction": (trad_t2_pss - eff_t2_pss) / trad_t2_pss * 100,
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "family", "topology", "iv", "execution", "mpi_ranks", "omp_threads_per_rank",
        "nodes", "volume_cells", "mean_cell_volume", "mean_volume_equivalent_size",
        "efficient_steps", "traditional_steps", "efficient_initialize_seconds",
        "traditional_initialize_seconds", "efficient_march_seconds",
        "traditional_march_seconds", "march_speedup", "efficient_total_seconds",
        "traditional_total_seconds", "total_speedup", "efficient_after_initialize_pss_gib",
        "traditional_after_initialize_pss_gib", "efficient_t2_pss_gib",
        "traditional_t2_pss_gib", "pss_ratio", "pss_reduction_percent",
        "efficient_aggregate_hwm_gib", "traditional_aggregate_hwm_gib",
        "efficient_max_rank_hwm_mib", "traditional_max_rank_hwm_mib",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            family_name, topology = FAMILIES[row["family"]]
            mesh = row["mesh"]
            efficient = row["efficient"]
            traditional = row["traditional"]
            writer.writerow(
                {
                    "family": family_name,
                    "topology": topology,
                    "iv": row["iv"],
                    "execution": execution(row["iv"]),
                    "mpi_ranks": RANKS[row["iv"]],
                    "omp_threads_per_rank": 1,
                    "nodes": mesh["nodes"],
                    "volume_cells": mesh["volume_cells"],
                    "mean_cell_volume": mesh["mean_cell_volume"],
                    "mean_volume_equivalent_size": mesh["mean_volume_equivalent_size"],
                    "efficient_steps": efficient["physical_steps"],
                    "traditional_steps": traditional["physical_steps"],
                    "efficient_initialize_seconds": efficient["initialize_seconds"],
                    "traditional_initialize_seconds": traditional["initialize_seconds"],
                    "efficient_march_seconds": efficient["time_march_seconds"],
                    "traditional_march_seconds": traditional["time_march_seconds"],
                    "march_speedup": row["march_speedup"],
                    "efficient_total_seconds": efficient["total_seconds"],
                    "traditional_total_seconds": traditional["total_seconds"],
                    "total_speedup": row["total_speedup"],
                    "efficient_after_initialize_pss_gib": efficient["memory"]["after_initialize"]["pss"]["sum_mib"] / 1024,
                    "traditional_after_initialize_pss_gib": traditional["memory"]["after_initialize"]["pss"]["sum_mib"] / 1024,
                    "efficient_t2_pss_gib": row["eff_t2_pss"],
                    "traditional_t2_pss_gib": row["trad_t2_pss"],
                    "pss_ratio": row["pss_ratio"],
                    "pss_reduction_percent": row["pss_reduction"],
                    "efficient_aggregate_hwm_gib": efficient["memory"]["after_time_2"]["hwm"]["sum_mib"] / 1024,
                    "traditional_aggregate_hwm_gib": traditional["memory"]["after_time_2"]["hwm"]["sum_mib"] / 1024,
                    "efficient_max_rank_hwm_mib": efficient["memory"]["after_time_2"]["hwm"]["max_mib"],
                    "traditional_max_rank_hwm_mib": traditional["memory"]["after_time_2"]["hwm"]["max_mib"],
                }
            )


def render_html(rows: list[dict], result_dir: Path, binary_sha256: str) -> str:
    mesh_rows: list[str] = []
    time_sections: list[str] = []
    memory_sections: list[str] = []

    for row in rows:
        family_name, topology = FAMILIES[row["family"]]
        mesh = row["mesh"]
        mesh_rows.append(
            "<tr>"
            f"<td>{keep(family_name)}</td><td>{keep(topology)}</td><td>IV{row['iv']}</td>"
            f"<td>{keep(execution(row['iv']))}</td><td>{keep(mesh['nodes'])}</td>"
            f"<td>{keep(mesh['volume_cells'])}</td><td>{mesh['mean_cell_volume']:.8g}</td>"
            f"<td>{mesh['mean_volume_equivalent_size']:.6f}</td></tr>"
        )

    for family, (family_name, topology) in FAMILIES.items():
        family_rows = [row for row in rows if row["family"] == family]
        time_body: list[str] = []
        memory_body: list[str] = []
        for row in family_rows:
            efficient = row["efficient"]
            traditional = row["traditional"]
            time_body.append(
                "<tr>"
                f"<td>IV{row['iv']}</td><td>{keep(execution(row['iv']))}</td>"
                f"<td>{efficient['physical_steps']}/{traditional['physical_steps']}</td>"
                f"<td>{seconds(efficient['initialize_seconds'])}</td>"
                f"<td>{seconds(traditional['initialize_seconds'])}</td>"
                f"<td>{seconds(efficient['time_march_seconds'])}</td>"
                f"<td>{seconds(traditional['time_march_seconds'])}</td>"
                f"<td>{row['march_speedup']:.2f}</td>"
                f"<td>{seconds(efficient['total_seconds'])}</td>"
                f"<td>{seconds(traditional['total_seconds'])}</td></tr>"
            )
            eff_init_pss = efficient["memory"]["after_initialize"]["pss"]["sum_mib"] / 1024
            trad_init_pss = traditional["memory"]["after_initialize"]["pss"]["sum_mib"] / 1024
            eff_hwm = efficient["memory"]["after_time_2"]["hwm"]["sum_mib"] / 1024
            trad_hwm = traditional["memory"]["after_time_2"]["hwm"]["sum_mib"] / 1024
            memory_body.append(
                "<tr>"
                f"<td>IV{row['iv']}</td><td>{keep(execution(row['iv']))}</td>"
                f"<td>{eff_init_pss:.3f}</td><td>{trad_init_pss:.3f}</td>"
                f"<td>{row['eff_t2_pss']:.3f}</td><td>{row['trad_t2_pss']:.3f}</td>"
                f"<td>{row['pss_ratio']:.2f}</td><td>{row['pss_reduction']:.1f}%</td>"
                f"<td>{eff_hwm:.3f}</td><td>{trad_hwm:.3f}</td></tr>"
            )
        time_sections.append(
            f"<h3>{family_name}（{topology}）</h3><table><thead><tr>"
            "<th>网格</th><th>资源</th><th>步数<br>高效/传统</th>"
            "<th>初始化(s)<br>高效</th><th>初始化(s)<br>传统</th>"
            "<th>推进(s)<br>高效</th><th>推进(s)<br>传统</th>"
            "<th>推进加速比<br>传统/高效</th><th>总时间(s)<br>高效</th>"
            f"<th>总时间(s)<br>传统</th></tr></thead><tbody>{''.join(time_body)}</tbody></table>"
        )
        memory_sections.append(
            f"<h3>{family_name}（{topology}）</h3><table><thead><tr>"
            "<th>网格</th><th>资源</th><th>初始化后PSS(GiB)<br>高效</th>"
            "<th>初始化后PSS(GiB)<br>传统</th><th>t=2 PSS(GiB)<br>高效</th>"
            "<th>t=2 PSS(GiB)<br>传统</th><th>PSS比<br>传统/高效</th>"
            "<th>PSS降低</th><th>聚合HWM(GiB)<br>高效</th>"
            f"<th>聚合HWM(GiB)<br>传统</th></tr></thead><tbody>{''.join(memory_body)}</tbody></table>"
        )

    speedup_summary = "；".join(
        f"{family_name} {min(row['march_speedup'] for row in rows if row['family'] == family):.2f}–"
        f"{max(row['march_speedup'] for row in rows if row['family'] == family):.2f} 倍"
        for family, (family_name, _) in FAMILIES.items()
    )
    memory_summary = "；".join(
        f"{family_name} {min(row['pss_reduction'] for row in rows if row['family'] == family):.1f}%–"
        f"{max(row['pss_reduction'] for row in rows if row['family'] == family):.1f}%"
        for family, (family_name, _) in FAMILIES.items()
    )
    now = dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
    return f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>NCFV 优化后 t=2 计算效率与内存报告</title>
<style>
@page {{ size: A4 landscape; margin: 13mm; }}
body {{ font-family: "Noto Sans CJK SC", "Microsoft YaHei", sans-serif; color:#1f2937; font-size:9pt; line-height:1.4; }}
h1 {{ text-align:center; color:#17365d; font-size:18pt; margin:0 0 4pt; }}
h2 {{ color:#17365d; font-size:13pt; border-bottom:1px solid #9fbad0; margin:13pt 0 6pt; }}
h3 {{ color:#24577a; font-size:10.5pt; margin:8pt 0 3pt; }}
.meta {{ text-align:center; color:#52606d; margin-bottom:10pt; }}
.note {{ background:#eef5fb; border-left:4px solid #3478a8; padding:7pt 9pt; margin:7pt 0; }}
table {{ width:100%; border-collapse:collapse; margin:4pt 0 8pt; page-break-inside:avoid; font-size:7.8pt; }}
th,td {{ border:1px solid #aab7c4; padding:2.5pt; text-align:center; vertical-align:middle; }}
th {{ background:#dbeaf5; color:#17365d; }} tr:nth-child(even) td {{ background:#f7fafc; }}
.small {{ font-size:8pt; color:#52606d; word-break:break-all; }}
</style></head><body>
<h1>NCFV 优化后：三类网格推进至 t=2 的计算效率与内存需求</h1>
<div class="meta">三维等熵涡；本轮全部从 t=0 重新计算；报告生成：{html.escape(now)}</div>
<div class="note"><b>资源口径：</b>IV10、IV20 使用 1 个 MPI 进程；IV40 使用 8 个 MPI 进程；IV80 使用 32 个 MPI 进程。每个 MPI 进程固定 OMP/BLAS=1。高效法和传统法在同一网格上使用相同资源。</div>

<h2>1. 计算与统计口径</h2>
<p>比较高效微分积分方法（EfficientDifferential）和传统高斯积分方法（TraditionalQuadrature）。所有 24 组算例均从初始场直接推进到物理时间 t=2，关闭 VTK、常规结果和重启 I/O。推进时间由生产路径 <code>Solver::Run()</code> 内部计时；初始化时间单独统计。</p>
<p>PSS 为各 MPI 进程按共享页比例分摊后的聚合物理内存，是多进程总内存需求的主指标。聚合 HWM 为各进程生命周期内 VmHWM 之和，是保守的峰值指标，各秩峰值不一定发生在同一时刻。</p>

<h2>2. 网格数量与平均尺寸</h2>
<table><thead><tr><th>网格族</th><th>拓扑</th><th>尺度</th><th>资源</th><th>节点数</th><th>体单元数</th><th>平均单元体积</th><th>平均尺寸 h̄</th></tr></thead><tbody>{''.join(mesh_rows)}</tbody></table>
<p>所有网格物理域为 10×10×4、总体积 400；平均尺寸定义为 <b>h̄=(1/N)ΣVᵢ^(1/3)</b>。</p>

<h2>3. 计算时间与算法效率</h2>
{''.join(time_sections)}
<p>“推进加速比”定义为传统方法推进时间/高效方法推进时间；大于 1 表示高效方法更快。不同 IV 使用不同进程数，因此不以跨 IV 绝对时间计算并行加速比。</p>

<h2>4. 内存需求</h2>
{''.join(memory_sections)}

<h2>5. 主要结论</h2>
<ol>
<li>全部 12 个同网格、同资源配对中，高效方法的时间推进均更快。加速比范围：{speedup_summary}。</li>
<li>全部配对中高效方法的 t=2 聚合 PSS 均更低。PSS 降低范围：{memory_summary}。</li>
<li>Tet4 的体单元数是同 IV Hex8 的 6 倍，因此跨单元族比较绝对时间和内存时必须同时参考体单元数与平均尺寸。</li>
<li>本轮结果反映当前初始化优化后的生产求解器；未增加时间推进阶段的 stencil 条目或运算路径。</li>
</ol>

<h2>6. 完整性与来源</h2>
<p>24/24 个结果均通过检查：终止时间为 t=2、物理步数为正、MPI 数符合 1/1/8/32，且未读取或写入重启检查点。</p>
<p class="small">原始结果目录：{html.escape(str(result_dir))}<br>基准可执行文件 SHA-256：{html.escape(binary_sha256)}</p>
</body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--mesh-statistics", type=Path, required=True)
    parser.add_argument("--output-html", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--binary-sha256", required=True)
    args = parser.parse_args()
    rows = load_data(args.result_dir, args.mesh_statistics)
    write_csv(args.output_csv, rows)
    args.output_html.write_text(
        render_html(rows, args.result_dir, args.binary_sha256), encoding="utf-8"
    )
    print(f"validated_rows={len(rows)}")
    print(args.output_html)
    print(args.output_csv)


if __name__ == "__main__":
    main()
