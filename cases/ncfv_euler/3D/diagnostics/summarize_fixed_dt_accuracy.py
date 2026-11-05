#!/usr/bin/env python3
"""Validate and summarize fixed-step Hex8/Tet4 NCFV accuracy probes."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path


CASE_PATTERN = re.compile(r"^(hex|tet)_iv(10|20|40|80)_(efficient|traditional)$")
GRID_SIZES = (10, 20, 40, 80)
MODES = ("efficient", "traditional")
FAMILIES = ("hex", "tet")
TIME_STEPS = {10: 1.0 / 5.0, 20: 1.0 / 10.0, 40: 1.0 / 20.0, 80: 1.0 / 40.0}


def read_cases(root: Path) -> list[dict]:
    cases: list[dict] = []
    for path in sorted(root.glob("*/accuracy.json")):
        match = CASE_PATTERN.match(path.parent.name)
        if match is None:
            continue
        family, n_text, mode = match.groups()
        n = int(n_text)
        payload = json.loads(path.read_text(encoding="utf-8"))
        cfg = payload["configuration"]
        final = payload["final"]
        expected_mode = "EfficientDifferential" if mode == "efficient" else "TraditionalQuadrature"
        expected_dt = TIME_STEPS[n]
        expected_iterations = round(1.6 / expected_dt)
        failures: list[str] = []
        if cfg["algorithm"]["mode"] != expected_mode:
            failures.append(f"algorithm={cfg['algorithm']['mode']}")
        if cfg["reconstruction"]["method"] != "SVDLeastSquares":
            failures.append(f"reconstruction={cfg['reconstruction']['method']}")
        if abs(cfg["time"]["timeStep"] - expected_dt) > 1e-13:
            failures.append(f"dt={cfg['time']['timeStep']}")
        if abs(cfg["time"]["endTime"] - 1.6) > 1e-13:
            failures.append(f"endTime={cfg['time']['endTime']}")
        if final["iteration"] != expected_iterations:
            failures.append(f"iterations={final['iteration']}")
        if abs(final["time"] - 1.6) > 1e-11:
            failures.append(f"final_time={final['time']}")
        if not (final["stencil_size_min"] == final["stencil_size_mean"] == final["stencil_size_max"] == 16):
            failures.append("stencil is not uniformly 16")
        if mode == "efficient" and final["gauss_points_stored"] != 0:
            failures.append(f"efficient gauss={final['gauss_points_stored']}")
        if mode == "traditional" and final["gauss_points_stored"] <= 0:
            failures.append("traditional gauss count is non-positive")
        cases.append(
            {
                "family": family,
                "n": n,
                "mode": mode,
                "path": path,
                "payload": payload,
                "failures": failures,
            }
        )
    return cases


def error(case: dict, field: str, norm: str) -> float:
    return case["payload"]["final"][field][norm]


def order(coarse: dict, fine: dict, field: str, norm: str) -> float:
    return math.log(error(coarse, field, norm) / error(fine, field, norm), 2.0)


def write_csv(cases: list[dict], output: Path) -> None:
    fields = [
        "family", "iv", "mode", "mpi_ranks", "dt", "steps", "final_time",
        "rho_L1", "rho_L2", "rho_Linf", "pressure_L1", "pressure_L2", "pressure_Linf",
        "gauss_points_stored", "wall_seconds", "reconstruction",
    ]
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for case in sorted(cases, key=lambda c: (c["family"], c["n"], c["mode"])):
            payload = case["payload"]
            cfg, final = payload["configuration"], payload["final"]
            writer.writerow(
                {
                    "family": case["family"], "iv": case["n"], "mode": case["mode"],
                    "mpi_ranks": payload["mpi_ranks"], "dt": cfg["time"]["timeStep"],
                    "steps": final["iteration"], "final_time": final["time"],
                    "rho_L1": final["rho_point"]["L1"], "rho_L2": final["rho_point"]["L2"],
                    "rho_Linf": final["rho_point"]["Linf"],
                    "pressure_L1": final["pressure_point"]["L1"],
                    "pressure_L2": final["pressure_point"]["L2"],
                    "pressure_Linf": final["pressure_point"]["Linf"],
                    "gauss_points_stored": final["gauss_points_stored"],
                    "wall_seconds": payload["wall_seconds"],
                    "reconstruction": cfg["reconstruction"]["method"],
                }
            )


def write_report(cases: list[dict], output: Path) -> None:
    lookup = {(c["family"], c["n"], c["mode"]): c for c in cases}
    lines = [
        "# Hex8/Tet4 固定时间步精度对照（t=1.6）",
        "",
        "## 设置",
        "",
        "- 网格：周期 Hex8 和 Tet4，IV = 10、20、40、80；每级空间尺度比为 2。",
        "- 时间：固定 `dt = 1/5, 1/10, 1/20, 1/40`，对应 8、16、32、64 步，终止时间 `t=1.6`。",
        "- 共同设置：无粘 Roe 格式、三阶 SSPRK、无激波限制器、配置默认的 16 点 `SVDLeastSquares` 重构。",
        "- 对照：`efficient` 为高效微分积分，`traditional` 为传统体积/界面积分。",
        "- 误差：解析等熵涡解的对偶体积加权恢复节点值误差。下表列出最终时刻的 L2 范数。",
        "",
        "## 最终误差",
        "",
        "| 网格 | IV | dt | 步数 | 高效 rho L2 | 传统 rho L2 | 传统/高效 | 高效 p L2 | 传统 p L2 | 传统/高效 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for family in FAMILIES:
        for n in GRID_SIZES:
            efficient = lookup[(family, n, "efficient")]
            traditional = lookup[(family, n, "traditional")]
            rho_e = error(efficient, "rho_point", "L2")
            rho_t = error(traditional, "rho_point", "L2")
            p_e = error(efficient, "pressure_point", "L2")
            p_t = error(traditional, "pressure_point", "L2")
            steps = efficient["payload"]["final"]["iteration"]
            lines.append(
                f"| {family.upper()} | {n} | {TIME_STEPS[n]:.5f} | {steps} | {rho_e:.8e} | {rho_t:.8e} | {rho_t / rho_e:.3f} | {p_e:.8e} | {p_t:.8e} | {p_t / p_e:.3f} |"
            )

    lines += [
        "",
        "## 相邻加密观察阶（L2）",
        "",
        "`p = log2(E_IVN / E_IV2N)`。因网格尺度和时间步同时减半，这是空间与时间误差的合成观察阶。",
        "",
        "| 网格 | 方法 | 10→20 rho | 20→40 rho | 40→80 rho | 10→20 p | 20→40 p | 40→80 p |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for family in FAMILIES:
        for mode in MODES:
            rows = [lookup[(family, n, mode)] for n in GRID_SIZES]
            rho_orders = [order(rows[i], rows[i + 1], "rho_point", "L2") for i in range(3)]
            p_orders = [order(rows[i], rows[i + 1], "pressure_point", "L2") for i in range(3)]
            lines.append(
                f"| {family.upper()} | {mode} | " + " | ".join(f"{x:.3f}" for x in rho_orders + p_orders) + " |"
            )

    lines += [
        "",
        "## 求积存储与运行审计",
        "",
        "| 网格 | IV | 方法 | MPI | 存储求积点 | 求解器墙钟时间 (s) |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for family in FAMILIES:
        for n in GRID_SIZES:
            for mode in MODES:
                case = lookup[(family, n, mode)]
                payload, final = case["payload"], case["payload"]["final"]
                lines.append(
                    f"| {family.upper()} | {n} | {mode} | {payload['mpi_ranks']} | {final['gauss_points_stored']} | {payload['wall_seconds']:.3f} |"
                )
    lines += [
        "",
        "墙钟时间使用了随网格调整的 MPI 进程数，只用于运行审计，不应用作两种算法的严格性能比较。",
        "",
        "## 审计结果",
        "",
        "全部 16 个算例均验证了积分模式、固定时间步、终止时间、步数、默认重构和 16 点模板。",
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="directory containing *_iv*_{efficient,traditional}/accuracy.json")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    output_dir = (args.output_dir or root).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = read_cases(root)
    expected = {(family, n, mode) for family in FAMILIES for n in GRID_SIZES for mode in MODES}
    actual = {(case["family"], case["n"], case["mode"]) for case in cases}
    missing = sorted(expected - actual)
    failures = [f"{case['path']}: {', '.join(case['failures'])}" for case in cases if case["failures"]]
    if missing or failures:
        details = []
        if missing:
            details.append("missing=" + repr(missing))
        details.extend(failures)
        raise SystemExit("\n".join(details))
    write_csv(cases, output_dir / "accuracy_summary.csv")
    write_report(cases, output_dir / "accuracy_report.md")


if __name__ == "__main__":
    main()
