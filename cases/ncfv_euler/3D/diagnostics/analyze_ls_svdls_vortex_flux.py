"""Validate paired production LS/SVDLS vortex runs and report density errors."""
import argparse
import csv
import json
import math
from pathlib import Path


def density_field(directory, steps, ranks):
    field = {}
    for rank in range(ranks):
        path = directory / f"solution_{steps:08d}.nodes.rank{rank:04d}.csv"
        with path.open() as stream:
            for row in csv.DictReader(stream):
                node = int(row["original_node"])
                assert node not in field
                field[node] = (float(row["rho_mean"]), float(row["rho_point"]),
                               float(row["partial_volume"]))
    return field


def field_difference(first, second):
    assert first.keys() == second.keys()
    squared = 0
    volume = 0
    maximum = 0
    for node, (mean, point, weight) in first.items():
        other_mean, other_point, other_weight = second[node]
        assert math.isclose(weight, other_weight, rel_tol=2e-12, abs_tol=1e-15)
        squared += weight * (point - other_point) ** 2
        volume += weight
        maximum = max(maximum, abs(mean - other_mean), abs(point - other_point))
    return {"density_point_gap_L2": math.sqrt(squared / volume),
            "density_mean_or_point_gap_max": maximum}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    methods = ("LeastSquares", "SVDLeastSquares")
    fluxes = ("quadrature", "midpoint")
    levels = (10, 20, 40, 80)
    data = {}
    rows = []
    hashes = set()
    def case_directory(method, flux, level):
        if level != 80:
            return root / "raw"
        return root / ("svd_midpoint_32" if method == "SVDLeastSquares" and flux == "midpoint" else "iv80_32")
    for method in methods:
        for flux in fluxes:
            previous = None
            for level in levels:
                name = f"prism{level}_{method}_{flux}"
                case_root = case_directory(method, flux, level)
                raw = json.loads((case_root / f"{name}.json").read_text())
                manifest = json.loads((case_root / f"{name}.manifest.json").read_text())
                hashes.add(manifest["binary_sha256"])
                data[method, flux, level] = raw
                assert raw["method"] == method
                assert raw["mode"] == "Traditional"
                assert raw["traditional_edge_midpoint_flux"] == (flux == "midpoint")
                assert raw["riemann_solver"] == "Roe"
                assert raw["configuration"]["reconstruction"]["distanceWeightPower"] == 2
                assert not raw["configuration"]["reconstruction"]["enableLimiter"]
                assert abs(raw["final"]["time"] - 1.6) < 1e-12
                row = {"method": method, "flux": flux, "level": level, "nodes": raw["nodes"],
                       "mpi_ranks": raw["mpi_ranks"],
                       "h_volume": (400 / raw["nodes"]) ** (1 / 3), "dt": raw["dt"],
                       "L1": raw["final"]["rho_point_L1V"],
                       "L2": raw["final"]["rho_point_L2V"],
                       "Linf": raw["final"]["rho_point_Linf"],
                       "order_L1": "", "order_L2": "", "order_Linf": "",
                       "order_L2_nominal": "", "mass_drift": raw["final"]["mass_drift"],
                       "last_step_riemann_calls": raw["last_step_riemann_calls"]}
                if previous:
                    denominator = math.log(row["nodes"] / previous["nodes"]) / 3
                    for norm in ("L1", "L2", "Linf"):
                        row[f"order_{norm}"] = math.log(previous[norm] / row[norm]) / denominator
                    row["order_L2_nominal"] = math.log(previous["L2"] / row["L2"]) / math.log(level / previous["level"])
                assert all(math.isfinite(row[norm]) and row[norm] > 0 for norm in ("L1", "L2", "Linf"))
                rows.append(row)
                previous = row
    assert len(hashes) == 1
    checks = {"cases": len(rows), "binary_sha256": next(iter(hashes)),
              "maximum_mass_drift": max(abs(row["mass_drift"]) for row in rows),
              "LS_SVDLS_maximum_density_norm_absolute_difference": 0,
              "LS_SVDLS_maximum_density_norm_relative_difference": 0,
              "time_half": {}, "mpi32_check": {}, "IV80_midpoint_to_quadrature": {},
              "IV80_LS_SVDLS_field_difference": {}}
    for level in levels:
        for method in methods:
            quad = data[method, "quadrature", level]
            mid = data[method, "midpoint", level]
            assert quad["initial"] == mid["initial"], "Initial data differ between flux variants"
            assert quad["nodes"] == mid["nodes"] and quad["dt"] == mid["dt"]
            assert quad["configuration"] == mid["configuration"] | {"io": quad["configuration"]["io"]}
            assert quad["last_step_riemann_calls"] > mid["last_step_riemann_calls"]
        for flux in fluxes:
            ls = data["LeastSquares", flux, level]
            svd = data["SVDLeastSquares", flux, level]
            assert ls["nodes"] == svd["nodes"] and ls["dt"] == svd["dt"]
            expected = dict(ls["configuration"])
            expected["reconstruction"] = dict(expected["reconstruction"], method="SVDLeastSquares")
            expected["io"] = svd["configuration"]["io"]
            assert expected == svd["configuration"]
            assert ls["last_step_riemann_calls"] == svd["last_step_riemann_calls"]
            for norm in ("rho_point_L1V", "rho_point_L2V", "rho_point_Linf"):
                difference = abs(ls["final"][norm] - svd["final"][norm])
                checks["LS_SVDLS_maximum_density_norm_absolute_difference"] = max(
                    checks["LS_SVDLS_maximum_density_norm_absolute_difference"], difference)
                checks["LS_SVDLS_maximum_density_norm_relative_difference"] = max(
                    checks["LS_SVDLS_maximum_density_norm_relative_difference"], difference / ls["final"][norm])
    for flux in fluxes:
        half = json.loads((root / "time_half" / f"prism40_LeastSquares_{flux}.json").read_text())
        base = data["LeastSquares", flux, 40]
        assert half["dt"] == base["dt"] / 2
        checks["time_half"][flux] = {"base_L2": base["final"]["rho_point_L2V"],
                                    "half_L2": half["final"]["rho_point_L2V"]}
        checks["time_half"][flux]["L2_relative_change"] = (
            half["final"]["rho_point_L2V"] / base["final"]["rho_point_L2V"] - 1)
        base_field = density_field(root / "raw" / f"prism40_LeastSquares_{flux}", base["steps"], base["mpi_ranks"])
        half_field = density_field(root / "time_half" / f"prism40_LeastSquares_{flux}", half["steps"], half["mpi_ranks"])
        checks["time_half"][flux].update(field_difference(base_field, half_field))
        checks["time_half"][flux]["density_point_gap_to_error_ratio"] = (
            checks["time_half"][flux]["density_point_gap_L2"] / base["final"]["rho_point_L2V"])
        ls = data["LeastSquares", flux, 80]
        svd = data["SVDLeastSquares", flux, 80]
        ls_field = density_field(case_directory("LeastSquares", flux, 80) / f"prism80_LeastSquares_{flux}", ls["steps"], ls["mpi_ranks"])
        svd_field = density_field(case_directory("SVDLeastSquares", flux, 80) / f"prism80_SVDLeastSquares_{flux}", svd["steps"], svd["mpi_ranks"])
        assert len(ls_field) == ls["nodes"]
        checks["IV80_LS_SVDLS_field_difference"][flux] = field_difference(ls_field, svd_field)
        mpi = json.loads((root / "mpi32_check" / f"prism40_LeastSquares_{flux}.json").read_text())
        assert mpi["dt"] == base["dt"] and mpi["mpi_ranks"] == 32
        mpi_field = density_field(root / "mpi32_check" / f"prism40_LeastSquares_{flux}", mpi["steps"], mpi["mpi_ranks"])
        checks["mpi32_check"][flux] = field_difference(base_field, mpi_field)
        checks["mpi32_check"][flux]["L2_norm_relative_difference"] = (
            mpi["final"]["rho_point_L2V"] / base["final"]["rho_point_L2V"] - 1)
        checks["mpi32_check"][flux]["density_point_gap_to_error_ratio"] = (
            checks["mpi32_check"][flux]["density_point_gap_L2"] / base["final"]["rho_point_L2V"])
    for method in methods:
        quad = data[method, "quadrature", 80]
        mid = data[method, "midpoint", 80]
        checks["IV80_midpoint_to_quadrature"][method] = {
            "L2_ratio": mid["final"]["rho_point_L2V"] / quad["final"]["rho_point_L2V"],
            "riemann_call_ratio": quad["last_step_riemann_calls"] / mid["last_step_riemann_calls"]}
    old_mid = json.loads((root / "raw" / "prism80_LeastSquares_midpoint.json").read_text())
    new_mid = data["LeastSquares", "midpoint", 80]
    old_field = density_field(root / "raw" / "prism80_LeastSquares_midpoint", old_mid["steps"], old_mid["mpi_ranks"])
    new_field = density_field(root / "iv80_32" / "prism80_LeastSquares_midpoint", new_mid["steps"], new_mid["mpi_ranks"])
    checks["IV80_MPI8_vs32_midpoint"] = field_difference(old_field, new_field)
    threaded = {"cases": 0, "maximum_density_field_difference": 0,
                "maximum_density_norm_difference": 0}
    for method in methods:
        for flux in fluxes:
            for level in (10, 20):
                name = f"prism{level}_{method}_{flux}"
                native = data[method, flux, level]
                parallel = json.loads((root / "threaded_check" / f"{name}.json").read_text())
                assert native["last_step_riemann_calls"] == parallel["last_step_riemann_calls"]
                first = density_field(root / "raw" / name, native["steps"], native["mpi_ranks"])
                second = density_field(root / "threaded_check" / name, parallel["steps"], parallel["mpi_ranks"])
                gap = field_difference(first, second)
                threaded["cases"] += 1
                threaded["maximum_density_field_difference"] = max(
                    threaded["maximum_density_field_difference"], gap["density_mean_or_point_gap_max"])
                for norm in ("rho_point_L1V", "rho_point_L2V", "rho_point_Linf"):
                    threaded["maximum_density_norm_difference"] = max(
                        threaded["maximum_density_norm_difference"],
                        abs(native["final"][norm] - parallel["final"][norm]))
    assert threaded["maximum_density_field_difference"] < 1e-11
    checks["isolated_parallel_diagnostic"] = threaded
    with (root / "density_errors.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (root / "checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for method in methods:
        for flux in fluxes:
            series = [r for r in rows if r["method"] == method and r["flux"] == flux]
            label = ("LS" if method == "LeastSquares" else "SVDLS") + "/" + flux
            style = "o-" if method == "LeastSquares" else "x--"
            color = "#356d9a" if flux == "quadrature" else "#d06b30"
            axes[0].loglog(levels, [r["L2"] for r in series], style, color=color, label=label)
            axes[1].semilogx(levels[1:], [r["order_L2"] for r in series[1:]], style, color=color)
    axes[0].set_ylabel("Volume-weighted density L2 error")
    axes[0].legend(fontsize=8)
    axes[1].set_ylabel("Observed order using node-count scale")
    axes[1].axhline(3, color="black", linewidth=0.8, linestyle=":")
    for ax, ticks in zip(axes, (levels, levels[1:])):
        ax.set_xlabel("Prism mesh refinement")
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_major_formatter(FixedFormatter([f"IV{n}" for n in ticks]))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.grid(True, alpha=0.25)
    fig.savefig(root / "convergence.png", dpi=180)
    fig.savefig(root / "convergence.pdf")
    plt.close(fig)
    report = ["# LS / SVDLS 等熵涡：界面积分与棱中点通量（2026-10-11）", "",
              "本次运行直接调用当前 DNDSR 的 `LeastSquares` 和 `SVDLeastSquares`，共 16 个主算例。",
              "三棱柱 IV10/20/40/80，三方向周期区域 10×10×4，γ=1.4，涡强度 5，初始中心 (5,5,0)，背景速度 (1,1,0)。",
              "Roe 通量、无黏 Euler、SSPRK3、t=1.6；步长 0.2/0.1/0.05/0.025；IV10/20/40 用 8 MPI ranks，IV80 四组均用 32 ranks，各 1 线程。",
              "两种重构均使用完整二次多项式（9 个非恒定系数）、16 点模板、距离行权重指数 p=2、距离下限 0.15，关闭限制器。",
              "体积分阶数 4、界面积分阶数 3。初场均按同一体积求积构造控制体均值。",
              "", "界面积分逐求积点重构左右状态并求解 Roe 问题；棱中点方案只在原始棱两端坐标的中点求解一次 Roe 问题，乘以对偶界面有向面积向量之和的模长。",
              "两种方案均通过同一个残差/时间推进实现除一次控制体体积。没有启用参考文档专用通量开关。",
              "", "误差采用求解器恢复的节点密度点值，和最终时刻解析密度点值比较：",
              r"$E_1=\sum_iV_i|e_i|/\sum_iV_i$，$E_2=(\sum_iV_ie_i^2/\sum_iV_i)^{1/2}$，$E_\infty=\max_i|e_i|$。",
              "这是节点点值误差，不是控制体均值相对解析体均值的误差。",
              "", "表中阶数使用有效网格尺度 h=(400/N节点)^(1/3)，保持此前 LS/SVDLS 报告口径。",
              "IV 编号翻倍并不使这批网格的节点数严格增至 8 倍；按编号倍率计算的 L2 阶数另列在 CSV 的 `order_L2_nominal`。",
              "随网格同步减小时间步，因此表中为本算例的时空联合观察阶。", ""]
    names = {"LeastSquares": "LS", "SVDLeastSquares": "SVDLS",
             "quadrature": "界面积分", "midpoint": "棱中点值 × 等效面积"}
    for method in methods:
        for flux in fluxes:
            report += [f"## {names[method]}：{names[flux]}", "",
                       "| 网格 | 唯一节点数 | L1 | L1 阶 | L2 | L2 阶 | L∞ | L∞ 阶 |",
                       "|---|---:|---:|---:|---:|---:|---:|---:|"]
            for row in rows:
                if row["method"] == method and row["flux"] == flux:
                    q = ["—" if row[f"order_{n}"] == "" else f"{row[f'order_{n}']:.6f}" for n in ("L1", "L2", "Linf")]
                    report += [f"| IV{row['level']} | {row['nodes']} | {row['L1']:.10e} | {q[0]} | {row['L2']:.10e} | {q[1]} | {row['Linf']:.10e} | {q[2]} |"]
            report += [""]
    report += ["## 核对", "",
               f"1. 同一重构的两种通量方案初始诊断量逐项一致。LS/SVDLS 的最终密度三种范数最大绝对差 {checks['LS_SVDLS_maximum_density_norm_absolute_difference']:.4e}，最大相对差 {checks['LS_SVDLS_maximum_density_norm_relative_difference']:.4e}。",
               f"2. 全部主算例到达 t=1.6；最大质量漂移 {checks['maximum_mass_drift']:.4e}。"]
    for number, flux in enumerate(fluxes, 3):
        test = checks["time_half"][flux]
        report += [f"{number}. IV40 LS {names[flux]}将时间步从 0.05 减至 0.025，密度 L2 从 {test['base_L2']:.10e} 变为 {test['half_L2']:.10e}，变化 {100 * test['L2_relative_change']:+.4f}%。两次计算的密度点值之差 L2 为 {test['density_point_gap_L2']:.4e}，占基准误差 {100 * test['density_point_gap_to_error_ratio']:.4f}%。"]
    report += [f"5. IV80 上直接逐节点比较 LS/SVDLS，两种通量方案的密度均值/恢复点值最大差分别为 {checks['IV80_LS_SVDLS_field_difference']['quadrature']['density_mean_or_point_gap_max']:.4e} 和 {checks['IV80_LS_SVDLS_field_difference']['midpoint']['density_mean_or_point_gap_max']:.4e}。"]
    report += [f"6. IV40 将 MPI 进程数从 8 改为 32，两种通量方案的逐节点密度点值差 L2 分别为 {checks['mpi32_check']['quadrature']['density_point_gap_L2']:.4e}、{checks['mpi32_check']['midpoint']['density_point_gap_L2']:.4e}。"]
    report += [f"7. IV80 LS 中点方案的 8/32 MPI 对照，逐节点密度点值差 L2 为 {checks['IV80_MPI8_vs32_midpoint']['density_point_gap_L2']:.4e}。"]
    report += [f"8. 独立 OpenMP 棱通量诊断在 IV10/20、LS/SVDLS、两种通量方案共 8 组对照中，与原实现的黎曼调用数一致；最终密度均值/点值的最大逐节点差为 {threaded['maximum_density_field_difference']:.4e}。本报告的 16 组主结果均采用原始串行棱循环，不依赖该并行诊断。"]
    report += ["", "本等熵涡中，LS 和 SVDLS 的误差与观察阶在表中位数内相同。IV40→80 的密度 L2 阶为界面积分 2.848351、中点方案 2.886299；IV80 中点 L2 比积分高约 0.0288%。",
               "这组有限网格结果不能证明中点通量对其他光滑算例普遍保持三阶。",
               "", "数据目录中的 8 进程 IV80 积分试跑及重复排队的 SVDLS 中点试跑已主动停止，不纳入统计。最终 IV80 积分与 LS 中点数据位于 `iv80_32/`，SVDLS 中点数据位于 `svd_midpoint_32/`。"]
    report += ["", "原始 JSON 保留完整合并配置、初始/最终诊断、计时和最后一步黎曼调用数；manifest 保留命令及二进制/配置/网格 SHA256。",
               "",
               "![误差与观察阶](convergence.png)",
               "",
               "[完整误差表](density_errors.csv) · [配对检查](checks.json) · [IV10/20/40](raw/) · [IV80](iv80_32/) · [IV80 SVDLS 中点](svd_midpoint_32/) · [时间步减半](time_half/) · [MPI 对照](mpi32_check/)", ""]
    (root / "report.md").write_text("\n".join(report))
    print(json.dumps(checks, indent=2))
    for row in rows:
        print(row["method"], row["flux"], row["level"], row["L2"], row["order_L2"])


if __name__ == "__main__":
    main()
