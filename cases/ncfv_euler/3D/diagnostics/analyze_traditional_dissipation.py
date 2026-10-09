"""Audit traditional first-reconstruction Fnum norms, MPI and quadrature sensitivity."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

NORMS = ("L1", "L2", "Linf")
FIELDS = ("rho", "rhou", "rhov", "rhow", "rhoE")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def check(actual, expected, message):
    require(np.isclose(actual, expected, rtol=3e-10, atol=2e-13), message)


def audit(directory):
    m = json.loads((directory / "metrics.json").read_text())
    require(m["time"] == m["time_steps"] == m["mean_state_max_change"] == 0, "State advanced")
    require(m["reconstruction_calls"] == m["rhs_evaluations"] == 1, "Not first reconstruction")
    require(m["coefficient_rows_min"] == m["coefficient_rows_max"] == 9, "Quadratic coefficients omitted")
    require(m["configuration"]["algorithm"]["mode"] == "TraditionalQuadrature", "Wrong mode")
    require(m["surface_rho_min"] > 0 and m["surface_pressure_min"] > 0, "Nonphysical face state")
    require(m["volume_quadrature_points"] > 0 and m["surface_quadrature_points"] > 0, "No quadrature")
    for key in ("M2_formula_max_mismatch", "face_area_relative_mismatch", "unit_normal_max_mismatch",
                "rhs_decomposition_max_mismatch"):
        require(m[key] < 1e-10, f"Identity failed: {key}")
    require(np.max(np.abs(m["global_rhs_parts_integral"])) < 1e-10, "Nonconservative assembly")
    check(m["volume"], 400.0, "Volume not 400")
    paths = sorted(directory.glob("edges.rank*.csv"))
    require(len(paths) == m["mpi_ranks"], "Missing MPI edge output")
    a = np.concatenate([np.loadtxt(p, delimiter=",", skiprows=1, ndmin=2) for p in paths])
    require(a.shape == (m["edges"], 23) and np.isfinite(a).all(), "Invalid edge data")
    require((a[:, 2:4] > 0).all(), "Nonpositive area or quadrature count")
    require(int(a[:, 3].sum()) == m["surface_quadrature_points"], "Point count mismatch")
    keys = np.sort(a[:, :2].astype(np.int64), axis=1)
    require(np.unique(keys, axis=0).shape[0] == m["edges"], "Repeated MPI edge ownership")
    area = a[:, 2]
    reproduced = {}
    for k, key in enumerate(("jump.rho", "M2.Fnum.rho", "Roe.Fnum.rho", "M2.Fnum.rhoE", "Roe.Fnum.rhoE")):
        offset = 4 + 3*k
        values = a[:, offset:offset+3]
        require((values >= 0).all(), "Invalid per-edge moment")
        v = dict(L1=float(values[:, 0].sum()/area.sum()),
                 L2=float(np.sqrt(values[:, 1].sum()/area.sum())), Linf=float(values[:, 2].max()))
        for norm in NORMS:
            check(v[norm], m["quadrature_norms"][key][norm], f"Quadrature norm mismatch {key}.{norm}")
        reproduced["quadrature."+key] = v
    for k, key in enumerate(("M2.Fnum.rho", "Roe.Fnum.rho", "M2.Fnum.rhoE", "Roe.Fnum.rhoE")):
        x = a[:, 19+k]
        v = dict(L1=float(area@np.abs(x)/area.sum()), L2=float(np.sqrt(area@(x*x)/area.sum())),
                 Linf=float(np.abs(x).max()))
        for norm in NORMS:
            check(v[norm], m["face_mean_norms"][key][norm], f"Face mean norm mismatch {key}.{norm}")
            require(v[norm] <= m["quadrature_norms"][key][norm] + 2e-12, "Averaging increased norm")
        reproduced["face_mean."+key] = v
    for location in ("quadrature_norms", "face_mean_norms"):
        for item in m[location].values():
            check(item["weight"], area.sum(), "Incorrect norm weighting")
    m["independent_aggregate_audit"] = {"passed": True, "norms": reproduced}
    print(f"Audited {directory.name}: {m['surface_quadrature_points']} internal surface points", flush=True)
    return m


def comparable_config(config):
    result = json.loads(json.dumps(config))
    result["mesh"].pop("meshFile")
    for key in ("outputPrefix", "vtkSeriesName", "restartPrefix"):
        result["io"].pop(key, None)
    return result


def order(a, b, ea, eb):
    return float(np.log(ea/eb)/np.log(a["h_3d"]/b["h_3d"])) if min(ea, eb) > 1e-12 else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.data, args.output = args.data.resolve(), args.output.resolve()
    root = Path(__file__).resolve().parents[4]
    runs = []
    for name in ("iv10", "iv20", "iv40", "iv80"):
        m = audit(args.data/name)
        m["mesh"] = name
        runs.append(m)
    cfg = comparable_config(runs[0]["configuration"])
    require(all(comparable_config(r["configuration"]) == cfg for r in runs), "Settings differ across meshes")
    require(cfg["algorithm"]["quadratureOrder"] == 4, "Unexpected baseline quadrature order")
    mpi = audit(args.data/"iv10_np2_check")
    require(mpi["mpi_ranks"] == 2 and runs[0]["mpi_ranks"] == 1, "Unexpected MPI comparison")
    require(comparable_config(mpi["configuration"]) == cfg, "MPI check changed settings")
    mpi_difference = 0.0
    for location in ("quadrature_norms", "face_mean_norms"):
        for key, values in runs[0][location].items():
            for norm in NORMS:
                check(values[norm], mpi[location][key][norm], "MPI reproducibility failed")
                mpi_difference = max(mpi_difference, abs(values[norm]-mpi[location][key][norm]))
    q6 = [audit(args.data/f"{name}_q6") for name in ("iv40", "iv80")]
    for m in q6:
        c = comparable_config(m["configuration"])
        require(c["algorithm"]["quadratureOrder"] == 6, "Wrong high-order rule")
        c["algorithm"]["quadratureOrder"] = 4
        require(c == cfg, "Quadrature check changed other settings")
    rows = []
    for i, run in enumerate(runs):
        run["orders"] = {}
        for location in ("quadrature_norms", "face_mean_norms"):
            run["orders"][location] = {}
            for key, values in run[location].items():
                orders = {n: order(runs[i-1], run, runs[i-1][location][key][n], values[n]) if i else None for n in NORMS}
                run["orders"][location][key] = orders
                rows.append(dict(mesh=run["mesh"], h=run["h_3d"], location=location, quantity=key,
                                 **{n: values[n] for n in NORMS}, **{f"order_{n}": orders[n] for n in NORMS}))
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output/"norms_and_orders.csv").open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    quadrature_check = []
    for i, r in enumerate(runs[-2:]):
        for key in ("jump.rho", "M2.Fnum.rho", "Roe.Fnum.rho", "M2.Fnum.rhoE", "Roe.Fnum.rhoE"):
            a, b = r["quadrature_norms"][key]["L2"], q6[i]["quadrature_norms"][key]["L2"]
            quadrature_check.append(dict(mesh=r["mesh"], quantity=key, q4_L2=a, q6_L2=b, relative_change=b/a-1))
    hashes = {}
    for p in ("src/NCFV/NCFVSpatial.cpp", "src/NCFV/NCFVReconstruction.cpp", "src/NCFV/NCFVDualGeometry.cpp",
              "src/NCFV/NCFVSolver.cpp", "src/Euler/Gas.hpp", "build/src/NCFV/libncfv_3D.a",
              "cases/ncfv_euler/3D/diagnostics/traditional_dissipation_probe.cpp"):
        hashes[p] = hashlib.sha256((root/p).read_bytes()).hexdigest()
    result = {"runs": runs, "q6_runs": q6, "quadrature_check": quadrature_check,
              "mpi_max_norm_difference": mpi_difference, "source_and_library_sha256": hashes}
    (args.output/"metrics.json").write_text(json.dumps(result, indent=2)+"\n")

    def table(location, quantities):
        text = "| 网格 | " + " | ".join(f"{label} L2 | 阶数" for _, label in quantities) + " |\n"
        text += "|---|" + "---:|---:|"*len(quantities) + "\n"
        for r in runs:
            cells = [r["mesh"]]
            for key, _ in quantities:
                p = r["orders"][location][key]["L2"]
                cells += [f'{r[location][key]["L2"]:.7e}', "—" if p is None else f"{p:.4f}"]
            text += "| " + " | ".join(cells) + " |\n"
        return text

    primary = table("quadrature_norms", [("jump.rho", "密度跳跃"), ("M2.Fnum.rho", "M2密度耗散"),
                                         ("Roe.Fnum.rho", "Roe密度耗散")])
    means = table("face_mean_norms", [("M2.Fnum.rho", "M2面平均耗散"), ("Roe.Fnum.rho", "Roe面平均耗散")])
    last = runs[-1]
    component_table = "| 分量 | M2 L1阶 | M2 L2阶 | M2 L∞阶 | Roe L1阶 | Roe L2阶 | Roe L∞阶 |\n|---|---:|---:|---:|---:|---:|---:|\n"
    for field in FIELDS:
        cells = [field]
        for scheme in ("M2", "Roe"):
            cells += ["—" if (v := last["orders"]["quadrature_norms"][f"{scheme}.Fnum.{field}"][n]) is None else f"{v:.4f}" for n in NORMS]
        component_table += "| " + " | ".join(cells) + " |\n"
    qtable = "| 网格 | 项 | q4 L2 | q6 L2 | 相对变化 |\n|---|---|---:|---:|---:|\n"
    for item in quadrature_check:
        qtable += f"| {item['mesh']} | {item['quantity']} | {item['q4_L2']:.7e} | {item['q6_L2']:.7e} | {100*item['relative_change']:.5f}% |\n"
    q6_m2_order = order(q6[0], q6[1], q6[0]["quadrature_norms"]["M2.Fnum.rho"]["L2"], q6[1]["quadrature_norms"]["M2.Fnum.rho"]["L2"])
    q6_roe_order = order(q6[0], q6[1], q6[0]["quadrature_norms"]["Roe.Fnum.rho"]["L2"], q6[1]["quadrature_norms"]["Roe.Fnum.rho"]["L2"])
    old = json.loads((root/"docs/reports/ncfv_first_roe_dissipation_20260907/metrics.json").read_text())
    comparison = "| 网格 | 高效M2耗散 L2 | 传统M2面平均耗散 L2 | 传统/高效 |\n|---|---:|---:|---:|\n"
    for r, efficient in zip(runs, old):
        require(r["periodic_unknowns"] == efficient["periodic_unknowns"], "Comparison mesh mismatch")
        a, b = efficient["face_norms"]["M2.D.rho"]["L2"], r["face_mean_norms"]["M2.Fnum.rho"]["L2"]
        comparison += f"| {r['mesh']} | {a:.7e} | {b:.7e} | {b/a:.6f} |\n"
    report = f"""# 传统三阶 NCFV：首次重构数值耗散阶数

## 计算设置

四套网格为 iv10/20/40/80，物理初始场与上一轮相同：β=5、γ=1.4，涡中心(5,5)，
背景速度(1,1,0)，周期盒[10,10,4]。时间始终为0；每套只初始化一次、重构一次、评估一次空间RHS。
主算例沿用原配置并在诊断实例中切换 `TraditionalQuadrature`，不改动原JSON或生产模块。
限制器与黏性关闭；保留论文归一化及全部9个二次基函数系数，不只保留一阶导数矩阵块。
初始化的控制体均值采用体积分，通量在各子三角面上的积分点求值。
主算例 `quadratureOrder=4`：每个微三角面{last['points_per_micro_triangle']}点、每个微四面体{last['points_per_micro_tetrahedron']}点，
使用现有 DNDSR 的 Hammer 型规则，不将其误称为 Gauss–Lobatto。
固定 CFL=0.5 保留，但没有时间更新，以下量不乘 dt，也不依赖 CFL。

## 只提取 Fnum 耗散修正项

    U_L(x_q) = U_i(point) + C_i^T b_i(x_q)
    U_R(x_q) = U_j(point) + C_j^T b_j(x_q)
    ΔU_q = U_R(x_q) − U_L(x_q)
    Fnum_q = Fhat_q − [F_nq(U_L)+F_nq(U_R)]/2
    Fnum_q(M2) = −α_q ΔU_q/2
    α_q = max(|u_L·n_q|+c_L, |u_R·n_q|+c_R)

`Roe_M2` 在当前库中是局部 Lax–Friedrichs/Rusanov 型标量耗散；
同一份首次重构状态上另调用 `Roe` 选项（含现有熵修正）作对照。
使用每个积分点所属子三角面的单位法向，不能用整个非平面对偶面的平均法向替代。
Fnum只包含耗散修正，不包含中心物理通量，也不以除体积后的残差代替它。
完整RHS分解仅用于验证提取正确，不纳入本报告耗散阶数表。

## 积分点耗散范数（主要结果）

对任一分量 a_q（跳跃或Fnum）：

    L1 = Σ_e,q w_eq |a_eq| / Σ_e,q w_eq
    L2 = sqrt(Σ_e,q w_eq a_eq² / Σ_e,q w_eq)
    L∞ = max_e,q |a_eq|

积分权重仅用于统计范数，不把 S·Fnum 当作单位面积耗散。
L∞是已采样积分点上的最大值，不是连续面上最大值的解析求解。
h=(400/N_periodic)^(1/3)，观察阶=log(E粗/E细)/log(h粗/h细)，不强制网格比等于2。

{primary}
## 最细两级各分量阶数

{component_table}
本算例 rho*w 恒为0；M2对应耗散为0，Roe可能有舍入误差水平的耦合残量，不拟合阶数。

## 整个对偶面的平均耗散（另一种口径）

先计算 Fnum_mean,e = (1/S_e)Σ_q w_eq Fnum_eq，再在边对偶面之间用面积加权求范数。
先平均再取绝对值/平方与先取绝对值/平方再积分不同，前者允许同一对偶面内部的正负抵消。

{means}
与之前高效模式的单位面积耗散相比：

{comparison}
这是两种完整模式的比较，而非只替换Riemann求解器的严格控制变量实验：
传统/高效模式的初始化积分、状态构造和面法向处理也不同。
高效模式直接以估计的左右面均值构造耗散；传统模式逐积分点构造后积分平均，二者非线性操作顺序不同。

## 提高积分精度的交叉检查

iv40、iv80 另外使用 `quadratureOrder=6`，包括初始化体积分和面通量积分一并提高。
新规则每个微三角面{q6[0]['points_per_micro_triangle']}点、每个微四面体{q6[0]['points_per_micro_tetrahedron']}点。
这不是额外时间推进；每个交叉算例仍只有一次重构。

{qtable}
提高积分精度后，最细两级密度耗散L2观察阶为 M2 {q6_m2_order:.4f}、Roe {q6_roe_order:.4f}。

## 核验与边界

- 时间0、推进步数0，守恒均值变化严格为0，完整9行重构矩阵已确认。
- 所有积分点左右状态物理可行，无正性回退；无额外限制器。
- M2标量耗散公式与库输出最大差 {max(r['M2_formula_max_mismatch'] for r in runs):.3e}。
- 中心积分与耗散积分之和组装得到的RHS，与生产RHS最大差 {max(r['rhs_decomposition_max_mismatch'] for r in runs):.3e}。
- 从逐边积分统计量独立复算密度/能量耗散范数，核对面面积、积分点数量及MPI边所有权，全部通过。
- iv10 使用1/2个MPI进程交叉验证，全部范数最大绝对差 {mpi_difference:.3e}。
- 本次验证的是耗散项的网格衰减阶，不是完整空间截断误差阶，也不是时间推进后的解误差阶。

## 文件

- [全部守恒分量、L1/L2/L∞及阶数](norms_and_orders.csv)
- [主算例与q6核验指标、配置和生产源码/库校验值](metrics.json)
- [密度耗散收敛图](convergence.png)
- 原始数据：`{args.data}`，每个目录包含逐边积分统计CSV和完成后的metrics.json。
- 诊断入口：`cases/ncfv_euler/3D/diagnostics/traditional_dissipation_probe.cpp`。
- 编译：`compile_reconstruction_probe.py --source traditional_dissipation_probe.cpp --output /path/to/probe`（source用完整路径）。
- 从 build 运行：`mpirun -np N /path/to/probe /path/to/config.json /path/to/new-output [quadratureOrder]`。
- 输出目录必须不存在；不会覆盖原高效模式或t=2算例结果。
"""
    (args.output/"report.md").write_text(report)
    plt.rcParams.update({"font.family": "DejaVu Sans", "text.usetex": False})
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1), constrained_layout=True)
    h = np.array([r["h_3d"] for r in runs])
    for ax, location, title in zip(axes, ("quadrature_norms", "face_mean_norms"),
                                  ("Density Fnum at quadrature points", "Face-averaged density Fnum")):
        for scheme, marker in (("M2", "o"), ("Roe", "s")):
            ax.loglog(h, [r[location][f"{scheme}.Fnum.rho"]["L2"] for r in runs], marker+"-", label=scheme)
        last_error = runs[-1][location]["M2.Fnum.rho"]["L2"]
        for p, style in ((2, "--"), (3, ":")):
            ax.loglog(h, last_error*(h/h[-1])**p, style, color="gray", label=f"slope {p}")
        ax.set(title=title, xlabel="h = (400 / N_periodic)^(1/3)", ylabel="Weighted L2 magnitude")
        ax.grid(which="both", alpha=0.25)
        ax.legend()
    fig.suptitle("Traditional third-order NCFV / first reconstruction / no time advancement")
    fig.savefig(args.output/"convergence.png", dpi=180)
    plt.close(fig)
    print(primary + "\n" + component_table + "\n" + qtable)
    print(f"q6 density L2 orders: M2={q6_m2_order}, Roe={q6_roe_order}")
    print(f"Report: {args.output/'report.md'}")


if __name__ == "__main__":
    main()
