"""Audit static first-reconstruction Roe/M2 diagnostics and report mesh orders.

Standalone NumPy analysis; does not import DNDSR Python bindings.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIELDS = ("rho", "rhou", "rhov", "rhow", "rhoE")
NORMS = ("L1", "L2", "Linf")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def weighted_norms(values, weights):
    return dict(L1=float(weights @ np.abs(values) / weights.sum()),
                L2=float(np.sqrt(weights @ (values * values) / weights.sum())),
                Linf=float(np.abs(values).max()))


def check_norms(actual, expected, label):
    for key in NORMS:
        require(np.isclose(actual[key], expected[key], rtol=2e-10, atol=2e-13),
                f"Independent norm mismatch: {label}.{key}")


def audit(directory):
    m = json.loads((directory / "metrics.json").read_text())
    require(m["time"] == m["time_steps"] == m["mean_state_max_change"] == 0,
            "Time or physical state was advanced")
    require(m["reconstruction_calls"] == m["rhs_evaluations"] == 1, "Not a first reconstruction")
    require(m["gauss_points_stored"] == 0, "Unexpected Gauss storage")
    require(m["surface_rho_min"] > 0 and m["surface_pressure_min"] > 0, "Nonphysical surface states")
    for key in ("M2_formula_max_mismatch", "Roe_consistency_max_mismatch",
                "Roe_orientation_max_mismatch", "rhs_decomposition_max_mismatch"):
        require(m[key] < 1e-10, f"Failed C++ identity: {key}")
    node_files = sorted(directory.glob("nodes.rank*.csv"))
    edge_files = sorted(directory.glob("edges.rank*.csv"))
    require(len(node_files) == len(edge_files) == m["mpi_ranks"], "Missing rank output")
    nodes = np.concatenate([np.loadtxt(p, delimiter=",", skiprows=1, ndmin=2) for p in node_files])
    require(nodes.shape == (m["nodes"], 25), "Wrong node data shape")
    require(np.isfinite(nodes).all(), "Nonfinite node data")
    require(len(np.unique(nodes[:, 0])) == m["nodes"], "Repeated original node ownership")
    volume = nodes[:, 4]
    require((volume > 0).all() and np.isclose(volume.sum(), 400, atol=1e-9), "Invalid dual volumes")
    independent = {}
    for offset, prefix in ((5, "M2.Rd"), (10, "Roe.Rd"), (15, "central_rhs"), (20, "production_rhs")):
        for j, field in enumerate(FIELDS):
            key = f"{prefix}.{field}"
            independent[key] = weighted_norms(nodes[:, offset+j], volume)
            check_norms(independent[key], m["node_norms"][key], key)
    mismatch = float(np.max(np.abs(nodes[:, 5:10] + nodes[:, 15:20] - nodes[:, 20:25])))
    require(mismatch < 1e-10, "Independent RHS decomposition failed")
    for offset, prefix in ((5, "M2"), (10, "Roe")):
        total = volume @ nodes[:, offset:offset+5]
        require(np.allclose(total, m[f"{prefix}_dissipation_global_integral"], atol=2e-12),
                "Independent global conservation mismatch")
        require(np.max(np.abs(total)) < 1e-10, "Nonconservative dissipation assembly")
    edges = np.concatenate([np.loadtxt(p, delimiter=",", skiprows=1, ndmin=2) for p in edge_files])
    require(edges.shape == (int(m["owned_edges_global"]), 10), "Wrong edge data shape")
    require(np.isfinite(edges).all() and (edges[:, 2:4] > 0).all(), "Invalid face data")
    edge_keys = np.sort(edges[:, :2].astype(np.int64), axis=1)
    require(np.unique(edge_keys, axis=0).shape[0] == edges.shape[0], "Repeated owned original edge")
    area = edges[:, 2]
    for offset, field in ((4, "rho"), (7, "rhoE")):
        for j, prefix in enumerate(("jump", "M2.D", "Roe.D")):
            key = f"{prefix}.{field}"
            independent[key] = weighted_norms(edges[:, offset+j], area)
            check_norms(independent[key], m["face_norms"][key], key)
        for j, prefix in ((1, "M2"), (2, "Roe")):
            key = f"{prefix}.integrated_D.{field}"
            check_norms(weighted_norms(area * edges[:, offset+j], area), m["face_norms"][key], key)
        require(np.max(np.abs(edges[:, offset+1] - 0.5 * edges[:, 3] * edges[:, offset])) < 1e-12,
                "Independent M2 = alpha/2 * jump identity failed")
    m["independent_audit"] = {"passed": True, "norms": independent,
                              "rhs_decomposition_max_mismatch": mismatch}
    return m


def comparable_config(configuration):
    c = json.loads(json.dumps(configuration))
    c["mesh"].pop("meshFile")
    for key in ("outputPrefix", "vtkSeriesName", "restartPrefix"):
        c["io"].pop(key, None)
    return c


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.data, args.output = args.data.resolve(), args.output.resolve()
    runs = []
    for name in ("iv10", "iv20", "iv40", "iv80"):
        m = audit(args.data / name)
        m["mesh"] = name
        runs.append(m)
        print(f"Audited {name}: faces={m['owned_edges_global']:.0f}, time={m['time']}", flush=True)
    require(all(comparable_config(r["configuration"]) == comparable_config(runs[0]["configuration"])
                for r in runs), "Non-mesh settings differ")
    mpi_check = None
    if (args.data / "iv10_np2_check" / "metrics.json").exists():
        other = audit(args.data / "iv10_np2_check")
        require(other["mpi_ranks"] == 2 and runs[0]["mpi_ranks"] == 1,
                "Unexpected MPI reproducibility pair")
        require(comparable_config(other["configuration"]) == comparable_config(runs[0]["configuration"]),
                "MPI check changed the numerical configuration")
        maximum = 0.0
        for section in ("face_norms", "node_norms"):
            for key in runs[0][section]:
                for norm in NORMS:
                    a, b = runs[0][section][key][norm], other[section][key][norm]
                    require(np.isclose(a, b, rtol=1e-10, atol=1e-12), "MPI norm mismatch")
                    maximum = max(maximum, abs(a-b))
        mpi_check = {"passed": True, "mesh": "iv10", "ranks": [1, 2], "max_norm_difference": maximum}
    rows = []
    for i, run in enumerate(runs):
        run["orders"] = {}
        for location in ("face_norms", "node_norms"):
            for key, values in run[location].items():
                orders = {}
                for norm in NORMS:
                    previous = runs[i-1][location][key][norm] if i else 0
                    # Machine-roundoff and identically-zero quantities have no meaningful order.
                    orders[norm] = (float(np.log(previous / values[norm]) /
                                          np.log(runs[i-1]["h_3d"] / run["h_3d"]))
                                    if i and previous > 1e-12 and values[norm] > 1e-12 else None)
                run["orders"][key] = orders
                rows.append(dict(mesh=run["mesh"], h=run["h_3d"], location=location, quantity=key,
                                 **{n: values[n] for n in NORMS},
                                 **{f"order_{n}": orders[n] for n in NORMS}))
    args.output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[4]
    sources = ("src/NCFV/NCFVSpatial.cpp", "src/NCFV/NCFVReconstruction.cpp",
               "src/Euler/Gas.hpp", "build/src/NCFV/libncfv_3D.a",
               "cases/ncfv_euler/3D/diagnostics/first_roe_dissipation_probe.cpp")
    provenance = {"sha256": {p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in sources},
                  "mpi_reproducibility": mpi_check}
    (args.output / "verification.json").write_text(json.dumps(provenance, indent=2) + "\n")
    with (args.output / "errors_and_orders.csv").open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "metrics.json").write_text(json.dumps(runs, indent=2) + "\n")

    def table(keys):
        title = "| 网格 | " + " | ".join(f"{label} L2 | 阶数" for _, _, label in keys) + " |\n"
        result = title + "|---|" + "---:|---:|" * len(keys) + "\n"
        for run in runs:
            cells = [run["mesh"]]
            for location, key, _ in keys:
                order = run["orders"][key]["L2"]
                cells += [f'{run[location][key]["L2"]:.7e}', "—" if order is None else f"{order:.4f}"]
            result += "| " + " | ".join(cells) + " |\n"
        return result

    rho_table = table([("face_norms", "jump.rho", "密度面均值跳跃"),
                       ("face_norms", "M2.D.rho", "M2密度单位面积耗散"),
                       ("node_norms", "M2.Rd.rho", "M2密度耗散RHS")])
    roe_table = table([("face_norms", "Roe.D.rho", "Roe密度单位面积耗散"),
                       ("node_norms", "Roe.Rd.rho", "Roe密度耗散RHS")])
    energy_table = table([("face_norms", "M2.D.rhoE", "M2能量单位面积耗散"),
                          ("node_norms", "M2.Rd.rhoE", "M2能量耗散RHS"),
                          ("node_norms", "Roe.Rd.rhoE", "Roe能量耗散RHS")])
    control_table = table([("face_norms", "control1.jump.rho", "仅换解析点值"),
                           ("face_norms", "control2.jump.rho", "仅换解析梯度"),
                           ("face_norms", "control3.jump.rho", "点值梯度均解析")])
    finest = runs[-1]
    mpi_note = (f"- iv10 额外采用1/2个 MPI 进程复算，全部范数一致，最大绝对差 {mpi_check['max_norm_difference']:.3e}。"
                if mpi_check else "")
    report = f"""# 首次重构后的 Roe 数值耗散：四网格静态验证

## 范围与实际格式

沿用上一轮 t=2 算例的四套网格、物理场及论文归一化后的高效格式。
每套只初始化一次、重构一次、计算一次空间 RHS；时间和时间步数始终为0，守恒均值未改变。
未调用 Solver.Run，未改动 NCFV/Euler 等生产模块，没有 Gauss 点存储，也没有积分点参考解。
原配置 CFL=0.5 保留，但本次没有时间更新，以下耗散不乘 dt，CFL 不决定这些静态数值。

原计算实际选项为 `Roe_M2`。当前 `src/Euler/Gas.hpp` 的 eigScheme=2 分支提前返回
局部 Lax–Friedrichs/Rusanov 通量，使用标量最大波速，并非标准 Roe 特征分解耗散。
本报告以实际 M2 为主，同时对完全相同的首次重构左右状态调用 DNDSR 的 `Roe` 选项作对照
（包含库中现有的熵修正），不改变主计算配置。

## 分解与量纲

高效算法给 Roe 的左右状态是整个边对偶面的左右估计面均值，不是原始边端点值，也不是 Gauss 点值：

    U_L = U_i(point) + (1/S_e) Σ_k G_k^T w^L_ek
    U_R = U_j(point) + (1/S_e) Σ_k G_k^T w^R_ek
    ΔU_e = U_R − U_L
    D_e = [F_n(U_L)+F_n(U_R)]/2 − F_num(U_L,U_R)
    D_e(M2) = α_e ΔU_e/2
    α_e = max(|u_L·n|+c_L, |u_R·n|+c_R)
    Φ_e = C_e − S_e D_e
    R_D,i = (1/V_i) Σ_e s_ie S_e D_e

其中 C_e 是生产程序高效全微分积分所得左右物理通量积分的平均。
因此生产 RHS = 中心积分 RHS + R_D。本次对所有五个守恒分量逐节点核对此恒等式。
S_e 严格用生产程序的标量对偶面面积，n 用面矢量归一化；二者不互相替代。
周期边界份额按原程序的体积加权规则合并，再计算耗散 RHS 范数。

必须区分三种量：D_e 是单位面积通量耗散；S_e D_e 是面积积分耗散；R_D 是守恒平均值的变化率贡献。
不能仅因三维面面积带来 h²，就把积分耗散的高幂次称为格式高阶精度。
压力不是独立守恒通量分量，故这里报告密度、动量及总能量耗散，不虚构“压力通量”。

## 范数和阶数

界面 L2 = sqrt(Σ_e S_e |D_e|² / Σ_e S_e)，节点 L2 = sqrt(Σ_i V_i |R_D,i|² / Σ_i V_i)。
L1 使用相同权重，L∞取最大绝对值；全部分量和三种范数见CSV。
每条原始边仅统计其 MPI 所有者；节点使用原始节点的对偶体积份额，周期总量不重复计数。
h=(400/N_periodic)^(1/3)，阶数=log(E粗/E细)/log(h粗/h细)，不强制网格比为2。
这些是耗散幅值的网格观察阶，不是时间推进后的解误差阶，也不等同于所有情况下的等效黏性系数。

## 当前 M2：密度

{rho_table}
## 同一重构状态上的标准 Roe 对照：密度

{roe_table}
## 能量分量

{energy_table}
## 面均值跳跃的辅助对照

以下仅在诊断通量计算中用解析点值/梯度替换对应输入；不重新求重构矩阵、不重新初始化均值、不推进时间。
三组都保留相同面权重。它们不是新的生产算例，也不能将非线性误差简单相加。

{control_table}
## 阶数量级如何解释

本组波速 α 的最小/最大范围为 {min(r['alpha_min'] for r in runs):.6g}～{max(r['alpha_max'] for r in runs):.6g}，为 O(1)。
因此当前 M2 的单位面积耗散与左右状态跳跃同阶；标准 Roe 则按特征波速度作用于同一跳跃。
若光滑问题的左右面均值均为三阶近似，则 ΔU=O(h³)，D=O(h³)，三维 S D=O(h⁵)。
经面积求和、除体积后，一般不依赖额外抵消的估计是 R_D=O(h²)；
规则/对称模板上的额外抵消可能提高它的观察阶，不能预先假设存在。
当前实测最细两级密度 M2 的 D 阶为 {finest['orders']['M2.D.rho']['L2']:.4f}，
R_D 阶为 {finest['orders']['M2.Rd.rho']['L2']:.4f}；
Roe 的相应阶数为 {finest['orders']['Roe.D.rho']['L2']:.4f}、{finest['orders']['Roe.Rd.rho']['L2']:.4f}。
就当前密度 L2 结果看，单位面积耗散随加密趋近三阶衰减，耗散 RHS 的观察阶约2.8；
粗网格区间的阶数明显更低。不能据此将先前 t=2 的误差全部归因于一个“始终一阶”的耗散项。
仅凭耗散项本身不能断定完整空间算子的阶数：中心积分项也有截断误差，二者可能抵消或叠加。

## 核验

- 所有算例时间0、步数0、重构1次，均值变化严格为0；所有表面状态物理可行，无正性回退。
- 标量 M2 公式与库输出最大差：{max(r['M2_formula_max_mismatch'] for r in runs):.3e}。
- 中心积分 RHS 加耗散 RHS 与生产 RHS 最大差：{max(r['rhs_decomposition_max_mismatch'] for r in runs):.3e}。
- 标准 Roe 左右相同状态一致性与交换左右/翻转法向测试均通过。
- 独立读取全部 MPI 输出，复算密度/能量界面范数、全部守恒分量节点范数与全域守恒积分，均通过。
- Gauss 点存储数为0；MPI 进程数依次为 {', '.join(str(r['mpi_ranks']) for r in runs)}。
{mpi_note}

## 文件与复现

- [全部分量、三种范数及阶数](errors_and_orders.csv)
- [完整配置及核验指标](metrics.json)
- [生产源码与链接库校验值、MPI一致性](verification.json)
- [耗散随网格尺度变化图](dissipation_convergence.png)
- 原始逐边/逐节点数据：`{args.data}`。
- 诊断入口：`cases/ncfv_euler/3D/diagnostics/first_roe_dissipation_probe.cpp`。
- 使用 `compile_reconstruction_probe.py --source ... --output ...` 链接当前 NCFV 库；
  从 build 目录执行 `mpirun -np N /path/to/probe /path/to/original-config.json /path/to/new-output`。
  输出目录必须不存在，诊断会关闭求解器常规输出及重启写入，不覆盖原 t=2 结果。
- 汇总：`venv/bin/python cases/ncfv_euler/3D/diagnostics/analyze_first_roe_dissipation.py --data {args.data} --output {args.output}`。
"""
    (args.output / "report.md").write_text(report)
    plt.rcParams.update({"font.family": "DejaVu Sans", "text.usetex": False})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    h = np.array([r["h_3d"] for r in runs])
    for ax, location, suffix, title in zip(axes, ("face_norms", "node_norms"), ("D", "Rd"),
                                          ("Per-area density dissipation", "Density dissipation RHS")):
        for scheme, marker in (("M2", "o"), ("Roe", "s")):
            values = np.array([r[location][f"{scheme}.{suffix}.rho"]["L2"] for r in runs])
            ax.loglog(h, values, marker + "-", label=scheme)
        end = runs[-1][location][f"M2.{suffix}.rho"]["L2"]
        for order, style in ((2, "--"), (3, ":")):
            ax.loglog(h, end * (h/h[-1])**order, style, color="gray", label=f"slope {order}")
        ax.set(xlabel="h = (400 / N_periodic)^(1/3)", ylabel="Weighted L2 magnitude", title=title)
        ax.grid(which="both", alpha=0.25)
        ax.legend()
    fig.suptitle("NCFV efficient / first reconstruction / t=0 / no time stepping")
    fig.savefig(args.output / "dissipation_convergence.png", dpi=180)
    plt.close(fig)
    print(rho_table + "\n" + roe_table)
    print(f"Report: {args.output / 'report.md'}")


if __name__ == "__main__":
    main()
