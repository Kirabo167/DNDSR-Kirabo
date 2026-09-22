"""Audit and summarize the static density-gradient implementation checks."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]


def read_run(directory):
    metadata = json.loads((directory / "metrics.json").read_text())
    if metadata["diagnostic_revision"] != 2 or metadata["production_operator_modified"]:
        raise ValueError("Wrong diagnostic revision or modified production operator")
    if metadata["time_steps"] != 0 or metadata["rhs_evaluations"] != 0:
        raise ValueError("This report is limited to static reconstruction")
    files = sorted(directory.glob("nodes.rank*.csv"))
    if len(files) != metadata["mpi_ranks"]:
        raise ValueError("Missing rank output")
    data = np.concatenate([np.genfromtxt(file, names=True, delimiter=",", ndmin=1) for file in files])
    data.sort(order="original_node")
    if not np.array_equal(data["original_node"], np.arange(metadata["nodes"])):
        raise ValueError("Missing or duplicate owned nodes")
    if not np.isfinite(data.view(np.float64)).all() or np.any(data["volume"] <= 0):
        raise ValueError("Invalid node data")
    volume = data["volume"].sum()
    if abs(volume - 400) > 1e-9:
        raise ValueError("Incorrect total volume")
    mapping = {
        "rho_initial_error": "rho_initial", "rho_reference_error": "rho_reference6",
        "cubic_error": "rho_cubic_taylor_term", "higher_remainder_error": "rho_higher_taylor_remainder",
        "init_gradient_change": "rho_initialization_gradient_change",
        "compact15_error": "rho_compact15_initial", "power4_error": "rho_power4_initial",
    }
    for column, key in mapping.items():
        error = data[column]
        independent = (np.sum(data["volume"]*error)/volume,
                       np.sqrt(np.sum(data["volume"]*error**2)/volume), error.max())
        for norm, value in zip(("L1", "L2", "Linf"), independent):
            if not np.isclose(value, metadata["errors"][key][norm], atol=2e-13, rtol=3e-11):
                raise ValueError(f"Independent error norm disagrees: {column} {norm}")
    errors = metadata["errors"]
    leading = errors["rho_cubic_taylor_term"]["L2"]
    higher = errors["rho_higher_taylor_remainder"]["L2"]
    cross = metadata["cubic_higher_inner_product"]
    reconstructed_squared = leading**2 + higher**2 + 2*cross
    if not np.isclose(reconstructed_squared, errors["rho_reference6"]["L2"]**2, atol=1e-15, rtol=1e-11):
        raise ValueError("Taylor error decomposition does not close")
    metadata["cubic_higher_cosine"] = cross / (leading*higher)
    metadata["higher_to_cubic_norm_ratio"] = higher / leading
    metadata["independent_node_reduction_verified"] = True
    metadata["source_directory"] = str(directory)
    return metadata, data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/out/NCFV/gradient_audit_v2")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/reports/ncfv_gradient_audit")
    args = parser.parse_args()
    previous = json.loads((ROOT / "docs/reports/ncfv_first_reconstruction/metrics.json").read_text())["runs"]
    args.output.mkdir(parents=True, exist_ok=True)
    runs = []
    for n, old in zip((10,20,40,80), previous):
        run, nodes = read_run(args.data / f"iv{n}")
        if n == 10:
            first_nodes = nodes
        if abs(run["errors"]["rho_initial"]["L2"] - old["errors"]["primitive.rho.gradient"]["L2"]) > 1e-13:
            raise ValueError("New baseline disagrees with first-reconstruction check")
        run["mesh"] = f"iv{n}"
        run["orders"] = {}
        if runs:
            for key, error in run["errors"].items():
                run["orders"][key] = {
                    norm: float(np.log(runs[-1]["errors"][key][norm]/error[norm]) / np.log(runs[-1]["h"]/run["h"]))
                    if min(error[norm], runs[-1]["errors"][key][norm]) > 1e-12 else None
                    for norm in ("L1","L2","Linf")}
        runs.append(run)
    repeated, repeat_nodes = read_run(args.data / "iv10_np2")
    max_node_difference = float(np.max(np.abs(first_nodes.view(np.float64)-repeat_nodes.view(np.float64))))
    if max_node_difference > 2e-12:
        raise ValueError(f"MPI node diagnostics disagree: {max_node_difference}")
    mpi_check = {
        "mesh": "iv10", "ranks": [1,2], "maximum_node_diagnostic_difference": max_node_difference,
        "baseline_gradient_L2_difference": abs(repeated["errors"]["rho_initial"]["L2"]-runs[0]["errors"]["rho_initial"]["L2"]),
        "compact_control_gradient_L2_difference": abs(repeated["errors"]["rho_compact15_initial"]["L2"]-runs[0]["errors"]["rho_compact15_initial"]["L2"]),
    }
    (args.output / "evidence.json").write_text(json.dumps({"runs":runs,"mpi_check":mpi_check},indent=2)+"\n")
    with (args.output / "errors_and_orders.csv").open("w",newline="") as stream:
        writer=csv.writer(stream)
        writer.writerow(("mesh","h","test","L1","L2","Linf","order_L1","order_L2","order_Linf"))
        for run in runs:
            for key,error in run["errors"].items():
                writer.writerow((run["mesh"],run["h"],key,*[error[n] for n in ("L1","L2","Linf")],
                                 *[run["orders"].get(key,{}).get(n) for n in ("L1","L2","Linf")]))

    def error_table(keys, labels):
        lines=["| 网格 | "+" | ".join(label+" L2 | 阶数" for label in labels)+" |",
               "|---|"+"---:|---:|"*len(keys)]
        for run in runs:
            row=[run["mesh"]]
            for key in keys:
                order=run["orders"].get(key,{}).get("L2")
                row += [f'{run["errors"][key]["L2"]:.6e}', "—" if order is None else f"{order:.3f}"]
            lines.append("| "+" | ".join(row)+" |")
        return "\n".join(lines)

    checks=["| 网格 | 独立二次再现缺陷 | 独立完整SVD梯度差 | 尺度换算不变性差 | 条件数 |",
            "|---|---:|---:|---:|---:|"]
    cancellation=["| 网格 | 三次项梯度贡献 L2 | 其余项 L2 | 夹角余弦 | 合计梯度误差 L2 |",
                  "|---|---:|---:|---:|---:|"]
    radius=["| 网格 | 模板中心最大xy距离的加权平均 | 梯度误差/h² | z半周期模板覆盖体积比例 |",
            "|---|---:|---:|---:|"]
    for run in runs:
        check=run["checks"];error=run["errors"]
        checks.append(f'| {run["mesh"]} | {check["independent_quadratic_identity_defect"]:.3e} | '
                      f'{check["independent_full_svd_gradient_difference"]:.3e} | {check["length_scale_invariance_difference"]:.3e} | {check["condition_number"]:.3f} |')
        cancellation.append(f'| {run["mesh"]} | {error["rho_cubic_taylor_term"]["L2"]:.6e} | '
                            f'{error["rho_higher_taylor_remainder"]["L2"]:.6e} | {run["cubic_higher_cosine"]:.4f} | '
                            f'{error["rho_reference6"]["L2"]:.6e} |')
        radius.append(f'| {run["mesh"]} | {run["volume_weighted_stencil_radius_xy"]:.4f} | '
                      f'{error["rho_initial"]["L2"]/run["h"]**2:.6f} | {run["halfbox_z_volume_fraction"]:.3f} |')
    report=r"""# NCFV 密度梯度低阶现象：实现与误差来源复核

## 结论

本次证据支持：iv10→iv20→iv40 的低表观阶主要来自当前宽模板对等熵涡的有限分辨率，
不是漏掉二次项、错误截取矩阵行或少除/多除一次尺度。
粗网格上，高阶余项与主导的二阶梯度误差明显反向抵消，使总误差尚不能用固定系数 C h² 描述。
随网格加密，余项迅速减小，观察阶向二阶逼近。

这不是“所有实现都已证明无错”的声明，也不处理完整流场时间推进的全部误差来源。
此处不计算通量、残差或时间步，因此 Roe/LLF 选择不参与本次梯度误差。
生产求解器、初始化算法、矩阵和模板均未修改。

## 1. 复现当前结果与初始化对照

使用同一解析等熵涡，检查梯度向量的体积加权 L2；
h=(400/N_periodic)^(1/3)，阶数使用实际加密比。
“原初始化”是高效全微分均值，“参考均值”是仅在诊断中用六阶规则积分的解析密度体均值。
参考积分不写入生产高效算法的 Gauss 点数组。

__INITIAL_TABLE__

换参考均值后，粗网格阶数仍为约1.31和1.79。
原均值与参考均值导致的梯度差 L2，iv10/20/40/80 分别约为
1.11e-3、3.78e-4、3.89e-5、4.03e-6；不是造成当前阶数差异的主因。
参考积分由五阶升为六阶后，梯度场差 L2 最大为3.20e-9，明显小于本次研究的误差。

## 2. 不是只看源代码：对每个节点独立检查矩阵

从对偶微四面体顶点独立积分中心一、二阶矩，再按周期平移组装各邻居的体平均二次基，
得到 A_ind。该过程不直接复用生产缓存矩来构造测试右端。
对实际保存的3×Ns梯度矩阵 P_g，检查

\[
P_g A_{ind}\simeq[\,I_3\;0_{3\times6}\,].
\]

这个检验同时覆盖三项线性基与六项二次基；常数项在均值差中严格消去。
它是在局部展开坐标/周期平移像中做的多项式一致性测试，
不是把一个非周期全局二次函数错误地跨周期边界连接。

另用 A_ind 重新求完整九行加权SVD伪逆，再取前三行计算实际梯度；
还将归一化长度改为2h，相应调整一次/二次列与最后的1/h，检查物理梯度不变。

__CHECK_TABLE__

表中的二次再现缺陷为无量纲矩阵最大绝对元素误差。
独立完整SVD所得实际梯度与生产梯度的最大差不超过9.1e-14，
远低于0.059、0.024、0.007这些观察误差。
所以本次未发现能解释低阶的基函数、截取、尺度换算或病态矩阵问题。

源码关键位置：

- `src/NCFV/NCFVReconstruction.cpp:225`：完整九列体均值基矩阵；
- `src/NCFV/NCFVReconstruction.cpp:242`：加权完整SVD；
- `src/NCFV/NCFVReconstruction.cpp:292`：求逆后截取前三行；
- `src/NCFV/NCFVReconstruction.cpp:351`：当前均值差乘矩阵，再除长度尺度；
- `src/NCFV/NCFVNodeHalo.cpp:317`、`:348`：最小周期位移及中心矩平移。

## 3. 为什么正确的二次重构在粗网格上只有1.31、1.78阶？

在每个节点附近展开密度，并对各控制体积分：

\[
\boldsymbol b=A\boldsymbol a_{\le2}+\boldsymbol b_3+
\boldsymbol r_{\ge4},\qquad
\boldsymbol e_\nabla=\frac{P_g\boldsymbol b_3}{h}
+\frac{P_g\boldsymbol r_{\ge4}}{h}.
\]

第一项为三次泰勒项投影产生的主导 O(h²) 梯度误差。
本次利用解析密度三阶导数与独立积分的三阶体积矩直接计算这一项，
再从参考均值的完整梯度误差中扣除它，得到剩余项。
剩余项包括四阶及更高阶项，并包含很小的参考积分/舍入影响。
解析三阶导数另由解析Hessian的中心差分复核，抽样最大差约2.14e-10。

__CANCELLATION_TABLE__

这里不能把两列L2范数直接相加；实际关系是

\[
\|e_\nabla\|_V^2=\|e_3\|_V^2+\|e_{rest}\|_V^2
+2\langle e_3,e_{rest}\rangle_V.
\]

iv10上，主导项约0.1264，剩余项约0.06867，二者夹角余弦约−0.982，
发生很强的反向抵消，合计误差约0.06036。
因此iv10总误差比只按主导 h² 项估计的小很多，而不代表网格已充分分辨。
到iv40和iv80，剩余项相对主导项已小得多，阶数便接近二阶。

__TAYLOR_ORDER_TABLE__

**主导项本身从最粗两级起就是约二阶；低阶出现在它与高阶余项合成后的总误差。**
各网格不是完全同构的等比例网格，所以主导项阶数也不必恰等于2.000。

## 4. 模板范围与函数分辨率的对照

当前模板整圈扩展，实际约26～40点。其xy半径随网格如下变化：

__RADIUS_TABLE__

此半径是节点到模板节点最大xy距离的体积加权平均，不包含邻居控制体额外延伸。
等熵涡的指数分布使用单位长度尺度；iv10的模板中心半径约2.03，
局部二次展开覆盖了较宽的涡结构范围，高阶余项不能忽略。

在完全相同的生产重构矩阵上，把待重构函数改为严格周期的
f_k=sin(2πkx/10)cos(2πky/10)，以参考体均值输入：

__WAVE_TABLE__

k=1从最粗两级起就接近二阶；k=2变化更快，最粗两级阶数约1.55，随后也向二阶逼近。
这一对照支持“模板相对函数变化尺度过宽”的解释，而不是矩阵固定只具有一阶精度。

## 5. 只在诊断中的模板对照

保持当前高效初始化不变，独立重算两个对照矩阵：
一是保留直接邻居、按距离补足到15点并检查完整二次秩；
二是保留整圈模板，仅将距离权重指数由1改为4。
这些矩阵只是诊断局部变量，没有替换任何生产算子。

__CONTROL_TABLE__

紧凑模板显著降低梯度误差，并使粗网格阶数更早接近二阶，但不会强制所有粗网格立即达到2.000。
本报告采用几何排序打破等距离点的选择平局，避免MPI分区编号影响诊断模板。
早期诊断版本使用编号顺序，紧凑模板存在微小分区差异；其结果未用于这里的正式对照表。
即使有这些正向结果，仍需完整边界、黏性和时间推进测试，才能将新模板作为生产修复。

## 6. 周期粗网格与MPI

iv10的模板触及z方向半周期距离，iv20/40/80没有这一情况。
此前iv10的∂ρ/∂z误差约1.00e-3，而较细网格接近舍入误差。
其平方只占iv10总梯度误差平方约0.029%，因此不能单凭z向误差解释主要的x/y梯度收敛问题。
这里记录周期模板非局部性的现象，未把它未经单独反事实验证地认定为新的周期代码缺陷。

__MPI_TEXT__

## 7. 建议与边界

当前最优先的方向不是补存Hessian或改为线性拟合，而是实现方向覆盖合理、
满秩且MPI一致的紧凑模板，并在同一初场上复核梯度误差与最差节点误差。
本次结论仅针对密度一阶导数的静态检查，不等同于解释或修复整个非定常格式的全部低阶现象。
此前格点值L∞的局部低阶仍需另外处理。

## 文件与复现

- [完整原始指标](evidence.json)
- [所有对照的L1/L2/L∞及阶数](errors_and_orders.csv)
- [主导项与余项收敛图](error_decomposition.png)
- 实际采用的输出：`data/out/NCFV/gradient_audit_v2/iv*/`。
- 生产模块未修改；新增独立诊断代码位于 `cases/NCFV/diagnostics/gradient_audit.cpp`。

从仓库根目录编译诊断：

```bash
cmake --build build --target NCFV -j 6
venv/bin/python cases/NCFV/diagnostics/compile_reconstruction_probe.py \
  --source cases/NCFV/diagnostics/gradient_audit.cpp --output /tmp/ncfv_gradient_audit
```

从build目录运行，必须使用尚不存在的输出目录：

```bash
OMP_NUM_THREADS=1 mpirun --bind-to none -np 4 /tmp/ncfv_gradient_audit \
  ../cases/NCFV/NCFV_iv40.json ../data/out/NCFV/gradient_audit_new/iv40
```

从仓库根目录复核本轮已完成的数据：

```bash
MPLCONFIGDIR=/tmp/ncfv-gradient-audit-mpl \
  venv/bin/python cases/NCFV/diagnostics/summarize_gradient_audit.py
```
"""
    replacements={
        "__INITIAL_TABLE__":error_table(["rho_initial","rho_reference6"],["原初始化","参考均值"]),
        "__CHECK_TABLE__":"\n".join(checks),"__CANCELLATION_TABLE__":"\n".join(cancellation),
        "__TAYLOR_ORDER_TABLE__":error_table(["rho_cubic_taylor_term","rho_higher_taylor_remainder"],["三次项贡献","剩余项"]),
        "__RADIUS_TABLE__":"\n".join(radius),
        "__WAVE_TABLE__":error_table(["wave1_reference","wave2_reference"],["k=1","k=2"]),
        "__CONTROL_TABLE__":error_table(["rho_compact15_initial","rho_power4_initial"],["15点模板","权重指数4"]),
        "__MPI_TEXT__":f'iv10的1/2进程对照中，基准密度梯度L2差为 {mpi_check["baseline_gradient_L2_difference"]:.3e}；'
                       f'紧凑模板对照的L2差为 {mpi_check["compact_control_gradient_L2_difference"]:.3e}。'
                       f'逐点输出的误差幅值和几何统计最大差为 {max_node_difference:.3e}。',
    }
    for key,value in replacements.items(): report=report.replace(key,value)
    (args.output / "report.md").write_text(report)
    plt.rcParams.update({"text.usetex":False,"font.family":"DejaVu Sans","font.size":10})
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),constrained_layout=True)
    h=np.array([r["h"] for r in runs])
    for key,label,marker in (("rho_reference6","Total gradient error","o"),("rho_cubic_taylor_term","Cubic Taylor contribution","s"),
                             ("rho_higher_taylor_remainder","Remaining contribution","^")):
        axes[0].loglog(h,[r["errors"][key]["L2"] for r in runs],marker+"-",label=label)
    for key,label,marker in (("rho_initial","Full-ring stencil","o"),("rho_compact15_initial","15-node diagnostic stencil","s"),
                             ("rho_power4_initial","Full ring, distance power 4","^")):
        axes[1].loglog(h,[r["errors"][key]["L2"] for r in runs],marker+"-",label=label)
    for ax in axes:
        ref=runs[-1]["errors"]["rho_initial"]["L2"]
        ax.loglog(h,ref*(h/h[-1])**2,"k:",label="slope 2")
        ax.set(xlabel="h = (400 / N_periodic)^(1/3)",ylabel="Volume-weighted density-gradient L2 error")
        ax.grid(which="both",alpha=0.25);ax.legend(fontsize=8)
    axes[0].set_title("Why coarse-grid observed orders are below 2")
    axes[1].set_title("Stencil sensitivity (diagnostic controls only)")
    fig.savefig(args.output / "error_decomposition.png",dpi=180)
    plt.close(fig)
    print(error_table(["rho_initial","rho_reference6","rho_compact15_initial"],["original","reference","compact15"]))
    print(mpi_check)
    print(args.output / "report.md")


if __name__=="__main__":
    main()
