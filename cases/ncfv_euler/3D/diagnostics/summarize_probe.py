"""Summarize controlled order diagnostics; exclude the stopped iv80 Roe run."""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze_vortex_convergence import audit_run

ROOT = Path(__file__).resolve().parents[4]
DATA = ROOT / "data/out/vertexFV/order_diagnosis"
OUTPUT = ROOT / "docs/reports/vertexfv_order_diagnosis"


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    probes = {str(n): json.loads((DATA / f"iv{n}_t0.json").read_text()) for n in (10, 20, 40, 80)}
    baseline = {r["nominal_resolution"]: r for r in json.loads(
        (ROOT / "docs/reports/vertexfv_convergence/metrics.json").read_text())["runs"]}
    controls = {}
    for name, pattern in (("Roe_full_ring", "iv{n}_roe"), ("LLF_compact15", "iv{n}_compact15_evolve_run")):
        records = []
        for n in (20, 40):
            directory = DATA / pattern.format(n=n)
            metrics, _, _, cfg = audit_run(n, directory)
            if name == "LLF_compact15":
                metadata = json.loads((DATA / f"iv{n}_compact15_evolve.json").read_text())
                if metadata["evolution_time"] != 2 or metadata["compact_stencil_target"] != 15:
                    raise ValueError("Incomplete or incorrect compact evolution")
                if not (metadata["stencil_min"] == metadata["stencil_max"] == 15):
                    raise ValueError("Unexpected compact stencil size")
            if cfg["physics"]["riemannSolver"] != ("Roe" if name.startswith("Roe") else "Roe_M2"):
                raise ValueError("Riemann solver selection differs from the controlled test")
            records.append(metrics)
        rate = np.log(records[0]["density_final"]["L2"] / records[1]["density_final"]["L2"]) / np.log(
            baseline[20]["h_3d"] / baseline[40]["h_3d"])
        controls[name] = {"runs": records, "order_L2_20_to_40": float(rate)}
    weights = {str(p): json.loads((DATA / f"iv40_weight{p}_t0.json").read_text()) for p in (0, 2, 4)}
    compact = json.loads((DATA / "iv40_compact15_t0.json").read_text())
    phase = json.loads((DATA / "iv40_t2.json").read_text())
    reference = json.loads((DATA / "iv40_t0_q6.json").read_text())
    modes = {
        "mode0": "computed point / computed gradient",
        "mode1": "exact point / computed gradient",
        "mode2": "exact point / exact gradient",
        "mode3": "computed point / exact gradient",
    }
    result = {
        "probes": probes, "probe_modes": modes, "controls": controls,
        "iv40_compact15_probe": compact, "iv40_distance_weights": weights,
        "iv40_translated_analytic_probe": phase, "iv40_reference_order6": reference,
        "iv80_Roe_evolution": {
            "status": "stopped_by_agent_after_logged_step_200",
            "reason": "Bound additional diagnostic cost; not used as a t=2 result or convergence datum",
            "preserved_directory": str(DATA / "iv80_roe")},
        "scope": "No production solver/module source or baseline configuration changed.",
    }
    (OUTPUT / "evidence.json").write_text(json.dumps(result, indent=2) + "\n")
    plt.rcParams.update({"text.usetex": False, "font.family": "DejaVu Sans", "font.size": 10})
    fig, ax = plt.subplots(figsize=(7.4, 4.6), constrained_layout=True)
    h = np.array([baseline[n]["h_3d"] for n in (10, 20, 40, 80)])
    for key, label, marker in (
        ("rhs_density_L2", "Full spatial-operator error", "o"),
        ("mode0_central_error_L2", "Central-flux error", "s"),
        ("mode0_dissipation_L2", "Dissipative contribution", "^"),
        ("mode2_central_error_L2", "Central rule with exact points/gradients", "D")):
        ax.loglog(h, [probes[str(n)][key] for n in (10, 20, 40, 80)], marker + "-", label=label)
    ref = probes["80"]["mode2_central_error_L2"]
    ax.loglog(h, ref * (h/h[-1])**3, "k:", label="slope 3")
    ax.set(xlabel="h = (V / N_periodic)^(1/3)", ylabel="Volume-weighted density RHS norm",
           title="Analytic vortex at t=0: error decomposition")
    ax.grid(which="both", alpha=0.2)
    ax.legend(fontsize=8)
    fig.savefig(OUTPUT / "operator_error_decomposition.png", dpi=180)
    plt.close(fig)
    p40, p80 = probes["40"], probes["80"]
    rows = []
    rows.append(f"| 原基准：LLF＋整圈模板 | {baseline[20]['density_final']['L2']:.6e} | "
                f"{baseline[40]['density_final']['L2']:.6e} | {baseline[40]['order_3d']['L2']:.3f} |")
    for key, label in (("Roe_full_ring", "仅换标准 Roe"), ("LLF_compact15", "仅改15点诊断模板")):
        control = controls[key]
        rows.append(f"| {label} | {control['runs'][0]['density_final']['L2']:.6e} | "
                    f"{control['runs'][1]['density_final']['L2']:.6e} | {control['order_L2_20_to_40']:.3f} |")
    table = "\n".join(rows)
    report = f"""# 高效格式实测收敛阶偏低：诊断记录

## 结论

已确认当前误差由数值通量的耗散分量主导，梯度重构误差通过左右面状态差进入该分量。
当前实现还存在两处与论文路线的偏离：基准选择的 Roe_M2 实际是 LLF，而非标准 Roe；
模板选择整圈扩展，实际规模明显大于论文的14～18个（文中实例15个）。
这些差异已经被独立对照证实会显著抬高误差，但不能据此宣称已证明全部降阶原因，
也不能把换通量或缩模板等同于修复后三阶已通过。

本次没有修改生产求解器代码、原始模块或基准配置。紧凑模板仅在独立诊断进程中
改变该进程自己的重构算子，用于反事实对照，未部署为求解器功能。

## 1. Roe_M2 实际走 LLF 分支

原参考配置的无黏通量名为 EFF_FLUX_ROE_HO；当前基准 JSON 选择 Roe_M2。
DNDSR 的 Gas.hpp 中，Roe_M2 转发到 eigScheme=2，随即提前返回

\\[
\\widehat{{F}}=\\frac{{F_L+F_R}}{{2}}
-\\frac{{\\alpha}}{{2}}(U_R-U_L),\\qquad
\\alpha=\\max(|u_{{n,L}}|+a_L,|u_{{n,R}}|+a_R).
\\]

这就是局部 Lax–Friedrichs/Rusanov 型标量耗散，对所有波统一用最大波速。
标准 Roe 按各特征波速施加耗散；两者不能因名称相近而视作同一个通量。
本算例全部无量纲，因此差别不来自单位转换。

代码定位：src/Euler/Gas.hpp:1081、:1458；
高效宏面组合：src/NCFV/NCFVSpatial.cpp:509。

## 2. 数值上主导的是耗散项，而非中心全微分积分

以下为解析初场 t=0 下的密度方程瞬时空间算子误差，不是最终 t=2 解误差：

| 检查量 | iv40 | iv80 |
|---|---:|---:|
| 完整空间算子误差 | {p40['rhs_density_L2']:.6e} | {p80['rhs_density_L2']:.6e} |
| 中心通量误差 | {p40['mode0_central_error_L2']:.6e} | {p80['mode0_central_error_L2']:.6e} |
| 耗散分量范数 | {p40['mode0_dissipation_L2']:.6e} | {p80['mode0_dissipation_L2']:.6e} |
| 保留计算点值、改用精确梯度后的误差 | {p40['mode3_total_error_L2']:.6e} | {p80['mode3_total_error_L2']:.6e} |
| 精确点值和梯度下的中心积分误差 | {p40['mode2_central_error_L2']:.6e} | {p80['mode2_central_error_L2']:.6e} |

分量范数不是可直接相加的“误差百分比”，不能把耗散范数与总范数之比当作归因比例。
但在 iv40、iv80 上耗散分量明显主导，替换梯度的对照也显著降低空间算子误差。
独立重组中心与耗散项后，与原库 RHS 的差分别仅
{p40['decomposition_match_L2']:.2e} 和 {p80['decomposition_match_L2']:.2e}，
确认诊断分解复现了实际计算路径。

采用精确点值/梯度时，中心积分的名义相邻阶约为2.92、2.96、2.94；
使用实际重构时，最后两级中心项的名义阶约3.25，而耗散项约2.80。
因此没有证据把主因归结为“无 Gauss 的中心积分只能二阶”。

## 3. 重构模板比论文规定宽

论文第3.2.1节要求优先选择面相邻模板，再从下一圈只补充所需部分，目标14～18个；
第5章计算量表使用15个。当前 BuildOperator 用 ceil(1.7*9)=16 作为最小数量，
但 stencil.insert 会整圈加入，然后才检查是否达到数量要求。
对 iv40 的周期图独立枚举，实际模板为26～40个，平均约34.08个。

这会改变二次最小二乘拟合的误差常数与空间分辨能力，尤其在涡心快速变化区域。
它不自动改变多项式精确性或理论阶数，不能简单概括为“模板越多阶数越低”。

在保留直接邻居、补最近邻并检查完整二次秩的15模板诊断中，
iv40 密度梯度误差从 {p40['gradient_rho_L2']:.6e} 降到 {compact['gradient_rho_L2']:.6e}，
空间算子误差从 {p40['rhs_density_L2']:.6e} 降到 {compact['rhs_density_L2']:.6e}。
仅把距离权重指数由1改为4时，空间算子误差也降到
{weights['4']['rhs_density_L2']:.6e}。这说明模板范围和权重确实是影响源，
但这些参数对照不是已获验证的生产修复方案。

代码定位：src/NCFV/NCFVReconstruction.cpp:174，特别是整圈追加及停止条件。

## 4. 完整物理时间对照

下表全部推进到 t=2，其余物理设置、CFL、时间步上限保持一致。
阶数采用实际周期自由度对应的 h，不强制加密比恰为2。

| 独立对照 | iv20 密度 L2 | iv40 密度 L2 | iv20→iv40 阶 |
|---|---:|---:|---:|
{table}

这组结果区分了“误差降低”与“阶数恢复”：只有完成网格加密验证，才能宣称稳定三阶。
附加 iv80 标准 Roe 长时复算在日志第200步后为控制诊断开销停止，未完成 t=2，
未参与任何最终误差或收敛阶结论；其日志及初场保留在 order_diagnosis/iv80_roe。
原先完成的 iv80 LLF 基准和本次 iv80 静态算子诊断均完整保留。

## 5. 已排除或不能误判的因素

- 时间步：既有 iv20/iv40 半步长解与基准的密度场 L2 差仅约3.16e-8、3.81e-8。
- 初始化体积分：iv40 用更精确体均值代替高效初场均值后，瞬时误差从
  {p40['rhs_density_L2']:.6e} 变为 {p40['rhs_exact_means_L2']:.6e}，影响很小。
- 参考积分误差：诊断中的解析体积分从五阶升至六阶，iv40 RHS 误差变化约
  {abs(reference['rhs_density_L2']-p40['rhs_density_L2']):.2e}。
  此积分仅用于诊断真值，不进入高效求解器，也不改变其零 Gauss 点存储。
- 几何与周期：此前闭合误差约1e-16，体积和400，守恒漂移约1e-14，
  本次独立 RHS 分解又核对至舍入误差。未发现可据此认定的几何或 MPI 硬错误。
- 一阶导数只有二阶精度：这是二次重构的正常结果，论文也明确如此。
  不能据此要求恢复显式 Hessian 或宣布整体至多二阶。
- 局部截断误差与最终解误差不是同一种阶数；
  非结构有限体积可以发生超收敛，不能仅由“面积除体积少一阶”断言整体阶数。
  参见 [Diskin–Thomas 原始论文](https://fun3d.larc.nasa.gov/papers/AIAA-2012-609.pdf)。

## 6. 数学上的误差传递与下一步

高效面积平均状态可以写成

\\[
U_{{e,L}}=U_i^{{pt}}+\\frac1{{S_e}}\\sum_m G_m^T\\ell_{{im}}^e,
\\qquad J_e=U_{{e,R}}-U_{{e,L}}.
\\]

当前 LLF 耗散正比于 S_e alpha_e J_e。梯度误差通过上式进入 J_e，
再经有限体积散度参与时间推进。对二次重构，梯度误差一般为 O(h²)，
积分权重除以面积为 O(h)，故该路径可以产生 O(h³) 的状态跳跃误差。
其局部散度和最终解误差的阶数还取决于网格、相邻误差抵消和演化过程。
本次证据显示误差仍明显受耗散、重构误差常数及有限网格分辨率影响；
既不能把2.434直接解释成永久理论阶数，也不能声称修改一个参数就已恢复三阶。

建议下一轮按独立改动逐项验证：

1. 明确采用论文对应的标准 Roe 路径，不把 Roe_M2 当作标准 Roe。
2. 实现方向覆盖与秩检查兼顾的14～18模板选取，保留直接邻居，仅补足需要的邻居。
3. 对两项组合重新做同一 t=2 的网格收敛，同时保留单项对照；
   若最细两级仍未接近三阶，再增加细网格或充分分辨的光滑周期制造解测试。

本报告是诊断结论，不是修复验收报告。可复现程序为
cases/ncfv_euler/3D/diagnostics/operator_probe.cpp 和 build_probe.py；
完整数值证据见同目录 evidence.json，全部运行日志位于
data/out/vertexFV/order_diagnosis。
"""
    (OUTPUT / "diagnosis.md").write_text(report)
    print(table)
    print(OUTPUT / "diagnosis.md")


if __name__ == "__main__":
    main()
