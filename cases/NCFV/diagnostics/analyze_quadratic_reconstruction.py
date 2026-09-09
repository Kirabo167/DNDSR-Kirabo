"""Audit the eight quadratic patch runs without importing DNDSR Python bindings."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
NAMES = ["rho", "rhou", "rhov", "rhow", "rhoE"]
COEFFICIENTS = np.array([
    [2000, 100, -60, 40, 30, 20, -15, 25, 12, 20],
    [600, -30, 50, 20, 12, -10, 8, -6, 9, 10],
    [400, 40, 20, -30, -8, 12, 6, 10, -7, 8],
    [200, 20, -10, 40, 6, 8, -10, 12, 5, -4],
    [5000, 150, -100, 80, 60, -25, 20, 40, 15, 30],
], dtype=np.longdouble) / 1000
CENTER = np.array([5, 5, 2], dtype=np.longdouble)
SCALE = np.array([5, 5, 2], dtype=np.longdouble)
# Fixed pre-run absolute acceptance bounds, not fitted to measured results.
VALUE_TOLERANCE = 1e-10
GRADIENT_TOLERANCE = 1e-8


def exact_field(xyz):
    """Matrix/Hessian form, independent of the probe's component formulas."""
    coordinate = (xyz - CENTER) / SCALE
    hessian = np.zeros((5, 3, 3), dtype=np.longdouble)
    hessian[:, 0, 0] = 2 * COEFFICIENTS[:, 4]
    hessian[:, 1, 1] = 2 * COEFFICIENTS[:, 7]
    hessian[:, 2, 2] = 2 * COEFFICIENTS[:, 9]
    for a, b, k in [(0, 1, 5), (0, 2, 6), (1, 2, 8)]:
        hessian[:, a, b] = hessian[:, b, a] = COEFFICIENTS[:, k]
    values = (COEFFICIENTS[:, 0] + coordinate @ COEFFICIENTS[:, 1:4].T
              + np.longdouble("0.5") * np.einsum("ni,vij,nj->nv", coordinate, hessian, coordinate))
    gradients = (COEFFICIENTS[:, 1:4][None, :, :]
                 + np.einsum("vij,nj->nvi", hessian, coordinate)) / SCALE
    return values, gradients


def audit_nodes(directory, metadata):
    accumulators = {}
    all_ids = []
    volume = np.longdouble(0)
    minimum = np.full(3, np.inf, dtype=np.longdouble)
    maximum = -minimum.copy()
    files = sorted(directory.glob("nodes.rank*.csv"))
    assert len(files) == metadata["mpi_ranks"]
    for path in files:
        data = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.longdouble, ndmin=2)
        assert data.shape[1] == 36 and np.isfinite(data).all()
        all_ids.append(data[:, 0].astype(np.int64))
        xyz, weights = data[:, 1:4], data[:, 4]
        assert np.all(weights > 0)
        volume += weights.sum()
        minimum = np.minimum(minimum, xyz.min(axis=0))
        maximum = np.maximum(maximum, xyz.max(axis=0))
        exact, gradient = exact_field(xyz)
        value_error = data[:, 16:21] - exact
        gradient_error = data[:, 21:36].reshape(-1, 5, 3) - gradient
        field_errors = {
            "mean": data[:, 6:11] - data[:, 11:16],
            "value": value_error,
            "gradient": np.sqrt(np.sum(gradient_error**2, axis=2)),
            **{f"d{axis}": gradient_error[:, :, d] for d, axis in enumerate("xyz")},
        }
        for quantity, differences in field_errors.items():
            for v, name in enumerate(NAMES):
                error = np.abs(differences[:, v])
                key = f"all.conservative.{name}.{quantity}"
                a = accumulators.setdefault(key, np.zeros(3, dtype=np.longdouble))
                a[0] += np.sum(weights * error)
                a[1] += np.sum(weights * error**2)
                a[2] = max(a[2], error.max())
    ids = np.concatenate(all_ids)
    assert len(ids) == len(np.unique(ids)) == metadata["nodes"]
    assert np.allclose(minimum, [0, 0, 0], rtol=0, atol=1e-10)
    assert np.allclose(maximum, [10, 10, 4], rtol=0, atol=1e-10)
    assert abs(volume - 400) < 1e-9
    discrepancy = 0.0
    for key, sums in accumulators.items():
        computed = [sums[0] / volume, np.sqrt(sums[1] / volume), sums[2]]
        for norm, value in zip(["L1", "L2", "Linf"], computed):
            expected = metadata["errors"][key][norm]
            delta = float(abs(value - expected))
            discrepancy = max(discrepancy, delta)
            assert delta < max(1e-17, abs(expected) * 5e-4), (directory, key, norm, value, expected)
    return {"unique_owned_nodes": len(ids), "volume": float(volume),
            "independent_norm_max_discrepancy": discrepancy}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/out/NCFV/quadratic_reconstruction_20260909")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/reports/ncfv_quadratic_reconstruction_20260909")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    runs, rows, audits = [], [], {}
    for mode in ["efficient", "traditional"]:
        for n in [10, 20, 40, 80]:
            directory = args.input / mode / f"iv{n}"
            metadata = json.loads((directory / "metrics.json").read_text())
            cfg = metadata["configuration"]
            assert metadata["time"] == metadata["iteration"] == metadata["rhs_evaluations"] == metadata["time_steps"] == 0
            assert metadata["compute_coefficients_calls"] == metadata["recover_point_values_calls"] == 1
            assert not metadata["periodic"] and cfg["mesh"]["periodicLengths"] == [0, 0, 0]
            assert not cfg["reconstruction"]["enableLimiter"]
            assert metadata["minimum_reconstruction_rank"] == 9
            assert metadata["inverse_rows"] == (3 if mode == "efficient" else 9)
            assert (metadata["stored_quadrature_points"] == 0) == (mode == "efficient")
            assert metadata["rho_min"] > 0 and metadata["pressure_min"] > 0
            assert metadata["volume_relative_error_max"] < 1e-11
            assert metadata["interior_nodes"] + metadata["boundary_nodes"] == metadata["nodes"]
            assert np.array_equal(np.asarray(metadata["field_coefficients_numerator"], dtype=np.longdouble) / 1000,
                                  COEFFICIENTS)
            for name in NAMES:
                for quantity, tolerance in [("mean", VALUE_TOLERANCE), ("value", VALUE_TOLERANCE),
                                             ("gradient", GRADIENT_TOLERANCE)]:
                    assert metadata["errors"][f"all.conservative.{name}.{quantity}"]["Linf"] < tolerance
            audits[f"{mode}/iv{n}"] = audit_nodes(directory, metadata)
            for key, norms in metadata["errors"].items():
                region, variables, component, quantity = key.split(".")
                rows.append({"mode": mode, "mesh": n, "nodes": metadata["nodes"],
                             "region": region, "variables": variables, "component": component,
                             "quantity": quantity, **norms})
            runs.append((mode, n, metadata))
    with (args.output / "errors.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    def metric(metadata, component, quantity, norm="L2", region="all", variables="conservative"):
        return metadata["errors"][f"{region}.{variables}.{component}.{quantity}"][norm]

    report = [
        "# NCFV 二次多项式重构再现试验（2026-09-09）", "",
        "## 结论", "",
        "四套网格、两种模式均已完成初始化及一次重构；未计算残差、通量或时间推进。"
        "各守恒分量的函数值和一阶导数均通过预设绝对容差（函数/平均值 1e-10，梯度向量 1e-8）。"
        "实际误差见下表；这些容差是回归检查上限，不代表实测误差。", "",
        "本试验检验的是二次多项式再现性，不是一般光滑解的收敛阶。"
        "舍入误差附近不拟合网格收敛阶，也不能据此证明完整求解器达到三阶。", "",
        "## 解析流场", "",
        "令 X=(x-5)/5，Y=(y-5)/5，Z=(z-2)/2，定义守恒量 "
        "U=(ρ,ρu,ρv,ρw,ρE)ᵀ，每个分量为：", "",
        "```text", "U_a = c0 + cx X + cy Y + cz Z + cxx X² + cxy XY + cxz XZ + cyy Y² + cyz YZ + czz Z²",
        "∂x U_a = (cx + 2 cxx X + cxy Y + cxz Z)/5",
        "∂y U_a = (cy + cxy X + 2 cyy Y + cyz Z)/5",
        "∂z U_a = (cz + cxz X + cyz Y + 2 czz Z)/2", "```", "",
        "| 分量 | c0 | cx | cy | cz | cxx | cxy | cxz | cyy | cyz | czz |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, coefficients in zip(NAMES, COEFFICIENTS):
        report.append("| " + name + " | " + " | ".join(f"{float(c):g}" for c in coefficients) + " |")
    lower_rho = COEFFICIENTS[0, 0] - np.abs(COEFFICIENTS[0, 1:]).sum()
    lower_energy = COEFFICIENTS[4, 0] - np.abs(COEFFICIENTS[4, 1:]).sum()
    upper_momentum = np.abs(COEFFICIENTS[1:4, 0]) + np.abs(COEFFICIENTS[1:4, 1:]).sum(axis=1)
    lower_pressure = np.longdouble("0.4") * (lower_energy - (upper_momentum**2).sum() / (2 * lower_rho))
    report += ["", "原始变量由 u=(ρu)/ρ、v=(ρv)/ρ、w=(ρw)/ρ、"
               "p=(γ−1)[ρE−((ρu)²+(ρv)²+(ρw)²)/(2ρ)] 给出，γ=1.4。"
               "这些原始变量一般不是二次多项式；报告同时给出它们由守恒量变换得到的函数和梯度误差。",
               "", f"在整个盒域 |X|,|Y|,|Z|≤1 上，保守下界 ρ≥{float(lower_rho):.6f}、"
               f"p≥{float(lower_pressure):.6f}，因此不是仅在采样点满足物理性。", "",
               "## 控制体平均值与重构过程", "",
               "1. 沿用四个 CGNS 网格、网格读取、对偶控制体生成以及论文半跨度归一化。"
               "二次多项式不能周期拼接，因此仅在诊断内存配置中关闭平移周期合并；"
               "保留边界处截断的完整对偶控制体，并使用单侧内部格点模板。"
               "所有边界分区设为非强制 FarField，但未调用边界通量或对制造场施加强制边界状态。",
               "2. 高效模式初始平均值采用生产初始化相同的全微分权重："
               "Ūᵢ=U(xᵢ)+Σⱼ Wᵢⱼ·∇U(xⱼ)。此处仅初始化使用解析梯度；"
               "真正送入重构函数的只有平均值，重构后输出的梯度不是解析梯度回填。",
               "3. 传统模式采用现有 quadratureOrder=4 的小四面体体数值积分生成平均值。"
               "这次未调用面通量，故本试验不检验面通量积分。",
               "4. 独立平均值参考采用 long double 小四面体重心坐标恒等式。"
               "设四顶点归一化坐标 sₐ，S=Σₐsₐ，则 ⟨s⟩=S/4，"
               "⟨ssᵀ⟩=(SSᵀ+Σₐsₐsₐᵀ)/20；代入二次多项式积分后按体积累加。"
               "体积也重新由四顶点行列式求出，不调用生产 RawMoments/积分权重。"
               "诊断保留了小四面体顶点用于此核对；高效模式没有存储 Gauss 点。",
               "5. 每次调用一次原有 ComputeCoefficients 和一次 RecoverPointValues。"
               "两模式均解九项二次零均值基的最小二乘问题；高效模式只存伪逆前三行，"
               "传统模式存全部九行。传统节点梯度由完整多项式在节点处求导。"
               "MPI 同步均值以及恢复点值所需的梯度/系数。未修改任何生产求解模块。", "",
               "对于精确二次场，模板满足 ΔŪ=Aa，因此满列秩时 A⁺ΔŪ=a。"
               "取伪逆前三行同样能得到正确的一阶系数。传统点值为 Ūᵢ−aᵢ·⟨bᵢ⟩；"
               "高效点值为 Ūᵢ−ΣⱼWᵢⱼ·∇Uⱼ。二次场梯度是线性的，"
               "算术均值构造点的梯度插值对它是精确的，所以两种点值恢复在精确算术下都应精确。", "",
               "## 密度函数值和梯度误差", "",
               "L1=ΣᵢVᵢ|eᵢ|/ΣᵢVᵢ；L2=(ΣᵢVᵢ|eᵢ|²/ΣᵢVᵢ)¹ᐟ²；L∞=maxᵢ|eᵢ|。"
               "梯度项使用三方向误差向量的欧氏模。权重为实际对偶体积，覆盖所有格点。", "",
               "| 模式 | 网格 | 格点数 | 初始均值 L2 | 点值 L2 | 点值 L∞ | 梯度 L2 | 梯度 L∞ |",
               "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for mode, n, metadata in runs:
        values = [metric(metadata, "rho", "mean"), metric(metadata, "rho", "value"),
                  metric(metadata, "rho", "value", "Linf"), metric(metadata, "rho", "gradient"),
                  metric(metadata, "rho", "gradient", "Linf")]
        report.append(f"| {'高效' if mode == 'efficient' else '传统'} | {n} | {metadata['nodes']} | "
                      + " | ".join(f"{v:.6e}" for v in values) + " |")
    report += ["", "## 全部守恒分量及边界核对", "",
               "下面各误差为五个守恒分量中的最大值，梯度仍为三方向误差向量模。", "",
               "| 模式 | 网格 | 所有点值 L∞ | 所有梯度 L∞ | 内部梯度 L∞ | 边界梯度 L∞ | 最大条件数 |",
               "|---|---:|---:|---:|---:|---:|---:|"]
    for mode, n, metadata in runs:
        quantities = [("value", "all"), ("gradient", "all"), ("gradient", "interior"), ("gradient", "boundary")]
        values = [max(metric(metadata, name, quantity, "Linf", region) for name in NAMES)
                  for quantity, region in quantities] + [metadata["condition_max"]]
        report.append(f"| {'高效' if mode == 'efficient' else '传统'} | {n} | "
                      + " | ".join(f"{v:.6e}" for v in values) + " |")
    report += ["", "梯度误差随加密略增，符合初始平均值和矩阵运算的舍入误差经求导约按 1/h 放大的特征。"
               "传统模式的初始平均值已经具有约 1e-14 的舍入误差，因此其梯度噪声高于高效模式。"
               "这是与数据相符的解释，并非对每个浮点误差来源的逐项归因；这里没有观察到二次多项式再现失效。", "",
               "## iv80 各守恒分量的分方向 L2 误差", "",
               "| 模式 | 分量 | 点值 | ∂x | ∂y | ∂z |",
               "|---|---|---:|---:|---:|---:|"]
    for mode, n, metadata in runs:
        if n != 80:
            continue
        for name in NAMES:
            values = [metric(metadata, name, quantity) for quantity in ["value", "dx", "dy", "dz"]]
            report.append(f"| {'高效' if mode == 'efficient' else '传统'} | {name} | "
                          + " | ".join(f"{v:.6e}" for v in values) + " |")
    report += ["", "## 运行与完整性记录", "",
               "| 模式 | 网格 | MPI 进程 | 边界点数 | 最小矩阵秩 | 存储求积点数 | 总体积 | 耗时/s |",
               "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for mode, n, m in runs:
        report.append(f"| {mode} | {n} | {m['mpi_ranks']} | {m['boundary_nodes']} | "
                      f"{m['minimum_reconstruction_rank']} | {m['stored_quadrature_points']} | "
                      f"{m['volume']:.12g} | {m['wall_seconds']:.3f} |")
    report += ["", "独立 NumPy 长精度后处理逐点重算了五个守恒分量的解析值和梯度，"
               "核对所有节点 ID 唯一、范围正确、总体积为 400，以及 CSV 与 MPI 汇总一致。"
               "完整 L1/L2/L∞、三个方向导数、原始变量及内部/边界分组数据见 errors.csv。", "",
               "本试验不能排除周期商网格、一般非多项式场截断误差、面通量积分、数值耗散、"
               "时间推进等环节的问题。尤其这是非周期多项式试验，不单独验证周期边界处理。", "",
               "## 复现", "", "从项目根目录编译最新 NCFV，再运行独立诊断。每次输出目录必须不存在。", "",
               "```bash", "CCACHE_DISABLE=1 cmake --build build -t NCFV -j6",
               "venv/bin/python cases/NCFV/diagnostics/compile_reconstruction_probe.py \\",
               "  --source cases/NCFV/diagnostics/quadratic_reconstruction_probe.cpp \\",
               "  --output /tmp/ncfv_quadratic_probe", "cd build",
               "OMP_NUM_THREADS=1 mpirun --bind-to none -np 4 /tmp/ncfv_quadratic_probe \\",
               "  ../cases/NCFV/t2_cfl05_thesis_20260907/iv20.json /tmp/ncfv_quadratic_fresh_output", "```", "",
               "传统模式改用 cases/NCFV/traditional_t2_cfl05_20260908/iv20.json；"
               "其余网格替换 iv20 为 iv10/iv40/iv80。加载后实际的诊断配置完整存于每次 metrics.json 中。", ""]
    (args.output / "report.md").write_text("\n".join(report))
    sources = ["src/NCFV/NCFVReconstruction.cpp", "src/NCFV/NCFVDualGeometry.cpp", "src/NCFV/NCFVSolver.cpp",
               "build/src/NCFV/libncfv.a", "cases/NCFV/diagnostics/quadratic_reconstruction_probe.cpp"]
    manifest = {"audits": audits, "value_tolerance": VALUE_TOLERANCE, "gradient_tolerance": GRADIENT_TOLERANCE,
                "sources_sha256": {s: hashlib.sha256((ROOT / s).read_bytes()).hexdigest() for s in sources}}
    (args.output / "audit.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    print(args.output / "report.md")


if __name__ == "__main__":
    main()
