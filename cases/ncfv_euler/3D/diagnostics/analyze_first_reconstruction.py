"""Independently audit the first reconstruction, with no DNDSR Python bindings."""
import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze_vortex_convergence import audit_mesh

ROOT = Path(__file__).resolve().parents[4]
NORMS = ("L1", "L2", "Linf")
CONSERVATIVE = ("rho", "rhou", "rhov", "rhow", "rhoE")
PRIMITIVE = ("rho", "u", "v", "w", "p")


def reference(xyz, cfg):
    """Direct primitive formulas, independently evaluated from the C++ reference."""
    gamma = cfg["physics"]["gamma"]
    center = np.asarray(cfg["initialField"]["vortexCenter"][:2])
    lengths = np.asarray(cfg["mesh"]["periodicLengths"][:2])
    delta = xyz[:, :2] - center
    ratio = delta / lengths
    delta -= lengths * np.copysign(np.floor(np.abs(ratio) + 0.5), ratio)
    x, y = delta.T
    a = cfg["initialField"]["vortexStrength"] / (2 * np.pi) * np.exp((1 - x*x - y*y) / 2)
    temperature = 1 - (gamma - 1) * a*a / (2 * gamma)
    rho = temperature ** (1 / (gamma - 1))
    pressure = temperature ** (gamma / (gamma - 1))
    velocity_x = cfg["physics"]["initialPrimitive"][1] - a*y
    velocity_y = cfg["physics"]["initialPrimitive"][2] + a*x
    zero = np.zeros(len(x))
    primitive = np.column_stack((rho, velocity_x, velocity_y, zero, pressure))
    gradient = np.zeros((len(x), 3, 5))
    d_temperature = ((gamma - 1) / gamma * a*a)[:, None] * np.column_stack((x, y, zero))
    gradient[:, :, 0] = (rho / ((gamma - 1) * temperature))[:, None] * d_temperature
    gradient[:, 0, 1] = a*x*y
    gradient[:, 1, 1] = a*(y*y - 1)
    gradient[:, 0, 2] = a*(1 - x*x)
    gradient[:, 1, 2] = -a*x*y
    gradient[:, :, 4] = (gamma * pressure / rho)[:, None] * gradient[:, :, 0]
    conservative = primitive.copy()
    velocity = primitive[:, 1:4]
    speed_squared = np.sum(velocity**2, axis=1)
    conservative[:, 1:4] = rho[:, None] * velocity
    conservative[:, 4] = pressure / (gamma - 1) + rho * speed_squared / 2
    conservative_gradient = gradient.copy()
    conservative_gradient[:, :, 1:4] = (
        gradient[:, :, 0, None] * velocity[:, None, :] + rho[:, None, None] * gradient[:, :, 1:4])
    conservative_gradient[:, :, 4] = (
        gradient[:, :, 4] / (gamma - 1) + speed_squared[:, None] * gradient[:, :, 0] / 2
        + rho[:, None] * np.sum(velocity[:, None, :] * gradient[:, :, 1:4], axis=2))
    return conservative, conservative_gradient, primitive, gradient


def primitive_from_reconstruction(state, gradient, gamma):
    rho = state[:, 0]
    velocity = state[:, 1:4] / rho[:, None]
    speed_squared = np.sum(velocity**2, axis=1)
    primitive = np.column_stack((rho, velocity, (gamma - 1) * (state[:, 4] - rho * speed_squared / 2)))
    derivatives = gradient.copy()
    derivatives[:, :, 1:4] = (
        gradient[:, :, 1:4] - gradient[:, :, 0, None] * velocity[:, None, :]) / rho[:, None, None]
    derivatives[:, :, 4] = (gamma - 1) * (
        gradient[:, :, 4] - np.sum(velocity[:, None, :] * gradient[:, :, 1:4], axis=2)
        + speed_squared[:, None] * gradient[:, :, 0] / 2)
    return primitive, derivatives


def norms(error, weights):
    return dict(zip(NORMS, map(float, (
        np.sum(weights * np.abs(error)) / np.sum(weights),
        np.sqrt(np.sum(weights * error**2) / np.sum(weights)),
        np.max(np.abs(error))))))


def audit(n, directory):
    metadata = json.loads((directory / "metrics.json").read_text())
    for key in ("time", "iteration", "time_steps", "rhs_evaluations", "gauss_points_stored"):
        if metadata[key] != 0:
            raise ValueError(f"iv{n}: {key} must be zero")
    for key in ("compute_coefficients_calls", "recover_point_values_calls"):
        if metadata[key] != 1:
            raise ValueError(f"iv{n}: not the first single reconstruction")
    files = sorted(directory.glob("nodes.rank*.csv"))
    if len(files) != metadata["mpi_ranks"]:
        raise ValueError("Missing rank output")
    data = np.concatenate([np.loadtxt(file, delimiter=",", skiprows=1, ndmin=2) for file in files])
    data = data[np.argsort(data[:, 0])]
    manifest, coordinates, _ = audit_mesh(n)
    if not np.array_equal(data[:, 0], np.arange(len(coordinates))):
        raise ValueError("Original node IDs are not uniquely and completely owned")
    if not np.allclose(data[:, 1:4], coordinates, atol=1e-13, rtol=0):
        raise ValueError("Node coordinates do not match the checksummed input mesh")
    if not np.isfinite(data).all() or data[:, 4].min() <= 0:
        raise ValueError("Invalid reconstruction data or dual volume")
    if metadata["nodes"] != len(data) or metadata["cells"] != manifest["cells"]:
        raise ValueError("Wrong mesh size")
    volume = float(data[:, 4].sum())
    if abs(volume - 400) > 1e-9:
        raise ValueError("Dual volume does not sum to domain volume")
    wrapped = coordinates.copy()
    for axis, length in enumerate(metadata["configuration"]["mesh"]["periodicLengths"]):
        wrapped[np.isclose(wrapped[:, axis], length, atol=1e-8, rtol=0), axis] = 0
    periodic_count = len(np.unique(np.round(wrapped, 8), axis=0))
    if periodic_count != metadata["periodic_unknowns"]:
        raise ValueError("Periodic node count disagrees with volume-fraction reduction")
    cfg = metadata["configuration"]
    computed = data[:, 10:15]
    derivative = data[:, 15:].reshape(-1, 5, 3).transpose(0, 2, 1)
    computed_primitive, primitive_derivative = primitive_from_reconstruction(computed, derivative, cfg["physics"]["gamma"])
    exact, exact_derivative, exact_primitive, exact_primitive_derivative = reference(coordinates, cfg)
    independent = {}
    worst_nodes = {}
    for family, names, q, dq, truth, truth_dq in (
        ("conservative", CONSERVATIVE, computed, derivative, exact, exact_derivative),
        ("primitive", PRIMITIVE, computed_primitive, primitive_derivative, exact_primitive, exact_primitive_derivative),
    ):
        for i, name in enumerate(names):
            quantities = {"value": q[:, i] - truth[:, i],
                          "gradient": np.linalg.norm(dq[:, :, i] - truth_dq[:, :, i], axis=1)}
            quantities.update({f"d{axis}": dq[:, d, i] - truth_dq[:, d, i] for d, axis in enumerate("xyz")})
            for quantity, error in quantities.items():
                key = f"{family}.{name}.{quantity}"
                independent[key] = norms(error, data[:, 4])
                worst = int(np.argmax(np.abs(error)))
                worst_nodes[key] = {"original_node": int(data[worst, 0]),
                                    "coordinate": coordinates[worst].tolist(),
                                    "error": float(error[worst])}
                for norm in NORMS:
                    if not np.isclose(independent[key][norm], metadata["errors"][key][norm], rtol=2e-10, atol=2e-13):
                        raise ValueError(f"Independent reference/reduction disagreement: iv{n} {key} {norm}")
    metadata.update(mesh=f"iv{n}", nominal_resolution=n, h_nominal=10/n,
                    independently_audited=True, source_directory=str(directory),
                    independent_errors=independent, worst_nodes=worst_nodes)
    return metadata, data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/out/NCFV/first_reconstruction")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/reports/ncfv_first_reconstruction")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    runs = []
    baseline20 = None
    for n in (10, 20, 40, 80):
        metadata, data = audit(n, args.data / f"iv{n}")
        if n == 20:
            baseline20 = data
        runs.append(metadata)
        print(f"iv{n}: audited {len(data)} nodes, exactly one reconstruction, t=0", flush=True)
    for previous, current in zip(runs, runs[1:]):
        if any(previous["configuration"][key] != current["configuration"][key]
               for key in ("algorithm", "reconstruction", "physics", "initialField")):
            raise ValueError("Meshes use different reconstruction/physics settings")
    rows = []
    for i, run in enumerate(runs):
        run["orders"] = {}
        for key, error in run["errors"].items():
            rates = {}
            for norm in NORMS:
                last_error = runs[i-1]["errors"][key][norm] if i else 0
                rates[norm] = (float(np.log(last_error/error[norm]) / np.log(runs[i-1]["h_3d"]/run["h_3d"]))
                               if i and min(last_error, error[norm]) > 1e-12 else None)
            run["orders"][key] = rates
            rows.append({"mesh": run["mesh"], "h_3d": run["h_3d"], "quantity": key,
                         **error, **{f"order_{norm}": rates[norm] for norm in NORMS}})
    mpi_check = None
    if (args.data / "iv20_np1/metrics.json").exists():
        _, serial20 = audit(20, args.data / "iv20_np1")
        maximum_difference = float(np.max(np.abs(serial20[:, 5:] - baseline20[:, 5:])))
        if maximum_difference > 2e-12:
            raise ValueError(f"MPI field disagreement: {maximum_difference}")
        mpi_check = {"mesh": "iv20", "ranks": [1, 2], "maximum_field_difference": maximum_difference}
    with (args.output / "errors_and_orders.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "metrics.json").write_text(json.dumps({"runs": runs, "mpi_check": mpi_check}, indent=2) + "\n")

    plt.rcParams.update({"text.usetex": False, "font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    h = np.array([run["h_3d"] for run in runs])
    for ax, quantity, order, title in zip(axes, ("value", "gradient"), (3, 2),
                                        ("Recovered node values", "First-derivative vectors")):
        for name, marker in zip(("rho", "p", "u", "v"), ("o", "s", "^", "D")):
            errors = np.array([run["errors"][f"primitive.{name}.{quantity}"]["L2"] for run in runs])
            ax.loglog(h, errors, marker + "-", label=name)
        reference_error = runs[-1]["errors"][f"primitive.rho.{quantity}"]["L2"]
        ax.loglog(h, reference_error * (h/h[-1])**order, "k:", label=f"slope {order}")
        ax.set(xlabel="h = (400 / N_periodic)^(1/3)", ylabel="Volume-weighted L2 error", title=title)
        ax.grid(which="both", alpha=0.25)
        ax.legend(fontsize=9)
    fig.suptitle("NCFV / efficient mode / first reconstruction at t=0 (no time steps)")
    fig.savefig(args.output / "convergence.png", dpi=180)
    plt.close(fig)

    def table(keys, labels, norm="L2"):
        header = "| 网格 | " + " | ".join(f"{label} {norm} | 阶数" for label in labels) + " |\n"
        header += "|---|" + "---:|---:|" * len(keys) + "\n"
        for run in runs:
            row = [run["mesh"]]
            for key in keys:
                rate = run["orders"][key][norm]
                row += [f'{run["errors"][key][norm]:.6e}', "—" if rate is None else f"{rate:.3f}"]
            header += "| " + " | ".join(row) + " |\n"
        return header

    mesh_table = "| 网格 | 原始节点 | 周期独立节点 | h | 模板数最小/平均/最大 |\n|---|---:|---:|---:|---:|\n"
    for run in runs:
        mesh_table += (f'| {run["mesh"]} | {run["nodes"]} | {run["periodic_unknowns"]} | {run["h_3d"]:.6f} | '
                       f'{run["stencil_min"]:.0f}/{run["stencil_mean_owned_nodes"]:.2f}/{run["stencil_max"]:.0f} |\n')
    last = runs[-1]
    mpi_text = (f'iv20 使用1和2进程复核，控制体均值、格点值及全部一阶导数的最大差为 '
                f'{mpi_check["maximum_field_difference"]:.3e}。' if mpi_check else '本轮未附加不同进程数对照。')
    report = f"""# NCFV 高效模式：第一次重构精度检查

NCFV 全称：Node Center Finite Volume Method。

## 检查范围

沿用 iv10、iv20、iv40、iv80 网格上的同一等熵涡，β=5、γ=1.4、
中心(5,5)、背景速度(1,1,0)、周期盒[10,10,4]。所有结果严格位于 t=0。
直接读取 Solver.Initialize 生成并完成周期同步的高效对偶均值，只调用一次
ComputeCoefficients 和一次 RecoverPointValues，不调用 Solver.Run 或 EvaluateRHS。
时间步数、残差评价次数和 Gauss 点存储数均为0。未改变生产模块、模板或权重设置。

这是“当前高效初始化→第一次重构→格点值恢复”整体的误差检查，
不是用精确体均值单独检验最小二乘算子的试验，也不是完整时间推进收敛测试。
压力与速度由恢复的守恒点值转换，其导数通过守恒变量导数的链式法则计算，
不是另对压力/速度再做一次最小二乘重构。

## 误差定义与网格尺度

对每个原始节点 owner 仅计一次，以其对偶控制体份额体积 V_i 加权：

L1 = Σ V_i |e_i| / Σ V_i；L2 = sqrt(Σ V_i |e_i|² / Σ V_i)；L∞ = max |e_i|。

梯度向量误差使用 |e_i| = ||∇q_i^h − ∇q_exact(x_i)||₂；dx、dy、dz 列则单独比较相应偏导数。
周期边界节点使用各自体积份额，合计体积为400，不重复计完整控制体体积。
阶数采用 p = log(E_coarse/E_fine)/log(h_coarse/h_fine)，h=(400/N_periodic)^(1/3)。
不强制把实际网格加密比取成2。误差已低于1e-12的量不计算阶数。

{mesh_table}
## 密度与压力：恢复格点值

{table(['primitive.rho.value', 'primitive.p.value'], ['ρ', 'p'])}
## 密度与压力：格点值最大误差（不可忽略）

{table(['primitive.rho.value', 'primitive.p.value'], ['ρ', 'p'], 'Linf')}
iv40→iv80 的格点值 L∞ 阶仅为：密度 {last['orders']['primitive.rho.value']['Linf']:.3f}，
压力 {last['orders']['primitive.p.value']['Linf']:.3f}。
因此只能说本组格点值在 L2 意义下收敛较快，不能宣称所有节点都已验证三阶。
iv80 密度最大点误差位于原始节点 {last['worst_nodes']['primitive.rho.value']['original_node']}
（坐标 {last['worst_nodes']['primitive.rho.value']['coordinate']}）；
压力最大点误差位于原始节点 {last['worst_nodes']['primitive.p.value']['original_node']}
（坐标 {last['worst_nodes']['primitive.p.value']['coordinate']}）。节点编号为0起始。
这里只定位最大误差，不将其未经验证地归因为某种网格或边界缺陷。

## 密度与压力：一阶导数向量

{table(['primitive.rho.gradient', 'primitive.p.gradient'], ['∇ρ', '∇p'])}
## 密度一阶偏导数分量

{table(['primitive.rho.dx', 'primitive.rho.dy', 'primitive.rho.dz'], ['∂ρ/∂x', '∂ρ/∂y', '∂ρ/∂z'])}
## 速度：恢复格点值及梯度

{table(['primitive.u.value', 'primitive.v.value'], ['u', 'v'])}
{table(['primitive.u.gradient', 'primitive.v.gradient'], ['∇u', '∇v'])}
## 解释及验证

最细两级的密度格点值 L2 阶为 {last['orders']['primitive.rho.value']['L2']:.3f}，
密度一阶导数向量 L2 阶为 {last['orders']['primitive.rho.gradient']['L2']:.3f}。
完整二次重构的一阶导数通常期望二阶；导数不是三阶本身不构成程序降阶证据。
对当前高效初始化与点值恢复，令 w_im 为归一化全微分权重，则

ū_i^h = u_exact(x_i) + Σ_m w_im · ∇u_exact(x_m)，
u_i^h = ū_i^h − Σ_m w_im · ∇u_m^h，
因此点值误差受 Σ_m w_im · (∇u_exact − ∇u_m^h) 控制，周期节点再作体积加权合并。
这解释了格点值与一阶导数可以呈现不同的收敛阶；较高点值实测阶不能直接当作整个求解器阶数。

本解析涡与z无关，且w、ρw及其导数严格为0；相应误差仅用于检查横向污染或舍入误差，
不应利用接近零的误差比推算阶数。
iv10 的 ∂ρ/∂z 确有约1.00e-3的 L2 误差；iv20、iv40、iv80 的同一误差约为
3.03e-16、8.83e-16、1.64e-15，已接近舍入误差水平。这一粗网格现象应与
x/y方向的正常加密趋势分别看待。

独立Python公式已复核全部50类量的 L1/L2/L∞，与C++ MPI归约结果一致。
也检查了原始节点唯一所有权、坐标/连接数组校验值、正体积及周期自由度计数。
{mpi_text}

## 文件与复现

- [完整误差与阶数CSV](errors_and_orders.csv)：守恒变量和原始变量的点值、三方向偏导及梯度向量范数。
- [完整指标与配置](metrics.json)。
- [误差收敛图](convergence.png)。
- 逐节点均值、恢复点值和守恒量梯度：`data/out/NCFV/first_reconstruction/iv*/nodes.rank*.csv`。

从仓库根目录构建与编译诊断：

```bash
cmake --build build --target NCFV -j 6
venv/bin/python cases/ncfv_euler/3D/diagnostics/compile_reconstruction_probe.py
```

从 build 目录运行（输出目录必须尚不存在；其他网格相同方式替换编号）：

```bash
OMP_NUM_THREADS=1 mpirun --bind-to none -np 4 /tmp/ncfv_first_reconstruction_probe \\
  ../cases/ncfv_euler/3D/NCFV_iv40.json ../data/out/NCFV/first_reconstruction_new/iv40
```

从仓库根目录汇总四套完整结果：

```bash
MPLCONFIGDIR=/tmp/ncfv-first-reconstruction-mpl \\
  venv/bin/python cases/ncfv_euler/3D/diagnostics/analyze_first_reconstruction.py \\
  --data data/out/NCFV/first_reconstruction_new \\
  --output docs/reports/ncfv_first_reconstruction_new
```
"""
    (args.output / "report.md").write_text(report)
    print(table(["primitive.rho.value", "primitive.rho.gradient", "primitive.p.value", "primitive.p.gradient"],
                ["ρ", "∇ρ", "p", "∇p"]))
    print(f"Report: {args.output / 'report.md'}")


if __name__ == "__main__":
    main()
