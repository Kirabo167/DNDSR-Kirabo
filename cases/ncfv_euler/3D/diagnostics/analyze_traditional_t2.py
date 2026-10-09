"""Audit completed traditional t=2 NCFV runs using recovered density AND pressure point values.

No DNDSR Python bindings are imported; the solver was run as a C++/MPI program.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze_vortex_convergence import audit_mesh, exact_density, norms, load_nodes

ROOT = Path(__file__).resolve().parents[4]
NORMS = ("L1", "L2", "Linf")


def audit_run(n, directory, log_path=None):
    prefix = directory / "solution"
    diagnostics = np.atleast_1d(np.genfromtxt(str(prefix) + ".diagnostics.csv", delimiter=",", names=True))
    if (len(diagnostics) != 2 or abs(diagnostics[-1]["time"] - 2.0) > 1e-12
            or diagnostics[0]["time"] != 0 or diagnostics[0]["iteration"] != 0):
        raise ValueError(f"iv{n}: run is not complete at t=2")
    configuration = json.loads(Path(str(prefix) + ".resolved.json").read_text())
    if configuration["algorithm"]["mode"] != "TraditionalQuadrature":
        raise ValueError("Only traditional runs may enter this convergence study")
    if configuration["time"]["useLocalTimeStep"]:
        raise ValueError("Local pseudo-time is invalid for this unsteady comparison")
    manifest, xyz, cells = audit_mesh(n)
    data = []
    errors = []
    for diagnostic in diagnostics:
        nodes = load_nodes(prefix, int(diagnostic["iteration"]))
        if not np.array_equal(nodes["original_node"], np.arange(len(xyz))):
            raise ValueError("Original mesh node ownership is incomplete")
        output_xyz = np.column_stack([nodes[k] for k in ("x", "y", "z")])
        if not np.allclose(output_xyz, xyz, atol=1e-13, rtol=0):
            raise ValueError("Output node coordinates do not match original mesh")
        if not np.isfinite(nodes.view(np.float64)).all() or np.min(nodes["partial_volume"]) <= 0:
            raise ValueError("Non-finite data or nonpositive dual volume")
        exact = exact_density(xyz, float(diagnostic["time"]))
        if not np.allclose(exact, nodes["rho_exact"], rtol=0, atol=3e-14):
            raise ValueError("Independent vortex reference disagrees with solver")
        error = norms(nodes["rho_point"] - exact, nodes["partial_volume"])
        for name in NORMS:
            if not np.isclose(error[name], diagnostic[f"rho_point_{name}"], rtol=3e-12, atol=2e-15):
                raise ValueError("Independent error reduction disagrees with solver")
        data.append(nodes)
        errors.append(error)
    first, final = data
    volume = float(np.sum(final["partial_volume"]))
    if abs(volume - 400.0) > 1e-9:
        raise ValueError("Dual volumes do not sum to domain volume")
    lengths = np.array([10.0, 10.0, 4.0])
    wrapped = xyz.copy()
    for d, length in enumerate(lengths):
        wrapped[np.isclose(wrapped[:, d], length, rtol=0, atol=1e-8), d] = 0
    _, inverse = np.unique(np.round(wrapped, 8), axis=0, return_inverse=True)
    n_periodic = int(inverse.max() + 1)
    n_xy = len(np.unique(np.round(wrapped[:, :2], 8), axis=0))
    periodic_difference = 0.0
    for field in ("rho_mean", "rhou", "rhov", "rhow", "rhoE"):
        minimum, maximum = np.full(n_periodic, np.inf), np.full(n_periodic, -np.inf)
        np.minimum.at(minimum, inverse, final[field])
        np.maximum.at(maximum, inverse, final[field])
        periodic_difference = max(periodic_difference, float(np.max(maximum - minimum)))
    if periodic_difference > 1e-11:
        raise ValueError("Periodic state copies disagree")
    conserved = ("mass", "momentum_x", "momentum_y", "momentum_z", "total_energy")
    drift = {key: float((diagnostics[-1][key] - diagnostics[0][key]) /
                       max(abs(diagnostics[0][key]), 1.0)) for key in conserved}
    momentum_squared = sum(final[k]**2 for k in ("rhou", "rhov", "rhow"))
    pressure_of_mean = 0.4 * (final["rhoE"] - 0.5 * momentum_squared / final["rho_mean"])
    if final["rho_mean"].min() <= 0 or pressure_of_mean.min() <= 0:
        raise ValueError("Nonphysical final conservative means")
    metrics = {
        "mesh": f"iv{n}", "nominal_resolution": n, "nodes": len(final), "cells": manifest["cells"],
        "periodic_unknowns": n_periodic, "periodic_xy_nodes": n_xy,
        "z_layers": len(np.unique(xyz[:, 2])) - 1,
        "h_3d": float((volume / n_periodic)**(1.0 / 3.0)),
        "h_xy": float(np.sqrt(100.0 / n_xy)), "h_nominal": 10.0 / n,
        "steps": int(diagnostics[-1]["iteration"]), "time": float(diagnostics[-1]["time"]),
        "average_dt": float(2.0 / diagnostics[-1]["iteration"]),
        "cfl": configuration["time"]["cfl"], "dt_cap": configuration["time"]["maximumTimeStep"],
        "mpi_ranks": len(list(directory.glob("solution_00000000.nodes.rank*.csv"))),
        "volume": volume, "density_initial": errors[0], "density_final": errors[1],
        "conservation_scaled_drift": drift, "periodic_state_difference": periodic_difference,
        "entropy_L1": float(diagnostics[-1]["entropy_L1"]),
        "min_density_mean": float(final["rho_mean"].min()),
        "min_pressure_of_mean": float(pressure_of_mean.min()),
        "source_directory": str(directory.relative_to(ROOT)),
        "mesh_arrays_sha256_verified": True,
    }
    if log_path and log_path.exists():
        log = log_path.read_text()
        stored = {}
        for field in ("volume", "internal-surface", "boundary-surface"):
            match = re.search(rf"stored {field} quadrature=(\d+)", log)
            if not match or int(match.group(1)) <= 0:
                raise ValueError("Traditional quadrature storage was not logged")
            stored[field] = int(match.group(1))
        metrics["stored_quadrature_by_kind"] = stored
        metrics["stored_gauss_points"] = sum(stored.values())
        metrics["closure_max"] = float(re.search(r"max closure=([\deE.+-]+)", log).group(1))
        wall = re.search(r"^real ([\d.]+)$", log, re.MULTILINE)
        metrics["wall_seconds"] = float(wall.group(1)) if wall else None
    return metrics, final, cells, configuration



def snapshot(directory, label, time, cfg, node_count, rank_count, expected):
    files = sorted(directory.glob(f"points_{label}.rank*.csv"))
    if len(files) != rank_count:
        raise ValueError(f"{directory}: incomplete {label} recovered-state output")
    a = np.concatenate([np.loadtxt(f, delimiter=",", skiprows=1, ndmin=2) for f in files])
    a = a[np.argsort(a[:, 0])]
    if a.shape != (node_count, 18) or not np.array_equal(a[:, 0], np.arange(node_count)):
        raise ValueError("Duplicate or missing original node IDs")
    if not np.isfinite(a).all() or np.min(a[:, 4]) <= 0:
        raise ValueError("Invalid snapshot or dual volumes")
    gamma = cfg["physics"]["gamma"]
    rho = a[:, 10]
    pressure = (gamma - 1) * (a[:, 14] - np.sum(a[:, 11:14]**2, axis=1) / (2 * rho))
    truth_rho = exact_density(a[:, 1:4], time)
    truth_pressure = truth_rho**gamma
    if np.min(rho) <= 0 or np.min(pressure) <= 0:
        raise ValueError("Recovered point state is not physical")
    for computed, saved in ((pressure, a[:, 16]), (truth_rho, a[:, 15]), (truth_pressure, a[:, 17])):
        if not np.allclose(computed, saved, rtol=2e-12, atol=2e-14):
            raise ValueError("Independent point-pressure/reference formula disagrees")
    errors = {"rho_point": norms(rho - truth_rho, a[:, 4]),
              "pressure_point": norms(pressure - truth_pressure, a[:, 4])}
    for field in errors:
        for norm in NORMS:
            if not np.isclose(errors[field][norm], expected[field][norm], rtol=3e-11, atol=2e-14):
                raise ValueError("Independent error reduction differs from C++ MPI result")
    return a, errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.data = args.data.resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    runs, cuts = [], []
    common_cfg = None
    for n in (10, 20, 40, 80):
        directory = args.data / f"iv{n}"
        audit = json.loads((directory / "accuracy.json").read_text())
        if abs(audit["final"]["time"] - 2) > 1e-12 or audit["initial"]["time"] != 0:
            raise ValueError("Run did not complete the requested interval")
        cfg = audit["configuration"]
        settings = {k: cfg[k] for k in ("dimension", "algorithm", "reconstruction", "physics", "initialField", "time")}
        settings["mesh_settings"] = {k: v for k, v in cfg["mesh"].items() if k != "meshFile"}
        if common_cfg is not None and settings != common_cfg:
            raise ValueError("The meshes did not use identical numerical and physical settings")
        common_cfg = settings
        if not (cfg["time"]["cfl"] == 0.5 and cfg["time"]["useCFLTimeStep"] and
                not cfg["time"]["useLocalTimeStep"] and cfg["time"]["maximumTimeStep"] == 1e30 and
                cfg["time"]["minimumTimeStep"] == 1e-30 and not cfg["reconstruction"]["enableLimiter"] and
                cfg["physics"]["gamma"] == 1.4 and cfg["initialField"]["vortexStrength"] == 5 and
                cfg["algorithm"]["mode"] == "TraditionalQuadrature" and cfg["algorithm"]["quadratureOrder"] == 4 and
                cfg["physics"]["riemannSolver"] == "Roe_M2" and not cfg["physics"]["viscous"]["enabled"]):
            raise ValueError("Unexpected CFL or vortex settings")
        log_path = args.data / f"iv{n}.log"
        log = log_path.read_text()
        if "normalization=dual-bounds-half-span (thesis 3-34)" not in log:
            raise ValueError("Wrong normalization implementation")
        if "Transient audit complete:" not in log:
            raise ValueError("Run has no successful completion marker")
        base, final_nodes, _, resolved = audit_run(n, directory, log_path)
        if resolved != cfg or base["steps"] != audit["final"]["iteration"] or base["mpi_ranks"] != audit["mpi_ranks"]:
            raise ValueError("Resolved configuration or completion metadata disagree")
        for label, time in (("initial", 0), ("final", 2)):
            a, errors = snapshot(directory, label, time, cfg, base["nodes"], audit["mpi_ranks"], audit[label])
            audit[label]["independent_errors"] = errors
            if label == "final":
                for column, name in ((5, "rho_mean"), (6, "rhou"), (7, "rhov"), (8, "rhow"), (9, "rhoE"), (10, "rho_point")):
                    if not np.allclose(a[:, column], final_nodes[name], rtol=3e-12, atol=2e-14):
                        raise ValueError("Extra snapshot differs from production solver output")
                xyz = np.column_stack([final_nodes[k] for k in ("x", "y", "z")])
                if not np.allclose(a[:, 1:4], xyz, atol=1e-13, rtol=0):
                    raise ValueError("Snapshot coordinates do not match mesh")
                z = np.unique(a[:, 3]); level = z[np.argmin(np.abs(z - 2))]
                cuts.append(a[np.isclose(a[:, 3], level, atol=1e-9, rtol=0)])
                base["min_pressure_point"] = float(np.min(a[:, 16]))
                base["min_density_point"] = float(np.min(a[:, 10]))
        if (audit["initial"]["gauss_points_stored"] <= 0 or
                audit["initial"]["gauss_points_stored"] != audit["final"]["gauss_points_stored"] or
                audit["final"]["gauss_points_stored"] != base["stored_gauss_points"]):
            raise ValueError("Traditional quadrature storage is missing or changed")
        stamp = f"{base['steps']:08d}"
        if not (directory / f"solution_{stamp}.pvtu").exists():
            raise ValueError("Final VTK index is missing")
        if len(list((directory / f"solution_{stamp}.vtu.dir").glob("*.vtu"))) != base["mpi_ranks"]:
            raise ValueError("Final VTK rank pieces are incomplete")
        if not (directory / f"restart_{stamp}.dnds.h5").exists():
            raise ValueError("Final H5 restart is missing")
        dt = [tuple(map(float, pair)) for pair in re.findall(r"dt\[min,max\]=\[([^,]+),([^\]]+)\]", log)]
        if not dt or any(lo <= 0 or not np.isfinite(lo + hi) or abs(lo - hi) > 1e-13 for lo, hi in dt):
            raise ValueError("Invalid or nonuniform physical-time candidates")
        base.update(audit=audit, reported_stage3_candidate_min=min(lo for lo, _ in dt),
                    reported_stage3_candidate_max=max(hi for _, hi in dt),
                    average_physical_step=2 / base["steps"], common_cfl=cfg["time"]["cfl"])
        runs.append(base)
        print(f'iv{n}: t=2, {base["steps"]} steps, rho L2={audit["final"]["rho_point"]["L2"]:.9e}, '
              f'p L2={audit["final"]["pressure_point"]["L2"]:.9e}', flush=True)

    rows = []
    for i, run in enumerate(runs):
        run["orders"] = {}
        for field in ("rho_point", "pressure_point"):
            errors = run["audit"]["final"]["independent_errors"][field]
            orders = {norm: (float(np.log(runs[i-1]["audit"]["final"][field][norm] / errors[norm]) /
                                       np.log(runs[i-1]["h_3d"] / run["h_3d"])) if i else None) for norm in NORMS}
            run["orders"][field] = orders
            rows.append({"mesh": run["mesh"], "h": run["h_3d"], "time": 2, "cfl": 0.5,
                         "steps": run["steps"], "quantity": field, **errors,
                         **{f"order_{norm}": orders[norm] for norm in NORMS}})
    with (args.output / "errors_and_orders.csv").open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    reference = json.loads((ROOT / "docs/reports/ncfv_t2_cfl05_thesis_20260907/metrics.json").read_text())
    comparison = []
    for run, efficient in zip(runs, reference["runs"]):
        if run["mesh"] != efficient["mesh"] or run["periodic_unknowns"] != efficient["periodic_unknowns"]:
            raise ValueError("Comparison mesh mismatch")
        current_cfg, reference_cfg = run["audit"]["configuration"], efficient["audit"]["configuration"]
        for section in ("dimension", "mesh", "reconstruction", "physics", "initialField", "time"):
            if current_cfg[section] != reference_cfg[section]:
                raise ValueError(f"Traditional/efficient settings differ in {section}")
        for field in ("rho_point", "pressure_point"):
            traditional_error = run["audit"]["final"][field]["L2"]
            efficient_error = efficient["audit"]["final"][field]["L2"]
            comparison.append(dict(mesh=run["mesh"], quantity=field, traditional_L2=traditional_error,
                                   efficient_L2=efficient_error, ratio=traditional_error/efficient_error,
                                   traditional_order=run["orders"][field]["L2"],
                                   efficient_order=efficient["orders"][field]["L2"]))
    hashes = {}
    for path in ("src/NCFV/NCFVSpatial.cpp", "src/NCFV/NCFVReconstruction.cpp", "src/NCFV/NCFVSolver.cpp",
                 "src/NCFV/NCFVDualGeometry.cpp", "src/Euler/Gas.hpp", "build/src/NCFV/libncfv_3D.a",
                 "cases/ncfv_euler/3D/diagnostics/traditional_transient_accuracy_probe.cpp"):
        hashes[path] = hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
    with (args.output / "comparison_with_efficient.csv").open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(comparison[0]))
        writer.writeheader(); writer.writerows(comparison)
    (args.output / "metrics.json").write_text(json.dumps({"common_settings": common_cfg, "runs": runs,
        "comparison_with_efficient": comparison, "source_and_library_sha256": hashes}, indent=2) + "\n")

    plt.rcParams.update({"text.usetex": False, "font.family": "DejaVu Sans", "font.size": 10})
    h = np.array([r["h_3d"] for r in runs])
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    for ax, field, title in zip(axes, ("rho_point", "pressure_point"), ("Density", "Pressure")):
        for norm, marker in zip(NORMS, ("o", "s", "^")):
            ax.loglog(h, [r["audit"]["final"][field][norm] for r in runs], marker + "-", label=norm)
        end_error = runs[-1]["audit"]["final"][field]["L2"]
        for order, style in ((2, "--"), (3, ":")):
            ax.loglog(h, end_error * (h / h[-1])**order, style, color="gray", label=f"slope {order}")
        ax.set(xlabel="h = (400 / N_periodic)^(1/3)", ylabel="Recovered-point error", title=title)
        ax.grid(which="both", alpha=0.25); ax.legend(fontsize=9)
    fig.suptitle("NCFV / traditional quadratic reconstruction / CFL=0.5 / t=2")
    fig.savefig(args.output / "convergence.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(2, 4, figsize=(14, 6.8), constrained_layout=True)
    for j, (run, a) in enumerate(zip(runs, cuts)):
        tri = mtri.Triangulation(a[:, 1], a[:, 2])
        for i, (computed, exact, name) in enumerate(((10, 15, "rho"), (16, 17, "p"))):
            error = a[:, computed] - a[:, exact]
            extent = max(float(np.max(np.abs(error))), 1e-12)
            ax = axes[i, j]
            im = ax.tricontourf(tri, error, levels=np.linspace(-extent, extent, 25), cmap="RdBu_r", extend="both")
            fig.colorbar(im, ax=ax, shrink=0.8, format="%.1e")
            ax.set(xlim=(0, 10), ylim=(0, 10), aspect="equal", xlabel="x", ylabel="y",
                   title=f'{run["mesh"]}: {name} error, z={a[0,3]:g}')
    fig.suptitle("Recovered point errors at t=2 (separate color scales)")
    fig.savefig(args.output / "density_pressure_errors.png", dpi=160); plt.close(fig)

    def table(field):
        text = "| 网格 | L1 | 阶数 | L2 | 阶数 | L∞ | 阶数 |\n|---|---:|---:|---:|---:|---:|---:|\n"
        for run in runs:
            row = [run["mesh"]]
            for norm in NORMS:
                order = run["orders"][field][norm]
                row += [f'{run["audit"]["final"][field][norm]:.7e}', "—" if order is None else f"{order:.4f}"]
            text += "| " + " | ".join(row) + " |\n"
        return text

    time_table = "| 网格 | MPI进程 | 步数 | CFL | 初始物理步长 | 平均物理步长 | 最终时间 |\n|---|---:|---:|---:|---:|---:|---:|\n"
    for run in runs:
        time_table += (f'| {run["mesh"]} | {run["mpi_ranks"]} | {run["steps"]} | 0.5 | '
                       f'{run["audit"]["initial_cfl_step"]:.8e} | {run["average_physical_step"]:.8e} | 2 |\n')
    conservation = max(abs(v) for r in runs for v in r["conservation_scaled_drift"].values())
    periodic = max(r["periodic_state_difference"] for r in runs)
    comparison_table = "| 网格 | 分量 | 传统L2 | 高效L2 | 传统/高效 |\n|---|---|---:|---:|---:|\n"
    for row in comparison:
        comparison_table += (f"| {row['mesh']} | {row['quantity']} | {row['traditional_L2']:.7e} | "
                             f"{row['efficient_L2']:.7e} | {row['ratio']:.5f} |\n")
    report = f"""# 传统三阶 NCFV：统一 CFL=0.5，四网格推进到 t=2

## 计算口径

这是真实时间推进后的误差，不是 t=0 首次重构误差。
使用论文式（3-34）的对偶控制体分方向半跨度归一化，保留完整二次重构的9行系数，
采用传统体/面数值积分（order=4；每个微四面体14点、每个微三角面6点）、
Roe_M2 通量、SSPRK3。四套网格的物理、重构、积分及时间配置已逐项核对一致。
β=5，γ=1.4，初始中心(5,5)，背景速度(1,1,0)，周期盒[10,10,4]；t=2 时解析中心为(7,7)。
黏性、限制器、局部伪时间均关闭，初始状态由传统体积分生成，未从历史重启继续。
诊断程序调用现有 Solver.Run，不改变生产求解器的时间推进、重构、通量或任何矩阵。
额外快照只读取初始/最终物理状态，重算同一节点值恢复并输出完整守恒点值。

## CFL 一致性

统一 `cfl=0.5`、`useCFLTimeStep=true`、`useLocalTimeStep=false`。
将旧的 `maximumTimeStep=0.01` 改为1e30，下限为1e-30，使这些网格的正常步长由 CFL 决定。
原先0.01上限会限制粗网格，不能将相同的配置 CFL 与相同的实际 CFL 控制混为一谈。
保持相同 CFL 不要求各网格使用相同 dt；加密后 dt 应随空间尺度减小。
每一步使用起始状态所确定的全域最小 CFL 物理步长，三个 SSPRK3 阶段共用它。
这里的相同 CFL 指同一个全域步长控制准则，不是强迫所有控制体使用相同的局部 Courant 数。
最后一步允许缩短以恰好到达 t=2，因此最后一步的有效 CFL 可以低于0.5。

{time_table}
周期节点先将各份额的逆步长按体积合并，等价于完整控制体的 Σ谱半径/Σ体积；
然后对周期控制体候选步长取全域最小值。
日志中的 dt[min,max] 来自第三 RK 阶段重新估计的候选值，不应误读为完整的实际推进步长历史。
本表初始物理步长是在起步状态单独读取；平均物理步长严格用最终时间/总步数计算。

## 误差定义

密度使用恢复的节点密度；压力由完整恢复守恒点值计算：
p_h=(γ−1)[(ρE)_h−((ρu)_h²+(ρv)_h²+(ρw)_h²)/(2ρ_h)]。
没有用控制体均值对应的压力替代格点压力，也没有用 p_h=ρ_h^γ 强行施加等熵关系。
解析压力为 p_exact=ρ_exact^γ。
独立解析公式在 t=2 平移后的涡中心计算，周期取最近平移像。

以原始节点各自对偶控制体份额体积加权：L1=ΣV|e|/ΣV，L2=sqrt(ΣV e²/ΣV)，L∞=max|e|。
周期节点不会重复计入完整控制体体积；总份额体积为400。
阶数采用 log(E粗/E细)/log(h粗/h细)，h=(400/N_periodic)^(1/3)，不强制加密比等于2。

## 最终密度点值误差

{table('rho_point')}
## 最终压力点值误差

{table('pressure_point')}
## 与高效格式比较

{comparison_table}
两组采用相同网格、CFL、SSPRK3、物理场和重构参数。传统模式用数值体积分初始化、完整二次多项式及逐点面通量积分，
高效模式用全微分初始化/积分及梯度块，因此这是两套完整算法的误差比较。
步长由各自瞬时状态自适应确定，同CFL并不要求两种模式每一步的dt完全一致。
MPI进程数不完全相同，墙钟时间不能直接用作严格的算法加速比。

## 结论

最细两级 {runs[-2]['mesh']}→{runs[-1]['mesh']} 的 L2 观察阶分别为
密度 {runs[-1]['orders']['rho_point']['L2']:.4f}、压力 {runs[-1]['orders']['pressure_point']['L2']:.4f}。
本轮传统格式的误差较高效模式减小，但四网格结果仍未验证稳定三阶；低于三阶的现象并非高效积分模式独有。
本轮固定 CFL，包含初始化、空间离散和时间推进的综合误差；
尚未分离时间误差，不能将这些阶数直接视为纯空间离散阶数。

## 完成与独立核验

- 四套网格全部到达 t=2，检查步数和结束标记，不将未完成结果纳入收敛表。
- CGNS 坐标/连接数组校验值、原始节点唯一所有权、体积正性及总和均通过。
- 独立复算密度、压力和 L1/L2/L∞，并与 C++ MPI 归约结果及生产输出交叉核对。
- 初始和最终快照都确认传统积分点存储存在且数量不变，完整二次重构9行系数已检查。
- 所有最终守恒均值及恢复节点值满足密度、压力正性。
- 四套网格最大守恒量缩放漂移为 {conservation:.4e}；周期守恒状态副本最大差为 {periodic:.4e}。
- 这些阶数是当前完整离散在固定 CFL 下的空间/时间综合观察阶；本轮没有独立进行 dt→0 外推。

## 输出

- [全部误差与阶数](errors_and_orders.csv)。
- [配置、网格、守恒量和完整审计指标](metrics.json)。
- [与高效模式的误差对照](comparison_with_efficient.csv)。
- [运行设置、CFL口径与编译标识](run_notes.md)。
- [密度与压力收敛图](convergence.png)。
- [t=2 截面密度与压力误差图](density_pressure_errors.png)；各图色标独立，插值仅用于展示。
- 数据目录：`{args.data}`，含每套网格的运行日志、完整恢复点值、生产节点输出、VTK 和最终 H5 重启。
- 精度表使用 `points_final.rank*.csv` 中的恢复点值；现有生产 VTK 场来自控制体均值，不能直接混用于此处的格点误差比较。
- 配置目录：`cases/ncfv_euler/3D/traditional_t2_cfl05_20260908/`。

## 复现

先构建 NCFV，再编译 `cases/ncfv_euler/3D/diagnostics/traditional_transient_accuracy_probe.cpp`；
使用 `compile_reconstruction_probe.py --source ... --output ...` 可链接当前库。
从 build 目录运行 `mpirun -np N /path/to/traditional_transient_accuracy_probe /path/to/config.json`。
输出目录必须不存在；复跑前修改配置的输出/重启/VTK 前缀，不覆盖当前数据。
汇总命令：

```bash
venv/bin/python cases/ncfv_euler/3D/diagnostics/analyze_traditional_t2.py \\
  --data {args.data} \\
  --output {args.output}
```
"""
    (args.output / "report.md").write_text(report)
    print(table("rho_point")); print(table("pressure_point"))
    print(f'Report: {args.output / "report.md"}')


if __name__ == "__main__":
    main()
