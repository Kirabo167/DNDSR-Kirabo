"""Measure periodic NCFV spatial dispersion and dissipation for LS and SVDLS.

The C++ probe applies the linearized production RHS to cosine/sine Euler waves.
This script launches that probe, validates its provenance, and exports CSV,
figures, and a Chinese report. It does not import the DNDSR Python bindings.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import shlex
import subprocess


ROOT = Path(__file__).resolve().parents[4]
CASE_ROOT = ROOT / "cases/ncfv_euler/3D"
METHODS = ("LeastSquares", "SVDLeastSquares")
INTEGRATIONS = ("Traditional", "Efficient")
DIRECTIONS = ("x", "y", "xy", "xyz")
WAVES = ("entropy", "acoustic_plus")
LABELS = {"LeastSquares": "LS", "SVDLeastSquares": "SVDLS"}
COLORS = {"LeastSquares": "#2471a3", "SVDLeastSquares": "#cb4335"}


def digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def configuration_and_mesh(family, level):
    if family == "prism":
        config = CASE_ROOT / "NCFV_traditional_variational_vortex.json"
        mesh = CASE_ROOT / f"iv{level}_3d_1_dnds.cgns"
    else:
        config = CASE_ROOT / f"NCFV_periodic_{family}_iv{level}.json"
        mesh = CASE_ROOT / "generated_periodic_meshes" / f"periodic_{family}_iv{level}.cgns"
    return config, mesh


def source_fingerprints():
    sources = sorted((ROOT / "src/NCFV").glob("*.cpp"))
    sources += sorted((ROOT / "src/NCFV").glob("*.hpp"))
    sources += [Path(__file__).resolve(), Path(__file__).with_name("dispersion_dissipation_probe.cpp")]
    return {str(path.relative_to(ROOT)): digest(path) for path in sources}


def validate_probe(data, args, family, method, integration, result_path):
    if data.get("mpi_ranks") != args.ranks:
        raise RuntimeError(f"Unexpected MPI rank count in {result_path}")
    if not math.isclose(data.get("amplitude", 0), args.amplitude, rel_tol=1e-12):
        raise RuntimeError(f"Unexpected amplitude in {result_path}")
    reconstruction = data["configuration"]["reconstruction"]
    if reconstruction.get("method") != method:
        raise RuntimeError(f"Unexpected reconstruction method in {result_path}")
    expected_mode = {"Traditional": "TraditionalQuadrature", "Efficient": "EfficientDifferential"}
    if data["configuration"]["algorithm"]["mode"] != expected_mode[integration]:
        raise RuntimeError(f"Unexpected integration method in {result_path}")
    xyz_max = min(args.max_harmonic, args.prism_xyz_max_harmonic) if family == "prism" else args.max_harmonic
    expected = {(wave, direction, harmonic) for wave, direction in itertools.product(WAVES, DIRECTIONS)
                for harmonic in range(1, (xyz_max if direction == "xyz" else args.max_harmonic) + 1)}
    actual = [(row["mode"], row["direction"], row["harmonic"]) for row in data["rows"]]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise RuntimeError(f"Incomplete or duplicate wave sweep in {result_path}")
    for row in data["rows"]:
        for field in ("omega_exact", "lambda_real", "lambda_imag", "phase_speed_ratio",
                      "damping_ratio", "attenuation_per_period", "leakage_ratio",
                      "constant_leakage", "central_difference_asymmetry", "phi_norm"):
            if not isinstance(row.get(field), (int, float)) or not math.isfinite(row[field]):
                raise RuntimeError(f"Nonfinite {field} in {result_path}: {row}")
        if row["phi_norm"] <= 0 or row["omega_exact"] <= 0:
            raise RuntimeError(f"Degenerate wave in {result_path}: {row}")


def export_csv(output, rows):
    preferred = ["family", "level", "method", "integration", "mode", "direction", "harmonic",
                 "nominal_reduced_wavenumber", "nominal_points_per_wavelength", "k_x", "k_y", "k_z",
                 "omega_exact", "lambda_real", "lambda_imag", "phase_speed_ratio",
                 "phase_speed_error", "damping_ratio", "attenuation_per_period", "leakage_ratio",
                 "constant_leakage", "central_difference_asymmetry", "rhs_seconds", "phi_norm",
                 "nodes", "cells", "periodic_unknowns", "volume", "uniform_rhs_max"]
    fields = preferred + sorted(set().union(*(row.keys() for row in rows)) - set(preferred))
    with (output / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def selected(rows, family, wave, direction, method, integration):
    return sorted((row for row in rows if row["family"] == family and row["mode"] == wave
                   and row["direction"] == direction and row["method"] == method
                   and row["integration"] == integration), key=lambda row: row["harmonic"])


def draw_curves(ax, rows, family, wave, direction, metric):
    for method, integration in itertools.product(METHODS, INTEGRATIONS):
        series = selected(rows, family, wave, direction, method, integration)
        ax.plot([row["nominal_reduced_wavenumber"] for row in series], [row[metric] for row in series],
                color=COLORS[method], linestyle="-" if integration == "Traditional" else "--",
                marker="o" if integration == "Traditional" else "x", markersize=4,
                label=f"{LABELS[method]} / {integration}")
    ax.axhline(1 if metric == "phase_speed_ratio" else 0, color="#777777", linewidth=0.8)
    ax.grid(True, alpha=0.25)


def export_figures(output, rows, families):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "savefig.dpi": 180})
    fig, axes = plt.subplots(4, len(families), figsize=(4.5 * len(families), 11),
                             squeeze=False, sharex=True)
    for column, family in enumerate(families):
        for wave_index, wave in enumerate(WAVES):
            for metric_index, (metric, label) in enumerate((("phase_speed_ratio", "Phase speed / exact"),
                                                          ("damping_ratio", "Damping / exact frequency"))):
                ax = axes[2 * wave_index + metric_index, column]
                draw_curves(ax, rows, family, wave, "x", metric)
                ax.set_title(f"{family} / {wave} / x")
                ax.set_ylabel(label)
        axes[-1, column].set_xlabel(r"Nominal $k_x h_x / \pi = 2m/N$")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.005))
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(output / "overview.png", bbox_inches="tight")
    fig.savefig(output / "overview.pdf", bbox_inches="tight")
    plt.close(fig)
    for family, wave in itertools.product(families, WAVES):
        fig, axes = plt.subplots(2, 4, figsize=(14, 6.5), sharex=True)
        for column, direction in enumerate(DIRECTIONS):
            for row_index, (metric, label) in enumerate((("phase_speed_ratio", "Phase speed / exact"),
                                                       ("damping_ratio", "Damping / exact frequency"))):
                ax = axes[row_index, column]
                draw_curves(ax, rows, family, wave, direction, metric)
                ax.set_title(f"{direction}: {label}")
                if column == 0:
                    ax.set_ylabel(label)
            axes[-1, column].set_xlabel(r"Nominal $k_d h_d / \pi = 2m/N$")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.975))
        fig.suptitle(f"NCFV / {family} / {wave}", y=1.015, fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        fig.savefig(output / f"{family}_{wave}.png", bbox_inches="tight")
        fig.savefig(output / f"{family}_{wave}.pdf", bbox_inches="tight")
        plt.close(fig)
    fig, axes = plt.subplots(2, len(families), figsize=(4.5 * len(families), 6),
                             squeeze=False, sharex=True)
    for column, family in enumerate(families):
        for wave_index, wave in enumerate(WAVES):
            ax = axes[wave_index, column]
            for method, integration in itertools.product(METHODS, INTEGRATIONS):
                for direction in DIRECTIONS:
                    series = selected(rows, family, wave, direction, method, integration)
                    ax.plot([row["nominal_reduced_wavenumber"] for row in series],
                            [row["leakage_ratio"] for row in series],
                            color=COLORS[method], alpha=0.7,
                            linestyle="-" if integration == "Traditional" else "--")
            ax.set_yscale("symlog", linthresh=1e-8)
            ax.set_title(f"{family} / {wave} / all directions")
            ax.set_ylabel("Relative mode leakage")
            ax.grid(True, alpha=0.25)
        axes[-1, column].set_xlabel(r"Nominal $k_d h_d / \pi = 2m/N$")
    fig.tight_layout()
    fig.savefig(output / "mode_leakage.png")
    fig.savefig(output / "mode_leakage.pdf")
    plt.close(fig)


def number(value):
    return f"{value:.6g}"


def export_report(output, rows, runs, args):
    now = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M %Z")
    lines = ["# NCFV 色散与耗散：LS / SVDLS，传统 / 高效积分", "", f"生成时间：{now}。", "",
             "本报告测量**半离散空间算子**对周期小振幅波的瞬时响应；没有引入时间积分器。"
             "测试包含密度熵波和正向声波，比较 LeastSquares（LS）、SVDLeastSquares（SVDLS）"
             "以及 TraditionalQuadrature、EfficientDifferential 两种积分路径。", "",
             "## 测量定义", "",
             "在均匀背景态附近，用正、负小扰动的中心差分获得线性化右端项；"
             "同时使用正弦与余弦两个分量，先由 Euler 左本征向量抽取目标波支，"
             "再进行周期代表节点的对偶体积加权复投影。"
             "提取的复响应记为 λ，精确正向模态频率记为 ω。", "",
             "- 相速度比：−Im(λ)/ω；理想值为 1。", "- 归一化耗散：−Re(λ)/ω；理想值为 0，正值表示衰减。",
             "- 等效每精确周期幅值比：exp(2π Re(λ)/ω)；理想值为 1。这是投影系数换算值，并非实际时间推进测量。",
             "- 模态泄漏：全保守变量线性响应减去 λ 乘目标波后的加权范数相对值。",
             "- 目标波支泄漏（mode_leakage_ratio）：左本征向量抽取后的标量响应残差。", "",
             "这些量是**投影响应，不是全离散矩阵的本征谱**。"
             "若模态泄漏较大，该模态会与其他模态耦合，每周期幅值比不能直接预测长时间增长或衰减。"
             "中心差分提取的主 λ 不受均匀态常量右端项影响；"
             "正负扰动不对称指标含均匀态残差除以扰动幅值的背景项，只作为差分误差诊断界限。", "",
             "## 参数与网格", "",
             f"扰动幅值 {args.amplitude:g}；谐波 m=1…{args.max_harmonic}；"
             f"方向 x、y、xy、xyz；每次运行 {args.ranks} 个 MPI ranks，最多 {args.jobs} 个任务并行。"
             "统一采用无黏性 Roe 通量、关闭限制器、stencilSizeFactor=1.7、distanceWeightPower=1；"
             "最终参数以 raw/*.json 的 configuration 为准。", "",
             "初始波采用单纯形上复指数函数的解析级数积分构造对偶体积平均值，"
             "并记录级数末项 phi_integration_tail 与微体积权重不匹配诊断。", "",
             "域长 (Lx,Ly,Lz)=(10,10,4)，非零波数分量为 2πm/Lj。"
             f"横轴为非零波数方向的名义约化波数 kd hd/π=2m/{args.level}（hd=各方向域长/{args.level}），"
             "它在非均匀棱柱网格上是标度，不能当成所有节点的实际局部网格间距。", "",
             "| 网格 | 节点 | 单元 | 周期未知量 | 体积 | 均匀态 RHS 最大值 |",
             "|---|---:|---:|---:|---:|---:|"]
    for family in args.families:
        subset = [run for run in runs if run["family"] == family]
        first = subset[0]
        lines.append(f"| {family} IV{args.level} | {first['nodes']} | {first['cells']} | "
                     f"{first['periodic_unknowns']} | {number(first['volume'])} | "
                     f"{number(max(run['uniform_rhs_max'] for run in subset))} |")
    if "prism" in args.families:
        lines += ["", f"棱柱 xyz 方向仅扫描 m≤{min(args.max_harmonic, args.prism_xyz_max_harmonic)}，"
                  "保证低于当前棱柱网格垂向分辨率的 Nyquist 范围。"
                  "棱柱 IV20 原网格在 z 方向约 7–8 个层间距；hex/tet IV20 为每方向 20 格。"
                  "跨网格比较时，x/y/xy 方向更适合比较，xyz 方向还受垂向分辨率差异影响。"
                  if args.level == 20 else
                  "棱柱网格的垂向分辨率与 hex/tet 不同；xyz 高频点需结合实际采样层数判断别名。"]
    lines += ["", "## 结果比较", "",
              "下表为 x 向波，c/ce 为相速度比，d 为归一化耗散。"
              "全部方向和等效每周期幅值比见 metrics.csv。", ""]
    harmonics = sorted({1, min(4, args.max_harmonic), args.max_harmonic})
    header = "| 网格 | 波 | 重构 | 积分 | " + " | ".join(f"m={m}: c/ce, d" for m in harmonics) + " |"
    lines += [header, "|---|---|---|---|" + "---:|" * len(harmonics)]
    for family, wave, method, integration in itertools.product(args.families, WAVES, METHODS, INTEGRATIONS):
        series = {row["harmonic"]: row for row in selected(rows, family, wave, "x", method, integration)}
        values = " | ".join(f"{number(series[m]['phase_speed_ratio'])}, {number(series[m]['damping_ratio'])}"
                            for m in harmonics)
        lines.append(f"| {family} | {wave} | {LABELS[method]} | {integration} | {values} |")
    indexed = {(row["family"], row["mode"], row["direction"], row["harmonic"], row["method"],
                row["integration"]): row for row in rows}
    comparisons = []
    for family, wave, integration in itertools.product(args.families, WAVES, INTEGRATIONS):
        pairs = [(indexed[(family, wave, direction, harmonic, METHODS[0], integration)],
                  indexed[(family, wave, direction, harmonic, METHODS[1], integration)])
                 for direction, harmonic in itertools.product(DIRECTIONS, range(1, args.max_harmonic + 1))
                 if (family, wave, direction, harmonic, METHODS[0], integration) in indexed]
        phase = max(abs(left["phase_speed_ratio"] - right["phase_speed_ratio"]) for left, right in pairs)
        damping = max(abs(left["damping_ratio"] - right["damping_ratio"]) for left, right in pairs)
        comparisons.append({"family": family, "mode": wave, "integration": integration,
                            "maximum_LS_SVDLS_phase_difference": phase,
                            "maximum_LS_SVDLS_damping_difference": damping})
    lines += ["", "LS 与 SVDLS 在所有方向和波数点的最大绝对差：", "",
              "| 网格 | 波 | 积分 | 相速度比差 | 归一化耗散差 |", "|---|---|---|---:|---:|"]
    for row in comparisons:
        lines.append(f"| {row['family']} | {row['mode']} | {row['integration']} | "
                     f"{number(row['maximum_LS_SVDLS_phase_difference'])} | "
                     f"{number(row['maximum_LS_SVDLS_damping_difference'])} |")
    write_json(output / "comparison_summary.json", comparisons)
    worst_leakage = max(rows, key=lambda row: row["leakage_ratio"])
    maximum_asymmetry = max(row["central_difference_asymmetry"] for row in rows)
    maximum_constant = max(row["constant_leakage"] for row in rows)
    maximum_integration_tail = max(row.get("phi_integration_tail", 0) for row in rows)
    maximum_volume_mismatch = max(row.get("micro_volume_weight_mismatch", 0) for row in rows)
    growth = [row for row in rows if row["damping_ratio"] < -1e-8]
    lines += ["", "## 有效性与限制", "",
              f"最大模态泄漏 {number(worst_leakage['leakage_ratio'])}，发生于 "
              f"{worst_leakage['family']} / {LABELS[worst_leakage['method']]} / "
              f"{worst_leakage['integration']} / {worst_leakage['mode']} / "
              f"{worst_leakage['direction']} / m={worst_leakage['harmonic']}。"
              f"正负扰动不对称最大值 {number(maximum_asymmetry)}；"
              f"常量泄漏最大值 {number(maximum_constant)}。", "",
              f"复指数积分级数末项最大值 {number(maximum_integration_tail)}；"
              f"微体积权重不匹配最大值 {number(maximum_volume_mismatch)}。", "",
              f"耗散比小于 −1e−8 的投影增长点：{len(growth)}/{len(rows)}。"
              "局部投影增长不等价于已经证明整体时间推进不稳定；"
              "可结合模态耦合、本征值和时间推进后续验证。", "",
              "本次数据不包含时间积分误差、非线性大振幅效应或限制器作用。"
              "波数高于网格可解析范围时，物理波与采样波必须区分。"
              "同一谐波在不同方向的 |k| 不同，因此各方向曲线按方向内精确频率归一化。", "",
              "## 数据与复现", "", "- `metrics.csv`：所有网格、波型、方向、谐波的原始度量。",
              "- `overview.png` / `.pdf`：x 向总览。",
              "- `<family>_<wave>.png` / `.pdf`：四个方向的相速度与耗散曲线。",
              "- `mode_leakage.png` / `.pdf`：模态泄漏诊断。",
              "- `raw/*.json`：完整配置与探针输出；对应日志和 manifest 保留命令与哈希。",
              "- `manifest.json`：全部运行的配置、网格、二进制和源文件 SHA256，拒绝跨版本误复用。", "",
              "复跑命令（项目根目录；探针必须从相同源文件和 build 库重新构建）：", "", "```bash",
              shlex.join([str(ROOT / "venv/bin/python"), str(Path(__file__).resolve()),
                          "--binary", str(args.binary.resolve()), "--output", str(output),
                          "--ranks", str(args.ranks), "--jobs", str(args.jobs),
                          "--level", str(args.level), "--max-harmonic", str(args.max_harmonic),
                          "--prism-xyz-max-harmonic", str(args.prism_xyz_max_harmonic),
                          "--amplitude", str(args.amplitude), "--families", *args.families]),
              "```", ""]
    (output / "report.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ranks", type=int, default=2)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--max-harmonic", type=int, default=8)
    parser.add_argument("--prism-xyz-max-harmonic", type=int, default=3)
    parser.add_argument("--families", nargs="+", choices=("hex", "tet", "prism"),
                        default=["hex", "tet", "prism"])
    parser.add_argument("--level", type=int, default=20)
    parser.add_argument("--amplitude", type=float, default=1e-5)
    args = parser.parse_args()
    if min(args.ranks, args.jobs, args.max_harmonic, args.prism_xyz_max_harmonic, args.level) <= 0:
        parser.error("ranks, jobs, maximum harmonic, and level must be positive")
    if not math.isfinite(args.amplitude) or not 0 < args.amplitude <= 1e-3:
        parser.error("amplitude must be finite, positive, and at most 1e-3")
    if len(set(args.families)) != len(args.families):
        parser.error("mesh families must not be repeated")
    if any(family in ("hex", "tet") for family in args.families) and 2 * args.max_harmonic >= args.level:
        parser.error("hex/tet harmonics must be strictly below the structured-grid Nyquist frequency (2m < level)")
    binary = args.binary.resolve()
    output = args.output.resolve()
    raw = output / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    sources = source_fingerprints()
    binary_hash = digest(binary)
    environment = dict(os.environ, OMP_NUM_THREADS="1", DNDS_DIST_OMP_NUM_THREADS="1",
                       OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       HWLOC_COMPONENTS="-gl,-cuda,-nvml")
    for key in list(environment):
        if key.startswith("NCFV_"):
            environment.pop(key)
    enforced_environment = {key: environment[key] for key in (
        "OMP_NUM_THREADS", "DNDS_DIST_OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS", "HWLOC_COMPONENTS")}

    def run(case):
        family, method, integration = case
        config, mesh = configuration_and_mesh(family, args.level)
        name = f"{family}_iv{args.level}_{LABELS[method]}_{integration}"
        result_path = raw / f"{name}.json"
        manifest_path = raw / f"{name}.manifest.json"
        command = ["mpirun", "--mca", "pml", "ob1", "--mca", "btl", "self,vader",
                   "--bind-to", "none", "-np", str(args.ranks), str(binary), str(config),
                   str(result_path), method, integration, str(args.amplitude),
                   str(args.max_harmonic), str(mesh),
                   str(min(args.max_harmonic, args.prism_xyz_max_harmonic)
                       if family == "prism" else args.max_harmonic)]
        manifest = {"schema_version": 1, "command": command, "working_directory": str(ROOT / "build"),
                    "binary_sha256": binary_hash, "config_sha256": digest(config),
                    "mesh_sha256": digest(mesh), "source_sha256": sources,
                    "environment": enforced_environment, "removed_environment_prefix": "NCFV_"}
        if result_path.exists():
            if not manifest_path.exists() or json.loads(manifest_path.read_text()) != manifest:
                raise RuntimeError(f"Refusing stale result with different provenance: {result_path}")
        else:
            if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
                raise RuntimeError(f"Refusing to replace a different run: {manifest_path}")
            write_json(manifest_path, manifest)
            print(f"Start {name}", flush=True)
            with (raw / f"{name}.log").open("w") as log:
                subprocess.run(command, cwd=ROOT / "build", env=environment,
                               stdout=log, stderr=subprocess.STDOUT, check=True)
        data = json.loads(result_path.read_text())
        validate_probe(data, args, family, method, integration, result_path)
        run_metadata = {key: data[key] for key in ("nodes", "cells", "periodic_unknowns", "volume", "uniform_rhs_max")}
        run_metadata.update(family=family, level=args.level, method=method, integration=integration,
                            result=str(result_path.relative_to(output)), manifest=manifest)
        rows = []
        for probe_row in data["rows"]:
            row = dict(probe_row)
            k = row.pop("k")
            row.update({key: value for key, value in run_metadata.items() if key not in ("result", "manifest")})
            row.update(k_x=k[0], k_y=k[1], k_z=k[2], nominal_reduced_wavenumber=2 * row["harmonic"] / args.level,
                       nominal_points_per_wavelength=args.level / row["harmonic"],
                       phase_speed_error=row["phase_speed_ratio"] - 1)
            rows.append(row)
        print(f"Done {name}: {len(rows)} wave responses", flush=True)
        return rows, run_metadata

    cases = list(itertools.product(args.families, METHODS, INTEGRATIONS))
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(run, cases))
    rows = [row for result_rows, _ in results for row in result_rows]
    rows.sort(key=lambda row: (row["family"], row["mode"], row["direction"], row["harmonic"],
                               row["method"], row["integration"]))
    runs = [run_metadata for _, run_metadata in results]
    if source_fingerprints() != sources or digest(binary) != binary_hash:
        raise RuntimeError("Sources or probe binary changed during the sweep; refusing mixed results")
    write_json(output / "manifest.json", {"schema_version": 1, "runs": runs})
    export_csv(output, rows)
    export_figures(output, rows, args.families)
    export_report(output, rows, runs, args)
    print(f"Wrote {len(rows)} responses, figures, and report to {output}", flush=True)


if __name__ == "__main__":
    main()
