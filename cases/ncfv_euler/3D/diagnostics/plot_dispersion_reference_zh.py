"""Export measured NCFV modified-wavenumber curves in a thesis figure style.

The entry point only reads CSV data and renders figures.  Measured curves stop
at the last available response point; only the exact spectral reference is
drawn over the full interval [0, pi].  No zero-wavenumber sample is inserted.
"""

import argparse
import csv
import itertools
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
from matplotlib.ft2font import FT2Font
from matplotlib.ticker import AutoMinorLocator, MultipleLocator


ROOT = Path(__file__).resolve().parents[4]
REPORT = ROOT / "docs/reports/ncfv_dispersion_dissipation_20261012"
DEFAULT_OUTPUT = REPORT / "reference_style"
FAMILIES = ("hex", "tet", "prism")
WAVES = ("entropy", "acoustic_plus")
DIRECTIONS = ("x", "y", "xy", "xyz")
PREVIEW_CASE = ("hex", "entropy", "x")
FAMILY_LABELS = {"hex": "六面体", "tet": "四面体", "prism": "三棱柱"}
WAVE_LABELS = {"entropy": "熵波", "acoustic_plus": "正向声波"}
DIRECTION_LABELS = {"x": "x 轴方向", "y": "y 轴方向", "xy": "xy 斜向", "xyz": "xyz 斜向"}
CURVES = (
    ("LeastSquares", "Traditional", "LS-传统积分", "-", "^", 0),
    ("LeastSquares", "Efficient", "LS-高效积分", "--", "v", 0),
    ("SVDLeastSquares", "Traditional", "SVDLS-传统积分", "-.", "o", 1),
    ("SVDLeastSquares", "Efficient", "SVDLS-高效积分", ":", "s", 1),
)
DERIVED_FIELDS = (
    "kappa", "modified_wavenumber_real", "modified_wavenumber_imag",
    "relative_dispersion_error",
)
AXIS_LABELS = {
    "modified_wavenumber_real": r"$\mathrm{Re}(k^{*})$",
    "modified_wavenumber_imag": r"$\mathrm{Im}(k^{*})$",
    "relative_dispersion_error": r"$\mathrm{Re}(k^{*})/\kappa-1$",
}


def configure_font(dpi):
    chinese_text = (
        "".join(FAMILY_LABELS.values()) + "".join(WAVE_LABELS.values())
        + "".join(DIRECTION_LABELS.values()) + "".join(curve[2] for curve in CURVES)
        + "精确谱图网格的色散与耗散特性曲线相对误差名义无量纲波数横轴"
        + "注重合空心标记交错显示仅连接实测点；（），：。"
    )
    for candidate in ("WenQuanYi Micro Hei", "Noto Sans CJK SC", "Noto Sans CJK JP"):
        try:
            path = font_manager.findfont(candidate, fallback_to_default=False)
        except ValueError:
            continue
        glyphs = FT2Font(path).get_charmap()
        if not all(ord(character) in glyphs for character in chinese_text):
            continue
        plt.rcParams.update({
            "font.family": candidate,
            "font.size": 9,
            "axes.unicode_minus": False,
            "mathtext.fontset": "stix",
            "axes.linewidth": 0.85,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": dpi,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        })
        print(f"中文字体：{candidate} ({path})", flush=True)
        return
    raise RuntimeError("未找到覆盖全部图中文字形的文泉驿或 Noto CJK 字体")


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or ())
        rows = list(reader)
    if not rows:
        raise ValueError(f"测量数据为空：{path}")
    required = {
        "family", "mode", "direction", "method", "integration", "harmonic",
        "nominal_reduced_wavenumber", "phase_speed_ratio", "damping_ratio",
    }
    if missing := required - fields:
        raise ValueError(f"CSV 缺少列：{sorted(missing)}")
    has_derived = all(field in fields for field in DERIVED_FIELDS)
    if any(field in fields for field in DERIVED_FIELDS) and not has_derived:
        raise ValueError("CSV 的修正波数列不完整")
    seen = set()
    for row in rows:
        for field in ("nominal_reduced_wavenumber", "phase_speed_ratio", "damping_ratio"):
            row[field] = float(row[field])
            if not math.isfinite(row[field]):
                raise ValueError(f"CSV 含非有限数据：{field}")
        row["harmonic"] = int(row["harmonic"])
        if row["nominal_reduced_wavenumber"] <= 0:
            raise ValueError("输入应为已有非零波数响应点；不会补入零波数数据")
        kappa = math.pi * row["nominal_reduced_wavenumber"]
        expected = (
            kappa, kappa * row["phase_speed_ratio"], -kappa * row["damping_ratio"],
            row["phase_speed_ratio"] - 1,
        )
        for field, value in zip(DERIVED_FIELDS, expected):
            if has_derived:
                row[field] = float(row[field])
                if not math.isfinite(row[field]) or not math.isclose(
                    row[field], value, rel_tol=1e-12, abs_tol=1e-12
                ):
                    raise ValueError(f"修正波数与原始相速度/耗散率不一致：{field}")
            else:
                row[field] = value
        if kappa > math.pi:
            raise ValueError("响应波数超出绘图区间 [0, π]")
        key = tuple(row[field] for field in (
            "family", "mode", "direction", "method", "integration", "harmonic"
        ))
        if key in seen:
            raise ValueError(f"测量点重复：{key}")
        seen.add(key)
    return rows


def select_series(rows, case, method, integration):
    family, wave, direction = case
    series = sorted((row for row in rows
                     if (row["family"], row["mode"], row["direction"]) == (family, wave, direction)
                     and row["method"] == method and row["integration"] == integration),
                    key=lambda row: row["kappa"])
    if not series:
        raise ValueError(f"缺少曲线：{case}/{method}/{integration}")
    return series


def case_label(case):
    family, wave, direction = case
    return f"{FAMILY_LABELS[family]}网格，{WAVE_LABELS[wave]}，{DIRECTION_LABELS[direction]}"


def draw_panel(ax, rows, case, metric):
    exact = [0, math.pi] if metric == "modified_wavenumber_real" else [0, 0]
    ax.plot([0, math.pi], exact, color="black", linewidth=0.65, label="精确谱", zorder=1)
    values = [0]
    last_kappa = 0
    measured_points = []
    for method, integration, label, linestyle, marker, marker_offset in CURVES:
        series = select_series(rows, case, method, integration)
        x_values = [row["kappa"] for row in series]
        y_values = [row[metric] for row in series]
        values.extend(y_values)
        measured_points.extend(zip(x_values, y_values))
        last_kappa = max(last_kappa, x_values[-1])
        ax.plot(x_values, y_values, color="black", linestyle=linestyle, linewidth=0.9,
                marker=marker, markersize=4.1, markerfacecolor="white",
                markeredgewidth=0.75, markevery=(marker_offset, 2), label=label, zorder=2)

    legend_anchor = None
    if metric == "modified_wavenumber_real":
        lower = min(values)
        ax.set_ylim(lower - 0.11 if lower < 0 else 0, math.pi + 0.035)
        ax.yaxis.set_major_locator(MultipleLocator(0.5))
        ax.yaxis.set_minor_locator(MultipleLocator(0.1))
        legend_location = "upper left"
    else:
        lower, upper = min(values), max(values)
        span = max(upper - lower, 0.1)
        ax.set_ylim(lower - 0.10 * span, upper + 0.075 * span)
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        legend_location = "lower right" if last_kappa < math.pi / 2 else "lower left"
        if legend_location == "lower left" and any(
            kappa < 1.65 and value < lower + 0.40 * span
            for kappa, value in measured_points
        ):
            legend_location = "upper right"
            # Keep the exact zero line above the legend and the descending
            # high-wavenumber curves below it on tetrahedral oblique cases.
            legend_anchor = (0.98, 0.88)

    ax.set_xlim(0, math.pi)
    ax.set_box_aspect(1)
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.tick_params(axis="both", which="major", direction="in", top=True, right=True,
                   length=4.0, width=0.75, labelsize=8, pad=3)
    ax.tick_params(axis="both", which="minor", direction="in", top=True, right=True,
                   length=2.1, width=0.6)
    ax.set_xlabel(r"$\kappa$", labelpad=1.5, fontsize=10)
    ax.set_ylabel(AXIS_LABELS[metric], labelpad=3, fontsize=10)
    ax.legend(loc=legend_location, bbox_to_anchor=legend_anchor,
              frameon=False, fontsize=7.7, handlelength=2.5,
              borderaxespad=0.65, labelspacing=0.3, handletextpad=0.65)


def save_figure(fig, output, stem):
    for extension in ("png", "pdf"):
        path = output / f"{stem}.{extension}"
        fig.savefig(path)
        print(f"已保存：{path}", flush=True)
    plt.close(fig)


def export_pair(output, rows, case):
    fig = plt.figure(figsize=(7.4, 4.05))
    for left, metric in zip((0.102, 0.565), DERIVED_FIELDS[1:3]):
        ax = fig.add_axes((left, 0.302, 0.335, 0.612))
        draw_panel(ax, rows, case, metric)
    fig.text(0.5, 0.192, f"图 1  NCFV 色散与耗散特性曲线（{case_label(case)}）",
             ha="center", va="center", fontsize=10)
    fig.text(0.5, 0.121, r"横轴为名义无量纲波数 $\kappa$；曲线仅连接实测点。",
             ha="center", va="center", fontsize=8.5)
    fig.text(0.5, 0.061, "注：LS 与 SVDLS 曲线重合，空心标记交错显示。",
             ha="center", va="center", fontsize=8.5)
    save_figure(fig, output, "_".join(case) + "_dispersion_dissipation")


def export_error(output, rows, case):
    fig = plt.figure(figsize=(4.8, 4.4))
    ax = fig.add_axes((0.2, 0.319, 0.60, 0.655))
    draw_panel(ax, rows, case, "relative_dispersion_error")
    fig.text(0.5, 0.213, "图 2  NCFV 相对色散误差曲线", ha="center", va="center", fontsize=10)
    fig.text(0.5, 0.157, f"（{case_label(case)}）", ha="center", va="center", fontsize=9.2)
    fig.text(0.5, 0.099, r"横轴为名义无量纲波数 $\kappa$；曲线仅连接实测点。",
             ha="center", va="center", fontsize=8.2)
    fig.text(0.5, 0.045, "注：LS 与 SVDLS 曲线重合，空心标记交错显示。",
             ha="center", va="center", fontsize=8.2)
    save_figure(fig, output, "_".join(case) + "_relative_error")


def export_combined(output, rows, case):
    fig = plt.figure(figsize=(7.4, 7.5))
    for left, metric in zip((0.102, 0.565), DERIVED_FIELDS[1:3]):
        ax = fig.add_axes((left, 0.596, 0.335, 0.331))
        draw_panel(ax, rows, case, metric)
    ax = fig.add_axes((0.333, 0.142, 0.335, 0.331))
    draw_panel(ax, rows, case, "relative_dispersion_error")
    fig.text(0.5, 0.527, f"图 1  NCFV 色散与耗散特性曲线（{case_label(case)}）",
             ha="center", va="center", fontsize=10)
    fig.text(0.5, 0.075, "图 2  NCFV 相对色散误差曲线", ha="center", va="center", fontsize=10)
    fig.text(0.5, 0.028,
             r"名义无量纲波数 $\kappa$；LS 与 SVDLS 曲线重合，空心标记交错显示。",
             ha="center", va="center", fontsize=8.5)
    save_figure(fig, output, "_".join(case) + "_combined")


def main():
    parser = argparse.ArgumentParser(description="参照论文黑白图，从已有 NCFV CSV 绘制中文修正波数曲线")
    parser.add_argument("--input", type=Path, default=DEFAULT_OUTPUT / "modified_wavenumber.csv",
                        help="已有测量或修正波数 CSV")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="PNG/PDF 输出目录")
    parser.add_argument("--dpi", type=int, default=240, help="PNG 分辨率，默认 240 dpi")
    parser.add_argument("--case", nargs=3, metavar=("FAMILY", "WAVE", "DIRECTION"),
                        help="只绘制一个算例，用于检查布局")
    args = parser.parse_args()
    if args.dpi <= 0:
        parser.error("--dpi 必须为正数")
    cases = [tuple(args.case)] if args.case else list(itertools.product(FAMILIES, WAVES, DIRECTIONS))
    if any(family not in FAMILIES or wave not in WAVES or direction not in DIRECTIONS
           for family, wave, direction in cases):
        parser.error("--case 的网格、波型或传播方向不受支持")
    rows = read_rows(args.input)
    configure_font(args.dpi)
    args.output.mkdir(parents=True, exist_ok=True)
    print(f"读取：{args.input}；共 {len(rows)} 个响应点；将绘制 {len(cases)} 个算例", flush=True)
    for case in cases:
        export_pair(args.output, rows, case)
        export_error(args.output, rows, case)
    preview_case = cases[0] if args.case else PREVIEW_CASE
    export_combined(args.output, rows, preview_case)
    figure_count = 2 * len(cases) + 1
    print(f"完成：{figure_count} 张黑白中文图，各含 PNG 和嵌入字体的 PDF。", flush=True)


if __name__ == "__main__":
    main()
