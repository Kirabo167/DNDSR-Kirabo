"""Redraw the NCFV dispersion/dissipation measurements with Chinese labels.

This is a plotting-only entry point: it reads the archived CSV and does not
launch a solver, alter the measurements, or change the original runner.
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
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT = ROOT / "docs/reports/ncfv_dispersion_dissipation_20261012"
FAMILIES = ("hex", "tet", "prism")
METHODS = ("LeastSquares", "SVDLeastSquares")
INTEGRATIONS = ("Traditional", "Efficient")
DIRECTIONS = ("x", "y", "xy", "xyz")
WAVES = ("entropy", "acoustic_plus")
FAMILY_LABELS = {"hex": "六面体", "tet": "四面体", "prism": "三棱柱"}
METHOD_LABELS = {"LeastSquares": "LS", "SVDLeastSquares": "SVDLS"}
INTEGRATION_LABELS = {"Traditional": "传统积分", "Efficient": "高效积分"}
DIRECTION_LABELS = {"x": "x轴方向", "y": "y轴方向", "xy": "xy斜向", "xyz": "xyz斜向"}
WAVE_LABELS = {"entropy": "熵波", "acoustic_plus": "正向声波"}
COLORS = {"LeastSquares": "#2471a3", "SVDLeastSquares": "#cb4335"}
LINESTYLES = {"Traditional": "-", "Efficient": "--"}
METHOD_MARKERS = {"LeastSquares": "o", "SVDLeastSquares": "x"}
DIRECTION_MARKERS = {"x": "o", "y": "s", "xy": "^", "xyz": "D"}
METRICS = (("phase_speed_ratio", "相速度比"), ("damping_ratio", "归一化耗散率"))
WAVENUMBER_LABEL = r"名义无量纲波数 $k_d h_d / \pi = 2m/N$"


def configure_font():
    candidates = ("WenQuanYi Micro Hei", "Noto Sans CJK SC", "Noto Sans CJK JP")
    chinese_text = "".join(
        list(FAMILY_LABELS.values()) + list(INTEGRATION_LABELS.values())
        + list(DIRECTION_LABELS.values()) + list(WAVE_LABELS.values())
        + [label for _, label in METRICS]
        + ["名义无量纲波数相对模态耦合重构方法积分方法传播方向色散与耗散轴向比较全方向理想值"]
    )
    for candidate in candidates:
        try:
            path = font_manager.findfont(candidate, fallback_to_default=False)
        except ValueError:
            continue
        glyphs = FT2Font(path).get_charmap()
        if all(ord(character) in glyphs for character in chinese_text):
            plt.rcParams.update({
                "font.family": candidate,
                "font.size": 10,
                "axes.unicode_minus": False,
                "mathtext.fontset": "dejavusans",
                "pdf.fonttype": 42,
                "ps.fonttype": 42,
                "savefig.dpi": 180,
            })
            print(f"中文字体：{candidate} ({path})", flush=True)
            return
    raise RuntimeError("没有找到包含所需中文字形的文泉驿或 Noto CJK 字体")


def read_rows(path):
    numeric_fields = ("harmonic", "nominal_reduced_wavenumber", "phase_speed_ratio", "damping_ratio", "leakage_ratio")
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"测量数据为空：{path}")
    for row in rows:
        for field in numeric_fields:
            value = float(row[field])
            if not math.isfinite(value):
                raise ValueError(f"测量数据包含非有限数：{field}")
            row[field] = int(value) if field == "harmonic" else value
    identifiers = ("family", "mode", "direction", "method", "integration", "harmonic")
    keys = [tuple(row[field] for field in identifiers) for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("测量数据包含重复的波数响应点")
    return rows


def selected(rows, family, wave, direction, method, integration):
    series = sorted((row for row in rows if row["family"] == family and row["mode"] == wave
                     and row["direction"] == direction and row["method"] == method
                     and row["integration"] == integration), key=lambda row: row["harmonic"])
    if not series:
        raise ValueError(f"缺少曲线数据：{family}/{wave}/{direction}/{method}/{integration}")
    return series


def draw_curves(ax, rows, family, wave, direction, metric):
    for method, integration in itertools.product(METHODS, INTEGRATIONS):
        series = selected(rows, family, wave, direction, method, integration)
        ax.plot([row["nominal_reduced_wavenumber"] for row in series], [row[metric] for row in series],
                color=COLORS[method], linestyle=LINESTYLES[integration],
                marker=METHOD_MARKERS[method], markersize=4.5, markerfacecolor="none",
                linewidth=1.4, label=f"{METHOD_LABELS[method]} / {INTEGRATION_LABELS[integration]}")
    ax.axhline(1 if metric == "phase_speed_ratio" else 0, color="#777777", linewidth=0.8)
    ax.grid(True, alpha=0.25)


def save_figure(fig, output, stem):
    for suffix in ("png", "pdf"):
        path = output / f"{stem}.{suffix}"
        fig.savefig(path, bbox_inches="tight")
        print(f"已保存：{path}", flush=True)
    plt.close(fig)


def export_overview(output, rows):
    fig, axes = plt.subplots(4, len(FAMILIES), figsize=(14.5, 11.5),
                             squeeze=False, sharex=True, sharey="row")
    for column, family in enumerate(FAMILIES):
        for wave_index, wave in enumerate(WAVES):
            for metric_index, (metric, label) in enumerate(METRICS):
                ax = axes[2 * wave_index + metric_index, column]
                draw_curves(ax, rows, family, wave, "x", metric)
                ax.set_title(f"{FAMILY_LABELS[family]} / {WAVE_LABELS[wave]} / x轴方向")
                if column == 0:
                    ax.set_ylabel(label)
        axes[-1, column].set_xlabel(WAVENUMBER_LABEL)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.97))
    fig.suptitle("NCFV 色散与耗散：三类网格的轴向比较", y=1.003, fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save_figure(fig, output, "overview")


def export_family_waves(output, rows):
    for family, wave in itertools.product(FAMILIES, WAVES):
        fig, axes = plt.subplots(2, 4, figsize=(15, 6.8), sharex=True)
        for column, direction in enumerate(DIRECTIONS):
            for row_index, (metric, label) in enumerate(METRICS):
                ax = axes[row_index, column]
                draw_curves(ax, rows, family, wave, direction, metric)
                ax.set_title(f"{DIRECTION_LABELS[direction]}：{label}")
                if column == 0:
                    ax.set_ylabel(label)
            axes[-1, column].set_xlabel(WAVENUMBER_LABEL, fontsize=9)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.97))
        fig.suptitle(f"NCFV 色散与耗散 / {FAMILY_LABELS[family]} / {WAVE_LABELS[wave]}",
                     y=1.015, fontsize=15)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        save_figure(fig, output, f"{family}_{wave}")


def export_mode_leakage(output, rows):
    fig, axes = plt.subplots(2, len(FAMILIES), figsize=(14.5, 7.7),
                             squeeze=False, sharex=True, sharey="row")
    for column, family in enumerate(FAMILIES):
        for wave_index, wave in enumerate(WAVES):
            ax = axes[wave_index, column]
            for method, integration, direction in itertools.product(METHODS, INTEGRATIONS, DIRECTIONS):
                series = selected(rows, family, wave, direction, method, integration)
                ax.plot([row["nominal_reduced_wavenumber"] for row in series],
                        [row["leakage_ratio"] for row in series],
                        color=COLORS[method], linestyle=LINESTYLES[integration],
                        marker=DIRECTION_MARKERS[direction], markersize=4,
                        markerfacecolor="none", alpha=0.8, linewidth=1.2)
            ax.set_yscale("symlog", linthresh=1e-8)
            ax.set_title(f"{FAMILY_LABELS[family]} / {WAVE_LABELS[wave]}")
            if column == 0:
                ax.set_ylabel("相对模态耦合")
            ax.grid(True, alpha=0.25)
        axes[-1, column].set_xlabel(WAVENUMBER_LABEL)

    method_handles = [Line2D([], [], color=COLORS[method], linewidth=2,
                             label=METHOD_LABELS[method]) for method in METHODS]
    integration_handles = [Line2D([], [], color="#555555", linestyle=LINESTYLES[integration], linewidth=2,
                                  label=INTEGRATION_LABELS[integration]) for integration in INTEGRATIONS]
    direction_handles = [Line2D([], [], color="#555555", linestyle="none", marker=DIRECTION_MARKERS[direction],
                                markerfacecolor="none", markersize=5,
                                label=DIRECTION_LABELS[direction]) for direction in DIRECTIONS]
    fig.legend(handles=method_handles, title="重构方法", loc="upper center",
               bbox_to_anchor=(0.135, 0.96), ncol=2)
    fig.legend(handles=integration_handles, title="积分方法", loc="upper center",
               bbox_to_anchor=(0.375, 0.96), ncol=2)
    fig.legend(handles=direction_handles, title="传播方向", loc="upper center",
               bbox_to_anchor=(0.755, 0.96), ncol=4)
    fig.suptitle("NCFV 相对模态耦合：三类网格与全部传播方向", y=1.005, fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.865))
    save_figure(fig, output, "mode_leakage")


def main():
    parser = argparse.ArgumentParser(description="从已有 NCFV 测量 CSV 重绘中文色散耗散图")
    parser.add_argument("--input", type=Path, default=DEFAULT_OUTPUT / "metrics.csv", help="已有测量 CSV")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="PNG/PDF 输出目录")
    args = parser.parse_args()
    rows = read_rows(args.input)
    configure_font()
    args.output.mkdir(parents=True, exist_ok=True)
    print(f"读取测量数据：{args.input}；共 {len(rows)} 个响应点", flush=True)
    export_overview(args.output, rows)
    export_family_waves(args.output, rows)
    export_mode_leakage(args.output, rows)
    print("完成：8 张中文图，各含 PNG 与嵌入字体的 PDF。", flush=True)


if __name__ == "__main__":
    main()
