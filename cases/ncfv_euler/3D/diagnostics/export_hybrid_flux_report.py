"""Export native flux crossover results, archived comparisons, and editable Word.

This script reads only completed runs; missing or partial results are errors.
It records source/binary hashes because the repository has pre-existing edits.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
from pathlib import Path
import re
import subprocess
from zipfile import ZipFile

from lxml import etree

ROOT = Path(__file__).resolve().parents[4]
LEVELS = (10, 20, 40, 80)
CASES = ("TT", "EE", "ET", "TE", "EE_native_initialization")
LABELS = {"TT": "传统平均＋传统耗散", "EE": "高效平均＋高效耗散",
          "ET": "高效平均＋传统耗散", "TE": "传统平均＋高效耗散",
          "EE_native_initialization": "原始高效（原始初值）"}


def read_csv(path: Path):
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def number(value, digits=7):
    return "—" if value is None else f"{float(value):.{digits - 1}e}"


def table(headers, rows):
    def cells(items, tag):
        return "".join(f"<{tag}>{html.escape(str(x))}</{tag}>" for x in items)
    return ("<table><thead><tr>" + cells(headers, "th") + "</tr></thead><tbody>" +
            "".join("<tr>" + cells(row, "td") + "</tr>" for row in rows) + "</tbody></table>")


def paragraph(value):
    return f"<p>{html.escape(value)}</p>"


def digest(path):
    return hashlib.file_digest(path.open("rb"), "sha256").hexdigest()


def format_word(path):
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    def tag(name):
        return f"{{{ns['w']}}}{name}"
    def child(parent, name):
        existing = parent.find(tag(name))
        return existing if existing is not None else etree.SubElement(parent, tag(name))
    temporary = path.with_suffix(".formatting.docx")
    if temporary.exists():
        raise FileExistsError(temporary)
    with ZipFile(path) as original, ZipFile(temporary, "w") as output:
        for entry in original.infolist():
            contents = original.read(entry.filename)
            if entry.filename in {"word/document.xml", "word/styles.xml"}:
                root = etree.fromstring(contents)
                if entry.filename == "word/styles.xml":
                    for style in root.findall("w:style", ns):
                        identity = style.get(tag("styleId"), "")
                        if identity in {"Normal", "BodyText", "Title", "Heading1", "Heading2"}:
                            properties = child(style, "rPr")
                            fonts = child(properties, "rFonts")
                            for kind in ("ascii", "hAnsi", "eastAsia"):
                                fonts.set(tag(kind), "Noto Sans CJK SC" if identity != "Normal" else "Noto Serif CJK SC")
                            sizes = {"Title": 34, "Heading1": 27, "Heading2": 23}
                            child(properties, "sz").set(tag("val"), str(sizes.get(identity, 20)))
                else:
                    # Some LibreOffice versions mis-import OMML delimiter endChr.
                    # Explicit editable math runs preserve the intended brackets.
                    math_ns = "http://schemas.openxmlformats.org/officeDocument/2006/math"
                    def math_tag(name):
                        return f"{{{math_ns}}}{name}"
                    for delimiter in list(root.iter(math_tag("d"))):
                        properties = delimiter.find(math_tag("dPr"))
                        begin = properties.find(math_tag("begChr")) if properties is not None else None
                        end = properties.find(math_tag("endChr")) if properties is not None else None
                        begin_char = begin.get(math_tag("val"), "(") if begin is not None else "("
                        end_char = end.get(math_tag("val"), ")") if end is not None else ")"
                        expressions = delimiter.findall(math_tag("e"))
                        assert len(expressions) == 1
                        def math_run(value):
                            run = etree.Element(math_tag("r"))
                            rpr = etree.SubElement(run, math_tag("rPr"))
                            etree.SubElement(rpr, math_tag("sty")).set(math_tag("val"), "p")
                            etree.SubElement(run, math_tag("t")).text = value
                            return run
                        replacements = ([math_run(begin_char)] if begin_char else [])
                        replacements += list(expressions[0])
                        replacements += [math_run(end_char)] if end_char else []
                        parent = delimiter.getparent()
                        position = parent.index(delimiter)
                        parent.remove(delimiter)
                        for offset, replacement in enumerate(replacements):
                            parent.insert(position + offset, replacement)
                    for section in root.findall(".//w:sectPr", ns):
                        page = child(section, "pgSz")
                        page.set(tag("w"), "11906")
                        page.set(tag("h"), "16838")
                        margin = child(section, "pgMar")
                        for side in ("top", "bottom", "left", "right"):
                            margin.set(tag(side), "850")
                    for tbl in root.findall(".//w:tbl", ns):
                        properties = child(tbl, "tblPr")
                        width = child(properties, "tblW")
                        width.set(tag("type"), "dxa")
                        width.set(tag("w"), "10206")
                        margins = child(properties, "tblCellMar")
                        for side in ("left", "right"):
                            margin = child(margins, side)
                            margin.set(tag("type"), "dxa")
                            margin.set(tag("w"), "65")
                        columns = tbl.findall("w:tblGrid/w:gridCol", ns)
                        old_widths = [int(column.get(tag("w"))) for column in columns]
                        total = sum(old_widths)
                        widths = [round(10206 * value / total) for value in old_widths]
                        widths[-1] += 10206 - sum(widths)
                        for column, value in zip(columns, widths):
                            column.set(tag("w"), str(value))
                        borders = child(properties, "tblBorders")
                        for kind in ("top", "left", "bottom", "right", "insideH", "insideV"):
                            border = child(borders, kind)
                            border.set(tag("val"), "single")
                            border.set(tag("sz"), "4")
                            border.set(tag("color"), "B8C5D1")
                        for row_index, row in enumerate(tbl.findall("w:tr", ns)):
                            if row_index == 0:
                                child(child(row, "trPr"), "tblHeader")
                            child(child(row, "trPr"), "cantSplit")
                            for column_index, cell in enumerate(row.findall("w:tc", ns)):
                                cell_properties = child(cell, "tcPr")
                                cell_width = child(cell_properties, "tcW")
                                cell_width.set(tag("type"), "dxa")
                                cell_width.set(tag("w"), str(widths[column_index]))
                                text = "".join(cell.xpath(".//w:t/text()", namespaces=ns))
                                scientific = bool(re.fullmatch(r"[-+]?\d\.\d+e[-+]\d+", text))
                                if scientific:
                                    child(cell_properties, "noWrap")
                                if row_index == 0:
                                    child(child(cell, "tcPr"), "shd").set(tag("fill"), "E8F0F5")
                                for run in cell.findall(".//w:r", ns):
                                    rpr = child(run, "rPr")
                                    child(rpr, "sz").set(tag("val"), "16")
                                    child(rpr, "szCs").set(tag("val"), "16")
                                    fonts = child(rpr, "rFonts")
                                    fonts.set(tag("eastAsia"), "Noto Sans CJK SC")
                                    fonts.set(tag("ascii"), "Liberation Sans")
                                    fonts.set(tag("hAnsi"), "Liberation Sans")
                                    if scientific:
                                        fonts.set(tag("ascii"), "Liberation Mono")
                                        fonts.set(tag("hAnsi"), "Liberation Mono")
                                        child(rpr, "sz").set(tag("val"), "15")
                                        child(rpr, "szCs").set(tag("val"), "15")
                                    if row_index == 0:
                                        child(rpr, "b")
                    # Keep direct run properties in OOXML schema order.
                    run_order = {name: position for position, name in enumerate((
                        "rStyle", "rFonts", "b", "bCs", "i", "iCs", "caps", "smallCaps",
                        "strike", "dstrike", "outline", "shadow", "emboss", "imprint",
                        "noProof", "snapToGrid", "vanish", "webHidden", "color", "spacing",
                        "w", "kern", "position", "sz", "szCs", "highlight", "u", "effect",
                        "bdr", "shd", "fitText", "vertAlign", "rtl", "cs", "em", "lang",
                        "eastAsianLayout", "specVanish", "oMath"))}
                    for rpr in root.findall(".//w:rPr", ns):
                        rpr[:] = sorted(rpr, key=lambda element: run_order.get(etree.QName(element).localname, 99))
                    # Properties must precede run text/row cells (not just have
                    # correctly ordered children), or Word readers ignore them.
                    for parent_name, property_name in (("r", "rPr"), ("tr", "trPr"), ("tc", "tcPr")):
                        for parent in root.findall(f".//w:{parent_name}", ns):
                            properties = parent.find(tag(property_name))
                            if properties is not None:
                                parent.remove(properties)
                                parent.insert(0, properties)
                contents = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)
            output.writestr(entry, contents)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "docs/dev/ncfv_hybrid_flux_20261013")
    parser.add_argument("--binary", type=Path, default=Path("/tmp/ncfv_hybrid_flux_accuracy_probe_20261013"))
    args = parser.parse_args()
    directory = args.results.resolve()
    data = {}
    for level in LEVELS:
        path = directory / ("iv10_np1.json" if level == 10 else f"iv{level}.json")
        data[level] = json.loads(path.read_text())
        assert data[level].get("complete") and set(data[level]["cases"]) == set(CASES), path
        assert data[level]["final_time"] == 1.6 and data[level]["mpi_ranks"] <= 64
    parallel = json.loads((directory / "iv10_np2.json").read_text())
    assert parallel.get("complete")
    geometry_audit = json.loads((directory / "geometry_iv10.json").read_text())
    temporal_log = (directory / "rk3_temporal_np1.log").read_text()
    temporal_orders = [(int(mode), int(method), float(order)) for mode, method, order in
                       re.findall(r"temporal order mode=(\d+) method=(\d+) order=([0-9.]+)", temporal_log)]
    assert len(temporal_orders) == 4 and all(2.8 < order < 3.3 for _, _, order in temporal_orders)
    assert "SUCCESS!" in temporal_log
    ghost_counts = {}
    for level in LEVELS:
        log_path = directory / ("iv10_np1.log" if level == 10 else f"iv{level}.log")
        counts = re.findall(r"retained runtime ghosts=(\d+)", log_path.read_text())
        assert len(counts) == 2, log_path
        ghost_counts[level] = [int(value) for value in counts]
    h = {level: (400.0 / data[level]["nodes"]) ** (1 / 3) for level in LEVELS}
    def eoc(coarse, fine, level0, level1):
        if min(coarse, fine) < 1e-12:
            return None
        return math.log(coarse / fine) / math.log(h[level0] / h[level1])
    def orders(values):
        return [f"{eoc(values[i - 1], values[i], LEVELS[i - 1], LEVELS[i]):.3f}" for i in range(1, 4)]
    def values(case, block="mean_conserved", component=0, norm="L2V"):
        return [data[level]["cases"][case][block][norm][component] for level in LEVELS]
    def history_orders(values):
        return orders([float(x) for x in values])

    # Full machine-readable norm table, including both native point recoveries.
    blocks = ("mean_conserved", "primitive_of_mean", "point_traditional_conserved",
              "point_efficient_conserved", "point_traditional_primitive", "point_efficient_primitive")
    metric_rows = []
    for case in CASES:
        for block in blocks:
            components = data[10]["primitive_components" if "primitive" in block else "conserved_components"]
            for component, label in enumerate(components):
                for norm in ("L1V", "L2V", "Linf"):
                    previous = None
                    for level in LEVELS:
                        error = data[level]["cases"][case][block][norm][component]
                        order = None if previous is None else eoc(previous[1], error, previous[0], level)
                        metric_rows.append({"grid": level, "case": case, "initialization": data[level]["cases"][case]["initialization"],
                            "quantity": block, "component": label, "norm": norm, "error": error,
                            "observed_order": "" if order is None else order, "nodes": data[level]["nodes"],
                            "h": h[level], "dt": data[level]["dt"], "steps": data[level]["steps"],
                            "time": 1.6, "mpi_ranks": data[level]["mpi_ranks"]})
                        previous = (level, error)
    csv_path = directory / "all_weighted_errors.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)

    final = data[80]["cases"]
    tt, ee, et, te = [final[k]["mean_conserved"]["L2V"][0] for k in ("TT", "EE", "ET", "TE")]
    reduction_d = 1 - et / ee
    reduction_c = 1 - te / ee
    gap_recovered = (ee - et) / (ee - tt)
    residual_gap = et / tt - 1
    native_e = final["EE_native_initialization"]["mean_conserved"]["L2V"][0]
    parallel_max = max(abs(data[10]["cases"][case][block][norm][c] - parallel["cases"][case][block][norm][c])
                       for case in CASES for block in blocks for norm in ("L1V", "L2V", "Linf") for c in range(5))
    density_parallel_max = max(abs(data[10]["cases"][case]["mean_conserved"]["L2V"][0] -
                                   parallel["cases"][case]["mean_conserved"]["L2V"][0]) for case in CASES)
    source_paths = [ROOT / "cases/ncfv_euler/3D/diagnostics/hybrid_flux_accuracy_probe.cpp",
                    ROOT / "cases/ncfv_euler/3D/diagnostics/hybrid_flux_geometry_audit.cpp",
                    Path(__file__).resolve(),
                    ROOT / "cases/ncfv_euler/3D/diagnostics/run_hybrid_flux_accuracy.py",
                    ROOT / "src/NCFV/NCFVSpatial.cpp", ROOT / "src/NCFV/NCFVSpatial.hpp",
                    ROOT / "src/NCFV/NCFVReconstruction.cpp", ROOT / "src/NCFV/NCFVDualGeometry.cpp",
                    ROOT / "src/NCFV/NCFVNodeHalo.cpp", ROOT / "src/NCFV/NCFVSolver.cpp",
                    ROOT / "build/src/NCFV/libncfv.a", args.binary]
    provenance = {str(path): digest(path) for path in source_paths}
    provenance["git_HEAD"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    (directory / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    analysis = {"IV80_swap_dissipation_error_reduction_fraction": reduction_d,
                "IV80_swap_average_error_reduction_fraction": reduction_c,
                "IV80_norm_gap_recovered_fraction": gap_recovered,
                "IV80_ET_over_TT_minus_one": residual_gap,
                "IV10_np1_np2_density_L2_max_difference": density_parallel_max,
                "IV10_np1_np2_all_norms_max_difference": parallel_max,
                "scope": "LS13, IV10/20/40/80, native Roe entropy fix, t=1.6, joint h/dt refinement"}
    (directory / "analysis.json").write_text(json.dumps(analysis, indent=2, ensure_ascii=False) + "\n")

    parts = ["""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>NCFV 高效平均通量与 Roe 耗散交叉验证报告</title>
<style>body{font-family:'Noto Serif CJK SC';font-size:10pt;line-height:1.4}
h1{font-size:18pt}h2{font-size:14pt}h3{font-size:11pt}table{border-collapse:collapse;width:100%;font-size:8pt}
th,td{border:1px solid #b8c5d1;padding:4px}th{background:#e8f0f5}pre{font-size:8pt;white-space:pre-wrap}</style></head><body>"""]
    parts += ["<h1>NCFV 高效平均通量与 Roe 耗散交叉验证报告</h1>",
              paragraph("三棱柱 IV10/20/40/80 · LS13 · 固定全局步长 · t=1.6 · 2026-10-13"),
              "<h2>1. 结论先行</h2>"]
    findings = [
        f"本轮控制变量测试表明，高效原生耗散项的构造是高效/传统总误差差异的主要来源。IV80 上，高效平均＋传统耗散的密度均值 L2 为 {number(et)}，相比完整高效 {number(ee)} 降低 {100 * reduction_d:.2f}%；传统平均＋高效耗散为 {number(te)}，仅降低 {100 * reduction_c:.2f}%。",
        f"以完整高效与完整传统的 L2 范数差为参照，更换耗散消除了 {100 * gap_recovered:.2f}% 的范数差；交叉 ET 仍比完整传统 {number(tt)} 高 {100 * residual_gap:.2f}%。这是一项终时刻控制实验，不是将非线性演化中的误差逐项线性分摊。",
        f"统一初值下，完整高效 IV40→IV80 的密度均值观测阶为 {orders(values('EE'))[-1]}；高效平均＋传统耗散为 {orders(values('ET'))[-1]}；完整传统为 {orders(values('TT'))[-1]}。改善耗散既降低误差常数，也改善这一网格段的观测斜率，但不据此宣称所有网格、所有问题都达到严格三阶。",
        f"两种方案使用同一个标准 Roe 分派器和原生熵修正。本实验定位的是“高效耗散的面状态／法向／面积合并与积分构造”，不是证明 Roe 特征分解公式或熵修正写错。原生左右恢复方式也保留在各自通量分量内部。",
        f"IV10 串行与 2 进程的密度均值 L2 最大差为 {number(density_parallel_max)}；测试推进与生产传统 SSPRK3 的终场最大差为零。本轮没有证据表明这部分额外误差来自 ghost 不足或时间推进实现不一致。",
    ]
    parts.append("<ol>" + "".join(f"<li>{html.escape(x)}</li>" for x in findings) + "</ol>")

    parts += ["<h2>2. 计算设置和误差口径</h2>", paragraph(
        "区域 [0,10]×[0,10]×[0,4]，三方向周期合并；背景速度 (1,1,0)、涡强 5、初始涡心 (5,5)、γ=1.4。无黏 Euler，二次 LS 正规方程重构，实际模板均为 13 点；所有者节点次序及全局模板 ID 在两套原生算子间逐个核对一致。限制器关闭；没有采用边中点诊断通量、参考文档通量或 0.15a 参考熵修正。"),
        table(["网格", "唯一节点", "原始三棱柱单元", "MPI", "全局 Δt", "步数", "终时间"],
              [[g, data[g]["nodes"], data[g]["cells"], data[g]["mpi_ranks"], data[g]["dt"], data[g]["steps"], 1.6] for g in LEVELS]),
        paragraph("时间推进逐字采用生产代码的三阶段 SSPRK3 组合；固定 dt，不使用全局 CFL 自适应或局部时间步。40 与 80 网格曾并行运行，合计 48 个单线程 MPI 进程；单作业最大 32 进程，均不超过 64 核。计时仅为诊断运行记录，不是算法效率基准。"),
        paragraph("四组主对照 TT/EE/ET/TE 使用相同的传统原生 quadratureOrder=4 体积求积初值。不启用此前取消的五阶初始化。第五组 EE_native_initialization 使用原始高效的解析点值＋解析梯度微几何均值恢复，与历史高效结果对照。共享控制体与原始网格，每套算子分别保留自己依赖驱动的完整运行时 node halo。"),
        paragraph("均值主指标比较数值守恒均值与高阶解析参考均值。解析参考仅用于误差评价：每个微四面体 order=6 的 24 点规则，并非修改求解器的初始化或通量积分。所有 L1/L2 均含节点对偶控制体体积权重，只累加 owned 节点，ghost 不重复计数。L∞ 为最大绝对误差。"),
        paragraph("E1,V = Σj[Vj·|ej|] / ΣjVj；E2,V = √{Σj[Vj·ej²] / ΣjVj}；E∞ = maxj|ej|。其中 ej 为数值量减相应解析参考量，Vj 为节点对偶控制体积。"),
        r"<p>$$h=(400/N)^{1/3},\qquad p=\frac{\log(E_{\mathrm{coarse}}/E_{\mathrm{fine}})}{\log(h_{\mathrm{coarse}}/h_{\mathrm{fine}})}.$$</p>",
        paragraph("N 是周期合并后的唯一节点数。由于网格和时间步同步加密，报告的阶数是时空联合观测阶，而非独立空间阶。primitive_of_mean 表示 W(Ū) 对 W(Ūexact) 的误差，不冒充原始变量真正的控制体均值。恢复点值则与解析节点值比较。")]

    parts += ["<h2>3. 原生通量拆分与严格交叉组装</h2>",
        paragraph("记平均通量为 C，数值通量相对于平均物理通量的 Roe 修正为 D。这里 D 自带 Roe 通量中的负号，不额外改号。传统方案在每个对偶微三角面原生求积点重构左右状态，使用该微面的局部法向。surfaceQuadratureOrder=3 对每个微三角面使用 3 点，对二次多项式精确；一张宏面包含若干微面，绝不是每张宏面只有 3 点。"),
        r"<p>$$C_T=\sum_q w_q\frac{F(U^T_{L,q},n_q)+F(U^T_{R,q},n_q)}{2}.$$</p>",
        r"<p>$$D_T=\sum_q w_q\left[\widehat F_{\mathrm{Roe}}(U^T_{L,q},U^T_{R,q},n_q)-\frac{F(U^T_{L,q},n_q)+F(U^T_{R,q},n_q)}{2}\right].$$</p>",
        paragraph("高效平均通量严格调用左右原生 EfficientIntegratedPhysicalFlux：节点零阶物理通量加预计算权重作用于物理通量一阶导数。高效耗散严格使用 EfficientSurfaceMean 所得左右面均值，以及面积向量之和的单位方向；A 表示所有微面标量面积之和。"),
        r"<p>$$C_E=\frac{I^E_L(F)+I^E_R(F)}{2}.$$</p>",
        r"<p>$$D_E=A\left[\widehat F_{\mathrm{Roe}}(\overline U^E_L,\overline U^E_R,n_S)-\frac{F(\overline U^E_L,n_S)+F(\overline U^E_R,n_S)}{2}\right].$$</p>",
        table(["编号", "平均部分", "耗散部分", "意义"], [[k, "高效" if k[0] == "E" else "传统", "高效" if k[1] == "E" else "传统", LABELS[k]] for k in CASES[:4]]),
        paragraph("诊断不自行重写以上公式。SpatialOperator::SetInteriorInviscidCorrectionScalesForDiagnostics(0) 提取该原生算子的 C，系数 1 得到完整通量；同一状态的完整 RHS 减中心 RHS 得到 D 的原生守恒散度。然后按 L_ET=L_E,central+L_T,full−L_T,central 和 L_TE=L_T,central+L_E,full−L_E,central 组装。无黏、全周期且边界控制面为空，因此不存在额外边界通量混入 D 的问题。线性散度保证该操作等价于逐条边先交叉组装再除体积。"),
        paragraph("传统恢复保留全二次系数与零均值基函数，高效恢复保留分布式一阶导数恢复模板；没有为了交叉测试强迫两者使用相同的点值，也没有以高效面均值冒充传统求积点左右迹。共用状态是同一 RK 阶段的控制体守恒均值，不是交换两个已经完成演化的解。两种 Roe 调用均走 Gas.hpp 的 eigScheme=0 原生 Harten–Yee 熵修正，fixScale=1。")]

    parts += ["<h2>4. 四组交叉实验的密度结果</h2>",
        "<h3>4.1 密度均值体积加权 L2 与观测阶</h3>",
        table(["组装／初值", "IV10", "IV20", "IV40", "IV80", "10→20", "20→40", "40→80"],
              [[LABELS[k]] + [number(x) for x in values(k)] + orders(values(k)) for k in CASES]),
        paragraph("前四行共用传统原生初值，第五行仅供原始高效方案与历史记录对照。表中高效平均＋传统耗散（ET）和传统平均＋高效耗散（TE）是用户要求新增的两种组装。"),
        "<h3>4.2 各网格的替换效果</h3>"]
    rows = []
    for g in LEVELS:
        f = data[g]["cases"]
        a, b, c, d = [f[k]["mean_conserved"]["L2V"][0] for k in ("TT", "EE", "ET", "TE")]
        rows.append([g, f"{100*(1-c/b):.2f}%", f"{100*(1-d/b):.2f}%", f"{100*(c/a-1):.2f}%", f"{100*(b-c)/(b-a):.2f}%"])
    parts += [table(["网格", "换传统耗散：EE→ET 降幅", "换传统平均：EE→TE 降幅", "ET 比 TT 高", "消除 EE−TT 范数差"], rows),
              paragraph("最后一列是范数差的控制实验指标，不是耗散占总误差的严格百分比。不同方案的解轨迹不同，平均项与耗散项之间也会相互作用。"),
              "<h3>4.3 密度均值完整三种范数</h3>",
              table(["网格", "组装", "L1,V", "L2,V", "L∞"],
                    [[g, k] + [number(data[g]["cases"][k]["mean_conserved"][n][0]) for n in ("L1V", "L2V", "Linf")] for g in LEVELS for k in CASES])]

    parts += ["<h2>5. 守恒变量和原始变量结果</h2>",
        paragraph("以下均为体积加权 L2。均值是本轮主判断口径。恢复点值同时给出传统恢复与高效恢复，避免将输出恢复差异误判为通量组装误差。解析解的 ρw、w 恒为零，其非零误差是微弱的数值横向扰动，不是非零解析分量的相对误差，也不应当作为格式整体阶数的主指标。各变量 L1、L2、L∞ 及逐级阶数完整保存在同目录 all_weighted_errors.csv（1800 条记录）；舍入量级的值不计算阶数。")]
    block_labels = {"mean_conserved": "5.1 控制体守恒均值",
                    "primitive_of_mean": "5.2 守恒均值转换的原始变量",
                    "point_traditional_conserved": "5.3 传统恢复的节点守恒变量",
                    "point_efficient_conserved": "5.4 高效恢复的节点守恒变量",
                    "point_traditional_primitive": "5.5 传统恢复的节点原始变量",
                    "point_efficient_primitive": "5.6 高效恢复的节点原始变量"}
    for block in blocks:
        components = data[10]["primitive_components" if "primitive" in block else "conserved_components"]
        parts += [f"<h3>{block_labels[block]}</h3>",
                  table(["网格", "组装"] + components,
                        [[g, k] + [number(x) for x in data[g]["cases"][k][block]["L2V"]] for g in LEVELS for k in CASES[:4]])]

    parts += ["<h2>6. 初始化、拆分与并行一致性检查</h2>",
        table(["网格", "传统初值密度 L2", "高效初值密度 L2", "两初值差 L2", "统一初值高效终误差", "原始初值高效终误差"],
              [[g, number(data[g]["traditional_initial_error"]["L2V"][0]),
                number(data[g]["efficient_initial_error"]["L2V"][0]), number(data[g]["initialization_difference"]["L2V"][0]),
                number(data[g]["cases"]["EE"]["mean_conserved"]["L2V"][0]),
                number(data[g]["cases"]["EE_native_initialization"]["mean_conserved"]["L2V"][0])] for g in LEVELS]),
        paragraph(f"IV80 初值选择引起的高效终误差范数变化约 {100*abs(ee-native_e)/native_e:.3f}%，显著小于更换耗散带来的 {100*reduction_d:.2f}% 改善。统一初值因此不是本轮耗散定位结论的主要驱动因素。"),
        table(["网格", "两套体积最大差", "传统边通量／RHS拆分不符", "高效边通量／RHS拆分不符", "运行时 ghost：传统／高效"],
              [[g, number(data[g]["checks"]["max_dual_volume_difference"]),
                number(data[g]["checks"]["traditional_part_divergence_identity_max"]),
                number(data[g]["checks"]["efficient_part_divergence_identity_max"]),
                f"{ghost_counts[g][0]} / {ghost_counts[g][1]}"] for g in LEVELS]),
        paragraph("ghost 数为全部 MPI ranks 的运行时 ghost 总和，包含不同 rank 上的重复通信副本；不是独立全局节点数。串行 IV10 无 MPI ghost，而两种 MPI 分区的数值误差相同。"),
        paragraph(f"IV10 np=1 与 np=2：密度均值 L2 最大差 {number(density_parallel_max)}；所有输出变量、两种点恢复、三种范数的最大差 {number(parallel_max)}。两种并行分区的传统测试终场与生产求解器的 SSPRK3 终场最大差均为零。完整 node halo 由各自重构和积分依赖收集并裁剪，非简单给定固定 ghost 层数。"),
        paragraph("补充通过现有 ncfv_test_parallel 的方向归一化、ghost 交换和几何缓存释放测试（np=2）：LS、SVDLS、变分重构分别在传统／高效模式再现二次多项式点值和导数；两 rank 分别有 2400、3240 个断言通过。该测试使用拉伸网格及 16 点模板，是重构前提的补充检查，不替代本轮 LS13 的串并行终场检查。"),
        paragraph("现有 Roe＋SSPRK3 时间自收敛测试也通过（np=1，12036 个断言）：固定网格，仅将 dt 从 0.05 依次减半到 0.025、0.0125，推进到 t=0.2，用相邻解差比较时间阶。测试涵盖两种积分模式与 LS／变分重构，结果如下；这不是本轮 LS13 混合格式的独立时间收敛试验。"),
        table(["积分模式", "重构", "时间自收敛阶"],
              [["传统" if mode == 1 else "高效", "LS" if method == 1 else "变分", f"{order:.6f}"]
               for mode, method, order in temporal_orders]),
        table(["网格", "TT质量漂移", "EE质量漂移", "ET质量漂移", "TE质量漂移", "原始EE质量漂移"],
              [[g] + [number(data[g]["cases"][k]["mass_drift"]) for k in CASES] for g in LEVELS]),
        paragraph("质量漂移比较同一计算的终质量与初质量，而不是比较终质量和解析参考质量；后者包含初始积分误差及参考误差。")]

    parts += ["<h2>7. 此前传统／高效结果归档</h2>",
        paragraph("本节整理已有记录，不把不同模板、重构、初始化或版本的数据混入本轮四组控制矩阵。除 7.8 节单独列出的减半时间步试验外，归档结果均在同样的四级三棱柱网格、dt=0.2/0.1/0.05/0.025、t=1.6 下得到。若历史记录只含节点误差，明确标为恢复点值，不能作为均值误差使用。"),
        "<h3>7.1 LS 模板 13–16 点：解析均值口径</h3>"]
    archive_mean = read_csv(ROOT / "docs/reports/ncfv_exact_mean_accuracy_20260921/density_pressure_summary.csv")
    rows = []
    for mode in ("traditional", "efficient"):
        for stencil in (13, 14, 15, 16):
            records = sorted((r for r in archive_mean if r["method"] == mode and int(r["stencil"]) == stencil), key=lambda r: int(r["grid"]))
            assert [int(r["grid"]) for r in records] == list(LEVELS)
            v = [float(r["rho_mean_L2"]) for r in records]
            rows.append([("传统" if mode == "traditional" else "高效") + f" LS{stencil}"] + [number(x) for x in v] + history_orders(v))
    parts += [table(["历史方案", "IV10", "IV20", "IV40", "IV80", "10→20", "20→40", "40→80"], rows),
        paragraph("来源：20260921 density_pressure_summary.csv。历史均值口径、二进制版本和分区与本轮可能有微小差别，例如 IV20 传统 LS13 历史值 5.9358263e-3，本轮 5.9353748e-3；本轮 IV40 原始高效 LS13 为 1.7808673e-3，20260924 为 1.7808669e-3。它们接近，但不能宣称所有历史值都逐位复现；本轮因果判断只采用同时复算、同一口径的四组数据。"),
        "<h3>7.2 历史 LS13–16 的压力均值函数误差</h3>"]
    rows = []
    for mode in ("traditional", "efficient"):
        for stencil in (13, 14, 15, 16):
            records = sorted((r for r in archive_mean if r["method"] == mode and int(r["stencil"]) == stencil), key=lambda r: int(r["grid"]))
            v = [float(r["pressure_of_mean_L2"]) for r in records]
            rows.append([("传统" if mode == "traditional" else "高效") + f" LS{stencil}"] + [number(x) for x in v] + history_orders(v))
    parts += [table(["历史方案", "IV10", "IV20", "IV40", "IV80", "10→20", "20→40", "40→80"], rows),
        "<h3>7.3 历史 LS13–16 的原生恢复密度／压力点值误差</h3>"]
    rows = []
    for mode in ("traditional", "efficient"):
        for stencil in (13, 14, 15, 16):
            records = sorted((r for r in archive_mean if r["method"] == mode and int(r["stencil"]) == stencil), key=lambda r: int(r["grid"]))
            for key, label in (("rho_point_L2", "ρ点值"), ("pressure_point_L2", "p点值")):
                v = [float(r[key]) for r in records]
                rows.append([("传统" if mode == "traditional" else "高效") + f" LS{stencil} {label}"] + [number(x) for x in v] + history_orders(v))
    parts += [table(["历史方案", "IV10", "IV20", "IV40", "IV80", "10→20", "20→40", "40→80"], rows),
        "<h3>7.4 LS/SVDLS：16 点原生恢复点值口径</h3>"]
    archive = read_csv(ROOT / "docs/reports/ncfv_ls_svdls_t1p6_dt_20260924/metrics.csv")
    rows = []
    for mode in ("traditional", "efficient"):
        for method in ("ls", "svdls"):
            records = sorted((r for r in archive if r["mesh_type"] == "prism" and r["mode"] == mode and r["method"] == method), key=lambda r: int(r["level"]))
            v = [float(r["rho_point_L2V_final"]) for r in records]
            assert len(v) == 4
            rows.append([("传统" if mode == "traditional" else "高效") + " " + method.upper() + "16"] + [number(x) for x in v] + history_orders(v))
    parts += [table(["历史方案", "IV10", "IV20", "IV40", "IV80", "10→20", "20→40", "40→80"], rows),
        paragraph("来源：20260924 LS/SVDLS metrics.csv；重构使用约 16 点模板，非本轮 LS13。LS 与 SVDLS 在相同模式下几乎一致，说明更换线性求解器本身不足以消除这里的高效通量差异。"),
        "<h3>7.5 变分重构：原生恢复点值口径</h3>"]
    rows = []
    for label, path in (("传统变分 w5/3次", "ncfv_traditional_variational_w5_t1p6_dt_20260924"),
                        ("高效变分一阶导数", "ncfv_efficient_variational_gradient_t1p6_dt_20260928")):
        records = sorted((r for r in read_csv(ROOT / f"docs/reports/{path}/metrics.csv") if r["mesh_type"] == "prism"), key=lambda r: int(r["level"]))
        v = [float(r["rho_point_L2V_final"]) for r in records]
        rows.append([label] + [number(x) for x in v] + history_orders(v))
    parts += [table(["历史方案", "IV10", "IV20", "IV40", "IV80", "10→20", "20→40", "40→80"], rows),
        paragraph("变分数据来自当前一次导数高效版本对应的两份归档。传统变分在有限网格段出现大于三阶的斜率，不等于形式精度为四阶。本轮交叉实验仅使用 LS13，未将其因果结论直接量化推广到变分方案。"),
        "<h3>7.6 此前 LS13 高效的完整／去耗散对照：均值口径</h3>"]
    zero_d = [0.026144867466240892, 0.006430618734830741, 0.00042979074513310435, 0.00003408118059317795]
    old_e = [0.027401225563209543, 0.008706520407154882, 0.0017808669495476006, 0.00026992002904962354]
    parts += [table(["历史方案", "IV10", "IV20", "IV40", "IV80", "10→20", "20→40", "40→80"],
                    [["原生高效 LS13＋Roe"] + [number(x) for x in old_e] + orders(old_e),
                     ["原生高效 LS13 无耗散"] + [number(x) for x in zero_d] + orders(zero_d)]),
        paragraph("来源：docs/dev/ncfv_ls13_accuracy_synthesis_20260924.md 及原始消融记录。去耗散 IV80 改善 87.4%，但中心通量缺少间断稳定性；本轮 ET 则保留传统 Roe 耗散，属于另一种有耗散对照，不能与去耗散结果混称。此前周期拼接审计中，完整高效 IV80 节点密度误差平方积分有 98.90% 位于 r<3 涡核，拼接带份额仅 0.0051%；该历史区域分布支持拼接非主要密度误差来源，但本轮未重复区域消融。")]

    parts.append("<h3>7.7 此前 LS13 两种算法的全部变量 L2</h3>")
    archive_variables = read_csv(ROOT / "docs/reports/ncfv_conservative_primitive_accuracy_20260922/results/all_variable_errors.csv")
    for kind, variables in (("conservative", ("rho", "rhou", "rhov", "rhow", "rhoE")),
                            ("primitive_of_mean", ("rho", "u", "v", "w", "p"))):
        selected = [r for r in archive_variables if int(r["stencil"]) == 13 and r["variable_type"] == kind]
        lookup = {(r["method"], int(r["grid"]), r["variable"]): float(r["L2"]) for r in selected}
        rows = [[g, "传统" if mode == "traditional" else "高效"] +
                [number(lookup[mode, g, v]) for v in variables] for g in LEVELS for mode in ("traditional", "efficient")]
        parts += [paragraph("历史控制体守恒均值" if kind == "conservative" else "历史原始变量：由守恒均值转换"),
                  table(["网格", "历史方案"] + list(variables), rows)]
    parts.append(paragraph("来源：20260922 all_variable_errors.csv。该历史报告已经明确采用 W(Ū) 对 W(Ūexact)，而非原始变量的真正体均值，变量定义与本轮一致；历史数值解及 MPI 分区可能不同，仍单列归档。完整历史 CSV 保留在原目录，未覆盖。"))

    parts.append("<h3>7.8 此前高效 LS16 减半时间步：原生恢复点值口径</h3>")
    time_control = sorted((r for r in read_csv(ROOT / "docs/reports/ncfv_efficient_ls_t1p6_dt_0p1_0p0125_20260924/comparison_old_new.csv")
                           if r["mesh_type"] == "prism"), key=lambda r: int(r["level"]))
    parts += [table(["网格", "原 Δt", "减半 Δt", "原密度点值 L2,V", "减半后 L2,V", "相对变化"],
                    [[r["level"], r["old_dt"], r["new_dt"], number(r["old_rho_point_L2V"]),
                      number(r["new_rho_point_L2V"]), f'{float(r["relative_change_percent"]):.3f}%'] for r in time_control]),
              paragraph("来源：20260924 时间加密复测。仅针对原生高效 LS16，减半 dt 后终时刻误差改变约 0.25%–0.40%，不能说时间误差严格为零，也不能当作本轮 LS13 两种混合格式已经单独完成的时间消融。")]

    parts += ["<h2>8. 为什么原生高效耗散产生额外误差</h2>",
        paragraph("由代码结构和本轮替换结果可以作如下推断：逐点 Roe 修正积分与“先平均左右状态、先合并法向、再求一次 Roe”不交换。高效平均通量对物理通量加入了导数积分修正，但原生高效 Roe 修正没有同样的面内变化积分处理。"),
        r"<p>$$\int_S D(U_L(x),U_R(x),n(x))\,dS\ne A D(\overline U_L,\overline U_R,n_S).$$</p>",
        paragraph("差异同时涉及：Roe 平均状态与特征矩阵的非线性、绝对特征速度与熵修正的分段变化；局部法向与宏面平均法向的区别；传统全二次迹与高效梯度恢复所产生的左右跳跃差异。非平面宏面的标量面积总和不等于面积向量模长。不能把这些因素简单归结为“面上一个 Gauss 点低阶”，因为高效路线没有传统 Gauss 面积分，也不能仅加大 surfaceQuadratureOrder 就直接改善它。"),
        "<h3>8.1 同一初场的原生耗散修正范数</h3>",
        table(["网格", "传统密度耗散 RHS 的 L2,V", "高效密度耗散 RHS 的 L2,V", "高效／传统"],
              [[g, number(data[g]["checks"]["traditional_initial_dissipation_rhs"]["L2V"][0]),
                number(data[g]["checks"]["efficient_initial_dissipation_rhs"]["L2V"][0]),
                f'{data[g]["checks"]["efficient_initial_dissipation_rhs"]["L2V"][0] / data[g]["checks"]["traditional_initial_dissipation_rhs"]["L2V"][0]:.4f}'] for g in LEVELS]),
        paragraph("该表是对同一传统原生初值计算的修正项范数，不是将最终误差线性分摊，也不能将平均通量 RHS 的原始范数当成平均通量截断误差。"),
        "<h3>8.2 固定纯接触跳跃的几何单独检查（IV10）</h3>",
        table(["检查项", "实测值"], [
            ["宏面数量", geometry_audit["macro_edges"]],
            ["非平面宏面数量（A/|面积向量|>1+1e−10）", geometry_audit["nonplanar_macro_edges"]],
            ["A/|面积向量| 的最大值", f'{geometry_audit["area_over_vector_measure_max"]:.8f}'],
            ["全部宏面上的原生三角面求积点数量范围", f'{geometry_audit["quadrature_points_per_macro_min"]}–{geometry_audit["quadrature_points_per_macro_max"]}'],
            ["宏面比逐点耗散更大的边数／更小的边数", f'{geometry_audit["macro_overdissipative_edges"]} / {geometry_audit["macro_underdissipative_edges"]}'],
            ["全部边绝对密度耗散之和：宏面／逐点", f'{geometry_audit["macro_over_traditional_dissipation_ratio"]:.8f}'],
            ["原生 Roe 纯接触恒等式最大不符", number(geometry_audit["pure_contact_identity_max_error"])]]),
        paragraph("此检查固定左右密度 0.9/1.1、压力 1、速度 (1,1,0)，直接调用原生 Roe，排除了迹随位置变化。纯接触密度修正分别等于 −0.1∫|u·nq|dS 与 −0.1A|u·nS|；本例局部已不等价，但全部边的绝对耗散总和接近一致。因此不能凭非平面几何或面积因子单独解释终时刻约四成的改善，也不能未经进一步验证就把 A 换为面积向量模长。它不是一个可全局推进的精确算例，只是隔离几何的单面诊断。"),
        paragraph("本轮已分开 C 和 D 两大模块，并单独检查了固定迹下的几何差异；尚未定量分离真实涡场 D 内部的“左右迹恢复”“宏法向合并”“Roe 特征变化”贡献。进一步应固定同一组真实左右迹，对比逐微面局部法向、宏法向以及同一 Roe 矩阵的耗散积分；所有实验都须保持熵修正和唯一边通量守恒装配。"),
        paragraph("二次 k-exact 重构与三阶 RK 本身不保证任意几何上的整体三阶。此次结果显示耗散构造提高了误差常数并改变有限网格的联合观测阶，不等于已经证明一项普遍的二阶截断误差。周期最小镜像等熵涡也不是严格光滑周期解，最终三阶验证仍需严格周期三维熵波／涡量波与独立时间细化。"),
        "<h2>9. 后续改进建议及复现</h2>",
        "<ol><li>优先把 ET 作为准确性参考：保留高效物理平均通量，耗散按传统左右迹、局部法向和原生三点微面求积组装。该建议来自本轮实测，不表示已把混合方案写入生产配置。</li>"
        "<li>若要保持高效成本，应研究逐微面少点耗散或面内变化修正；必须同时处理 Roe 矩阵、左右跳跃和法向，而不是仅提高平均通量精度。</li>"
        "<li>对改进耗散补做光滑三维周期波、接触间断、激波及正性测试；保留原生熵修正。不要把直接去掉 Roe 耗散作为通用生产修复。</li></ol>",
        paragraph("本轮仅新增独立诊断源码和报告生成脚本，未修改生产通量、重构、初始化、边界或时间推进代码；保留任务开始时已有的所有未提交修改。本报告测试范围为用户确认的 IV10/20/40/80，不含 160/320。"),
        "<pre>cmake --build build -t ncfv_euler3D -j 8\n"
        "venv/bin/python cases/ncfv_euler/3D/diagnostics/compile_reconstruction_probe.py \\\n"
        "  --source cases/ncfv_euler/3D/diagnostics/hybrid_flux_accuracy_probe.cpp \\\n"
        "  --output /tmp/ncfv_hybrid_flux_accuracy_probe_new\n"
        "cd build\nHWLOC_COMPONENTS=-gl,-cuda,-nvml OMP_NUM_THREADS=1 \\\n"
        "  mpirun --bind-to none -np 32 /tmp/ncfv_hybrid_flux_accuracy_probe_new \\\n"
        "  ../cases/ncfv_euler/3D/NCFV_iv80.json \\\n"
        "  ../cases/ncfv_euler/3D/iv80_3d_1_dnds.cgns \\\n"
        "  /tmp/iv80_hybrid_new.json 0.025 1.6</pre>",
        paragraph("复现应使用新的结果路径。也可以使用 run_hybrid_flux_accuracy.py --binary /tmp/ncfv_hybrid_flux_accuracy_probe_new --output /tmp/ncfv_hybrid_new --parallel-fine-grids 自动完成所有网格及串并行检查（合计最多 48 ranks）。原始运行日志、各级 JSON、all_weighted_errors.csv、分析比例和 source/binary SHA256 位于 docs/dev/ncfv_hybrid_flux_20261013/。同目录 source_snapshot.tar.gz 保存相关实现与诊断源码；库摘要与 git HEAD 不能替代源码快照，因为工作树已有未提交修改。"),
        paragraph("通量实现定位：src/NCFV/NCFVSpatial.cpp 的 EfficientIntegratedPhysicalFlux、EfficientSurfaceMean、EvaluateOwnedEdgeFlux 和 EvaluateRHS；点值恢复位于 src/NCFV/NCFVReconstruction.cpp 的 RecoverPointValues；生产初值与 RK3 位于 src/NCFV/NCFVSolver.cpp。"),
        paragraph("几何检查复现需单独以 compile_reconstruction_probe.py 编译 hybrid_flux_geometry_audit.cpp，再在 build 目录串行运行：geometry-audit.exe ../cases/ncfv_euler/3D/NCFV_iv10.json /tmp/geometry_iv10_new.json。补充单元测试为 ncfv_test_parallel 的 directional normalization 与 SSPRK3 过滤项；它们的完整通过日志均随本轮结果保存。"),
        "</body></html>"]
    source = directory / "report.html"
    source.write_text("\n".join(parts), encoding="utf-8")
    destination = ROOT / "docs/dev/NCFV_交叉通量误差报告_20261013.docx"
    subprocess.run(["pandoc", str(source), "--from=html+tex_math_dollars", "--to=docx", "--standalone", "-o", str(destination)], check=True)
    format_word(destination)
    with ZipFile(destination) as file:
        root = etree.fromstring(file.read("word/document.xml"))
        ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
              "m": "http://schemas.openxmlformats.org/officeDocument/2006/math"}
        report_text = "".join(root.xpath("//w:t/text()", namespaces=ns))
        assert "传统平均＋高效耗散" in report_text and "高效平均＋传统耗散" in report_text
        assert len(root.xpath("//m:oMath", namespaces=ns)) >= 6, "Missing editable formulas"
        assert len(root.xpath("//w:tbl", namespaces=ns)) >= 15, "Missing result tables"
    print(json.dumps({"word": str(destination), "weighted_error_rows": len(metric_rows), **analysis}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
