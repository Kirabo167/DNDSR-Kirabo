"""Build the archived NCFV weighted-error synthesis as an HTML Word source.

Run from the repository root, then convert the generated HTML with LibreOffice.
The generated report only reads existing result CSVs; it does not run a solver.
"""

from __future__ import annotations

import csv
import html
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "docs/reports"
NODES = {10: 512, 20: 3896, 40: 30416, 80: 239968}
H = {grid: (400.0 / count) ** (1.0 / 3.0) for grid, count in NODES.items()}


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def number(value: object, digits: int = 6) -> str:
    if value is None or value == "":
        return "—"
    return f"{float(value):.{digits}g}"


def order(coarse: float, fine: float, coarse_grid: int, fine_grid: int) -> float:
    return math.log(coarse / fine) / math.log(H[coarse_grid] / H[fine_grid])


def table(headers: list[str], data: list[list[object]], css_class: str = "") -> str:
    head = "".join(f"<th>{html.escape(str(item))}</th>" for item in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(cell))}</td>" for cell in row) + "</tr>"
        for row in data
    )
    return f'<table class="{css_class}"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


def paragraph(text: str) -> str:
    return f"<p>{text}</p>"


def build() -> str:
    all_variables = rows(
        SOURCE
        / "ncfv_conservative_primitive_accuracy_20260922/results/all_variable_errors.csv"
    )
    density_summary = rows(
        SOURCE / "ncfv_exact_mean_accuracy_20260921/density_pressure_summary.csv"
    )
    point_ls = rows(
        SOURCE / "ncfv_ls_stencil_13_15_20260921/point_errors_t1p6.csv"
    )
    first_reconstruction = rows(
        SOURCE / "ncfv_ls_vs_svdls_dt_20260921/static_reconstruction.csv"
    )
    entropy_ls = rows(SOURCE / "ncfv_efficient_3d_entropy_20260920/summary.csv")
    assert len(all_variables) == 320
    assert len(density_summary) == 32

    # Snapshot from the two 2026-09-22 variational reports and their accuracy.json
    # records. The raw /tmp runs may be removed later, so the published numbers
    # are retained here without making this generator depend on volatile paths.
    vr = {
        "高效变分": {
            10: (0.012231826624752504, 0.030107630256096016, 0.23439635517192614,
                 0.04691774350818645, 0.02471939114558542),
            20: (0.002624312982205993, 0.006263866059857945, 0.04665176340364763,
                 0.01029556680650263, 0.005952732268950918),
            40: (0.000418654714570321, 0.0010658797177319139, 0.007462072431201672,
                 0.0014588336018201268, 0.0010484918807588015),
            80: (5.111304737549875e-05, 0.0001278455110827778, 0.0009420823809394019,
                 0.00017967735848002837, 0.00012718681692576456),
        },
        "传统变分": {
            10: (0.009808332569596057, 0.024275372247304385, 0.1885810802468496,
                 0.039013135304692276, 0.0192011317905),
            20: (0.001512645508350042, 0.0036067067727266473, 0.025625400569215195,
                 0.004758334887161685, 0.00350458970230),
            40: (0.0001418836879746659, 0.00037917567446170707, 0.0034663837978188283,
                 0.0005185335581052947, 0.000378501874875),
            80: (8.325306258841098e-06, 2.1244078146105694e-05, 0.0001665281207285929,
                 2.643835586069781e-05, 2.13947909415e-05),
        },
    }
    vr_pressure_point = {
        "高效变分": {
            10: (0.01658303046296549, 0.04691774350818645, 0.35540395455117024),
            20: (0.0036402953669313768, 0.01029556680650263, 0.0787506427041248),
            40: (0.0005184092506578322, 0.0014588336018201268, 0.011318803402396438),
            80: (6.371270860308931e-05, 0.00017967735848002837, 0.0014922523258440457),
        },
        "传统变分": {
            10: (0.01384455480817248, 0.039013135304692276, 0.29869501598648207),
            20: (0.0018618342976420756, 0.004758334887161685, 0.04122284873363069),
            40: (0.00017524405691460678, 0.0005185335581052947, 0.006202460976586666),
            80: (9.455705888417285e-06, 2.643835586069781e-05, 0.00033245150926425326),
        },
    }
    entropy_vr = {
        10: 0.03700991589233393,
        20: 0.004457166081799132,
        40: 0.00044561631739026134,
        80: 0.00005078830515303717,
    }

    parts: list[str] = []
    parts.append("""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>NCFV三棱柱网格体积加权误差综合报告</title>
<style>
@page { size: A4 landscape; margin: 1.5cm 1.6cm; }
body { font-family: 'Noto Sans CJK SC', 'Liberation Sans', sans-serif; color: #17212b; font-size: 9.5pt; line-height: 1.32; }
h1 { color: #12384b; font-size: 19pt; margin-bottom: 0.25em; }
h2 { color: #155d74; font-size: 13pt; margin-top: 1.25em; margin-bottom: 0.35em; }
h3 { color: #205369; font-size: 10.5pt; margin-top: 0.9em; margin-bottom: 0.25em; }
p { margin: 0.32em 0; }
.subtitle { color: #5b6671; font-size: 10pt; }
.formula { font-family: 'Liberation Serif', serif; font-size: 11pt; margin: 0.55em 0; padding: 0.35em 0.7em; background: #edf5f7; }
table { border-collapse: collapse; width: 100%; margin: 0.55em 0 0.9em; table-layout: fixed; font-size: 8pt; }
th { background: #e4f0f4; color: #193c4c; font-weight: bold; }
th, td { border: 0.5pt solid #b9cbd3; padding: 0.22em 0.4em; vertical-align: top; overflow-wrap: anywhere; }
tr:nth-child(even) td { background: #f7fafb; }
thead { display: table-header-group; }
.small { font-size: 7.3pt; }
.note { border-left: 3pt solid #b68a3c; padding: 0.4em 0.65em; background: #fff8eb; }
.pagebreak { page-break-before: always; }
.source { font-size: 8pt; color: #405363; overflow-wrap: anywhere; }
</style></head><body>""")
    parts.append("<h1>NCFV 三棱柱网格：体积加权误差综合报告</h1>")
    parts.append('<p class="subtitle">截至 2026-09-22 的已有计算结果整理 · 无新增求解 · 统一误差口径</p>')
    parts.append('<p class="note"><b>阅读要点：</b>原 LS 高效法在等熵涡 40→80 网格上约为 2.74 阶；高效变分重构、通量仍只使用一阶导数时，同一区间的恢复格点密度约为 3.08 阶。全部 L1/L2 均按节点对偶体积加权；L∞ 是不加权的最大误差。不同误差对象不能仅凭数值大小直接比较。</p>')

    parts.append("<h2>1. 统一的误差定义与可比范围</h2>")
    parts.append(paragraph("令 Vᵢ 为节点 i 在计算域中的对偶控制体体积份额，仅统计各 MPI 进程拥有的节点一次；本系列周期盒总体积 ΣᵢVᵢ≈400。对标量误差 eᵢ，使用："))
    parts.append('<p class="formula">L₁,V = Σᵢ Vᵢ |eᵢ| / Σᵢ Vᵢ；　L₂,V = [Σᵢ Vᵢ eᵢ² / Σᵢ Vᵢ]¹ᐟ²；　L∞ = maxᵢ |eᵢ|。</p>')
    parts.append(paragraph("相对误差为 L₂,V(e)/L₂,V(qref)；带符号偏差为 ΣᵢVᵢeᵢ/ΣᵢVᵢ。若参考量恒为零（如本涡的 w、ρw），相对误差与表观阶数不应解释为整体格式精度。收敛阶 p=ln(E粗/E细)/ln(h粗/h细)，h=(400/N周期独立节点)¹ᐟ³；N=512、3896、30416、239968。"))
    parts.append(paragraph("恢复格点值误差：eᵢ=Uᵢʳᵉᶜ−Uexact(xᵢ,t)。状态—参考均值误差：eᵢ=Wᵢ−Vᵢ⁻¹∫VᵢUexact dV。传统法的 Wᵢ 是求积得到的体均值；高效法的 Wᵢ 由格点值和梯度微分权重构造，只是体均值的高阶近似，故表中明确称“状态—参考均值”，不把它与传统体均值完全等同。"))
    parts.append(paragraph("原始变量表中的 p、u、v、w 均由数值或参考的守恒均值代入状态方程得到，即 P(Ū)，不是先算解析原始变量再体积分的平均值。传统法的 surfaceQuadratureOrder=3 对每个微三角面用 3 点、精确到二次；高效法不存储面 Gauss 点，不能将该参数解释为高效通量的求积阶。"))

    parts.append("<h2>2. 最新高效／传统变分对照：同一恢复格点值口径</h2>")
    vr_rows: list[list[object]] = []
    ls15 = {int(item["grid"]): item for item in point_ls if item["stencil"] == "15"}
    ls15_mean = {
        (item["method"], int(item["grid"])): item
        for item in density_summary if item["stencil"] == "15"
    }
    for method in ("高效 LS15", "高效变分", "传统变分"):
        previous_grid: int | None = None
        previous_error: float | None = None
        for grid in (10, 20, 40, 80):
            if method == "高效 LS15":
                item = ls15[grid]
                rho_l1 = float(item["rho_L1"])
                rho_l2 = float(item["rho_L2"])
                rho_linf = float(item["rho_Linf"])
                p_l2 = float(item["pressure_L2"])
                mean_l2 = float(ls15_mean[("efficient", grid)]["rho_mean_L2"])
            else:
                rho_l1, rho_l2, rho_linf, p_l2, mean_l2 = vr[method][grid]
            p_order = (order(previous_error, rho_l2, previous_grid, grid)
                       if previous_error is not None and previous_grid is not None else None)
            vr_rows.append([
                method, grid, number(rho_l1), number(rho_l2), number(rho_linf),
                number(p_l2), number(p_order, 4), number(mean_l2),
            ])
            previous_grid, previous_error = grid, rho_l2
    parts.append(table(
        ["方法", "网格", "ρ点 L1,V", "ρ点 L2,V", "ρ点 L∞", "p点 L2,V", "ρ点相邻阶", "状态—参考均值ρ L2,V"],
        vr_rows,
    ))
    pressure_rows = []
    for method in ("高效变分", "传统变分"):
        for grid in (10, 20, 40, 80):
            p_l1, p_l2, p_linf = vr_pressure_point[method][grid]
            pressure_rows.append([method, grid, number(p_l1), number(p_l2), number(p_linf)])
    parts.append(table(["变分方法", "网格", "p点 L1,V", "p点 L2,V", "p点 L∞"], pressure_rows))
    parts.append(paragraph("全部方法均用 Roe（含熵修正）、无限制器、SSPRK3 和固定全局步长 Δt=0.2/0.1/0.05/0.025，分别推进 8/16/32/64 步到 t=1.6。变分方案以 15 点 LS 为初值，再进行 30 轮邻点耦合更新。高效变分的点值由完整二次系数恢复，但积分物理通量只读取格点值和一阶导数；这一对照同时改变了重构与点恢复，不能把误差下降全部归给其中之一。"))
    parts.append(paragraph("80 网格高效变分的点值密度 L2,V 比高效 LS15 低约 55.1%，但约为传统变分的 6.02 倍。传统变分 40→80 的约 4.19 阶属于特定涡、网格和误差抵消产生的观测斜率，不意味着理论四阶。"))

    parts.append("<h2>3. LS 13–16 点：32 组密度与压力的加权 L2 总表</h2>")
    summary_rows = []
    for item in sorted(density_summary, key=lambda r: (r["method"], int(r["stencil"]), int(r["grid"]))):
        summary_rows.append([
            "传统" if item["method"] == "traditional" else "高效",
            item["stencil"], item["grid"], number(item["rho_mean_L2"]),
            number(item["rho_mean_L2_order"], 4), number(item["pressure_of_mean_L2"]),
            number(item["pressure_of_mean_L2_order"], 4), number(item["rho_point_L2"]),
        ])
    parts.append(table(
        ["方法", "模板点", "网格", "状态—参考均值ρ L2,V", "ρ阶", "p(状态) L2,V", "p阶", "恢复点ρ L2,V"],
        summary_rows, "small",
    ))
    parts.append(paragraph("这 32 组均为正规方程 LS；高效算法存储自由度与解析均值比较时含有表示误差。原始 L1,V、L2,V、L∞、相对 L2,V、偏差和各守恒／原始变量见附录的 320 行。13/14 点优于 16 点是本等熵涡上的事实，不是所有网格上的普适最优点数。"))

    parts.append("<h2>4. 首次重构、三维熵波与排除性实验</h2>")
    first_rows = []
    for item in first_reconstruction:
        first_rows.append([
            item["grid"], number(item["ls_rho_point_L2"]),
            number(item["point_order"], 4), number(item["ls_rho_gradient_L2"]),
            number(item["gradient_order"], 4),
            number(abs(float(item["ls_rho_point_L2"]) - float(item["svdls_rho_point_L2"])), 3),
        ])
    parts.append("<h3>4.1 t=0、16 点 LS/SVDLS 的体积加权重构误差</h3>")
    parts.append(table(
        ["网格", "ρ点 L2,V", "ρ点阶", "∇ρ L2,V", "∇ρ阶", "LS−SVDLS点值L2差"],
        first_rows,
    ))
    parts.append(paragraph("二次重构的梯度误差在细端接近二阶，点值误差达到三阶左右；两种线性代数求解器差异为舍入级。梯度 L2,V 中每节点先取向量差的欧氏范数，再用 Vᵢ 加权。"))

    entropy_rows = []
    previous_ls = previous_vr = None
    previous_grid = None
    for item in entropy_ls:
        grid = int(item["grid"])
        ls_error = float(item["rho_recovered_point_L2"])
        vr_error = entropy_vr[grid]
        entropy_rows.append([
            grid, number(ls_error),
            number(order(previous_ls, ls_error, previous_grid, grid), 4) if previous_ls else "—",
            number(vr_error),
            number(order(previous_vr, vr_error, previous_grid, grid), 4) if previous_vr else "—",
        ])
        previous_ls, previous_vr, previous_grid = ls_error, vr_error, grid
    parts.append("<h3>4.2 严格周期三维斜向熵波：恢复点密度 L2,V</h3>")
    parts.append(table(["网格", "高效 LS16", "相邻阶", "高效变分", "相邻阶"], entropy_rows))
    parts.append(paragraph("熵波是三方向变化的 Euler 精确解。LS16 细端约 2.99 阶；高效变分三段均超过 3。熵波变分使用 16 点 LS 初值，而上文等熵涡变分使用 15 点；本表只作光滑解测阶，不把熵波和等熵涡的绝对误差横向比较。"))

    experiments = [
        ["五阶体积分初始化", "高效 LS15；80 网格", "t=0 密度状态—参考均值 L2,V：1.227×10⁻⁶ → 4.433×10⁻¹⁴；t=1.6：2.83255×10⁻⁴ → 2.83767×10⁻⁴", "初值均值误差不是终态低阶的主因；终态略增来自误差抵消"],
        ["模板数 16→13", "高效 LS；80 网格", "密度状态—参考均值 L2,V：3.18665×10⁻⁴ → 2.69920×10⁻⁴（−15.3%）", "模板宽度改变误差常数，未独自恢复三阶"],
        ["解析节点值＋梯度覆盖", "历史高效 16 点；80 网格", "恢复点密度 L2,V：4.67117×10⁻⁴ → 9.09465×10⁻⁵（降低 5.14 倍）", "重构场是重要来源；旧基线与现行 LS15/变分不能直接作百分比分摊"],
        ["消除周期拼接", "历史传统 16 点；80 网格", "原涡/紧支撑平衡涡恢复点密度 L2,V：3.65595655×10⁻⁴ / 3.65594409×10⁻⁴", "接缝不是该传统对照的密度主因；不能直接替代高效法专项消融"],
        ["时间步减半", "历史传统 16 点；80 网格", "密度点 L2,V：3.655956548×10⁻⁴ → 3.644579605×10⁻⁴（−0.311%）", "该对照不支持 RK3 时间误差主导；高效变分仍需独立减半检查"],
        ["串并行", "高效变分；10 网格", "1/2 进程密度点 L2,V：0.0301076302561 / 0.0301076302555", "没有发现可见的 ghost 层不足造成误差阶损失"],
        ["折曲宏面几何", "IV10/IV20", "最小 |ΣAₖnₖ|/ΣAₖ = 0.858475", "这是面积度量差而非解误差百分比；单次宏面 Roe 与逐微面 Roe 尚未做纯消融"],
    ]
    parts.append("<h3>4.3 诊断实验与证据边界</h3>")
    parts.append(table(["试验", "对象", "加权误差／几何结果", "可支持的判断"], experiments, "small"))

    parts.append("<h2>5. 误差机制与尚需分离的项</h2>")
    parts.append(paragraph("对光滑涡，二次重构的格点梯度误差一般为 O(h²)，点值及左右界面迹的首个遗漏三次项为 O(h³)。高效通量把节点物理通量 F(Uᵢ) 和 ∇F(Uₖ) 与预计算微分权重组合；Roe 耗散则作用于左右宏面均值的跳量。单个三维面对偶面积 O(h²) 乘 O(h³) 通量密度误差、再除 O(h³) 控制体积，局部残差可有 O(h²) 项。整体三阶要求相邻面领先项有符号抵消，或传播中产生更高阶抵消；二次多项式再现本身不保证它。"))
    parts.append(paragraph("当前证据最有力地指向“重构／点恢复误差与通量、对偶体散度耦合”造成 LS 等熵涡的预渐近 2.7 阶。变分重构使最细一级恢复三阶，却仍比传统变分误差大约 6 倍；宏面一次 Roe、物理通量微分积分、不同初始化和点恢复的贡献尚未单独量化。三维熵波已证实高效离散不存在普遍的二阶上限。"))
    parts.append(paragraph("下一步验收宜在同一变分重构和同一初值下做 2×2 消融：完整二次点恢复与梯度微分点恢复；宏面一次 Roe 与逐微三角面 Roe。逐项保存中心、耗散及总 RHS 的体积加权 L2,V 和终态点误差，再检查 40/80/更细网格及时间步减半。"))

    parts.append('<div class="pagebreak"></div>')
    parts.append("<h2>附录 A. 320 条守恒／原始变量完整加权误差</h2>")
    parts.append(paragraph("下表逐条保留原始后处理结果。L1,V、L2,V、相对 L2,V 与偏差均含体积权重；L∞ 不加权；阶数由相邻网格 L2,V 计算。高效行的“均值”对象是高效存储状态与解析体均值的差。"))
    for method in ("traditional", "efficient"):
        for variable_type in ("conservative", "primitive_of_mean"):
            subset = [
                item for item in all_variables
                if item["method"] == method and item["variable_type"] == variable_type
            ]
            assert len(subset) == 80
            subset.sort(key=lambda item: (int(item["stencil"]), int(item["grid"]), item["variable"]))
            title = ("传统面积求积" if method == "traditional" else "高效微分") + (
                " · 守恒量" if variable_type == "conservative" else " · P(状态) 原始变量"
            )
            parts.append(f"<h3>{html.escape(title)}（80 行）</h3>")
            appendix_rows = [
                [item["stencil"], item["grid"], item["variable"],
                 number(item["L1"]), number(item["L2"]), number(item["Linf"]),
                 number(item["relative_L2"]), number(item["signed_bias"]),
                 number(item["L2_order"], 4)]
                for item in subset
            ]
            parts.append(table(
                ["模板", "网格", "变量", "L1,V", "L2,V", "L∞", "相对L2,V", "带符号偏差", "L2阶"],
                appendix_rows, "small",
            ))

    parts.append("<h2>附录 B. 数据出处与版本约束</h2>")
    sources = [
        "docs/reports/ncfv_conservative_primitive_accuracy_20260922/results/all_variable_errors.csv（附录 A，320 行）",
        "docs/reports/ncfv_exact_mean_accuracy_20260921/density_pressure_summary.csv（32 组密度／压力）",
        "docs/reports/ncfv_ls_stencil_13_15_20260921/point_errors_t1p6.csv（LS 点值误差）",
        "docs/dev/ncfv_efficient_variational_accuracy_20260922.md（高效变分）",
        "docs/dev/ncfv_variational_accuracy_20260922.md（传统变分）",
        "docs/reports/ncfv_ls_vs_svdls_dt_20260921/static_reconstruction.csv（首次重构）",
        "docs/reports/ncfv_efficient_3d_entropy_20260920/summary.csv（三维熵波 LS）",
        "docs/reports/ncfv_vortex_oracle_traces_20260920/report.md（解析重构轨迹）",
        "docs/reports/ncfv_gauss5_initialization_ls_20260921/report.md（初始化）",
        "docs/reports/ncfv_periodic_seam_20260918/report.md（接缝）",
        "docs/reports/ncfv_consistency_recheck_20260918/report.md（时间步及残差）",
        "docs/reports/ncfv_geometry_stencil_audit_20260920/report.md（宏面折曲几何）",
        "docs/dev/generate_ncfv_weighted_error_report.py（变分精度数据快照及本报告生成源）",
    ]
    parts.extend(f'<p class="source">{html.escape(source)}</p>' for source in sources)
    parts.append(paragraph("本报告仅整理截至所示日期已有计算，不新增数值模拟。早期“旧 16 点”探针与现行完整一环模板、局部矩实现不完全相同，故只用于机制定位，不能混为同一版本的严格单因素对照。"))
    parts.append("</body></html>")
    return "\n".join(parts)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: generate_ncfv_weighted_error_report.py OUTPUT.html")
    output = Path(sys.argv[1]).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build(), encoding="utf-8")
    print(output)
