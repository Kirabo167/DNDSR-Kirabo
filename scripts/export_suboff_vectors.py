#!/usr/bin/env python3
"""Export editable vector graphics and velocity arrows from an existing flow report."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET
import zipfile

from render_suboff_flow_report import (ROOT, plt, pv, np, colors, LineCollection,
                                      isoline_levels, make_sections, outline_segments, reference_lut, save_figure)


SURFACE_CP_ISOLINE_INTERVAL = .05


def surface_cp_levels(body):
    values = np.asarray(body.point_data["Cp"])
    first = int(np.ceil(values.min() / SURFACE_CP_ISOLINE_INTERVAL))
    last = int(np.floor(values.max() / SURFACE_CP_ISOLINE_INTERVAL))
    return np.round(np.arange(first, last + 1) * SURFACE_CP_ISOLINE_INTERVAL, 12)


def surface_isolines(surface, levels, front):
    """Project 3D Cp contours, retaining only the camera-visible segments."""
    cut = surface.contour(isosurfaces=levels, scalars="Cp")
    bounds = np.asarray(surface.bounds).reshape(3, 2)
    axis = 0 if front else 2
    span = np.linalg.norm(bounds[:, 1] - bounds[:, 0])
    near, far = (bounds[axis, 0] - span, bounds[axis, 1] + span) if front else (
        bounds[axis, 1] + span, bounds[axis, 0] - span)
    visible = []
    cursor = 0
    while cursor < len(cut.lines):
        count = int(cut.lines[cursor])
        points = cut.points[cut.lines[cursor + 1:cursor + count + 1]]
        for a, b in zip(points[:-1], points[1:]):
            midpoint = .5 * (a + b)
            start, end = midpoint.copy(), midpoint.copy()
            start[axis], end[axis] = near, far
            hit, _ = surface.ray_trace(start, end, first_point=True)
            if np.size(hit) and np.linalg.norm(np.asarray(hit).reshape(-1, 3)[0] - midpoint) < span * 1e-6:
                visible.append(np.asarray([a, b])[:, [2, 1] if front else [0, 1]])
        cursor += count + 1
    return visible


def cp_view(body, out, front, metadata, compact=False, smooth=False, isolines=False):
    surface = body.triangulate()
    original_cells = surface.n_cells
    if compact and not smooth and original_cells > 6000:
        surface.set_active_scalars("Cp")
        surface = surface.decimate(1 - 6000 / original_cells, attribute_error=True,
                                   scalars=True, scalars_weight=1., boundary_constraints=True)
        # Reprobe the original Cp rather than use the decimator's optimized values.
        from scipy.spatial import cKDTree
        _, closest = cKDTree(body.points).query(surface.points)
        surface.point_data["Cp"] = np.asarray(body.point_data["Cp"])[closest]
    indices = surface.faces.reshape(-1, 4)[:, 1:]
    # Orthographic projection with back-to-front ordering of the actual surface
    # triangles. Gouraud fills remain vector gradients in SVG/PDF.
    xyz = np.asarray(surface.points, dtype=float)
    xy = xyz[:, [2, 1]] if front else xyz[:, :2]
    depth = -xyz[:, 0] if front else xyz[:, 2]
    a, b, c = xy[indices[:, 0]], xy[indices[:, 1]], xy[indices[:, 2]]
    area = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    indices = indices[np.abs(area) > 1e-20]
    indices = indices[np.argsort(np.mean(depth[indices], axis=1), kind="stable")]
    fig, ax = plt.subplots(figsize=(8, 9) if front else (15, 5))
    cmap = colors.ListedColormap(np.asarray(reference_lut().values, dtype=float) / 255)
    artist = ax.tripcolor(xy[:, 0], xy[:, 1], indices, np.asarray(surface.point_data["Cp"]),
                         shading="gouraud", cmap=cmap, norm=colors.Normalize(-2.694, 1.009, clip=True))
    cp_levels = surface_cp_levels(body)
    if isolines:
        segments = surface_isolines(surface, cp_levels, front)
        ax.add_collection(LineCollection(segments, colors="#172733", linewidths=.32, alpha=.75, zorder=4))
    ax.set_aspect("equal")
    ax.margins(.06)
    ax.set_xlabel("z" if front else "x")
    ax.set_ylabel("y")
    label = "正视图（从艇首沿 +x 观察）" if front else "侧视图（沿 −z 观察）"
    variant = " · 有等值线" if isolines else (" · 无等值线" if smooth else "")
    ax.set_title(f"艇体表面邻壁压力系数 Cp · {label}{variant}\n第 {metadata['step']} 步，t={metadata['physical_time']:.3f}", fontsize=15)
    bar = fig.colorbar(artist, ax=ax, orientation="horizontal", pad=.17 if front else .24, fraction=.055)
    bar.set_ticks([-2.694, -2, -1, 0, 1.009])
    bar.set_label("Cp = (p − p∞) / (0.5 ρ U∞²)")
    fig.subplots_adjust(top=.87, bottom=.15, left=.09, right=.95)
    note = "原始曲面、原始节点 Cp；共享节点连续渐变；未降采样、未过滤场值。" if smooth else ("连续三角形渐变；仅显示曲面简化，Cp 取原曲面最近节点值，原始计算数据保留。" if compact else "连续颜色采用曲面三角形节点插值；原网格与压力值保留。")
    if isolines:
        note = (f"Cp 等值线：{cp_levels[0]:g} 至 {cp_levels[-1]:g}，间隔 {SURFACE_CP_ISOLINE_INTERVAL:g}；"
                "取自同一曲面 Cp 场，已剔除背面遮挡线段。") if len(cp_levels) else "当前曲面 Cp 范围内无等值线。"
    fig.text(.5, .025, note, ha="center", fontsize=10)
    name = "surface_Cp_front" if front else "surface_Cp_side"
    save_figure(fig, out / f"{name}.png", vector=True, formats=(".pdf",) if smooth else (".svg", ".pdf"))
    plt.close(fig)
    print(f"Saved vector {name}", flush=True)
    return dict(original_triangles=original_cells, display_triangles=surface.n_cells)


def arrows(section, body, out, near, metadata):
    b = metadata["mesh_bounds"]
    xmin, xmax, ymin, ymax = (-.5, 7., -1., 1.) if near else (b[0], b[1], b[2], b[3])
    nx = 63 if near else 57
    ny = max(17, int(nx * (ymax - ymin) / (xmax - xmin)))
    dx = (xmax - xmin) / nx
    gx, gy = np.meshgrid(np.linspace(xmin + dx / 2, xmax - dx / 2, nx),
                         np.linspace(ymin + dx / 2, ymax - dx / 2, ny))
    points = pv.PolyData(np.column_stack((gx.ravel(), gy.ravel(), np.zeros(gx.size))))
    sampled = points.sample(section, tolerance=1e-6)
    valid = np.asarray(sampled.point_data["vtkValidPointMask"]).astype(bool)
    if np.mean(valid) < .7:
        raise RuntimeError("Insufficient valid velocity samples in the section")
    xyz = points.points[valid]
    velocity = np.asarray(sampled.point_data["Velocity"])[valid]
    speed = np.linalg.norm(velocity, axis=1)
    fig, ax = plt.subplots(figsize=(15, 5.4) if near else (14, 9.5))
    quiver = ax.quiver(xyz[:, 0], xyz[:, 1], velocity[:, 0], velocity[:, 1], speed,
                       cmap="turbo", norm=colors.Normalize(*metadata["display_ranges"]["Speed"]),
                       angles="xy", scale_units="xy", scale=metadata["u_inf"] / (.78 * dx),
                       pivot="mid", width=.0016, headwidth=3.1, headlength=4.1, minlength=0)
    ax.add_collection(LineCollection(outline_segments(body), colors="#25374b", linewidths=.85))
    ax.quiverkey(quiver, .83, 1.04, metadata["u_inf"], "面内速度 = 1", labelpos="E", coordinates="axes")
    ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax), xlabel="x", ylabel="y")
    ax.set_aspect("equal")
    label = "潜艇近场" if near else "整个求解域"
    ax.set_title(f"{label}速度矢量 · z=0\n第 {metadata['step']} 步，t={metadata['physical_time']:.3f}", fontsize=16, pad=25)
    bar = fig.colorbar(quiver, ax=ax, fraction=.035, pad=.025)
    bar.set_label("三维速度模 |U|")
    fig.subplots_adjust(top=.82, bottom=.13, left=.07, right=.92)
    fig.text(.5, .02, "箭头：面内速度 (Ux, Uy)，长度与速度成正比；颜色：三维速度模。固体内部不采样。", ha="center", fontsize=10)
    name = "near_field_velocity_vectors" if near else "whole_domain_velocity_vectors"
    save_figure(fig, out / f"{name}.png", vector=True)
    plt.close(fig)
    np.savetxt(out / f"{name}.csv", np.column_stack((xyz, velocity, speed)), delimiter=",",
               header="x,y,z,Ux,Uy,Uz,speed", comments="", fmt="%.12e")
    print(f"Saved {name}: {len(xyz)} arrows", flush=True)


def export_variant(report, metadata, section, near, body, compact, smooth, isolines, display_cache):
    out = report / ("vector_smooth_isolines" if isolines else (
        "vector_smooth" if smooth else ("vector_compact" if compact else "vector")))
    out.mkdir(exist_ok=True)
    b = metadata["mesh_bounds"]
    make_sections(out, section, near, (b[0], b[1], b[2], b[3]), metadata["display_ranges"],
                  metadata["step"], metadata["physical_time"], outline_segments(body), vector=True,
                  compact=compact, smooth=smooth, isolines=isolines, display_cache=display_cache)
    surface_audit = {}
    for front in (False, True):
        surface_audit["front" if front else "side"] = cp_view(body, out, front, metadata, compact, smooth, isolines)
    for close in (True, False):
        arrows(section, body, out, close, metadata)
    audit = {}
    for svg in sorted(out.glob("*.svg")):
        tree = ET.parse(svg)
        images = [e for e in tree.iter() if e.tag.rsplit("}", 1)[-1] == "image"]
        if images:
            raise RuntimeError(f"Embedded raster detected in {svg}")
        audit[svg.name] = dict(embedded_rasters=0, bytes=svg.stat().st_size)
    pdf = out / f"SUBOFF_vector_report_step_{metadata['step']:08d}.pdf"
    if shutil.which("pdfunite"):
        selected = [out / f"{name}.pdf" for name in ("near_field_side", "whole_domain_side", "surface_Cp_side", "surface_Cp_front", "near_field_velocity_vectors", "whole_domain_velocity_vectors")]
        subprocess.run(["pdfunite", *(str(p) for p in selected), str(pdf)], check=True)
    info = dict(step=metadata["step"], physical_time=metadata["physical_time"], source_report=str(report),
                compact=compact, smooth=smooth, isolines=isolines, surface_audit=surface_audit,
                display_ranges=metadata["display_ranges"],
                contour_method="Isolines and continuous shading share exactly the same sampled display field" if isolines else ("No isolines; continuous vector mesh shading" if smooth else "Filled contour bands"),
                field_processing="Geometric cubic display interpolation with original nodes retained; log interpolation for positive fields; new samples bounded by the original parent triangle's nodal range; no original data modification" if smooth else "See contour and surface methods",
                section_isoline_levels={field: [float(v) for v in isoline_levels(metadata['display_ranges'], field)] for field in metadata["display_ranges"]} if isolines else {},
                surface_isoline_levels=[float(v) for v in surface_cp_levels(body)] if isolines else [],
                surface_isoline_interval=SURFACE_CP_ISOLINE_INTERVAL if isolines else None,
                formats="Six-page native-vector PDF, PNG previews, velocity-arrow SVGs" if smooth else "SVG and PDF",
                svg_audit=audit, surface_method="Depth-sorted orthographic surface triangles, vector Gouraud gradients",
                velocity_method="Linear probing of the z=0 section; invalid/solid samples excluded; in-plane vector lengths proportional to velocity")
    if smooth:
        info["shared_display_fields"] = {
            region: dict(points=data.n_points, triangles=len(triangles.triangles),
                         field_sha256={field: hashlib.sha256(np.asarray(data[field]).tobytes()).hexdigest()
                                       for field in metadata["display_ranges"]})
            for region, (data, triangles) in display_cache.items()}
    (out / "vector_metadata.json").write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n")
    bundle = out / f"SUBOFF_vector_figures_step_{metadata['step']:08d}.zip"
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(out.iterdir()):
            include = p.suffix in (".svg", ".pdf", ".png", ".csv", ".json")
            if compact:
                include = p.suffix in (".svg", ".csv", ".json") or p == pdf
            if include:
                z.write(p, p.name)
    print(json.dumps(dict(directory=str(out), pdf=str(pdf), bundle=str(bundle)), indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--full-detail", action="store_true", help="Retain dense paths and duplicate individual plots")
    parser.add_argument("--contours", action="store_true", help="Use legacy compact filled contour bands")
    parser.add_argument("--line-variant", choices=("both", "without", "with"), default="both",
                        help="By default export both versions using one shared display field")
    args = parser.parse_args()
    report = args.report.resolve()
    metadata = json.loads((report / "metadata.json").read_text())
    compact = not args.full_detail
    smooth = compact and not args.contours
    if not smooth and args.line_variant == "with":
        parser.error("Isoline overlays require the default smooth display mode")
    plt.rcParams.update({"pdf.fonttype": 3, "svg.fonttype": "none"})
    section = pv.read(report / "whole_domain_side_slice.vtp")
    near = pv.read(report / "near_field_side_slice.vtp")
    body = pv.read(report / "submarine_surface_fields.vtp")
    variants = (False, True) if smooth and args.line_variant == "both" else (args.line_variant == "with",)
    display_cache = {}
    for isolines in variants:
        export_variant(report, metadata, section, near, body, compact, smooth, isolines, display_cache)


if __name__ == "__main__":
    main()
