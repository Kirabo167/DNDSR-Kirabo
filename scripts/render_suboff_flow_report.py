#!/usr/bin/env python3
"""Export consistent SUBOFF side sections and front/side wall-adjacent Cp views."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re
import tempfile
import zipfile

os.environ.setdefault("EGL_PLATFORM", "surfaceless")
os.environ.setdefault("VTK_DEFAULT_OPENGL_WINDOW", "vtkEGLRenderWindow")
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("VTK_SMP_MAX_THREADS", "4")
os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="suboff-report-mpl-"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors, font_manager, tri
from matplotlib.collections import LineCollection
import numpy as np
from PIL import Image
import pyvista as pv


ROOT = Path(__file__).resolve().parents[1]
FONT = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
if FONT.exists():
    font_manager.fontManager.addfont(str(FONT))
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(FONT)).get_name()
plt.rcParams.update({"axes.unicode_minus": False, "font.size": 12, "axes.titleweight": "medium"})


def scalar_range(values):
    values = np.asarray(values)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite field in postprocessing")
    return [float(np.min(values)), float(np.max(values))]


def reference_lut():
    reference = ROOT / "data/out/GPU/SUBOFF_SST_BDF2_20261104/snapshots/reference_zCFD_SUBOFF_pressure.png"
    pixels = np.asarray(Image.open(reference).convert("RGB"))
    positions = (np.arange(40) + .5) / 40
    anchors = pixels[96, (340 + positions * 551).astype(int)].astype(float)
    rgb = np.column_stack([np.interp(np.linspace(0, 1, 4096), positions, anchors[:, c]) for c in range(3)])
    lut = pv.LookupTable(n_values=4096)
    lut.values = np.column_stack((np.rint(rgb).astype(np.uint8), np.full(4096, 255, dtype=np.uint8)))
    lut.scalar_range = (-2.694, 1.009)
    return lut


def triangulation(section):
    section = section.triangulate()
    faces = section.faces.reshape(-1, 4)
    if not np.all(faces[:, 0] == 3):
        raise ValueError("Section contains non-triangular faces")
    indices = faces[:, 1:]
    xy = np.asarray(section.points[:, :2], dtype=float)
    a, b, c = xy[indices[:, 0]], xy[indices[:, 1]], xy[indices[:, 2]]
    area2 = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    indices = indices[np.abs(area2) > 1e-22]
    return section, tri.Triangulation(xy[:, 0], xy[:, 1], indices)


def outline_segments(body):
    cut = body.slice(normal=(0, 0, 1), origin=(0, 0, 0))
    segments = []
    offset = 0
    while offset < len(cut.lines):
        count = int(cut.lines[offset])
        indices = cut.lines[offset + 1:offset + 1 + count]
        segments.append(cut.points[indices, :2])
        offset += count + 1
    return segments


def smooth_section_display(section, limits, width):
    """Sample one C1 display field for both shading and isolines, within fluid cells.

    Original vertices are retained. Refinement only affects display triangles;
    no point is introduced across a solid boundary or across a missing cell.
    """
    section = section.copy()
    section.clear_cell_data()
    section = section.clean(tolerance=1e-10, absolute=True)
    section, original = triangulation(section)
    xy = np.column_stack((original.x, original.y))
    indices = original.triangles.copy()
    original_centers = np.mean(xy[indices], axis=1)
    triangle_parents = np.arange(len(indices))
    node_parents = np.full(len(xy), -1, dtype=np.int64)
    normalized = []
    for field, (low, high) in limits.items():
        values = np.asarray(section[field])
        if field in ("TurbulenceK", "VorticityMagnitude"):
            values, low, high = np.log(np.maximum(values, low)), np.log(low), np.log(high)
        normalized.append(np.clip((values - low) / (high - low), 0, 1))
    normalized = np.column_stack(normalized)
    # Split shared long edges consistently. Centroid fans avoid hanging nodes
    # while leaving the dense near-wall mesh untouched.
    for _ in range(2):
        edge_pairs = np.sort(indices[:, [[0, 1], [1, 2], [2, 0]]], axis=2)
        edges, first, inverse = np.unique(edge_pairs.reshape(-1, 2), axis=0, return_index=True, return_inverse=True)
        contrast = np.max(np.abs(normalized[edges[:, 0]] - normalized[edges[:, 1]]), axis=1)
        selected = ((np.linalg.norm(xy[edges[:, 0]] - xy[edges[:, 1]], axis=1) > width / 220)
                    & (contrast > 1 / 96))
        if not selected.any():
            break
        edge_nodes = np.full(len(edges), -1, dtype=np.int64)
        edge_nodes[selected] = len(xy) + np.arange(np.count_nonzero(selected))
        mids = edge_nodes[inverse].reshape(-1, 3)
        changed = np.any(mids >= 0, axis=1)
        changed_indices = indices[changed]
        changed_parents = triangle_parents[changed]
        centers = np.mean(xy[changed_indices], axis=1)
        center_start = len(xy) + np.count_nonzero(selected)
        xy = np.vstack((xy, np.mean(xy[edges[selected]], axis=1), centers))
        normalized = np.vstack((normalized, np.mean(normalized[edges[selected]], axis=1),
                                np.mean(normalized[changed_indices], axis=1)))
        node_parents = np.r_[node_parents, triangle_parents[first[selected] // 3], changed_parents]
        replacement = []
        replacement_parents = []
        for i, (corners, split) in enumerate(zip(changed_indices, mids[changed])):
            boundary = []
            for corner, mid in zip(corners, split):
                boundary.append(int(corner))
                if mid >= 0:
                    boundary.append(int(mid))
            replacement.extend((boundary[j], boundary[(j + 1) % len(boundary)], center_start + i)
                               for j in range(len(boundary)))
            replacement_parents.extend([changed_parents[i]] * len(boundary))
        indices = np.vstack((indices[~changed], np.asarray(replacement, dtype=np.int64)))
        triangle_parents = np.r_[triangle_parents[~changed], replacement_parents]
    display = pv.PolyData(np.column_stack((xy, np.zeros(len(xy)))),
                          np.column_stack((np.full(len(indices), 3), indices)).ravel())
    for field in limits:
        values = np.asarray(section.point_data[field])
        logarithmic = field in ("TurbulenceK", "VorticityMagnitude")
        base = np.log(np.maximum(values, limits[field][0])) if logarithmic else values
        interpolator = tri.CubicTriInterpolator(original, base, kind="geom")
        result = np.asarray(interpolator(xy[:, 0], xy[:, 1]).filled(np.nan))
        # A rounded midpoint on the sail boundary can fall a few ulps outside
        # the point locator. Evaluate just inside its known fluid parent cell;
        # keep the actual display coordinates and solid mask unchanged.
        for epsilon in (1e-8, 1e-6, 1e-4):
            bad = ~np.isfinite(result)
            bad[:len(base)] = False
            if not bad.any():
                break
            query = xy[bad] + epsilon * (original_centers[node_parents[bad]] - xy[bad])
            result[bad] = interpolator(query[:, 0], query[:, 1]).filled(np.nan)
        # Limit new samples to their original fluid triangle's nodal range:
        # cubic interpolation must not create spurious local extrema.
        result = np.clip(result, np.min(base), np.max(base))
        local_values = base[original.triangles]
        parents = node_parents[len(base):]
        result[len(base):] = np.clip(result[len(base):], np.min(local_values, axis=1)[parents],
                                     np.max(local_values, axis=1)[parents])
        result[:len(base)] = base
        if not np.isfinite(result).all():
            bad = ~np.isfinite(result)
            raise ValueError(f"Invalid smooth display interpolation: {field}, {np.count_nonzero(bad)} points, first={xy[bad][:5]}")
        display.point_data[field] = np.exp(result) if logarithmic else result
    print(f"Smooth display: {original.triangles.shape[0]} -> {len(indices)} triangles", flush=True)
    return triangulation(display)


def isoline_levels(limits, field):
    low, high = limits[field]
    return (np.geomspace(low, high, 9) if field in ("TurbulenceK", "VorticityMagnitude")
            else np.linspace(low, high, 9))[1:-1]


def label_isolines(ax, lines, region, outline, logarithmic):
    """Label a few clear contour locations without covering the body or each other."""
    from scipy.spatial import cKDTree
    wall = cKDTree(np.vstack(outline))
    width, height = region[1] - region[0], region[3] - region[2]
    placed = []
    for level, path in zip(lines.levels, lines.get_paths()):
        starts = np.flatnonzero(path.codes == 1) if path.codes is not None else np.array([0])
        candidates = []
        for start, end in zip(starts, np.r_[starts[1:], len(path.vertices)]):
            chain = path.vertices[start:end]
            if len(chain) < 8 or np.sum(np.linalg.norm(np.diff(chain, axis=0), axis=1)) < .14 * width:
                continue
            candidates.extend(chain[np.linspace(1, len(chain) - 2, min(30, len(chain) - 2)).astype(int)])
        if not candidates:
            continue
        candidates = np.asarray(candidates)
        clearance, _ = wall.query(candidates)
        for i in np.argsort(clearance)[::-1]:
            x, y = candidates[i]
            if not (region[0] + .05 * width < x < region[1] - .05 * width
                    and region[2] + .07 * height < y < region[3] - .07 * height):
                continue
            if clearance[i] < .025 * height or any(abs(x - px) < .13 * width and abs(y - py) < .08 * height
                                                   for px, py in placed):
                continue
            placed.append((x, y))
            break
        if len(placed) >= 4:
            break
    if placed:
        from matplotlib import patheffects
        labels = ax.clabel(lines, manual=placed, inline=True, inline_spacing=3, fontsize=7,
                           fmt=(lambda v: f"{v:.1e}") if logarithmic else (lambda v: f"{v:.4g}"))
        for text in labels:
            text.set_path_effects([patheffects.Stroke(linewidth=1.2, foreground="white"), patheffects.Normal()])


def simplify_closed_path(path, tolerance):
    """Reduce contour vertices within a data-space tolerance; retain every ring."""
    from matplotlib.path import Path as MplPath
    vertices, codes = [], []
    starts = np.flatnonzero(path.codes == MplPath.MOVETO)
    for start, end in zip(starts, np.r_[starts[1:], len(path.vertices)]):
        ring = path.vertices[start:end]
        if len(ring) < 4:
            vertices.extend(ring)
            codes.extend(path.codes[start:end])
            continue
        # CLOSEPOLY's coordinate may be a placeholder rather than a vertex.
        if path.codes[end - 1] == MplPath.CLOSEPOLY:
            ring = np.vstack((ring[:-1], ring[0]))
        keep = np.zeros(len(ring), dtype=bool)
        keep[[0, -1]] = True
        stack = [(0, len(ring) - 1)]
        while stack:
            first, last = stack.pop()
            if last <= first + 1:
                continue
            delta = ring[last] - ring[first]
            length2 = np.dot(delta, delta)
            offset = ring[first + 1:last] - ring[first]
            fraction = np.clip(offset @ delta / length2, 0, 1) if length2 else np.zeros(len(offset))
            distance2 = np.sum((offset - fraction[:, None] * delta) ** 2, axis=1)
            split = int(np.argmax(distance2))
            if distance2[split] > tolerance ** 2:
                middle = first + 1 + split
                keep[middle] = True
                stack.extend(((first, middle), (middle, last)))
        reduced = ring[keep]
        if len(reduced) < 4:
            reduced = ring  # Keep small islands/holes rather than erase them.
        vertices.extend(reduced)
        codes.extend([MplPath.MOVETO] + [MplPath.LINETO] * (len(reduced) - 2) + [MplPath.CLOSEPOLY])
    return MplPath(np.asarray(vertices).reshape(-1, 2), np.asarray(codes, dtype=np.uint8))


def draw_isolines(ax, triangles, values, limits, field, region, outline):
    """Use the identical displayed field for line overlays in reports and videos."""
    from matplotlib import patheffects
    logarithmic = field in ("VorticityMagnitude", "TurbulenceK")
    if logarithmic:
        values = np.maximum(values, limits[field][0])
    lines = ax.tricontour(triangles, values, levels=isoline_levels(limits, field),
                          colors="#172733", linewidths=.42, alpha=.7, zorder=4)
    lines.set_path_effects([patheffects.Stroke(linewidth=1., foreground="white", alpha=.45), patheffects.Normal()])
    label_isolines(ax, lines, region, outline, logarithmic)


def section_panel(fig, ax, section, triangles, field, limits, full, near, outline, compact=False, smooth=False, isolines=False):
    settings = {
        "Speed": ("速度模 |U|", "turbo", False),
        "VorticityMagnitude": ("涡量模 |curl U|（对数色标）", "inferno", True),
        "PressureRelative": ("相对压力 p − p∞", "coolwarm", False),
        "TurbulenceK": ("湍动能 k（对数色标）", "viridis", True),
    }
    title, cmap, logarithmic = settings[field]
    low, high = limits[field]
    values = np.asarray(section.point_data[field])
    if logarithmic:
        levels = np.geomspace(low, high, 128 if compact else 256)
        # Values below the declared display floor use the lowest color only.
        values = np.maximum(values, low)
        norm = colors.LogNorm(low, high)
        extend = "min"
    else:
        levels = np.linspace(low, high, 256 if compact else 512)
        norm = colors.Normalize(low, high)
        extend = "neither"
    if smooth:
        # Native PDF Gouraud shading shares the original nodal colors across
        # triangle edges. No contour bands, path simplification, or data filter.
        artist = ax.tripcolor(triangles, values, shading="gouraud",
                              cmap=plt.get_cmap(cmap, 4096), norm=norm)
    else:
        artist = ax.tricontourf(triangles, values, levels=levels, cmap=cmap, norm=norm, extend=extend)
    if compact and not smooth:
        region_width = 7.5 if near else full[1] - full[0]
        artist.set_paths([simplify_closed_path(path, region_width / 12000)
                          for path in artist.get_paths()])
    if isolines:
        draw_isolines(ax, triangles, values, limits, field, full if not near else (-.5, 7., -1., 1.), outline)
    ax.add_collection(LineCollection(outline, colors="#425466", linewidths=.55, zorder=3))
    region = full if not near else (-.5, 7.0, -1.0, 1.0)
    ax.set_xlim(region[:2])
    ax.set_ylim(region[2:])
    ax.set_aspect("equal", adjustable="box")
    ax.set_facecolor("white")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title, loc="left", fontsize=14, pad=9)
    ax.tick_params(direction="out", labelsize=10)
    for spine in ax.spines.values():
        spine.set_color("#728092")
    bar = fig.colorbar(artist, ax=ax, pad=.022, fraction=.035, shrink=.95,
                       **({"extend": extend} if smooth else {}))
    if logarithmic:
        powers = np.arange(np.ceil(np.log10(low)), np.floor(np.log10(high)) + 1)
        if len(powers) > 5:
            powers = powers[np.linspace(0, len(powers) - 1, 5).astype(int)]
        ticks = list(10.**powers)
        labels = [rf"$10^{{{int(p)}}}$" for p in powers]
        if not ticks or high / ticks[-1] > 1.5:
            ticks.append(high)
            labels.append(f"{high:.2g}")
        bar.set_ticks(ticks)
        bar.set_ticklabels(labels)
    else:
        from matplotlib.ticker import MaxNLocator
        bar.locator = MaxNLocator(nbins=5)
        bar.update_ticks()
    bar.ax.tick_params(labelsize=10)


def save_figure(fig, destination, vector=False, formats=(".svg", ".pdf")):
    fig.savefig(destination, dpi=220, facecolor="white")
    if vector:
        # Matplotlib rasterizes dense colorbars automatically; explicitly retain
        # paths/polygons in these exports, including the colorbar itself.
        for ax in fig.axes:
            for collection in ax.collections:
                collection.set_rasterized(False)
                if isinstance(collection, matplotlib.collections.QuadMesh) or (
                        isinstance(collection, matplotlib.contour.ContourSet) and collection.filled):
                    # Overlap adjacent filled paths slightly to prevent white
                    # stitching seams in PDF/SVG viewers at fractional zoom.
                    collection.set_edgecolor("face")
                    collection.set_linewidth(.1)
        for extension in formats:
            fig.savefig(destination.with_suffix(extension), facecolor="white")


def make_sections(out, section, near_section, full, limits, step, physical_time, outline, vector=False, compact=False, smooth=False, isolines=False, display_cache=None):
    fields = ("Speed", "VorticityMagnitude", "PressureRelative", "TurbulenceK")
    products = []
    for near in (True, False):
        if smooth:
            cache_key = "near" if near else "full"
            if display_cache is not None and cache_key in display_cache:
                data, triangles = display_cache[cache_key]
            else:
                data, triangles = smooth_section_display(near_section if near else section, limits,
                                                         7.5 if near else full[1] - full[0])
                if display_cache is not None:
                    display_cache[cache_key] = (data, triangles)
        else:
            data, triangles = triangulation(near_section if near else section)
        label = "潜艇周围流场" if near else "整个求解域流场"
        name = "near_field_side" if near else "whole_domain_side"
        fig, axes = plt.subplots(4, 1, figsize=(14, 19)) if near else plt.subplots(2, 2, figsize=(16, 11))
        for ax, field in zip(np.asarray(axes).ravel(), fields):
            section_panel(fig, ax, data, triangles, field, limits, full, near, outline, compact, smooth, isolines)
        variant = " · 有等值线" if isolines else (" · 无等值线" if smooth else "")
        fig.suptitle(f"{label} · 侧视截面 z=0{variant}\n第 {step} 步，t={physical_time:.3f}", fontsize=19, y=.985)
        fig.subplots_adjust(top=.925, bottom=.065, left=.075, right=.94, hspace=.45, wspace=.28)
        note = "同源三次显示插值并局部限幅，原节点值保留；正值场按对数插值；计算数据未修改。内迭代未达收敛标准。" if smooth else "同一场量在近场/全域使用相同色标；单元值插值到节点；涡量为三维速度场后处理旋度。内迭代未达收敛标准。"
        fig.text(.5, .015, note, ha="center", fontsize=10, color="#6c4b27")
        destination = out / f"{name}.png"
        # SVG expands every Gouraud triangle into many gradient paths; PDF has
        # native mesh shading and keeps the complete display mesh compact.
        save_figure(fig, destination, vector, formats=(".pdf",) if smooth else (".svg", ".pdf"))
        plt.close(fig)
        products.append(destination)
        print(f"Saved {destination.name}", flush=True)
        if compact:
            continue  # Dashboards already contain all four fields.
        for field in fields:
            fig, ax = plt.subplots(figsize=(14, 5) if near else (14, 9))
            section_panel(fig, ax, data, triangles, field, limits, full, near, outline)
            fig.suptitle(f"{label} · z=0 · step={step} · t={physical_time:.3f}", fontsize=17, y=.98)
            fig.subplots_adjust(top=.87, bottom=.13, left=.07, right=.93)
            destination = out / f"{name}_{field}.png"
            save_figure(fig, destination, vector)
            plt.close(fig)
            products.append(destination)
    return products


def surface_view(body, out, front, step, physical_time):
    width, height = (1600, 1600) if front else (3000, 1200)
    plotter = pv.Plotter(off_screen=True, window_size=(width, height), shape=(3, 1), border=False)
    plotter.enable_anti_aliasing("ssaa")
    for i, viewport in enumerate(((0, .85, 1, 1), (0, .17, 1, .85), (0, 0, 1, .17))):
        plotter.subplot(i, 0)
        plotter.background_color = "white"
        plotter.renderer.SetViewport(*viewport)
        plotter.camera.parallel_projection = True
    plotter.subplot(0, 0)
    label = "Front view / looking downstream (+x)" if front else "Side view / looking along -z"
    plotter.add_text("SUBOFF | wall-adjacent Cp", position="upper_left", font_size=20, color="#14334b")
    plotter.add_text(f"{label}\nstep={step}    t={physical_time:.3f}", position="lower_left", font_size=16, color="#425466")
    plotter.subplot(1, 0)
    actor = plotter.add_mesh(body, scalars="Cp", preference="point", cmap=reference_lut(), clim=(-2.694, 1.009),
                             show_scalar_bar=False, show_edges=False, smooth_shading=True,
                             interpolate_before_map=True, ambient=.65, diffuse=.35, specular=0)
    bounds = np.asarray(body.bounds)
    center = .5 * (bounds[::2] + bounds[1::2])
    plotter.camera.focal_point = tuple(center)
    plotter.camera.position = tuple(center + (np.array([-20., 0, 0]) if front else np.array([0., 0, 20.])))
    plotter.camera.up = (0, 1, 0)
    extent = bounds[1::2] - bounds[::2]
    horizontal = extent[2] if front else extent[0]
    plotter.camera.parallel_scale = max(extent[1] / 2, horizontal / (2 * width / (.68 * height))) * 1.12
    plotter.add_axes(xlabel="x", ylabel="y", zlabel="z", color="#425466")
    plotter.subplot(2, 0)
    for value in (-2.694, -2., -1., 0., 1.009):
        actor.mapper.lookup_table.SetAnnotation(value, f"{value:g}")
    plotter.add_scalar_bar(title="Cp = (p-p_inf)/(0.5 rho U_inf^2)", mapper=actor.mapper,
                          vertical=False, position_x=.18, position_y=.40, width=.64, height=.22,
                          title_font_size=18 if front else 22, label_font_size=15 if front else 18,
                          color="#14334b", n_labels=0, n_colors=4096)
    plotter.add_text("Interpolated display | Inner convergence criteria not met", position="lower_right",
                     font_size=10 if front else 14, color="#815623")
    destination = out / ("surface_Cp_front.png" if front else "surface_Cp_side.png")
    plotter.screenshot(destination)
    plotter.close()
    print(f"Saved {destination.name}", flush=True)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--step", type=int)
    args = parser.parse_args()
    active = json.loads((ROOT / "data/out/GPU/SUBOFF_active_run.json").read_text())
    run = args.run.resolve() if args.run else Path(active["directory"])
    status = json.loads((run / "status.json").read_text())
    entries = json.loads(next((run / "fields").glob("*.series")).read_text())["files"]
    if args.step is not None:
        entries = [e for e in entries if e["name"].endswith(f"_{args.step:08d}.vtkhdf")]
    entry = max(entries, key=lambda e: e["time"])
    source = run / "fields" / entry["name"]
    step = int(re.search(r"_(\d+)\.vtkhdf$", source.name)[1])
    physical_time = float(entry["time"])
    out = run / "postprocess" / f"step_{step:08d}"
    out.mkdir(parents=True, exist_ok=True)
    config = json.loads(Path(status["config"]).read_text())
    rho = float(config["acmSettings"]["rho0"])
    far = config["acmSettings"]["farFieldValue"]
    u_inf = float(np.linalg.norm(far[:3]))
    p_inf = float(far[3])
    if config["acmSettings"]["pressureStorage"] != "PhysicalP":
        raise ValueError("Cp export requires physical pressure storage")
    print(f"Reading completed step {step}, t={physical_time}: {source}", flush=True)
    mesh = pv.read(source)
    raw_ranges = {name: scalar_range(mesh.cell_data[name]) for name in ("Velocity", "Pressure", "TurbulenceK", "TurbulenceOmega")}
    for name in list(mesh.cell_data):
        if name not in ("Velocity", "Pressure", "TurbulenceK", "TurbulenceOmega"):
            del mesh.cell_data[name]
    print("Cell-to-point interpolation and full three-dimensional curl/Q/divergence", flush=True)
    mesh = mesh.cell_data_to_point_data(pass_cell_data=False)
    mesh = mesh.compute_derivative(scalars="Velocity", gradient=False, vorticity="Vorticity",
                                   qcriterion="QCriterion", divergence="Divergence", preference="point", faster=False)
    mesh.point_data["Speed"] = np.linalg.norm(mesh.point_data["Velocity"], axis=1)
    mesh.point_data["VorticityMagnitude"] = np.linalg.norm(mesh.point_data["Vorticity"], axis=1)
    mesh.point_data["PressureRelative"] = np.asarray(mesh.point_data["Pressure"]) - p_inf
    mesh.point_data["Cp"] = mesh.point_data["PressureRelative"] / (.5 * rho * u_inf**2)
    point_ranges = {name: scalar_range(mesh.point_data[name]) for name in mesh.point_data}
    print("Extracting z=0 section and submarine surface", flush=True)
    section = mesh.slice(normal=(0, 0, 1), origin=(0, 0, 0))
    near = section.clip_box((-.5, 7., -1., 1., -.001, .001), invert=False).extract_surface(algorithm="dataset_surface")
    surface = mesh.extract_surface(algorithm="dataset_surface").connectivity()
    bodies = []
    for region_id in np.unique(surface.cell_data["RegionId"]):
        region = surface.threshold([float(region_id), float(region_id)], scalars="RegionId")
        extents = np.asarray(region.bounds)[1::2] - np.asarray(region.bounds)[::2]
        if 2 < extents[0] < 6 and max(extents[1:]) < 3:
            bodies.append(region)
    if not bodies:
        raise RuntimeError("Cannot identify the submarine inner boundary")
    body = pv.merge(bodies).extract_surface(algorithm="dataset_surface").clean()
    bounds = list(mesh.bounds)
    full = (bounds[0], bounds[1], bounds[2], bounds[3])
    section_ranges = {name: scalar_range(section.point_data[name]) for name in ("Speed", "VorticityMagnitude", "PressureRelative", "TurbulenceK")}
    pmax = max(abs(v) for v in section_ranges["PressureRelative"])
    wmax = section_ranges["VorticityMagnitude"][1]
    kmax = section_ranges["TurbulenceK"][1]
    limits = {"Speed": [0., section_ranges["Speed"][1] * 1.00001],
              "VorticityMagnitude": [max(wmax * 1e-5, 1e-12), wmax * 1.00001],
              "PressureRelative": [-pmax * 1.00001, pmax * 1.00001],
              "TurbulenceK": [max(section_ranges["TurbulenceK"][0], 1e-14), kmax * 1.00001]}
    print(f"Section cells={section.n_cells}; body faces={body.n_cells}; ranges={section_ranges}", flush=True)
    section.save(out / "whole_domain_side_slice.vtp")
    near.save(out / "near_field_side_slice.vtp")
    body.save(out / "submarine_surface_fields.vtp")
    np.savetxt(out / "submarine_surface_Cp.csv", np.column_stack((body.points, body.point_data["Pressure"], body.point_data["Cp"])),
               delimiter=",", header="x,y,z,pressure,Cp", comments="", fmt="%.12e")
    # Preserve mesh-native sampling; no geometric smoothing or invented cells.
    products = make_sections(out, section, near, full, limits, step, physical_time, outline_segments(body))
    products += [surface_view(body, out, False, step, physical_time), surface_view(body, out, True, step, physical_time)]
    cp = np.asarray(body.point_data["Cp"])
    metadata = dict(source=str(source), step=step, physical_time=physical_time, cells=mesh.n_cells,
                    mesh_bounds=bounds, section_plane="z=0; x horizontal, y vertical; no out-of-plane projection",
                    near_bounds=[-.5, 7., -1., 1.], side_view="camera along -z", front_view="camera along +x, from bow",
                    raw_cell_ranges=raw_ranges, derived_point_ranges=point_ranges, section_ranges=section_ranges,
                    display_ranges=limits, cp_display_range=[-2.694, 1.009], cp_color_levels=4096,
                    surface_point_cp_range=scalar_range(cp), cp_saturated_below=int(np.sum(cp < -2.694)),
                    cp_saturated_above=int(np.sum(cp > 1.009)), rho=rho, u_inf=u_inf, p_inf=p_inf,
                    cp_formula="(p-p_inf)/(0.5*rho*U_inf**2)", inner_converged=False,
                    interpolation="Volume cell values averaged to shared mesh vertices, then linearly interpolated for display; geometry unchanged.",
                    derivative_method="VTK full 3D point-velocity gradient, faster=False; curl before slicing; approximate postprocessing derivative, not the solver reconstruction gradient.",
                    vorticity_log_floor=limits["VorticityMagnitude"][0], units="Solver units; Cp dimensionless; no rescaling to an external dimensional freestream.",
                    surface_pressure_method="Wall-adjacent volume pressures averaged to surface vertices, without wall-normal extrapolation.",
                    additional_vtp_fields=["Velocity", "Vorticity", "QCriterion", "Divergence", "TurbulenceK", "TurbulenceOmega"])
    (out / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    fig = plt.figure(figsize=(12, 9), facecolor="white")
    fig.text(.075, .90, "SUBOFF 流场与表面压力系数报告", fontsize=25, color="#14334b")
    lines = [f"保存步：{step}    物理时间：{physical_time:.3f}    体单元数：{mesh.n_cells:,}",
             "侧视截面：z=0；x 水平、y 竖直。正视图：从艇首沿 +x 观察。",
             f"全域 x=[{bounds[0]:.3f}, {bounds[1]:.3f}]，y=[{bounds[2]:.3f}, {bounds[3]:.3f}]。",
             "近场 x=[−0.5, 7.0]，y=[−1.0, 1.0]；近场与全域的同一场量共用色标。",
             "速度、压力和 SST 变量来自已保存解；变量采用求解器单位。",
             "涡量为三维速度场后处理旋度，先计算三维梯度再截取中央截面。",
             "节点速度由单元平均值插值得到，因此涡量不等同于求解器内部重构梯度。",
             "图中涡量和湍动能使用对数色标；涡量显示下限见 metadata.json。",
             "Cp=(p−p∞)/(0.5ρU∞²)，本工况 ρ=1、U∞=1、p∞=0，因此 Cp=2p。",
             "表面 Cp 由邻壁单元压力插值到表面节点；采用此前的连续参考色带。",
             "Cp 固定色标 [−2.694, 1.009]，超出范围的值按端点颜色显示。",
             "平滑仅用于插值显示，未移动网格、未增加数值分辨率。",
             "当前内迭代未达到收敛标准；本报告为瞬时计算快照，不作精度认证。",
             "同时提供单项高清图片、截面/表面 VTP、表面 Cp CSV 和元数据。"]
    for i, line in enumerate(lines):
        fig.text(.075, .81 - i * .047, line, fontsize=12, color="#314355")
    cover = out / "report_cover.png"
    fig.savefig(cover, dpi=180)
    plt.close(fig)
    page_paths = [cover, out / "near_field_side.png", out / "whole_domain_side.png", out / "surface_Cp_side.png", out / "surface_Cp_front.png"]
    pages = [Image.open(p).convert("RGB") for p in page_paths]
    pdf = out / f"SUBOFF_flow_report_step_{step:08d}.pdf"
    pages[0].save(pdf, save_all=True, append_images=pages[1:], resolution=180.)
    for page in pages:
        page.close()
    bundle = out / f"SUBOFF_flow_report_step_{step:08d}.zip"
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
        for path in sorted(out.iterdir()):
            if path.suffix in (".png", ".pdf", ".vtp", ".csv", ".json"):
                archive.write(path, path.name)
    result = dict(output=str(out), step=step, physical_time=physical_time, pdf=str(pdf), bundle=str(bundle))
    (run / "latest_flow_report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
