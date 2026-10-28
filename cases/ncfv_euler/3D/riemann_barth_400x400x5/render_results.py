"""Render the four thin-slab NCFV Riemann cases without loading a full PVTU.

Read each VTU piece separately, enable Density/Pressure only, check the complete
piece for physical values, then retain its z slice. Cache the merged 2D slices
on disk. No solver output is modified or deleted. All videos use the same
absolute-time axis: the previous saved state is held between outputs and a case
is frozen after its final time. There is no interpolation in time.

Example, from the project root::

    EGL_PLATFORM=surfaceless venv/bin/python \
      cases/ncfv_euler/3D/riemann_barth_400x400x5/render_results.py \
      --root data/out/NCFV/riemann_barth_400x400x5 --workers 2
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET

os.environ.setdefault("EGL_PLATFORM", "surfaceless")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/dndsr-riemann-matplotlib")
os.environ.setdefault("MESA_SHADER_CACHE_DIR", "/tmp/dndsr-riemann-mesa")

import numpy as np
from PIL import Image
import pyvista as pv


FIELDS = ("Density", "Pressure")
CACHE_VERSION = 1


def group_ranges(xy, values):
    """Return the scalar ranges at each (x,y), including all z layers."""
    _, inverse = np.unique(np.round(xy, 12), axis=0, return_inverse=True)
    count = int(inverse.max()) + 1
    lower = np.full((count, len(FIELDS)), np.inf)
    upper = np.full((count, len(FIELDS)), -np.inf)
    np.minimum.at(lower, inverse, values[:, :len(FIELDS)])
    np.maximum.at(upper, inverse, values[:, -len(FIELDS):])
    first = np.full(count, len(xy), dtype=np.int64)
    np.minimum.at(first, inverse, np.arange(len(xy)))
    return np.round(xy[first], 12), lower, upper


def prepare_slice(task):
    """Process one saved time; each process owns its VTK state."""
    pvtu_name, z, cache_name = task
    pvtu = Path(pvtu_name)
    cache = Path(cache_name)
    meta_file = cache.with_suffix(".json")
    pieces = [pvtu.parent / element.attrib["Source"] for element in
              ET.parse(pvtu).getroot().findall("./PUnstructuredGrid/Piece")]
    if not pieces:
        raise ValueError(f"No VTU pieces in {pvtu}")
    fingerprint = {"version": CACHE_VERSION, "z": z,
                   "inputs": [[str(p.resolve()), p.stat().st_size,
                               p.stat().st_mtime_ns] for p in [pvtu, *pieces]]}
    if cache.exists() and meta_file.exists():
        metadata = json.loads(meta_file.read_text())
        if metadata.get("fingerprint") == fingerprint:
            return str(cache), metadata["validation"]

    slices, profiles = [], []
    volume_points = 0
    lower, upper = np.full(2, np.inf), np.full(2, -np.inf)
    for piece in pieces:
        reader = pv.get_reader(piece)
        reader.disable_all_point_arrays()
        reader.disable_all_cell_arrays()
        for name in FIELDS:
            if name not in reader.point_array_names:
                raise ValueError(f"Missing point field {name} in {piece}")
            reader.enable_point_array(name)
        mesh = reader.read()
        values = np.column_stack([mesh.point_data[name] for name in FIELDS])
        if not np.isfinite(values).all() or np.any(values <= 0):
            raise ValueError(f"Nonpositive or nonfinite density/pressure: {piece}")
        lower = np.minimum(lower, values.min(axis=0))
        upper = np.maximum(upper, values.max(axis=0))
        volume_points += mesh.n_points
        xy, lo, hi = group_ranges(mesh.points[:, :2], values)
        profiles.append((xy, lo, hi))
        if mesh.bounds[4] <= z <= mesh.bounds[5]:
            cut = mesh.slice(normal=(0, 0, 1), origin=(0, 0, z))
            # DNDS may write an unused unnamed array even with array selection.
            for name in list(cut.point_data):
                if name not in FIELDS:
                    del cut.point_data[name]
            cut.cell_data.clear()
            if cut.n_cells:
                slices.append(cut)
        del mesh, values
    if not slices:
        raise ValueError(f"No cells intersect z={z} in {pvtu}")
    combined = slices[0].merge(slices[1:], merge_points=True, tolerance=1e-10)
    if not isinstance(combined, pv.PolyData):
        combined = combined.extract_surface()
    _, z_lo, z_hi = group_ranges(
        np.concatenate([record[0] for record in profiles]),
        np.concatenate([np.column_stack(record[1:]) for record in profiles]))
    span = z_hi - z_lo
    validation = {
        "input": str(pvtu.resolve()), "pieces": len(pieces),
        "volume_points_including_MPI_copies": volume_points,
        "slice_points": combined.n_points, "slice_cells": combined.n_cells,
        "z": z,
        **{name: {"minimum": float(lower[k]), "maximum": float(upper[k]),
                  "maximum_span_along_z": float(span[:, k].max()),
                  "rms_span_along_z": float(np.sqrt(np.mean(span[:, k] ** 2)))}
           for k, name in enumerate(FIELDS)},
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    combined.save(cache)
    meta_file.write_text(json.dumps({"fingerprint": fingerprint,
                                    "validation": validation}, indent=2) + "\n")
    return str(cache), validation


def locate_ffmpeg(explicit):
    if explicit:
        candidate = Path(explicit)
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        return str(candidate.resolve())
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass
    for parent in (Path("/tmp/dndsr_video_deps"),
                   *sorted(Path("/tmp").glob("vertexfv-video-tools.*"))):
        candidates = sorted((parent / "imageio_ffmpeg/binaries").glob("ffmpeg-*"))
        if candidates:
            return str(candidates[-1])
    raise RuntimeError("ffmpeg is unavailable; provide --ffmpeg /path/to/ffmpeg")


def load_series(path):
    entries = sorted(json.loads(path.read_text())["files"], key=lambda e: e["time"])
    times = np.asarray([entry["time"] for entry in entries], dtype=float)
    if len(entries) < 2 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError(f"Expected two or more strictly increasing saved times: {path}")
    if abs(times[0]) > 1e-12:
        raise ValueError(f"The initial t=0 output is missing: {path}")
    for entry in entries:
        entry["path"] = str((path.parent / entry["name"]).resolve())
        if not Path(entry["path"]).is_file():
            raise FileNotFoundError(entry["path"])
    return entries, times


def add_panel(plotter, column, mesh, field, clim, width, height, cmap="coolwarm"):
    """Give titles, data, and colorbars separate white renderer strips."""
    x0, x1 = column / 2, (column + 1) / 2
    title_index, data_index, bar_index = column * 3, column * 3 + 1, column * 3 + 2
    for index, y0, y1 in ((title_index, .92, 1), (data_index, .14, .92),
                          (bar_index, 0, .14)):
        plotter.subplot(index, 0)
        plotter.renderer.SetViewport(x0, y0, x1, y1)
        plotter.background_color = "white"
        plotter.camera.parallel_projection = True
    plotter.subplot(title_index, 0)
    title = plotter.add_text("", position="lower_edge", color="black", font_size=16)
    plotter.subplot(data_index, 0)
    actor = plotter.add_mesh(mesh, scalars=field, cmap=cmap,
                            clim=clim, show_scalar_bar=False,
                            lighting=False, show_edges=False)
    bounds = mesh.bounds
    cx, cy = (bounds[0] + bounds[1]) / 2, (bounds[2] + bounds[3]) / 2
    domain_width, domain_height = bounds[1] - bounds[0], bounds[3] - bounds[2]
    view_aspect = (width / 2) / (height * .78)
    plotter.camera.position = (cx, cy, bounds[5] + 5 * max(domain_width, domain_height))
    plotter.camera.focal_point = (cx, cy, bounds[4])
    plotter.camera.view_up = (0, 1, 0)
    plotter.camera.parallel_scale = max(domain_height, domain_width / view_aspect) * .61
    positions, labels = [], []
    for x in np.linspace(bounds[0], bounds[1], 3):
        positions.append((x - .015 * domain_width, bounds[2] - .05 * domain_height,
                          bounds[4]))
        labels.append(f"{x:g}")
    for y in np.linspace(bounds[2], bounds[3], 3):
        positions.append((bounds[0] - .075 * domain_width, y - .012 * domain_height,
                          bounds[4]))
        labels.append(f"{y:g}")
    positions.extend(((cx, bounds[2] - .10 * domain_height, bounds[4]),
                      (bounds[0] - .12 * domain_width, cy, bounds[4])))
    labels.extend(("x", "y"))
    plotter.add_point_labels(positions, labels, show_points=False, shape=None,
                            font_size=13, text_color="black", bold=False,
                            always_visible=True)
    plotter.subplot(bar_index, 0)
    plotter.add_scalar_bar(title=field, mapper=actor.mapper, vertical=False,
                          position_x=.2, position_y=.32, width=.6, height=.18,
                          label_font_size=14, title_font_size=16,
                          n_labels=5, fmt="%.3g", color="black")
    return title, actor


def render_case(name, entries, caches, clims, directory, size, z, final_label=True,
                contour_count=25, cmap="coolwarm"):
    directory.mkdir(parents=True, exist_ok=True)
    plotter = pv.Plotter(off_screen=True, window_size=size, shape=(6, 1), border=False)
    mesh = pv.read(caches[0])
    panels = [add_panel(plotter, k, mesh, field, clims[field], *size, cmap=cmap)
              for k, field in enumerate(FIELDS)]
    levels = {field: np.linspace(*clims[field], contour_count + 2)[1:-1]
              for field in FIELDS}
    images = []
    for index, (entry, cache) in enumerate(zip(entries, caches)):
        if index:
            updated = pv.read(cache)
            if (updated.n_points != mesh.n_points or updated.n_cells != mesh.n_cells
                    or not np.array_equal(updated.points, mesh.points)
                    or not np.array_equal(updated.faces, mesh.faces)):
                raise ValueError(f"Slice topology changed at {entry['path']}")
            for field in FIELDS:
                mesh.point_data[field] = updated.point_data[field]
        for column, (field, (title, actor)) in enumerate(zip(FIELDS, panels)):
            suffix = " (final)" if final_label and index == len(entries) - 1 else ""
            title.set_text("lower_edge", f"{name}  |  {field}  |  t = {entry['time']:.5f}{suffix}")
            actor.mapper.scalar_range = clims[field]
            plotter.subplot(column * 3 + 1, 0)
            contour_name = f"contours_{field}"
            plotter.remove_actor(contour_name, reset_camera=False, render=False)
            if contour_count:
                contours = mesh.contour(levels[field], scalars=field,
                                        compute_scalars=False, preference="point")
                if contours.n_cells:
                    # Lift the lines slightly toward the camera to avoid z fighting.
                    contours.translate((0, 0, 1e-5 * (mesh.bounds[1] - mesh.bounds[0])),
                                       inplace=True)
                    plotter.add_mesh(contours, color="black", line_width=.7, opacity=.65,
                                     lighting=False, show_scalar_bar=False, name=contour_name,
                                     reset_camera=False, render=False)
        image = directory / f"{index:06d}.png"
        plotter.render()
        plotter.screenshot(image)
        images.append(image)
        if index % 10 == 0 or index == len(entries) - 1:
            print(f"Rendered {name}: {index + 1}/{len(entries)}, z={z:g}", flush=True)
    plotter.close()
    return images


def encode(ffmpeg, files, output, fps):
    with tempfile.TemporaryDirectory(prefix="ncfv-riemann-frames-") as tmp:
        for index, source in enumerate(files):
            (Path(tmp) / f"{index:06d}.png").symlink_to(source.resolve())
        subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                        "-framerate", str(fps), "-i", str(Path(tmp) / "%06d.png"),
                        "-c:v", "libx264", "-threads", "2", "-crf", "18",
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                        str(output)], check=True)
    subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-xerror",
                    "-threads", "2", "-i", str(output), "-f", "null", "-"],
                   check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/out/NCFV/riemann_barth_400x400x5"))
    parser.add_argument("--output", type=Path, help="Default: ROOT/videos")
    parser.add_argument("--slice-cache-root", type=Path,
                        help="Reuse existing validated slices, e.g. ROOT/videos/slices")
    parser.add_argument("--contours", type=int, default=25,
                        help="Fixed, equally spaced interior contour levels per field; zero disables lines")
    parser.add_argument("--cmap", default="coolwarm",
                        help="Low-to-high color map; default coolwarm is blue-white-red")
    parser.add_argument("--cases", nargs="+", default=["RP1", "RP2", "RP3", "RP4"])
    parser.add_argument("--series-name", default="series.pvtu.series")
    parser.add_argument("--ffmpeg", type=Path)
    parser.add_argument("--z", type=float, default=.00625)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--seconds", type=float, default=12)
    parser.add_argument("--final-hold", type=float, default=1)
    parser.add_argument("--width", type=int, default=1536)
    parser.add_argument("--height", type=int, default=850)
    parser.add_argument("--z-tolerance", type=float, default=1e-7,
                        help="Report z spans above this threshold in validation.json")
    args = parser.parse_args()
    if (args.workers < 1 or args.fps < 1 or args.seconds <= 0 or args.final_hold < 0
            or args.width % 4 or args.height % 2 or args.contours < 0):
        parser.error("Use positive workers/fps/seconds, nonnegative hold/contours, width divisible by 4 and even height")
    if len(args.cases) not in (1, 4):
        parser.error("Select one case, or four cases for the 2x2 overview")
    ffmpeg = locate_ffmpeg(args.ffmpeg)
    output = (args.output or args.root / "videos").resolve()
    slice_root = (args.slice_cache_root or output / "slices").resolve()
    output.mkdir(parents=True, exist_ok=True)
    cases = {}
    validation = {}
    global_limits = {field: [np.inf, -np.inf] for field in FIELDS}
    for name in args.cases:
        entries, times = load_series(args.root / name / args.series_name)
        tasks = [(entry["path"], args.z,
                  str(slice_root / name / f"{k:06d}.vtp"))
                 for k, entry in enumerate(entries)]
        if args.workers == 1:
            results = list(map(prepare_slice, tasks))
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                results = list(executor.map(prepare_slice, tasks))
        validation[name] = [result[1] for result in results]
        for record in validation[name]:
            for field in FIELDS:
                global_limits[field][0] = min(global_limits[field][0], record[field]["minimum"])
                global_limits[field][1] = max(global_limits[field][1], record[field]["maximum"])
        cases[name] = {"entries": entries, "times": times,
                       "caches": [result[0] for result in results]}
        print(f"Prepared {name}: {len(entries)} slices, t_end={times[-1]:g}", flush=True)
    report = {"physical_values": "All checked volume Density/Pressure are finite and positive",
              "z_tolerance": args.z_tolerance, "records": validation,
              "z_variation_exceeds_tolerance": any(
                  record[field]["maximum_span_along_z"] > args.z_tolerance
                  for records in validation.values() for record in records for field in FIELDS)}
    (output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    clims = {field: (float(lo), float(hi if hi > lo else lo + 1e-9))
             for field, (lo, hi) in global_limits.items()}
    t_end = max(case["times"][-1] for case in cases.values())
    timeline = np.linspace(0, t_end, max(2, round(args.fps * args.seconds)))
    timeline = np.r_[timeline, np.full(round(args.fps * args.final_hold), t_end)]
    for name, case in cases.items():
        case["images"] = render_case(name, case["entries"], case["caches"], clims,
                                      output / "frames" / name,
                                      (args.width, args.height), args.z,
                                      contour_count=args.contours, cmap=args.cmap)
        case["indices"] = np.minimum(np.searchsorted(case["times"], timeline + 1e-12,
                                                     side="right") - 1,
                                     len(case["times"]) - 1)
        encode(ffmpeg, [case["images"][k] for k in case["indices"]],
               output / f"{name}_density_pressure.mp4", args.fps)
        shutil.copyfile(case["images"][-1], output / f"{name}_final_density_pressure.png")

    if len(args.cases) == 4:
        overview_dir = output / "frames" / "overview"
        overview_dir.mkdir(parents=True, exist_ok=True)
        composed, overview = {}, []
        for index in range(len(timeline)):
            key = tuple(int(case["indices"][index]) for case in cases.values())
            if key not in composed:
                canvas = Image.new("RGB", (args.width, 2 * args.height), "white")
                for panel, (name, saved_index) in enumerate(zip(args.cases, key)):
                    with Image.open(cases[name]["images"][saved_index]) as frame:
                        density = frame.crop((0, 0, args.width // 2, args.height))
                        canvas.paste(density, ((panel % 2) * (args.width // 2),
                                               (panel // 2) * args.height))
                destination = overview_dir / f"{len(composed):06d}.png"
                canvas.save(destination)
                composed[key] = destination
            overview.append(composed[key])
        encode(ffmpeg, overview, output / "RP1_RP4_density_overview.mp4", args.fps)
        shutil.copyfile(overview[-1], output / "RP1_RP4_final_density_overview.png")
    manifest = {
        "slice_z": args.z, "fps": args.fps,
        "colormap": args.cmap,
        "color_direction": "low blue, high red" if args.cmap == "coolwarm" else "increasing along colormap",
        "contour_count_per_field": args.contours,
        "contour_levels": {field: np.linspace(*clims[field], args.contours + 2)[1:-1].tolist()
                           for field in FIELDS},
        "contour_style": {"color": "black", "line_width": .7, "opacity": .65},
        "slice_cache_root": str(slice_root),
        "color_limits_global_all_cases_all_saved_volume_states": clims,
        "timeline": "Absolute physical time; hold preceding saved output; freeze each case after t_end",
        "temporal_interpolation": False, "final_hold_seconds": args.final_hold,
        "display_times": timeline.tolist(),
        "cases": {name: {"saved_times": case["times"].tolist(),
                         "video_saved_indices": case["indices"].tolist(),
                         "final_time": float(case["times"][-1]),
                         "video": f"{name}_density_pressure.mp4"}
                  for name, case in cases.items()},
        "overview_panel_order": args.cases,
        "overview_video": "RP1_RP4_density_overview.mp4" if len(args.cases) == 4 else None,
    }
    (output / "render_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Videos, final figures, validation and time mapping: {output}", flush=True)
    if report["z_variation_exceeds_tolerance"]:
        print("Notice: z variation exceeds --z-tolerance; see validation.json", flush=True)


if __name__ == "__main__":
    main()
