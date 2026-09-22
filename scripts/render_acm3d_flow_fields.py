#!/usr/bin/env python3
"""Render 3-D ACM cylinder VTKHDF results as scalar and flow videos.

Products
--------
* five 3-D videos: pressure, u, v, w, and velocity magnitude;
* two five-panel section videos with filled fields and contour lines;
* one instantaneous-streamline video;
* one time-dependent particle-pathline video.

Each VTKHDF snapshot is read once. All products for that snapshot reuse the
same point-interpolated mesh, clipped view, and two sections. Color limits are
fixed across the complete selected sequence, avoiding animation flicker.

The default case is ``CylinderRe3000_coarse_laminar``. Its snapshots are
expected every 50 solver steps, beginning with step 0.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("EGL_PLATFORM", "surfaceless")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/dndsr-matplotlib-cache")
os.environ.setdefault("MESA_SHADER_CACHE_DIR", "/tmp/dndsr-mesa-cache")

# Lazy imports keep ``--help`` available before the rendering environment is
# installed. Type annotations are deferred by ``from __future__`` above.
h5py: Any = None
pv: Any = None


DEFAULT_INPUT = Path(
    "/fs2/home/mrz/DNDSR-Kirabo/data/outACM/"
    "acm3D_CylinderRe3000_coarse_laminar"
)
DEFAULT_PATTERN = "CylinderRe3000_coarse_laminar_*.vtkhdf"

FIELD_SPECS = {
    "p": {"title": "Pressure p", "cmap": "RdBu_r"},
    "u": {"title": "Velocity u", "cmap": "RdBu_r"},
    "v": {"title": "Velocity v", "cmap": "RdBu_r"},
    "w": {"title": "Velocity w", "cmap": "RdBu_r"},
    "speed": {"title": "Velocity magnitude |V|", "cmap": "turbo"},
}
MODES = ("scalar3d", "sections", "streamlines", "pathlines")


@dataclass(frozen=True)
class Snapshot:
    path: Path
    step: int
    time: float | None


@dataclass
class ParticleTrack:
    points: deque[np.ndarray] = field(default_factory=deque)
    speeds: deque[float] = field(default_factory=deque)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render ACM3D p/u/v/w/|V|, sections, streamlines, and pathlines."
    )
    parser.add_argument(
        "input_directory",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"VTKHDF directory (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        help="default: <input>/visualization/acm3d_flow_fields",
    )
    parser.add_argument("--pattern", default=DEFAULT_PATTERN)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=MODES,
        default=MODES,
        help="products to render (default: all)",
    )
    parser.add_argument("--stride", type=int, default=1, help="use every Nth saved snapshot")
    parser.add_argument("--start-step", type=int)
    parser.add_argument("--end-step", type=int)
    parser.add_argument("--max-frames", type=int, help="useful for previews and validation")
    parser.add_argument(
        "--expected-step-spacing",
        type=int,
        default=50,
        help="warn when adjacent filenames are not this many solver steps apart",
    )

    parser.add_argument("--slice-1-axis", choices=("x", "y", "z"), default="z")
    parser.add_argument(
        "--slice-1-position",
        type=float,
        help="default: quarter-span z plane (z=0.375*pi for the supplied mesh)",
    )
    parser.add_argument("--slice-2-axis", choices=("x", "y", "z"), default="z")
    parser.add_argument(
        "--slice-2-position",
        type=float,
        help="default: three-quarter-span z plane (z=1.125*pi for the supplied mesh)",
    )
    parser.add_argument(
        "--view-bounds",
        nargs=6,
        type=float,
        metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
        help="crop shown region; default emphasizes the cylinder and near wake",
    )

    parser.add_argument("--contour-levels", type=int, default=13)
    parser.add_argument("--surface-opacity", type=float, default=0.07)
    parser.add_argument("--slice-opacity", type=float, default=0.68)
    parser.add_argument("--isosurface-opacity", type=float, default=0.48)
    parser.add_argument(
        "--streamline-seeds",
        nargs=2,
        type=int,
        metavar=("NY", "NZ"),
        default=(11, 5),
    )
    parser.add_argument("--pathline-release-every", type=int, default=4)
    parser.add_argument("--pathline-trail-steps", type=int, default=32)
    parser.add_argument(
        "--pathline-dt",
        type=float,
        default=0.25,
        help="advection time between snapshots when no .vtkhdf.series time exists",
    )

    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument(
        "--seconds-per-snapshot",
        type=float,
        default=0.25,
        help="screen time assigned to each saved solver snapshot",
    )
    parser.add_argument(
        "--window-size",
        nargs=2,
        type=int,
        metavar=("WIDTH", "HEIGHT"),
        default=(1920, 1080),
    )
    parser.add_argument("--no-video", action="store_true")
    return parser.parse_args()


def load_dependencies() -> None:
    global h5py, pv
    try:
        import h5py as h5py_module
        import pyvista as pyvista_module
    except ModuleNotFoundError as error:
        raise SystemExit(
            "Rendering requires h5py and pyvista. Install them in the active "
            "environment with: python -m pip install h5py pyvista"
        ) from error
    h5py = h5py_module
    pv = pyvista_module
    pv.OFF_SCREEN = True


def step_from_path(path: Path) -> int:
    match = re.search(r"_(\d+)\.vtkhdf$", path.name)
    if match is None:
        raise ValueError(f"Cannot extract solver step from {path.name}")
    return int(match.group(1))


def series_times(directory: Path) -> dict[str, float]:
    series_files = sorted(directory.glob("*.vtkhdf.series"))
    if len(series_files) != 1:
        return {}
    with series_files[0].open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    return {
        Path(str(entry["name"])).name: float(entry["time"])
        for entry in payload.get("files", [])
        if "name" in entry and "time" in entry
    }


def discover_snapshots(args: argparse.Namespace) -> list[Snapshot]:
    if not args.input_directory.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {args.input_directory}")
    if args.stride <= 0:
        raise ValueError("--stride must be positive")
    paths = sorted(args.input_directory.glob(args.pattern), key=step_from_path)
    times = series_times(args.input_directory)
    snapshots = [
        Snapshot(path, step_from_path(path), times.get(path.name)) for path in paths
    ]
    if args.start_step is not None:
        snapshots = [item for item in snapshots if item.step >= args.start_step]
    if args.end_step is not None:
        snapshots = [item for item in snapshots if item.step <= args.end_step]
    snapshots = snapshots[:: args.stride]
    if args.max_frames is not None:
        if args.max_frames <= 0:
            raise ValueError("--max-frames must be positive")
        snapshots = snapshots[: args.max_frames]
    if not snapshots:
        raise FileNotFoundError(
            f"No selected snapshots match {args.input_directory / args.pattern}"
        )
    expected = args.expected_step_spacing * args.stride
    gaps = [
        (left.step, right.step)
        for left, right in zip(snapshots, snapshots[1:])
        if right.step - left.step != expected
    ]
    if gaps:
        print(f"Warning: unexpected solver-step gaps: {gaps[:8]}", flush=True)
    return snapshots


def validate_args(args: argparse.Namespace) -> None:
    if args.fps <= 0 or args.seconds_per_snapshot <= 0:
        raise ValueError("--fps and --seconds-per-snapshot must be positive")
    if args.contour_levels < 2:
        raise ValueError("--contour-levels must be at least 2")
    if args.pathline_release_every <= 0 or args.pathline_trail_steps < 2:
        raise ValueError("pathline release interval must be positive and trail length >= 2")
    if args.pathline_dt <= 0:
        raise ValueError("--pathline-dt must be positive")
    if any(size <= 0 or size % 2 for size in args.window_size):
        raise ValueError("--window-size entries must be positive even integers")
    for name in ("surface_opacity", "slice_opacity", "isosurface_opacity"):
        value = getattr(args, name)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be in [0, 1]")


def global_color_limits(snapshots: list[Snapshot]) -> dict[str, tuple[float, float]]:
    maxima = {name: 0.0 for name in ("p", "u", "v", "w")}
    maximum_speed = 0.0
    for index, snapshot in enumerate(snapshots):
        with h5py.File(snapshot.path, "r") as handle:
            group = handle["VTKHDF/CellData"]
            if "Pressure" not in group or "Velocity" not in group:
                raise KeyError(f"Missing Pressure or Velocity in {snapshot.path}")
            pressure = np.asarray(group["Pressure"], dtype=float)
            velocity = np.asarray(group["Velocity"], dtype=float)
        if velocity.ndim != 2 or velocity.shape[1] < 3:
            raise ValueError(
                f"Velocity in {snapshot.path} has shape {velocity.shape}, expected (n, 3)"
            )
        arrays = {
            "p": pressure,
            "u": velocity[:, 0],
            "v": velocity[:, 1],
            "w": velocity[:, 2],
        }
        for name, values in arrays.items():
            maxima[name] = max(maxima[name], float(np.nanmax(np.abs(values))))
        maximum_speed = max(
            maximum_speed,
            float(np.nanmax(np.linalg.norm(velocity[:, :3], axis=1))),
        )
        print(
            f"Color scan {index + 1}/{len(snapshots)}: {snapshot.path.name}",
            flush=True,
        )
    epsilon = 1.0e-12
    limits = {
        name: (-max(1.02 * value, epsilon), max(1.02 * value, epsilon))
        for name, value in maxima.items()
    }
    limits["speed"] = (0.0, max(1.02 * maximum_speed, epsilon))
    return limits


def add_derived_point_fields(mesh: Any) -> Any:
    point_mesh = mesh.cell_data_to_point_data(pass_cell_data=False)
    pressure = np.asarray(point_mesh.point_data["Pressure"], dtype=float)
    velocity = np.asarray(point_mesh.point_data["Velocity"], dtype=float)
    point_mesh.point_data["p"] = pressure
    point_mesh.point_data["u"] = velocity[:, 0]
    point_mesh.point_data["v"] = velocity[:, 1]
    point_mesh.point_data["w"] = velocity[:, 2]
    point_mesh.point_data["speed"] = np.linalg.norm(velocity[:, :3], axis=1)
    point_mesh.set_active_vectors("Velocity")
    return point_mesh


def default_view_bounds(bounds: tuple[float, ...]) -> tuple[float, ...]:
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    lx, ly = xmax - xmin, ymax - ymin
    cy = 0.5 * (ymin + ymax)
    # For the supplied domain this gives approximately x=[-4.4, 20.2],
    # y=[-8, 8], retaining the cylinder and near wake without the far field.
    return (
        xmin + 0.06 * lx,
        xmin + 0.47 * lx,
        cy - 0.16 * ly,
        cy + 0.16 * ly,
        zmin,
        zmax,
    )


def resolved_slice_position(
    mesh_bounds: tuple[float, ...],
    axis: str,
    requested: float | None,
    default_fraction: float,
) -> float:
    axis_index = "xyz".index(axis)
    if requested is not None:
        return requested
    lower = mesh_bounds[2 * axis_index]
    upper = mesh_bounds[2 * axis_index + 1]
    return lower + default_fraction * (upper - lower)


def make_slice(mesh: Any, axis: str, position: float) -> Any:
    origin = list(mesh.center)
    origin["xyz".index(axis)] = position
    section = mesh.slice(normal=axis, origin=origin)
    if section.n_cells == 0:
        raise ValueError(
            f"Slice {axis}={position:g} does not intersect mesh bounds {mesh.bounds}"
        )
    return section


def scalar_levels(dataset: Any, name: str, count: int) -> np.ndarray:
    values = np.asarray(dataset.point_data[name], dtype=float)
    low, high = float(np.nanmin(values)), float(np.nanmax(values))
    span = max(abs(low), abs(high), 1.0)
    if not np.isfinite(low + high) or high - low <= 1.0e-12 * span:
        return np.empty(0)
    return np.linspace(low, high, count + 2)[1:-1]


def add_contour_lines(
    plotter: Any,
    section: Any,
    name: str,
    count: int,
    opacity: float = 0.62,
    line_width: float = 1.25,
) -> None:
    levels = scalar_levels(section, name, count)
    if levels.size == 0:
        return
    lines = section.contour(isosurfaces=levels, scalars=name)
    if lines.n_cells:
        plotter.add_mesh(
            lines,
            color="#202020",
            opacity=opacity,
            line_width=line_width,
            show_scalar_bar=False,
        )


def snapshot_label(snapshot: Snapshot) -> str:
    if snapshot.time is None:
        return f"step = {snapshot.step}"
    return f"step = {snapshot.step},  t = {snapshot.time:.6g}"


def configure_3d_scene(plotter: Any, view_bounds: tuple[float, ...]) -> None:
    plotter.show_grid(
        bounds=view_bounds,
        xtitle="x / D",
        ytitle="y / D",
        ztitle="z / D",
        color="black",
        font_size=16,
        n_xlabels=5,
        n_ylabels=5,
        n_zlabels=4,
    )
    plotter.add_axes(color="black", line_width=2)
    plotter.view_isometric()
    plotter.reset_camera()
    plotter.camera.zoom(1.08)


def render_scalar3d(
    focused: Any,
    outer_surface: Any,
    sections: tuple[Any, Any],
    name: str,
    limits: dict[str, tuple[float, float]],
    snapshot: Snapshot,
    output: Path,
    args: argparse.Namespace,
    view_bounds: tuple[float, ...],
) -> None:
    spec = FIELD_SPECS[name]
    plotter = pv.Plotter(off_screen=True, window_size=args.window_size)
    plotter.background_color = "white"
    plotter.add_mesh(
        outer_surface,
        color="#b8c1cc",
        opacity=args.surface_opacity,
        show_scalar_bar=False,
    )
    for section_index, section in enumerate(sections):
        plotter.add_mesh(
            section,
            scalars=name,
            cmap=spec["cmap"],
            clim=limits[name],
            opacity=args.slice_opacity,
            show_scalar_bar=section_index == 0,
            scalar_bar_args={
                "title": spec["title"],
                "vertical": False,
                "position_x": 0.25,
                "position_y": 0.055,
                "width": 0.50,
                "height": 0.065,
                "label_font_size": 18,
                "title_font_size": 21,
                "fmt": "%.3g",
                "color": "black",
            },
        )
        add_contour_lines(plotter, section, name, args.contour_levels)

    levels = scalar_levels(focused, name, max(5, args.contour_levels // 2))
    if levels.size:
        isosurfaces = focused.contour(isosurfaces=levels, scalars=name)
        if isosurfaces.n_cells:
            plotter.add_mesh(
                isosurfaces,
                scalars=name,
                cmap=spec["cmap"],
                clim=limits[name],
                opacity=args.isosurface_opacity,
                show_scalar_bar=False,
            )
    plotter.add_text(
        f"3-D {spec['title']}\n{snapshot_label(snapshot)}",
        position="upper_edge",
        font_size=18,
        color="black",
    )
    configure_3d_scene(plotter, view_bounds)
    plotter.screenshot(output)
    plotter.close()


def camera_for_section(
    section: Any, axis: str, panel_aspect: float
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...], float]:
    normal_index = "xyz".index(axis)
    horizontal_index, vertical_index = {
        "x": (1, 2),
        "y": (0, 2),
        "z": (0, 1),
    }[axis]
    center = np.asarray(section.center, dtype=float)
    normal, up = np.zeros(3), np.zeros(3)
    normal[normal_index] = 1.0
    if axis == "y":
        # Looking from negative y keeps +x pointing to screen right.
        normal[normal_index] = -1.0
    up[vertical_index] = 1.0
    width = section.bounds[2 * horizontal_index + 1] - section.bounds[2 * horizontal_index]
    height = section.bounds[2 * vertical_index + 1] - section.bounds[2 * vertical_index]
    scale = 0.54 * max(height, width / panel_aspect)
    return tuple(center + max(section.length, 1.0) * normal), tuple(center), tuple(up), scale


def render_section_panel(
    section: Any,
    axis: str,
    position: float,
    limits: dict[str, tuple[float, float]],
    snapshot: Snapshot,
    output: Path,
    args: argparse.Namespace,
) -> None:
    width, height = args.window_size
    panel_aspect = (width / 3.0) / (height / 2.0)
    camera_position, focal_point, view_up, parallel_scale = camera_for_section(
        section, axis, panel_aspect
    )
    plotter = pv.Plotter(
        shape=(2, 3), off_screen=True, window_size=args.window_size, border=False
    )
    for panel, (name, spec) in enumerate(FIELD_SPECS.items()):
        plotter.subplot(panel // 3, panel % 3)
        plotter.background_color = "white"
        plotter.add_mesh(
            section,
            scalars=name,
            cmap=spec["cmap"],
            clim=limits[name],
            opacity=0.96,
            show_scalar_bar=True,
            scalar_bar_args={
                "title": spec["title"],
                "vertical": True,
                "position_x": 0.84,
                "position_y": 0.13,
                "width": 0.075,
                "height": 0.68,
                "label_font_size": 12,
                "title_font_size": 14,
                "fmt": "%.3g",
                "color": "black",
            },
        )
        add_contour_lines(plotter, section, name, args.contour_levels)
        plotter.camera.parallel_projection = True
        plotter.camera.position = camera_position
        plotter.camera.focal_point = focal_point
        plotter.camera.view_up = view_up
        plotter.camera.parallel_scale = parallel_scale
        plotter.show_bounds(
            location="outer",
            ticks="outside",
            xtitle="x / D",
            ytitle="y / D",
            ztitle="z / D",
            font_size=11,
            color="black",
            n_xlabels=4,
            n_ylabels=4,
            n_zlabels=3,
        )
        plotter.add_text(spec["title"], position="upper_edge", font_size=14, color="black")

    plotter.subplot(1, 2)
    plotter.background_color = "white"
    plotter.add_text(
        f"Section {axis} = {position:.6g}\n\n"
        f"{snapshot_label(snapshot)}\n\n"
        "Filled field + contour lines\n"
        "Fixed color limits over all snapshots",
        position="upper_left",
        font_size=18,
        color="black",
    )
    plotter.screenshot(output)
    plotter.close()


def seed_points(view_bounds: tuple[float, ...], ny: int, nz: int) -> np.ndarray:
    if ny < 2 or nz < 2:
        raise ValueError("--streamline-seeds NY NZ must both be at least 2")
    xmin, xmax, ymin, ymax, zmin, zmax = view_bounds
    seed_x = xmin + 0.035 * (xmax - xmin)
    ys = np.linspace(ymin + 0.12 * (ymax - ymin), ymax - 0.12 * (ymax - ymin), ny)
    zs = np.linspace(zmin + 0.08 * (zmax - zmin), zmax - 0.08 * (zmax - zmin), nz)
    return np.asarray([(seed_x, y, z) for y in ys for z in zs], dtype=float)


def instantaneous_streamlines(
    focused: Any, seeds: np.ndarray, view_bounds: tuple[float, ...]
) -> Any:
    source = pv.PolyData(seeds)
    maximum_length = 1.25 * (view_bounds[1] - view_bounds[0])
    return focused.streamlines_from_source(
        source,
        vectors="Velocity",
        integration_direction="forward",
        integrator_type=45,
        initial_step_length=0.04,
        step_unit="l",
        min_step_length=0.005,
        max_step_length=0.20,
        max_steps=2500,
        terminal_speed=1.0e-8,
        max_length=maximum_length,
        compute_vorticity=False,
        interpolator_type="point",
    )


def add_flow_background(
    plotter: Any,
    outer_surface: Any,
    sections: tuple[Any, Any],
    speed_limits: tuple[float, float],
    args: argparse.Namespace,
) -> None:
    plotter.add_mesh(
        outer_surface,
        color="#c5cbd3",
        opacity=max(args.surface_opacity, 0.045),
        show_scalar_bar=False,
    )
    for section in sections:
        plotter.add_mesh(
            section,
            scalars="speed",
            cmap="Greys",
            clim=speed_limits,
            opacity=0.18,
            show_scalar_bar=False,
        )
        add_contour_lines(
            plotter,
            section,
            "speed",
            max(7, args.contour_levels // 2),
            opacity=0.28,
            line_width=0.8,
        )


def render_streamlines(
    focused: Any,
    outer_surface: Any,
    sections: tuple[Any, Any],
    seeds: np.ndarray,
    limits: dict[str, tuple[float, float]],
    snapshot: Snapshot,
    output: Path,
    args: argparse.Namespace,
    view_bounds: tuple[float, ...],
) -> None:
    lines = instantaneous_streamlines(focused, seeds, view_bounds)
    plotter = pv.Plotter(off_screen=True, window_size=args.window_size)
    plotter.background_color = "white"
    add_flow_background(plotter, outer_surface, sections, limits["speed"], args)
    if lines.n_cells:
        tubes = lines.tube(radius=0.025, n_sides=8)
        plotter.add_mesh(
            tubes,
            scalars="speed",
            cmap="turbo",
            clim=limits["speed"],
            opacity=0.94,
            scalar_bar_args={
                "title": "Velocity magnitude |V|",
                "vertical": False,
                "position_x": 0.25,
                "position_y": 0.055,
                "width": 0.50,
                "height": 0.065,
                "fmt": "%.3g",
                "color": "black",
            },
        )
    plotter.add_points(seeds, color="#111111", point_size=4, render_points_as_spheres=True)
    plotter.add_text(
        f"Instantaneous streamlines\n{snapshot_label(snapshot)}",
        position="upper_edge",
        font_size=20,
        color="black",
    )
    configure_3d_scene(plotter, view_bounds)
    plotter.screenshot(output)
    plotter.close()


def sample_velocity(mesh: Any, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(points) == 0:
        return np.empty((0, 3)), np.empty(0, dtype=bool)
    sampled = pv.PolyData(points).sample(
        mesh,
        pass_cell_data=False,
        pass_point_data=True,
        locator="static_cell",
    )
    valid = np.asarray(sampled.point_data["vtkValidPointMask"], dtype=bool)
    velocity = np.asarray(sampled.point_data["Velocity"], dtype=float)
    return velocity, valid


def advance_tracks(
    tracks: list[ParticleTrack],
    mesh: Any,
    dt: float,
    view_bounds: tuple[float, ...],
    trail_steps: int,
) -> list[ParticleTrack]:
    if not tracks:
        return tracks
    starts = np.asarray([track.points[-1] for track in tracks])
    velocity_0, valid_0 = sample_velocity(mesh, starts)
    midpoint = starts + 0.5 * dt * velocity_0
    velocity_mid, valid_mid = sample_velocity(mesh, midpoint)
    ends = starts + dt * velocity_mid
    xmin, xmax, ymin, ymax, zmin, zmax = view_bounds
    inside = (
        (ends[:, 0] >= xmin)
        & (ends[:, 0] <= xmax)
        & (ends[:, 1] >= ymin)
        & (ends[:, 1] <= ymax)
        & (ends[:, 2] >= zmin)
        & (ends[:, 2] <= zmax)
    )
    retained: list[ParticleTrack] = []
    for index, track in enumerate(tracks):
        if not (valid_0[index] and valid_mid[index] and inside[index]):
            continue
        track.points.append(ends[index].copy())
        track.speeds.append(float(np.linalg.norm(velocity_mid[index])))
        while len(track.points) > trail_steps:
            track.points.popleft()
            track.speeds.popleft()
        retained.append(track)
    return retained


def release_tracks(tracks: list[ParticleTrack], seeds: np.ndarray, mesh: Any) -> None:
    velocity, valid = sample_velocity(mesh, seeds)
    for point, value, is_valid in zip(seeds, velocity, valid):
        if is_valid:
            tracks.append(
                ParticleTrack(
                    points=deque([point.copy()]),
                    speeds=deque([float(np.linalg.norm(value))]),
                )
            )


def tracks_to_polydata(tracks: list[ParticleTrack]) -> Any | None:
    drawable = [track for track in tracks if len(track.points) >= 2]
    if not drawable:
        return None
    points: list[np.ndarray] = []
    speeds: list[float] = []
    cells: list[np.ndarray] = []
    offset = 0
    for track in drawable:
        count = len(track.points)
        points.extend(track.points)
        speeds.extend(track.speeds)
        cells.append(np.concatenate(([count], np.arange(offset, offset + count))))
        offset += count
    polydata = pv.PolyData(np.asarray(points))
    polydata.lines = np.concatenate(cells).astype(np.int64)
    polydata.point_data["ParticleSpeed"] = np.asarray(speeds)
    return polydata


def render_pathlines(
    outer_surface: Any,
    sections: tuple[Any, Any],
    tracks: list[ParticleTrack],
    limits: dict[str, tuple[float, float]],
    snapshot: Snapshot,
    output: Path,
    args: argparse.Namespace,
    view_bounds: tuple[float, ...],
) -> None:
    paths = tracks_to_polydata(tracks)
    plotter = pv.Plotter(off_screen=True, window_size=args.window_size)
    plotter.background_color = "white"
    add_flow_background(plotter, outer_surface, sections, limits["speed"], args)
    if paths is not None:
        tubes = paths.tube(radius=0.040, n_sides=8)
        plotter.add_mesh(
            tubes,
            scalars="ParticleSpeed",
            cmap="turbo",
            clim=limits["speed"],
            opacity=0.93,
            scalar_bar_args={
                "title": "Particle speed |V|",
                "vertical": False,
                "position_x": 0.25,
                "position_y": 0.055,
                "width": 0.50,
                "height": 0.065,
                "fmt": "%.3g",
                "color": "black",
            },
        )
    if tracks:
        heads = np.asarray([track.points[-1] for track in tracks])
        plotter.add_points(
            heads,
            color="#101010",
            point_size=5,
            render_points_as_spheres=True,
        )
    plotter.add_text(
        f"Time-dependent particle pathlines\n{snapshot_label(snapshot)}",
        position="upper_edge",
        font_size=20,
        color="black",
    )
    configure_3d_scene(plotter, view_bounds)
    plotter.screenshot(output)
    plotter.close()


def encode_video(
    frame_paths: list[Path], output: Path, fps: int, seconds_per_snapshot: float
) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        try:
            import imageio_ffmpeg
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "Video encoding requires ffmpeg or imageio-ffmpeg. Install the "
                "fallback with: python -m pip install imageio-ffmpeg"
            ) from error
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    repetitions = max(1, round(fps * seconds_per_snapshot))
    with tempfile.TemporaryDirectory(prefix="dndsr_acm3d_video_") as temp_name:
        temporary = Path(temp_name)
        sequence_index = 0
        for frame_path in frame_paths:
            for _ in range(repetitions):
                (temporary / f"{sequence_index:06d}.png").symlink_to(
                    frame_path.resolve()
                )
                sequence_index += 1
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-framerate",
                str(fps),
                "-i",
                str(temporary / "%06d.png"),
                "-c:v",
                "libx264",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(output),
            ],
            check=True,
        )


def product_names(args: argparse.Namespace) -> list[str]:
    names: list[str] = []
    if "scalar3d" in args.modes:
        names.extend(f"scalar3d_{name}" for name in FIELD_SPECS)
    if "sections" in args.modes:
        names.extend(("section_1", "section_2"))
    if "streamlines" in args.modes:
        names.append("streamlines")
    if "pathlines" in args.modes:
        names.append("pathlines")
    return names


def main() -> None:
    args = parse_args()
    validate_args(args)
    load_dependencies()
    snapshots = discover_snapshots(args)
    output_directory = args.output_directory or (
        args.input_directory / "visualization" / "acm3d_flow_fields"
    )
    frames_root = output_directory / "frames"
    videos_directory = output_directory / "videos"
    names = product_names(args)
    frame_paths = {name: [] for name in names}
    for name in names:
        (frames_root / name).mkdir(parents=True, exist_ok=True)
    if not args.no_video:
        videos_directory.mkdir(parents=True, exist_ok=True)

    print(f"Selected {len(snapshots)} snapshots", flush=True)
    limits = global_color_limits(snapshots)
    print(f"Fixed color limits: {json.dumps(limits)}", flush=True)

    first_mesh = pv.read(snapshots[0].path)
    original_bounds = tuple(float(value) for value in first_mesh.bounds)
    view_bounds = (
        tuple(args.view_bounds)
        if args.view_bounds is not None
        else default_view_bounds(original_bounds)
    )
    slice_specs = (
        (
            args.slice_1_axis,
            resolved_slice_position(
                original_bounds,
                args.slice_1_axis,
                args.slice_1_position,
                default_fraction=0.25,
            ),
        ),
        (
            args.slice_2_axis,
            resolved_slice_position(
                original_bounds,
                args.slice_2_axis,
                args.slice_2_position,
                default_fraction=0.75,
            ),
        ),
    )
    del first_mesh
    seeds = seed_points(view_bounds, *args.streamline_seeds)
    # Pathlines use a less dense rake than streamlines to keep long animations clear.
    pathline_seeds = seed_points(
        view_bounds,
        max(5, args.streamline_seeds[0] // 2),
        max(3, args.streamline_seeds[1] // 2),
    )
    tracks: list[ParticleTrack] = []
    previous_time: float | None = None

    for frame_index, snapshot in enumerate(snapshots):
        print(
            f"Rendering snapshot {frame_index + 1}/{len(snapshots)}: {snapshot.path.name}",
            flush=True,
        )
        mesh = add_derived_point_fields(pv.read(snapshot.path))
        focused = mesh.clip_box(view_bounds, invert=False)
        outer_surface = focused.extract_surface(algorithm="dataset_surface")
        sections = tuple(
            make_slice(focused, axis, position) for axis, position in slice_specs
        )

        if "scalar3d" in args.modes:
            for name in FIELD_SPECS:
                product = f"scalar3d_{name}"
                output = frames_root / product / f"frame_{frame_index:06d}.png"
                render_scalar3d(
                    focused,
                    outer_surface,
                    sections,
                    name,
                    limits,
                    snapshot,
                    output,
                    args,
                    view_bounds,
                )
                frame_paths[product].append(output)

        if "sections" in args.modes:
            for section_index, (section, (axis, position)) in enumerate(
                zip(sections, slice_specs), start=1
            ):
                product = f"section_{section_index}"
                output = frames_root / product / f"frame_{frame_index:06d}.png"
                render_section_panel(
                    section,
                    axis,
                    position,
                    limits,
                    snapshot,
                    output,
                    args,
                )
                frame_paths[product].append(output)

        if "streamlines" in args.modes:
            output = frames_root / "streamlines" / f"frame_{frame_index:06d}.png"
            render_streamlines(
                focused,
                outer_surface,
                sections,
                seeds,
                limits,
                snapshot,
                output,
                args,
                view_bounds,
            )
            frame_paths["streamlines"].append(output)

        if "pathlines" in args.modes:
            if frame_index:
                if snapshot.time is not None and previous_time is not None:
                    particle_dt = snapshot.time - previous_time
                    if particle_dt <= 0:
                        raise ValueError("Series times must be strictly increasing")
                else:
                    particle_dt = args.pathline_dt
                tracks = advance_tracks(
                    tracks,
                    focused,
                    particle_dt,
                    view_bounds,
                    args.pathline_trail_steps,
                )
            if frame_index % args.pathline_release_every == 0:
                release_tracks(tracks, pathline_seeds, focused)
            output = frames_root / "pathlines" / f"frame_{frame_index:06d}.png"
            render_pathlines(
                outer_surface,
                sections,
                tracks,
                limits,
                snapshot,
                output,
                args,
                view_bounds,
            )
            frame_paths["pathlines"].append(output)
            previous_time = snapshot.time

        del sections, outer_surface, focused, mesh

    manifest = {
        "input_directory": str(args.input_directory.resolve()),
        "source_files": [str(item.path.resolve()) for item in snapshots],
        "steps": [item.step for item in snapshots],
        "times": [item.time for item in snapshots],
        "modes": list(args.modes),
        "products": names,
        "original_bounds": original_bounds,
        "view_bounds": view_bounds,
        "sections": [
            {"axis": axis, "position": position} for axis, position in slice_specs
        ],
        "color_limits": limits,
        "contour_levels": args.contour_levels,
        "opacity": {
            "surface": args.surface_opacity,
            "section_in_3d": args.slice_opacity,
            "isosurface": args.isosurface_opacity,
        },
        "fps": args.fps,
        "seconds_per_snapshot": args.seconds_per_snapshot,
        "pathline_dt_fallback": args.pathline_dt,
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    manifest_path = output_directory / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    if not args.no_video:
        for index, name in enumerate(names):
            print(f"Encoding video {index + 1}/{len(names)}: {name}", flush=True)
            encode_video(
                frame_paths[name],
                videos_directory / f"{name}.mp4",
                args.fps,
                args.seconds_per_snapshot,
            )
    print(f"Frames: {frames_root}")
    print(f"Manifest: {manifest_path}")
    if not args.no_video:
        print(f"Videos: {videos_directory}")


if __name__ == "__main__":
    main()
