#!/usr/bin/env python3
"""Render pressure and near-wall speed on a rotating SUBOFF surface."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np

os.environ.setdefault("EGL_PLATFORM", "surfaceless")
import pyvista as pv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--reynolds", type=float, default=12_000_000.0)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--preview-only", action="store_true")
    return parser.parse_args()


def select_body(mesh: pv.DataSet) -> pv.DataSet:
    surface = mesh.extract_surface(algorithm="dataset_surface")
    connected = surface.connectivity()
    region_ids = np.unique(np.asarray(connected.cell_data["RegionId"]))
    if region_ids.size < 2:
        raise ValueError("Expected distinct SUBOFF and far-field surface regions")

    candidates = []
    for region_id in region_ids:
        region = connected.threshold(
            [float(region_id), float(region_id)], scalars="RegionId"
        )
        extents = np.array(
            [
                region.bounds.x_max - region.bounds.x_min,
                region.bounds.y_max - region.bounds.y_min,
                region.bounds.z_max - region.bounds.z_min,
            ]
        )
        candidates.append((float(np.linalg.norm(extents)), region))
    return min(candidates, key=lambda item: item[0])[1]


def add_panel(
    plotter: pv.Plotter,
    row: int,
    body: pv.DataSet,
    field: str,
    title: str,
    cmap: str,
    clim: tuple[float, float],
) -> None:
    plotter.subplot(row, 0)
    plotter.background_color = "white"
    plotter.add_mesh(
        body,
        scalars=field,
        preference="cell",
        cmap=cmap,
        clim=clim,
        show_edges=False,
        smooth_shading=True,
        ambient=0.35,
        diffuse=0.75,
        specular=0.12,
        specular_power=18.0,
        scalar_bar_args={
            "title": title,
            "vertical": False,
            "position_x": 0.27,
            "position_y": 0.045,
            "width": 0.46,
            "height": 0.065,
            "label_font_size": 15,
            "title_font_size": 17,
            "color": "black",
            "fmt": "%.3g",
        },
    )
    plotter.add_text(title, position="upper_edge", font_size=17, color="black")
    plotter.add_axes(
        xlabel="x", ylabel="y", zlabel="z", color="black", line_width=2
    )
    plotter.camera.parallel_projection = True


def set_camera(plotter: pv.Plotter, row: int, center: np.ndarray, angle: float,
               parallel_scale: float) -> None:
    yaw = math.radians(28.0) * math.sin(angle)
    elevation = math.radians(12.0 + 5.0 * math.sin(2.0 * angle))
    radial = np.array(
        [
            math.sin(yaw) * math.cos(elevation),
            math.cos(yaw) * math.cos(elevation),
            math.sin(elevation),
        ]
    )
    world_up = np.array([0.0, 0.0, 1.0])
    up = world_up - np.dot(world_up, radial) * radial
    up /= np.linalg.norm(up)
    plotter.subplot(row, 0)
    plotter.camera.position = tuple(center + 10.0 * radial)
    plotter.camera.focal_point = tuple(center)
    plotter.camera.up = tuple(up)
    plotter.camera.parallel_scale = parallel_scale
    plotter.camera.clipping_range = (0.01, 30.0)


def encode_video(frame_directory: Path, output: Path, fps: int) -> None:
    import imageio_ffmpeg

    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(frame_directory / "frame_%04d.png"),
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    if args.frames < 1 or args.fps < 1 or args.width < 2 or args.height < 2:
        raise SystemExit("frames, fps, width, and height must be positive")
    if args.width % 2 or args.height % 2:
        raise SystemExit("H.264 yuv420p requires even width and height")
    if not args.input.is_file():
        raise SystemExit(f"Input field does not exist: {args.input}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    mesh = pv.read(args.input)
    body = select_body(mesh)
    body.cell_data["SpeedMagnitude"] = np.linalg.norm(
        np.asarray(body.cell_data["Velocity"]), axis=1
    )

    pressure = np.asarray(body.cell_data["Pressure"])
    speed = np.asarray(body.cell_data["SpeedMagnitude"])
    pressure_limit = 1.02 * float(np.nanmax(np.abs(pressure)))
    speed_limit = 1.02 * float(np.nanmax(speed))

    pv.OFF_SCREEN = True
    plotter = pv.Plotter(
        off_screen=True,
        window_size=(args.width, args.height),
        shape=(2, 1),
        border=False,
    )
    add_panel(
        plotter,
        0,
        body,
        "Pressure",
        "Surface-adjacent pressure p",
        "coolwarm",
        (-pressure_limit, pressure_limit),
    )
    add_panel(
        plotter,
        1,
        body,
        "SpeedMagnitude",
        "Surface-adjacent speed |V|",
        "turbo",
        (0.98 * float(np.nanmin(speed)), speed_limit),
    )

    bounds = body.bounds
    center = np.array(
        [
            0.5 * (bounds.x_min + bounds.x_max),
            0.5 * (bounds.y_min + bounds.y_max),
            0.5 * (bounds.z_min + bounds.z_max),
        ]
    )
    body_length = bounds.x_max - bounds.x_min
    panel_aspect = args.width / (0.5 * args.height)
    parallel_scale = 0.5 * body_length * 1.10 / panel_aspect

    metadata = (
        f"SUBOFF AFF-1  |  Re = {args.reynolds:.3e}  |  "
        f"Ncells = {mesh.n_cells:,}  |  pseudo step = {args.step}"
    )
    for row in (0, 1):
        plotter.subplot(row, 0)
        plotter.add_text(metadata, position="lower_right", font_size=10, color="black")

    frame_count = 1 if args.preview_only else args.frames
    with tempfile.TemporaryDirectory(
        prefix="suboff_surface_frames_", dir=args.output.parent
    ) as temporary:
        frame_directory = Path(temporary)
        for frame in range(frame_count):
            angle = 2.0 * math.pi * frame / args.frames
            for row in (0, 1):
                set_camera(plotter, row, center, angle, parallel_scale)
            frame_path = frame_directory / f"frame_{frame:04d}.png"
            plotter.render()
            plotter.screenshot(frame_path)
            if frame % max(1, frame_count // 10) == 0:
                print(f"Rendered frame {frame + 1}/{frame_count}", flush=True)

        preview = args.output.with_suffix(".png")
        preview.write_bytes((frame_directory / "frame_0000.png").read_bytes())
        if not args.preview_only:
            encode_video(frame_directory, args.output, args.fps)

    plotter.close()
    print(f"body cells: {body.n_cells}")
    print(f"pressure range: {float(np.nanmin(pressure)):.9g} to {float(np.nanmax(pressure)):.9g}")
    print(f"speed range: {float(np.nanmin(speed)):.9g} to {float(np.nanmax(speed)):.9g}")
    print(preview)
    if not args.preview_only:
        print(args.output)


if __name__ == "__main__":
    main()
