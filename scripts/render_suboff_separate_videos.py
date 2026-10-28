#!/usr/bin/env python3
"""Render separate pressure, speed, u, v, and w SUBOFF history videos."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np
from PIL import Image

os.environ.setdefault("EGL_PLATFORM", "surfaceless")
import pyvista as pv

from render_suboff_domain_history import (
    FIELDS,
    attach_fields,
    global_ranges,
    load_frames,
    sha256,
    step_from_name,
)


FIELD_SLUGS = {
    "Pressure": "pressure",
    "SpeedMagnitude": "speed_magnitude",
    "VelocityU": "u",
    "VelocityV": "v",
    "VelocityW": "w",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_directory", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument(
        "--series", default="SUBOFF_AFF1_SST.retained.vtkhdf.series"
    )
    parser.add_argument("--prefix", default="SUBOFF_AFF1_Re12000000_Ncells990914")
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--reynolds", type=float, default=12_000_000.0)
    parser.add_argument("--cell-count", type=int, default=990_914)
    parser.add_argument("--mpi-ranks", type=int, default=64)
    parser.add_argument("--contours", action="store_true")
    parser.add_argument("--contour-levels", type=int, default=17)
    parser.add_argument("--preview-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def configure_camera(plotter: pv.Plotter, plane: str) -> None:
    center_x = 4.0
    if plane == "xy":
        plotter.camera.position = (center_x, 0.0, 20.0)
        plotter.camera.focal_point = (center_x, 0.0, 0.0)
        plotter.camera.up = (0.0, 1.0, 0.0)
    else:
        plotter.camera.position = (center_x, 20.0, 0.0)
        plotter.camera.focal_point = (center_x, 0.0, 0.0)
        plotter.camera.up = (0.0, 0.0, 1.0)
    plotter.camera.parallel_projection = True
    viewport_aspect = plotter.window_size[0] / 5.0 / plotter.window_size[1]
    plotter.camera.parallel_scale = 1.04 * max(2.0, 5.0 / viewport_aspect)


def render_strip_frame(
    path: Path,
    _time_value: float,
    output: Path,
    limits: dict[str, tuple[float, float]],
    args: argparse.Namespace,
) -> None:
    mesh = pv.read(path)
    if mesh.n_cells != args.cell_count:
        raise ValueError(
            f"Unexpected cell count in {path}: {mesh.n_cells}, expected {args.cell_count}"
        )
    attach_fields(mesh)
    xy = mesh.slice(normal=(0.0, 0.0, 1.0), origin=(0.0, 0.0, 0.0))
    xz = mesh.slice(normal=(0.0, 1.0, 0.0), origin=(0.0, 0.0, 0.0))
    xy = xy.cell_data_to_point_data(pass_cell_data=False)
    xz = xz.cell_data_to_point_data(pass_cell_data=False)

    pv.OFF_SCREEN = True
    plotter = pv.Plotter(
        off_screen=True,
        window_size=(args.width * len(FIELDS), args.height),
        shape=(1, len(FIELDS)),
        border=False,
    )
    step = step_from_name(path.name)
    for column, (field, title, cmap) in enumerate(FIELDS):
        section = xz if field == "VelocityW" else xy
        plane = "xz" if field == "VelocityW" else "xy"
        plotter.subplot(0, column)
        plotter.background_color = "white"
        plotter.add_mesh(
            section,
            scalars=field,
            preference="point",
            cmap=cmap,
            clim=limits[field],
            show_edges=False,
            lighting=False,
            scalar_bar_args={
                "title": "",
                "vertical": False,
                "position_x": 0.25,
                "position_y": 0.04,
                "width": 0.50,
                "height": 0.045,
                "label_font_size": 13,
                "title_font_size": 13,
                "color": "black",
                "fmt": "%.3g",
            },
        )
        if args.contours:
            contour_values = np.linspace(
                limits[field][0], limits[field][1], args.contour_levels + 2
            )[1:-1]
            contours = section.contour(isosurfaces=contour_values, scalars=field)
            if contours.n_points:
                plotter.add_mesh(
                    contours,
                    color="black",
                    line_width=0.8,
                    opacity=0.55,
                    lighting=False,
                )
        outline = section.extract_feature_edges(
            boundary_edges=True,
            feature_edges=False,
            manifold_edges=False,
            non_manifold_edges=False,
        )
        plotter.add_mesh(outline, color="black", line_width=1.2, lighting=False)
        plotter.add_text(title, position="upper_edge", font_size=16, color="black")
        plotter.add_text(
            f"Re_L = {args.reynolds:.3e}  |  cells = {args.cell_count:,}  |  "
            f"step = {step}",
            position="lower_right",
            font_size=10,
            color="black",
        )
        configure_camera(plotter, plane)

    plotter.screenshot(output)
    plotter.close()


def encode_crop(
    frame_directory: Path,
    output: Path,
    fps: int,
    width: int,
    height: int,
    column: int,
) -> None:
    import imageio_ffmpeg

    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(frame_directory / "frame_%05d.png"),
        "-vf",
        f"crop={width}:{height}:{column * width}:0",
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
    if args.fps < 1 or args.width < 2 or args.height < 2:
        raise SystemExit("fps, width, and height must be positive")
    if args.width % 2 or args.height % 2:
        raise SystemExit("H.264 yuv420p requires even width and height")
    if args.contour_levels < 1:
        raise SystemExit("--contour-levels must be positive")

    frames = load_frames(args.input_directory, args.series)
    render_frames = [frames[0], frames[-1]] if args.preview_only else frames
    args.output_directory.mkdir(parents=True, exist_ok=True)
    output_suffix = "_contours" if args.contours else ""
    outputs = {
        field: args.output_directory
        / f"{args.prefix}_steps{step_from_name(frames[0][0].name)}_"
        f"{step_from_name(frames[-1][0].name)}_{FIELD_SLUGS[field]}"
        f"{output_suffix}.mp4"
        for field, _, _ in FIELDS
    }
    if not args.force:
        existing = [path for path in outputs.values() if path.exists()]
        if existing:
            raise SystemExit(f"Output already exists: {existing[0]}; use --force")

    limits = global_ranges(frames)
    with tempfile.TemporaryDirectory(
        prefix="suboff_separate_fields_", dir=args.output_directory
    ) as temporary:
        frame_directory = Path(temporary)
        for index, (path, time_value) in enumerate(render_frames):
            frame_path = frame_directory / f"frame_{index:05d}.png"
            render_strip_frame(path, time_value, frame_path, limits, args)
            print(
                f"Rendered {index + 1}/{len(render_frames)}: "
                f"step {step_from_name(path.name)}",
                flush=True,
            )

        final_strip = Image.open(
            frame_directory / f"frame_{len(render_frames) - 1:05d}.png"
        )
        for column, (field, _, _) in enumerate(FIELDS):
            preview = outputs[field].with_suffix(".png")
            final_strip.crop(
                (
                    column * args.width,
                    0,
                    (column + 1) * args.width,
                    args.height,
                )
            ).save(preview)
            if not args.preview_only:
                encode_crop(
                    frame_directory,
                    outputs[field],
                    args.fps,
                    args.width,
                    args.height,
                    column,
                )

    manifest = {
        "source_series": str((args.input_directory / args.series).resolve()),
        "frame_count": len(frames),
        "first_step": step_from_name(frames[0][0].name),
        "last_step": step_from_name(frames[-1][0].name),
        "fps": args.fps,
        "resolution": [args.width, args.height],
        "reynolds": args.reynolds,
        "cell_count": args.cell_count,
        "mpi_ranks": args.mpi_ranks,
        "contours": args.contours,
        "contour_levels": args.contour_levels if args.contours else 0,
        "field_ranges": {key: list(value) for key, value in limits.items()},
        "outputs": {},
    }
    for field, title, _ in FIELDS:
        path = outputs[field]
        manifest["outputs"][FIELD_SLUGS[field]] = {
            "field": field,
            "title": title,
            "plane": "y=0" if field == "VelocityW" else "z=0",
            "video": str(path.resolve()) if not args.preview_only else None,
            "preview": str(path.with_suffix(".png").resolve()),
            "sha256": sha256(path) if not args.preview_only else None,
        }
    manifest_path = args.output_directory / (
        f"{args.prefix}_steps100_5000{output_suffix}_manifest.json"
    )
    with manifest_path.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
