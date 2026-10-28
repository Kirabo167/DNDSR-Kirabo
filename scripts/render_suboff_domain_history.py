#!/usr/bin/env python3
"""Render a five-field SUBOFF ACM domain-history dashboard video."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

import h5py
import numpy as np

os.environ.setdefault("EGL_PLATFORM", "surfaceless")
import pyvista as pv


STEP_PATTERN = re.compile(r"_(\d+)\.vtkhdf$")
FIELDS = (
    ("Pressure", "Pressure  p  (z = 0)", "coolwarm"),
    ("SpeedMagnitude", "Speed magnitude  |V|  (z = 0)", "turbo"),
    ("VelocityU", "Streamwise velocity  u  (z = 0)", "turbo"),
    ("VelocityV", "Transverse velocity  v  (z = 0)", "coolwarm"),
    ("VelocityW", "Vertical velocity  w  (y = 0)", "coolwarm"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_directory", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--series",
        default="SUBOFF_AFF1_SST.retained.vtkhdf.series",
        help="series JSON within input_directory",
    )
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--width", type=int, default=2560)
    parser.add_argument("--height", type=int, default=1440)
    parser.add_argument("--reynolds", type=float, default=12_000_000.0)
    parser.add_argument("--cell-count", type=int, default=990_914)
    parser.add_argument("--mpi-ranks", type=int, default=64)
    parser.add_argument("--preview-only", action="store_true")
    parser.add_argument(
        "--max-frames",
        type=int,
        help="render an evenly sampled subset; the last frame is always included",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def step_from_name(name: str) -> int:
    match = STEP_PATTERN.search(name)
    if match is None:
        raise ValueError(f"Cannot parse pseudo step from {name}")
    return int(match.group(1))


def load_frames(input_directory: Path, series_name: str) -> list[tuple[Path, float]]:
    series_path = input_directory / series_name
    with series_path.open("r", encoding="utf-8") as stream:
        document = json.load(stream)
    frames = []
    missing = []
    for entry in document.get("files", []):
        path = input_directory / Path(str(entry["name"])).name
        if path.is_file():
            frames.append((path, float(entry["time"])))
        else:
            missing.append(path.name)
    frames.sort(key=lambda item: item[1])
    if not frames:
        raise ValueError(f"No existing VTK-HDF files referenced by {series_path}")
    if any(right[1] <= left[1] for left, right in zip(frames, frames[1:])):
        raise ValueError("Series times must be strictly increasing")
    if missing:
        print(f"Ignored {len(missing)} missing series entries", flush=True)
    return frames


def sample_frames(
    frames: list[tuple[Path, float]], max_frames: int | None
) -> list[tuple[Path, float]]:
    if max_frames is None or max_frames >= len(frames):
        return frames
    if max_frames < 2:
        raise ValueError("--max-frames must be at least 2")
    indices = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
    return [frames[index] for index in np.unique(indices)]


def global_ranges(frames: list[tuple[Path, float]]) -> dict[str, tuple[float, float]]:
    pressure_abs = 0.0
    speed_max = 0.0
    u_min, u_max = np.inf, -np.inf
    v_abs, w_abs = 0.0, 0.0
    for index, (path, _) in enumerate(frames, start=1):
        with h5py.File(path, "r") as handle:
            pressure = np.asarray(handle["VTKHDF/CellData/Pressure"][:], dtype=float)
            velocity = np.asarray(handle["VTKHDF/CellData/Velocity"][:], dtype=float)
        if not np.isfinite(pressure).all() or not np.isfinite(velocity).all():
            raise ValueError(f"Non-finite field value in {path}")
        pressure_abs = max(pressure_abs, float(np.max(np.abs(pressure))))
        speed_max = max(speed_max, float(np.max(np.linalg.norm(velocity, axis=1))))
        u_min = min(u_min, float(np.min(velocity[:, 0])))
        u_max = max(u_max, float(np.max(velocity[:, 0])))
        v_abs = max(v_abs, float(np.max(np.abs(velocity[:, 1]))))
        w_abs = max(w_abs, float(np.max(np.abs(velocity[:, 2]))))
        if index == 1 or index % 10 == 0 or index == len(frames):
            print(f"Scanned field range {index}/{len(frames)}", flush=True)
    u_padding = max(0.02 * (u_max - u_min), 1.0e-6)
    return {
        "Pressure": (-1.02 * pressure_abs, 1.02 * pressure_abs),
        "SpeedMagnitude": (0.0, max(1.02 * speed_max, 1.0e-6)),
        "VelocityU": (u_min - u_padding, u_max + u_padding),
        "VelocityV": (-max(1.02 * v_abs, 1.0e-6), max(1.02 * v_abs, 1.0e-6)),
        "VelocityW": (-max(1.02 * w_abs, 1.0e-6), max(1.02 * w_abs, 1.0e-6)),
    }


def attach_fields(mesh: pv.DataSet) -> None:
    velocity = np.asarray(mesh.cell_data["Velocity"])
    mesh.cell_data["SpeedMagnitude"] = np.linalg.norm(velocity, axis=1)
    mesh.cell_data["VelocityU"] = velocity[:, 0]
    mesh.cell_data["VelocityV"] = velocity[:, 1]
    mesh.cell_data["VelocityW"] = velocity[:, 2]


def configure_camera(plotter: pv.Plotter, plane: str, bounds: pv.BoundsTuple) -> None:
    center_x = 4.0
    if plane == "xy":
        center_vertical = 0.0
        plotter.camera.position = (center_x, center_vertical, 20.0)
        plotter.camera.focal_point = (center_x, center_vertical, 0.0)
        plotter.camera.up = (0.0, 1.0, 0.0)
        vertical_extent = 4.0
    else:
        center_vertical = 0.0
        plotter.camera.position = (center_x, 20.0, center_vertical)
        plotter.camera.focal_point = (center_x, 0.0, center_vertical)
        plotter.camera.up = (0.0, 0.0, 1.0)
        vertical_extent = 4.0
    horizontal_extent = 10.0
    viewport_aspect = (plotter.window_size[0] / 3.0) / (plotter.window_size[1] / 2.0)
    plotter.camera.parallel_projection = True
    plotter.camera.parallel_scale = 1.04 * max(
        0.5 * vertical_extent, 0.5 * horizontal_extent / viewport_aspect
    )


def add_field_panel(
    plotter: pv.Plotter,
    row: int,
    column: int,
    section: pv.DataSet,
    field: str,
    title: str,
    cmap: str,
    limits: tuple[float, float],
    plane: str,
) -> None:
    plotter.subplot(row, column)
    plotter.background_color = "white"
    plotter.add_mesh(
        section,
        scalars=field,
        preference="point",
        cmap=cmap,
        clim=limits,
        show_edges=False,
        lighting=False,
        scalar_bar_args={
            "title": "",
            "vertical": False,
            "position_x": 0.22,
            "position_y": 0.025,
            "width": 0.56,
            "height": 0.045,
            "label_font_size": 11,
            "title_font_size": 12,
            "color": "black",
            "fmt": "%.3g",
        },
    )
    outline = section.extract_feature_edges(
        boundary_edges=True,
        feature_edges=False,
        manifold_edges=False,
        non_manifold_edges=False,
    )
    plotter.add_mesh(outline, color="black", line_width=1.0, lighting=False)
    plotter.add_text(title, position="upper_edge", font_size=13, color="black")
    configure_camera(plotter, plane, section.bounds)


def render_frame(
    path: Path,
    time_value: float,
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
        window_size=(args.width, args.height),
        shape=(2, 3),
        border=False,
    )
    locations = ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1))
    for (field, title, cmap), (row, column) in zip(FIELDS, locations):
        section = xz if field == "VelocityW" else xy
        plane = "xz" if field == "VelocityW" else "xy"
        add_field_panel(
            plotter, row, column, section, field, title, cmap, limits[field], plane
        )

    step = step_from_name(path.name)
    plotter.subplot(1, 2)
    plotter.background_color = "white"
    plotter.add_text(
        "DARPA SUBOFF AFF-1\n\n"
        "ACM + k-omega SST\n"
        f"Re_L = {args.reynolds:.3e}\n"
        f"cells = {args.cell_count:,}\n"
        f"MPI ranks = {args.mpi_ranks}\n\n"
        f"steady pseudo step = {step}\n"
        f"series time = {time_value:g}\n\n"
        "Freestream direction: +x\n"
        "p, |V|, u, v: z = 0 plane\n"
        "w: y = 0 plane",
        position="upper_left",
        font_size=16,
        color="black",
    )
    plotter.camera.parallel_projection = True
    plotter.screenshot(output)
    plotter.close()


def encode_video(frame_directory: Path, output: Path, fps: int) -> None:
    import imageio_ffmpeg

    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(frame_directory / "frame_%05d.png"),
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if args.fps < 1 or args.width < 2 or args.height < 2:
        raise SystemExit("fps, width, and height must be positive")
    if args.width % 2 or args.height % 2:
        raise SystemExit("H.264 yuv420p requires even width and height")
    if args.output.exists() and not args.force:
        raise SystemExit(f"Output already exists: {args.output}; use --force to replace")

    all_frames = load_frames(args.input_directory, args.series)
    frames = sample_frames(all_frames, 2 if args.preview_only else args.max_frames)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    limits = global_ranges(all_frames)

    with tempfile.TemporaryDirectory(
        prefix="suboff_domain_history_", dir=args.output.parent
    ) as temporary:
        frame_directory = Path(temporary)
        for index, (path, time_value) in enumerate(frames):
            frame_path = frame_directory / f"frame_{index:05d}.png"
            render_frame(path, time_value, frame_path, limits, args)
            print(
                f"Rendered {index + 1}/{len(frames)}: step {step_from_name(path.name)}",
                flush=True,
            )
        preview = args.output.with_suffix(".png")
        shutil.copy2(frame_directory / f"frame_{len(frames) - 1:05d}.png", preview)
        if not args.preview_only:
            encode_video(frame_directory, args.output, args.fps)

    metadata = {
        "video": str(args.output.resolve()) if not args.preview_only else None,
        "preview": str(preview.resolve()),
        "source_series": str((args.input_directory / args.series).resolve()),
        "source_frame_count": len(all_frames),
        "rendered_frame_count": len(frames),
        "first_step": step_from_name(frames[0][0].name),
        "last_step": step_from_name(frames[-1][0].name),
        "fps": args.fps,
        "resolution": [args.width, args.height],
        "reynolds": args.reynolds,
        "cell_count": args.cell_count,
        "mpi_ranks": args.mpi_ranks,
        "field_ranges": {key: list(value) for key, value in limits.items()},
        "planes": {
            "Pressure": "z=0",
            "SpeedMagnitude": "z=0",
            "VelocityU": "z=0",
            "VelocityV": "z=0",
            "VelocityW": "y=0",
        },
    }
    if not args.preview_only:
        metadata["sha256"] = sha256(args.output)
    metadata_path = args.output.with_suffix(".json")
    with metadata_path.open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)
        stream.write("\n")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
