"""Render actual DNDSR PVTU time steps as a density/pressure MP4.

Reads the native uncompressed inline-binary VTU format without VTK or pybind11.
Uses the original triangular faces on the selected z plane. Spatial shading is
linear; video frames hold saved states, with no temporal interpolation.
"""

import argparse
import base64
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
import matplotlib.tri as mtri
import numpy as np


DTYPES = {"Float64": "f8", "Float32": "f4", "Int64": "i8", "Int32": "i4", "UInt8": "u1"}


def data_array(element):
    dtype = np.dtype("<" + DTYPES[element.attrib["type"]])
    text = "".join(element.itertext()).strip()
    if element.attrib.get("format", "ascii") == "ascii":
        values = np.fromstring(text, dtype=dtype, sep=" ")
    elif element.attrib["format"] == "binary":
        packed = base64.b64decode(text)
        size = int(np.frombuffer(packed, dtype="<u8", count=1)[0])
        if len(packed) != size + 8:
            raise ValueError("VTU binary payload size mismatch")
        values = np.frombuffer(packed, dtype=dtype, offset=8)
    else:
        raise ValueError("Only DNDS inline binary/ASCII VTU is supported")
    components = int(element.attrib.get("NumberOfComponents", 1))
    return values.reshape(-1, components) if components > 1 else values


def read_piece(path):
    root = ET.parse(path).getroot()
    if root.attrib.get("compressor") or root.attrib.get("header_type") != "UInt64":
        raise ValueError("Unexpected compressed VTU/header format")
    piece = root.find("./UnstructuredGrid/Piece")
    coordinates = data_array(piece.find("./Points/DataArray"))
    fields = {a.attrib["Name"]: data_array(a) for a in piece.findall("./PointData/DataArray")}
    rho, pressure = fields["Density"], fields["Pressure"]
    if not (np.isfinite(rho).all() and np.isfinite(pressure).all() and np.all(rho > 0) and np.all(pressure > 0)):
        raise ValueError(f"Nonphysical density/pressure in {path}")
    cells = {a.attrib["Name"]: data_array(a) for a in piece.findall("./Cells/DataArray")}
    if not np.all(cells["types"] == 13) or not np.all(np.diff(np.r_[0, cells["offsets"]]) == 6):
        raise ValueError("This visualization expects the supplied six-node prism mesh")
    return coordinates, rho, pressure, cells["connectivity"].reshape(-1, 6)


def read_slice(path, z):
    pieces = ET.parse(path).getroot().findall("./PUnstructuredGrid/Piece")
    records = [read_piece(path.parent / p.attrib["Source"]) for p in pieces]
    cuts = [np.flatnonzero(np.isclose(r[0][:, 2], z, rtol=0, atol=1e-9)) for r in records]
    xyz = np.concatenate([r[0][cut] for r, cut in zip(records, cuts)])
    rho = np.concatenate([r[1][cut] for r, cut in zip(records, cuts)])
    pressure = np.concatenate([r[2][cut] for r, cut in zip(records, cuts)])
    if len(xyz) == 0:
        raise ValueError(f"No mesh plane at z={z}; this script does not interpolate slices")
    _, first, inverse = np.unique(np.round(xyz, 10), axis=0, return_index=True, return_inverse=True)
    # MPI ghost copies must contain the same density and pressure.
    for field in (rho, pressure):
        if np.max(np.abs(field - field[first][inverse])) > 2e-11:
            raise ValueError("MPI ghost copies disagree on the slice")
    triangles = []
    offset = 0
    for (coordinates, _, _, cells), cut in zip(records, cuts):
        local_to_slice = np.full(len(coordinates), -1, dtype=np.int64)
        local_to_slice[cut] = inverse[offset:offset + len(cut)]
        offset += len(cut)
        for face in (cells[:, :3], cells[:, 3:]):
            mapped = local_to_slice[face]
            triangles.extend(mapped[np.all(mapped >= 0, axis=1)].tolist())
    triangles = np.unique(np.sort(np.asarray(triangles), axis=1), axis=0)
    points = xyz[first]
    ab = points[triangles[:, 1], :2] - points[triangles[:, 0], :2]
    ac = points[triangles[:, 2], :2] - points[triangles[:, 0], :2]
    clockwise = ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0] < 0
    triangles[clockwise] = triangles[clockwise][:, [0, 2, 1]]
    return points, triangles, rho[first], pressure[first]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("series", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--ffmpeg", required=True, type=Path)
    parser.add_argument("--z", type=float, default=2.0)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--motion-seconds", type=float, default=12.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(args.output)
    entries = sorted(json.loads(args.series.read_text())["files"], key=lambda e: e["time"])
    if len(entries) < 2:
        raise ValueError("At least two saved time states are required")
    times = np.array([e["time"] for e in entries])
    if not np.all(np.diff(times) > 0):
        raise ValueError("Saved physical times must be strictly increasing")
    steps = [int(Path(e["name"]).stem.rsplit("_", 1)[1]) for e in entries]
    slices = []
    coordinates, triangles = None, None
    for i, entry in enumerate(entries):
        points, faces, rho, pressure = read_slice(args.series.parent / entry["name"], args.z)
        if coordinates is None:
            coordinates, triangles = points, faces
        elif not np.array_equal(points, coordinates) or not np.array_equal(faces, triangles):
            raise ValueError("Mesh slice changed during the series")
        slices.append((rho, pressure))
        if i % 20 == 0 or i == len(entries) - 1:
            print(f"Loaded state {i + 1}/{len(entries)}, step={steps[i]}, t={times[i]:.8f}", flush=True)
    font = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=font).get_name(),
                         "text.usetex": False, "font.size": 12, "axes.unicode_minus": False,
                         "axes.spines.top": False, "axes.spines.right": False})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tri = mtri.Triangulation(coordinates[:, 0], coordinates[:, 1], triangles)
    # One fixed color scale per field for every time step, rounded outwards.
    scales = []
    for variable in (0, 1):
        minimum = min(float(s[variable].min()) for s in slices)
        maximum = max(float(s[variable].max()) for s in slices)
        scales.append((np.floor(minimum * 20) / 20, np.ceil(maximum * 20) / 20))
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 7.2), dpi=150)
    fig.subplots_adjust(left=0.075, right=0.955, bottom=0.245, top=0.785, wspace=0.29)
    fig.patch.set_facecolor("#f8fafc")
    fig.text(0.065, 0.932, "等熵涡输运  |  密度与压力", fontsize=24, weight="bold", color="#102138")
    fig.text(0.066, 0.879, f"高效全微分积分 · 三维三棱柱网格 · z = {args.z:g} 截面", fontsize=13, color="#475569")
    time_text = fig.text(0.947, 0.893, "", ha="right", fontsize=15, color="#102138")
    artists = []
    for j, (ax, label, cmap) in enumerate(zip(axes, ("密度  ρ", "压力  p"), ("viridis", "magma"))):
        artist = ax.tripcolor(tri, slices[0][j], shading="gouraud", cmap=cmap, norm=Normalize(*scales[j]), rasterized=True)
        artists.append(artist)
        ax.set(xlim=(0, 10), ylim=(0, 10), xlabel="x", ylabel="y", aspect="equal")
        ax.set_title(label, loc="left", fontsize=17, pad=12)
        ax.set_xticks(np.arange(0, 11, 2))
        ax.set_yticks(np.arange(0, 11, 2))
        ax.tick_params(labelsize=10)
        colorbar = fig.colorbar(artist, ax=ax, fraction=0.046, pad=0.04)
        colorbar.ax.tick_params(labelsize=10)
        colorbar.outline.set_visible(False)
    fig.text(0.066, 0.112, "MPI 4  ·  60,832 个三棱柱  ·  无量纲变量  ·  色标固定", fontsize=11, color="#334155")
    fig.text(0.066, 0.068, "实际计算输出：每 2 步一帧；空间线性着色，时间上不插值。", fontsize=10, color="#64748b")
    progress_ax = fig.add_axes([0.065, 0.152, 0.88, 0.009])
    progress_ax.set(xlim=(0, 1), ylim=(0, 1))
    progress_ax.axis("off")
    progress_ax.add_patch(Rectangle((0, 0), 1, 1, color="#dbe4ed"))
    progress = Rectangle((0, 0), 0, 1, color="#197f94")
    progress_ax.add_patch(progress)
    progress_text = fig.text(0.947, 0.112, "", ha="right", fontsize=11, color="#334155")
    motion_frames = int(round(args.motion_seconds * args.fps))
    virtual_times = np.linspace(times[0], times[-1], motion_frames)
    selected = np.searchsorted(times, virtual_times, side="right") - 1
    schedule = np.r_[np.zeros(args.fps, dtype=int), selected, np.full(2 * args.fps, len(times) - 1, dtype=int)]
    if len(np.unique(schedule)) != len(entries):
        raise ValueError("Video duration/fps is too short to show all saved time states")
    command = [str(args.ffmpeg), "-hide_banner", "-loglevel", "error", "-nostdin", "-f", "rawvideo",
               "-pix_fmt", "rgba", "-s", "1920x1080", "-r", str(args.fps), "-i", "pipe:0", "-an",
               "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
               "-movflags", "+faststart", str(args.output)]
    if args.overwrite:
        command.insert(1, "-y")
    log_file = args.output.with_suffix(".ffmpeg.log")
    preview_indices = {0, len(entries) // 2, len(entries) - 1}
    with log_file.open("w") as log:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=log)
        previous = None
        try:
            for frame, state in enumerate(schedule):
                if state != previous:
                    for j in (0, 1):
                        artists[j].set_array(slices[state][j])
                    time_text.set_text(f"t = {times[state]:.5f}    |    第 {steps[state]} 步")
                    progress.set_width((times[state] - times[0]) / (times[-1] - times[0]))
                    progress_text.set_text(f"{times[state] / 10:.1%} 周期")
                    fig.canvas.draw()
                    rgba = np.asarray(fig.canvas.buffer_rgba()).tobytes()
                    if state in preview_indices:
                        fig.savefig(args.output.parent / f"density_pressure_step{steps[state]:06d}.png", dpi=150)
                    previous = state
                process.stdin.write(rgba)
                if frame % args.fps == 0:
                    print(f"Encoded {frame}/{len(schedule)} frames; t={times[state]:.5f}", flush=True)
        finally:
            process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError(log_file.read_text())
    plt.close(fig)
    for variable, label, cmap, name in ((0, "密度 ρ", "viridis", "density_final.png"),
                                        (1, "压力 p", "magma", "pressure_final.png")):
        fig, ax = plt.subplots(figsize=(7, 6), dpi=180, constrained_layout=True)
        plot = ax.tripcolor(tri, slices[-1][variable], shading="gouraud", cmap=cmap, norm=Normalize(*scales[variable]))
        fig.colorbar(plot, ax=ax, label="无量纲")
        ax.set(xlim=(0, 10), ylim=(0, 10), xlabel="x", ylabel="y", aspect="equal",
               title=f"等熵涡 · {label}\nt = {times[-1]:g}，z = {args.z:g}")
        fig.savefig(args.output.parent / name, dpi=180)
        plt.close(fig)
    metadata = {"series": str(args.series), "output": str(args.output), "z": args.z,
                "saved_states": len(entries), "slice_nodes": len(coordinates), "slice_triangles": len(triangles),
                "steps": steps, "times": times.tolist(), "fps": args.fps, "video_frames": len(schedule),
                "duration_seconds": len(schedule) / args.fps, "resolution": [1920, 1080],
                "density_scale": scales[0], "pressure_scale": scales[1],
                "temporal_interpolation": False, "spatial_rendering": "linear Gouraud shading on original mesh triangles",
                "field_semantics": "Density and Pressure from the solver VTK nodal fields derived from dual conservative means"}
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in metadata.items() if k not in ("steps", "times")}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
