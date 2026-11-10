#!/usr/bin/env python3
"""Render pinned GPU SUBOFF history with fixed ranges and paired isoline movies."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ("Speed", "VorticityMagnitude", "PressureRelative", "TurbulenceK")


def plan_history(run, cutoff, last_source=None, last_time=None):
    """Follow actual restart ancestry; truncate each parent at its restart time."""
    run = Path(run).resolve()
    frames, ancestry, visited = {}, [], set()
    ceiling = float("inf") if last_time is None else float(last_time)
    physics = None
    while run not in visited:
        visited.add(run)
        status = json.loads((run / "status.json").read_text())
        config = json.loads(Path(status["config"]).read_text())
        settings = config["acmSettings"]
        signature = (settings["rho0"], tuple(settings["farFieldValue"]), settings["dynamicViscosity"],
                     config["turbulenceSettings"]["model"], settings["pressureStorage"])
        if physics is None:
            physics = signature
        if signature != physics or settings["pressureStorage"] != "PhysicalP":
            raise ValueError("Restart history contains incompatible physical settings")
        series = sorted((run / "fields").glob("*.series"))
        if len(series) != 1:
            raise ValueError(f"Expected exactly one output series in {run}")
        for entry in json.loads(series[0].read_text())["files"]:
            source = (run / "fields" / entry["name"]).resolve()
            time = float(entry["time"])
            if time > ceiling + 1e-9 or not source.is_file() or source.stat().st_mtime > cutoff.timestamp():
                continue
            match = re.search(r"_(\d+)\.vtkhdf$", source.name)
            if match is None or source.stat().st_size == 0:
                raise ValueError(f"Invalid saved snapshot: {source}")
            # The newer phase owns a duplicated restart time.
            frames.setdefault(round(time, 9), dict(source=str(source), physical_time=time,
                step=int(match[1]), run=str(run), bytes=source.stat().st_size,
                completed_utc=datetime.fromtimestamp(source.stat().st_mtime, timezone.utc).isoformat()))
        restart = config.get("restartSettings", {})
        ancestry.append(dict(run=str(run), config=status["config"], upper_physical_time=None if ceiling == float("inf") else ceiling,
                             mesh_file=config["meshSettings"]["meshFile"], ranks=status["ranks"]))
        if not restart.get("flowFile"):
            break
        ceiling = min(ceiling, float(restart["physicalTime"]))
        source_run = config.get("caseNotes", {}).get("restartSourceRun")
        run = ((ROOT / "build-gpu-solvers" / source_run).resolve() if source_run else
               (ROOT / "build-gpu-solvers" / restart["flowFile"]).resolve().parent.parent)
    else:
        raise ValueError("Cycle in restart ancestry")
    ordered = [frames[t] for t in sorted(frames)]
    if len(ordered) < 2:
        raise ValueError("At least two complete saved snapshots are needed")
    if last_source is not None and Path(ordered[-1]["source"]) != Path(last_source).resolve():
        raise ValueError("Video final snapshot does not match the pinned report snapshot")
    return dict(cutoff=cutoff.isoformat(), frames=ordered, ancestry=ancestry,
                rho=physics[0], p_inf=physics[1][3], fields=list(FIELDS),
                selection="Complete series entries only; actual restart ancestry; parent histories truncated at restart; newer duplicate retained")


def geometry_fingerprint(handle):
    digest = hashlib.sha256()
    group = handle["VTKHDF"]
    for key in ("Points", "Connectivity", "Offsets", "Types", "NumberOfPoints", "NumberOfCells"):
        data = group[key]
        digest.update(f"{key}:{data.shape}:{data.dtype}".encode())
        digest.update(data[()].tobytes())
    return digest.hexdigest()


def prepare_sections(plan, directory):
    import h5py
    from render_suboff_flow_report import pv, np
    minimum = {field: float("inf") for field in FIELDS}
    maximum = {field: -float("inf") for field in FIELDS}
    geometry, current_run, grid, body, bounds = {}, None, None, None, None
    for index, frame in enumerate(plan["frames"]):
        if frame["run"] != current_run:
            grid = pv.read(frame["source"])
            grid.clear_data()
            if grid.n_cells != 3914184:
                raise ValueError(f"Unexpected SUBOFF mesh size: {grid.n_cells}")
            with h5py.File(frame["source"]) as handle:
                geometry[frame["run"]] = geometry_fingerprint(handle)
                point_counts = handle["VTKHDF/NumberOfPoints"][()]
                cell_counts = handle["VTKHDF/NumberOfCells"][()]
            current_run = frame["run"]
            if body is None:
                surface = grid.extract_surface(algorithm="dataset_surface").connectivity()
                bodies = []
                for region in np.unique(surface.cell_data["RegionId"]):
                    part = surface.threshold([float(region), float(region)], scalars="RegionId")
                    extents = np.asarray(part.bounds)[1::2] - np.asarray(part.bounds)[::2]
                    if 2 < extents[0] < 6 and max(extents[1:]) < 3:
                        bodies.append(part)
                if not bodies:
                    raise ValueError("Cannot identify the submarine surface")
                body = pv.merge(bodies).extract_surface(algorithm="dataset_surface").clean()
                bounds = list(grid.bounds)
        with h5py.File(frame["source"]) as handle:
            if (not np.array_equal(point_counts, handle["VTKHDF/NumberOfPoints"][()])
                    or not np.array_equal(cell_counts, handle["VTKHDF/NumberOfCells"][()])):
                raise ValueError("Mesh partition sizes changed within one run")
            for field in ("Velocity", "Pressure", "TurbulenceK"):
                values = handle[f"VTKHDF/CellData/{field}"][()]
                if values.shape[0] != grid.n_cells or not np.isfinite(values).all():
                    raise ValueError(f"Invalid saved field {field} in {frame['source']}")
                grid.cell_data[field] = values
        point = grid.cell_data_to_point_data(pass_cell_data=False)
        point = point.compute_derivative(scalars="Velocity", gradient=False, vorticity="Vorticity",
                                         preference="point", faster=False)
        point.point_data["Speed"] = np.linalg.norm(point["Velocity"], axis=1)
        point.point_data["VorticityMagnitude"] = np.linalg.norm(point["Vorticity"], axis=1)
        point.point_data["PressureRelative"] = point["Pressure"] - plan["p_inf"]
        section = point.slice(normal=(0, 0, 1), origin=(0, 0, 0))
        section.clear_cell_data()
        for field in list(section.point_data):
            if field not in FIELDS:
                del section.point_data[field]
        for field in FIELDS:
            values = np.asarray(section[field])
            if not np.isfinite(values).all():
                raise ValueError(f"Non-finite section {field}")
            minimum[field] = min(minimum[field], float(np.min(values)))
            maximum[field] = max(maximum[field], float(np.max(values)))
        section.save(directory / f"section_{index:06d}.vtp")
        del point, section
        print(f"Prepared {index+1}/{len(plan['frames'])}: step={frame['step']}, t={frame['physical_time']:.4f}", flush=True)
    pmax = max(abs(minimum["PressureRelative"]), abs(maximum["PressureRelative"]), 1e-12) * 1.00001
    wmax = max(maximum["VorticityMagnitude"], 1e-12) * 1.00001
    kmin = max(minimum["TurbulenceK"], 1e-14)
    ranges = {"Speed": [0., max(maximum["Speed"] * 1.00001, 1e-12)],
              "VorticityMagnitude": [max(wmax * 1e-5, 1e-16), wmax],
              "PressureRelative": [-pmax, pmax],
              "TurbulenceK": [kmin, max(maximum["TurbulenceK"] * 1.00001, kmin * 1.0001)]}
    return body, bounds, ranges, geometry


def render_movies(plan, output, width, height, fps):
    import shutil
    import imageio_ffmpeg
    from render_suboff_flow_report import (pv, np, plt, smooth_section_display,
        section_panel, outline_segments, draw_isolines)
    if width % 2 or height % 2:
        raise ValueError("H.264 image dimensions must be even")
    output.mkdir(parents=True, exist_ok=True)
    (output / "frame_manifest.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n")
    videos = {}
    with tempfile.TemporaryDirectory(prefix=".evolution-work-", dir=output) as temporary:
        temporary = Path(temporary)
        body, bounds, ranges, geometry = prepare_sections(plan, temporary)
        outline = outline_segments(body)
        full = (bounds[0], bounds[1], bounds[2], bounds[3])
        (output / "display_ranges.json").write_text(json.dumps(ranges, indent=2) + "\n")
        folders = {}
        for region in ("near_field", "whole_domain"):
            for variant in ("without_isolines", "with_isolines"):
                key = region + "_" + variant
                folders[key] = temporary / key
                folders[key].mkdir()
        for index, frame in enumerate(plan["frames"]):
            section = pv.read(temporary / f"section_{index:06d}.vtp")
            for near in (True, False):
                region = "near_field" if near else "whole_domain"
                source = (section.clip_box((-.5, 7., -1., 1., -.001, .001), invert=False)
                          .extract_surface(algorithm="dataset_surface")) if near else section
                data, triangles = smooth_section_display(source, ranges, 7.5 if near else full[1] - full[0])
                fig, axes = plt.subplots(2, 2, figsize=(width / 120, height / 120), dpi=120)
                for ax, field in zip(axes.ravel(), FIELDS):
                    section_panel(fig, ax, data, triangles, field, ranges, full, near, outline,
                                  compact=True, smooth=True)
                title = fig.suptitle("", fontsize=20, y=.985)
                fig.subplots_adjust(top=.88, bottom=.11, left=.065, right=.94, hspace=.48, wspace=.32)
                fig.text(.5, .025, "全片固定色标；同一平滑显示场提取等值线；涡量为三维后处理旋度。内迭代未达收敛标准。",
                         ha="center", fontsize=10, color="#6c4b27")
                for isolines in (False, True):
                    variant = "with_isolines" if isolines else "without_isolines"
                    if isolines:
                        for ax, field in zip(axes.ravel(), FIELDS):
                            draw_isolines(ax, triangles, np.asarray(data[field]), ranges, field,
                                          (-.5, 7., -1., 1.) if near else full, outline)
                    title.set_text(f"SUBOFF · {'潜艇近场' if near else '整个求解域'} · 侧视 z=0 · {'有等值线' if isolines else '无等值线'}\n"
                                   f"第 {frame['step']} 步，物理时间 t={frame['physical_time']:.4f}")
                    target = folders[region + "_" + variant] / f"frame_{index:06d}.png"
                    fig.savefig(target, dpi=120, facecolor="white")
                    if index in (0, len(plan["frames"]) - 1):
                        shutil.copyfile(target, output / f"{region}_{variant}_{'first' if index == 0 else 'last'}.png")
                plt.close(fig)
                del data, triangles
            print(f"Rendered {index+1}/{len(plan['frames'])}", flush=True)
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        for key, folder in folders.items():
            target = output / f"SUBOFF_evolution_{key}.mp4"
            partial = target.with_suffix(".partial.mp4")
            command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-framerate", str(fps),
                       "-i", str(folder / "frame_%06d.png"), "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                       "-pix_fmt", "yuv420p", "-threads", "2", "-movflags", "+faststart", str(partial)]
            subprocess.run(command, check=True, timeout=1800)
            count, duration = imageio_ffmpeg.count_frames_and_secs(str(partial))
            if count != len(plan["frames"]):
                raise ValueError(f"Encoded frame count mismatch in {partial}: {count}")
            partial.replace(target)
            videos[key] = dict(path=str(target), frames=count, duration_seconds=duration,
                               width=width, height=height, fps=fps, bytes=target.stat().st_size)
            print(f"Encoded {target.name}: {count} frames", flush=True)
    result = dict(state="completed", videos=videos, display_ranges=ranges, geometry_sha256_by_run=geometry,
                  cutoff=plan["cutoff"], first_frame=plan["frames"][0], last_frame=plan["frames"][-1],
                  temporal_processing="Actual saved snapshots in physical-time order; no synthesized intermediate flow fields",
                  field_processing="Cell-to-point interpolation; full 3D curl before slicing; bounded cubic display interpolation; paired variants share identical display samples",
                  color_processing="Global ranges over all selected saved frames; fixed across time, views, and isoline variants",
                  surface="z=0 side section; x horizontal, y vertical; original solid mask retained",
                  temporary_frames_removed=True)
    (output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, help="Optional preview subset; scheduled runs use all saved frames")
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    args = parser.parse_args()
    if args.fps <= 0 or args.width < 640 or args.height < 360:
        parser.error("Invalid video dimensions or frame rate")
    plan = json.loads(args.manifest.read_text())
    if args.max_frames is not None:
        if args.max_frames < 2:
            parser.error("Preview requires at least two frames")
        frames = plan["frames"]
        count = min(args.max_frames, len(frames))
        indices = sorted({round(i * (len(frames) - 1) / (count - 1)) for i in range(count)})
        plan["frames"] = [frames[i] for i in indices]
        plan["preview_subset"] = True
    for key, value in dict(EGL_PLATFORM="surfaceless", VTK_DEFAULT_OPENGL_WINDOW="vtkEGLRenderWindow",
                           LIBGL_ALWAYS_SOFTWARE="1", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1",
                           VTK_SMP_MAX_THREADS="2").items():
        os.environ[key] = value
    result = render_movies(plan, args.output.resolve(), args.width, args.height, args.fps)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
