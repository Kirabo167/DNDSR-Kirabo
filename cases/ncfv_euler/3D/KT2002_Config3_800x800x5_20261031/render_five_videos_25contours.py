"""Create synchronized 25-contour videos for five KT Configuration 3 cases.

Immutable saved states are sliced and checked while the solvers finish. Final
color limits and videos are produced only after every case reaches t=0.3.
No simulation data or solver process is modified by this workflow.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import argparse
import fcntl
import hashlib
import importlib.util
import json
import multiprocessing
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import traceback
from zoneinfo import ZoneInfo

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("EGL_PLATFORM", "surfaceless")
os.environ.setdefault("LP_NUM_THREADS", "2")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/ncfv-five-videos-mpl")
os.environ.setdefault("MESA_SHADER_CACHE_DIR", "/tmp/ncfv-five-videos-mesa")

import numpy as np
from PIL import Image

NAMES = ["SG_BJ_PRES", "SG_SIGN_BUDGET", "FC_IDP_BINARY", "FC_IDP_CONT", "LLF_FIRST_ORDER"]
TARGET = .3
CONTOURS = 25
STOP = False
HELPER = None


def now(): return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def sha256(path):
    with path.open("rb") as f: return hashlib.file_digest(f, "sha256").hexdigest()


def stop(*unused):
    global STOP
    STOP = True


def renderer(path):
    global HELPER
    if HELPER is None:
        spec = importlib.util.spec_from_file_location("ncfv_five_video_helper", path)
        HELPER = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = HELPER
        spec.loader.exec_module(HELPER)
    return HELPER


def prepare(task):
    helper, pvtu, z, cache = task
    return renderer(helper).prepare_slice((pvtu, z, cache))


def video_probe(ffmpeg, video, expected_frames, expected_size):
    # Decode the entire H.264 stream and count decoded frames. This works even
    # on installations whose bundled ffmpeg does not ship a separate ffprobe.
    process = subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "info", "-xerror", "-threads", "2",
                              "-i", str(video), "-progress", "pipe:1", "-f", "null", "-"],
                             text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    counts = re.findall(r"^frame=(\d+)$", process.stdout, re.M)
    if not counts or int(counts[-1]) != expected_frames:
        raise ValueError("Unexpected decoded frame count: " + str(video))
    width, height = expected_size
    if not re.search(rf"\b{width}x{height}\b", process.stderr):
        raise ValueError("Unexpected encoded dimensions: " + str(video))
    return {"frames": expected_frames, "width": width, "height": height,
            "all_frames_decoded": True, "bytes": video.stat().st_size, "sha256": sha256(video)}


def encode(helper, ffmpeg, images, destination, fps):
    temporary = destination.with_name(destination.stem + ".encoding.mp4")
    helper.encode(ffmpeg, images, temporary, fps)
    probe = video_probe(ffmpeg, temporary, len(images), Image.open(images[0]).size)
    temporary.replace(destination)
    return probe


def compose(cases, output, timeline, fps, helper, ffmpeg):
    directory = output / "frames/overview"
    directory.mkdir(parents=True, exist_ok=True)
    # Five columns; density above pressure. Each field occupies a separate
    # square-data panel and retains the method label, source time and colorbar.
    tile = (640, 720)
    size = (tile[0] * 5, tile[1] * 2)
    composed, images = {}, []
    for i in range(len(timeline)):
        key = tuple(int(cases[name]["indices"][i]) for name in NAMES)
        if key not in composed:
            canvas = Image.new("RGB", size, "white")
            for col, (name, saved) in enumerate(zip(NAMES, key)):
                with Image.open(cases[name]["images"][saved]) as frame:
                    w, h = frame.size
                    for row in (0, 1):
                        field = frame.crop((row * (w // 2), 0, (row + 1) * (w // 2), h))
                        canvas.paste(field.resize(tile, Image.Resampling.LANCZOS), (col * tile[0], row * tile[1]))
            path = directory / f"{len(composed):06d}.png"
            canvas.save(path)
            composed[key] = path
        images.append(composed[key])
    destination = output / "KT3_five_limiters_density_pressure_comparison.mp4"
    probe = encode(helper, ffmpeg, images, destination, fps)
    shutil.copy2(images[-1], output / "KT3_five_limiters_final_comparison.png")
    return destination, probe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--renderer", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--z", type=float, default=.02)
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / "render.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    status = {"status": "waiting_for_cases", "started_at": now(), "watcher_pid": os.getpid(),
              "target_physical_time": TARGET, "slice_z": args.z, "contour_levels": CONTOURS,
              "mesh_cells": [800, 800, 5],
              "colormap": "coolwarm", "cases": {}, "videos": {}, "output": str(output),
              "renderer_sha256": sha256(args.renderer.resolve()), "watcher_sha256": sha256(Path(__file__))}
    status_path = output / "video_status.json"
    if status_path.exists() and json.loads(status_path.read_text()).get("status") == "completed":
        raise FileExistsError("Videos are already complete")
    helper = renderer(str(args.renderer.resolve()))
    ffmpeg = helper.locate_ffmpeg(None)
    status["ffmpeg"] = ffmpeg
    prepared = {}
    validation = {}
    save(status_path, status)
    print("WATCHER", now(), "Waiting for all five t=0.3 results; slicing z=", args.z, flush=True)
    try:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            while not STOP:
                tasks, task_keys = [], {}
                complete = True
                for name in NAMES:
                    source_status = json.loads((root / name / "run_status.json").read_text())
                    if source_status["status"] in ("failed", "stopped"):
                        raise RuntimeError("Case did not complete: " + name + " / " + source_status["status"])
                    finished = source_status["status"] == "completed"
                    complete = complete and finished
                    status["cases"][name] = {"status": source_status["status"], "physical_time": source_status.get("physical_time", 0),
                                             "cached_frames": sum(k[0] == name for k in prepared)}
                    series_path = root / name / "series.pvtu.series"
                    if not series_path.exists(): continue
                    try:
                        entries, times = helper.load_series(series_path)
                    except (OSError, ValueError, KeyError, json.JSONDecodeError):
                        if finished: raise
                        continue
                    if finished and (abs(times[-1] - TARGET) > 1e-12 or abs(source_status["physical_time"] - TARGET) > 1e-12):
                        raise ValueError("Completed case lacks its requested final time: " + name)
                    for i, entry in enumerate(entries):
                        key = (name, i)
                        if key in prepared: continue
                        pvtu = Path(entry["path"])
                        if not finished and time.time() - pvtu.stat().st_mtime < 10: continue
                        cache = output / "slices" / name / f"{i:06d}.vtp"
                        task = (str(args.renderer.resolve()), str(pvtu), args.z, str(cache))
                        tasks.append(task)
                        task_keys[str(pvtu)] = (key, finished)
                if tasks:
                    status["status"] = "waiting_and_caching" if not complete else "validating_saved_states"
                    save(status_path, status)
                    futures = {pool.submit(prepare, task): task for task in tasks}
                    for future in as_completed(futures):
                        task = futures[future]
                        key, finished = task_keys[task[1]]
                        try:
                            cache, record = future.result()
                            prepared[key], validation[key] = cache, record
                        except (OSError, ValueError, KeyError) as error:
                            if finished: raise
                            status["last_deferred_frame"] = {"input": task[1], "reason": str(error)}
                        status["cases"][key[0]]["cached_frames"] = sum(k[0] == key[0] for k in prepared)
                        status["updated_at"] = now()
                        save(status_path, status)
                print("PROGRESS", now(), json.dumps(status["cases"]), flush=True)
                if complete: break
                status["status"] = "waiting_for_cases"
                save(status_path, status)
                for unused in range(30):
                    if STOP: break
                    time.sleep(1)
        if STOP: raise InterruptedError("Video workflow stopped")
        cases = {}
        clims = {field: [np.inf, -np.inf] for field in helper.FIELDS}
        for name in NAMES:
            entries, times = helper.load_series(root / name / "series.pvtu.series")
            if abs(times[-1] - TARGET) > 1e-12: raise ValueError("Final time mismatch")
            records = [validation[(name, i)] for i in range(len(entries))]
            for record in records:
                for field in helper.FIELDS:
                    clims[field][0] = min(clims[field][0], record[field]["minimum"])
                    clims[field][1] = max(clims[field][1], record[field]["maximum"])
            cases[name] = {"entries": entries, "times": times, "caches": [prepared[(name, i)] for i in range(len(entries))]}
        clims = {k: [float(lo), float(max(hi, lo + 1e-12))] for k, (lo, hi) in clims.items()}
        save(output / "validation.json", {"physical_values": "All saved volume density/pressure finite and positive",
             "records": {name: [validation[(name, i)] for i in range(len(cases[name]["entries"]))] for name in NAMES}})
        fps = 24
        timeline = np.r_[np.linspace(0, TARGET, 480), np.full(24, TARGET)]
        manifest = {"slice_z": args.z, "resolution_per_case": [1920, 1080], "fps": fps,
                    "colormap": "coolwarm", "color_direction": "low blue, high red", "contour_count_per_field": CONTOURS,
                    "mesh_cells": [800, 800, 5],
                    "contour_levels": {k: np.linspace(*v, CONTOURS + 2)[1:-1].tolist() for k, v in clims.items()},
                    "color_limits_global_all_cases_all_saved_volume_states": clims,
                    "temporal_interpolation": False, "timeline": "Hold preceding saved state on a common absolute physical-time axis",
                    "display_times": timeline.tolist(), "final_hold_seconds": 1., "cases": {}, "overview_panel_order": NAMES,
                    "overview_layout": "Five method columns, density top row, pressure bottom row"}
        for name in NAMES:
            status.update(status="rendering", current_case=name, updated_at=now())
            save(status_path, status)
            case = cases[name]
            case["images"] = helper.render_case(name, case["entries"], case["caches"], clims,
                                                output / "frames" / name, (1920, 1080), args.z,
                                                contour_count=CONTOURS, cmap="coolwarm")
            case["indices"] = np.clip(np.searchsorted(case["times"], timeline + 1e-12, side="right") - 1, 0, len(case["times"]) - 1)
            destination = output / f"{name}_density_pressure.mp4"
            probe = encode(helper, ffmpeg, [case["images"][k] for k in case["indices"]], destination, fps)
            shutil.copy2(case["images"][-1], output / f"{name}_final_density_pressure.png")
            status["videos"][name] = {"path": str(destination), **probe}
            manifest["cases"][name] = {"video": str(destination), "saved_times": case["times"].tolist(),
                                        "video_saved_indices": case["indices"].tolist(), "final_time": float(case["times"][-1])}
            save(status_path, status)
        status.update(status="composing_comparison", updated_at=now())
        save(status_path, status)
        video, probe = compose(cases, output, timeline, fps, helper, ffmpeg)
        status["videos"]["comparison"] = {"path": str(video), **probe}
        manifest["overview_video"] = str(video)
        save(output / "render_manifest.json", manifest)
        status.update(status="completed", finished_at=now(), updated_at=now())
        save(status_path, status)
        print("COMPLETED", json.dumps(status["videos"]), flush=True)
    except InterruptedError as error:
        status.update(status="stopped", reason=str(error), stopped_at=now())
        save(status_path, status)
    except Exception as error:
        status.update(status="failed", reason=repr(error), failed_at=now())
        save(status_path, status)
        traceback.print_exc()
        raise


if __name__ == "__main__": main()
