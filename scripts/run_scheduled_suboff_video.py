#!/usr/bin/env python3
"""One-shot cutoff-pinned flow-evolution movie action for the SUBOFF timer."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from run_scheduled_suboff_report import ROOT, choose_snapshot
from render_suboff_gpu_evolution import plan_history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--due", required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    due = datetime.fromisoformat(args.due)
    if due.tzinfo is None:
        parser.error("The scheduled cutoff requires an explicit timezone")
    if args.dry_run:
        cutoff = min(due, datetime.now(timezone.utc))
        selected = choose_snapshot(cutoff)
        plan = plan_history(selected["run"], cutoff, selected["source"], selected["physical_time"])
        print(json.dumps(dict(selected_snapshot=selected, complete_saved_frames=len(plan["frames"]),
                             first_frame=plan["frames"][0], last_frame=plan["frames"][-1],
                             historical_phases=len(plan["ancestry"]),
                             variants=["near_field_without_isolines", "near_field_with_isolines",
                                       "whole_domain_without_isolines", "whole_domain_with_isolines"],
                             width=1920, height=1080, fps=12), ensure_ascii=False, indent=2))
        return
    if datetime.now(timezone.utc) < due:
        raise RuntimeError("Video worker started before the scheduled cutoff")
    directory = args.state.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "status.json"
    state = json.loads(path.read_text()) if path.exists() else {}
    if state.get("state") == "completed":
        raise RuntimeError("This scheduled video action has already completed")
    state.update(state="running", due=due.isoformat(), worker_pid=os.getpid(),
                 started_utc=datetime.now(timezone.utc).isoformat())

    def persist():
        state["updated_utc"] = datetime.now(timezone.utc).isoformat()
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(path)

    persist()
    try:
        selected = choose_snapshot(due)
        plan = plan_history(selected["run"], due, selected["source"], selected["physical_time"])
        manifest = directory / "selected_frames.json"
        manifest.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n")
        state.update(selected_snapshot=selected, selected_frames=len(plan["frames"]),
                     first_physical_time=plan["frames"][0]["physical_time"],
                     last_physical_time=plan["frames"][-1]["physical_time"],
                     manifest=str(manifest), phase="rendering")
        persist()
        output = directory / "output"
        env = os.environ.copy()
        env.pop("DISPLAY", None)
        env.update(OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1", VTK_SMP_MAX_THREADS="2",
                   EGL_PLATFORM="surfaceless", LIBGL_ALWAYS_SOFTWARE="1")
        command = [sys.executable, str(ROOT / "scripts/render_suboff_gpu_evolution.py"),
                   "--manifest", str(manifest), "--output", str(output)]
        with (directory / "render.log").open("w") as log:
            subprocess.run(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                           stdout=log, stderr=subprocess.STDOUT, check=True, timeout=20700)
        result = json.loads((output / "result.json").read_text())
        if result.get("state") != "completed" or len(result["videos"]) != 4:
            raise RuntimeError("Video renderer did not produce all four required variants")
        if any(not Path(video["path"]).is_file() for video in result["videos"].values()):
            raise RuntimeError("A rendered movie is missing")
        state.update(state="completed", phase="completed", videos=result["videos"],
                     result=str(output / "result.json"))
    except Exception as error:
        state.update(state="failed", error=str(error))
        raise
    finally:
        persist()
        print(json.dumps(state, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
