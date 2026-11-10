#!/usr/bin/env python3
"""One-shot report action for a host timer; pin the last completed field at a cutoff."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def choose_snapshot(cutoff):
    active = json.loads((ROOT / "data/out/GPU/SUBOFF_active_run.json").read_text())
    run = Path(active["directory"])
    series = list((run / "fields").glob("*.series"))
    choices = []
    if series:
        for entry in json.loads(series[0].read_text())["files"]:
            source = run / "fields" / entry["name"]
            if source.is_file() and source.stat().st_mtime <= cutoff.timestamp():
                choices.append((entry, source))
    if choices:
        # Steady snapshots retain the source physical time; select the latest
        # iteration when several complete outputs share that time coordinate.
        entry, source = max(choices, key=lambda pair: (pair[0]["time"],
                            int(re.search(r"_(\d+)\.vtkhdf$", pair[1].name)[1])))
    else:
        # If the active phase is still initializing, its explicit restart field
        # is the latest completed state belonging to that continuation.
        configuration = json.loads(Path(active["config"]).read_text())
        restart = configuration["restartSettings"]
        source_file = configuration.get("caseNotes", {}).get("restartSourceField", restart["flowFile"])
        source = (ROOT / "build-gpu-solvers" / source_file).resolve()
        if not restart["flowFile"] or not source.is_file() or source.stat().st_mtime > cutoff.timestamp():
            raise RuntimeError("No completed saved field available by the scheduled cutoff")
        run = source.parent.parent
        entry = dict(name=source.name, time=restart["physicalTime"])
    step = int(re.search(r"_(\d+)\.vtkhdf$", source.name)[1])
    return dict(run=str(run), source=str(source), step=step, physical_time=entry["time"],
                file_completed_utc=datetime.fromtimestamp(source.stat().st_mtime, timezone.utc).isoformat())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--due", required=True, help="ISO timestamp with explicit timezone")
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    due = datetime.fromisoformat(args.due)
    if due.tzinfo is None:
        raise ValueError("Scheduled report requires a timezone-aware cutoff")
    if args.dry_run:
        print(json.dumps(choose_snapshot(min(due, datetime.now(timezone.utc))), indent=2))
        return
    if datetime.now(timezone.utc) < due:
        raise RuntimeError("Report action started before its scheduled cutoff")
    directory = args.state.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "status.json"
    state = json.loads(path.read_text()) if path.exists() else {}
    if state.get("state") == "completed":
        raise RuntimeError("The scheduled report has already completed")
    state.update(state="running", due=due.isoformat(), worker_pid=os.getpid(),
                 started_utc=datetime.now(timezone.utc).isoformat())

    def persist():
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(path)

    persist()
    try:
        selected = choose_snapshot(due)
        state["selected_snapshot"] = selected
        persist()
        env = os.environ.copy()
        env.pop("DISPLAY", None)
        env.update(OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1", VTK_SMP_MAX_THREADS="4",
                   EGL_PLATFORM="surfaceless", LIBGL_ALWAYS_SOFTWARE="1")
        command = [sys.executable, str(ROOT / "scripts/render_suboff_flow_report.py"),
                   "--run", selected["run"], "--step", str(selected["step"])]
        with (directory / "render.log").open("w") as log:
            subprocess.run(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                           stdout=log, stderr=subprocess.STDOUT, check=True, timeout=3600)
        report = Path(selected["run"]) / "postprocess" / f"step_{selected['step']:08d}"
        command = [sys.executable, str(ROOT / "scripts/export_suboff_vectors.py"), "--report", str(report)]
        with (directory / "vectors.log").open("w") as log:
            subprocess.run(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                           stdout=log, stderr=subprocess.STDOUT, check=True, timeout=3600)
        vector_pdfs, vector_bundles = {}, {}
        for variant, folder in (("without_isolines", "vector_smooth"), ("with_isolines", "vector_smooth_isolines")):
            vector_pdfs[variant] = str(report / folder / f"SUBOFF_vector_report_step_{selected['step']:08d}.pdf")
            vector_bundles[variant] = str(report / folder / f"SUBOFF_vector_figures_step_{selected['step']:08d}.zip")
            if not Path(vector_pdfs[variant]).is_file() or not Path(vector_bundles[variant]).is_file():
                raise RuntimeError(f"Missing scheduled vector report variant: {variant}")
        state.update(state="completed", report_directory=str(report),
                     pdf=str(report / f"SUBOFF_flow_report_step_{selected['step']:08d}.pdf"),
                     data_bundle=str(report / f"SUBOFF_flow_report_step_{selected['step']:08d}.zip"),
                     vector_bundle=vector_bundles["without_isolines"],
                     vector_pdfs=vector_pdfs, vector_bundles=vector_bundles)
    except Exception as error:
        state.update(state="failed", error=str(error))
        raise
    finally:
        state["updated_utc"] = datetime.now(timezone.utc).isoformat()
        persist()
        print(json.dumps(state, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
