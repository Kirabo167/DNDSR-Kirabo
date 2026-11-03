"""Audit native z layers for RP3 (t=0.3) and RP4 (t=0.2)."""
import argparse
import csv
import json
import os
from pathlib import Path
import re

os.environ.setdefault("MPLCONFIGDIR", "/tmp/ncfv-efficient-derivation-mpl")
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

NAMES = ["SG_BJ_PRES", "SG_SIGN_BUDGET", "FC_IDP_BINARY", "FC_IDP_CONT", "LLF_FIRST_ORDER"]


def read_checkpoint(path):
    before = path.stat()
    with h5py.File(path) as f:
        u = f["state/father/array/data"][:].reshape(-1, 5)
        ids = f["state/origIndex"][:].astype(np.int64)
        t, iteration = float(f.attrs["simulationTime"]), int(f.attrs["iteration"])
        config = json.loads(f.attrs["configuration"])
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("Checkpoint is still changing")
    n = len(u)
    side = int(round(np.sqrt(n / 5)))
    if side * side * 5 != n:
        raise ValueError("Unexpected Cartesian layer shape")
    if u.shape != (n, 5) or not np.array_equal(np.sort(ids), np.arange(n)):
        raise ValueError("Unexpected independent periodic node identities")
    ordered = np.empty_like(u)
    ordered[ids] = u
    cube = ordered.reshape(5, side, side, 5)
    rho = cube[..., 0]
    p = .4 * (cube[..., 4] - .5 * np.sum(cube[..., 1:4] ** 2, axis=-1) / rho)
    if not np.isfinite(cube).all() or np.any(rho <= 0) or np.any(p <= 0):
        raise ValueError("Nonphysical saved state")
    return {"rho": rho, "p": p, "w": cube[..., 3] / rho}, t, iteration, config


def export(case, path):
    fields, t, iteration, config = read_checkpoint(path)
    out = case / "post" / f"step_{iteration:08d}"
    out.mkdir(parents=True, exist_ok=True)
    dz = config["mesh"]["periodicLengths"][2] / 5
    z_layers = (np.arange(5) * dz).tolist()
    result = {"name": case.name, "time": t, "iteration": iteration, "checkpoint": str(path),
              "problem": case.parent.name, "z_layers": z_layers, "fields": {},
              "definition": "Conservative dual-volume means at retained native z-node layers; no z interpolation",
              "active_reconstruction_settings": config["reconstruction"]}
    x = np.linspace(0, 1, fields["rho"].shape[-1])
    for key, values in fields.items():
        span = np.ptp(values, axis=0)
        result["fields"][key] = {"min": float(values.min()), "max": float(values.max()),
                                 "max_abs": float(np.abs(values).max()), "z_span_max": float(span.max()),
                                 "z_span_rms": float(np.sqrt(np.mean(span ** 2)))}
        if key not in ("rho", "p"):
            continue
        lo, hi = float(values.min()), float(values.max())
        hi = max(hi, lo + 1e-12)
        levels = np.linspace(lo, hi, 27)[1:-1]
        fig, axes = plt.subplots(1, 5, figsize=(18, 4), layout="constrained")
        for k, ax in enumerate(axes):
            im = ax.pcolormesh(x, x, values[k], cmap="coolwarm", vmin=lo, vmax=hi, shading="nearest", rasterized=True)
            valid = levels[(levels > values[k].min()) & (levels < values[k].max())]
            if len(valid): ax.contour(x, x, values[k], levels=valid, colors="black", linewidths=.3, alpha=.6)
            ax.set(title=f"z={z_layers[k]:g}", xlabel="x", ylabel="y", aspect="equal")
        fig.colorbar(im, ax=axes, shrink=.8, label=key)
        fig.suptitle(f"{case.parent.name} / {case.name}: {key}, t={t:.7f}; {len(x)-1} x {len(x)-1} x 5; 25 contour levels")
        fig.savefig(out / f"{key}_all_z_layers.png", dpi=150)
        plt.close(fig)
    np.savez_compressed(out / "fields.npz", **fields, time=t)
    log = (case / "solver.log").read_text(errors="replace") if (case / "solver.log").exists() else ""
    timings = [float(v) for v in re.findall(r"step wall\[s\]=([0-9.eE+-]+)", log)]
    result["reported_step_seconds"] = {"count": len(timings), "mean_after_first_two": float(np.mean(timings[2:])) if len(timings) > 2 else None,
                                        "median_after_first_two": float(np.median(timings[2:])) if len(timings) > 2 else None,
                                        "note": "Reported steps only; checkpoint writing may not be included in step wall timing."}
    result["rejected_steps"] = log.count("NCFV rejected iteration=")
    (out / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    (case / "latest_metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def compare(root):
    records = []
    for name in NAMES:
        p = root / name / "latest_metrics.json"
        if p.exists(): records.append(json.loads(p.read_text()))
    post = root / "comparison"
    post.mkdir(exist_ok=True)
    (post / "latest_metrics.json").write_text(json.dumps(records, indent=2) + "\n")
    # Compare only states at the final common physical time. Partial states are
    # reported separately above and never silently labelled as t=0.3.
    target = {"RP3": .3, "RP4": .2}[root.name]
    finals = [r for r in records if abs(r["time"] - target) < 1e-12]
    for key in ("rho", "p"):
        if not finals: continue
        arrays = [np.load(root / r["name"] / "post" / f"step_{r['iteration']:08d}" / "fields.npz")[key] for r in finals]
        lo = min(float(a.min()) for a in arrays)
        hi = max(float(a.max()) for a in arrays)
        levels = np.linspace(lo, hi, 27)[1:-1]
        fig, axes = plt.subplots(1, len(finals), figsize=(5 * len(finals), 5), squeeze=False, layout="constrained")
        x = np.linspace(0, 1, 801)
        for ax, r, a in zip(axes[0], finals, arrays):
            im = ax.pcolormesh(x, x, a[2], cmap="coolwarm", vmin=lo, vmax=hi, shading="nearest", rasterized=True)
            valid = levels[(levels > a[2].min()) & (levels < a[2].max())]
            if len(valid): ax.contour(x, x, a[2], levels=valid, colors="black", linewidths=.35, alpha=.6)
            ax.set(title=f"{r['name']}\nz span={r['fields'][key]['z_span_max']:.4g}", xlabel="x", ylabel="y", aspect="equal")
        fig.colorbar(im, ax=axes[0], shrink=.8, label=key)
        fig.suptitle(f"{root.name}: t={target:g}, z=0.02, common colors and 25 contour levels; {len(finals)}/{len(NAMES)} cases complete")
        fig.savefig(post / f"final_{key}_comparison.png", dpi=170)
        plt.close(fig)
    with (post / "metrics.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["case", "physical_time", "iteration", "rho_min", "p_min", "rho_z_span_max", "p_z_span_max", "max_abs_w", "rejected_steps"])
        for r in records:
            f = r["fields"]
            writer.writerow([r["name"], r["time"], r["iteration"], f["rho"]["min"], f["p"]["min"], f["rho"]["z_span_max"], f["p"]["z_span_max"], f["w"]["max_abs"], r["rejected_steps"]])
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    print(json.dumps(compare(args.root), indent=2))
