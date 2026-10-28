"""Read-only RP1 checkpoint slices, with shared physical-variable color scales."""

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/ncfv-rp1-z-slices-matplotlib")
os.environ.setdefault("EGL_PLATFORM", "surfaceless")

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
import numpy as np
import pyvista as pv


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_checkpoint(path, layers):
    with h5py.File(path) as data:
        values = data["state/father/array/data"][:].reshape(-1, 5)
        ids = data["state/origIndex"][:].astype(np.int64)
        raw = data.attrs["configuration"]
        configuration = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
        time = float(data.attrs["simulationTime"])
        iteration = int(data.attrs["iteration"])
    side = int(round(np.sqrt(len(ids) / layers)))
    if layers * side * side != len(ids) or not np.array_equal(np.sort(ids), np.arange(len(ids))):
        raise ValueError("Expected contiguous z-periodic Cartesian quotient node identities")
    periods = configuration["mesh"]["periodicLengths"]
    if periods[0] != 0 or periods[1] != 0 or periods[2] <= 0:
        raise ValueError("This reader requires only z periodicity")
    ordered = np.empty_like(values)
    ordered[ids] = values
    cube = ordered.reshape(layers, side, side, 5)
    rho = cube[..., 0]
    velocity = cube[..., 1:4] / rho[..., None]
    pressure = (configuration["physics"]["gamma"] - 1) * (
        cube[..., 4] - .5 * rho * np.sum(velocity ** 2, axis=-1))
    if not np.isfinite(cube).all() or np.any(rho <= 0) or np.any(pressure <= 0):
        raise ValueError("Nonphysical checkpoint")
    fields = {"rho": rho, "p": pressure, "u": velocity[..., 0], "v": velocity[..., 1],
              "w": velocity[..., 2], "speed": np.linalg.norm(velocity, axis=-1)}
    return cube, fields, np.linspace(0, 1, side), np.arange(layers) * periods[2] / layers, time, iteration


def panels(fields, names, labels, x, z, time, path, derivative=False):
    figure, axes = plt.subplots(len(names), len(z), figsize=(19, 4.3 * len(names) + .7),
                                squeeze=False, layout="constrained", facecolor="white")
    figure.set_constrained_layout_pads(h_pad=.12, hspace=.10, w_pad=.04, wspace=.03)
    for row, (name, label) in enumerate(zip(names, labels)):
        values = fields[name]
        low, high = float(values.min()), float(values.max())
        if derivative or name == "w":
            bound = max(abs(low), abs(high), 1e-15)
            norm = TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound)
        else:
            norm = Normalize(low, high if high > low else low + 1e-15)
        levels = np.linspace(norm.vmin, norm.vmax, 27)[1:-1]
        for column, position in enumerate(z):
            axis = axes[row, column]
            image = axis.pcolormesh(x, x, values[column], shading="gouraud", cmap="coolwarm", norm=norm)
            valid = levels[(levels > values[column].min()) & (levels < values[column].max())]
            if name == "w" or derivative:
                valid = valid[np.abs(valid) > 1e-12 * max(1.0, abs(norm.vmin), abs(norm.vmax))]
            if len(valid):
                axis.contour(x, x, values[column], levels=valid, colors="black", linewidths=.35, alpha=.60)
            axis.set(xlabel="x" if row == len(names) - 1 else "", ylabel="y", xlim=(0, 1), ylim=(0, 1), aspect="equal",
                     xticks=[0, .5, 1], yticks=[0, .5, 1])
            axis.set_title(f"z={position:.4g}\n{label}", fontsize=11)
        figure.colorbar(image, ax=list(axes[row]), shrink=.83, pad=.015, label=label, format="%.3g")
    note = "Raw least-squares gradients; pressure/velocity chain rule at recovered points" if derivative else "Physical variables converted from conservative dual-volume means"
    contour_note = "Fixed levels; zero contour omitted for signed gradients/w." if derivative or "w" in names else "25 contour levels."
    figure.suptitle(f"RP1, 400 x 400 x 5, z-periodic, t={time:g}\n{note}\n"
                    "Blue: low; red: high. Shared scale for each variable across all slices. " + contour_note, fontsize=13)
    figure.savefig(path, dpi=180, facecolor="white")
    plt.close(figure)


def gradient_panels(summary_path, out, x, z, time):
    summary = json.loads(summary_path.read_text())
    if abs(summary["physical_time"] - time) > 1e-12:
        raise ValueError("Gradient audit time does not match the slice time")
    names = ("rho", "p", "u", "v", "w")
    columns = summary["node_table_columns"]
    width = summary["node_table_width"]
    count = len(z) * len(x) ** 2
    fields = {name: np.empty(count) for name in names}
    seen = np.zeros(count, dtype=bool)
    for rank in range(summary["mpi_ranks"]):
        path = Path(summary["node_table_pattern"].replace("<RANK:06d>", f"{rank:06d}"))
        rows = np.fromfile(path, dtype="<f8").reshape(-1, width)
        ids = rows[:, columns.index("original_node")].astype(np.int64)
        if len(np.unique(ids)) != len(ids) or seen[ids].any():
            raise ValueError("Duplicated gradient node identities")
        expected = np.column_stack((x[ids % len(x)], x[(ids // len(x)) % len(x)], z[ids // len(x) ** 2]))
        if np.max(np.abs(rows[:, [columns.index(c) for c in ("x", "y", "z")]] - expected)) > 1e-12:
            raise ValueError("Gradient coordinates do not match quotient node ordering")
        seen[ids] = True
        for name in names:
            fields[name][ids] = rows[:, columns.index("point_chain_dz_" + name)]
    if not seen.all():
        raise ValueError("Missing gradient nodes")
    fields = {name: values.reshape(len(z), len(x), len(x)) for name, values in fields.items()}
    panels(fields, ("rho", "p"), ("Raw density z derivative", "Raw pressure z derivative"),
           x, z, time, out / "raw_dz_density_pressure_slices.png", derivative=True)
    panels(fields, ("u", "v", "w"), ("Raw u z derivative", "Raw v z derivative", "Raw w z derivative"),
           x, z, time, out / "raw_dz_velocity_slices.png", derivative=True)
    np.savez_compressed(out / "raw_z_derivatives.npz", x=x, z=z, time=time, **fields)
    metadata = {"audit_summary": str(summary_path.resolve()), "definition": summary["limiter_note"],
                "max_abs_dz": {name: float(np.abs(values).max()) for name, values in fields.items()}}
    (out / "gradient_render_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--restart", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", type=int, default=5)
    parser.add_argument("--gradient-summary", type=Path)
    parser.add_argument("--figures-only", action="store_true", help="Refresh generated figures from this output's own cached data")
    args = parser.parse_args()
    path = args.restart.resolve()
    out = args.output.resolve()
    if args.figures_only:
        metadata = json.loads((out / "slice_metrics.json").read_text())
        if digest(path) != metadata["sha256_before"]:
            raise ValueError("Checkpoint no longer matches the cached slices")
        with np.load(out / "slice_fields.npz") as data:
            x, z, time = data["x"], data["z"], float(data["time"])
            fields = {name: data[name] for name in ("rho", "p", "u", "v", "w", "speed")}
        panels(fields, ("rho", "p"), ("Density", "Pressure"), x, z, time, out / "density_pressure_z_slices.png")
        panels(fields, ("u", "v", "w"), ("Velocity u", "Velocity v", "Velocity w"), x, z, time, out / "velocity_z_slices.png")
        if args.gradient_summary:
            gradient_panels(args.gradient_summary, out, x, z, time)
        return
    out.mkdir(parents=True, exist_ok=False)
    before = digest(path)
    cube, fields, x, z, time, iteration = load_checkpoint(path, args.layers)
    metrics = {"checkpoint": str(path), "sha256_before": before, "time": time,
               "iteration": iteration, "mesh_cells": [len(x) - 1, len(x) - 1, args.layers],
               "z_planes": z.tolist(), "state_kind": "conservative dual-volume means converted to physical variables",
               "periodic_endpoint_note": "The z=Lz plane is identical to z=0 and is represented by the same quotient nodes.",
               "fields": {}, "slices": []}
    for name, values in fields.items():
        span = np.ptp(values, axis=0)
        k, j, i = np.unravel_index(np.abs(values).argmax(), values.shape)
        metrics["fields"][name] = {"minimum": float(values.min()), "maximum": float(values.max()),
                                    "maximum_z_span": float(span.max()),
                                    "max_abs": float(np.abs(values[k, j, i])),
                                    "max_abs_coordinate": [float(x[i]), float(x[j]), float(z[k])]}
    vtk = out / "vtk_slices"
    vtk.mkdir()
    xx, yy = np.meshgrid(x, x, indexing="xy")
    for plane, position in enumerate(z):
        record = {"z": float(position), "fields": {name: {"minimum": float(values[plane].min()),
                    "maximum": float(values[plane].max())} for name, values in fields.items()}}
        grid = pv.StructuredGrid(xx, yy, np.full_like(xx, position))
        for name, label in (("rho", "Density"), ("p", "Pressure"), ("u", "u"),
                            ("v", "v"), ("w", "w"), ("speed", "Speed")):
            grid.point_data[label] = fields[name][plane].ravel(order="F")
        grid.point_data["Velocity"] = np.column_stack([fields[name][plane].ravel(order="F") for name in ("u", "v", "w")])
        grid.point_data["rhoE"] = cube[plane, ..., 4].ravel(order="F")
        target = vtk / f"z_plane_{plane}.vts"
        grid.save(target)
        record["vtk"] = str(target)
        metrics["slices"].append(record)
    panels(fields, ("rho", "p"), ("Density", "Pressure"), x, z, time, out / "density_pressure_z_slices.png")
    panels(fields, ("u", "v", "w"), ("Velocity u", "Velocity v", "Velocity w"), x, z, time, out / "velocity_z_slices.png")
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for axis, (field, y) in zip(axes.flat, (("rho", .5), ("p", .5), ("rho", .95), ("p", .95))):
        row = int(round(y * (len(x) - 1)))
        for plane, position in enumerate(z):
            axis.plot(x, fields[field][plane, row], linewidth=.9, label=f"z={position:.4g}")
        axis.set(xlabel="x", ylabel="Density" if field == "rho" else "Pressure", title=f"y={x[row]:g}", xlim=(0, 1))
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
    figure.suptitle(f"RP1 slice profiles, t={time:g}; same saved checkpoint")
    figure.savefig(out / "slice_profiles.png", dpi=180, facecolor="white")
    plt.close(figure)
    metrics["sha256_after"] = digest(path)
    if before != metrics["sha256_after"]:
        raise ValueError("Checkpoint changed during read-only post-processing")
    (out / "slice_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    np.savez_compressed(out / "slice_fields.npz", x=x, z=z, time=time, **fields)
    if args.gradient_summary:
        gradient_panels(args.gradient_summary, out, x, z, time)
    print(json.dumps({key: metrics[key] for key in ("time", "iteration", "z_planes", "fields")}, indent=2))


if __name__ == "__main__":
    main()
