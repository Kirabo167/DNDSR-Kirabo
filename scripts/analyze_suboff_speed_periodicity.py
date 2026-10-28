#!/usr/bin/env python3
"""Diagnose apparent periodic speed changes in a SUBOFF VTK-HDF series."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from scipy import signal


STEP_PATTERN = re.compile(r"_(\d+)\.vtkhdf$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_directory", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument(
        "--series", default="SUBOFF_AFF1_SST.retained.vtkhdf.series"
    )
    parser.add_argument("--sample-cells", type=int, default=50_000)
    return parser.parse_args()


def step_from_name(name: str) -> int:
    match = STEP_PATTERN.search(name)
    if match is None:
        raise ValueError(f"Cannot parse pseudo step from {name}")
    return int(match.group(1))


def load_frames(input_directory: Path, series_name: str) -> list[tuple[int, Path]]:
    with (input_directory / series_name).open("r", encoding="utf-8") as stream:
        document = json.load(stream)
    frames = []
    for entry in document["files"]:
        path = input_directory / Path(entry["name"]).name
        if path.is_file():
            frames.append((step_from_name(path.name), path))
    frames.sort()
    if len(frames) < 3:
        raise ValueError("At least three existing frames are required")
    return frames


def weighted_stats(values: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    weight_sum = float(np.sum(weights))
    mean = float(np.sum(weights * values) / weight_sum)
    mean_square = float(np.sum(weights * values * values) / weight_sum)
    return {
        "mean": mean,
        "rms": mean_square**0.5,
        "std": max(0.0, mean_square - mean * mean) ** 0.5,
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def spectrum_summary(values: np.ndarray, step_interval: int) -> dict[str, float | None]:
    values = signal.detrend(np.asarray(values, dtype=float), type="linear")
    frequencies, power = signal.periodogram(
        values,
        fs=1.0 / step_interval,
        window="hann",
        detrend=False,
        scaling="spectrum",
    )
    positive = frequencies > 0.0
    frequencies = frequencies[positive]
    power = power[positive]
    peak_index = int(np.argmax(power))
    peak_frequency = float(frequencies[peak_index])
    peak_period = 1.0 / peak_frequency
    power_fraction = float(power[peak_index] / np.sum(power))

    autocorrelation = signal.correlate(values, values, mode="full", method="auto")
    autocorrelation = autocorrelation[len(values) - 1 :]
    autocorrelation /= autocorrelation[0]
    maximum_lag = len(values) // 2
    peaks, properties = signal.find_peaks(
        autocorrelation[2:maximum_lag], prominence=0.05
    )
    if len(peaks):
        best = int(np.argmax(properties["prominences"]))
        autocorrelation_lag = int(peaks[best] + 2)
        autocorrelation_period = float(autocorrelation_lag * step_interval)
        autocorrelation_peak = float(autocorrelation[autocorrelation_lag])
    else:
        autocorrelation_period = None
        autocorrelation_peak = None

    return {
        "fft_peak_period_steps": peak_period,
        "fft_peak_power_fraction": power_fraction,
        "fft_resolved_cycles": float(
            len(values) * step_interval / peak_period
        ),
        "autocorrelation_peak_period_steps": autocorrelation_period,
        "autocorrelation_peak_value": autocorrelation_peak,
    }


def main() -> None:
    args = parse_args()
    frames = load_frames(args.input_directory, args.series)
    args.output_directory.mkdir(parents=True, exist_ok=True)

    mesh = pv.read(frames[0][1])
    centers = np.asarray(mesh.cell_centers().points)
    volumes = np.asarray(
        mesh.compute_cell_sizes(length=False, area=False, volume=True).cell_data[
            "Volume"
        ],
        dtype=float,
    )
    if np.any(volumes <= 0.0):
        raise ValueError("Non-positive cell volume encountered")
    radius = np.hypot(centers[:, 1], centers[:, 2])
    masks = {
        "global": np.ones(mesh.n_cells, dtype=bool),
        "near_hull":
            (centers[:, 0] >= -0.25)
            & (centers[:, 0] <= 4.75)
            & (radius <= 1.0),
        "near_wake":
            (centers[:, 0] >= 4.0)
            & (centers[:, 0] <= 9.0)
            & (radius <= 1.0),
    }
    region_indices = {name: np.flatnonzero(mask) for name, mask in masks.items()}
    analysis_mask = (
        (centers[:, 0] >= -0.25)
        & (centers[:, 0] <= 9.0)
        & (radius <= 1.0)
    )
    candidates = np.flatnonzero(analysis_mask)
    rng = np.random.default_rng(20260916)
    sample_count = min(args.sample_cells, len(candidates))
    sample_indices = np.sort(rng.choice(candidates, sample_count, replace=False))

    steps = np.asarray([item[0] for item in frames])
    equal_spacing = int(np.min(np.diff(steps)))
    regular_mask = (steps <= 4950) & (steps % 50 == 0)
    regular_frame_indices = np.flatnonzero(regular_mask)
    regular_steps = steps[regular_mask]
    regular_interval = int(np.unique(np.diff(regular_steps)).item())
    sampled_speed = np.empty(
        (len(regular_frame_indices), sample_count), dtype=np.float32
    )

    rows: list[dict[str, float | int]] = []
    previous_speed: np.ndarray | None = None
    previous_step: int | None = None
    regular_lookup = {
        frame_index: regular_index
        for regular_index, frame_index in enumerate(regular_frame_indices)
    }
    for frame_index, (step, path) in enumerate(frames):
        with h5py.File(path, "r") as handle:
            velocity = np.asarray(handle["VTKHDF/CellData/Velocity"][:], dtype=float)
        if velocity.shape != (mesh.n_cells, 3) or not np.isfinite(velocity).all():
            raise ValueError(f"Invalid velocity array in {path}")
        speed = np.linalg.norm(velocity, axis=1)
        row: dict[str, float | int] = {"step": step}
        for name, indices in region_indices.items():
            stats = weighted_stats(speed[indices], volumes[indices])
            for statistic, value in stats.items():
                row[f"{name}_{statistic}"] = value
            if previous_speed is None:
                row[f"{name}_delta_rms"] = float("nan")
                row[f"{name}_delta_max"] = float("nan")
            else:
                difference = speed[indices] - previous_speed[indices]
                row[f"{name}_delta_rms"] = weighted_stats(
                    difference, volumes[indices]
                )["rms"]
                row[f"{name}_delta_max"] = float(np.max(np.abs(difference)))
        row["step_interval"] = 0 if previous_step is None else step - previous_step
        rows.append(row)
        if frame_index in regular_lookup:
            sampled_speed[regular_lookup[frame_index]] = speed[sample_indices]
        previous_speed = speed
        previous_step = step
        print(f"Scanned speed frame {frame_index + 1}/{len(frames)}: step {step}")

    csv_path = args.output_directory / "speed_statistics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    detrended = signal.detrend(sampled_speed.astype(float), axis=0, type="linear")
    sample_weights = volumes[sample_indices]
    sample_weights /= np.sum(sample_weights)
    weighted_field = detrended * np.sqrt(sample_weights)[None, :]
    covariance = weighted_field @ weighted_field.T
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]
    pod_score = eigenvectors[:, 0] * np.sqrt(eigenvalues[0])
    if pod_score[-1] < pod_score[0]:
        pod_score *= -1.0
    energy_fractions = eigenvalues / np.sum(eigenvalues)

    regular_near_mean = np.asarray(
        [rows[index]["near_hull_mean"] for index in regular_frame_indices],
        dtype=float,
    )
    full_pod_spectrum = spectrum_summary(pod_score, regular_interval)
    full_mean_spectrum = spectrum_summary(regular_near_mean, regular_interval)
    midpoint = len(pod_score) // 2
    first_half_spectrum = spectrum_summary(pod_score[: midpoint + 1], regular_interval)
    second_half_spectrum = spectrum_summary(pod_score[midpoint:], regular_interval)

    region_summary = {}
    for name, indices in region_indices.items():
        means = np.asarray([row[f"{name}_mean"] for row in rows], dtype=float)
        region_summary[name] = {
            "cell_count": int(len(indices)),
            "volume": float(np.sum(volumes[indices])),
            "speed_mean_first": float(means[0]),
            "speed_mean_last": float(means[-1]),
            "speed_mean_range": float(np.ptp(means)),
            "speed_mean_range_percent_of_average": float(
                100.0 * np.ptp(means) / np.mean(means)
            ),
        }

    last_rows = [row for row in rows if int(row["step"]) >= 4994]
    last_step_delta = {
        name: {
            "mean_delta_rms": float(
                np.nanmean([row[f"{name}_delta_rms"] for row in last_rows[1:]])
            ),
            "max_delta_rms": float(
                np.nanmax([row[f"{name}_delta_rms"] for row in last_rows[1:]])
            ),
        }
        for name in masks
    }

    summary = {
        "source_series": str((args.input_directory / args.series).resolve()),
        "frame_count": len(frames),
        "step_range": [int(steps[0]), int(steps[-1])],
        "frame_step_intervals": sorted({int(value) for value in np.diff(steps)}),
        "regular_sequence": {
            "step_range": [int(regular_steps[0]), int(regular_steps[-1])],
            "frame_count": len(regular_steps),
            "step_interval": regular_interval,
        },
        "regions": {
            "near_hull": "-0.25 <= x <= 4.75 and sqrt(y^2+z^2) <= 1.0",
            "near_wake": "4.0 <= x <= 9.0 and sqrt(y^2+z^2) <= 1.0",
            "global": "all fluid cells",
        },
        "region_statistics": region_summary,
        "last_steps_4994_5000": last_step_delta,
        "pod": {
            "sampled_cell_count": sample_count,
            "sample_region": "-0.25 <= x <= 9.0 and sqrt(y^2+z^2) <= 1.0",
            "leading_mode_energy_fraction": float(energy_fractions[0]),
            "first_five_mode_energy_fractions": [
                float(value) for value in energy_fractions[:5]
            ],
            "full_sequence_spectrum": full_pod_spectrum,
            "first_half_spectrum": first_half_spectrum,
            "second_half_spectrum": second_half_spectrum,
        },
        "near_hull_mean_speed_spectrum": full_mean_spectrum,
        "notes": [
            "Statistics are volume weighted except extrema.",
            "Spectra use only the uniformly sampled steps 100 through 4950.",
            "POD input is linearly detrended at every sampled cell.",
            f"Minimum adjacent frame interval is {equal_spacing} pseudo step(s).",
        ],
    }
    json_path = args.output_directory / "speed_periodicity_summary.json"
    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)
        stream.write("\n")

    figure, axes = plt.subplots(3, 2, figsize=(14, 12), constrained_layout=True)
    for name, color in (("global", "0.2"), ("near_hull", "C0"), ("near_wake", "C1")):
        axes[0, 0].plot(
            steps,
            [row[f"{name}_mean"] for row in rows],
            label=name,
            color=color,
        )
        axes[0, 1].plot(
            steps,
            [row[f"{name}_max"] for row in rows],
            label=name,
            color=color,
        )
        axes[1, 0].semilogy(
            steps[1:],
            [row[f"{name}_delta_rms"] for row in rows[1:]],
            label=name,
            color=color,
        )
    axes[0, 0].set(title="Volume-weighted mean speed", ylabel="Speed")
    axes[0, 1].set(title="Maximum cell speed", ylabel="Speed")
    axes[1, 0].set(
        title="RMS speed change from previous saved field",
        ylabel="RMS delta speed",
    )
    for axis in axes[0, :]:
        axis.legend()
    axes[1, 0].legend()

    tail_steps = np.asarray([row["step"] for row in last_rows])
    for name, color in (("global", "0.2"), ("near_hull", "C0"), ("near_wake", "C1")):
        axes[1, 1].semilogy(
            tail_steps[1:],
            [row[f"{name}_delta_rms"] for row in last_rows[1:]],
            marker="o",
            label=name,
            color=color,
        )
    axes[1, 1].set(
        title="Consecutive fields at steps 4994-5000",
        xlabel="Pseudo step",
        ylabel="RMS delta speed per step",
    )
    axes[1, 1].legend()

    axes[2, 0].plot(regular_steps, pod_score, color="C3")
    axes[2, 0].axhline(0.0, color="0.6", linewidth=0.8)
    axes[2, 0].set(
        title="Leading detrended speed POD coefficient",
        xlabel="Pseudo step",
        ylabel="Coefficient",
    )

    frequencies, power = signal.periodogram(
        signal.detrend(pod_score),
        fs=1.0 / regular_interval,
        window="hann",
        detrend=False,
        scaling="spectrum",
    )
    positive = frequencies > 0.0
    axes[2, 1].plot(1.0 / frequencies[positive], power[positive], color="C3")
    axes[2, 1].set_xscale("log")
    axes[2, 1].set_yscale("log")
    axes[2, 1].set(
        title="POD coefficient periodogram",
        xlabel="Period (pseudo steps)",
        ylabel="Spectral power",
    )
    for axis in axes.flat:
        axis.grid(True, alpha=0.25)
    png_path = args.output_directory / "speed_periodicity_diagnostic.png"
    figure.savefig(png_path, dpi=180)
    plt.close(figure)

    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {png_path}")
    print(json.dumps(summary["pod"], indent=2))


if __name__ == "__main__":
    main()
