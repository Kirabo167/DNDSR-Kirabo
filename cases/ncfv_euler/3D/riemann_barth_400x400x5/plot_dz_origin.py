"""Plot the measured column-weight leakage and pure reconstruction tests."""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/ncfv-dz-origin-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np


def main():
    root = Path(__file__).resolve().parents[4]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=root / "data/out/NCFV/reconstruction_dz_diagnostic/20261026")
    parser.add_argument("--columns", type=Path, default=root / "data/out/NCFV/z_column_actual_audit")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    figure, axes = plt.subplots(2, 2, figsize=(12, 9), layout="constrained", facecolor="white")
    norm = TwoSlopeNorm(vmin=-5 / 3, vcenter=0, vmax=5 / 3)
    cmap = matplotlib.colormaps["coolwarm"].copy()
    cmap.set_bad("#eeeeee")
    for axis, name, title in zip(axes[0], ("compact_actual_max", "complete_same_node"),
                                 ("Compact 16-point stencil", "Complete 24-point ring")):
        path = args.columns / f"{name}.json.columns.csv"
        rows = np.atleast_1d(np.genfromtxt(path, delimiter=",", names=True))
        matrix = np.full((5, 5), np.nan)
        for row in rows:
            ix = int(round((row["x"] - .475) / .025)) + 2
            iy = int(round((row["y"] - .5) / .025)) + 2
            matrix[iy, ix] = row["column_weight_sum"]
        image = axis.imshow(matrix, origin="lower", extent=(-2.5, 2.5, -2.5, 2.5),
                            cmap=cmap, norm=norm, interpolation="nearest")
        for iy, ix in zip(*np.where(np.isfinite(matrix))):
            value = matrix[iy, ix]
            axis.text(ix - 2, iy - 2, "0" if abs(value) < 1e-12 else f"{value:+.3f}",
                      ha="center", va="center", fontsize=10)
        axis.set(xlabel=r"$(x-x_i)/h$", ylabel=r"$(y-y_i)/h$", xticks=range(-2, 3), yticks=range(-2, 3),
                 title=title + "\nSum of z-derivative weights in each (x,y) column")
    figure.colorbar(image, ax=list(axes[0]), shrink=.85, label="Column weight sum (1 / length)")
    sizes = np.array([20, 40, 80])
    for axis, field, title in zip(axes[1], (3, 4),
                                 (r"Smooth $q=\sin(2\pi x)\cos(2\pi y)$", "RP1 density discontinuity")):
        for strategy, color, label in (("compact", "#c33e32", "Compact"),
                                        ("complete", "#2666a5", "Complete ring")):
            values = []
            for size in sizes:
                stem = f"flat_{strategy}_np1" if size == 40 else f"flat_{size}_{strategy}_np1"
                data = json.loads((args.data / (stem + ".json")).read_text())
                if any(v != 0 for v in data["mean_maximum_column_span"]):
                    raise ValueError("The input is not strictly independent of z")
                values.append(data["raw_gradient_abs_max"][10 + field])
            axis.loglog(sizes, values, "o-", color=color, linewidth=1.8, label=label)
            if strategy == "compact":
                exponent = -2 if field == 3 else 1
                axis.loglog(sizes, values[0] * (sizes / sizes[0]) ** exponent, "k--", alpha=.6,
                            label=r"$N^{-2}$ reference" if exponent == -2 else r"$N$ reference")
        axis.set(xlabel="Cells in x and y, N (5 z cells; h=1/N)", ylabel=r"$\max|D_zq|$", title=title,
                 xticks=sizes, xticklabels=[str(n) for n in sizes])
        axis.grid(alpha=.2, which="both")
        axis.legend(fontsize=9)
    figure.suptitle("Why a 3-D reconstruction creates a z derivative for 2-D data\n"
                    "Exact dual-volume means; raw reconstruction only, without limiting or time advancement", fontsize=14)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180, facecolor="white")
    plt.close(figure)
    print(args.output)


if __name__ == "__main__":
    main()
