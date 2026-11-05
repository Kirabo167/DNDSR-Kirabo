"""Plot the volume-weighted density errors of the entropy-wave LS sweep."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter, NullLocator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    with (output / "metrics.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    plt.rcParams.update({"font.size": 10, "axes.grid": True,
                         "grid.alpha": 0.25, "savefig.dpi": 180})
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    for row_index, mode in enumerate(("Traditional", "Efficient")):
        for power, color in ((1, "#456990"), (2, "#d1495b")):
            series = sorted((r for r in rows if r["mode"] == mode and
                             float(r["weight_power"]) == power),
                            key=lambda r: int(r["level"]))
            if not series:
                continue
            x = [int(r["nodes"]) ** (1 / 3) for r in series]
            for column, field, title in ((0, "mean_density_L2", "Dual mean density"),
                                         (1, "point_density_L2", "Recovered point density")):
                ax = axes[row_index, column]
                ax.loglog(x, [float(r[field]) for r in series], marker="o",
                          color=color, label=f"p={power}")
                ax.set_title(f"{mode}: {title}")
                ax.set_ylabel("Volume-weighted L2 error")
                ax.legend()
    for ax in axes[-1]:
        ax.xaxis.set_major_locator(FixedLocator(x))
        ax.xaxis.set_major_formatter(FixedFormatter([f"IV{r['level']}" for r in series]))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("Prism mesh refinement")
    fig.tight_layout()
    fig.savefig(output / "convergence.png")
    fig.savefig(output / "convergence.pdf")


if __name__ == "__main__":
    main()
