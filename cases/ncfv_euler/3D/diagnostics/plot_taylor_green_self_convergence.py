"""Plot four-grid Taylor--Green self-differences at fixed shared nodes."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    with (output / "all_four_self_convergence.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    x = [1.0, (3896 / 512) ** (1 / 3), (30416 / 512) ** (1 / 3)]
    fields = (("velocity", "Velocity vector"), ("density", "Density"),
              ("pressure", "Pressure"))
    plt.rcParams.update({"font.size": 10, "axes.grid": True,
                         "grid.alpha": 0.25, "savefig.dpi": 180})
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), sharex=True)
    for ax, (field, title) in zip(axes, fields):
        for mode, color in (("Traditional", "#456990"),
                            ("Efficient", "#d1495b")):
            row = next(r for r in rows if r["mode"] == mode and r["field"] == field)
            y = [float(row[f"difference_{pair}"]) for pair in
                 ("10_20", "20_40", "40_80")]
            ax.loglog(x, y, "o-", color=color, label=mode)
        ax.set_title(title)
        ax.set_ylabel("Volume-weighted L2 difference")
        ax.set_xlabel("Refinement pair")
        ax.xaxis.set_major_locator(FixedLocator(x))
        ax.xaxis.set_major_formatter(FixedFormatter(
            ["IV10–20", "IV20–40", "IV40–80"]))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.legend()
    fig.tight_layout()
    fig.savefig(output / "self_convergence.png")
    fig.savefig(output / "self_convergence.pdf")


if __name__ == "__main__":
    main()
