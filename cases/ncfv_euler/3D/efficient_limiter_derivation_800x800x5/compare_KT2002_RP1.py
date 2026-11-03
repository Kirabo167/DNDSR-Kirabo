"""Qualitative paper comparison; the paper provides figures, not exact fields."""
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil

os.environ.setdefault("MPLCONFIGDIR", "/tmp/ncfv-kt2002-comparison")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[4]
RUN = ROOT / "data/out/NCFV/RP1_efficient_limiter_derivation_800x800x5_np128_smt_20261029"
OUTPUT = RUN / "paper_comparison_KT2002_20261030"
NAMES = ["SG_BJ_PRES", "SG_SIGN_BUDGET", "FC_IDP_BINARY", "FC_IDP_CONT", "LLF_FIRST_ORDER"]


def main():
    shutil.copy2(Path(__file__), OUTPUT / "inputs" / Path(__file__).name)
    initial = json.loads((ROOT / "data/mesh/riemann_800x800x5_Lz0p05_initial/RP1.metadata.json").read_text())
    expected = [[.138, 1.206, 1.206, 0., .029], [.5323, 0., 1.206, 0., .3],
                [.5323, 1.206, 0., 0., .3], [1.5, 0., 0., 0., 1.5]]
    if not np.array_equal(initial["primitive_states"], expected):
        raise ValueError("RP1 does not match paper Configuration 3")
    video = json.loads((RUN / "videos_five_cases_t025_25contours_20261030/render_manifest.json").read_text())
    levels = video["contour_levels"]["Density"]
    data, records = {}, []
    t = .25
    paper_t = .3
    x = np.linspace(0, 1, 801)
    similarity = (x - .5) / t
    bound = .5 / paper_t
    low, high = .5323, 1.5
    shock_speed = (high * 0. - low * 1.206) / (high - low)
    reference = .5 + shock_speed * t
    cut_index = 720  # y=0.9 or x=0.9, outside the central interaction region.

    def crossing(profile, level):
        candidates = np.flatnonzero((profile[:-1] - level) * (profile[1:] - level) <= 0)
        candidates = [i for i in candidates if abs((x[i] + x[i + 1]) / 2 - reference) < .06 and profile[i + 1] != profile[i]]
        values = [x[i] + (level - profile[i]) / (profile[i + 1] - profile[i]) * (x[i + 1] - x[i]) for i in candidates]
        if not values:
            raise ValueError("Cannot identify the isolated shock")
        return min(values, key=lambda v: abs(v - reference))

    for name in NAMES:
        metric = json.loads((RUN / name / "latest_metrics.json").read_text())
        if abs(metric["time"] - t) > 1e-12:
            raise ValueError("Saved result time differs")
        with np.load(RUN / name / "post" / f"step_{metric['iteration']:08d}" / "fields.npz") as archive:
            rho, pressure, w = archive["rho"], archive["p"], archive["w"]
        plane = rho[2]
        data[name] = plane
        record = dict(case=name, physical_time=t, rho_min=float(rho.min()), rho_max=float(rho.max()),
                      p_min=float(pressure.min()), p_max=float(pressure.max()),
                      rho_z_span_max=float(np.ptp(rho, axis=0).max()), rho_z_span_rms=float(np.sqrt(np.mean(np.ptp(rho, axis=0)**2))),
                      p_z_span_max=float(np.ptp(pressure, axis=0).max()), max_abs_w=float(np.abs(w).max()),
                      rho_xy_reflection_linf=float(np.abs(plane-plane.T).max()))
        window = abs(x - reference) <= .04
        for label, profile in (("horizontal_y09", plane[cut_index, :]), ("vertical_x09", plane[:, cut_index])):
            p10 = crossing(profile, low + .1 * (high - low))
            p90 = crossing(profile, low + .9 * (high - low))
            mid = crossing(profile, .5 * (low + high))
            record[label + "_shock_midpoint"] = mid
            record[label + "_shock_position_abs_difference"] = abs(mid - reference)
            record[label + "_width_10_90"] = abs(p90 - p10)
            record[label + "_width_in_mesh_intervals"] = abs(p90 - p10) * 800
            record[label + "_local_density_overshoot"] = max(0., float(profile[window].max()) - high)
            record[label + "_local_density_undershoot"] = max(0., low - float(profile[window].min()))
        records.append(record)
    with (OUTPUT / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    fig, axes = plt.subplots(2, 5, figsize=(21, 9), layout="constrained")
    titles = [r"Paper Fig.3.3(a): 2nd order, $\theta=2$", "Paper Fig.3.3(b): 3rd order",
              r"Paper Fig.3.3(c): 2nd order, $\theta=1$"]
    for i, letter in enumerate("abc"):
        ax = axes[0, i]
        ax.imshow(Image.open(OUTPUT / "inputs" / f"figure_3_3_{letter}.png"),
                  extent=[-bound, bound, -bound, bound], origin="upper")
        ax.set(title=titles[i] + "\n400 x 400 gridpoints; t=0.3", xlabel=r"$\xi=(x-0.5)/t$", ylabel=r"$\eta=(y-0.5)/t$")
    for ax in axes[0, 3:]:
        ax.axis("off")
    axes[0, 3].text(0, .95, "Matched: Configuration 3\nFour primitive quadrant states\nIdeal gas, gamma = 1.4\nDiscontinuities at x=y=0.5\n\nDifferent: time and resolution\nPaper: t=0.3, CFL=0.475\nNCFV: t=0.25, CFL=0.5\n800 x 800 x 5 primal cells\nz=0.02 native layer", va="top", fontsize=12)
    axes[0, 4].text(0, .95, "Qualitative similarity comparison\nU(x,y,t) -> U(xi,eta)\n\nPaper contour levels unavailable.\nNCFV: 25 common density levels.\nNo exact-solution error inferred.\nFinite-domain and discretization\neffects are not removed by scaling.\n\nSource: Kurganov & Tadmor\n(2002), Fig.3.3, p.594.", va="top", fontsize=12)
    for ax, name, record in zip(axes[1], NAMES, records):
        plane = data[name]
        valid = [v for v in levels if plane.min() < v < plane.max()]
        ax.contour(similarity, similarity, plane, levels=valid, colors="black", linewidths=.55)
        ax.set(xlim=(-bound, bound), ylim=(-bound, bound), aspect="equal",
               title=f"{name}\nt=0.25; max z density span={record['rho_z_span_max']:.3g}",
               xlabel=r"$\xi=(x-0.5)/t$", ylabel=r"$\eta=(y-0.5)/t$")
    fig.suptitle("RP1 = KT Configuration 3: density wave patterns in self-similar coordinates", fontsize=16)
    fig.savefig(OUTPUT / "RP1_vs_KT2002_Fig3p3_similarity.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for name in NAMES:
        for ax, profile in zip(axes, (data[name][cut_index, :], data[name][:, cut_index])):
            ax.plot(x, profile, label=name, linewidth=1.3)
    for ax, label in zip(axes, ("Horizontal profile at y=0.9", "Vertical profile at x=0.9")):
        ax.axvline(reference, color="black", linestyle=":", label="Mass-jump shock position")
        ax.set(xlim=(reference-.04, reference+.04), ylim=(.45, 1.65), title=label,
               xlabel="x or y", ylabel="density")
        ax.grid(alpha=.2)
    axes[1].legend(fontsize=8)
    fig.suptitle("RP1 at t=0.25, z=0.02: isolated outer-shock profiles")
    fig.savefig(OUTPUT / "RP1_outer_shock_profiles.png", dpi=170)
    plt.close(fig)
    pdf = OUTPUT / "inputs/Kurganov_Tadmor_2002.pdf"
    summary = dict(paper=dict(title="Solution of two-dimensional Riemann problems for gas dynamics without Riemann problem solvers",
                              authors="A. Kurganov and E. Tadmor", year=2002, doi="10.1002/num.10025",
                              configuration=3, figure="3.3", printed_page=594, pdf_page=11,
                              time=paper_t, gridpoints=[400, 400], cfl=.475, pdf_sha256=hashlib.file_digest(pdf.open("rb"),"sha256").hexdigest()),
                   ncfv=dict(time=t, primal_cells=[800, 800, 5], independent_nodes=[801, 801, 5], cfl=.5,
                             slice_z=.02, initial_states_match=True, contour_levels=levels),
                   comparison="Qualitative self-similar wave shapes; paper supplies no exact field or contour values",
                   derived_mass_jump_shock_speed=shock_speed, derived_shock_position_at_t025=reference,
                   derived_shock_position_at_t03=.5+shock_speed*paper_t,
                   profile_definition="Density dual means, native z=0.02. 10%-90% crossings nearest derived outer shock; position reference is not an exact full 2D solution.",
                   metrics=records)
    (OUTPUT / "comparison_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
