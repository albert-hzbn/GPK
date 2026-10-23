#!/usr/bin/env python3
"""
plot_theory_foundations.py — summary chart for the foundational GPK theory
validation suite (demos/validate_theory_foundations.cpp):
T2 (Gauss-Bonnet shape-independence), T3 (von Neumann-Mullins), T4 (Mullins
perturbation decay), T6 (convergence order), T7 (magnetic perimeter-scaling),
T8 (combined energy monotonicity). Values are hardcoded from the suite's own
printed, already-validated output (12/12 checks pass) -- this script is a
visualization of a fixed, recorded result set, not a live re-derivation.

Writes: build/output/validate_theory_foundations/theory_foundations_summary.png
"""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
OUTROOT = ROOT / "build" / "output" / "validate_theory_foundations"
OUTROOT.mkdir(parents=True, exist_ok=True)

DARK_BG, DARK_AX, SPINE_C, TEXT_C, TITLE_C = "#0d0d0d", "#111111", "#444444", "#cccccc", "#ffcc44"
COLORS = ["#4fc3f7", "#ff8a65", "#69f0ae", "#ce93d8", "#ffcc02"]


def style_ax(ax, title=""):
    ax.set_facecolor(DARK_AX)
    for sp in ax.spines.values():
        sp.set_edgecolor(SPINE_C)
    ax.tick_params(colors=TEXT_C, labelsize=8)
    ax.grid(True, color="#333333", linewidth=0.5)
    if title:
        ax.set_title(title, color=TITLE_C, fontsize=10, fontweight="bold")


def lbl(ax, xlabel="", ylabel=""):
    ax.set_xlabel(xlabel, color=TEXT_C, fontsize=9)
    ax.set_ylabel(ylabel, color=TEXT_C, fontsize=9)


def main():
    fig = plt.figure(figsize=(17, 11))
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle("GPK Foundational Theory Validation — 12/12 Checks Pass\n"
                 "Classical, shape-/model-independent results, including two magnetic-driving-force checks",
                 color="white", fontsize=15, fontweight="bold")
    gs = fig.add_gridspec(2, 3, top=0.87, bottom=0.08, hspace=0.4, wspace=0.3)

    # T2: Gauss-Bonnet shape-independent area law.
    ax = fig.add_subplot(gs[0, 0])
    shapes = ["square", "ellipse"]
    meas = [-1.631e-8, -1.688e-8]
    theory = [-2.036e-8, -2.036e-8]
    x = np.arange(2)
    ax.bar(x-0.18, meas, 0.36, label="measured", color=COLORS[0])
    ax.bar(x+0.18, theory, 0.36, label="theory (shape-independent)", color=COLORS[1])
    ax.set_xticks(x); ax.set_xticklabels(shapes, color=TEXT_C)
    lbl(ax, "", "dA/dt [m^2/s]")
    style_ax(ax, "T2: Gauss-Bonnet shape-independent\narea-loss law")
    ax.legend(fontsize=7, facecolor=DARK_AX, edgecolor=SPINE_C, labelcolor=TEXT_C)

    # T3: von Neumann-Mullins.
    ax = fig.add_subplot(gs[0, 1])
    n = np.array([4,5,6,7,8])
    meas3 = np.array([-1.302e-8,-7.266e-9,4.07e-10,4.950e-9,1.424e-8])
    th3 = (np.pi/3)*1e-8*0.324*(n-6)
    ax.plot(n, meas3, "o-", color=COLORS[0], label="measured (bin mean)")
    ax.plot(n, th3, "--", color=COLORS[1], label="theory (pi/3)*M*sigma*(n-6)")
    ax.axhline(0, color=TEXT_C, lw=0.5)
    lbl(ax, "neighbour count n", "dA_i/dt [m^2/s]")
    style_ax(ax, "T3: von Neumann-Mullins law\n(r=0.91 individual-grain correlation)")
    ax.legend(fontsize=7, facecolor=DARK_AX, edgecolor=SPINE_C, labelcolor=TEXT_C)

    # T4: Mullins perturbation decay.
    ax = fig.add_subplot(gs[0, 2])
    t4 = np.array([0.0, 0.0403, 0.0809, 0.1201, 0.1600])
    A4 = np.array([9.92, 7.45, 4.99, 3.33, 2.13])
    A4_theory = 9.92*np.exp(-7.807*t4)
    ax.semilogy(t4, A4, "o-", color=COLORS[0], label="measured")
    ax.semilogy(t4, A4_theory, "--", color=COLORS[1], label="theory exp(-M*sigma*k^2*t)")
    lbl(ax, "t [s]", "amplitude [cells]")
    style_ax(ax, "T4: Mullins perturbation\nlinear-stability decay")
    ax.legend(fontsize=7, facecolor=DARK_AX, edgecolor=SPINE_C, labelcolor=TEXT_C)

    # T6: convergence order.
    ax = fig.add_subplot(gs[1, 0])
    dx = np.array([0.4e-6, 0.2e-6, 0.1e-6])
    err = np.array([14.80, 17.51, 8.32])
    ax.loglog(dx, err, "o-", color=COLORS[2])
    lbl(ax, "dx [m]", "error [%]")
    style_ax(ax, "T6: convergence-order study\n(fitted p=0.415)")

    # T7: magnetic perimeter-scaling law.
    ax = fig.add_subplot(gs[1, 1])
    shapes7 = ["circle", "square"]
    meas7 = [2.229e-8, 2.388e-8]
    theory7 = [2.177e-8, 2.702e-8]
    x = np.arange(2)
    ax.bar(x-0.18, meas7, 0.36, label="measured", color=COLORS[0])
    ax.bar(x+0.18, theory7, 0.36, label="theory (perimeter-scaled)", color=COLORS[1])
    ax.set_xticks(x); ax.set_xticklabels(shapes7, color=TEXT_C)
    lbl(ax, "", "dA/dt [m^2/s]")
    style_ax(ax, "T7: magnetic driving-force\nperimeter-scaling law")
    ax.legend(fontsize=7, facecolor=DARK_AX, edgecolor=SPINE_C, labelcolor=TEXT_C)

    # T8: combined energy monotonicity (start/end bar, illustrative trend).
    ax = fig.add_subplot(gs[1, 2])
    t8 = np.linspace(0, 1, 50)
    E8 = -0.4152 + (-0.4668 - (-0.4152)) * (1 - np.exp(-3*t8))  # illustrative smooth decay matching endpoints
    ax.plot(t8, E8, "-", color=COLORS[0])
    ax.axhline(-0.4152, color=TEXT_C, lw=0.6, ls=":")
    ax.axhline(-0.4668, color=COLORS[1], lw=0.6, ls=":")
    lbl(ax, "relaxation progress (illustrative)", "E_total [J/m]")
    style_ax(ax, "T8: combined energy functional\n(net -12.4%, 0/250 violations)")

    out_path = OUTROOT / "theory_foundations_summary.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
