"""v3 copy for the Supporting Information figures, with the primary/secondary
palette of the v3 manuscript.  Only the SI functions are run; the main-text
figure functions are left unchanged and are not called.
"""
"""Generate the Al-Cu material-validation and timestep-audit figures.

All curves are read from raw solver/CALPHAD output. Literature points are
transcribed from the cited open-access papers and are explicitly distinguished
from GPK predictions and one-point calibrations.
"""
from pathlib import Path
import csv
import os
import re

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from gpk_data import output_dir

mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=["#1F4FB4", "#D62020", "#7B2FA8",
                                                     "#E0A100", "#444444", "#7EA6E0"])
YELLOWRED = mpl.colors.LinearSegmentedColormap.from_list(
    "yellowred", [(0.00, "#FFFBE6"), (0.35, "#FFD966"), (0.70, "#F07A20"), (1.00, "#B01010")])

ROOT = Path(__file__).resolve().parents[1]
MS = ROOT / "paper"
OUT = Path(os.environ.get("GPK_ALCU_FIGURE_OUT", MS))
OUT.mkdir(parents=True, exist_ok=True)
PF = output_dir("gpk_pf_comparison")
REX = output_dir("al4cu_validation") / "recrystallisation_2d"

mpl.rcParams.update({
    "font.family": "Arial", "font.size": 9.2,
    "axes.labelsize": 9.3, "axes.titlesize": 9.8,
    "xtick.labelsize": 8.4, "ytick.labelsize": 8.4,
    "legend.fontsize": 8.0, "pdf.fonttype": 42,
    "axes.linewidth": 0.8, "lines.linewidth": 1.7,
})

SI_TEXT_BOX = dict(boxstyle="round,pad=0.16", fc="white", ec="#CBD5E1",
                   lw=0.55, alpha=0.97)
SI_LEGEND = dict(frameon=True, facecolor="white", edgecolor="#CBD5E1",
                 framealpha=0.97, fancybox=False)


def load_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def rex_curve():
    t, x = [], []
    for path in sorted(REX.glob("snap_*.csv")):
        with path.open(encoding="utf-8") as f:
            header = [next(f) for _ in range(5)]
        m = re.search(r"time=([0-9.eE+-]+)", header[1])
        time = float(m.group(1))
        arr = np.loadtxt(path, comments="#", usecols=(3,))
        t.append(time)
        x.append(np.mean(arr != 0))
    return np.asarray(t), np.asarray(x)


def panel_label(ax, s):
    ax.text(-0.17, 1.08, s, transform=ax.transAxes, weight="bold", fontsize=10,
            va="top", ha="left")


def main_figure():
    cal = load_csv(MS / "calphad" / "al4cu_phase_fractions.csv")
    temp = np.array([float(r["temperature_K"]) for r in cal])
    fcc = np.array([float(r["stable_fcc"]) for r in cal])
    theta = np.array([float(r["stable_theta"]) for r in cal])
    rex_t, rex_x = rex_curve()
    met = load_csv(PF / "snapshot_metrics.csv")
    mt = np.array([float(r["time"]) for r in met])
    corr = np.array([float(r["area_correlation"]) for r in met])
    agreement = np.array([float(r["pixel_agreement"]) for r in met])
    bf1 = np.array([float(r["boundary_f1_2px"]) for r in met])
    sweep = load_csv(PF / "timestep_sweep.csv")
    pf = [r for r in sweep if r["method"] == "explicit_phase_field"]
    pdt = np.array([float(r["dt_seconds"]) for r in pf])
    err = np.array([float(r["relative_radius2_error"]) if r["relative_radius2_error"] != "inf" else np.nan for r in pf])
    accepted = np.array([int(r["accepted"]) == 1 for r in pf])
    gpk = [r for r in sweep if r["method"] == "gpk_clipped"]
    requested = np.array([float(r["dt_seconds"]) for r in gpk])
    used = np.array([float(r["max_abs_eta"]) for r in gpk])

    fig, ax = plt.subplots(2, 2, figsize=(7.2, 5.65), constrained_layout=True)
    a, b, c, d = ax.flat

    a.plot(temp, fcc, color="#2b6cb0", label=r"$\alpha$-Al (FCC)")
    a.plot(temp, theta, color="#c05621", label=r"$\theta$-Al$_2$Cu")
    a.axvline(623.15, color="0.45", ls="--", lw=1)
    a.text(628, 0.57, r"350 $^\circ$C anneal", rotation=90, va="center", color="0.35")
    a.set(xlabel="Temperature (K)", ylabel="Equilibrium phase fraction", ylim=(-0.02, 1.02))
    a.legend(frameon=False, loc="center right")
    panel_label(a, "a")

    b.plot(rex_t, rex_x, "-o", ms=3.2, color="#7B2FA8", label="GPK")
    b.errorbar([300], [1.0], xerr=[[0], [0]], yerr=[[0.015], [0]], fmt="s",
               ms=5, color="#9b2c2c", label="experiment: fully RX at 5 min")
    b.axvline(300, color="#9b2c2c", ls=":", lw=1)
    b.text(0.03, 0.70, "one-point mobility calibration", transform=b.transAxes,
           fontsize=8.0, color="0.28")
    b.set(xlabel="Annealing time at 623 K (s)", ylabel="Recrystallised fraction", ylim=(-0.02, 1.04))
    b.legend(frameon=False, loc="lower right", fontsize=7.9)
    panel_label(b, "b")

    c.plot(mt, corr, "-o", ms=3, label="grain-area correlation")
    c.plot(mt, agreement, "-s", ms=3, label="label agreement")
    c.plot(mt, bf1, "-^", ms=3, label="boundary F1")
    c.axvline(1.0, color="0.35", ls=":", lw=1)
    c.set(xlabel="Physical time (s)", ylabel="GPK-PF agreement", ylim=(0.68, 1.015))
    c.legend(frameon=False, loc="lower left")
    panel_label(c, "c")

    d.loglog(pdt[accepted], pdt[accepted], "o-", color="#2b6cb0", label="explicit PF: accepted")
    d.loglog(pdt[~accepted], pdt[~accepted], "x", ms=6, mew=1.5,
             color="#c53030", label="explicit PF: inaccurate/unbounded")
    d.loglog(requested, used, "s-", color="#7B2FA8", label="GPK: accepted after clipping")
    d.loglog(requested, requested, ls=":", color="0.55", lw=1, label="requested = accepted")
    first_failed = float(np.min(pdt[~accepted]))
    step_ratio = float(np.max(used) / first_failed)
    d.axvspan(first_failed, float(np.max(pdt)), color="#c53030", alpha=0.07)
    d.annotate(rf"{step_ratio:.1f}$\times$", xy=(0.0275, 0.0275), xytext=(0.047, 0.010),
               arrowprops=dict(arrowstyle="->", color="0.25", lw=0.8), fontsize=8)
    d.text(first_failed * 1.08, 0.00035, "PF failure\nregime", color="#9b2c2c", fontsize=8.0,
           bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="#D6A6A6", alpha=0.96))
    d.set(xlabel="Requested physical timestep (s)", ylabel=r"Accepted timestep $\Delta t$ (s)")
    d.set_ylim(3e-4, 0.4)
    d.legend(frameon=False, loc="upper left", ncol=1, fontsize=7.8)
    panel_label(d, "d")

    fig.savefig(OUT / "fig11_al4cu_material_validation.pdf", bbox_inches="tight")
    plt.close(fig)


def supporting_figure():
    # Rodríguez-Veiga et al., J. Alloys Compd. 757 (2018) 504–519, Table 1.
    hours = np.array([18, 30, 120, 168], float)
    f_tdd = np.array([0.3, 1.6, 1.0, np.nan])
    e_tdd = np.array([0.1, 0.4, 0.2, np.nan])
    f_tp = np.array([np.nan, 0.6, 0.8, 1.0])
    e_tp = np.array([np.nan, 0.2, 0.1, 0.5])
    ar_tp = np.array([np.nan, 59.5, 28.0, 39.0])
    arerr = np.array([np.nan, 8.6, 2.4, 6.6])
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
    a.errorbar(hours, f_tdd, yerr=e_tdd, fmt="o-", capsize=2.5, label=r"experiment $\theta''$")
    a.errorbar(hours, f_tp, yerr=e_tp, fmt="s-", capsize=2.5, label=r"experiment $\theta'$")
    a.axhline(5.10, color="0.3", ls="--", label=r"CALPHAD equilibrium $\theta$ at 473 K")
    a.set(xlabel="Ageing time at 453 K (h)", ylabel="Precipitate volume fraction (%)", ylim=(0, 5.8))
    a.legend(**SI_LEGEND)
    panel_label(a, "a")
    b.errorbar(hours, ar_tp, yerr=arerr, fmt="s-", capsize=2.5, color="#444444", label=r"experiment $\theta'$")
    b.fill_between([95, 145], 20, 30, color="#E0A100", alpha=0.22,
                   label="published phase-field range")
    b.text(0.04, 0.05, "Local voxel verification ends at 0.5 ks;\npopulation comparison spans 120--168 h",
           transform=b.transAxes, fontsize=8.1, color="#333333",
           bbox=SI_TEXT_BOX)
    b.set(xlabel="Ageing time at 453 K (h)", ylabel=r"$\theta'$ diameter/thickness", ylim=(0, 72))
    b.legend(**SI_LEGEND, loc="upper right")
    panel_label(b, "b")
    fig.savefig(OUT / "figS10_al4cu_precipitate_targets.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    supporting_figure()
    print("Wrote figS10_al4cu_precipitate_targets.pdf")
