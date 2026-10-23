#!/usr/bin/env python3
"""
GPK Literature Validation: Grain Growth and Recrystallisation
=============================================================
Reads precomputed C++ CSV output, fits kinetic models,
and produces a 2-page comparison figure against classical theory.

Physics validated
-----------------
Grain growth     Burke-Turnbull parabolic law   <R>^2 = R0^2 + K*t, K = 2*M*sigma/3
Recryst. 2D      JMAK site-saturated 2D         n_theory = 2
Recryst. 3D      JMAK site-saturated 3D         n_theory = 3

References
----------
Burke & Turnbull (1952) Prog. Metal Phys. 3, 220
Kolmogorov (1937) Izv. Akad. Nauk SSSR, Ser. Mat. 1, 355
Avrami (1939, 1940, 1941) J. Chem. Phys.
Johnson & Mehl (1939) Trans. AIME 135, 416
"""

import re
import sys
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── paths ────────────────────────────────────────────────────────────────────
REPO   = Path(__file__).resolve().parents[1]
BUILD  = REPO / "build"
OUT    = REPO / "build" / "output"

# ── colour palette (dark theme) ──────────────────────────────────────────────
BG   = "#0d0d0d"
AX   = "#111111"
SPL  = "#444444"
TXT  = "#cccccc"
TTL  = "#ffcc44"
C1   = "#4fc3f7"   # GPK data
C2   = "#ff8a65"   # theory
C3   = "#69f0ae"   # fit
C4   = "#ce93d8"   # secondary data
C5   = "#ffcc02"   # accent


# ═══════════════════════════════════════════════════════════════════════════
#  CSV I/O
# ═══════════════════════════════════════════════════════════════════════════

def read_csv(path):
    meta = {}
    with open(path) as f:
        for line in f:
            if not line.startswith('#'):
                break
            for key in ('time', 'dx', 'Nz', 'Ny', 'Nx'):
                m = re.search(rf'{key}=([\d.e+\-]+)', line)
                if m:
                    meta[key] = float(m.group(1))
    data = np.loadtxt(path, comments='#')
    if data.ndim == 1:
        data = data[np.newaxis, :]
    return {
        'x': data[:, 0], 'y': data[:, 1], 'z': data[:, 2],
        'grain_id': data[:, 3].astype(int),
        'alpha': data[:, 4], 'comp': data[:, 5],
        'meta': meta,
    }


def load_snaps(sim_dir):
    files = sorted(sim_dir.glob("snap_*.csv"))
    snaps = []
    for f in files:
        try:
            snaps.append(read_csv(f))
        except Exception as e:
            print(f"  Warning skipping {f.name}: {e}", file=sys.stderr, flush=True)
    return snaps


# ═══════════════════════════════════════════════════════════════════════════
#  Grain-property utilities
# ═══════════════════════════════════════════════════════════════════════════

def mean_radius(snap):
    gid  = snap['grain_id']
    dx   = snap['meta'].get('dx', 1e-6)
    Nz   = int(snap['meta'].get('Nz', 1))
    ids  = np.unique(gid); ids = ids[ids != 0]
    if len(ids) == 0:
        return 0.0, 0
    radii = []
    for g in ids:
        n = (gid == g).sum()
        r = (n / np.pi)**0.5 * dx if Nz == 1 else (3*n/(4*np.pi))**(1/3) * dx
        radii.append(r)
    return float(np.mean(radii)), len(ids)


def rex_fraction(snap):
    gid = snap['grain_id']
    return float((gid != 0).mean())


def radius_distribution(snap):
    gid  = snap['grain_id']
    dx   = snap['meta'].get('dx', 1e-6)
    Nz   = int(snap['meta'].get('Nz', 1))
    ids  = np.unique(gid); ids = ids[ids != 0]
    radii = []
    for g in ids:
        n = (gid == g).sum()
        r = (n / np.pi)**0.5 * dx if Nz == 1 else (3*n/(4*np.pi))**(1/3) * dx
        radii.append(r * 1e6)   # µm
    return np.array(radii)


# ═══════════════════════════════════════════════════════════════════════════
#  Axis styling
# ═══════════════════════════════════════════════════════════════════════════

def style(ax, title="", xlabel="", ylabel="", legend=False):
    ax.set_facecolor(AX)
    for sp in ax.spines.values():
        sp.set_edgecolor(SPL)
    ax.tick_params(colors=TXT, labelsize=9)
    ax.grid(True, color="#2a2a2a", linewidth=0.6, zorder=0)
    if title:
        ax.set_title(title, color=TTL, fontsize=11, fontweight='bold', pad=6)
    if xlabel:
        ax.set_xlabel(xlabel, color=TXT, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, color=TXT, fontsize=10)
    if legend:
        ax.legend(fontsize=9, facecolor=AX, edgecolor=SPL, labelcolor=TXT,
                  framealpha=0.85)


# ═══════════════════════════════════════════════════════════════════════════
#  GRAIN GROWTH VALIDATION
# ═══════════════════════════════════════════════════════════════════════════
# Normal grain growth is tested by the parabolic network relation
# <R>^2 - R0^2 = K*t. Unlike an isolated circle, the network coefficient K
# depends on the size statistic and topology; M*sigma is only a normalization.

def validate_grain_growth(fig, gs_row):
    sim_dir = OUT / "validation_grain_growth_dense_2d"
    snaps   = load_snaps(sim_dir)
    if not snaps:
        print("No grain growth snapshots found — skipping.")
        return

    M0     = 5.0e-11  # m^4/(J·s), production input
    sigma0 = 0.324    # J/m^2, production input
    K_scale = M0 * sigma0

    # Extract <R>(t) from CSV
    times, R_vals, N_vals = [], [], []
    for s in snaps:
        t = s['meta'].get('time', 0.0)
        R, N = mean_radius(s)
        if R > 0 and N > 1:
            times.append(t)
            R_vals.append(R * 1e6)   # µm
            N_vals.append(N)

    times  = np.array(times)
    R_vals = np.array(R_vals)
    N_vals = np.array(N_vals)

    # Fit R^2 = R0^2 + K*t  →  linear regression R^2 vs t
    R2 = R_vals**2
    slope, intercept, r_val, *_ = stats.linregress(times, R2)
    K_fit = slope * 1e-12    # µm^2/s → m^2/s

    t_fine   = np.linspace(0, times[-1], 300)
    R_fit_at_t = np.sqrt(np.maximum(intercept + slope * times, 0))
    residual_pct = (R_vals - R_fit_at_t) / R_fit_at_t * 100

    # ── Panel A: R^2 vs t ────────────────────────────────────────────────
    ax_R2 = fig.add_subplot(gs_row[0])
    R0_sq_fit = intercept
    ax_R2.scatter(times * 1e3, R2, s=40, c=C1, zorder=5, label='GPK simulation')
    ax_R2.plot(t_fine * 1e3,
               (slope * t_fine + R0_sq_fit),
               '--', color=C3, lw=2,
               label=f'Linear fit  K={slope/1e3:.3f} µm²/ms')
    style(ax_R2, r"Grain Growth: $\langle R\rangle^2$ vs $t$",
          r"Time  [ms]", r"$\langle R\rangle^2$  [µm²]", legend=True)
    ax_R2.text(0.97, 0.07,
               f'$K/(M\\sigma)$ = {K_fit / K_scale:.3f}\n'
               f'$R^2$ = {r_val**2:.4f}',
               transform=ax_R2.transAxes, ha='right', va='bottom',
               color=C3, fontsize=9,
               bbox=dict(boxstyle='round,pad=0.4', facecolor='#1a1a1a',
                         edgecolor=C3, alpha=0.9))

    # ── Panel B: % error vs t ────────────────────────────────────────────
    ax_err = fig.add_subplot(gs_row[1])
    ax_err.axhline(0, color=C2, lw=1, ls='--', alpha=0.5, label='Linear fit')
    ax_err.axhspan(-2, 2, alpha=0.10, color=C2, label='±2% band')
    ax_err.scatter(times * 1e3, residual_pct, s=40, c=C1, zorder=5)
    ax_err.plot(times * 1e3, residual_pct, '-', color=C1, lw=1.2, alpha=0.5)
    ax_err.set_ylim(-5, 5)
    style(ax_err, r"Residual from Parabolic Fit",
          r"Time  [ms]", r"$(R-R_\mathrm{fit})/R_\mathrm{fit}$  [%]",
          legend=True)
    mean_abs = np.mean(np.abs(residual_pct))
    ax_err.text(0.97, 0.97, f'Mean |err| = {mean_abs:.1f}%',
                transform=ax_err.transAxes, ha='right', va='top',
                color=TXT, fontsize=9,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#1a1a1a',
                          edgecolor=SPL, alpha=0.9))

    # ── Panel C: grain-size histograms t_initial vs t_final ──────────────
    ax_hist = fig.add_subplot(gs_row[2])
    r_init = radius_distribution(snaps[0])
    r_fin  = radius_distribution(snaps[-1])
    bins = np.linspace(0, max(r_fin.max(), r_init.max()) * 1.05, 16)
    ax_hist.hist(r_init, bins=bins, color=C1, alpha=0.6, label=f't = {snaps[0]["meta"]["time"]*1e3:.0f} ms',
                 edgecolor=BG)
    ax_hist.hist(r_fin,  bins=bins, color=C4, alpha=0.7, label=f't = {snaps[-1]["meta"]["time"]*1e3:.0f} ms',
                 edgecolor=BG)
    style(ax_hist, "Grain-Size Distribution",
          r"Equiv. radius  [µm]", "Count", legend=True)

    # ── Panel D: number of grains vs t ───────────────────────────────────
    ax_N = fig.add_subplot(gs_row[3])
    ax_N.scatter(times * 1e3, N_vals, s=35, c=C4, zorder=5)
    ax_N.plot(times * 1e3, N_vals, '-', color=C4, lw=1.5, alpha=0.7)
    style(ax_N, "Grain Count vs Time",
          r"Time  [ms]", "Number of grains")

    return K_fit / K_scale, r_val**2, mean_abs


# ═══════════════════════════════════════════════════════════════════════════
#  RECRYSTALLISATION VALIDATION (JMAK / Avrami)
# ═══════════════════════════════════════════════════════════════════════════
# JMAK theory (Kolmogorov 1937, Avrami 1939-41, Johnson-Mehl 1939):
#   X(t) = 1 - exp(-k*t^n)
#   ln(-ln(1-X)) = n*ln(t) + ln(k)
# Site-saturated 2D: n = 2
# Site-saturated 3D: n = 3

def avrami_data(snaps, x_min=0.01, x_max=0.99):
    lnt, lnln = [], []
    t_all, x_all = [], []
    for s in snaps:
        t = s['meta'].get('time', 0.0)
        X = rex_fraction(s)
        t_all.append(t)
        x_all.append(X)
        if t > 0 and x_min < X < x_max:
            lnt.append(math.log(t))
            lnln.append(math.log(-math.log(1 - X)))
    return (np.array(lnt), np.array(lnln),
            np.array(t_all), np.array(x_all))


def validate_recrystallisation(fig, gs_row):
    # Load 2D and 3D datasets
    snaps_2d = load_snaps(OUT / "validation_rex_current_2d")
    snaps_3d = load_snaps(OUT / "validation_rex_current_3d")

    lnt2, lnln2, t2, x2 = avrami_data(snaps_2d)
    lnt3, lnln3, t3, x3 = avrami_data(snaps_3d)

    # Fits
    slope2, icpt2, r2_2, *_ = stats.linregress(lnt2, lnln2)
    slope3, icpt3, r2_3, *_ = stats.linregress(lnt3, lnln3)

    # ── Panel A: Avrami plot 2D ───────────────────────────────────────────
    ax_av2 = fig.add_subplot(gs_row[0])
    ax_av2.scatter(lnt2, lnln2, s=45, c=C1, zorder=5, label='GPK 2D simulation')
    t_fit2 = np.linspace(lnt2.min(), lnt2.max(), 100)
    ax_av2.plot(t_fit2, slope2 * t_fit2 + icpt2, '--',
                color=C3, lw=2, label=f'Fit:  n = {slope2:.2f}  (R²={r2_2**2:.3f})')
    # Theoretical n=2 line through centroid
    cx = lnt2.mean(); cy = lnln2.mean()
    ax_av2.plot(t_fit2, 2 * (t_fit2 - cx) + cy,
                '-', color=C2, lw=1.5, alpha=0.8,
                label='Theory: n = 2  (Kolmogorov-Avrami)')
    style(ax_av2, "Avrami Plot — 2D site-saturated",
          r"$\ln(t)$  [ln s]", r"$\ln(-\ln(1-X))$", legend=True)
    ax_av2.text(0.03, 0.97,
                'Site-saturated 2D\nTheory: n = 2',
                transform=ax_av2.transAxes, va='top', color=C2, fontsize=9,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#1a1a1a',
                          edgecolor=C2, alpha=0.9))

    # ── Panel B: Avrami plot 3D ───────────────────────────────────────────
    ax_av3 = fig.add_subplot(gs_row[1])
    ax_av3.scatter(lnt3, lnln3, s=45, c=C4, zorder=5, label='GPK 3D simulation')
    t_fit3 = np.linspace(lnt3.min(), lnt3.max(), 100)
    ax_av3.plot(t_fit3, slope3 * t_fit3 + icpt3, '--',
                color=C3, lw=2, label=f'Fit:  n = {slope3:.2f}  (R²={r2_3**2:.3f})')
    cx3 = lnt3.mean(); cy3 = lnln3.mean()
    ax_av3.plot(t_fit3, 3 * (t_fit3 - cx3) + cy3,
                '-', color=C2, lw=1.5, alpha=0.8,
                label='Theory: n = 3  (Kolmogorov-Avrami)')
    style(ax_av3, "Avrami Plot — 3D site-saturated",
          r"$\ln(t)$  [ln s]", r"$\ln(-\ln(1-X))$", legend=True)
    ax_av3.text(0.03, 0.97,
                'Site-saturated 3D\nTheory: n = 3',
                transform=ax_av3.transAxes, va='top', color=C2, fontsize=9,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#1a1a1a',
                          edgecolor=C2, alpha=0.9))

    # ── Panel C: X(t) vs t for 2D ────────────────────────────────────────
    ax_xt2 = fig.add_subplot(gs_row[2])
    # GPK data
    mask2 = t2 > 0
    ax_xt2.scatter(t2[mask2], x2[mask2] * 100, s=30, c=C1, zorder=5, label='GPK 2D')
    # JMAK analytical curve from fit: X = 1 - exp(-exp(n*ln(t)+b))
    t_fine2 = np.linspace(t2[mask2].min(), t2[mask2].max(), 300)
    X_fit2  = 1 - np.exp(-np.exp(slope2 * np.log(t_fine2) + icpt2))
    ax_xt2.plot(t_fine2, X_fit2 * 100, '--', color=C3, lw=2,
                label=f'JMAK fit (n={slope2:.2f})')
    # Theory n=2
    k2_th = np.exp(icpt2 + (2 - slope2) * np.mean(np.log(t_fine2)))
    X_th2 = 1 - np.exp(-k2_th * t_fine2**2)
    ax_xt2.plot(t_fine2, X_th2 * 100, '-', color=C2, lw=1.5, alpha=0.8,
                label='Theory n = 2')
    style(ax_xt2, r"Rex Fraction vs Time — 2D",
          r"Time  [s]", r"Rex fraction  $X$  [%]", legend=True)
    ax_xt2.set_ylim(-2, 105)

    # ── Panel D: X(t) vs t for 3D ────────────────────────────────────────
    ax_xt3 = fig.add_subplot(gs_row[3])
    mask3 = t3 > 0
    ax_xt3.scatter(t3[mask3], x3[mask3] * 100, s=30, c=C4, zorder=5, label='GPK 3D')
    t_fine3 = np.linspace(t3[mask3].min(), t3[mask3].max(), 300)
    X_fit3  = 1 - np.exp(-np.exp(slope3 * np.log(t_fine3) + icpt3))
    ax_xt3.plot(t_fine3, X_fit3 * 100, '--', color=C3, lw=2,
                label=f'JMAK fit (n={slope3:.2f})')
    k3_th = np.exp(icpt3 + (3 - slope3) * np.mean(np.log(t_fine3)))
    X_th3 = 1 - np.exp(-k3_th * t_fine3**3)
    ax_xt3.plot(t_fine3, X_th3 * 100, '-', color=C2, lw=1.5, alpha=0.8,
                label='Theory n = 3')
    style(ax_xt3, r"Rex Fraction vs Time — 3D",
          r"Time  [s]", r"Rex fraction  $X$  [%]", legend=True)
    ax_xt3.set_ylim(-2, 105)

    return slope2, r2_2, slope3, r2_3


# ═══════════════════════════════════════════════════════════════════════════
#  SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════════════════

def print_summary(k_ratio, gg_r2, gg_err, slope2, r2_2, slope3, r2_3):
    print()
    print("=" * 65)
    print("  GPK vs Literature — Validation Summary")
    print("=" * 65)
    print()
    print("  Grain Growth (Burke-Turnbull parabolic law)")
    print("  ─────────────────────────────────────────")
    print("  Network test:  <R>^2 = R0^2 + K*t")
    print("  M*sigma is a normalization scale, not an exact network prefactor")
    print(f"  K_fit/(M*sigma) = {k_ratio:.3f}   (parabolic law R²={gg_r2:.4f})")
    print(f"  Mean |fit residual| = {gg_err:.2f}%")
    print()
    print("  Recrystallisation (JMAK / Avrami-Kolmogorov theory)")
    print("  ──────────────────────────────────────────────────")
    print(f"  {'Case':<30} {'n_fit':>7} {'R²':>7} {'n_theory':>10} {'Δn':>8}")
    print(f"  {'-'*62}")
    print(f"  {'2D site-saturated':<30} {slope2:>7.3f} {r2_2**2:>7.4f} {'2':>10} {slope2-2:>+8.3f}")
    print(f"  {'3D site-saturated':<30} {slope3:>7.3f} {r2_3**2:>7.4f} {'3':>10} {slope3-3:>+8.3f}")
    print()
    print("  References:")
    print("   Kolmogorov (1937) Izv. Akad. Nauk SSSR 1, 355")
    print("   Avrami (1939-41) J. Chem. Phys. 7/8/9")
    print("   Johnson & Mehl (1939) Trans. AIME 135, 416")
    print("   Burke & Turnbull (1952) Prog. Metal Phys. 3, 220")
    print("=" * 65)
    print()


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    required = {
        "2-D grain growth": OUT / "validation_grain_growth_dense_2d",
        "2-D recrystallisation": OUT / "validation_rex_current_2d",
        "3-D recrystallisation": OUT / "validation_rex_current_3d",
    }
    missing = [f"{label}: {path}" for label, path in required.items()
               if not list(path.glob("snap_*.csv"))]
    if missing:
        print("Required simulation snapshots are missing; no plots were written.",
              file=sys.stderr)
        for item in missing:
            print(f"  - {item}", file=sys.stderr)
        print("Run the corresponding grain-growth and recrystallisation cases first, "
              "or use scripts/validate_literature.py for the self-contained gated checks.",
              file=sys.stderr)
        return 2

    # ── Figure 1: Grain Growth ────────────────────────────────────────────
    fig1 = plt.figure(figsize=(12, 9.5))
    fig1.patch.set_facecolor(BG)
    fig1.suptitle(
        "GPK Grain Growth — Validation Against Burke-Turnbull Parabolic Law",
        color="white", fontsize=14, fontweight='bold', y=1.00)
    gs1 = gridspec.GridSpec(2, 2, figure=fig1, wspace=0.30, hspace=0.36,
                            left=0.08, right=0.97, top=0.92, bottom=0.08)
    k_ratio, gg_r2, gg_err = validate_grain_growth(fig1, gs1)
    out1 = REPO / "figures" / "grain_growth_validation.png"
    fig1.savefig(out1, dpi=300, bbox_inches='tight', facecolor=BG)
    print(f"Saved -> {out1}")

    # ── Figure 2: Recrystallisation ───────────────────────────────────────
    fig2 = plt.figure(figsize=(12, 9.5))
    fig2.patch.set_facecolor(BG)
    fig2.suptitle(
        "GPK Recrystallisation — Validation Against JMAK/Avrami-Kolmogorov Theory",
        color="white", fontsize=14, fontweight='bold', y=1.00)
    gs2 = gridspec.GridSpec(2, 2, figure=fig2, wspace=0.30, hspace=0.36,
                            left=0.08, right=0.97, top=0.92, bottom=0.08)
    slope2, r2_2, slope3, r2_3 = validate_recrystallisation(fig2, gs2)
    out2 = REPO / "figures" / "recrystallisation_validation.png"
    fig2.savefig(out2, dpi=300, bbox_inches='tight', facecolor=BG)
    print(f"Saved -> {out2}")

    print_summary(k_ratio, gg_r2, gg_err, slope2, r2_2, slope3, r2_3)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
