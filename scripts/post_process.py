#!/usr/bin/env python3
"""
post_process.py — GPK Microstructure Post-Processing
=====================================================
Reads CSV column snapshots produced by the C++ GPK demos and computes
per-grain statistics, then generates a multi-panel analysis figure.

Usage:
    python3 post_process.py                       # auto-detect all demo outputs
    python3 post_process.py --sim grain_growth    # single simulation
    python3 post_process.py --dir output/solidification --out sol_analysis.png

Grain properties measured per snapshot:
  • Count, mean/std equivalent radius
  • Volume (area in 2-D), aspect ratio (via inertia tensor)
  • Grain boundary length (interface perimeter / area)
  • Volume-weighted size distribution (histogram)
  • Avrami parameter for recrystallisation
  • Solute distribution statistics for solidification / Ostwald
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import ndimage

BUILD_ROOT = Path(__file__).parent.parent / "build" / "output"

# ── CSV reader ──────────────────────────────────────────────────────────────

def read_csv(path):
    """
    Read a GPK CSV snapshot.
    Returns a dict with keys: x, y, z, grain_id, alpha, composition,
                               temperature, orientation, meta (dict of header info)
    """
    meta = {}
    with open(path) as f:
        for line in f:
            if not line.startswith('#'):
                break
            m = re.search(r'snap=(\d+)', line)
            if m: meta['snap'] = int(m.group(1))
            m = re.search(r'time=([\d.e+\-]+)', line)
            if m: meta['time'] = float(m.group(1))
            m = re.search(r'Nz=(\d+)', line)
            if m: meta['Nz'] = int(m.group(1))
            m = re.search(r'Ny=(\d+)', line)
            if m: meta['Ny'] = int(m.group(1))
            m = re.search(r'Nx=(\d+)', line)
            if m: meta['Nx'] = int(m.group(1))
            m = re.search(r'dx=([\d.e+\-]+)', line)
            if m: meta['dx'] = float(m.group(1))

    data = np.loadtxt(path, comments='#')
    if data.ndim == 1:
        data = data[np.newaxis, :]

    return {
        'x':           data[:, 0],
        'y':           data[:, 1],
        'z':           data[:, 2],
        'grain_id':    data[:, 3].astype(int),
        'alpha':       data[:, 4],
        'composition': data[:, 5],
        'temperature': data[:, 6],
        'orientation': data[:, 7],
        'meta':        meta,
    }


# ── grain property measurement ─────────────────────────────────────────────

def grain_properties(snap):
    """
    Compute per-grain properties from a snapshot dict.

    Returns a dict of arrays (one entry per unique non-zero grain):
      grain_id, volume_m3 (or area_m2 in 2D), radius_eq, aspect_ratio,
      centroid_x, centroid_y, centroid_z, mean_composition
    """
    gid     = snap['grain_id']
    dx      = snap['meta'].get('dx', 1e-6)
    Nz      = snap['meta'].get('Nz', 1)
    labels  = np.unique(gid)
    labels  = labels[labels != 0]   # exclude matrix (phi=0)

    results = {k: [] for k in ('grain_id', 'n_cells', 'radius_eq',
                                'aspect_ratio', 'cx', 'cy', 'cz',
                                'mean_composition', 'mean_orientation')}

    for lab in labels:
        mask = gid == lab
        n    = mask.sum()

        vol_m  = n * dx**3 if Nz > 1 else n * dx**2
        R_eq   = (3*n/(4*np.pi))**(1/3) * dx if Nz > 1 else (n/np.pi)**0.5 * dx

        # Centroid
        cx = snap['x'][mask].mean()
        cy = snap['y'][mask].mean()
        cz = snap['z'][mask].mean()

        # Aspect ratio via inertia tensor (2D: 2×2, 3D: 3×3)
        rx = snap['x'][mask] - cx
        ry = snap['y'][mask] - cy
        rz = snap['z'][mask] - cz
        if Nz > 1:
            Ixx = (ry**2 + rz**2).mean()
            Iyy = (rx**2 + rz**2).mean()
            Izz = (rx**2 + ry**2).mean()
            Ixy = -(rx*ry).mean()
            Ixz = -(rx*rz).mean()
            Iyz = -(ry*rz).mean()
            I   = np.array([[Ixx, Ixy, Ixz],
                             [Ixy, Iyy, Iyz],
                             [Ixz, Iyz, Izz]])
        else:
            Ixx = (ry**2).mean()
            Iyy = (rx**2).mean()
            Ixy = -(rx*ry).mean()
            I   = np.array([[Ixx, Ixy],
                             [Ixy, Iyy]])

        eigvals = np.linalg.eigvalsh(I)
        eigvals = np.sort(np.abs(eigvals) + 1e-30)
        aspect  = float(eigvals[-1] / eigvals[0]) if eigvals[0] > 0 else 1.0

        results['grain_id'].append(lab)
        results['n_cells'].append(n)
        results['radius_eq'].append(R_eq)
        results['aspect_ratio'].append(aspect)
        results['cx'].append(cx); results['cy'].append(cy); results['cz'].append(cz)
        results['mean_composition'].append(snap['composition'][mask].mean())
        results['mean_orientation'].append(snap['orientation'][mask].mean())

    for k in results:
        results[k] = np.array(results[k])
    return results


def boundary_length(snap):
    """
    Estimate total grain boundary length (2-D: perimeter; 3-D: surface area)
    by counting interface-crossing neighbour pairs.
    Returns value in metres (or m^2 for 3-D).
    """
    dx   = snap['meta'].get('dx', 1e-6)
    Nz   = snap['meta'].get('Nz', 1)
    Ny   = snap['meta'].get('Ny', 1)
    Nx   = snap['meta'].get('Nx', 1)
    gid  = snap['grain_id'].reshape(Nz, Ny, Nx)

    interfaces = 0
    # x-direction
    interfaces += (gid[:, :, 1:] != gid[:, :, :-1]).sum()
    # y-direction
    interfaces += (gid[:, 1:, :] != gid[:, :-1, :]).sum()
    # z-direction
    if Nz > 1:
        interfaces += (gid[1:, :, :] != gid[:-1, :, :]).sum()

    # Each interface counts as 1 face of area dx^2 (3D) or edge of length dx (2D)
    if Nz > 1:
        return interfaces * dx**2   # m²
    else:
        return interfaces * dx      # m


def avrami_stats(snaps_data):
    """
    Compute Avrami (JMAK) statistics from a sequence of recrystallisation
    snapshots.  Returns (log_t, log_neg_log_1mX, slope, intercept).
    """
    log_t, ln_ln = [], []
    for sd in snaps_data:
        t = sd['meta'].get('time', 0)
        if t <= 0: continue
        all_ids = sd['grain_id']
        # fraction with phi != 0 = rex fraction
        X = (all_ids != 0).mean()
        if 0.01 < X < 0.999:
            log_t.append(np.log(t))
            ln_ln.append(np.log(-np.log(1 - X)))

    if len(log_t) < 2:
        return np.array(log_t), np.array(ln_ln), 0.0, 0.0

    coeffs = np.polyfit(log_t, ln_ln, 1)
    return np.array(log_t), np.array(ln_ln), float(coeffs[0]), float(coeffs[1])


# ── plotting helpers ────────────────────────────────────────────────────────

DARK_BG   = "#0d0d0d"
DARK_AX   = "#111111"
SPINE_C   = "#444444"
TEXT_C    = "#cccccc"
TITLE_C   = "#ffcc44"
COLORS    = ["#4fc3f7", "#69f0ae", "#ff8a65", "#ce93d8",
             "#80cbc4", "#ffcc02", "#f48fb1"]


def style_ax(ax, title=""):
    ax.set_facecolor(DARK_AX)
    for sp in ax.spines.values():
        sp.set_edgecolor(SPINE_C)
    ax.tick_params(colors=TEXT_C, labelsize=9)
    ax.grid(True, color="#333333", linewidth=0.5)
    if title:
        ax.set_title(title, color=TITLE_C, fontsize=11, fontweight='bold')


def lbl(ax, xlabel="", ylabel=""):
    ax.set_xlabel(xlabel, color=TEXT_C, fontsize=10)
    ax.set_ylabel(ylabel, color=TEXT_C, fontsize=10)


# ── per-simulation analysis panels ─────────────────────────────────────────

def analyse_grain_growth(snaps, ax_hist, ax_radt, ax_gb):
    """Radius distribution, R vs t, grain boundary length vs t."""
    times, r_means, r_stds, gb_lengths, n_grains = [], [], [], [], []

    for sd in snaps:
        props  = grain_properties(sd)
        t      = sd['meta'].get('time', 0)
        if len(props['radius_eq']) == 0: continue
        times.append(t * 1e3)   # ms
        r_means.append(props['radius_eq'].mean() * 1e6)
        r_stds.append(props['radius_eq'].std()  * 1e6)
        gb_lengths.append(boundary_length(sd) * 1e6)
        n_grains.append(len(props['grain_id']))

    # Histogram of radii at last snapshot
    last = snaps[-1]
    props_last = grain_properties(last)
    if len(props_last['radius_eq']) > 1:
        ax_hist.hist(props_last['radius_eq'] * 1e6, bins=15,
                     color=COLORS[0], edgecolor="#0d0d0d", alpha=0.85)
        lbl(ax_hist, "Equivalent radius [µm]", "Count")
        style_ax(ax_hist, "Grain size distribution (final snap)")

    # R vs t
    if times:
        ax_radt.errorbar(times, r_means, yerr=r_stds, fmt='o-',
                         color=COLORS[0], ecolor=COLORS[2], lw=2, ms=5,
                         label=r"$\langle R \rangle ± σ$")
        lbl(ax_radt, "Time [ms]", "Radius [µm]")
        style_ax(ax_radt, "Mean grain radius vs time")
        ax_radt.legend(fontsize=9, facecolor=DARK_AX, edgecolor=SPINE_C,
                       labelcolor=TEXT_C)

    # Grain boundary length vs t
    if times:
        ax_gb.plot(times, gb_lengths, 's-', color=COLORS[1], lw=2, ms=5)
        lbl(ax_gb, "Time [ms]", "GB length [µm]")
        style_ax(ax_gb, "Total grain boundary length")


def analyse_solidification(snaps, ax_sf, ax_cmax, ax_hist):
    """Solid fraction, max solute, and composition distribution."""
    times, sf, cmax_vals = [], [], []
    for sd in snaps:
        t = sd['meta'].get('time', 0)
        times.append(t * 1e3)
        c = sd['composition']
        gid = sd['grain_id']
        sf.append((gid == 1).mean() * 100)
        cmax_vals.append(c.max())

    ax_sf.plot(times, sf, 'o-', color=COLORS[1], lw=2, ms=5)
    lbl(ax_sf, "Time [ms]", "Solid fraction [%]")
    style_ax(ax_sf, "Solid fraction vs time")

    ax_cmax.plot(times, np.array(cmax_vals) * 100, 's-', color=COLORS[2], lw=2, ms=5)
    lbl(ax_cmax, "Time [ms]", "Max solute [mol%]")
    style_ax(ax_cmax, "Max solute enrichment vs time")

    # Composition histogram at last snapshot
    c_last = snaps[-1]['composition']
    gid_last = snaps[-1]['grain_id']
    c_solid  = c_last[gid_last == 1]
    c_liquid = c_last[gid_last == 0]
    if len(c_solid) > 1:
        ax_hist.hist(c_solid  * 100, bins=20, alpha=0.7,
                     color=COLORS[0], label='Solid', edgecolor='#0d0d0d')
    if len(c_liquid) > 1:
        ax_hist.hist(c_liquid * 100, bins=20, alpha=0.7,
                     color=COLORS[2], label='Liquid', edgecolor='#0d0d0d')
    lbl(ax_hist, "Composition [mol%]", "Count")
    style_ax(ax_hist, "Composition distribution (final)")
    ax_hist.legend(fontsize=9, facecolor=DARK_AX, edgecolor=SPINE_C,
                   labelcolor=TEXT_C)


def analyse_recrystallisation(snaps, ax_avrami, ax_rf, ax_hist):
    """Avrami plot, rex fraction vs t, grain size distribution."""
    log_t, ln_ln, slope, intercept = avrami_stats(snaps)

    if len(log_t) >= 2:
        ax_avrami.plot(log_t, ln_ln, 'D', color=COLORS[3], ms=7,
                       label='GPK data')
        t_fit = np.linspace(log_t[0], log_t[-1], 100)
        ax_avrami.plot(t_fit, slope * t_fit + intercept, '--',
                       color=COLORS[2], lw=2,
                       label=f'slope n = {slope:.2f}')
        ax_avrami.legend(fontsize=9, facecolor=DARK_AX, edgecolor=SPINE_C,
                         labelcolor=TEXT_C)
    lbl(ax_avrami, r"$\ln(t)$", r"$\ln(-\ln(1-X))$")
    style_ax(ax_avrami, "Avrami (JMAK) plot")

    # Rex fraction vs time
    times, rf = [], []
    for sd in snaps:
        t = sd['meta'].get('time', 0)
        X = (sd['grain_id'] != 0).mean()
        times.append(t); rf.append(X * 100)
    ax_rf.plot(times, rf, 'o-', color=COLORS[3], lw=2, ms=5)
    lbl(ax_rf, "Time [s]", "Rex fraction [%]")
    style_ax(ax_rf, "Recrystallised fraction vs time")

    # Grain size distribution at last snap with any rex
    for sd in reversed(snaps):
        props = grain_properties(sd)
        if len(props['radius_eq']) > 1:
            ax_hist.hist(props['radius_eq'] * 1e6, bins=15,
                         color=COLORS[3], edgecolor='#0d0d0d', alpha=0.85)
            lbl(ax_hist, "Grain radius [µm]", "Count")
            style_ax(ax_hist, "Rex grain size distribution")
            break


def analyse_ostwald(snaps, ax_r_vs_t, ax_nhist, ax_cfield):
    """Mean/max radius vs t, N(t), composition profile."""
    times, n_part, r_means, r_maxs = [], [], [], []

    for sd in snaps:
        t    = sd['meta'].get('time', 0) * 1e3
        gid  = sd['grain_id']
        labels = np.unique(gid); labels = labels[labels != 0]
        n = len(labels)
        if n == 0:
            times.append(t); n_part.append(0)
            r_means.append(0); r_maxs.append(0)
            continue
        props = grain_properties(sd)
        times.append(t)
        n_part.append(n)
        r_means.append(props['radius_eq'].mean() * 1e6)
        r_maxs.append(props['radius_eq'].max()  * 1e6)

    ax_r_vs_t.plot(times, r_means, 'o-', color=COLORS[0], lw=2, ms=5,
                   label=r'$\langle R \rangle$')
    ax_r_vs_t.plot(times, r_maxs, 's--', color=COLORS[2], lw=1.5, ms=5,
                   label=r'$R_{\max}$')
    lbl(ax_r_vs_t, "Time [ms]", "Radius [µm]")
    style_ax(ax_r_vs_t, "Particle radius vs time (coarsening)")
    ax2 = ax_r_vs_t.twinx()
    ax2.set_facecolor(DARK_AX)
    ax2.tick_params(colors=TEXT_C, labelsize=9)
    ax2.plot(times, n_part, '^:', color=COLORS[1], lw=1.5, ms=5)
    ax2.set_ylabel("N particles", color=TEXT_C, fontsize=10)
    ax2.spines['right'].set_edgecolor(SPINE_C)
    ax2.spines['left'].set_edgecolor(SPINE_C)
    ax_r_vs_t.legend(fontsize=9, facecolor=DARK_AX, edgecolor=SPINE_C,
                     labelcolor=TEXT_C)

    # Particle count histogram across snapshots
    all_n = [n for n in n_part if n > 0]
    if all_n:
        ax_nhist.plot(range(len(all_n)), all_n, 'o-',
                      color=COLORS[1], lw=2, ms=5)
        lbl(ax_nhist, "Snapshot index", "N surviving particles")
        style_ax(ax_nhist, "Particle dissolution curve")

    # Composition field at first snapshot (1-D profile through centre)
    sd0 = snaps[0]
    Ny  = sd0['meta'].get('Ny', 1)
    Nx  = sd0['meta'].get('Nx', 1)
    Nz  = sd0['meta'].get('Nz', 1)
    dx  = sd0['meta'].get('dx', 1e-6)
    c   = sd0['composition'].reshape(Nz, Ny, Nx)
    row = c[Nz//2, Ny//2, :]
    x_um = np.arange(Nx) * dx * 1e6
    ax_cfield.plot(x_um, row * 100, color=COLORS[2], lw=2)
    lbl(ax_cfield, "x [µm]", "Composition [mol%]")
    style_ax(ax_cfield, "Composition profile (1-D mid-row, t₀)")


# ── main figure assembly ────────────────────────────────────────────────────

def make_figure(sim_data, out_path):
    n_sims = len(sim_data)
    fig = plt.figure(figsize=(18, 5.5 * n_sims))
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle("GPK Microstructure Post-Processing — Grain Properties",
                 color="white", fontsize=16, fontweight='bold')

    outer = gridspec.GridSpec(n_sims, 1, figure=fig,
                              top=0.95, bottom=0.04, hspace=0.55)

    row_idx = 0
    for sim_name, snaps in sim_data.items():
        if not snaps: continue
        inner = outer[row_idx].subgridspec(1, 3, wspace=0.35)
        axes  = [fig.add_subplot(inner[0, j]) for j in range(3)]

        fig.text(0.005, (0.95 - row_idx / n_sims * 0.91 - 0.02),
                 sim_name.replace("_", " ").title(),
                 color=TITLE_C, fontsize=13, fontweight='bold', va='top',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1a1a',
                           edgecolor=TITLE_C, linewidth=1.2))

        if 'grain_growth' in sim_name:
            analyse_grain_growth(snaps, *axes)
        elif 'solidification' in sim_name:
            analyse_solidification(snaps, *axes)
        elif 'recrystallisation' in sim_name or 'recrystallization' in sim_name:
            analyse_recrystallisation(snaps, *axes)
        elif 'ostwald' in sim_name:
            analyse_ostwald(snaps, *axes)
        else:
            # Generic: radius distribution + boundary length
            analyse_grain_growth(snaps, *axes)

        row_idx += 1

    fig.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f"Saved → {out_path}")


# ── CLI ─────────────────────────────────────────────────────────────────────

def load_snaps(sim_dir: Path, max_snaps: int = 50):
    """Load all CSV snapshots from a simulation output directory."""
    files = sorted(sim_dir.glob("snap_*.csv"))
    if not files:
        return []
    files = files[:max_snaps]
    snaps = []
    for f in files:
        try:
            snaps.append(read_csv(f))
        except Exception as e:
            print(f"  Warning: skipping {f.name}: {e}", file=sys.stderr)
    return snaps


def main():
    parser = argparse.ArgumentParser(description="GPK post-processing")
    parser.add_argument("--sim",  default=None,
                        help="Simulation name (grain_growth|solidification|...)")
    parser.add_argument("--dir",  default=None,
                        help="Path to CSV snapshot directory")
    parser.add_argument("--out",  default="grain_analysis.png",
                        help="Output PNG path (default: grain_analysis.png)")
    parser.add_argument("--max-snaps", type=int, default=20,
                        help="Maximum snapshots per simulation to load")
    args = parser.parse_args()

    sim_data = {}

    if args.dir:
        d = Path(args.dir)
        snaps = load_snaps(d, args.max_snaps)
        sim_data[d.name] = snaps
        print(f"Loaded {len(snaps)} snapshots from {d}")
    elif args.sim:
        d = BUILD_ROOT / args.sim
        snaps = load_snaps(d, args.max_snaps)
        sim_data[args.sim] = snaps
        print(f"Loaded {len(snaps)} snapshots from {d}")
    else:
        # Auto-discover all simulation output directories
        if not BUILD_ROOT.exists():
            print(f"No output directory found at {BUILD_ROOT}")
            print("Run the C++ demos first.")
            sys.exit(1)
        for d in sorted(BUILD_ROOT.iterdir()):
            if d.is_dir():
                snaps = load_snaps(d, args.max_snaps)
                if snaps:
                    sim_data[d.name] = snaps
                    print(f"  {d.name}: {len(snaps)} snapshots")

    if not sim_data or all(len(v) == 0 for v in sim_data.values()):
        print("No CSV snapshots found.  Run the C++ demos first.")
        sys.exit(1)

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = Path(__file__).parent / out_path

    make_figure(sim_data, out_path)


if __name__ == "__main__":
    main()
