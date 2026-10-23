"""
plot_microstructures.py
=======================
Runs all four GPK C++ demos, captures their console output, and produces a
single publication-quality figure with microstructure snapshots and
quantitative metric plots for each simulation type.

Usage:
    python plot_microstructures.py
"""

import subprocess
import os
import re
import io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.image import imread
from pathlib import Path

BUILD = Path(__file__).parent.parent / "build"
OUT   = Path(__file__).parent / "microstructures_result.png"

# ── helpers ────────────────────────────────────────────────────────────────

def run_demo(name: str) -> str:
    """Run a demo executable and return its stdout."""
    exe = BUILD / name
    print(f"  Running {name} …", flush=True)
    result = subprocess.run(
        [str(exe)], capture_output=True, text=True, cwd=BUILD
    )
    if result.returncode != 0:
        raise RuntimeError(f"{name} failed:\n{result.stderr}")
    return result.stdout


def parse_table(stdout: str, header_keyword: str) -> np.ndarray:
    """
    Extract the first numeric table that appears after a line containing
    *header_keyword*.  Returns an (N, M) float array (non-numeric rows skipped).
    """
    lines = stdout.splitlines()
    in_table = False
    rows = []
    for line in lines:
        if header_keyword in line:
            in_table = True
            continue
        if not in_table:
            continue
        # Try to parse the whole line as floats
        tokens = line.split()
        try:
            row = [float(t) for t in tokens]
            if len(row) >= 2:
                rows.append(row)
        except ValueError:
            if rows:            # first non-numeric line after data → stop
                break
    return np.array(rows, dtype=float) if rows else np.empty((0, 0))


def load_ppm(path) -> np.ndarray:
    """Load a PPM file with matplotlib (requires Pillow)."""
    return imread(str(path))


def pick_snaps(pattern: str, n: int = 3) -> list:
    """Return up to *n* evenly-spaced PPM paths matching *pattern* glob."""
    files = sorted(Path(BUILD).glob(pattern))
    if not files:
        return []
    idx = np.linspace(0, len(files) - 1, min(n, len(files)), dtype=int)
    return [files[i] for i in idx]

# ── run all demos ──────────────────────────────────────────────────────────

print("Running C++ demos …")
gg_out   = run_demo("grain_growth")
sol_out  = run_demo("solidification")
rex_out  = run_demo("recrystallisation")
osw_out  = run_demo("ostwald_ripening")
print("All demos finished.\n")

# ── parse quantitative data ────────────────────────────────────────────────

# Grain growth: columns  Time[ms]  R_GPK[µm]  R_th[µm]  Error[%]  Grains
gg_data  = parse_table(gg_out,  "Time [ms]")
# Solidification: columns  Time[ms]  Solid_frac  Solute_max  Steps
sol_data = parse_table(sol_out, "Time[ms]")
# Recrystallisation: columns  Time[s]  Rex_frac  ln(-ln(1-X))  N_grains
rex_data = parse_table(rex_out, "Time[s]")
# Ostwald: columns  Time[ms]  N_part  R_mean[µm]  R_min[µm]  R_max[µm]
osw_data = parse_table(osw_out, "Time[ms]")

# ── build figure ───────────────────────────────────────────────────────────

fig = plt.figure(figsize=(18, 20))
fig.patch.set_facecolor("#0d0d0d")

TITLE_KW  = dict(color="white", fontsize=13, fontweight="bold", pad=8)
LABEL_KW  = dict(color="#cccccc", fontsize=10)
TICK_KW   = dict(colors="#aaaaaa", labelsize=9)
SPINE_C   = "#444444"
GRID_KW   = dict(color="#333333", linewidth=0.5)

def style_ax(ax):
    ax.set_facecolor("#111111")
    for sp in ax.spines.values():
        sp.set_edgecolor(SPINE_C)
    ax.tick_params(axis="both", **TICK_KW)
    ax.grid(True, **GRID_KW)

# ── Section headers ────────────────────────────────────────────────────────
section_y = [0.97, 0.73, 0.49, 0.25]
section_titles = [
    "① Grain Growth  —  Curvature-Driven Coarsening",
    "② Dendritic Solidification  —  Binary Alloy (Al-Cu-like)",
    "③ Recrystallisation  —  Avrami Kinetics",
    "④ Ostwald Ripening  —  Size-Selective Dissolution",
]
for y, title in zip(section_y, section_titles):
    fig.text(0.01, y, title, color="#ffcc44", fontsize=14,
             fontweight="bold", va="top",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1a1a",
                       edgecolor="#ffcc44", linewidth=1.2))

outer = gridspec.GridSpec(
    4, 1, figure=fig,
    top=0.96, bottom=0.03,
    hspace=0.38,
    height_ratios=[1, 1, 1, 1],
)

# ══════════════════════════════════════════════════════════════════════════
# ROW 1 — Grain Growth
# ══════════════════════════════════════════════════════════════════════════
gg_gs = outer[0].subgridspec(1, 2, wspace=0.08, width_ratios=[1, 1.6])

# Microstructure image
ax_gg_img = fig.add_subplot(gg_gs[0])
gg_img_path = BUILD / "grain_growth_final.ppm"
if gg_img_path.exists():
    ax_gg_img.imshow(load_ppm(gg_img_path))
ax_gg_img.set_title("Final grain map", **TITLE_KW)
ax_gg_img.axis("off")

# R vs t plot
ax_gg = fig.add_subplot(gg_gs[1])
style_ax(ax_gg)
if gg_data.size:
    t   = gg_data[:, 0]      # ms
    r   = gg_data[:, 1]      # µm  GPK
    r_t = gg_data[:, 2]      # µm  theory
    ax_gg.plot(t, r,   "o-", color="#4fc3f7", lw=2, ms=6,
               label="GPK simulation")
    ax_gg.plot(t, r_t, "--", color="#ff8a65", lw=2,
               label=r"Theory  $\langle R\rangle = \sqrt{R_0^2 + K t}$")
    ax_gg.set_xlabel("Time  [ms]", **LABEL_KW)
    ax_gg.set_ylabel(r"Mean grain radius  $\langle R \rangle$  [µm]", **LABEL_KW)
    ax_gg.set_title("Von Neumann–Mullins parabolic growth law", **TITLE_KW)
    leg = ax_gg.legend(fontsize=9, facecolor="#1a1a1a", edgecolor=SPINE_C,
                       labelcolor="white")

# ══════════════════════════════════════════════════════════════════════════
# ROW 2 — Solidification
# ══════════════════════════════════════════════════════════════════════════
sol_gs = outer[1].subgridspec(1, 5, wspace=0.05)

# 4 phase snapshots  (evenly spaced)
phase_snaps = pick_snaps("solid_snap_*_phase.ppm", 4)
for i, path in enumerate(phase_snaps):
    ax = fig.add_subplot(sol_gs[i])
    ax.imshow(load_ppm(path))
    snap_idx = int(re.search(r"snap_(\d+)", path.name).group(1))
    if sol_data.size and snap_idx < len(sol_data):
        t_ms = sol_data[snap_idx, 0]
        sf   = sol_data[snap_idx, 1]
        ax.set_title(f"t={t_ms:.2f} ms\nf_s={sf:.2f}", **TITLE_KW)
    ax.axis("off")

# Solid fraction vs time
ax_sol = fig.add_subplot(sol_gs[4])
style_ax(ax_sol)
if sol_data.size:
    ax_sol.plot(sol_data[:, 0], sol_data[:, 1] * 100,
                "s-", color="#69f0ae", lw=2, ms=6)
    ax_sol.set_xlabel("Time  [ms]", **LABEL_KW)
    ax_sol.set_ylabel("Solid fraction  [%]", **LABEL_KW)
    ax_sol.set_title("Solid fraction vs time", **TITLE_KW)

# ══════════════════════════════════════════════════════════════════════════
# ROW 3 — Recrystallisation
# ══════════════════════════════════════════════════════════════════════════
rex_gs = outer[2].subgridspec(1, 4, wspace=0.08, width_ratios=[1, 1, 1, 1.5])

rex_snaps = pick_snaps("rex_snap_*.ppm", 3)
snap_indices = [int(re.search(r"snap_(\d+)", p.name).group(1))
                for p in rex_snaps]

for col, (path, sidx) in enumerate(zip(rex_snaps, snap_indices)):
    ax = fig.add_subplot(rex_gs[col])
    ax.imshow(load_ppm(path))
    if rex_data.size and sidx < len(rex_data):
        t_s = rex_data[sidx, 0]
        X   = rex_data[sidx, 1]
        ax.set_title(f"t={t_s:.1f} s\nX={X:.2f}", **TITLE_KW)
    ax.axis("off")

# Avrami plot
ax_rex = fig.add_subplot(rex_gs[3])
style_ax(ax_rex)
if rex_data.size:
    mask = (rex_data[:, 1] > 0.01) & (rex_data[:, 1] < 0.999)
    t_r  = rex_data[mask, 0]
    avrami = rex_data[mask, 2]
    if len(t_r) > 1:
        ax_rex.plot(np.log(t_r), avrami, "D-", color="#ce93d8", lw=2, ms=6,
                    label="GPK")
        # Fit slope
        coeffs = np.polyfit(np.log(t_r), avrami, 1)
        t_fit  = np.linspace(np.log(t_r[0]), np.log(t_r[-1]), 50)
        ax_rex.plot(t_fit, np.polyval(coeffs, t_fit), "--",
                    color="#ff8a65", lw=1.5,
                    label=f"slope = {coeffs[0]:.2f}")
        ax_rex.legend(fontsize=9, facecolor="#1a1a1a", edgecolor=SPINE_C,
                      labelcolor="white")
    ax_rex.set_xlabel(r"$\ln(t)$", **LABEL_KW)
    ax_rex.set_ylabel(r"$\ln(-\ln(1-X))$", **LABEL_KW)
    ax_rex.set_title("Avrami (JMAK) plot", **TITLE_KW)

# ══════════════════════════════════════════════════════════════════════════
# ROW 4 — Ostwald Ripening
# ══════════════════════════════════════════════════════════════════════════
osw_gs = outer[3].subgridspec(1, 4, wspace=0.08, width_ratios=[1, 1, 1, 1.5])

osw_phase_snaps = pick_snaps("ostwald_snap_*_phase.ppm", 3)
for col, path in enumerate(osw_phase_snaps):
    ax = fig.add_subplot(osw_gs[col])
    ax.imshow(load_ppm(path))
    sidx = int(re.search(r"snap_(\d+)", path.name).group(1))
    if osw_data.size and sidx < len(osw_data):
        t_ms = osw_data[sidx, 0]
        n    = int(osw_data[sidx, 1])
        rm   = osw_data[sidx, 2]
        ax.set_title(f"t={t_ms:.0f} ms\nN={n}  ⟨R⟩={rm:.1f}µm", **TITLE_KW)
    ax.axis("off")

# Particle count + mean radius vs time
ax_osw = fig.add_subplot(osw_gs[3])
style_ax(ax_osw)
if osw_data.size:
    t_ms   = osw_data[:, 0]
    n_part = osw_data[:, 1]
    r_mean = osw_data[:, 2]
    r_max  = osw_data[:, 4]

    color1, color2, color3 = "#4fc3f7", "#69f0ae", "#ff8a65"
    ln1, = ax_osw.plot(t_ms, r_mean, "o-", color=color1, lw=2, ms=5,
                       label=r"$\langle R \rangle$")
    ln2, = ax_osw.plot(t_ms, r_max,  "s--", color=color3, lw=1.5, ms=5,
                       label=r"$R_{\max}$")
    ax_osw.set_xlabel("Time  [ms]", **LABEL_KW)
    ax_osw.set_ylabel("Radius  [µm]", **LABEL_KW)
    ax_osw.set_title("Coarsening: particle radii vs time", **TITLE_KW)

    ax2 = ax_osw.twinx()
    ax2.set_facecolor("#111111")
    ax2.tick_params(axis="y", **TICK_KW)
    ln3, = ax2.plot(t_ms, n_part, "^:", color=color2, lw=1.5, ms=5,
                    label="N particles")
    ax2.set_ylabel("Particle count", color="#aaaaaa", fontsize=10)
    ax2.spines["right"].set_edgecolor(SPINE_C)
    ax2.spines["left"].set_edgecolor(SPINE_C)

    lines  = [ln1, ln2, ln3]
    labels = [l.get_label() for l in lines]
    ax_osw.legend(lines, labels, fontsize=9, facecolor="#1a1a1a",
                  edgecolor=SPINE_C, labelcolor="white")

# ── main title & save ──────────────────────────────────────────────────────

fig.suptitle(
    "GPK Microstructure Evolution  —  C++ Phase-Field Simulations",
    color="white", fontsize=16, fontweight="bold", y=0.995,
)

fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved → {OUT}")
