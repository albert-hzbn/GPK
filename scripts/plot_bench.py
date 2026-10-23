#!/usr/bin/env python3
"""Plot the C++ GPK-vs-PF benchmark (reads CSVs written by bench_gpk_vs_pf.exe)."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(__file__), "..")
A = np.genfromtxt(os.path.join(ROOT, "build/bench_partA.csv"), delimiter=",", names=True)
C = np.genfromtxt(os.path.join(ROOT, "build/bench_partC_gpk.csv"), delimiter=",", names=True)

BLUE, ORANGE, GREEN, GREY, RED = "#1f4e9c", "#e07a1f", "#2a9d3a", "#888888", "#bb1111"
fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))

# (a) Matched-grid throughput vs grid size
ax[0].plot(A["N"], A["ca_rate"], "^-", color=RED, label="Cellular automaton")
ax[0].plot(A["N"], A["gpk_rate"], "o-", color=BLUE, label="GPK (C++ core)")
ax[0].plot(A["N"], A["pf_rate"], "s-", color=ORANGE, label="Phase field (Fan-Chen)")
ax[0].set_xlabel("Grid size N  (N×N)")
ax[0].set_ylabel("Throughput  [cell-steps / s]")
ax[0].set_title("(a) Matched-grid throughput")
ax[0].set_yscale("log")
ax[0].grid(True, which="both", alpha=0.3)
ax[0].legend()

# (b) Equal-physics speedup vs rho (measured per-cell constants + rho^4 scaling)
c_gpk = 1.0 / A["gpk_rate"][-1]
c_pf = 1.0 / A["pf_rate"][-1]
rho = np.array([2, 5, 10, 20, 50, 100], float)
cellsteps = rho**4
wall = cellsteps * (c_pf / c_gpk)
ax[1].plot(rho, wall, "o-", color=GREEN, label="wall-time PF/GPK (measured const)")
ax[1].plot(rho, cellsteps, "s--", color=GREY, label=r"cell-steps $\rho^{4}$ (ideal)")
ax[1].axhline(1.0, color="k", lw=0.8, ls=":")
ax[1].set_xlabel(r"Grid coarsening  $\rho=\Delta x_{\mathrm{GPK}}/\Delta x_{\mathrm{PF}}$")
ax[1].set_ylabel("PF / GPK cost ratio  (>1 ⇒ GPK faster)")
ax[1].set_title("(b) Equal-physics cost, fixed domain")
ax[1].set_xscale("log"); ax[1].set_yscale("log")
ax[1].grid(True, which="both", alpha=0.3)
ax[1].legend()

# (c) GPK parabolic kinetics vs von Neumann-Mullins theory
t_ms = C["t"] * 1e3
R2_gpk = (C["R_gpk"] * 1e6)**2
R2_th = (C["R_theory"] * 1e6)**2
ax[2].plot(t_ms, R2_gpk, "o", color=BLUE, label="GPK ⟨R⟩²")
ax[2].plot(t_ms, R2_th, "-", color="k", label=r"theory $R_0^2+Kt$,  $K=2M\sigma/3$")
ax[2].set_xlabel("time  [ms]")
ax[2].set_ylabel(r"⟨R⟩²  [µm²]")
ax[2].set_title("(c) GPK parabolic kinetics (von Neumann–Mullins)")
ax[2].grid(True, alpha=0.3)
ax[2].legend()

fig.tight_layout()
out = os.path.join(ROOT, "figures", "bench_gpk_vs_pf.png")
fig.savefig(out, dpi=160, bbox_inches="tight")
print("saved", out)

# ---------------------------------------------------------------------------
# Second figure: three-way equal-physics positioning (CA / GPK / PF)
# Normalised to CA = 1, fixed physical domain, rho = 10 (PF resolves interface).
# Cost ladder uses MEASURED per-cell constants + analytic rho^(d+2) for PF.
# ---------------------------------------------------------------------------
c_ca  = 1.0 / A["ca_rate"][-1]
c_gpk = 1.0 / A["gpk_rate"][-1]
c_pf  = 1.0 / A["pf_rate"][-1]
rho = 10.0
ca_cost  = 1.0
gpk_cost = c_gpk / c_ca                  # same coarse grid as CA
pf_cost  = (rho**4) * (c_pf / c_ca)      # PF needs rho^2 cells x rho^2 steps

fig2, bx = plt.subplots(figsize=(9.5, 3.6))
labels = ["Cellular\nAutomaton", "GPK", "Phase\nField"]
vals   = [ca_cost, gpk_cost, pf_cost]
cols   = [RED, BLUE, ORANGE]
bars = bx.barh(labels, vals, color=cols, height=0.6)
bx.set_xscale("log")
bx.set_xlabel(f"Equal-physics cost, fixed domain  (normalised to CA = 1,  ρ = {rho:.0f})")
bx.set_title("Where GPK sits:  CA-class cost,  PF-class physics")
ann = [f"1×\ngrid-locked κ, no thermo",
       f"{gpk_cost:.0f}×\nreal κ + CALPHAD + anisotropy",
       f"{pf_cost:,.0f}×\nsame physics, must resolve interface"]
for b, a in zip(bars, ann):
    bx.text(b.get_width() * 1.25, b.get_y() + b.get_height() / 2, a,
            va="center", ha="left", fontsize=9)
bx.set_xlim(0.5, pf_cost * 30)
bx.grid(True, axis="x", which="both", alpha=0.3)
fig2.tight_layout()
out2 = os.path.join(ROOT, "figures", "bench_positioning.png")
fig2.savefig(out2, dpi=160, bbox_inches="tight")
print("saved", out2)
print(f"ladder: CA=1x, GPK={gpk_cost:.1f}x, PF={pf_cost:,.0f}x  (rho={rho:.0f})")
