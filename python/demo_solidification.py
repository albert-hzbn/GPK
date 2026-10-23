"""
Demo 2: Dendritic solidification of a binary alloy

Solid seed growing into undercooled liquid.
GPK uses:
  - SolidificationThermo for ΔG_chem from undercooling
  - 4-fold anisotropy in σ (eps4=0.04) for dendritic morphology
  - Stefan diffusion (solute redistribution at interface)

Run:  python demo_solidification.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gpk import GPKGrid, SolidificationThermo


def run():
    rng = np.random.default_rng(7)
    Ny, Nx = 300, 300
    dx = 0.5e-6   # 0.5 µm

    # ---------- Thermodynamics ----------
    thermo = SolidificationThermo(
        sigma0 = 0.2,       # J/m²
        eps4   = 0.04,      # 4-fold anisotropy → dendritic
        M0     = 5e-8,      # m³/(J·s)
        m_liq  = -500.0,    # K / (mol frac)
        k_part = 0.3,
        T_melt = 933.0,
        L      = 1e8,       # J/m³
    )

    # ---------- Initial conditions ----------
    # Liquid everywhere, small solid seed at centre
    phi   = np.zeros((Ny, Nx), dtype=np.int32)   # 0 = liquid
    alpha = np.ones((Ny, Nx))
    c0    = 0.02          # 2 mol% solute
    c     = np.full((Ny, Nx), c0)
    T_liq = thermo.T_melt + thermo.m_liq * c0
    undercooling = 5.0    # K
    T     = np.full((Ny, Nx), T_liq - undercooling)
    g     = np.full((Ny, Nx), 0.0)   # single crystal orientation (aligned with grid)

    # Solid seed: 5×5 cells at centre
    cy, cx = Ny // 2, Nx // 2
    r_seed = 3
    phi[cy-r_seed:cy+r_seed, cx-r_seed:cx+r_seed] = 1   # solid
    c[cy-r_seed:cy+r_seed, cx-r_seed:cx+r_seed] = c0 * thermo.k_part

    grid = GPKGrid(phi, alpha, c, T, g,
                   dx=dx, thermo=thermo,
                   D=3e-9,   # solute diffusivity in liquid [m²/s]
                   n_phases=2)

    # ---------- Run ----------
    snapshots = []
    t_end     = 2e-3    # 2 ms
    n_snaps   = 6
    snap_times = np.linspace(t_end / n_snaps, t_end, n_snaps)

    print("Running dendritic solidification...")
    for t_snap in snap_times:
        grid.run(t_snap, dt=None)   # auto stable dt via max_dt()
        snapshots.append((grid.time, grid.phi.copy(), grid.c.copy()))
        solid_frac = np.mean(grid.phi)
        print(f"  t={grid.time*1e3:.2f} ms   solid fraction={solid_frac:.3f}")

    # ---------- Plot ----------
    fig, axes = plt.subplots(2, n_snaps, figsize=(18, 6))
    for k, (t, phi_snap, c_snap) in enumerate(snapshots):
        ax = axes[0, k]
        # render_smooth blends solid/liquid colours using alpha at the
        # interface, giving smooth dendritic boundaries instead of staircases.
        ax.imshow(grid.render_smooth("phase"), origin="lower")
        ax.set_title(f"t={t*1e3:.2f} ms")
        ax.axis("off")

        ax = axes[1, k]
        im = ax.imshow(c_snap, cmap="RdYlBu_r", origin="lower",
                       vmin=c0 * thermo.k_part, vmax=c0 * 1.5)
        ax.axis("off")
        if k == 0:
            ax.set_ylabel("Composition")

    axes[0, 0].set_ylabel("Phase (solid=blue, α-blended)")
    plt.suptitle("GPK: Dendritic solidification (4-fold anisotropy, binary alloy)")
    plt.tight_layout()
    plt.savefig("solidification_result.png", dpi=150)
    print("Saved: solidification_result.png")


if __name__ == "__main__":
    run()
