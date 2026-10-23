"""
Demo 1: Isotropic grain growth  (GPK vs analytical)

von Neumann–Mullins law:  ⟨R²⟩ = ⟨R₀²⟩ + 2 M σ t / 3
Expected growth exponent n = 2  (mean radius ∝ t^(1/2)).

Run:  python demo_grain_growth.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from gpk import GPKGrid, ThermoModel


def make_voronoi_microstructure(Ny, Nx, n_seeds, rng):
    """Create a Voronoi tessellation with n_seeds grains."""
    seeds = rng.integers(0, [Ny, Nx], size=(n_seeds, 2))
    Y, X = np.ogrid[:Ny, :Nx]
    phi = np.zeros((Ny, Nx), dtype=np.int32)
    for k, (sy, sx) in enumerate(seeds):
        if k == 0:
            dist_min = (Y - sy)**2 + (X - sx)**2
            phi[:] = 0
        else:
            dist = (Y - sy)**2 + (X - sx)**2
            phi = np.where(dist < dist_min, k, phi)
            dist_min = np.minimum(dist_min, dist)
    return phi


def count_grain_radius(phi):
    """Mean grain radius from active grains only (non-zero area)."""
    areas = np.bincount(phi.ravel())
    areas = areas[areas > 0]          # skip dissolved grains (area = 0)
    if len(areas) == 0:
        return 0.0
    return np.mean(np.sqrt(areas / np.pi))


def run():
    rng = np.random.default_rng(42)
    Ny, Nx = 128, 128
    n_grains = 40
    dx = 1e-6       # 1 µm

    # Isotropic thermo: ΔG_chem = 0, M = M0 (misorientation-independent), σ = σ0
    class IsotropicThermo(ThermoModel):
        def mobility(self, nx, ny, g_p, g_q, T):
            return self.M0   # constant, ignore misorientation

    thermo = IsotropicThermo(sigma0=0.5, eps4=0.0, M0=1e-8)

    phi0 = make_voronoi_microstructure(Ny, Nx, n_grains, rng)
    alpha0 = np.ones((Ny, Nx))
    c0     = np.zeros((Ny, Nx))
    T0     = np.full((Ny, Nx), 800.0)
    g0     = rng.uniform(0, np.pi/2, (Ny, Nx))   # random crystal orientations

    grid = GPKGrid(phi0, alpha0, c0, T0, g0, dx=dx, thermo=thermo, D=0.0)

    # Analytical prediction:  ⟨R²⟩ = R₀² + K t   (K = parabolic growth constant)
    R0 = count_grain_radius(grid.phi) * dx
    # Dimensionally: dR²/dt = 2Mσ/3  (von Neumann-Mullins 2D approximation)
    analytical_K = (2.0 / 3.0) * thermo.M0 * thermo.sigma0   # [m²/s]

    times, R_gpk, R_theory = [0.0], [R0], [R0]

    # Stable dt: based on interface CFL  Δt < 0.4 Δx² / (M₀ σ₀)
    dt = 0.4 * dx**2 / (thermo.M0 * thermo.sigma0)
    n_snapshots = 10
    # Run until mean radius grows by ~50 % (R² increases by factor 2.25)
    t_end = 2.25 * R0**2 / analytical_K

    print(f"{'Time [s]':>14}  {'R_GPK [µm]':>12}  {'R_theory [µm]':>14}  {'error %':>8}")
    for snap in range(n_snapshots):
        grid.run(t_end * (snap + 1) / n_snapshots, dt=dt)
        R_sim = count_grain_radius(grid.phi) * dx
        t = grid.time
        R_th = np.sqrt(R0**2 + analytical_K * t)
        times.append(t)
        R_gpk.append(R_sim)
        R_theory.append(R_th)
        err = abs(R_sim - R_th) / R_th * 100 if R_th > 0 else 0
        print(f"{t:14.3e}  {R_sim*1e6:12.4f}  {R_th*1e6:14.4f}  {err:8.2f}")

    # ---------- Plot ----------
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    ax = axes[0]
    # render_smooth blends grain colours using alpha at interface cells,
    # eliminating the staircase-pixelated grain boundary.
    ax.imshow(grid.render_smooth("grains"), origin="lower")
    ax.set_title("Final microstructure (GPK, α-blended boundaries)")
    ax.set_xlabel("x"); ax.set_ylabel("y")

    ax = axes[1]
    ax.plot([t * 1e3 for t in times], [r * 1e6 for r in R_gpk], "o-", label="GPK")
    ax.plot([t * 1e3 for t in times], [r * 1e6 for r in R_theory], "--", label="Analytical")
    ax.set_xlabel("Time [ms]"); ax.set_ylabel("Mean grain radius [µm]")
    ax.set_title("Grain growth kinetics"); ax.legend()

    ax = axes[2]
    t_arr = np.array(times[1:])
    R_arr = np.array(R_gpk[1:])
    if len(t_arr) > 2:
        log_t = np.log(t_arr)
        log_R = np.log(R_arr)
        n_fit = np.polyfit(log_t, log_R, 1)[0]
        ax.loglog(t_arr * 1e3, R_arr * 1e6, "o-", label=f"GPK  (n={n_fit:.3f})")
        ax.loglog(t_arr * 1e3, [r * 1e6 for r in R_theory[1:]], "--", label="Analytical (n=0.5)")
    ax.set_xlabel("Time [ms]"); ax.set_ylabel("Mean radius [µm]")
    ax.set_title("Log-log: growth exponent"); ax.legend()

    plt.tight_layout()
    plt.savefig("grain_growth_result.png", dpi=150)
    print("\nSaved: grain_growth_result.png")


if __name__ == "__main__":
    run()
