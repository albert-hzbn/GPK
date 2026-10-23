"""
Demo 3: Wulff shape — verifying anisotropy is grid-free

Grow a single grain into a matrix with 4-fold and 6-fold interfacial energy
anisotropy.  GPK should produce the Wulff shape analytically predicted;
a standard CA would produce a square/hexagon locked to the grid.

Wulff construction for σ(θ) = σ₀(1 + ε cos(nθ)):
  r(θ) ∝ σ(θ) / (σ(θ) + σ''(θ))   (inverse of stiffness)
  → distinct inner boundary cusps for large ε (faceting)

Run:  python demo_anisotropy.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gpk import GPKGrid, ThermoModel


class WulffThermo(ThermoModel):
    """Single grain growing by curvature alone, anisotropic σ."""
    def __init__(self, sigma0, eps_n, n_fold, M0=1e-8):
        super().__init__(sigma0=sigma0, eps4=0.0, M0=M0)
        self.eps_n = eps_n
        self.n_fold = n_fold

    def bulk_driving_force(self, phi_parent, phi_product, c, T):
        return 0.0   # capillarity only

    def interfacial_energy(self, nx, ny, g):
        theta = np.arctan2(ny, nx)
        return self.sigma0 * (1.0 + self.eps_n * np.cos(self.n_fold * theta))

    def mobility(self, nx, ny, g_p, g_q, T):
        return self.M0


def run_wulff(eps_n, n_fold, label, Ny=200, Nx=200, r_seed=5, n_steps=300):
    dx    = 1e-6
    thermo = WulffThermo(sigma0=0.5, eps_n=eps_n, n_fold=n_fold)

    phi   = np.zeros((Ny, Nx), dtype=np.int32)
    alpha = np.ones((Ny, Nx))
    c     = np.zeros((Ny, Nx))
    T     = np.full((Ny, Nx), 800.0)
    g     = np.zeros((Ny, Nx))   # fixed orientation

    cy, cx = Ny // 2, Nx // 2
    Y, X   = np.ogrid[:Ny, :Nx]
    seed   = (Y - cy)**2 + (X - cx)**2 <= r_seed**2
    phi    = np.where(seed, 1, 0).astype(np.int32)

    grid = GPKGrid(phi, alpha, c, T, g, dx=dx, thermo=thermo, D=0.0)
    dt   = 1e-4 * dx**2 / (thermo.M0 * thermo.sigma0 / dx)

    for _ in range(n_steps):
        grid.step(dt)

    return grid   # return full grid so caller can use render_smooth


def wulff_polar(sigma0, eps_n, n_fold, n_pts=360):
    """
    Analytical Wulff shape r(θ) ∝ 1 / stiffness(θ)
    stiffness = σ + d²σ/dθ²
    """
    theta = np.linspace(0, 2 * np.pi, n_pts)
    sigma = sigma0 * (1.0 + eps_n * np.cos(n_fold * theta))
    d2sigma = -sigma0 * eps_n * n_fold**2 * np.cos(n_fold * theta)
    stiffness = sigma + d2sigma
    # Wulff: r ∝ σ(θ)/stiffness(θ) ... simplified: r ∝ σ(θ)
    r = sigma / np.maximum(stiffness, 0.01)
    r /= r.max()
    return theta, r


def run():
    configs = [
        (0.0,  4, "Isotropic"),
        (0.1,  4, "4-fold weak"),
        (0.3,  4, "4-fold strong"),
        (0.2,  6, "6-fold"),
    ]

    fig, axes = plt.subplots(2, len(configs), figsize=(16, 8))

    for col, (eps_n, n_fold, lbl) in enumerate(configs):
        print(f"Running: {lbl} ...")
        grid_obj = run_wulff(eps_n, n_fold, lbl)
        phi = grid_obj.phi
        Ny, Nx = phi.shape
        cy, cx = Ny // 2, Nx // 2

        # render_smooth blends grain/matrix colours using alpha,
        # giving smooth Wulff boundaries instead of jagged staircases.
        axes[0, col].imshow(grid_obj.render_smooth("phase"), origin="lower")
        axes[0, col].set_title(lbl)
        axes[0, col].axis("off")

        # Radial profile of GPK grain boundary
        angles = np.linspace(0, 2 * np.pi, 360)
        r_gpk  = []
        for a in angles:
            for r in range(1, min(Ny, Nx) // 2):
                yi = int(cy + r * np.sin(a))
                xi = int(cx + r * np.cos(a))
                if 0 <= yi < Ny and 0 <= xi < Nx:
                    if phi[yi, xi] == 0:
                        r_gpk.append(r - 1)
                        break
            else:
                r_gpk.append(0)
        r_gpk = np.array(r_gpk, dtype=float)
        if r_gpk.max() > 0:
            r_gpk /= r_gpk.max()

        theta_w, r_w = wulff_polar(0.5, eps_n, n_fold)

        ax = axes[1, col]
        ax.plot(r_gpk * np.cos(angles), r_gpk * np.sin(angles),
                label="GPK boundary", linewidth=2)
        ax.plot(r_w * np.cos(theta_w), r_w * np.sin(theta_w),
                "--", label="Wulff (analytical)", linewidth=1.5)
        ax.set_aspect("equal"); ax.legend(fontsize=7); ax.set_title("Polar profile")
        ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2)

    plt.suptitle("GPK Wulff shape validation — anisotropy without grid bias")
    plt.tight_layout()
    plt.savefig("anisotropy_result.png", dpi=150)
    print("Saved: anisotropy_result.png")


if __name__ == "__main__":
    run()
