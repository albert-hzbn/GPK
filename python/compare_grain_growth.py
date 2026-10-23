"""
compare_grain_growth.py
=======================
Side-by-side visual comparison of grain growth produced by:
  1. A standard Allen-Cahn / multi-phase-field (PF) model
  2. The GPK sharp-interface model (python/gpk)

Both simulations start from the **identical** Voronoi initial condition
(same random seed, grid size, physical parameters).

Output:  grain_growth_comparison.png

Usage:
    cd d:/Work/GPK
    python python/compare_grain_growth.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from gpk import GPKGrid, ThermoModel

# ---------------------------------------------------------------------------
# Shared simulation parameters (identical for both models)
# ---------------------------------------------------------------------------
RNG_SEED    = 42
Ny, Nx      = 128, 128
N_GRAINS    = 10              # fewer grains so R >> eps* for PF
DX          = 1.0e-6         # 1 µm cell size (GPK grid)
# PF grid: same resolution as GPK (128×128 @ 1 µm/cell)
DX_PF       = 1.0e-6
PF_SCALE    = 1
Ny_pf       = Ny             # 128
Nx_pf       = Nx             # 128
EPS_PF_STAR = 3                        # interface width in PF cells (3µm)
M0          = 1.0e-8         # m³/(J·s)
SIGMA0      = 0.5            # J/m²
N_SNAPS     = 3              # initial, midpoint, final


# ---------------------------------------------------------------------------
# Shared initial microstructure: Voronoi tessellation
# ---------------------------------------------------------------------------
def make_voronoi(Ny, Nx, n_seeds, rng):
    seeds = rng.integers(0, [Ny, Nx], size=(n_seeds, 2))
    dist_min = np.full((Ny, Nx), np.inf)
    phi = np.zeros((Ny, Nx), dtype=np.int32)
    Y, X = np.mgrid[:Ny, :Nx]
    for k, (sy, sx) in enumerate(seeds):
        dy = np.minimum(np.abs(Y - sy), Ny - np.abs(Y - sy))   # periodic
        dx_ = np.minimum(np.abs(X - sx), Nx - np.abs(X - sx))
        dist = dy**2 + dx_**2
        mask = dist < dist_min
        dist_min = np.where(mask, dist, dist_min)
        phi = np.where(mask, k, phi)
    return phi


# ---------------------------------------------------------------------------
# Allen-Cahn multi-phase-field (PF) model  (non-dimensionalised)
# ---------------------------------------------------------------------------
class PFModel:
    """
    Allen-Cahn phase field for grain growth (explicit, coarse grid).

    Runs on a (Ny_pf × Nx_pf) grid at dx_pf = PF_SCALE * dx.
    Interface width = EPS_PF_STAR cells on the PF grid.
    Results are upsampled back to (Ny × Nx) for visualization.

    All dimensionless: x*=x/dx_pf, t*=t*M0*σ0/dx_pf².
    dI_i/dt* = ε*²·∇*²I_i − 2W*·I²(I−1)(2I−1)
    """

    def __init__(self, phi0_fine: np.ndarray, g0: np.ndarray,
                 dx: float = DX, M0: float = M0, sigma0: float = SIGMA0,
                 pf_scale: int = PF_SCALE, eps_star: int = EPS_PF_STAR):
        self.dx_fine    = dx
        self.pf_scale   = pf_scale
        self.dx_pf      = dx * pf_scale
        self.time_scale = self.dx_pf**2 / (M0 * sigma0)
        self.eps_star   = float(eps_star)
        self.W_star     = 2.0 / self.eps_star**2
        self.time       = 0.0

        # Downsample phi to coarse PF grid by majority vote (mode)
        Ny_f, Nx_f = phi0_fine.shape
        Ny_c, Nx_c = Ny_f // pf_scale, Nx_f // pf_scale
        self.Ny, self.Nx = Ny_c, Nx_c
        n_grains = int(phi0_fine.max()) + 1
        self.n_grains   = n_grains
        phi_c = phi0_fine.reshape(Ny_c, pf_scale, Nx_c, pf_scale)
        phi_c = phi_c.transpose(0,2,1,3).reshape(Ny_c, Nx_c, pf_scale*pf_scale)
        phi_coarse = np.apply_along_axis(
            lambda x: np.bincount(x, minlength=n_grains).argmax(), 2, phi_c
        ).astype(np.int32)
        self._phi_c = phi_coarse

        self.I = np.zeros((n_grains, Ny_c, Nx_c), dtype=np.float64)
        for i in range(n_grains):
            # Use sharp (step-function) init: 1 inside, 0 outside
            # Allen-Cahn will naturally form the diffuse interface profile
            self.I[i] = (phi_coarse == i).astype(np.float64)

        # Calibrate mobility correction from a small circle shrinkage test
        self.mu = self._calibrate_mu()

    def _calibrate_mu(self, cal_size: int = 48, r0: int = 14, n_steps: int = 300) -> float:
        """Measure effective mobility from a circle shrinkage test, return correction."""
        eps = self.eps_star; W = self.W_star
        # Small calibration grid
        ph = np.zeros((cal_size, cal_size), int)
        Y, X = np.mgrid[:cal_size, :cal_size]
        ph[np.sqrt((Y - cal_size//2)**2 + (X - cal_size//2)**2) < r0] = 1
        Ic = np.array([(ph==0).astype(float), (ph==1).astype(float)])
        A0 = np.sum(ph == 1)
        dt_c = 0.0016   # small stable dt*
        for _ in range(n_steps):
            fp = np.concatenate([Ic[...,-1:], Ic, Ic[...,0:1]], axis=-1)
            fp = np.concatenate([fp[...,-2:-1,:], fp, fp[...,1:2,:]], axis=-2)
            lap = fp[...,1:-1,2:]+fp[...,1:-1,:-2]+fp[...,2:,1:-1]+fp[...,:-2,1:-1]-4.0*Ic
            f_ = 2.0*W*Ic**2*(Ic-1)*(2*Ic-1)
            Ic = np.clip(Ic + dt_c * (eps**2 * lap - f_), 0.0, 1.0)
        A1 = np.sum(np.argmax(Ic, axis=0) == 1)
        dR2_dt_star = (A1/np.pi - A0/np.pi) / (n_steps * dt_c)
        return float((-2.0/3.0) / dR2_dt_star)  # mu s.t. effective rate = -(2/3)

    def step(self, dt: float):
        dt_star = dt / self.time_scale * self.mu   # mu corrects effective mobility
        I       = self.I
        e2      = self.eps_star**2
        W       = self.W_star

        # Periodic Laplacian via padding (vectorised over all grains)
        fp  = np.concatenate([I[...,-1:], I, I[...,0:1]], axis=-1)
        fp  = np.concatenate([fp[...,-2:-1,:], fp, fp[...,1:2,:]], axis=-2)
        lap = (fp[...,1:-1,2:] + fp[...,1:-1,:-2] +
               fp[...,2:,1:-1] + fp[...,:-2,1:-1] - 4.0*I)

        f_prime = 2.0 * W * I**2 * (I - 1.0) * (2.0 * I - 1.0)
        self.I  = np.clip(I + dt_star * (e2 * lap - f_prime), 0.0, 1.0)
        self._phi_c = np.argmax(self.I, axis=0).astype(np.int32)
        self.time += dt

    def grain_label_map(self) -> np.ndarray:
        """Upsampled grain label map at fine grid resolution."""
        return np.repeat(np.repeat(self._phi_c, self.pf_scale, axis=0),
                         self.pf_scale, axis=1)

    def run(self, t_end: float, dt: float):
        while self.time < t_end:
            self.step(min(dt, t_end - self.time))

    def mean_grain_radius(self) -> float:
        phi   = self._phi_c
        areas = np.bincount(phi.ravel(), minlength=self.n_grains).astype(float)
        active = areas[areas > 0]
        return float(np.mean(np.sqrt(active / np.pi))) * self.dx_pf



# ---------------------------------------------------------------------------
# GPK IsotropicThermo (grain growth: no driving force, constant mobility)
# ---------------------------------------------------------------------------
class IsotropicThermo(ThermoModel):
    def mobility(self, nx, ny, g_p, g_q, T):
        return self.M0


# ---------------------------------------------------------------------------
# Helper: mean grain radius from GPK phi map
# ---------------------------------------------------------------------------
def gpk_mean_radius(phi: np.ndarray, dx: float) -> float:
    areas = np.bincount(phi.ravel())
    areas = areas[areas > 0]
    if len(areas) == 0:
        return 0.0
    return float(np.mean(np.sqrt(areas / np.pi))) * dx


# ---------------------------------------------------------------------------
# Consistent colour mapping: grain i → colour index stable across snapshots
# ---------------------------------------------------------------------------
def remap_colors(phi: np.ndarray, n_colors: int = 20) -> np.ndarray:
    """Map grain labels to a small color index so visuals are consistent."""
    return phi % n_colors


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run():
    rng = np.random.default_rng(RNG_SEED)

    # ---- Shared initial microstructure ----
    phi0 = make_voronoi(Ny, Nx, N_GRAINS, rng)
    g0   = rng.uniform(0, np.pi / 2, (Ny, Nx))    # random orientations

    # ---- Time plan ----
    # Estimate t_end from von Neumann-Mullins: <R²> doubles from R0²
    areas0 = np.bincount(phi0.ravel())
    areas0 = areas0[areas0 > 0].astype(float)
    R0 = float(np.mean(np.sqrt(areas0 / np.pi))) * DX
    analytical_K = (2.0 / 3.0) * M0 * SIGMA0
    t_end   = 0.4 * R0**2 / analytical_K        # ~doubles mean area
    dt_gpk  = 0.4 * DX**2 / (M0 * SIGMA0)       # CFL for GPK

    # PF timestep: dt*_eff = dt_pf/time_scale * mu < 0.4/(eps*²*π²)
    dx_pf         = DX * PF_SCALE
    time_scale_pf = dx_pf**2 / (M0 * SIGMA0)
    # Estimated mu from calibration formula; actual mu computed in PFModel.__init__
    mu_approx     = {3: 0.063, 4: 0.0190, 5: 0.0161, 6: 0.0091}.get(EPS_PF_STAR, 0.02)
    dt_star_max   = 0.4 / (EPS_PF_STAR**2 * np.pi**2) / mu_approx * 0.8  # 20% safety margin
    dt_pf         = min(dt_gpk, dt_star_max * time_scale_pf)
    n_pf_total    = int(np.ceil(t_end / dt_pf))
    print(f"PF grid: {Ny_pf}×{Nx_pf} @ {dx_pf*1e6:.0f}µm, eps*={EPS_PF_STAR}, dt_pf={dt_pf:.3e}s ({n_pf_total} steps)")

    snap_times = [t_end * (s + 1) / N_SNAPS for s in range(N_SNAPS)]

    print(f"Grid: {Ny}×{Nx},  N_grains: {N_GRAINS},  dx: {DX*1e6:.1f} µm")
    print(f"R0: {R0*1e6:.2f} µm,  t_end: {t_end*1e3:.2f} ms")
    print(f"dt_gpk: {dt_gpk:.3e} s,  dt_pf: {dt_pf:.3e} s")
    print(f"Snap times: {[f'{t*1e3:.2f} ms' for t in snap_times]}")

    # ---- Initialise models ----
    thermo = IsotropicThermo(sigma0=SIGMA0, eps4=0.0, M0=M0)
    alpha0 = np.ones((Ny, Nx))
    c0     = np.zeros((Ny, Nx))
    T0     = np.full((Ny, Nx), 800.0)

    gpk_grid = GPKGrid(phi0.copy(), alpha0, c0, T0, g0, dx=DX,
                       thermo=thermo, D=0.0, n_phases=N_GRAINS + 1)
    pf_model = PFModel(phi0.copy(), g0, dx=DX, M0=M0, sigma0=SIGMA0,
                       pf_scale=PF_SCALE, eps_star=EPS_PF_STAR)
    # ---- Collect snapshots ----
    # Columns: initial | snap1 | snap2 | ... | final
    snaps_gpk = [phi0.copy()]
    snaps_pf  = [pf_model.grain_label_map().copy()]
    t_prev    = 0.0

    gpk_radii = [R0]
    pf_radii  = [R0]
    times_plot = [0.0]

    print("\nRunning simulations …")
    for i, t in enumerate(snap_times):
        print(f"  Advancing to t = {t*1e3:.2f} ms …", end="", flush=True)

        gpk_grid.run(t, dt=dt_gpk)
        pf_model.run(t, dt=dt_pf)

        snaps_gpk.append(gpk_grid.phi.copy())
        snaps_pf.append(pf_model.grain_label_map().copy())
        gpk_radii.append(gpk_mean_radius(gpk_grid.phi, DX))
        pf_radii.append(pf_model.mean_grain_radius())
        times_plot.append(t)
        print(f" done  (R_GPK={gpk_radii[-1]*1e6:.2f} µm, R_PF={pf_radii[-1]*1e6:.2f} µm)")

    # ---- Figure ----
    n_cols = len(snaps_gpk)   # initial + N_SNAPS
    fig, axes = plt.subplots(3, n_cols, figsize=(3.5 * n_cols, 10))

    col_labels = ["Initial"] + [f"t = {t*1e3:.1f} ms" for t in snap_times]
    cmap = plt.cm.get_cmap("tab20", 20)

    for col in range(n_cols):
        # Row 0: GPK
        ax = axes[0, col]
        ax.imshow(remap_colors(snaps_gpk[col]), cmap=cmap, vmin=0, vmax=19,
                  origin="lower", interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        if col == 0:
            ax.set_ylabel("GPK", fontsize=13, fontweight="bold")
        ax.set_title(col_labels[col], fontsize=10)

        # Row 1: PF
        ax = axes[1, col]
        ax.imshow(remap_colors(snaps_pf[col]), cmap=cmap, vmin=0, vmax=19,
                  origin="lower", interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        if col == 0:
            ax.set_ylabel("Phase Field", fontsize=13, fontweight="bold")

        # Row 2: Overlay difference (grain boundary positions)
        # Draw grain boundaries from each model and overlay
        ax = axes[2, col]
        def grain_boundaries(phi):
            """1 where a boundary exists between neighbours, 0 elsewhere."""
            bd = (
                (phi != np.roll(phi,  1, axis=0)) |
                (phi != np.roll(phi, -1, axis=0)) |
                (phi != np.roll(phi,  1, axis=1)) |
                (phi != np.roll(phi, -1, axis=1))
            ).astype(float)
            return bd

        bd_gpk = grain_boundaries(snaps_gpk[col])
        bd_pf  = grain_boundaries(snaps_pf[col])

        # Composite: GPK=red channel, PF=blue channel, overlap=magenta
        img = np.zeros((Ny, Nx, 3))
        img[:, :, 0] = bd_gpk           # red = GPK boundaries
        img[:, :, 2] = bd_pf            # blue = PF boundaries
        # Lighten background where no boundary
        bg = 1.0 - np.maximum(bd_gpk, bd_pf)
        img[:, :, 0] += 0.92 * bg
        img[:, :, 1] += 0.92 * bg
        img[:, :, 2] += 0.92 * bg
        img = np.clip(img, 0, 1)
        ax.imshow(img, origin="lower", interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        if col == 0:
            ax.set_ylabel("Boundaries\n(red=GPK, blue=PF)", fontsize=11)

    # ---- Kinetics inset panel ----
    fig2, ax_k = plt.subplots(figsize=(6, 4))
    t_ms = [t * 1e3 for t in times_plot]
    ax_k.plot(t_ms, [r * 1e6 for r in gpk_radii], "o-", color="tab:red",   label="GPK")
    ax_k.plot(t_ms, [r * 1e6 for r in pf_radii],  "s-", color="tab:blue",  label="Phase Field")
    t_arr = np.array(times_plot[1:])
    R_theory = np.sqrt(R0**2 + analytical_K * t_arr)
    ax_k.plot(t_ms[1:], R_theory * 1e6, "k--", label="Analytical (vN-M)")
    ax_k.set_xlabel("Time [ms]")
    ax_k.set_ylabel("Mean grain radius [µm]")
    ax_k.set_title("Grain growth kinetics comparison")
    ax_k.legend()
    ax_k.grid(True, alpha=0.3)
    plt.tight_layout()
    fig2.savefig("grain_growth_kinetics.png", dpi=150)
    print("Saved: grain_growth_kinetics.png")

    plt.figure(fig.number)
    plt.suptitle(
        f"Grain Growth: GPK vs Phase Field\n"
        f"({Ny}×{Nx} grid, {N_GRAINS} grains, dx={DX*1e6:.0f} µm, "
        f"M₀={M0:.0e} m³/(J·s), σ₀={SIGMA0} J/m²)",
        fontsize=12, y=1.01
    )
    plt.tight_layout()
    plt.savefig("grain_growth_comparison.png", dpi=150, bbox_inches="tight")
    print("Saved: grain_growth_comparison.png")


if __name__ == "__main__":
    run()
