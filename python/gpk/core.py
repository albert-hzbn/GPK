"""
Gaussian-Projected Kinetics (GPK) — core engine.

State per cell (2-D implementation):
    phi   : int   — phase label (0…N_phases-1)
    alpha : float — volume fraction of phase phi in this cell  [0,1]
    c     : float — composition (extend to array for multicomponent)
    T     : float — temperature
    g     : float — in-plane crystallographic orientation [radians]  (simplified SO(2))

Sign convention (sharp-interface Allen-Cahn limit):
    n̂  = outward normal of parent phase  (computed as -∇I/|∇I|, I = same-phase indicator)
    κ  = ∇·n̂  (positive for convex parent)
    v_n = M · (ΔG_chem − σ · κ)     ← capillarity opposes growth of convex grains
    v_n < 0 → interface retreats inward → parent shrinks
"""

import numpy as np
from dataclasses import dataclass
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# Thermodynamic models
# ---------------------------------------------------------------------------

@dataclass
class ThermoModel:
    """
    Free energy model.  Override bulk_driving_force / interfacial_energy / mobility
    for real alloy systems (e.g. CALPHAD).
    """
    sigma0: float = 0.5           # J/m²  baseline interfacial energy
    eps4:   float = 0.04          # 4-fold anisotropy strength
    M0:     float = 1e-8          # m³/(J·s)  baseline mobility
    Q:      float = 0.0           # activation energy
    R:      float = 8.314

    def bulk_driving_force(self, phi_parent: int, phi_product: int,
                           c: float, T: float) -> float:
        """
        ΔG_chem = G_parent − G_product  [J/m³].
        Positive  → parent is less stable → interface moves toward parent (parent shrinks).
        Negative  → parent is more stable → parent grows.
        Zero for grain growth (equal-energy phases).
        """
        return 0.0

    def interfacial_energy(self, nx: float, ny: float, g: float) -> float:
        """σ(n̂, g)  with 4-fold cubic anisotropy in crystal frame."""
        cos_g, sin_g = np.cos(g), np.sin(g)
        nx_c =  cos_g * nx + sin_g * ny
        ny_c = -sin_g * nx + cos_g * ny
        theta_c = np.arctan2(ny_c, nx_c)
        return self.sigma0 * (1.0 + self.eps4 * np.cos(4.0 * theta_c))

    def mobility(self, nx: float, ny: float, g_parent: float,
                 g_product: float, T: float) -> float:
        """Read-Shockley misorientation-dependent mobility."""
        delta_g = abs(g_parent - g_product)
        delta_g = min(delta_g, np.pi / 2 - delta_g % (np.pi / 2))
        theta_m = np.radians(15.0)
        if delta_g < 1e-10:
            mis_factor = 0.0
        elif delta_g >= theta_m:
            mis_factor = 1.0
        else:
            frac = delta_g / theta_m
            mis_factor = frac * (1.0 - np.log(frac))
        return self.M0 * mis_factor * np.exp(-self.Q / (self.R * max(T, 1.0)))


@dataclass
class SolidificationThermo(ThermoModel):
    """
    Binary alloy solidification (dilute solution approximation).
    Phase 0 = liquid, phase 1 = solid.
    """
    m_liq:   float = -500.0
    k_part:  float = 0.3
    T_melt:  float = 933.0
    L:       float = 1e8        # J/m³
    V_m:     float = 1e-5       # m³/mol

    def bulk_driving_force(self, phi_parent, phi_product, c, T):
        T_liq = self.T_melt + self.m_liq * c
        undercooling = T_liq - T          # positive in undercooled liquid
        dG = (self.L / self.T_melt) * undercooling   # > 0 when undercooled
        if phi_parent == 0 and phi_product == 1:
            # liquid→solid: ΔG = G_solid − G_liquid = −L·ΔT/Tm < 0 (spontaneous when undercooled)
            return -dG
        if phi_parent == 1 and phi_product == 0:
            # solid→liquid: ΔG = G_liquid − G_solid = +L·ΔT/Tm > 0 (opposed)
            # v_n = M(+dG − σ/R) > 0 for R > R* → solid clips at 1 (no capillary erosion)
            # v_n = M(+dG − σ/R) < 0 for R < R* → solid melts correctly
            return +dG
        return 0.0

    def mobility(self, nx, ny, g_parent, g_product, T):
        return self.M0   # isotropic for solidification


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _roll2(a: np.ndarray, di: int, dj: int) -> np.ndarray:
    if di != 0:
        a = np.roll(a, -di, axis=0)
    if dj != 0:
        a = np.roll(a, -dj, axis=1)
    return a


# ---------------------------------------------------------------------------
# GPK grid
# ---------------------------------------------------------------------------

class GPKGrid:
    """
    2-D Gaussian-Projected Kinetics simulation domain.

    Parameters
    ----------
    phi       : (Ny, Nx) integer array — initial phase labels
    alpha     : (Ny, Nx) float  — volume fractions (1 = fully parent phase)
    c         : (Ny, Nx) float  — composition
    T         : (Ny, Nx) float  — temperature
    g         : (Ny, Nx) float  — crystal orientations [radians]
    dx        : cell size [m]
    thermo    : ThermoModel
    D         : diffusivity [m²/s]
    """

    def __init__(self, phi, alpha, c, T, g, dx, thermo, D=1e-13, n_phases=2):
        self.phi     = phi.astype(np.int32)
        self.alpha   = alpha.astype(np.float64)
        self.c       = c.astype(np.float64)
        self.T       = T.astype(np.float64)
        self.g       = g.astype(np.float64)
        self.dx      = float(dx)
        self.thermo  = thermo
        self.D       = float(D)
        self.n_phases = n_phases
        self.time    = 0.0
        self.Ny, self.Nx = phi.shape

        self._nx_field      = np.zeros_like(alpha)
        self._ny_field      = np.zeros_like(alpha)
        self._kappa         = np.zeros_like(alpha)
        self._vn            = np.zeros_like(alpha)
        self._competing_phi = phi.copy()   # grain that will invade each interface cell

    # ------------------------------------------------------------------
    # Step 1: interface mask
    # ------------------------------------------------------------------
    def _interface_mask(self) -> np.ndarray:
        p = self.phi
        return (
            (p != np.roll(p,  1, axis=0)) |
            (p != np.roll(p, -1, axis=0)) |
            (p != np.roll(p,  1, axis=1)) |
            (p != np.roll(p, -1, axis=1))
        )

    # ------------------------------------------------------------------
    # Step 2: normal and curvature from phase-label field
    # ------------------------------------------------------------------
    def _reconstruct_geometry(self, mask: np.ndarray):
        """
        Compute n̂ and κ using a Gaussian-smoothed per-grain binary indicator.

        For each grain label p present at the interface:
          I_p = Gaussian_smooth((phi == p), σ)        smooth binary indicator
          n̂_p = −∇I_p / |∇I_p|                       outward normal of grain p
          κ_p = ∇·n̂_p  (divergence of normalised gradient of SAME smooth field)

        Using the same smooth field for both n̂ and κ avoids the mixed-normal
        sign-flip artefact that appears when computing ∇·n̂ across a grain boundary.

        Gaussian smoothing (σ ≈ 2 cells) creates a diffuse zone ~4 cells wide,
        making second derivatives accurate enough to recover κ = 1/R for circles.
        Cost: one gaussian_filter call per unique grain at the interface, O(N_grains·N).
        """
        from scipy.ndimage import gaussian_filter

        phi  = self.phi
        dx   = self.dx
        nx_f = self._nx_field
        ny_f = self._ny_field
        sigma = 2.0   # Gaussian width in cells

        kappa_full = np.zeros((self.Ny, self.Nx))
        nx_full    = np.zeros((self.Ny, self.Nx))
        ny_full    = np.zeros((self.Ny, self.Nx))

        for p in np.unique(phi[mask]):
            I    = (phi == p).astype(np.float64)
            I_s  = gaussian_filter(I, sigma=sigma)

            # Gradient of smooth indicator (central differences, in m⁻¹ effectively)
            Ix = (np.roll(I_s, -1, axis=1) - np.roll(I_s, 1, axis=1)) / (2.0 * dx)
            Iy = (np.roll(I_s, -1, axis=0) - np.roll(I_s, 1, axis=0)) / (2.0 * dx)
            mag = np.sqrt(Ix**2 + Iy**2)
            mag_safe = np.where(mag < 1e-20, 1.0, mag)

            # n̂ = −∇I / |∇I|
            nx_p = -Ix / mag_safe
            ny_p = -Iy / mag_safe

            # κ = ∇·n̂  (computed from the same smooth field — no cross-boundary mixing)
            d_nx_dx = (np.roll(nx_p, -1, axis=1) - np.roll(nx_p, 1, axis=1)) / (2.0 * dx)
            d_ny_dy = (np.roll(ny_p, -1, axis=0) - np.roll(ny_p, 1, axis=0)) / (2.0 * dx)
            kappa_p = d_nx_dx + d_ny_dy

            iface_p = (phi == p) & mask
            kappa_full = np.where(iface_p, kappa_p,  kappa_full)
            nx_full    = np.where(iface_p, nx_p,     nx_full)
            ny_full    = np.where(iface_p, ny_p,     ny_full)

        self._kappa[:] = kappa_full
        nx_f[:] = nx_full
        ny_f[:] = ny_full

    # ------------------------------------------------------------------
    # Step 3: interface velocities
    # ------------------------------------------------------------------
    def _compute_velocities(self, mask: np.ndarray):
        """
        v_n = M · (ΔG_chem − σ · κ)

        Sign: v_n < 0 → interface retreats inward → parent loses alpha → may convert.
        Positive ΔG_chem means the parent phase is thermodynamically unfavourable (unstable).
        """
        phi   = self.phi
        c     = self.c
        T     = self.T
        g     = self.g
        nx_f  = self._nx_field
        ny_f  = self._ny_field
        kappa = self._kappa
        thermo = self.thermo

        # Determine competing (product) phase: majority different-phase neighbour
        # For N-phase: find neighbour with most-negative driving force (simplest: majority count)
        # For 2-phase: just flip
        if self.n_phases == 2:
            phi_product = 1 - phi
        else:
            # Multi-phase grain growth: product = dominant differing neighbour
            phi_product = self._dominant_neighbour(phi)

        dG_chem = np.vectorize(thermo.bulk_driving_force)(phi, phi_product, c, T)
        sigma   = np.vectorize(thermo.interfacial_energy)(nx_f, ny_f, g)
        M_mob   = np.vectorize(thermo.mobility)(
            nx_f, ny_f, g,
            _roll2(g, 0, 1),    # use east-neighbour g as proxy for product grain
            T
        )

        # v_n = M * (ΔG_chem - σ * κ)
        # ΔG_chem > 0 and κ < 0 → parent is unstable and concave → v_n > 0 → grows
        # ΔG_chem = 0 and κ > 0 → capillarity drives shrinkage → v_n < 0 ✓
        driving = dG_chem - sigma * kappa
        self._vn[:] = np.where(mask, M_mob * driving, 0.0)

        # Competing grain: the neighbour in the outward-normal direction
        # (the phase that will invade when alpha ≤ 0)
        self._competing_phi[:] = self._normal_direction_neighbour()

    def _dominant_neighbour(self, phi: np.ndarray) -> np.ndarray:
        """Return the most-common different-phase label among 4-neighbours."""
        best = phi.copy()
        best_count = np.zeros_like(phi)
        for shift, ax in [(1, 0), (-1, 0), (1, 1), (-1, 1)]:
            nbr = np.roll(phi, shift, axis=ax)
            different = nbr != phi
            # Count occurrences: use a cheap heuristic — just take first different neighbour
            update = different & (best_count == 0)
            best = np.where(update, nbr, best)
            best_count = np.where(update, 1, best_count)
        return best

    def _normal_direction_neighbour(self) -> np.ndarray:
        """
        Return the phase label of the neighbour cell in the +n̂ direction.
        This is the 'invading' phase when the current phase loses alpha.
        """
        nx_f = self._nx_field
        ny_f = self._ny_field
        phi  = self.phi

        # Dominant axis of n̂
        use_x = np.abs(nx_f) >= np.abs(ny_f)

        # x-direction neighbour
        east  = np.roll(phi, -1, axis=1)   # phi at (i, j+1)
        west  = np.roll(phi,  1, axis=1)   # phi at (i, j-1)
        nbr_x = np.where(nx_f >= 0, east, west)

        # y-direction neighbour
        south = np.roll(phi, -1, axis=0)   # phi at (i+1, j)
        north = np.roll(phi,  1, axis=0)   # phi at (i-1, j)
        nbr_y = np.where(ny_f >= 0, south, north)

        return np.where(use_x, nbr_x, nbr_y)

    # ------------------------------------------------------------------
    # Step 4: update volume fractions and phase labels
    # ------------------------------------------------------------------
    def _update_alpha(self, dt: float, mask: np.ndarray):
        dx     = self.dx
        dalpha = self._vn * dt / dx

        alpha_new = self.alpha + dalpha

        # Cells where alpha > 1: parent grew, clip
        alpha_new = np.minimum(alpha_new, 1.0)

        # Cells where alpha ≤ 0: parent fully lost → convert to competitor
        revert = mask & (alpha_new <= 0.0)
        self.phi   = np.where(revert, self._competing_phi, self.phi)
        alpha_new  = np.where(revert, 1.0, alpha_new)

        self.alpha = np.clip(alpha_new, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Step 5: diffusion with Stefan flux-jump
    # ------------------------------------------------------------------
    def _diffuse_composition(self, dt: float, mask: np.ndarray):
        D   = self.D
        dx  = self.dx
        c   = self.c
        lap = (
            np.roll(c, 1,  axis=0) + np.roll(c, -1, axis=0) +
            np.roll(c, 1,  axis=1) + np.roll(c, -1, axis=1) - 4 * c
        ) / dx**2
        D_eff    = np.where(mask, D * self.alpha, D)
        self.c   = c + dt * D_eff * lap

    # ------------------------------------------------------------------
    # Stable timestep
    # ------------------------------------------------------------------
    def max_dt(self) -> float:
        dt_diff  = 0.4 * self.dx**2 / (4 * max(self.D, 1e-30))
        vmax     = np.max(np.abs(self._vn))
        dt_iface = 0.4 * self.dx / max(vmax, 1e-30)
        return min(dt_diff, dt_iface)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
    def step(self, dt: Optional[float] = None) -> float:
        mask = self._interface_mask()
        self._reconstruct_geometry(mask)
        self._compute_velocities(mask)
        if dt is None:
            dt = self.max_dt()
        self._update_alpha(dt, mask)
        self._diffuse_composition(dt, mask)
        self.time += dt
        return dt

    def run(self, t_end: float, dt: Optional[float] = None,
            callback: Optional[Callable] = None,
            callback_interval: int = 100) -> int:
        step_count = 0
        while self.time < t_end:
            dt_used = self.step(dt)
            step_count += 1
            if callback and step_count % callback_interval == 0:
                callback(self)
        return step_count

    # ------------------------------------------------------------------
    # Smooth rendering using sub-cell alpha blending
    # ------------------------------------------------------------------
    def render_smooth(self,
                      field: str = "grains",
                      cmap_name: str = "tab20",
                      phase_colors: Optional[tuple] = None) -> np.ndarray:
        """
        Return an (Ny, Nx, 3) float32 RGB image with smooth, anti-aliased
        grain boundaries.

        At every interface cell the pixel colour is blended:
            colour = alpha * colour(phi_own) + (1 - alpha) * colour(phi_comp)

        This converts the staircase step-change of the integer phi field
        into a smooth gradient that accurately represents the sub-cell
        boundary position encoded in alpha.

        Parameters
        ----------
        field : "grains" | "phase" | "alpha" | "composition"
            What to render.
        cmap_name : matplotlib colormap name used for "grains" mode.
        phase_colors : (solid_rgb, liquid_rgb) tuples for "phase" mode;
            defaults to ((0.12, 0.39, 0.78), (0.9, 0.9, 0.9)).

        Returns
        -------
        img : (Ny, Nx, 3) float32 array in [0, 1].
        """
        import matplotlib.pyplot as plt

        Ny, Nx = self.phi.shape
        phi  = self.phi
        a    = np.clip(self.alpha, 0.0, 1.0)
        comp = self._competing_phi   # set during step_compute_velocities

        # ---- "grains" mode -----------------------------------------------
        if field == "grains":
            cmap = plt.get_cmap(cmap_name)
            n_max = int(phi.max()) + 2
            # Build (n_max, 3) colour table — one row per grain label
            palette = np.array([cmap(k % 20)[:3] for k in range(n_max)],
                               dtype=np.float32)

            img_own  = palette[phi]        # (Ny, Nx, 3)
            img_comp = palette[np.clip(comp, 0, n_max - 1)]

            # Blend where the cell is at an interface and competitor differs
            is_iface = (comp != phi)[..., np.newaxis]          # (Ny, Nx, 1)
            w        = a[..., np.newaxis]                       # (Ny, Nx, 1)
            img = np.where(is_iface,
                           w * img_own + (1.0 - w) * img_comp,
                           img_own).astype(np.float32)
            return np.clip(img, 0.0, 1.0)

        # ---- "phase" mode (solidification / precipitation) ---------------
        elif field == "phase":
            if phase_colors is None:
                solid_col  = np.array([0.12, 0.39, 0.78], dtype=np.float32)
                liquid_col = np.array([0.90, 0.90, 0.90], dtype=np.float32)
            else:
                solid_col  = np.asarray(phase_colors[0], dtype=np.float32)
                liquid_col = np.asarray(phase_colors[1], dtype=np.float32)

            # Base image: solid=blue, liquid=grey
            is_solid = (phi == 1)[..., np.newaxis]
            img_base = np.where(is_solid, solid_col, liquid_col)

            # Blend interface cells
            is_iface = (comp != phi)[..., np.newaxis]
            w        = a[..., np.newaxis]
            comp_col = np.where(is_solid, liquid_col, solid_col)
            img = np.where(is_iface,
                           w * img_base + (1.0 - w) * comp_col,
                           img_base).astype(np.float32)
            return np.clip(img, 0.0, 1.0)

        # ---- "alpha" mode ------------------------------------------------
        elif field == "alpha":
            # Grey-scale image of the alpha field
            img = np.stack([a, a, a], axis=-1).astype(np.float32)
            return img

        # ---- "composition" mode ------------------------------------------
        elif field == "composition":
            cmap = plt.get_cmap("RdYlBu_r")
            c = self.c
            lo, hi = c.min(), c.max()
            if hi > lo:
                t = (c - lo) / (hi - lo)
            else:
                t = np.zeros_like(c)
            img = np.array([[cmap(v)[:3] for v in row]
                            for row in t], dtype=np.float32)
            return img

        else:
            raise ValueError(f"Unknown field '{field}'. "
                             "Use 'grains', 'phase', 'alpha', or 'composition'.")
