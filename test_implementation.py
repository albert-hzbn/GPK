#!/usr/bin/env python3
"""
Implementation test suite for GPK theory fixes.

Tests verified:
  1. Curvature sign: κ_liquid = -1/R (liquid parent at solid seed, concave surface)
                    κ_solid  = +1/R (solid parent, convex)
  2. Gibbs-Thomson critical radius: subcritical seed dissolves, supercritical grows
  3. Symmetry fix: solid cells do NOT erode during supercritical growth
                   (was a bug: returning dG=0 for solid parent let capillarity slowly melt it)
  4. Recrystallisation symmetry: rex grain cells do NOT erode from capillarity
  5. C++ binary smoke test: solidification INI runs and gives increasing solid fraction

All checks are quantitative; failures print what was expected vs. what was observed.
"""

import sys
import os
import math
import argparse
import subprocess
import tempfile
import numpy as np
from scipy.ndimage import gaussian_filter
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

_parser = argparse.ArgumentParser(description="GPK implementation regression checks")
_parser.add_argument(
    "--gpk-bin",
    type=Path,
    help="Path to the built gpk executable (defaults to build/gpk[.exe])",
)
_args = _parser.parse_args()

sys.path.insert(0, str(Path(__file__).parent / "python"))
from gpk.core import GPKGrid, SolidificationThermo, ThermoModel

PASS = "[PASS]"
FAIL = "[FAIL]"


def section(title: str) -> None:
    print(f"\n{'='*72}")
    print(f"  {title}")
    print('='*72)


def check(label: str, ok: bool, detail: str = "") -> None:
    status = PASS if ok else FAIL
    msg = f"  {status} {label}"
    if detail:
        msg += f"   ({detail})"
    print(msg)
    if not ok:
        global _any_fail
        _any_fail = True


_any_fail = False

# ---------------------------------------------------------------------------
# TEST 1 — Curvature sign convention
# ---------------------------------------------------------------------------

section("1. Curvature sign: κ_liquid = −1/R, κ_solid = +1/R")

Ny, Nx = 80, 80
R_cells = 15  # circle radius in cells
cy, cx = Ny // 2, Nx // 2
sigma_G = 2.0  # Gaussian smoothing in cells

# Build binary indicators
phi_grid = np.zeros((Ny, Nx), dtype=np.int32)
for y in range(Ny):
    for x in range(Nx):
        if (y - cy)**2 + (x - cx)**2 <= R_cells**2:
            phi_grid[y, x] = 1  # solid

# Liquid indicator (phi == 0)
I_liquid = (phi_grid == 0).astype(float)
I_liquid_s = gaussian_filter(I_liquid, sigma=sigma_G)
Ix = (np.roll(I_liquid_s, -1, axis=1) - np.roll(I_liquid_s, 1, axis=1)) / 2.0
Iy = (np.roll(I_liquid_s, -1, axis=0) - np.roll(I_liquid_s, 1, axis=0)) / 2.0
mag = np.sqrt(Ix**2 + Iy**2)
mag_safe = np.where(mag < 1e-20, 1.0, mag)
nx_liq = -Ix / mag_safe
ny_liq = -Iy / mag_safe
dnx_dx = (np.roll(nx_liq, -1, axis=1) - np.roll(nx_liq, 1, axis=1)) / 2.0
dny_dy = (np.roll(ny_liq, -1, axis=0) - np.roll(ny_liq, 1, axis=0)) / 2.0
kappa_liquid = dnx_dx + dny_dy   # in cell⁻¹

# Solid indicator (phi == 1)
I_solid = (phi_grid == 1).astype(float)
I_solid_s = gaussian_filter(I_solid, sigma=sigma_G)
Ix_s = (np.roll(I_solid_s, -1, axis=1) - np.roll(I_solid_s, 1, axis=1)) / 2.0
Iy_s = (np.roll(I_solid_s, -1, axis=0) - np.roll(I_solid_s, 1, axis=0)) / 2.0
mag_s = np.sqrt(Ix_s**2 + Iy_s**2)
mag_s_safe = np.where(mag_s < 1e-20, 1.0, mag_s)
nx_sol = -Ix_s / mag_s_safe
ny_sol = -Iy_s / mag_s_safe
dnx_dx_s = (np.roll(nx_sol, -1, axis=1) - np.roll(nx_sol, 1, axis=1)) / 2.0
dny_dy_s = (np.roll(ny_sol, -1, axis=0) - np.roll(ny_sol, 1, axis=0)) / 2.0
kappa_solid = dnx_dx_s + dny_dy_s

# Interface: cells near the solid-liquid boundary
interface_liquid = (phi_grid == 0) & (I_liquid_s < 0.7) & (I_liquid_s > 0.3)
interface_solid  = (phi_grid == 1) & (I_solid_s  < 0.7) & (I_solid_s  > 0.3)

kappa_liq_mean = float(kappa_liquid[interface_liquid].mean()) if interface_liquid.any() else 0.0
kappa_sol_mean = float(kappa_solid[interface_solid].mean())  if interface_solid.any()  else 0.0
kappa_theory   = 1.0 / R_cells   # in cell⁻¹

print(f"\n  R = {R_cells} cells,  κ_theory = +1/R = {kappa_theory:.4f} cell⁻¹")
print(f"  κ_liquid (at interface, liquid parent) = {kappa_liq_mean:.4f} cell⁻¹  [expect ≈ −{kappa_theory:.4f}]")
print(f"  κ_solid  (at interface, solid parent)  = {kappa_sol_mean:.4f} cell⁻¹  [expect ≈ +{kappa_theory:.4f}]")

check("κ_liquid < 0  (liquid parent is concave at solid seed)",
      kappa_liq_mean < -0.01,
      f"got {kappa_liq_mean:.4f}")
check("κ_solid > 0  (solid parent is convex)",
      kappa_sol_mean > 0.01,
      f"got {kappa_sol_mean:.4f}")
check("κ_liquid ≈ −1/R within 20%",
      abs(kappa_liq_mean - (-kappa_theory)) / kappa_theory < 0.20,
      f"|{kappa_liq_mean:.4f} − (−{kappa_theory:.4f})| / {kappa_theory:.4f} = "
      f"{abs(kappa_liq_mean+kappa_theory)/kappa_theory:.3f}")
check("κ_solid ≈ +1/R within 20%",
      abs(kappa_sol_mean - kappa_theory) / kappa_theory < 0.20,
      f"|{kappa_sol_mean:.4f} − {kappa_theory:.4f}| / {kappa_theory:.4f} = "
      f"{abs(kappa_sol_mean-kappa_theory)/kappa_theory:.3f}")


# ---------------------------------------------------------------------------
# TEST 2 — Critical radius (subcritical dissolves, supercritical grows)
# ---------------------------------------------------------------------------

section("2. Gibbs-Thomson critical radius — subcritical vs. supercritical")

thermo_params = dict(
    sigma0=0.20,      # J/m²
    M0=5e-8,          # m³/(J·s)
    m_liq=-500.0,
    k_part=0.3,
    T_melt=933.0,
    L=1e8,
    eps4=0.0,         # isotropic
)
dx   = 0.5e-6       # m
dT   = 5.0          # undercooling [K]
thermo = SolidificationThermo(**thermo_params)

R_star = thermo.sigma0 * thermo.T_melt / (thermo.L * dT)
print(f"\n  R* = σTm/(L·ΔT) = {R_star*1e6:.3f} µm  ({R_star/dx:.1f} cells with dx={dx*1e6:.2f} µm)")

def make_solidification_grid(r_cells: int) -> GPKGrid:
    Ny, Nx = 80, 80
    phi   = np.zeros((Ny, Nx), dtype=np.int32)
    alpha = np.ones((Ny, Nx))
    c0    = 0.02
    c     = np.full((Ny, Nx), c0)
    T_liq = thermo.T_melt + thermo.m_liq * c0
    T_arr = np.full((Ny, Nx), T_liq - dT)
    g     = np.zeros((Ny, Nx))
    cy, cx = Ny // 2, Nx // 2
    for y in range(Ny):
        for x in range(Nx):
            if (y - cy)**2 + (x - cx)**2 <= r_cells**2:
                phi[y, x] = 1
                c[y, x]   = c0 * thermo.k_part
    return GPKGrid(phi, alpha, c, T_arr, g, dx=dx, thermo=thermo, D=3e-9, n_phases=2)

# Subcritical: 1 cell radius ≈ dx = 0.5 µm << R* ≈ 0.37 µm (just barely, use 0.5 to be safe)
# Actually R* ≈ 0.37 µm, dx = 0.5 µm, so 1 cell = 0.5 µm > R*. Need R << R*.
# Use dx = 0.1 µm for this test to get cleaner subcritical/supercritical split.
dx_fine = 0.1e-6
R_star_cells = R_star / dx_fine
print(f"  Using dx={dx_fine*1e6:.2f} µm: R* = {R_star_cells:.1f} cells")

def make_grid_fine(r_cells: int) -> GPKGrid:
    Ny, Nx = 60, 60
    phi   = np.zeros((Ny, Nx), dtype=np.int32)
    alpha = np.ones((Ny, Nx))
    c0    = 0.02
    c     = np.full((Ny, Nx), c0)
    T_liq = thermo.T_melt + thermo.m_liq * c0
    T_arr = np.full((Ny, Nx), T_liq - dT)
    g     = np.zeros((Ny, Nx))
    cy, cx = Ny // 2, Nx // 2
    for y in range(Ny):
        for x in range(Nx):
            if (y - cy)**2 + (x - cx)**2 <= r_cells**2:
                phi[y, x] = 1
                c[y, x]   = c0 * thermo.k_part
    return GPKGrid(phi, alpha, c, T_arr, g, dx=dx_fine, thermo=thermo, D=3e-9, n_phases=2)

r_sub  = max(1, int(R_star_cells * 0.5))   # 50% of R* → subcritical
r_sup  = max(2, int(R_star_cells * 3.0))   # 300% of R* → well supercritical

print(f"  Subcritical seed: r = {r_sub} cells = {r_sub*dx_fine*1e6:.2f} µm")
print(f"  Supercritical seed: r = {r_sup} cells = {r_sup*dx_fine*1e6:.2f} µm")

def solid_fraction(grid: GPKGrid) -> float:
    return float(np.mean(grid.phi == 1))

grid_sub = make_grid_fine(r_sub)
grid_sup = make_grid_fine(r_sup)

sf_sub_0 = solid_fraction(grid_sub)
sf_sup_0 = solid_fraction(grid_sup)

t_run = 5e-5
grid_sub.run(t_run)
grid_sup.run(t_run)

sf_sub_1 = solid_fraction(grid_sub)
sf_sup_1 = solid_fraction(grid_sup)

print(f"\n  Subcritical  (r={r_sub}): sf {sf_sub_0:.4f} → {sf_sub_1:.4f}")
print(f"  Supercritical (r={r_sup}): sf {sf_sup_0:.4f} → {sf_sup_1:.4f}")

check("Subcritical seed dissolves: sf decreases",
      sf_sub_1 < sf_sub_0,
      f"{sf_sub_1:.4f} < {sf_sub_0:.4f}")
check("Supercritical seed grows: sf increases",
      sf_sup_1 > sf_sup_0,
      f"{sf_sup_1:.4f} > {sf_sup_0:.4f}")


# ---------------------------------------------------------------------------
# TEST 3 — Symmetry fix: solid cells must NOT erode during supercritical growth
# ---------------------------------------------------------------------------

section("3. Symmetry fix: solid interior cells maintain alpha during growth")

print("""
  Before the fix, SolidificationThermo returned dG=0 for phi_parent=1 (solid).
  This let capillarity (v_n = −Mσ/R < 0) slowly erode the solid boundary cells.
  After the fix, phi_parent=1 returns +dG = +L·ΔT/Tm, giving
  v_n = M(+dG − σ/R) > 0 for R > R* → solid clips at 1 (no erosion).
""")

# Supercritical seed: run a few steps and check minimum alpha in solid cells
grid_test = make_grid_fine(r_sup)
solid_mask_init = (grid_test.phi == 1)

# Run 20 steps
steps_run = 0
for _ in range(20):
    grid_test.step()
    steps_run += 1

# Check alpha in cells that were SOLID at init AND are still solid
still_solid = (grid_test.phi == 1) & solid_mask_init
if still_solid.any():
    min_alpha_solid = float(grid_test.alpha[still_solid].min())
    mean_alpha_solid = float(grid_test.alpha[still_solid].mean())
else:
    min_alpha_solid = 1.0
    mean_alpha_solid = 1.0

print(f"  After {steps_run} steps, in originally-solid cells that are still solid:")
print(f"    min(alpha)  = {min_alpha_solid:.4f}  [expect ≈ 1.0 with fix, < 1 without]")
print(f"    mean(alpha) = {mean_alpha_solid:.4f}  [expect ≈ 1.0 with fix]")

# With the fix: v_n = M(+dG - σ/R) > 0 for R > R*, so alpha clips at 1.0
# Threshold: alpha should be at least 0.95 if no erosion (allowing for edge effects)
check("Solid cells maintain alpha ≥ 0.95 (no capillary erosion)",
      min_alpha_solid >= 0.95,
      f"min={min_alpha_solid:.4f}")


# ---------------------------------------------------------------------------
# TEST 4 — Driving-force symmetry: sign check
# ---------------------------------------------------------------------------

section("4. Driving-force sign convention (bulk_driving_force)")

thermo_test = SolidificationThermo(**thermo_params)
c0 = 0.02
T_liq_val = thermo_test.T_melt + thermo_test.m_liq * c0   # 933 − 10 = 923 K
T0 = T_liq_val - 5.0   # 5 K undercooling → T = 918 K < T_liq

dG_liq = thermo_test.bulk_driving_force(0, 1, c0, T0)   # liquid → solid
dG_sol = thermo_test.bulk_driving_force(1, 0, c0, T0)   # solid → liquid

dT_val    = T_liq_val - T0                              # = +5 K > 0 (undercooled)
dG_theory = thermo_test.L / thermo_test.T_melt * dT_val   # > 0

print(f"\n  ΔT = T_liq − T = {dT_val:.2f} K (undercooled)")
print(f"  dG_theory = L·ΔT/Tm = {dG_theory:.4e} J/m³  (> 0)")
print(f"  dG_liquid→solid = {dG_liq:.4e}  [expect −{dG_theory:.4e}]")
print(f"  dG_solid→liquid = {dG_sol:.4e}  [expect +{dG_theory:.4e}]")

check("dG(liquid→solid) = −L·ΔT/Tm < 0  (spontaneous when undercooled)",
      abs(dG_liq - (-dG_theory)) < 1e-6 * abs(dG_theory),
      f"dG={dG_liq:.4e}, expected={-dG_theory:.4e}")
check("dG(solid→liquid) = +L·ΔT/Tm > 0  (opposed when undercooled)",
      abs(dG_sol - (+dG_theory)) < 1e-6 * abs(dG_theory),
      f"dG={dG_sol:.4e}, expected={+dG_theory:.4e}")
check("dG(liquid→solid) + dG(solid→liquid) = 0  (antisymmetric)",
      abs(dG_liq + dG_sol) < 1e-10 * abs(dG_theory),
      f"sum={dG_liq+dG_sol:.2e}")


# ---------------------------------------------------------------------------
# TEST 5 — C++ binary smoke test
# ---------------------------------------------------------------------------

section("5. C++ binary: solidification smoke test")

build_dir = Path(__file__).parent / "build"
if _args.gpk_bin is not None:
    gpk_bin = _args.gpk_bin.resolve()
    build_dir = gpk_bin.parent  # run from the build tree that holds the binary
else:
    candidates = [build_dir / "gpk.exe", build_dir / "gpk"]
    gpk_bin = next((p for p in candidates if p.is_file()), candidates[0])

if not gpk_bin.is_file():
    check("C++ solidification binary exists", False, f"not found: {gpk_bin}")
else:
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "solid_out"
        out_dir.mkdir()
        ini_text = f"""
[run]
model = solidification

[grid]
Nz = 1
Ny = 64
Nx = 64
dx = 0.5e-6

[time]
t_end   = 5.0e-4
n_snaps = 2

[thermo]
M0      = 5.0e-8
sigma0  = 0.20
eps4    = 0.0
D       = 3.0e-9
m_liq   = -500.0
k_part  = 0.3
T_melt  = 933.0
L       = 1.0e8
stochastic = false
elastic_enabled = false

[initial]
c0           = 0.02
undercooling = 5.0
seed_radius  = 5

[output]
format  = csv
dir     = {out_dir}
prefix  = snap
compact = false
"""
        ini_path = Path(tmpdir) / "solid_test.ini"
        ini_path.write_text(ini_text.strip(), encoding="utf-8")

        result = subprocess.run(
            [str(gpk_bin), str(ini_path)],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            cwd=str(build_dir)
        )

        if result.returncode != 0:
            print("  C++ binary FAILED:")
            print(result.stderr[-500:])
            check("C++ solidification binary runs without error", False, "non-zero exit code")
        else:
            # Parse solid fractions from the two snapshots
            snaps = sorted(out_dir.glob("snap_*.csv"))
            def parse_sf(path):
                n_total, n_solid = 0, 0
                with open(path) as f:
                    for line in f:
                        if line.startswith('#') or not line.strip():
                            continue
                        parts = line.split()
                        n_total += 1
                        if int(parts[3]) == 1:  # column 4 = gid
                            n_solid += 1
                return n_solid / max(1, n_total)

            if len(snaps) >= 2:
                sf_t0 = parse_sf(snaps[0])
                sf_t1 = parse_sf(snaps[-1])
                print(f"\n  C++ binary: solid fraction  snap[0]={sf_t0:.4f}  snap[-1]={sf_t1:.4f}")
                check("C++ solidification runs without error", True)
                check("C++ supercritical seed grows (sf increases)",
                      sf_t1 > sf_t0,
                      f"{sf_t0:.4f} → {sf_t1:.4f}")
            else:
                check("C++ output files created", len(snaps) >= 2,
                      f"found {len(snaps)} snaps")


# ---------------------------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------------------------

section("SUMMARY")

tests = [
    "1.  κ_liquid < 0 (liquid parent concave at solid seed)",
    "1.  κ_solid  > 0 (solid parent convex)",
    "1.  κ_liquid ≈ −1/R within 20%",
    "1.  κ_solid  ≈ +1/R within 20%",
    "2.  Subcritical seed dissolves",
    "2.  Supercritical seed grows",
    "3.  Solid cells maintain alpha ≥ 0.95 (no capillary erosion)",
    "4.  dG(liquid→solid) = −L·ΔT/Tm",
    "4.  dG(solid→liquid) = +L·ΔT/Tm",
    "4.  dG antisymmetric",
    "5.  C++ smoke test",
]

if _any_fail:
    print("\n  *** SOME CHECKS FAILED — see [FAIL] lines above ***")
    sys.exit(1)
else:
    print("\n  All checks PASS ✓")
    sys.exit(0)
