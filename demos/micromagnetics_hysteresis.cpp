/*
 * micromagnetics_hysteresis.cpp — standalone demo/validation for the
 * optional sub-grid domain-wall extension (include/gpk_micromagnetics.hpp).
 *
 * This is a SEPARATE module from the main GPK simulation loop: it is not
 * wired into GPKGrid::step() and costs nothing unless explicitly used, per
 * the header's design notes. This demo exercises it standalone: build a
 * polycrystal with randomly oriented (in-plane) easy axes, then trace out a
 * hysteresis loop M(H)/Ms with the deterministic (T=0) Stoner-Wohlfarth
 * switching rule, gated by the single-domain-radius criterion
 *.
 *
 * Validation target: for a RANDOM, IN-PLANE (2-D) distribution of uniaxial
 * easy axes, at H=0 a saturated particle's magnetization relaxes exactly
 * onto its own easy axis (the Zeeman term vanishes), on the same side it was
 * saturated toward, so M_remanent = Ms*cos(theta) where theta is the angle
 * between that grain's easy axis and the field direction. Averaging
 * |cos(theta)| UNIFORMLY over theta in [0, pi/2) (planar, not solid-angle,
 * distribution -- no sin(theta) weight, unlike the classical 3-D-isotropic
 * derivation that gives the famous Mr/Ms=0.5):
 *
 *   <cos theta> = [sin(theta)]_0^(pi/2) / (pi/2) = 2/pi ~= 0.6366
 *
 * is the planar analogue of that 3-D result, and is what this demo checks.
 *
 * Usage: ./micromagnetics_hysteresis
 */

#include "gpk_core.hpp"
#include "gpk_thermo.hpp"
#include "gpk_micromagnetics.hpp"

#include <cstdio>
#include <cmath>
#include <climits>
#include <random>
#include <vector>

using namespace gpk;

int main()
{
    const int   Ny = 256, Nx = 256;
    const float dx = 5.0e-9f;     // 5 nm cells -- fine relative to R_sd so most
                                   // grains land at/below the single-domain radius
    const int   N_GRAINS = 2300;
    const uint32_t seed = 7;

    // Nd2Fe14B-like parameters (Ku, Ms match inputs/magnetic_annealing.ini;
    // A_exch a representative literature exchange stiffness).
    const float Ku     = 4.3e6f;
    const float Ms     = 1.28e6f;
    const float A_exch = 7.7e-12f;

    auto thermo = std::make_shared<IsotropicGrowthThermo>(1e-8f, 0.324f);
    GPKGrid grid(1, Ny, Nx, dx, thermo, 0.f);

    // Simple nearest-seed Voronoi tessellation with in-plane random easy axes
    // (g uniform over [0, pi): uniaxial easy-axis symmetry, matching
    // MicromagneticExtension's use of g as the easy-axis angle).
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> ry(0, Ny - 1), rx(0, Nx - 1);
    std::uniform_real_distribution<float> rg(0.f, (float)M_PI);
    std::vector<int> sy(N_GRAINS), sx(N_GRAINS);
    std::vector<float> sg(N_GRAINS);
    for (int k = 0; k < N_GRAINS; ++k) { sy[k] = ry(rng); sx[k] = rx(rng); sg[k] = rg(rng); }

    for (int y = 0; y < Ny; ++y)
    for (int x = 0; x < Nx; ++x) {
        long best_d = LONG_MAX; int best = 0;
        for (int k = 0; k < N_GRAINS; ++k) {
            long dy = y - sy[k], dxp = x - sx[k];
            long d = dy*dy + dxp*dxp;
            if (d < best_d) { best_d = d; best = k; }
        }
        int i = grid.idx(0, y, x);
        grid.phi[i]   = best + 1;
        grid.g[i]     = sg[best];
        grid.alpha[i] = 1.f;
    }

    MicromagneticExtension mm(Ku, Ms, A_exch, seed);
    mm.alpha_damping = 0.8f; // strong damping -> fast convergence to equilibrium per field step
    mm.init_from_grid(grid);

    int n_single = 0, n_multi = 0;
    const float rsd = mm.R_sd();
    for (auto& kv : mm.states())
        (kv.second.n_domains == 1) ? ++n_single : ++n_multi;

    printf("=== GPK Micromagnetic Extension: Hysteresis Loop ===\n");
    printf("Nd2Fe14B-like: Ku=%.2e J/m^3  Ms=%.3e A/m  A=%.2e J/m\n", (double)Ku, (double)Ms, (double)A_exch);
    printf("Single-domain radius R_sd = %.2f nm (cell size dx=%.1f nm)\n", rsd*1e9, dx*1e9);
    printf("%d grains: %d single-domain (coherent macrospin), %d multi-domain (sub-spin partition)\n\n",
           N_GRAINS, n_single, n_multi);

    const float H_K     = 2.f * Ku / (MicromagneticExtension::MU0 * Ms);
    const float H_max   = 1.3f * H_K;
    const int   n_steps = 60;
    const float theta_H = 0.f;

    mm.saturate_along(theta_H);

    std::vector<float> H_hist, M_hist;

    printf("%-12s %-10s\n", "H/H_K", "M/Ms");
    // Descending branch: +H_max -> -H_max
    for (int k = 0; k <= n_steps; ++k) {
        float H = H_max * (1.f - 2.f * k / n_steps);
        mm.relax_to_equilibrium(H, theta_H, 0.f, 2.0e-14f, 300);
        float M = mm.mean_reduced_magnetization(theta_H);
        H_hist.push_back(H); M_hist.push_back(M);
        if (k % 6 == 0) printf("%-12.3f %-10.4f\n", (double)(H/H_K), (double)M);
    }
    const size_t n_descending = H_hist.size();
    // Ascending branch: -H_max -> +H_max
    for (int k = 0; k <= n_steps; ++k) {
        float H = -H_max * (1.f - 2.f * k / n_steps);
        mm.relax_to_equilibrium(H, theta_H, 0.f, 2.0e-14f, 300);
        float M = mm.mean_reduced_magnetization(theta_H);
        H_hist.push_back(H); M_hist.push_back(M);
    }

    // Remanence: M at |H| closest to 0 on the descending branch.
    size_t i0 = 0; float best = 1e30f;
    for (size_t k = 0; k < n_descending; ++k)
        if (std::fabs(H_hist[k]) < best) { best = std::fabs(H_hist[k]); i0 = k; }
    const float Mr        = M_hist[i0];
    const float Mr_theory = 2.f / (float)M_PI;
    const float err       = std::fabs(Mr - Mr_theory) / Mr_theory * 100.f;

    printf("\nRemanence Mr/Ms = %.4f  (theory, random in-plane easy axes: 2/pi = %.4f)  error = %.2f%%\n",
           (double)Mr, (double)Mr_theory, (double)err);

    const bool passed = err < 10.f; // single-domain-only subset + finite-ensemble tolerance
    printf("\nMicromagnetic hysteresis test: %s\n",
           passed ? "PASS -- remanence matches random-in-plane Stoner-Wohlfarth theory"
                  : "FAIL -- remanence deviates from theory beyond tolerance");
    return passed ? 0 : 1;
}
