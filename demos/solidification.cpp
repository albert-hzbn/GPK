/*
 * Demo 2 — Dendritic Solidification of a Binary Alloy (3-D)
 * Input:  inputs/solidification.ini
 * Output: output/solidification/snap_NNNN.{vtk|csv|ppm}
 */
#include "gpk_core.hpp"
#include "gpk_thermo.hpp"
#include "gpk_io.hpp"

#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>
#include <chrono>
#include <omp.h>

using namespace gpk;

int main(int argc, char* argv[]) {
    const char* ini = (argc > 1) ? argv[1]
                                  : "inputs/solidification.ini";
    InputParser cfg(ini);

    const int    Nz    = cfg.get_int   ("grid",  "Nz");
    const int    Ny    = cfg.get_int   ("grid",  "Ny");
    const int    Nx    = cfg.get_int   ("grid",  "Nx");
    const float  dx    = cfg.get_float ("grid",  "dx");
    const double t_end = cfg.get_double("time",  "t_end");
    const int    nsnap = cfg.get_int   ("time",  "n_snaps");

    const float M0     = cfg.get_float("thermo","M0");
    const float sig0   = cfg.get_float("thermo","sigma0");
    const float eps4   = cfg.get_float("thermo","eps4", 0.04f);
    const float D      = cfg.get_float("thermo","D");
    const float m_liq  = cfg.get_float("thermo","m_liq");
    const float k_part = cfg.get_float("thermo","k_part");
    const float T_melt = cfg.get_float("thermo","T_melt");
    const float L      = cfg.get_float("thermo","L");
    const bool  stochastic = cfg.get_bool("thermo", "stochastic", false);
    const float noise_scale = cfg.get_float("thermo", "noise_scale", 1.0f);
    const bool  elastic_enabled = cfg.get_bool("thermo", "elastic_enabled", false);
    const float elastic_modulus = cfg.get_float("thermo", "elastic_modulus", 0.0f);
    const float eigenstrain_coeff = cfg.get_float("thermo", "eigenstrain_coeff", 0.0f);
    const int   elastic_iters = cfg.get_int("thermo", "elastic_smooth_iters", 8);

    const float c0     = cfg.get_float("initial","c0");
    const float dT     = cfg.get_float("initial","undercooling");
    const int   seed_r = cfg.get_int  ("initial","seed_radius", 4);
    const float c_ref = cfg.get_float("thermo", "c_ref", c0);

    printf("=== GPK Solidification (3-D) ===\n");
    printf("Grid %d×%d×%d, dx=%.2f µm\n", Nz,Ny,Nx, dx*1e6f);
    printf("OpenMP threads: %d\n\n", omp_get_max_threads());

    auto thermo = std::make_shared<SolidificationThermo>(
        M0, sig0, eps4, m_liq, k_part, T_melt, L);
    thermo->stochastic = stochastic;
    thermo->noise_scale = noise_scale;
    thermo->elastic_enabled = elastic_enabled;
    thermo->elastic_modulus = elastic_modulus;
    thermo->eigenstrain_coeff = eigenstrain_coeff;
    thermo->c_ref = c_ref;
    thermo->elastic_smooth_iters = elastic_iters;

    GPKGrid grid(Nz, Ny, Nx, dx, thermo, D);

    float T_liq = T_melt + m_liq * c0;
    std::fill(grid.c.begin(), grid.c.end(), c0);
    std::fill(grid.T.begin(), grid.T.end(), T_liq - dT);

    float R_star = sig0 * T_melt / (L * dT);
    printf("Gibbs-Thomson R* = %.3f µm\n", R_star * 1e6f);
    printf("Seed radius      = %.1f µm (must be > R*)\n\n", seed_r*dx*1e6f);
    if (stochastic) {
        printf("Stochastic kinetics enabled, noise_scale = %.3f\n", (double)noise_scale);
    }
    if (elastic_enabled) {
        printf("Elastic coupling enabled, modulus = %.3e J/m^3, coeff = %.3e\n\n",
               (double)elastic_modulus, (double)eigenstrain_coeff);
    }

    // Solid seed at domain centre
    int cz=Nz/2, cy=Ny/2, cx=Nx/2;
    for (int dz=-seed_r; dz<=seed_r; ++dz)
    for (int dy=-seed_r; dy<=seed_r; ++dy)
    for (int dx_=-seed_r; dx_<=seed_r; ++dx_) {
        if (dz*dz+dy*dy+dx_*dx_ > seed_r*seed_r) continue;
        int zz=cz+dz, yy=cy+dy, xx=cx+dx_;
        if (zz<0||zz>=Nz||yy<0||yy>=Ny||xx<0||xx>=Nx) continue;
        int i = grid.idx(zz,yy,xx);
        grid.phi[i] = 1;
        grid.c[i]   = c0 * k_part;
    }

    OutputWriter writer(cfg);
    double snap_dt = t_end / nsnap;

    printf("%-10s %-14s %-14s %-10s\n",
           "Time[ms]", "Solid frac", "Solute max", "Steps");

    auto t_wall0 = std::chrono::high_resolution_clock::now();
    int total_steps = 0;

    for (int snap = 0; snap < nsnap; ++snap) {
        int steps = grid.run(snap_dt * (snap+1), -1.f);
        total_steps += steps;

        float sf   = grid.solid_fraction();
        float cmax = *std::max_element(grid.c.begin(), grid.c.end());
        printf("%-10.3f %-14.4f %-14.4f %-10d\n",
               grid.time*1e3, sf, cmax, steps);

        writer.write(grid, snap, grid.time);
    }

    double wall_s = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - t_wall0).count();
    printf("\nWall time: %.2f s  (%d steps, %.2f ms/step)\n",
           wall_s, total_steps, 1e3*wall_s/total_steps);
    return 0;
}
