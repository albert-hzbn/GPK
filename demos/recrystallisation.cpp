/*
 * Demo 3 — Recrystallisation (3-D)
 * Input:  inputs/recrystallisation.ini
 * Output: output/recrystallisation/snap_NNNN.{vtk|csv|ppm}
 */
#include "gpk_core.hpp"
#include "gpk_thermo.hpp"
#include "gpk_io.hpp"

#include <cstdio>
#include <cmath>
#include <vector>
#include <random>
#include <chrono>
#include <omp.h>

using namespace gpk;

static void nucleate(GPKGrid& grid, int n_nuclei, int r_nuc,
                     std::mt19937& rng, int& next_gid) {
    const int Nz=grid.Nz, Ny=grid.Ny, Nx=grid.Nx;
    bool is2d = (Nz == 1);
    std::uniform_int_distribution<int> rz(is2d ? 0 : r_nuc,
                                          is2d ? 0 : (Nz - r_nuc - 1));
    std::uniform_int_distribution<int> ry(r_nuc,Ny-r_nuc-1);
    std::uniform_int_distribution<int> rx(r_nuc,Nx-r_nuc-1);
    std::uniform_real_distribution<float> rg(0.f,(float)M_PI*0.5f);

    for (int n = 0; n < n_nuclei; ++n) {
        int cz=rz(rng), cy=ry(rng), cx=rx(rng);
        float gval = rg(rng);
        int gid = ++next_gid;
        for (int dz=-r_nuc; dz<=r_nuc; ++dz)
        for (int dy=-r_nuc; dy<=r_nuc; ++dy)
        for (int dx_=-r_nuc; dx_<=r_nuc; ++dx_) {
            if (dz*dz+dy*dy+dx_*dx_ > r_nuc*r_nuc) continue;
            int z=cz+dz, y=cy+dy, x=cx+dx_;
            if (z<0||z>=Nz||y<0||y>=Ny||x<0||x>=Nx) continue;
            if (is2d && dz != 0) continue;
            int i = grid.idx(z,y,x);
            if (grid.phi[i] == 0) {
                grid.phi[i]   = gid;
                grid.g[i]     = gval;
                grid.alpha[i] = 1.f;
            }
        }
    }
}

static int stochastic_nucleation_step(GPKGrid& grid,
                                      float Jdot,
                                      float dt,
                                      int r_nuc,
                                      std::mt19937& rng,
                                      int& next_gid)
{
    if (Jdot <= 0.f || dt <= 0.f) return 0;

    const int Nz = grid.Nz, Ny = grid.Ny, Nx = grid.Nx;
    const float Vcell = grid.dx * grid.dx * grid.dx;
    const float P = 1.f - std::exp(-Jdot * Vcell * dt);
    if (P <= 0.f) return 0;

    std::uniform_real_distribution<float> u01(0.f, 1.f);
    std::uniform_real_distribution<float> rg(0.f, (float)M_PI * 0.5f);

    int added = 0;
    for (int z = r_nuc; z < Nz - r_nuc; ++z)
    for (int y = r_nuc; y < Ny - r_nuc; ++y)
    for (int x = r_nuc; x < Nx - r_nuc; ++x) {
        int i = grid.idx(z, y, x);
        if (grid.phi[i] != 0) continue;
        if (u01(rng) > P) continue;

        int gid = ++next_gid;
        float gval = rg(rng);
        for (int dz = -r_nuc; dz <= r_nuc; ++dz)
        for (int dy = -r_nuc; dy <= r_nuc; ++dy)
        for (int dx_ = -r_nuc; dx_ <= r_nuc; ++dx_) {
            if (dz*dz + dy*dy + dx_*dx_ > r_nuc*r_nuc) continue;
            int zz = z + dz, yy = y + dy, xx = x + dx_;
            int j = grid.idx(zz, yy, xx);
            if (grid.phi[j] == 0) {
                grid.phi[j] = gid;
                grid.g[j] = gval;
                grid.alpha[j] = 1.f;
            }
        }
        ++added;
    }

    if (Nz == 1) {
        const int z = 0;
        for (int y = r_nuc; y < Ny - r_nuc; ++y)
        for (int x = r_nuc; x < Nx - r_nuc; ++x) {
            int i = grid.idx(z, y, x);
            if (grid.phi[i] != 0) continue;
            if (u01(rng) > P) continue;

            int gid = ++next_gid;
            float gval = rg(rng);
            for (int dy = -r_nuc; dy <= r_nuc; ++dy)
            for (int dx_ = -r_nuc; dx_ <= r_nuc; ++dx_) {
                if (dy*dy + dx_*dx_ > r_nuc*r_nuc) continue;
                int yy = y + dy, xx = x + dx_;
                int j = grid.idx(0, yy, xx);
                if (grid.phi[j] == 0) {
                    grid.phi[j] = gid;
                    grid.g[j] = gval;
                    grid.alpha[j] = 1.f;
                }
            }
            ++added;
        }
        return added;
    }

    return added;
}

int main(int argc, char* argv[]) {
    const char* ini = (argc > 1) ? argv[1]
                                  : "inputs/recrystallisation.ini";
    InputParser cfg(ini);

    const int    Nz       = cfg.get_int   ("grid",  "Nz");
    const int    Ny       = cfg.get_int   ("grid",  "Ny");
    const int    Nx       = cfg.get_int   ("grid",  "Nx");
    const float  dx       = cfg.get_float ("grid",  "dx");
    const double t_end    = cfg.get_double("time",  "t_end");
    const int    n_snaps  = cfg.get_int   ("time",  "n_snaps");

    const float  M0       = cfg.get_float("thermo","M0");
    const float  sig0     = cfg.get_float("thermo","sigma0");
    const float  E_stored = cfg.get_float("thermo","E_stored");

    const int    N_NUC    = cfg.get_int  ("nucleation","n_nuclei");
    const int    R_NUC    = cfg.get_int  ("nucleation","nucleus_r");
    const float  Jdot     = cfg.get_float("nucleation","Jdot", 0.f);
    const int    seed     = cfg.get_int  ("nucleation","seed", 42);

    printf("=== GPK Recrystallisation (3-D) ===\n");
    printf("Grid %d×%d×%d, dx=%.1f µm\n", Nz,Ny,Nx, dx*1e6f);
    printf("E_stored = %.2e J/m³, OpenMP: %d threads\n\n",
           (double)E_stored, omp_get_max_threads());

    float R_star = sig0 / E_stored;
    printf("Critical R* = %.3f µm (nucleus radius = %d cells = %.1f µm)\n\n",
           R_star*1e6f, R_NUC, R_NUC*dx*1e6f);

    auto thermo = std::make_shared<RecrystallisationThermo>(M0, sig0, E_stored);
    GPKGrid grid(Nz, Ny, Nx, dx, thermo, 0.f);

    std::fill(grid.phi.begin(),   grid.phi.end(),   0);
    std::fill(grid.alpha.begin(), grid.alpha.end(), 1.f);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> rg_dist(0.f, (float)M_PI*0.5f);
    for (auto& v : grid.g) v = rg_dist(rng);

    int next_gid = 0;
    nucleate(grid, N_NUC, R_NUC, rng, next_gid);
        printf("Nucleated %d rex grains (radius %d cells), Jdot = %.3e 1/(m^3 s)\n\n",
            N_NUC, R_NUC, (double)Jdot);

    OutputWriter writer(cfg);
    double snap_dt = t_end / n_snaps;

    printf("%-10s %-12s %-14s %-10s\n",
           "Time[s]", "Rex frac", "ln(-ln(1-X))", "N_grains");

    auto t_wall0 = std::chrono::high_resolution_clock::now();

    for (int snap = 0; snap < n_snaps; ++snap) {
        double t_target = snap_dt * (snap + 1);
        while (grid.time < t_target) {
            float dt = grid.step(-1.f);
            stochastic_nucleation_step(grid, Jdot, dt, R_NUC, rng, next_gid);
        }

        int rex_cnt = 0;
        for (auto v : grid.phi) if (v != 0) ++rex_cnt;
        float X = (float)rex_cnt / (float)grid.N();
        float avrami = (X>0.f && X<1.f) ? std::log(-std::log(1.f-X))
                                         : (X>=1.f ? 99.f : -99.f);

        printf("%-10.4f %-12.4f %-14.3f %-10d\n",
               grid.time, X, avrami, grid.n_grains());

        writer.write(grid, snap, grid.time);
        if (X > 0.99f) { printf("Fully recrystallised.\n"); break; }
    }

    double wall_s = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - t_wall0).count();
    printf("\nWall time: %.2f s\n", wall_s);
    return 0;
}
