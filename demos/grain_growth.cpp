/*
 * Demo 1 — Polycrystal Grain Growth (3-D)
 * Input:  inputs/grain_growth.ini
 * Output: output/grain_growth/snap_NNNN.{vtk|csv|ppm}
 *
 * Validation: von Neumann-Mullins parabolic law  <R>^2 = <R0>^2 + K*t
 */
#include "gpk_core.hpp"
#include "gpk_thermo.hpp"
#include "gpk_io.hpp"

#include <cstdio>
#include <cmath>
#include <vector>
#include <random>
#include <climits>
#include <chrono>
#include <omp.h>

using namespace gpk;

static void voronoi_init(GPKGrid& grid, int n_seeds, uint32_t seed) {
    const int Nz = grid.Nz, Ny = grid.Ny, Nx = grid.Nx;
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> rz(0,Nz-1), ry(0,Ny-1), rx(0,Nx-1);
    std::uniform_real_distribution<float> rg(0.f, (float)M_PI * 0.5f);

    std::vector<int> sz(n_seeds), sy(n_seeds), sx(n_seeds);
    std::vector<float> sg(n_seeds);
    for (int k = 0; k < n_seeds; ++k) {
        sz[k]=rz(rng); sy[k]=ry(rng); sx[k]=rx(rng); sg[k]=rg(rng);
    }

    #pragma omp parallel for schedule(static)
    for (int z = 0; z < Nz; ++z)
    for (int y = 0; y < Ny; ++y)
    for (int x = 0; x < Nx; ++x) {
        long long best_d = LLONG_MAX; int best = 0;
        for (int k = 0; k < n_seeds; ++k) {
            long long dz=z-sz[k], dy=y-sy[k], dx_=x-sx[k];
            long long d = dz*dz + dy*dy + dx_*dx_;
            if (d < best_d) { best_d=d; best=k; }
        }
        int i = grid.idx(z,y,x);
        grid.phi[i]   = best + 1;
        grid.g[i]     = sg[best];
        grid.alpha[i] = 1.f;
    }
}

int main(int argc, char* argv[]) {
    const char* ini = (argc > 1) ? argv[1]
                                  : "inputs/grain_growth.ini";
    InputParser cfg(ini);

    const int   Nz       = cfg.get_int   ("grid",  "Nz");
    const int   Ny       = cfg.get_int   ("grid",  "Ny");
    const int   Nx       = cfg.get_int   ("grid",  "Nx");
    const float dx       = cfg.get_float ("grid",  "dx");
    const double t_end   = cfg.get_double("time",  "t_end");
    const int   n_snaps  = cfg.get_int   ("time",  "n_snaps");
    const float M0       = cfg.get_float ("thermo","M0");
    const float sigma0   = cfg.get_float ("thermo","sigma0");
    const int   N_GRAINS = cfg.get_int   ("grain_growth","n_grains");
    const int   seed     = cfg.get_int   ("grain_growth","seed", 42);

    const float K_theory = 2.f / 3.f * M0 * sigma0;

    printf("=== GPK Grain Growth (3-D) ===\n");
    printf("Grid %d×%d×%d, dx=%.1f µm, %d grains\n", Nz,Ny,Nx, dx*1e6f, N_GRAINS);
    printf("OpenMP threads: %d\n", omp_get_max_threads());
    printf("Theory K = 2Mσ/3 = %.3e m²/s\n\n", (double)K_theory);

    auto thermo = std::make_shared<IsotropicGrowthThermo>(M0, sigma0);
    GPKGrid grid(Nz, Ny, Nx, dx, thermo, 0.f);
    voronoi_init(grid, N_GRAINS, (uint32_t)seed);

    double R0 = grid.mean_grain_radius();
    OutputWriter writer(cfg);

    printf("%-12s %-14s %-14s %-10s %-10s\n",
           "Time [s]", "R_GPK [µm]", "R_th [µm]", "Err [%]", "Grains");

    auto t_wall0 = std::chrono::high_resolution_clock::now();

    for (int snap = 0; snap < n_snaps; ++snap) {
        double t_snap = t_end * (snap + 1.0) / n_snaps;
        grid.run(t_snap, -1.f);

        if (grid.n_grains() <= 1) break;  // fully coarsened

        double R    = grid.mean_grain_radius();
        double R_th = std::sqrt(R0*R0 + K_theory * grid.time);
        double err  = std::fabs(R - R_th) / R_th * 100.0;

        printf("%-12.4f %-14.4f %-14.4f %-10.2f %-10d\n",
               grid.time, R*1e6, R_th*1e6, err, grid.n_grains());

        writer.write(grid, snap, grid.time);
    }

    double wall_s = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - t_wall0).count();
    printf("\nWall time: %.2f s\n", wall_s);
    return 0;
}
