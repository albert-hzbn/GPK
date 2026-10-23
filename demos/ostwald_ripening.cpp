/*
 * Demo 4 — Ostwald Ripening (3-D)
 * Input:  inputs/ostwald_ripening.ini
 * Output: output/ostwald_ripening/snap_NNNN.{vtk|csv|ppm}
 */
#include "gpk_core.hpp"
#include "gpk_thermo.hpp"
#include "gpk_io.hpp"

#include <cstdio>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <unordered_map>
#include <chrono>
#include <omp.h>

using namespace gpk;

static int place_particles(GPKGrid& grid, int n_try,
                            float r_mean_cells, float r_ln_sigma,
                            float c_part, float c_matrix, uint32_t seed) {
    const int Nz=grid.Nz, Ny=grid.Ny, Nx=grid.Nx;
    std::mt19937 rng(seed);
    std::lognormal_distribution<float> r_dist(std::log(r_mean_cells), r_ln_sigma);
    std::uniform_int_distribution<int> rz(0,Nz-1), ry(0,Ny-1), rx(0,Nx-1);

    std::fill(grid.phi.begin(),   grid.phi.end(),   0);
    std::fill(grid.alpha.begin(), grid.alpha.end(), 1.f);
    std::fill(grid.c.begin(),     grid.c.end(),     c_matrix);

    int placed = 0;
    for (int attempt = 0; attempt < n_try*50 && placed < n_try; ++attempt) {
        int cap = (Nz == 1) ? std::min(Ny,Nx)/4 : std::min({Nz,Ny,Nx})/4;
        int ri = std::max(1, (int)std::round(
            std::clamp(r_dist(rng), 1.5f, (float)cap)));
        int cz=rz(rng), cy=ry(rng), cx=rx(rng);

        // 2-D mode: Nz=1, skip z margin check
        bool z_ok = (Nz == 1) || (cz-ri>=1 && cz+ri<=Nz-2);
        if (!z_ok || cy-ri<1||cy+ri>=Ny-1||cx-ri<1||cx+ri>=Nx-1) continue;

        bool overlap = false;
        for (int dz=-(ri+1); dz<=ri+1 && !overlap; ++dz)
        for (int dy=-ri-1; dy<=ri+1 && !overlap; ++dy)
        for (int dx_=-ri-1; dx_<=ri+1 && !overlap; ++dx_) {
            int oz=cz+dz, oy=cy+dy, ox=cx+dx_;
            if (oz<0||oz>=Nz||oy<0||oy>=Ny||ox<0||ox>=Nx) continue;
            if (grid.phi[grid.idx(oz,oy,ox)] != 0) overlap=true;
        }
        if (overlap) continue;

        int gid = ++placed;
        for (int dz=-ri; dz<=ri; ++dz)
        for (int dy=-ri; dy<=ri; ++dy)
        for (int dx_=-ri; dx_<=ri; ++dx_) {
            if (dz*dz+dy*dy+dx_*dx_ > ri*ri) continue;
            int oz=cz+dz, oy=cy+dy, ox=cx+dx_;
            if (oz<0||oz>=Nz||oy<0||oy>=Ny||ox<0||ox>=Nx) continue;
            int i = grid.idx(oz, oy, ox);
            grid.phi[i]   = gid;
            grid.c[i]     = c_part;
            grid.alpha[i] = 1.f;
        }
    }
    return placed;
}

struct PStats { double mean, min_r, max_r; int count; };
static PStats particle_stats(const GPKGrid& grid) {
    std::unordered_map<int32_t,int> cnt;
    for (auto v : grid.phi) if (v != 0) cnt[v]++;
    if (cnt.empty()) return {0,0,0,0};
    double sum=0, rmin=1e30, rmax=0;
    for (auto& [k,a] : cnt) {
        double r = (grid.Nz>1)
                   ? std::cbrt(3.0*a/(4.0*M_PI))
                   : std::sqrt((double)a/M_PI);
        sum+=r; if(r<rmin) rmin=r; if(r>rmax) rmax=r;
    }
    float dx=grid.dx; int n=(int)cnt.size();
    return {sum/n*dx, rmin*dx, rmax*dx, n};
}

int main(int argc, char* argv[]) {
    const char* ini = (argc > 1) ? argv[1]
                                  : "inputs/ostwald_ripening.ini";
    InputParser cfg(ini);

    const int    Nz    = cfg.get_int   ("grid","Nz");
    const int    Ny    = cfg.get_int   ("grid","Ny");
    const int    Nx    = cfg.get_int   ("grid","Nx");
    const float  dx    = cfg.get_float ("grid","dx");
    const double t_end = cfg.get_double("time","t_end");
    const int    nsnap = cfg.get_int   ("time","n_snaps");

    const float M0     = cfg.get_float("thermo","M0");
    const float sig0   = cfg.get_float("thermo","sigma0");
    const float D      = cfg.get_float("thermo","D");
    const float c_sat  = cfg.get_float("thermo","c_sat", 0.15f);
    const float Omega  = cfg.get_float("thermo","Omega", 5e8f);

    const int    N_try    = cfg.get_int  ("particles","n_particles");
    const float  r_mean   = cfg.get_float("particles","r_mean_cells");
    const float  r_lns    = cfg.get_float("particles","r_lognormal_s");
    const float  c_part   = cfg.get_float("particles","c_particle");
    const float  c_matrix = cfg.get_float("particles","c_matrix");
    const int    seed     = cfg.get_int  ("particles","seed", 999);

    printf("=== GPK Ostwald Ripening (3-D) ===\n");
    printf("Grid %d×%d×%d, dx=%.2f µm, D=%.1e m²/s\n",
           Nz,Ny,Nx, dx*1e6f, (double)D);
    printf("OpenMP threads: %d\n\n", omp_get_max_threads());

    auto thermo = std::make_shared<OstwaldThermo>(M0, sig0, c_sat, Omega);
    GPKGrid grid(Nz, Ny, Nx, dx, thermo, D);

    int placed = place_particles(grid, N_try, r_mean, r_lns, c_part, c_matrix, seed);
    auto s0 = particle_stats(grid);
    printf("Placed %d particles; R_mean=%.2f µm, R_min=%.2f µm, R_max=%.2f µm\n\n",
           placed, s0.mean*1e6f, s0.min_r*1e6f, s0.max_r*1e6f);

    OutputWriter writer(cfg);
    double snap_dt = t_end / nsnap;

    printf("%-10s %-10s %-12s %-12s %-12s\n",
           "Time[ms]","N_part","R_mean[µm]","R_min[µm]","R_max[µm]");

    auto t_wall0 = std::chrono::high_resolution_clock::now();
    int total_steps = 0;

    for (int snap = 0; snap < nsnap; ++snap) {
        int steps = grid.run(snap_dt*(snap+1), -1.f);
        total_steps += steps;
        auto s = particle_stats(grid);
        printf("%-10.2f %-10d %-12.3f %-12.3f %-12.3f\n",
               grid.time*1e3, s.count,
               s.mean*1e6f, s.min_r*1e6f, s.max_r*1e6f);

        writer.write(grid, snap, grid.time);
        if (s.count == 0) { printf("All particles dissolved.\n"); break; }
    }

    double wall_s = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - t_wall0).count();
    printf("\nWall time: %.2f s  (%d steps)\n", wall_s, total_steps);
    return 0;
}
