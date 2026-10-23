/*
 * Demo 5 — Multi-phase precipitation
 * Input:  inputs/precipitation.ini
 * Output: output/precipitation/snap_NNNN.{vtk|csv|ppm}
 */
#include "gpk_core.hpp"
#include "gpk_thermo.hpp"
#include "gpk_io.hpp"

#include <cstdio>
#include <vector>
#include <string>
#include <sstream>
#include <random>
#include <unordered_map>
#include <algorithm>
#include <chrono>
#include <omp.h>

using namespace gpk;

static std::vector<float> parse_csv_floats(const std::string& s) {
    std::vector<float> out;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        if (!tok.empty()) out.push_back(std::stof(tok));
    }
    return out;
}

int main(int argc, char* argv[]) {
    const char* ini = (argc > 1) ? argv[1] : "inputs/precipitation.ini";
    InputParser cfg(ini);

    const int Nz = cfg.get_int("grid", "Nz");
    const int Ny = cfg.get_int("grid", "Ny");
    const int Nx = cfg.get_int("grid", "Nx");
    const float dx = cfg.get_float("grid", "dx");

    const double t_end = cfg.get_double("time", "t_end");
    const int nsnap = cfg.get_int("time", "n_snaps");

    const float M0 = cfg.get_float("thermo", "M0");
    const float sig0 = cfg.get_float("thermo", "sigma0");
    const float D = cfg.get_float("thermo", "D", 0.f);
    const int n_phases = cfg.get_int("thermo", "n_phases");
    const float T_ref = cfg.get_float("thermo", "T_ref", 900.f);

    std::vector<float> a = parse_csv_floats(cfg.get_str("thermo", "a"));
    std::vector<float> b = parse_csv_floats(cfg.get_str("thermo", "b"));
    std::vector<float> c2 = parse_csv_floats(cfg.get_str("thermo", "c2"));
    std::vector<float> dT = parse_csv_floats(cfg.get_str("thermo", "dT"));

    const float c0 = cfg.get_float("initial", "c0");
    const float T0 = cfg.get_float("initial", "T0");
    const int n_seeds = cfg.get_int("initial", "n_seeds");
    const int seed_r = cfg.get_int("initial", "seed_radius", 2);
    const int seed = cfg.get_int("initial", "seed", 123);

    printf("=== GPK Multi-phase Precipitation ===\n");
    printf("Grid %d x %d x %d, phases=%d, dx=%.2f um\n", Nz, Ny, Nx, n_phases, dx * 1e6f);
    printf("OpenMP threads: %d\n\n", omp_get_max_threads());

    auto thermo = std::make_shared<PrecipitationThermo>(M0, sig0, n_phases, a, b, c2, dT, T_ref);
    GPKGrid grid(Nz, Ny, Nx, dx, thermo, D);

    std::fill(grid.phi.begin(), grid.phi.end(), 0);
    std::fill(grid.alpha.begin(), grid.alpha.end(), 1.f);
    std::fill(grid.c.begin(), grid.c.end(), c0);
    std::fill(grid.T.begin(), grid.T.end(), T0);

    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> rz(seed_r, std::max(seed_r, Nz - seed_r - 1));
    std::uniform_int_distribution<int> ry(seed_r, std::max(seed_r, Ny - seed_r - 1));
    std::uniform_int_distribution<int> rx(seed_r, std::max(seed_r, Nx - seed_r - 1));
    std::uniform_int_distribution<int> rp(1, std::max(1, n_phases - 1));

    for (int n = 0; n < n_seeds; ++n) {
        int cz = rz(rng), cy = ry(rng), cx = rx(rng);
        int p = rp(rng);
        bool overlap = false;
        for (int dz = -seed_r - 1; dz <= seed_r + 1 && !overlap; ++dz)
        for (int dy = -seed_r - 1; dy <= seed_r + 1 && !overlap; ++dy)
        for (int dx_ = -seed_r - 1; dx_ <= seed_r + 1 && !overlap; ++dx_) {
            int zz = cz + dz, yy = cy + dy, xx = cx + dx_;
            if (zz < 0 || zz >= Nz || yy < 0 || yy >= Ny || xx < 0 || xx >= Nx) continue;
            int j = grid.idx(zz, yy, xx);
            if (grid.phi[j] != 0) overlap = true;
        }
        if (overlap) continue;

        for (int dz = -seed_r; dz <= seed_r; ++dz)
        for (int dy = -seed_r; dy <= seed_r; ++dy)
        for (int dx_ = -seed_r; dx_ <= seed_r; ++dx_) {
            if (dz*dz + dy*dy + dx_*dx_ > seed_r*seed_r) continue;
            int zz = cz + dz, yy = cy + dy, xx = cx + dx_;
            if (zz < 0 || zz >= Nz || yy < 0 || yy >= Ny || xx < 0 || xx >= Nx) continue;
            int i = grid.idx(zz, yy, xx);
            grid.phi[i] = p;
        }
    }

    OutputWriter writer(cfg);
    const double snap_dt = t_end / nsnap;

    auto t0 = std::chrono::high_resolution_clock::now();

    printf("%-10s %-12s %-12s %-12s\n", "Time[s]", "Phase0", "Phase1", "Phase2+");
    for (int snap = 0; snap < nsnap; ++snap) {
        grid.run(snap_dt * (snap + 1), -1.f);

        std::unordered_map<int32_t, int> cnt;
        for (auto p : grid.phi) cnt[p]++;
        float f0 = (float)cnt[0] / (float)grid.N();
        float f1 = (float)cnt[1] / (float)grid.N();
        float f2p = std::max(0.f, 1.f - f0 - f1);

        printf("%-10.4f %-12.4f %-12.4f %-12.4f\n", grid.time, f0, f1, f2p);
        writer.write(grid, snap, grid.time);
    }

    double wall_s = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
    printf("\nWall time: %.2f s\n", wall_s);
    return 0;
}
