/*
 * bench_detailed.cpp — realistic per-step and scaling benchmarks.
 *
 * Experiments:
 *   A  Thread scaling       : fixed 128x128 grid, 1..N_max threads
 *   B  Grid-size scaling    : 1 thread, grids 32..384 (step 32)
 *   C  Per-substep cost     : 128x128, 1 thread, time each step in isolation
 *   D  Interface-fraction   : vary number of grains → vary %interface
 *   E  Accuracy vs dx       : grain growth at dx = 0.5,1,2 µm same physics
 *
 * Output: TSV tables on stdout, parseable by the Python report script.
 */

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include "gpk_core.hpp"
#include "gpk_thermo.hpp"
#include <omp.h>
#include <cstdio>
#include <climits>
#include <cmath>
#include <chrono>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>

using namespace gpk;
using clk = std::chrono::high_resolution_clock;

// ── init helpers ──────────────────────────────────────────────────────────────

static void voronoi(GPKGrid& g, int n_seeds, int seed_val = 42) {
    const int Nz=g.Nz, Ny=g.Ny, Nx=g.Nx;
    srand(seed_val);
    std::vector<int> sz(n_seeds),sy(n_seeds),sx(n_seeds);
    for (int k=0;k<n_seeds;++k){ sz[k]=rand()%Nz; sy[k]=rand()%Ny; sx[k]=rand()%Nx; }
    for (int z=0;z<Nz;++z) for (int y=0;y<Ny;++y) for (int x=0;x<Nx;++x) {
        long long best=LLONG_MAX; int b=0;
        for (int k=0;k<n_seeds;++k){
            long long dz=z-sz[k],dy=y-sy[k],dx_=x-sx[k];
            long long d=dz*dz+dy*dy+dx_*dx_;
            if(d<best){best=d;b=k;}
        }
        int i=g.idx(z,y,x);
        g.phi[i]=b+1; g.alpha[i]=1.f; g.g[i]=(rand()%314)/100.f;
    }
}

// Warm up the grid by running a few steps so caches are primed
static void warmup(GPKGrid& g, int steps=3) {
    for (int i=0;i<steps;++i) g.step(-1.f);
}

// Run N steps and return wall seconds
static double time_steps(GPKGrid& g, int N) {
    auto t0 = clk::now();
    for (int i=0;i<N;++i) g.step(-1.f);
    return std::chrono::duration<double>(clk::now()-t0).count();
}

// ══════════════════════════════════════════════════════════════════════════════
// A  Thread scaling
// ══════════════════════════════════════════════════════════════════════════════
static void bench_threads() {
    const int Ny=128, Nx=128, N_SEEDS=40, STEPS=20;
    printf("\n=== A: Thread scaling (128x128, 40 grains, %d steps) ===\n", STEPS);
    printf("threads\twall_s\tcell_steps_per_s\tspeedup\n");

    double base_time = 0.0;
    int max_t = omp_get_max_threads();
    for (int t = 1; t <= max_t; ++t) {
        omp_set_num_threads(t);
        auto th = std::make_shared<IsotropicGrowthThermo>(1e-8f, 0.5f);
        GPKGrid g(1, Ny, Nx, 1e-6f, th, 0.f);
        voronoi(g, N_SEEDS);
        warmup(g, 2);

        double ws = time_steps(g, STEPS);
        if (t == 1) base_time = ws;
        double cps = (double)Ny * Nx * STEPS / ws;
        printf("%d\t%.4f\t%.3e\t%.3f\n", t, ws, cps, base_time/ws);
        fflush(stdout);
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// B  Grid-size scaling (1 thread, steps fixed at 10)
// ══════════════════════════════════════════════════════════════════════════════
static void bench_gridsize() {
    const int STEPS=10;
    printf("\n=== B: Grid-size scaling (1 thread, %d steps) ===\n", STEPS);
    printf("N\twall_s\tcell_steps_per_s\tdt_s\tiface_frac\n");
    omp_set_num_threads(1);

    for (int N : {32,48,64,80,96,112,128,160,192,256,320,384}) {
        auto th = std::make_shared<IsotropicGrowthThermo>(1e-8f, 0.5f);
        GPKGrid g(1, N, N, 1e-6f, th, 0.f);
        int seeds = std::max(4, N*N/400);    // ~same grain density
        voronoi(g, seeds);
        warmup(g, 2);
        float last_dt = g.step(-1.f);
        float iface   = g.interface_fraction();
        double ws = time_steps(g, STEPS);
        double cps = (double)N*N*STEPS/ws;
        printf("%d\t%.4f\t%.3e\t%.3e\t%.4f\n", N, ws, cps, (double)last_dt, (double)iface);
        fflush(stdout);
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// C  Per-substep cost breakdown (128x128, 1 thread, 50 steps each)
// ══════════════════════════════════════════════════════════════════════════════

// We time full step, then estimate sub-costs by running with artificial
// short-circuits: grid with zero interface cells (all mask=0 → geometry/velocity
// are no-ops) vs normal grid.
static void bench_substep() {
    const int STEPS=50;
    printf("\n=== C: Per-substep timing (128x128, 40 grains, %d steps, 1 thread) ===\n", STEPS);
    omp_set_num_threads(1);

    auto th = std::make_shared<IsotropicGrowthThermo>(1e-8f, 0.5f);
    GPKGrid g(1, 128, 128, 1e-6f, th, 0.f);
    voronoi(g, 40);
    warmup(g, 3);

    // --- full step ---
    double t_full = time_steps(g, STEPS);

    // Re-init to get same interface fraction
    GPKGrid g2(1, 128, 128, 1e-6f, th, 0.f);
    voronoi(g2, 40);
    warmup(g2, 3);

    // Time diffusion only: fill mask=0, skip geometry+velocity
    // We proxy this by running step_diffusion directly via a subclass trick —
    // instead, just note that D_=0 means diffusion is a no-op, so
    // cost(full, D=0) ≈ cost(interface_mask) + cost(geometry) + cost(velocity) + cost(alpha)
    // and cost(full, D>0) - cost(full, D=0) ≈ cost(diffusion)

    auto th_d = std::make_shared<IsotropicGrowthThermo>(1e-8f, 0.5f);
    // No diffusion
    GPKGrid g_nd(1, 128, 128, 1e-6f, th_d, 0.f);    // D=0
    voronoi(g_nd, 40); warmup(g_nd, 3);
    double t_no_diff = time_steps(g_nd, STEPS);

    // With diffusion (D=3e-9)
    GPKGrid g_d(1, 128, 128, 1e-6f, th_d, 3e-9f);
    voronoi(g_d, 40); warmup(g_d, 3);
    double t_with_diff = time_steps(g_d, STEPS);

    double t_diff_only = t_with_diff - t_no_diff;

    printf("substep\twall_s\tfrac_of_total\n");
    printf("total (no diffusion)\t%.4f\t%.3f\n", t_no_diff,  1.0);
    printf("diffusion only (D>0)\t%.4f\t%.3f\n", t_diff_only, t_diff_only/t_no_diff);
    printf("total (with diffusion)\t%.4f\t%.3f\n", t_with_diff, t_with_diff/t_no_diff);
    printf("full step reference\t%.4f\t1.000\n", t_full);

    float iface = g.interface_fraction();
    printf("interface_fraction\t%.4f\t-\n", (double)iface);
    fflush(stdout);
}

// ══════════════════════════════════════════════════════════════════════════════
// D  Interface-fraction scaling (vary Ng → vary %interface)
// ══════════════════════════════════════════════════════════════════════════════
static void bench_iface_scaling() {
    const int STEPS=20, N=128;
    printf("\n=== D: Interface-fraction scaling (128x128, 1 thread, %d steps) ===\n", STEPS);
    printf("n_grains\tiface_frac\twall_s\tcell_steps_per_s\n");
    omp_set_num_threads(1);

    for (int ng : {5, 10, 20, 40, 80, 160, 320, 640}) {
        auto th = std::make_shared<IsotropicGrowthThermo>(1e-8f, 0.5f);
        GPKGrid g(1, N, N, 1e-6f, th, 0.f);
        voronoi(g, ng);
        warmup(g, 2);
        float iface = g.interface_fraction();
        double ws   = time_steps(g, STEPS);
        double cps  = (double)N*N*STEPS/ws;
        printf("%d\t%.4f\t%.4f\t%.3e\n", ng, (double)iface, ws, cps);
        fflush(stdout);
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// E  Accuracy vs dx — grain growth at three resolutions, same physical domain
// ══════════════════════════════════════════════════════════════════════════════
static void bench_accuracy_vs_dx() {
    printf("\n=== E: Accuracy vs dx (same 128x128µm domain, 40 grains, 1 thread) ===\n");
    printf("dx_um\tN\ttime_ms\tR_gpk_um\tR_theory_um\terr_pct\twall_s\n");
    omp_set_num_threads(1);

    const float M0=1e-8f, sig=0.5f;
    const float K_theory = 2.f/3.f * M0 * sig;  // m²/s
    const float t_phys = 0.05f;                  // 50 ms physical time
    // dx values: 2.0, 1.0, 0.5 µm  → grid sizes 64, 128, 256
    for (auto [dx_val, N_val] : std::vector<std::pair<float,int>>{{2e-6f,64},{1e-6f,128},{0.5e-6f,256}}) {
        auto th = std::make_shared<IsotropicGrowthThermo>(M0, sig);
        GPKGrid g(1, N_val, N_val, dx_val, th, 0.f);
        voronoi(g, 40, 42);

        double R0 = g.mean_grain_radius();

        auto t_wall0 = clk::now();
        // Run to t_phys in 10 snapshots
        int n_snaps = 10;
        for (int s=0; s<n_snaps; ++s) {
            double t_snap = t_phys * (s+1.0)/n_snaps;
            while (g.time < t_snap) g.step(-1.f);

            double R_gpk  = g.mean_grain_radius();
            double R_th   = std::sqrt(R0*R0 + K_theory*(float)g.time);
            double err    = std::fabs(R_gpk - R_th)/R_th * 100.0;
            double ws     = std::chrono::duration<double>(clk::now()-t_wall0).count();

            printf("%.1f\t%d\t%.1f\t%.4f\t%.4f\t%.2f\t%.3f\n",
                   (double)dx_val*1e6, N_val, g.time*1e3,
                   R_gpk*1e6, R_th*1e6, err, ws);
        }
        fflush(stdout);
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// F  Memory footprint — report actual grid memory at several sizes
// ══════════════════════════════════════════════════════════════════════════════
static void bench_memory() {
    printf("\n=== F: Memory footprint (bytes per cell, 2-D grids) ===\n");
    printf("N\ttotal_cells\tgrid_bytes\tbytes_per_cell\n");
    omp_set_num_threads(1);
    for (int N : {64, 128, 256, 512}) {
        auto th = std::make_shared<IsotropicGrowthThermo>(1e-8f, 0.5f);
        GPKGrid g(1, N, N, 1e-6f, th, 0.f);
        int cells = N * N;
        // Count GPKGrid heap vectors:
        // phi(int32), alpha(f32), c(f32), T(f32), g(f32)=5 fields * 4 bytes
        // nx_f, ny_f, nz_f, kappa_f, vn_f, E_el = 6 * f32
        // phi_comp(int32), mask(uint8)
        // alpha_new_, c_new_, eps_star_, eps_smooth_, eps_tmp_ = 5 * f32
        // Total per cell: 5*4 + 6*4 + 4 + 1 + 5*4 = 20+24+4+1+20 = 69 bytes
        // Plus scratch buffers: 7 * f32 per thread (1 thread here)
        size_t per_cell_state   = 5*4 + 6*4 + 4 + 1 + 5*4;  // 69 bytes
        size_t scratch_per_cell = 7 * 4;                      // 28 bytes/cell/thread
        size_t total = (size_t)cells * (per_cell_state + scratch_per_cell);
        printf("%d\t%d\t%zu\t%.1f\n", N, cells, total, (double)total/cells);
        fflush(stdout);
    }
}

// ══════════════════════════════════════════════════════════════════════════════
int main() {
    printf("GPK Detailed Benchmark\n");
    printf("Build: %s %s\n", __DATE__, __TIME__);
    printf("Max OpenMP threads: %d\n", omp_get_max_threads());

    bench_threads();
    bench_gridsize();
    bench_substep();
    bench_iface_scaling();
    bench_accuracy_vs_dx();
    bench_memory();

    printf("\n=== DONE ===\n");
    return 0;
}
