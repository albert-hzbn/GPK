/*
 * Benchmark — OpenMP thread & grid-size scaling (3-D)
 */
#include "gpk_core.hpp"
#include "gpk_thermo.hpp"

#include <cstdio>
#include <climits>
#include <string>
#include <chrono>
#include <omp.h>

using namespace gpk;

static void init_voronoi_fast(GPKGrid& grid, int n_seeds) {
    const int Nz=grid.Nz, Ny=grid.Ny, Nx=grid.Nx;
    std::vector<int> sz(n_seeds), sy(n_seeds), sx(n_seeds);
    srand(42);
    for (int k=0; k<n_seeds; ++k) {
        sz[k]=rand()%Nz; sy[k]=rand()%Ny; sx[k]=rand()%Nx;
    }
    for (int z=0; z<Nz; ++z)
    for (int y=0; y<Ny; ++y)
    for (int x=0; x<Nx; ++x) {
        long long best=LLONG_MAX; int b=0;
        for (int k=0; k<n_seeds; ++k) {
            long long dz=z-sz[k], dy_=y-sy[k], dx_=x-sx[k];
            long long d=dz*dz+dy_*dy_+dx_*dx_;
            if(d<best){best=d;b=k;}
        }
        int i=grid.idx(z,y,x);
        grid.phi[i]=b+1; grid.alpha[i]=1.f;
    }
}

static double bench(int Nz, int Ny, int Nx, int n_steps) {
    auto th = std::make_shared<IsotropicGrowthThermo>(1e-8f, 0.5f);
    GPKGrid grid(Nz, Ny, Nx, 1e-6f, th, 0.f);
    init_voronoi_fast(grid, 4);
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i=0; i<n_steps; ++i) grid.step(-1.f);
    double s = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now()-t0).count();
    return (double)Nz*Ny*Nx * n_steps / s;
}

int main() {
    const int N=32, STEPS=5, MAX_T=omp_get_max_threads();
    printf("=== GPK 3-D Parallelisation Benchmark ===\n\n");

    printf("--- Thread scaling: %d³ grid, %d steps ---\n", N, STEPS);
    printf("%-8s %-18s %-12s %-12s\n",
           "Threads","Cell-steps/s","Speedup","Efficiency");

    double base = 0.0;
    for (int t=1; t<=MAX_T; ++t) {
        omp_set_num_threads(t);
        double rate = bench(N,N,N,STEPS);
        if (t==1) base = rate;
        printf("%-8d %-18.3e %-12.2f %-12.1f%%\n",
               t, rate, rate/base, rate/base/t*100.0);
    }
    omp_set_num_threads(MAX_T);

    printf("\n--- Grid-size scaling: %d threads, %d steps ---\n", MAX_T, STEPS);
    printf("%-12s %-18s %-14s\n","Grid","Cell-steps/s","Wall[s]");
    for (int n : {8, 16, 32, 48, 64}) {
        auto t0 = std::chrono::high_resolution_clock::now();
        double rate = bench(n,n,n,STEPS);
        double wall = std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now()-t0).count();
        printf("%-12s %-18.3e %-14.3f\n",
               (std::to_string(n)+"³").c_str(), rate, wall);
    }
    return 0;
}
