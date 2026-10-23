/*
 * Benchmark: GPK (C++ core) vs. Phase-Field (Fan-Chen) — 2-D grain growth.
 * =======================================================================
 * Both engines are C++, OpenMP, compiled with identical flags, so the
 * comparison is a fair like-for-like.  Three parts:
 *
 *   PART A  Matched-grid throughput.  Same N x N grid, fixed step count.
 *           Reports wall-time and cell-steps/s for each engine.
 *
 *   PART B  Equal-physics cost for a fixed physical domain.  Phase field must
 *           resolve the diffuse interface (dx_PF <= xi/m_cells); GPK does not,
 *           so for the same domain PF needs rho^d more cells and rho^2 more
 *           timesteps (rho = dx_GPK/dx_PF).  The per-cell-step constants come
 *           from the Part-A measurement; only the rho^(d+2) scaling is analytic.
 *
 *   PART C  Parabolic-kinetics correctness.  Both must reproduce
 *           <R>^2 = a + K t (von Neumann-Mullins), so neither is "fast" by
 *           skipping the physics.  GPK uses its mean_grain_radius(); PF uses a
 *           boundary-length proxy R ~ (cells)/(boundary cells).
 *
 * Build (from repo root, reusing the existing core lib):
 *   g++ -O3 -march=native -DNDEBUG -D_USE_MATH_DEFINES -fopenmp -std=c++17 \
 *       -Iinclude demos/bench_gpk_vs_pf.cpp build/libgpk_core.a -o build/bench_gpk_vs_pf
 */
#include "gpk_core.hpp"
#include "gpk_thermo.hpp"

#include <cstdio>
#include <cstdint>
#include <climits>
#include <cmath>
#include <vector>
#include <random>
#include <chrono>
#include <string>
#include <omp.h>

using namespace gpk;
using clk = std::chrono::high_resolution_clock;

// ---------------------------------------------------------------------------
// Shared Voronoi initial condition (periodic), returns labels 0..n_grains-1
// ---------------------------------------------------------------------------
static std::vector<int> voronoi_labels(int N, int n_grains, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> r(0, N - 1);
    std::vector<int> sy(n_grains), sx(n_grains);
    for (int k = 0; k < n_grains; ++k) { sy[k] = r(rng); sx[k] = r(rng); }
    std::vector<int> lab(N * N, 0);
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < N; ++y)
        for (int x = 0; x < N; ++x) {
            long long best = LLONG_MAX; int b = 0;
            for (int k = 0; k < n_grains; ++k) {
                long long dy = std::llabs((long long)y - sy[k]); dy = std::min(dy, (long long)N - dy);
                long long dx = std::llabs((long long)x - sx[k]); dx = std::min(dx, (long long)N - dx);
                long long d = dy * dy + dx * dx;
                if (d < best) { best = d; b = k; }
            }
            lab[y * N + x] = b;
        }
    return lab;
}

static long long boundary_count(const std::vector<int>& p, int N) {
    long long b = 0;
    #pragma omp parallel for reduction(+:b) schedule(static)
    for (int y = 0; y < N; ++y)
        for (int x = 0; x < N; ++x) {
            int c = p[y * N + x];
            int up = p[((y + N - 1) % N) * N + x];
            int dn = p[((y + 1) % N) * N + x];
            int lf = p[y * N + (x + N - 1) % N];
            int rt = p[y * N + (x + 1) % N];
            if (c != up || c != dn || c != lf || c != rt) b++;
        }
    return b;
}

// ---------------------------------------------------------------------------
// GPK 2-D grain growth (Nz = 1), isotropic capillarity
// ---------------------------------------------------------------------------
static GPKGrid make_gpk(const std::vector<int>& lab, int N, int n_grains,
                        std::shared_ptr<ThermoModel> th) {
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> rg(0.f, (float)M_PI * 0.5f);
    std::vector<float> go(n_grains);
    for (int k = 0; k < n_grains; ++k) go[k] = rg(rng);
    GPKGrid grid(1, N, N, 1e-6f, th, 0.f);
    for (int i = 0; i < N * N; ++i) {
        grid.phi[i]   = lab[i] + 1;
        grid.g[i]     = go[lab[i]];
        grid.alpha[i] = 1.f;
    }
    return grid;
}

static std::vector<int> gpk_labels(const GPKGrid& g, int N) {
    std::vector<int> p(N * N);
    for (int i = 0; i < N * N; ++i) p[i] = g.phi[i];
    return p;
}

// ---------------------------------------------------------------------------
// Phase-Field reference: Fan-Chen multi-order-parameter grain growth
//   f = sum_i (-eta_i^2/2 + eta_i^4/4) + gamma sum_{i<j} eta_i^2 eta_j^2
//   d eta_i/dt = -L ( -eta_i + eta_i^3 + 2 gamma eta_i (S - eta_i^2)
//                     - kappa lap(eta_i) ),  S = sum_j eta_j^2
// ---------------------------------------------------------------------------
struct PhaseField {
    int N, Q;
    float kappa, L, gamma, dx, dt;
    std::vector<std::vector<float>> eta, eta_new;

    PhaseField(const std::vector<int>& lab, int N_, int Q_,
               float kappa_ = 2.f, float L_ = 1.f, float gamma_ = 1.5f, float dx_ = 1.f)
        : N(N_), Q(Q_), kappa(kappa_), L(L_), gamma(gamma_), dx(dx_) {
        dt = 0.20f * dx * dx / (kappa * L);
        eta.assign(Q, std::vector<float>(N * N, 0.f));
        eta_new = eta;
        for (int i = 0; i < N * N; ++i) eta[lab[i] % Q][i] = 1.f;
    }

    void step() {
        const int NN = N * N;
        std::vector<float> S(NN, 0.f);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < NN; ++i) {
            float s = 0.f;
            for (int q = 0; q < Q; ++q) { float e = eta[q][i]; s += e * e; }
            S[i] = s;
        }
        for (int q = 0; q < Q; ++q) {
            const std::vector<float>& e = eta[q];
            std::vector<float>& en = eta_new[q];
            #pragma omp parallel for schedule(static)
            for (int y = 0; y < N; ++y)
                for (int x = 0; x < N; ++x) {
                    int i = y * N + x;
                    float c  = e[i];
                    float up = e[((y + N - 1) % N) * N + x];
                    float dn = e[((y + 1) % N) * N + x];
                    float lf = e[y * N + (x + N - 1) % N];
                    float rt = e[y * N + (x + 1) % N];
                    float lap = (up + dn + lf + rt - 4.f * c) / (dx * dx);
                    float df  = -c + c * c * c + 2.f * gamma * c * (S[i] - c * c);
                    en[i] = c - dt * L * (df - kappa * lap);
                }
        }
        eta.swap(eta_new);
    }

    std::vector<int> grain_field() const {
        std::vector<int> g(N * N, 0);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < N * N; ++i) {
            float best = -1e30f; int bq = 0;
            for (int q = 0; q < Q; ++q) if (eta[q][i] > best) { best = eta[q][i]; bq = q; }
            g[i] = bq;
        }
        return g;
    }
};

// ---------------------------------------------------------------------------
// Cellular Automaton reference: deterministic curvature-driven grain growth.
//   Moore (8) neighbourhood; a cell flips to the majority neighbour label when
//   >=5 of 8 neighbours share it.  This is the standard CA caricature of
//   curvature flow: convex protrusions get out-voted and shrink.  One integer
//   label field, ~8 reads + a small tally per cell -> cheapest per cell.
// ---------------------------------------------------------------------------
struct CellularAutomaton {
    int N;
    std::vector<int> phi, phi_new;
    CellularAutomaton(const std::vector<int>& lab, int N_) : N(N_), phi(lab), phi_new(lab) {}

    void step() {
        #pragma omp parallel for schedule(static)
        for (int y = 0; y < N; ++y)
            for (int x = 0; x < N; ++x) {
                int i = y * N + x;
                int ym = (y + N - 1) % N, yp = (y + 1) % N;
                int xm = (x + N - 1) % N, xp = (x + 1) % N;
                int nb[8] = {
                    phi[ym * N + xm], phi[ym * N + x], phi[ym * N + xp],
                    phi[y  * N + xm],                  phi[y  * N + xp],
                    phi[yp * N + xm], phi[yp * N + x], phi[yp * N + xp]
                };
                int bestlab = phi[i], bestc = 0;
                for (int a = 0; a < 8; ++a) {
                    int c = 1;
                    for (int b = a + 1; b < 8; ++b) if (nb[b] == nb[a]) ++c;
                    if (c > bestc) { bestc = c; bestlab = nb[a]; }
                }
                phi_new[i] = (bestc >= 5) ? bestlab : phi[i];
            }
        phi.swap(phi_new);
    }
};

// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    const int MAXT = omp_get_max_threads();
    int STEPS   = 20;
    int NGRAINS = 40;   // GPK polycrystal grain count
    int Q       = 20;   // PF order parameters
    if (argc > 1) STEPS = std::atoi(argv[1]);

    std::vector<int> sizes = {64, 128, 192, 256, 384, 512};

    printf("# GPK (C++ core) vs Phase-Field — 2-D Grain-Growth Benchmark\n");
    printf("OpenMP threads: %d | timing steps: %d | GPK n_grains=%d | PF Q=%d\n\n",
           MAXT, STEPS, NGRAINS, Q);

    // ---------------- PART A : matched-grid throughput ----------------------
    printf("## PART A - Matched-grid raw throughput (cell-steps/s)\n\n");
    printf("| Grid | GPK | PF | CA | GPK/PF | GPK/CA |\n");
    printf("|---|---|---|---|---|---|\n");

    FILE* fa = std::fopen("build/bench_partA.csv", "w");
    if (fa) std::fprintf(fa, "N,gpk_wall,gpk_rate,pf_wall,pf_rate,ca_wall,ca_rate\n");

    double gpk_cost = 0, pf_cost = 0, ca_cost = 0;  // per-cell-step s at largest grid
    for (int N : sizes) {
        auto lab = voronoi_labels(N, NGRAINS, 42);

        auto th = std::make_shared<IsotropicGrowthThermo>(1e-8f, 0.5f);
        GPKGrid grid = make_gpk(lab, N, NGRAINS, th);
        grid.step(-1.f);                              // warm-up
        auto t0 = clk::now();
        for (int s = 0; s < STEPS; ++s) grid.step(-1.f);
        double gw = std::chrono::duration<double>(clk::now() - t0).count();
        double gr = (double)N * N * STEPS / gw;

        PhaseField pf(lab, N, Q);
        pf.step();                                    // warm-up
        t0 = clk::now();
        for (int s = 0; s < STEPS; ++s) pf.step();
        double pw = std::chrono::duration<double>(clk::now() - t0).count();
        double pr = (double)N * N * STEPS / pw;

        CellularAutomaton ca(lab, N);
        ca.step();                                    // warm-up
        t0 = clk::now();
        for (int s = 0; s < STEPS; ++s) ca.step();
        double cw = std::chrono::duration<double>(clk::now() - t0).count();
        double cr = (double)N * N * STEPS / cw;

        printf("| %d^2 | %.2e | %.2e | %.2e | %.2fx | %.2fx |\n",
               N, gr, pr, cr, gr / pr, gr / cr);
        if (fa) std::fprintf(fa, "%d,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e\n",
                             N, gw, gr, pw, pr, cw, cr);
        gpk_cost = 1.0 / gr;
        pf_cost  = 1.0 / pr;
        ca_cost  = 1.0 / cr;
    }
    if (fa) std::fclose(fa);
    printf("\nPer-cell-step cost (largest grid): GPK=%.3e s, PF=%.3e s, CA=%.3e s\n",
           gpk_cost, pf_cost, ca_cost);
    printf("=> GPK is %.1fx the per-cell cost of CA; PF is %.2fx the per-cell cost of GPK.\n",
           gpk_cost / ca_cost, pf_cost / gpk_cost);

    // ---------------- PART B : equal-physics projection ---------------------
    printf("\n## PART B - Equal-physics cost, fixed physical domain (d=2)\n\n");
    printf("Per-cell-step cost (largest grid): GPK=%.3e s, PF=%.3e s (PF/GPK const=%.2fx)\n\n",
           gpk_cost, pf_cost, pf_cost / gpk_cost);
    printf("| rho=dx_GPK/dx_PF | PF cells x | PF steps x | cell-steps PF/GPK (rho^4) | wall PF/GPK |\n");
    printf("|---|---|---|---|---|\n");
    for (int rho : {5, 10, 20, 50, 100}) {
        double spatial = (double)rho * rho;
        double temporal = (double)rho * rho;
        double cs = spatial * temporal;
        double wall = cs * (pf_cost / gpk_cost);
        printf("| %dx | %.0fx | %.0fx | %.3e x | %.3e x |\n",
               rho, spatial, temporal, cs, wall);
    }

    // ---------------- PART C : parabolic kinetics ---------------------------
    printf("\n## PART C - Parabolic-kinetics correctness check\n\n");
    {
        int N = 256, ng = 120;   // larger initial grains (~13-cell R0) -> curvature well resolved
        auto lab = voronoi_labels(N, ng, 123);

        // GPK: mean grain radius over physical time
        auto th = std::make_shared<IsotropicGrowthThermo>(1e-8f, 0.5f);
        GPKGrid grid = make_gpk(lab, N, ng, th);
        double R0 = grid.mean_grain_radius();
        double K_th = 2.0 / 3.0 * 1e-8 * 0.5;
        std::vector<double> gt, gR2;
        int n_snaps = 12;
        double t_end = 0.10;
        FILE* fc = std::fopen("build/bench_partC_gpk.csv", "w");
        if (fc) std::fprintf(fc, "t,R_gpk,R_theory,grains\n");
        printf("GPK: K_theory = 2Mσ/3 = %.3e m^2/s,  R0 = %.3f um\n", K_th, R0 * 1e6);
        printf("| t [s] | R_GPK [um] | R_theory [um] | err %% | grains |\n");
        printf("|---|---|---|---|---|\n");
        for (int s = 1; s <= n_snaps; ++s) {
            double ts = t_end * s / n_snaps;
            grid.run(ts, -1.f);
            if (grid.n_grains() <= 10) break;   // stop before finite-N regime
            double R = grid.mean_grain_radius();
            double Rth = std::sqrt(R0 * R0 + K_th * grid.time);
            gt.push_back(grid.time); gR2.push_back(R * R);
            if (fc) std::fprintf(fc, "%.6e,%.6e,%.6e,%d\n", grid.time, R, Rth, grid.n_grains());
            printf("| %.4f | %.3f | %.3f | %.2f | %d |\n",
                   grid.time, R * 1e6, Rth * 1e6,
                   std::fabs(R - Rth) / Rth * 100.0, grid.n_grains());
        }
        if (fc) std::fclose(fc);
        // linear fit R^2 = a + K t, report R^2 of fit
        int n = (int)gt.size();
        double sx = 0, sy = 0, sxx = 0, sxy = 0, syy = 0;
        for (int i = 0; i < n; ++i) {
            sx += gt[i]; sy += gR2[i]; sxx += gt[i] * gt[i];
            sxy += gt[i] * gR2[i]; syy += gR2[i] * gR2[i];
        }
        double denom = (n * sxx - sx * sx) * (n * syy - sy * sy);
        double r2 = denom > 0 ? std::pow(n * sxy - sx * sy, 2) / denom : 0.0;
        double Kfit = (n * sxy - sx * sy) / (n * sxx - sx * sx);
        printf("\nGPK fit: K_measured = %.3e m^2/s (theory %.3e), linearity R^2 = %.4f\n",
               Kfit, K_th, r2);

        // PF: boundary-length coarsening proxy
        PhaseField pf(lab, N, Q);
        std::vector<double> pt, pR2;
        double tt = 0;
        int pf_steps = 3000, psamp = pf_steps / 15;
        for (int s = 0; s < pf_steps; ++s) {
            pf.step(); tt += pf.dt;
            if (s % psamp == 0) {
                long long b = boundary_count(pf.grain_field(), N);
                if (b > 0) { pt.push_back(tt); pR2.push_back(std::pow((double)N * N / b, 2)); }
            }
        }
        int m = (int)pt.size();
        sx = sy = sxx = sxy = syy = 0;
        for (int i = 0; i < m; ++i) {
            sx += pt[i]; sy += pR2[i]; sxx += pt[i] * pt[i];
            sxy += pt[i] * pR2[i]; syy += pR2[i] * pR2[i];
        }
        denom = (m * sxx - sx * sx) * (m * syy - sy * sy);
        double pr2 = denom > 0 ? std::pow(m * sxy - sx * sy, 2) / denom : 0.0;
        printf("PF  fit: <R>^2 grew %.2fx over run, parabolic linearity R^2 = %.4f\n",
               pR2.back() / pR2.front(), pr2);
    }
    return 0;
}
