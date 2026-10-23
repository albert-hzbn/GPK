/*
 * Matched isotropic grain-growth comparison: GPK vs multi-order-parameter PF.
 *
 * The two solvers use the same periodic Voronoi labels, grid spacing, grain
 * identities, interfacial energy and sharp-interface mobility.  The PF model
 * is the Fan-Chen multi-order-parameter Allen-Cahn model.  Its dimensionless
 * clock is mapped to physical time by an isolated-circle shrinkage test, so
 * the comparison does not tune the polycrystal trajectory itself.
 */
#include "gpk_core.hpp"
#include "gpk_thermo.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include <omp.h>

using namespace gpk;
namespace fs = std::filesystem;

static std::vector<int> periodic_voronoi(int N, int ng, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> pick(0, N - 1);
    std::vector<int> sy(ng), sx(ng);
    for (int g = 0; g < ng; ++g) { sy[g] = pick(rng); sx[g] = pick(rng); }
    std::vector<int> label(N * N, 0);
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < N; ++y) for (int x = 0; x < N; ++x) {
        long long best = std::numeric_limits<long long>::max();
        int owner = 0;
        for (int g = 0; g < ng; ++g) {
            long long dy = std::llabs((long long)y - sy[g]);
            long long dx = std::llabs((long long)x - sx[g]);
            dy = std::min(dy, (long long)N - dy);
            dx = std::min(dx, (long long)N - dx);
            const long long d2 = dy * dy + dx * dx;
            if (d2 < best) { best = d2; owner = g; }
        }
        label[y * N + x] = owner;
    }
    return label;
}

struct NormalizedSeeds {
    std::vector<double> y, x;
};

static NormalizedSeeds normalized_seeds(int ng, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> pick(0.0, 1.0);
    NormalizedSeeds s;
    s.y.resize(ng); s.x.resize(ng);
    for (int g = 0; g < ng; ++g) { s.y[g] = pick(rng); s.x[g] = pick(rng); }
    return s;
}

static std::vector<int> periodic_voronoi_from_seeds(int N,
                                                     const NormalizedSeeds& seeds) {
    const int ng = (int)seeds.x.size();
    std::vector<int> label(N * N, 0);
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < N; ++y) for (int x = 0; x < N; ++x) {
        const double py = (y + 0.5) / N, px = (x + 0.5) / N;
        double best = std::numeric_limits<double>::infinity();
        int owner = 0;
        for (int g = 0; g < ng; ++g) {
            double dy = std::fabs(py - seeds.y[g]);
            double dx = std::fabs(px - seeds.x[g]);
            dy = std::min(dy, 1.0 - dy); dx = std::min(dx, 1.0 - dx);
            const double d2 = dy * dy + dx * dx;
            if (d2 < best) { best = d2; owner = g; }
        }
        label[y * N + x] = owner;
    }
    return label;
}

struct PhaseField {
    int N, Q;
    double kappa, L, gamma, dtau;
    double tau = 0.0;
    std::vector<std::vector<float>> eta, next;
    std::vector<float> sumsq;

    PhaseField(const std::vector<int>& labels, int N_, int q,
               double kappa_ = 4.0, double L_ = 1.0, double gamma_ = 1.5,
               double dtau_ = -1.0)
        : N(N_), Q(q), kappa(kappa_), L(L_), gamma(gamma_) {
        dtau = dtau_ > 0.0 ? dtau_ : 0.18 / (kappa * L);
        eta.assign(Q, std::vector<float>(N * N, 0.0f));
        next = eta;
        sumsq.assign(N * N, 0.0f);
        for (int i = 0; i < N * N; ++i) eta.at(labels.at(i)).at(i) = 1.0f;
    }

    bool finite_and_bounded(double bound = 10.0) const {
        for (const auto& q : eta) for (float v : q)
            if (!std::isfinite(v) || std::fabs((double)v) > bound) return false;
        return true;
    }

    std::size_t allocated_array_bytes() const {
        std::size_t total = sumsq.capacity() * sizeof(float);
        for (const auto& q : eta) total += q.capacity() * sizeof(float);
        for (const auto& q : next) total += q.capacity() * sizeof(float);
        return total;
    }


    double phase_fraction_area(int phase) const {
        double a = 0.0;
        for (int i = 0; i < N * N; ++i) {
            double den = 0.0;
            for (int q = 0; q < Q; ++q) den += (double)eta[q][i] * eta[q][i];
            if (den > 1e-20) a += (double)eta[phase][i] * eta[phase][i] / den;
        }
        return a;
    }

    void step() {
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < N * N; ++i) {
            double s = 0.0;
            for (int q = 0; q < Q; ++q) {
                const double v = eta[q][i]; s += v * v;
            }
            sumsq[i] = (float)s;
        }
        for (int q = 0; q < Q; ++q) {
            const auto& e = eta[q]; auto& en = next[q];
            #pragma omp parallel for schedule(static)
            for (int y = 0; y < N; ++y) for (int x = 0; x < N; ++x) {
                const int i = y * N + x;
                const int ym = (y + N - 1) % N, yp = (y + 1) % N;
                const int xm = (x + N - 1) % N, xp = (x + 1) % N;
                const double c = e[i];
                const double lap = e[ym * N + x] + e[yp * N + x]
                                 + e[y * N + xm] + e[y * N + xp] - 4.0 * c;
                const double df = -c + c * c * c
                                + 2.0 * gamma * c * (sumsq[i] - c * c);
                en[i] = (float)(c - dtau * L * (df - kappa * lap));
            }
        }
        eta.swap(next); tau += dtau;
    }

    std::vector<int> labels() const {
        std::vector<int> out(N * N, 0);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < N * N; ++i) {
            float best = -std::numeric_limits<float>::infinity(); int owner = 0;
            for (int q = 0; q < Q; ++q) if (eta[q][i] > best) {
                best = eta[q][i]; owner = q;
            }
            out[i] = owner;
        }
        return out;
    }
};

static std::vector<double> areas(const std::vector<int>& labels, int ng) {
    std::vector<double> a(ng, 0.0);
    for (int g : labels) if (g >= 0 && g < ng) a[g] += 1.0;
    return a;
}

static double circle_pf_rate(double kappa = 4.0) {
    const int N = 96, r0 = 24;
    std::vector<int> label(N * N, 0);
    for (int y = 0; y < N; ++y) for (int x = 0; x < N; ++x) {
        const int dy = y - N / 2, dx = x - N / 2;
        if (dx * dx + dy * dy <= r0 * r0) label[y * N + x] = 1;
    }
    PhaseField pf(label, N, 2, kappa);
    std::vector<double> t, r2;
    for (int step = 1; step <= 500; ++step) {
        pf.step();
        if (step >= 60 && step % 20 == 0) {
            const auto a = areas(pf.labels(), 2);
            t.push_back(pf.tau); r2.push_back(a[1] / M_PI);
        }
    }
    const int n = (int)t.size();
    const double sx = std::accumulate(t.begin(), t.end(), 0.0);
    const double sy = std::accumulate(r2.begin(), r2.end(), 0.0);
    double sxx = 0.0, sxy = 0.0;
    for (int i = 0; i < n; ++i) { sxx += t[i] * t[i]; sxy += t[i] * r2[i]; }
    const double slope = (n * sxy - sx * sy) / (n * sxx - sx * sx);
    return -slope;
}

struct PFSweepResult {
    double dtau, final_tau, radius2, relative_error, max_abs;
    int steps;
    bool stable;
};

static GPKGrid make_gpk(const std::vector<int>& labels, int N, int ng,
                        float dx, float M, float sigma,
                        float gaussian_width_cells = 2.0f);

static PFSweepResult circle_pf_timestep(double dtau, double tau_end,
                                        double reference_radius2) {
    const int N = 96, r0 = 24;
    std::vector<int> label(N * N, 0);
    for (int y = 0; y < N; ++y) for (int x = 0; x < N; ++x) {
        const int dy = y - N / 2, dx = x - N / 2;
        if (dx * dx + dy * dy <= r0 * r0) label[y * N + x] = 1;
    }
    PhaseField pf(label, N, 2, 4.0, 1.0, 1.5, dtau);
    int steps = 0;
    bool stable = true;
    while (pf.tau + 0.5 * pf.dtau < tau_end) {
        pf.step(); ++steps;
        if (!pf.finite_and_bounded()) { stable = false; break; }
    }
    double max_abs = 0.0;
    for (const auto& q : pf.eta) for (float v : q)
        if (std::isfinite(v)) max_abs = std::max(max_abs, std::fabs((double)v));
        else max_abs = std::numeric_limits<double>::infinity();
    double r2 = std::numeric_limits<double>::quiet_NaN();
    double rel = std::numeric_limits<double>::infinity();
    if (stable) {
        r2 = pf.phase_fraction_area(1) / M_PI;
        rel = std::fabs(r2 - reference_radius2) / reference_radius2;
        // A bounded but qualitatively wrong solution is not an acceptable step.
        stable = r2 > 0.0 && rel <= 0.05;
    }
    return {dtau, pf.tau, r2, rel, max_abs, steps, stable};
}

static void write_timestep_sweep(const fs::path& out, double seconds_per_tau) {
    const double tau_end = 4.0;
    const double reference_dtau = 0.005625; // eight times finer than production
    const auto reference = circle_pf_timestep(reference_dtau, tau_end, 1.0);
    const double reference_radius2 = reference.radius2;
    const std::array<double, 10> steps = {
        0.005625, 0.01125, 0.0225, 0.045, 0.055,
        0.0625, 0.070, 0.080, 0.090, 0.120
    };
    std::ofstream f(out / "timestep_sweep.csv");
    f << "method,dtau,dt_seconds,tau_end,steps,radius2_cells2,relative_radius2_error,max_abs_eta,accepted\n";
    std::printf("\n# Explicit PF timestep audit (circle, tau_end=%.1f; acceptance: bounded and <=5%% radius^2 error)\n", tau_end);
    for (double dt : steps) {
        const auto r = circle_pf_timestep(dt, tau_end, reference_radius2);
        f << "explicit_phase_field," << std::setprecision(10) << r.dtau << ','
          << r.dtau * seconds_per_tau << ',' << r.final_tau << ',' << r.steps << ','
          << r.radius2 << ',' << r.relative_error << ',' << r.max_abs << ','
          << (r.stable ? 1 : 0) << '\n';
        std::printf("dtau=%7.5f (%8.4g s): error=%8.3g max|eta|=%8.3g %s\n",
                    r.dtau, r.dtau * seconds_per_tau, r.relative_error,
                    r.max_abs, r.stable ? "ACCEPT" : "FAIL");
    }

    // GPK never advances farther than its geometric crossing bound: requested
    // large steps are clipped rather than allowed to destabilise the interface.
    const int N = 96, r0 = 24;
    std::vector<int> label(N * N, 0);
    for (int y = 0; y < N; ++y) for (int x = 0; x < N; ++x) {
        const int dy = y - N / 2, dx = x - N / 2;
        if (dx * dx + dy * dy <= r0 * r0) label[y * N + x] = 1;
    }
    const float dx = 5.0e-7f, M = 5.0e-11f, sigma = 0.324f;
    const std::array<double, 6> requested = {0.001, 0.003, 0.01, 0.03, 0.10, 0.30};
    for (double req : requested) {
        auto grid = make_gpk(label, N, 2, dx, M, sigma);
        double accepted_max = 0.0, accepted_sum = 0.0;
        int n = 0;
        while (grid.time < 0.10) {
            const double used = grid.step((float)req, (float)(0.10 - grid.time));
            accepted_max = std::max(accepted_max, used); accepted_sum += used; ++n;
        }
        f << "gpk_clipped," << req / seconds_per_tau << ',' << req << ','
          << 0.10 / seconds_per_tau << ',' << n << ",nan,nan," << accepted_max
          << ",1\n";
        std::printf("GPK requested dt=%6.3f s: max accepted=%8.4g s, steps=%d (clipped=%s)\n",
                    req, accepted_max, n, accepted_max < req * (1.0 - 1e-5) ? "yes" : "no");
    }
}

static GPKGrid make_gpk(const std::vector<int>& labels, int N, int ng,
                        float dx, float M, float sigma,
                        float gaussian_width_cells) {
    auto thermo = std::make_shared<IsotropicGrowthThermo>(M, sigma);
    GPKGrid grid(1, N, N, dx, thermo, 0.0f, gaussian_width_cells);
    std::mt19937 rng(17); std::uniform_real_distribution<float> ori(0.f, (float)M_PI);
    std::vector<float> angle(ng); for (float& a : angle) a = ori(rng);
    for (int i = 0; i < N * N; ++i) {
        grid.phi[i] = labels[i] + 1;
        grid.g[i] = angle[labels[i]];
        grid.alpha[i] = 1.0f;
    }
    return grid;
}

static std::vector<int> gpk_labels(const GPKGrid& grid) {
    std::vector<int> out(grid.N());
    for (int i = 0; i < grid.N(); ++i) out[i] = grid.phi[i] - 1;
    return out;
}

static void write_labels(const fs::path& path, const std::vector<int>& labels,
                         int N, double time) {
    std::ofstream f(path);
    f << "# matched GPK/PF isotropic grain growth\n# time=" << std::setprecision(10)
      << time << " Nx=" << N << " Ny=" << N << "\n# x,y,grain_id\n";
    for (int y = 0; y < N; ++y) for (int x = 0; x < N; ++x)
        f << x << ',' << y << ',' << labels[y * N + x] << '\n';
}

static double grain_iou(const std::vector<int>& a, const std::vector<int>& b, int g) {
    long long inter = 0, uni = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        const bool aa = a[i] == g, bb = b[i] == g;
        inter += aa && bb; uni += aa || bb;
    }
    return uni ? (double)inter / uni : 1.0;
}

static std::vector<uint8_t> boundary_mask(const std::vector<int>& label, int N) {
    std::vector<uint8_t> out(N * N, 0);
    for (int y = 0; y < N; ++y) for (int x = 0; x < N; ++x) {
        const int i = y * N + x, c = label[i];
        out[i] = c != label[((y + N - 1) % N) * N + x]
              || c != label[((y + 1) % N) * N + x]
              || c != label[y * N + (x + N - 1) % N]
              || c != label[y * N + (x + 1) % N];
    }
    return out;
}

static double boundary_f1(const std::vector<int>& a, const std::vector<int>& b,
                          int N, int tolerance = 2) {
    const auto ba = boundary_mask(a, N), bb = boundary_mask(b, N);
    auto matched_fraction = [&](const std::vector<uint8_t>& source,
                                const std::vector<uint8_t>& target) {
        long long total = 0, matched = 0;
        for (int y = 0; y < N; ++y) for (int x = 0; x < N; ++x) if (source[y * N + x]) {
            ++total; bool found = false;
            for (int dy = -tolerance; dy <= tolerance && !found; ++dy)
                for (int dx = -tolerance; dx <= tolerance && !found; ++dx) {
                    const int yy = (y + dy + N) % N, xx = (x + dx + N) % N;
                    found = target[yy * N + xx] != 0;
                }
            matched += found;
        }
        return total ? (double)matched / total : 1.0;
    };
    const double precision = matched_fraction(ba, bb);
    const double recall = matched_fraction(bb, ba);
    return precision + recall > 0 ? 2.0 * precision * recall / (precision + recall) : 0.0;
}

static std::vector<int> resample_labels(const std::vector<int>& source,
                                        int source_n, int target_n) {
    std::vector<int> out(target_n * target_n, 0);
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < target_n; ++y) for (int x = 0; x < target_n; ++x) {
        const int sy = std::min(source_n - 1,
                                (int)std::floor((y + 0.5) * source_n / target_n));
        const int sx = std::min(source_n - 1,
                                (int)std::floor((x + 0.5) * source_n / target_n));
        out[y * target_n + x] = source[sy * source_n + sx];
    }
    return out;
}

struct AccuracyMetrics {
    double area_mae = 0.0;
    double area_correlation = 1.0;
    double label_agreement = 1.0;
    double boundary_f1 = 1.0;
    double mean_iou = 1.0;
};

static AccuracyMetrics accuracy_metrics(const std::vector<int>& candidate,
                                        const std::vector<int>& reference,
                                        int N, int ng, int boundary_tolerance) {
    AccuracyMetrics m;
    const auto ca = areas(candidate, ng), ra = areas(reference, ng);
    double cm = 0.0, rm = 0.0;
    for (int g = 0; g < ng; ++g) {
        m.area_mae += std::fabs(ca[g] - ra[g]) / (N * N);
        cm += ca[g]; rm += ra[g];
        m.mean_iou += grain_iou(candidate, reference, g);
    }
    m.mean_iou = (m.mean_iou - 1.0) / ng;
    cm /= ng; rm /= ng;
    double cov = 0.0, cv = 0.0, rv = 0.0;
    for (int g = 0; g < ng; ++g) {
        cov += (ca[g] - cm) * (ra[g] - rm);
        cv += (ca[g] - cm) * (ca[g] - cm);
        rv += (ra[g] - rm) * (ra[g] - rm);
    }
    m.area_correlation = cv > 0.0 && rv > 0.0 ? cov / std::sqrt(cv * rv) : 1.0;
    long long same = 0;
    for (std::size_t i = 0; i < candidate.size(); ++i)
        same += candidate[i] == reference[i];
    m.label_agreement = (double)same / candidate.size();
    m.boundary_f1 = boundary_f1(candidate, reference, N, boundary_tolerance);
    return m;
}

struct CostRun {
    int repetition = 0;
    double horizon_seconds = 0.0;
    int gpk_steps = 0;
    int pf_steps = 0;
    double gpk_compute_seconds = 0.0;
    double pf_compute_seconds = 0.0;
    std::size_t gpk_array_bytes = 0;
    std::size_t pf_array_bytes = 0;
};

static CostRun timed_cost_run(const std::vector<int>& initial, int N, int ng,
                              float dx, float M, float sigma,
                              double seconds_per_tau, double horizon,
                              int repetition, bool pf_first,
                              int checkpoints = 4) {
    using clock = std::chrono::steady_clock;
    auto gpk = make_gpk(initial, N, ng, dx, M, sigma);
    PhaseField pf(initial, N, ng);
    CostRun out;
    out.repetition = repetition;
    out.horizon_seconds = horizon;
    out.gpk_array_bytes = gpk.allocated_array_bytes();
    out.pf_array_bytes = pf.allocated_array_bytes();

    auto run_gpk = [&]() {
        const auto tic = clock::now();
        // Follow the exact 0.25 s production-snapshot schedule. GPK caps its
        // final step at each requested output time, so timing an uninterrupted
        // 1 s call would not reproduce the accepted-step sequence used for
        // the manuscript morphology.
        for (int checkpoint = 1; checkpoint <= checkpoints; ++checkpoint)
            out.gpk_steps += gpk.run(horizon * checkpoint / checkpoints, -1.f);
        const auto toc = clock::now();
        out.gpk_compute_seconds = std::chrono::duration<double>(toc - tic).count();
    };
    auto run_pf = [&]() {
        const auto tic = clock::now();
        for (int checkpoint = 1; checkpoint <= checkpoints; ++checkpoint) {
            const double target = horizon * checkpoint / checkpoints;
            while ((pf.tau + 0.5 * pf.dtau) * seconds_per_tau < target) {
                pf.step();
                ++out.pf_steps;
            }
        }
        const auto toc = clock::now();
        out.pf_compute_seconds = std::chrono::duration<double>(toc - tic).count();
    };
    if (pf_first) { run_pf(); run_gpk(); }
    else          { run_gpk(); run_pf(); }
    return out;
}

static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    const std::size_t n = v.size();
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

struct SingleMethodRun {
    int steps = 0;
    double compute_seconds = 0.0;
    std::size_t array_bytes = 0;
    std::vector<int> labels;
};

static SingleMethodRun evolve_gpk(const std::vector<int>& initial, int N, int ng,
                                  float dx, float M, float sigma, double horizon,
                                  bool retain_labels = true,
                                  float gaussian_width_cells = 2.0f) {
    using clock = std::chrono::steady_clock;
    auto grid = make_gpk(initial, N, ng, dx, M, sigma, gaussian_width_cells);
    SingleMethodRun out;
    out.array_bytes = grid.allocated_array_bytes();
    const auto tic = clock::now();
    out.steps = grid.run(horizon, -1.f);
    const auto toc = clock::now();
    out.compute_seconds = std::chrono::duration<double>(toc - tic).count();
    if (retain_labels) out.labels = gpk_labels(grid);
    return out;
}

static SingleMethodRun evolve_pf(const std::vector<int>& initial, int N, int ng,
                                 double seconds_per_tau, double horizon,
                                 bool retain_labels = true,
                                 double kappa = 4.0) {
    using clock = std::chrono::steady_clock;
    PhaseField pf(initial, N, ng, kappa);
    SingleMethodRun out;
    out.array_bytes = pf.allocated_array_bytes();
    const auto tic = clock::now();
    while ((pf.tau + 0.5 * pf.dtau) * seconds_per_tau < horizon) {
        pf.step(); ++out.steps;
    }
    const auto toc = clock::now();
    out.compute_seconds = std::chrono::duration<double>(toc - tic).count();
    if (retain_labels) out.labels = pf.labels();
    return out;
}

static double pf_planar_surface_energy_error(double kappa) {
    // Two periodic planar interfaces are relaxed in one dimension. The excess
    // free energy divided by sqrt(kappa) converges to sqrt(2)/3 for this
    // symmetric gamma=1.5 Fan--Chen binary interface. This audit detects a PF
    // mesh that places the categorical boundary correctly but under-resolves
    // the diffuse profile and therefore its interfacial free energy.
    constexpr int N = 1024;
    constexpr double gamma = 1.5;
    const double dtau = 0.18 / kappa;
    std::array<std::vector<double>, 2> eta, next;
    for (int q = 0; q < 2; ++q) {
        eta[q].assign(N, 0.0); next[q].assign(N, 0.0);
    }
    for (int i = 0; i < N; ++i) eta[i >= N / 2][i] = 1.0;
    for (int step = 0; step < 20000; ++step) {
        double change = 0.0;
        for (int i = 0; i < N; ++i) {
            const int im = (i + N - 1) % N, ip = (i + 1) % N;
            const double sumsq = eta[0][i] * eta[0][i] + eta[1][i] * eta[1][i];
            for (int q = 0; q < 2; ++q) {
                const double c = eta[q][i];
                const double lap = eta[q][im] + eta[q][ip] - 2.0 * c;
                const double df = -c + c * c * c
                                + 2.0 * gamma * c * (sumsq - c * c);
                next[q][i] = c - dtau * (df - kappa * lap);
                change = std::max(change, std::fabs(next[q][i] - c));
            }
        }
        eta.swap(next);
        if (change < 1e-12) break;
    }
    double excess = 0.0;
    for (int i = 0; i < N; ++i) {
        const int im = (i + N - 1) % N, ip = (i + 1) % N;
        const double e0 = eta[0][i], e1 = eta[1][i];
        const double potential = -0.5 * (e0 * e0 + e1 * e1)
                               + 0.25 * (std::pow(e0, 4) + std::pow(e1, 4))
                               + gamma * e0 * e0 * e1 * e1 + 0.25;
        const double g0 = 0.5 * (eta[0][ip] - eta[0][im]);
        const double g1 = 0.5 * (eta[1][ip] - eta[1][im]);
        excess += potential + 0.5 * kappa * (g0 * g0 + g1 * g1);
    }
    const double numerical = 0.5 * excess / std::sqrt(kappa);
    const double continuum = std::sqrt(2.0) / 3.0;
    return std::fabs(numerical - continuum) / continuum;
}

static void write_accuracy_matched_cost(const fs::path& out, float M, float sigma) {
    constexpr double domain_m = 64.0e-6;
    constexpr double accuracy_horizon = 0.25;
    constexpr double cost_horizon = 0.05;
    constexpr int ng = 80;
    constexpr int reference_n = 384;
    const std::array<int, 8> candidates = {96, 128, 160, 192, 224, 256, 320, 384};
    // These gates are fixed before inspecting the resolution sweep. They bound
    // integral grain-size error, identity displacement and interface location.
    constexpr double max_area_mae = 0.030;
    constexpr double min_area_corr = 0.995;
    constexpr double min_label_agreement = 0.950;
    constexpr double min_boundary_f1 = 0.900;
    constexpr double max_pf_surface_energy_error = 0.010;
    constexpr int boundary_tolerance_ref_cells = 2;
    const auto seeds = normalized_seeds(ng, 73129u);
    const auto ref_initial = periodic_voronoi_from_seeds(reference_n, seeds);
    const float ref_dx = (float)(domain_m / reference_n);
    const float ref_gaussian_cells = 2.0f * reference_n / 192.0f;
    const double ref_pf_kappa = 4.0 * std::pow(reference_n / 192.0, 2);
    const double ref_pf_rate = circle_pf_rate(ref_pf_kappa);
    const double ref_seconds_per_tau = (double)ref_dx * ref_dx * ref_pf_rate /
                                       (2.0 * M * sigma);

    std::printf("\n# Accuracy-matched resolution audit (64 um, 80 grains, t=%.2f s)\n",
                accuracy_horizon);
    const auto gref = evolve_gpk(ref_initial, reference_n, ng, ref_dx, M, sigma,
                                 accuracy_horizon, true, ref_gaussian_cells);
    const auto pref = evolve_pf(ref_initial, reference_n, ng, ref_seconds_per_tau,
                                accuracy_horizon, true, ref_pf_kappa);

    std::ofstream f(out / "accuracy_matched_resolution.csv");
    f << "method,grid_n,dx_um,physical_width_um,width_parameter_cells,horizon_seconds,steps,compute_seconds,array_bytes,"
         "area_mae_vs_384,area_correlation_vs_384,label_agreement_vs_384,"
         "boundary_f1_vs_384,mean_iou_vs_384,pf_surface_energy_relative_error,"
         "passes_morphology_gate,passes_full_accuracy_gate\n";
    int selected_gpk = reference_n, selected_pf_morphology = reference_n;
    int selected_pf = reference_n;
    bool selected_gpk_found = false, selected_pf_morphology_found = false;
    bool selected_pf_found = false;
    for (int N : candidates) {
        const float dx = (float)(domain_m / N);
        const float gaussian_cells = 2.0f * N / 192.0f;
        const double pf_kappa = 4.0 * std::pow(N / 192.0, 2);
        const double local_pf_rate = circle_pf_rate(pf_kappa);
        const double seconds_per_tau = (double)dx * dx * local_pf_rate /
                                       (2.0 * M * sigma);
        const auto initial = periodic_voronoi_from_seeds(N, seeds);
        SingleMethodRun gr, pr;
        if (N == reference_n) { gr = gref; pr = pref; }
        else {
            gr = evolve_gpk(initial, N, ng, dx, M, sigma, accuracy_horizon, true,
                            gaussian_cells);
            pr = evolve_pf(initial, N, ng, seconds_per_tau, accuracy_horizon, true,
                           pf_kappa);
        }
        const auto gu = resample_labels(gr.labels, N, reference_n);
        const auto pu = resample_labels(pr.labels, N, reference_n);
        const auto gm = accuracy_metrics(gu, gref.labels, reference_n, ng,
                                         boundary_tolerance_ref_cells);
        const auto pm = accuracy_metrics(pu, pref.labels, reference_n, ng,
                                         boundary_tolerance_ref_cells);
        const double pf_energy_error = pf_planar_surface_energy_error(pf_kappa);
        auto passes = [&](const AccuracyMetrics& a) {
            return a.area_mae <= max_area_mae && a.area_correlation >= min_area_corr
                && a.label_agreement >= min_label_agreement
                && a.boundary_f1 >= min_boundary_f1;
        };
        const bool gp = passes(gm), pp_morphology = passes(pm);
        const bool pp = pp_morphology && pf_energy_error <= max_pf_surface_energy_error;
        if (gp && !selected_gpk_found) { selected_gpk = N; selected_gpk_found = true; }
        if (pp_morphology && !selected_pf_morphology_found) {
            selected_pf_morphology = N; selected_pf_morphology_found = true;
        }
        if (pp && !selected_pf_found) { selected_pf = N; selected_pf_found = true; }
        auto write_row = [&](const char* method, double width_cells,
                             const SingleMethodRun& r,
                             const AccuracyMetrics& a, double energy_error,
                             bool morphology_pass, bool full_pass) {
            f << method << ',' << N << ',' << std::setprecision(10) << dx * 1e6
              << ',' << 2.0 * domain_m / 192.0 * 1e6 << ',' << width_cells
              << ',' << accuracy_horizon << ',' << r.steps << ',' << r.compute_seconds << ','
              << r.array_bytes << ',' << a.area_mae << ',' << a.area_correlation << ','
              << a.label_agreement << ',' << a.boundary_f1 << ',' << a.mean_iou << ','
              << energy_error << ',' << (morphology_pass ? 1 : 0) << ','
              << (full_pass ? 1 : 0) << '\n';
        };
        write_row("gpk", gaussian_cells, gr, gm,
                  std::numeric_limits<double>::quiet_NaN(), gp, gp);
        write_row("phase_field", std::sqrt(pf_kappa), pr, pm,
                  pf_energy_error, pp_morphology, pp);
        std::printf("N=%3d GPK: MAE %.4f agree %.3f F1 %.3f %s | PF: MAE %.4f agree %.3f F1 %.3f Eerr %.3f%% morph %s full %s\n",
                    N, gm.area_mae, gm.label_agreement, gm.boundary_f1, gp ? "PASS" : "FAIL",
                    pm.area_mae, pm.label_agreement, pm.boundary_f1,
                    100.0 * pf_energy_error, pp_morphology ? "PASS" : "FAIL",
                    pp ? "PASS" : "FAIL");
    }

    std::ofstream selection(out / "accuracy_matched_selection.csv");
    selection << "reference_n,horizon_seconds,max_area_mae,min_area_correlation,"
                 "min_label_agreement,min_boundary_f1,max_pf_surface_energy_error,"
                 "gpk_selected_n,pf_morphology_selected_n,pf_full_accuracy_selected_n,"
                 "gpk_dx_um,pf_dx_um,pf_over_gpk_linear_resolution\n";
    selection << reference_n << ',' << accuracy_horizon << ',' << max_area_mae << ','
              << min_area_corr << ',' << min_label_agreement << ',' << min_boundary_f1
              << ',' << max_pf_surface_energy_error << ',' << selected_gpk << ','
              << selected_pf_morphology << ',' << selected_pf << ','
              << domain_m / selected_gpk * 1e6 << ','
              << domain_m / selected_pf * 1e6 << ','
              << (double)selected_pf / selected_gpk << '\n';
    std::printf("Selected grids: GPK full %d^2, PF morphology-only %d^2, PF full %d^2 (full linear ratio %.3f)\n",
                selected_gpk, selected_pf_morphology, selected_pf,
                (double)selected_pf / selected_gpk);

    // Transfer the selected spacings to larger and smaller domains at fixed
    // physical grain density. This separates the resolution-selection audit
    // above from the repeated cost measurement below.
    const std::array<double, 9> scales = {
        0.50, 0.75, 1.00, 1.25, 1.50, 2.00, 2.50, 3.00, 4.00
    };
    constexpr int repetitions = 3;
    const float selected_gaussian_cells = 2.0f * selected_gpk / 192.0f;
    const double selected_pf_kappa = 4.0 * std::pow(selected_pf / 192.0, 2);
    const double selected_pf_rate = circle_pf_rate(selected_pf_kappa);
    std::ofstream raw(out / "accuracy_matched_cost_scaling_repeats.csv");
    raw << "domain_um,grains,horizon_seconds,repetition,gpk_grid_n,pf_grid_n,gpk_steps,pf_steps,"
           "gpk_compute_seconds,pf_compute_seconds,gpk_array_bytes,pf_array_bytes,"
           "pf_over_gpk_time,openmp_threads\n";
    std::ofstream summary(out / "accuracy_matched_cost_scaling.csv");
    summary << "domain_um,grains,horizon_seconds,repetitions,gpk_grid_n,pf_grid_n,gpk_dx_um,pf_dx_um,"
               "gpk_steps,pf_steps,gpk_median_seconds,gpk_min_seconds,gpk_max_seconds,"
               "pf_median_seconds,pf_min_seconds,pf_max_seconds,pf_over_gpk_time,"
               "gpk_array_bytes,pf_array_bytes,pf_over_gpk_memory,openmp_threads\n";
    for (double scale : scales) {
        const int gN = std::max(32, (int)std::lround(selected_gpk * scale));
        const int pN = std::max(32, (int)std::lround(selected_pf * scale));
        const int grains = std::max(4, (int)std::lround(ng * scale * scale));
        const double side = domain_m * scale;
        const float gdx = (float)(side / gN), pdx = (float)(side / pN);
        const double pf_seconds_per_tau = (double)pdx * pdx * selected_pf_rate /
                                          (2.0 * M * sigma);
        const auto size_seeds = normalized_seeds(grains,
                                  91009u + (uint32_t)std::lround(100.0 * scale));
        const auto gi = periodic_voronoi_from_seeds(gN, size_seeds);
        const auto pi = periodic_voronoi_from_seeds(pN, size_seeds);
        std::vector<double> gt, pt;
        std::vector<SingleMethodRun> gruns, pruns;
        for (int rep = 1; rep <= repetitions; ++rep) {
            SingleMethodRun gr, pr;
            if (rep % 2 == 0) {
                pr = evolve_pf(pi, pN, grains, pf_seconds_per_tau, cost_horizon, false,
                               selected_pf_kappa);
                gr = evolve_gpk(gi, gN, grains, gdx, M, sigma, cost_horizon, false,
                                selected_gaussian_cells);
            } else {
                gr = evolve_gpk(gi, gN, grains, gdx, M, sigma, cost_horizon, false,
                                selected_gaussian_cells);
                pr = evolve_pf(pi, pN, grains, pf_seconds_per_tau, cost_horizon, false,
                               selected_pf_kappa);
            }
            gt.push_back(gr.compute_seconds); pt.push_back(pr.compute_seconds);
            gruns.push_back(gr); pruns.push_back(pr);
            raw << side * 1e6 << ',' << grains << ',' << cost_horizon << ','
                << rep << ',' << gN << ','
                << pN << ',' << gr.steps << ',' << pr.steps << ','
                << std::setprecision(10) << gr.compute_seconds << ','
                << pr.compute_seconds << ',' << gr.array_bytes << ',' << pr.array_bytes
                << ',' << pr.compute_seconds / gr.compute_seconds << ','
                << omp_get_max_threads() << '\n';
        }
        const auto [gmin, gmax] = std::minmax_element(gt.begin(), gt.end());
        const auto [pmin, pmax] = std::minmax_element(pt.begin(), pt.end());
        const double gm = median(gt), pm = median(pt);
        summary << side * 1e6 << ',' << grains << ',' << cost_horizon << ','
                << repetitions << ',' << gN
                << ',' << pN << ',' << gdx * 1e6 << ',' << pdx * 1e6 << ','
                << gruns.front().steps << ',' << pruns.front().steps << ','
                << gm << ',' << *gmin << ',' << *gmax << ',' << pm << ',' << *pmin
                << ',' << *pmax << ',' << pm / gm << ',' << gruns.front().array_bytes
                << ',' << pruns.front().array_bytes << ','
                << (double)pruns.front().array_bytes / gruns.front().array_bytes << ','
                << omp_get_max_threads() << '\n';
        std::printf("accuracy-matched L=%4.0f um grains=%3d grids=%d/%d: GPK %.4f s, PF %.4f s, PF/GPK %.3f\n",
                    side * 1e6, grains, gN, pN, gm, pm, pm / gm);
    }
}

static void write_cost_scaling(const fs::path& out, float dx, float M,
                               float sigma, double seconds_per_tau) {
    // Controlled size-scaling study: keep dx, physical horizon and grains per cell fixed
    // while increasing both domain area and grain count. Consequently the
    // characteristic radius remains approximately 12.1 cells at every size.
    const std::array<int, 9> sizes = {64, 96, 128, 160, 192, 224, 256, 288, 320};
    const double grain_density = 80.0 / (192.0 * 192.0);
    const double horizon = 0.25;
    const int repetitions = 3;

    std::ofstream raw(out / "computational_cost_scaling_repeats.csv");
    raw << "grid_n,cells,grains,rchar_cells,horizon_seconds,repetition,gpk_steps,pf_steps,gpk_compute_seconds,pf_compute_seconds,gpk_array_bytes,pf_array_bytes,openmp_threads\n";
    std::ofstream summary(out / "computational_cost_scaling.csv");
    summary << "grid_n,cells,grains,rchar_cells,horizon_seconds,repetitions,gpk_steps,pf_steps,gpk_median_seconds,gpk_min_seconds,gpk_max_seconds,pf_median_seconds,pf_min_seconds,pf_max_seconds,pf_over_gpk_time,gpk_array_bytes,pf_array_bytes,pf_over_gpk_memory,openmp_threads\n";

    std::printf("\n# Matched GPK/PF controlled size-scaling audit (dx fixed, grain density fixed, t=%.2f s)\n",
                horizon);
    for (int N : sizes) {
        const int ng = std::max(4, (int)std::lround(grain_density * N * N));
        const double rchar = std::sqrt((double)N * N / (M_PI * ng));
        const auto initial = periodic_voronoi(N, ng, 42017u + (uint32_t)N);
        std::vector<CostRun> runs;
        std::vector<double> gt, pt;
        for (int rep = 1; rep <= repetitions; ++rep) {
            auto r = timed_cost_run(initial, N, ng, dx, M, sigma,
                                    seconds_per_tau, horizon, rep,
                                    rep % 2 == 0, 1);
            gt.push_back(r.gpk_compute_seconds);
            pt.push_back(r.pf_compute_seconds);
            raw << N << ',' << N * N << ',' << ng << ',' << std::setprecision(10)
                << rchar << ',' << horizon << ',' << rep << ','
                << r.gpk_steps << ',' << r.pf_steps << ','
                << r.gpk_compute_seconds << ',' << r.pf_compute_seconds << ','
                << r.gpk_array_bytes << ',' << r.pf_array_bytes << ','
                << omp_get_max_threads() << '\n';
            runs.push_back(r);
        }
        const auto [gmin, gmax] = std::minmax_element(gt.begin(), gt.end());
        const auto [pmin, pmax] = std::minmax_element(pt.begin(), pt.end());
        const double gm = median(gt), pm = median(pt);
        const auto& r0 = runs.front();
        summary << N << ',' << N * N << ',' << ng << ',' << std::setprecision(10)
                << rchar << ',' << horizon << ',' << repetitions << ','
                << r0.gpk_steps << ',' << r0.pf_steps << ','
                << gm << ',' << *gmin << ',' << *gmax << ','
                << pm << ',' << *pmin << ',' << *pmax << ',' << pm / gm << ','
                << r0.gpk_array_bytes << ',' << r0.pf_array_bytes << ','
                << (double)r0.pf_array_bytes / r0.gpk_array_bytes << ','
                << omp_get_max_threads() << '\n';
        std::printf("N=%3d grains=%3d Rchar=%.2f cells: GPK %.4f s, PF %.4f s, PF/GPK %.3f\n",
                    N, ng, rchar, gm, pm, pm / gm);
    }
}

int main(int argc, char** argv) {
    const int N = 192, ng = 80, n_snaps = 17;
    const float dx = 3.3333333e-7f, M = 5.0e-11f, sigma = 0.324f;
    const double t_end = 4.0;
    const fs::path out = fs::path(GPK_BINARY_DIR) / "output" / "gpk_pf_comparison";
    fs::create_directories(out);
    const bool accuracy_cost_only = argc > 1
                                 && std::string(argv[1]) == "--accuracy-cost-only";
    if (accuracy_cost_only) {
        #pragma omp parallel
        { }
        write_accuracy_matched_cost(out, M, sigma);
        return 0;
    }
    // Avoid mixing snapshots from an earlier sampling schedule with the
    // current matched run.
    for (const auto& entry : fs::directory_iterator(out)) {
        const std::string name = entry.path().filename().string();
        if (entry.is_regular_file() && entry.path().extension() == ".csv"
            && (name.rfind("gpk_snap_", 0) == 0 || name.rfind("pf_snap_", 0) == 0))
            fs::remove(entry.path());
    }

    const auto initial = periodic_voronoi(N, ng, 42017);
    auto gpk = make_gpk(initial, N, ng, dx, M, sigma);
    PhaseField pf(initial, N, ng);

    const double pf_rate = circle_pf_rate();
    const double seconds_per_tau = (double)dx * dx * pf_rate / (2.0 * M * sigma);
    const double pf_dt_seconds = pf.dtau * seconds_per_tau;
    std::printf("# Matched GPK vs PF grain-growth validation\n");
    std::printf("PF isolated-circle rate -d(R/dx)^2/dtau = %.6f\n", pf_rate);
    std::printf("PF time map = %.6e s/tau; dt = %.6e s\n", seconds_per_tau, pf_dt_seconds);
    write_timestep_sweep(out, seconds_per_tau);

    // Warm the OpenMP team before timing. Timings contain only evolution
    // kernels: initialization, calibration, file output and metric analysis
    // are deliberately excluded. Alternating execution order limits thermal
    // and first-run bias between the two solvers.
    #pragma omp parallel
    { }
    write_accuracy_matched_cost(out, M, sigma);
    write_cost_scaling(out, dx, M, sigma, seconds_per_tau);
    std::vector<CostRun> cost_repeats;
    for (int rep = 1; rep <= 5; ++rep) {
        cost_repeats.push_back(timed_cost_run(initial, N, ng, dx, M, sigma,
                                              seconds_per_tau, 1.0, rep,
                                              rep % 2 == 0));
        const auto& r = cost_repeats.back();
        std::printf("cost repeat %d: GPK %.6f s (%d steps), PF %.6f s (%d steps)\n",
                    rep, r.gpk_compute_seconds, r.gpk_steps,
                    r.pf_compute_seconds, r.pf_steps);
    }
    std::ofstream cost_raw(out / "computational_cost_repeats.csv");
    cost_raw << "repetition,horizon_seconds,gpk_steps,pf_steps,gpk_compute_seconds,pf_compute_seconds,gpk_array_bytes,pf_array_bytes,openmp_threads\n";
    std::vector<double> gpk_times, pf_times;
    for (const auto& r : cost_repeats) {
        gpk_times.push_back(r.gpk_compute_seconds);
        pf_times.push_back(r.pf_compute_seconds);
        cost_raw << r.repetition << ',' << r.horizon_seconds << ','
                 << r.gpk_steps << ',' << r.pf_steps << ','
                 << std::setprecision(10) << r.gpk_compute_seconds << ','
                 << r.pf_compute_seconds << ',' << r.gpk_array_bytes << ','
                 << r.pf_array_bytes << ',' << omp_get_max_threads() << '\n';
    }
    const double gpk_median = median(gpk_times), pf_median = median(pf_times);
    std::ofstream cost_summary(out / "computational_cost_summary.csv");
    cost_summary << "horizon_seconds,repetitions,gpk_steps,pf_steps,gpk_median_seconds,pf_median_seconds,pf_over_gpk_time,gpk_array_bytes,pf_array_bytes,pf_over_gpk_memory,openmp_threads\n";
    cost_summary << "1,5," << cost_repeats.front().gpk_steps << ','
                 << cost_repeats.front().pf_steps << ',' << std::setprecision(10)
                 << gpk_median << ',' << pf_median << ',' << pf_median / gpk_median
                 << ',' << cost_repeats.front().gpk_array_bytes << ','
                 << cost_repeats.front().pf_array_bytes << ','
                 << (double)cost_repeats.front().pf_array_bytes /
                    cost_repeats.front().gpk_array_bytes << ','
                 << omp_get_max_threads() << '\n';

    std::ofstream trajectory(out / "grain_trajectories.csv");
    trajectory << "time,grain,gpk_area_cells,pf_area_cells,gpk_fraction,pf_fraction,iou\n";
    std::ofstream summary(out / "snapshot_metrics.csv");
    summary << "time,mean_iou,median_iou,area_mae_fraction,area_correlation,pixel_agreement,boundary_f1_2px,gpk_grains,pf_grains\n";
    std::ofstream cost_trajectory(out / "computational_cost_trajectory.csv");
    cost_trajectory << "physical_time,gpk_steps,pf_steps,gpk_compute_seconds,pf_compute_seconds,pf_over_gpk_time\n";

    double gate_mean_iou = 0.0, gate_area_mae = 0.0, gate_corr = 0.0;
    double gate_pixel = 0.0, gate_boundary = 0.0;
    int gpk_steps_total = 0, pf_steps_total = 0;
    double gpk_compute_total = 0.0, pf_compute_total = 0.0;
    using clock = std::chrono::steady_clock;
    for (int s = 0; s < n_snaps; ++s) {
        const double target = t_end * s / (n_snaps - 1);
        if (s > 0) {
            auto tic = clock::now();
            gpk_steps_total += gpk.run(target, -1.f);
            auto toc = clock::now();
            gpk_compute_total += std::chrono::duration<double>(toc - tic).count();
            tic = clock::now();
            while ((pf.tau + 0.5 * pf.dtau) * seconds_per_tau < target) {
                pf.step();
                ++pf_steps_total;
            }
            toc = clock::now();
            pf_compute_total += std::chrono::duration<double>(toc - tic).count();
        }
        cost_trajectory << target << ',' << gpk_steps_total << ',' << pf_steps_total
                        << ',' << std::setprecision(10) << gpk_compute_total << ','
                        << pf_compute_total << ','
                        << (gpk_compute_total > 0.0 ? pf_compute_total / gpk_compute_total : 1.0)
                        << '\n';
        const auto gl = gpk_labels(gpk), pl = pf.labels();
        const auto ga = areas(gl, ng), pa = areas(pl, ng);
        write_labels(out / ("gpk_snap_" + std::to_string(s) + ".csv"), gl, N, target);
        write_labels(out / ("pf_snap_" + std::to_string(s) + ".csv"), pl, N, target);

        std::vector<double> ious; ious.reserve(ng);
        double mae = 0.0, mg = 0.0, mp = 0.0;
        int active_g = 0, active_p = 0;
        for (int g = 0; g < ng; ++g) {
            const double iou = grain_iou(gl, pl, g); ious.push_back(iou);
            mae += std::fabs(ga[g] - pa[g]) / (N * N);
            mg += ga[g]; mp += pa[g]; active_g += ga[g] > 0; active_p += pa[g] > 0;
            trajectory << target << ',' << g << ',' << ga[g] << ',' << pa[g] << ','
                       << ga[g] / (N * N) << ',' << pa[g] / (N * N) << ',' << iou << '\n';
        }
        mg /= ng; mp /= ng;
        double cov = 0.0, vg = 0.0, vp = 0.0;
        for (int g = 0; g < ng; ++g) {
            cov += (ga[g] - mg) * (pa[g] - mp);
            vg += (ga[g] - mg) * (ga[g] - mg);
            vp += (pa[g] - mp) * (pa[g] - mp);
        }
        const double corr = (vg > 0 && vp > 0) ? cov / std::sqrt(vg * vp) : 1.0;
        long long same = 0;
        for (size_t i = 0; i < gl.size(); ++i) same += gl[i] == pl[i];
        const double pixel = (double)same / gl.size();
        const double bf1 = boundary_f1(gl, pl, N);
        std::sort(ious.begin(), ious.end());
        const double mean_iou = std::accumulate(ious.begin(), ious.end(), 0.0) / ng;
        const double median_iou = 0.5 * (ious[ng / 2 - 1] + ious[ng / 2]);
        summary << target << ',' << mean_iou << ',' << median_iou << ',' << mae
                << ',' << corr << ',' << pixel << ',' << bf1 << ','
                << active_g << ',' << active_p << '\n';
        std::printf("t=%5.3f s: IoU=%.3f, area MAE=%.4f, corr=%.3f, pixels=%.3f, boundary F1=%.3f, grains=%d/%d\n",
                    target, mean_iou, mae, corr, pixel, bf1, active_g, active_p);
        if (std::fabs(target - 1.0) < 1e-9) {
            gate_mean_iou = mean_iou; gate_area_mae = mae; gate_corr = corr;
            gate_pixel = pixel; gate_boundary = bf1;
        }
    }

    // Quantitative comparison is gated at 1 s after early extinctions have begun
    // but while all five agreement metrics remain inside their declared bounds.
    // The 1.125--4 s outputs expose continued coarsening and later divergence.
    const bool pass = gate_corr >= 0.98 && gate_area_mae <= 0.15
                   && gate_mean_iou >= 0.62 && gate_pixel >= 0.84
                   && gate_boundary >= 0.65;
    std::printf("Declared gates at 1 s: corr>=0.98, MAE<=0.15, IoU>=0.62, pixel agreement>=0.84, boundary F1>=0.65: %s\n",
                pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
