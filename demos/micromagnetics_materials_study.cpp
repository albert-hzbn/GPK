/*
 * micromagnetics_materials_study.cpp — multi-material comparison using the
 * optional sub-grid domain-wall extension (include/gpk_micromagnetics.hpp).
 * Companion to the single-material validation demo
 * demos/micromagnetics_hysteresis.cpp (which checks the physics against a
 * closed-form target); this demo instead runs several real materials through
 * the SAME fixed microstructure and compares against literature coercivity.
 *
 * Materials (see MaterialSpec table below for sourcing):
 *   - Nd2Fe14B  : hard, uniaxial, NUCLEATION-type hard magnet
 *   - SmCo5     : hard, uniaxial, PINNING-type hard magnet (deliberately a
 *                 model-mismatch case -- this module only implements
 *                 nucleation-type/Stoner-Wohlfarth switching,
 *                 so it is EXPECTED to disagree with SmCo5's real mechanism)
 *   - Fe-Ga (Galfenol): soft magnetostrictive material; K1/Ms/A taken
 *                 directly from Yi & Xu 2014 [37] (already used elsewhere in
 *                 this project) -- the only material here with a directly
 *                 traceable source rather than recalled literature ranges
 *
 * All three are tessellated on the SAME grid/grain layout (same seed), so any
 * difference in single-/multi-domain fraction or hysteresis-loop shape is due
 * ONLY to material parameters, not microstructure.
 *
 * Honesty note (see this file's printed output):
 * the model computes alpha_K=1 (defect-free, Kronmuller factor at its
 * maximum) unless told otherwise, i.e. it reports the THEORETICAL, intrinsic
 * nucleation field H_K -- an upper bound real magnets fall well short of due
 * to microstructural defects (the reduction is exactly what the nucleation model's
 * alpha_K parameter is for, and exactly the open problem in hard-magnet
 * physics sometimes called "Brown's paradox": real coercivities are commonly
 * only 20-40% of the theoretical H_K). This demo reports both the model's
 * intrinsic H_K and representative literature achieved-Hc ranges side by
 * side specifically so that gap is visible, not hidden.
 *
 * Writes per-material CSVs to build/output/micromagnetics_materials/<name>/:
 *   hysteresis.csv   (H_kA_per_m, M_over_Ms)
 *   grains.csv       (grain_id, R_grain_nm, n_domains, cx, cy -- for the
 *                     microstructure/domain-map plot)
 * and a combined summary.csv across all materials.
 *
 * Usage: ./micromagnetics_materials_study
 */

#include "gpk_core.hpp"
#include "gpk_thermo.hpp"
#include "gpk_micromagnetics.hpp"
#include "gpk_io.hpp"   // mkdir_p

#include <cstdio>
#include <cmath>
#include <climits>
#include <random>
#include <string>
#include <vector>

using namespace gpk;

struct MaterialSpec {
    const char* name;
    float Ku;        // [J/m^3]
    float Ms;        // [A/m]
    float A_exch;     // [J/m]
    float Hc_lit_lo;  // [kA/m] representative literature ACHIEVED coercivity, low end
    float Hc_lit_hi;  // [kA/m] representative literature ACHIEVED coercivity, high end
    float alpha_K;    // Kronmuller microstructural quality factor,
                      // INDEPENDENTLY sourced from literature-reported alpha ranges for
                      // the mechanism in question, NOT back-solved from Hc_lit above --
                      // see alpha_K_source for the reasoning behind each value.
    const char* alpha_K_source;
    const char* mechanism;
    const char* source;
};

static const MaterialSpec MATERIALS[] = {
    // Ku [J/m^3], Ms [A/m], A [J/m], Hc_lit range [kA/m], alpha_K, alpha_K_source, mechanism, source
    { "Nd2Fe14B", 4.3e6f, 1.28e6f, 7.7e-12f,  900.f, 2000.f,
      0.32f,
      "typical reported Kronmuller alpha for well-processed sintered Nd-Fe-B "
      "(commonly cited range ~0.2-0.4; recalled, not re-verified this session)",
      "nucleation-type",
      "representative sintered-magnet literature range (recalled, not re-verified this session)" },
    { "SmCo5",    17.0e6f, 0.90e6f, 12.0e-12f, 1500.f, 2500.f,
      0.095f,
      "within the commonly reported pinning-coefficient range ~0.02-0.15 for cellular-"
      "precipitate Sm-Co-type magnets, well below nucleation-type alpha (recalled, not "
      "re-verified this session) -- SAME functional form as the Kronmuller relation, but here alpha stands "
      "in for pinning strength at precipitate boundaries, not nucleation-site quality",
      "PINNING-type (mechanism mismatch: this module only models nucleation-type switching)",
      "representative literature range (recalled, not re-verified this session)" },
    { "FeGa_Galfenol", 2.0e4f, 1.432e6f, 1.0e-11f, 0.5f, 10.f,
      1.00f,
      "no calibration applicable -- see printed note (0% single-domain at this grain size)",
      "soft magnetostrictive (K1 used as effective uniaxial Ku; true anisotropy is cubic)",
      "Yi & Xu 2014: K1=2e4 J/m^3, Ms=1.432e6 A/m, Ae~=1e-11 J/m" },
};
static const int N_MATERIALS = sizeof(MATERIALS) / sizeof(MATERIALS[0]);

int main()
{
    const int   Ny = 250, Nx = 250;
    const float dx = 2.0e-9f;      // 2 nm cells, 500x500 nm domain
    const int   N_GRAINS = 175;    // avg grain radius ~21 nm -- straddles the
                                    // R_sd of Nd2Fe14B/SmCo5 differently, while
                                    // being far above Fe-Ga's (~1.6 nm)
    const uint32_t seed = 11;

    auto thermo = std::make_shared<IsotropicGrowthThermo>(1e-8f, 0.324f);
    GPKGrid grid(1, Ny, Nx, dx, thermo, 0.f);

    // Same tessellation + same easy-axis orientations for every material.
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

    const std::string outroot = "output/micromagnetics_materials";
    mkdir_p(outroot);
    FILE* summary = std::fopen((outroot + "/summary.csv").c_str(), "w");
    std::fprintf(summary,
        "material,Ku_J_per_m3,Ms_A_per_m,A_J_per_m,R_sd_nm,n_grains,n_single,n_multi,"
        "frac_single,H_K_kA_per_m,Mr_over_Ms,"
        "Hc_intrinsic_kA_per_m,alpha_K_used,Hc_calibrated_kA_per_m,"
        "Hc_lit_lo_kA_per_m,Hc_lit_hi_kA_per_m,ratio_intrinsic,ratio_calibrated,mechanism\n");

    printf("=== GPK Micromagnetic Materials Study ===\n");
    printf("Shared microstructure: %d x %d @ dx=%.1f nm, %d grains, avg radius ~%.1f nm\n\n",
           Ny, Nx, dx*1e9, N_GRAINS, std::sqrt((double)Ny*Nx/N_GRAINS/M_PI)*dx*1e9);

    // Runs a full descending+ascending hysteresis sweep at a given alpha_K
    // and returns (Mr, Hc); optionally writes the loop to a CSV.
    auto run_sweep = [&](MaterialSpec const& mat, float alpha_K_val, float theta_H,
                        const std::string& csv_path, float& Mr_out, float& Hc_out) {
        MicromagneticExtension mm(mat.Ku, mat.Ms, mat.A_exch, seed);
        mm.alpha0 = alpha_K_val;
        mm.alpha_damping = 0.8f; // strong damping -> fast convergence to equilibrium per field step
        mm.init_from_grid(grid);
        mm.saturate_along(theta_H);

        const float H_K   = 2.f * mat.Ku / (MicromagneticExtension::MU0 * mat.Ms);
        // Scale the sweep range by alpha_K_val too: the switching fields that
        // actually matter live near alpha_K_val*H_K, not the bare H_K. Using a
        // fixed H_max=1.5*H_K regardless of alpha_K would make the (fixed
        // step count) sweep resolve the transition region increasingly
        // coarsely as alpha_K shrinks -- a quantization artifact, not physics.
        const float H_max = 1.5f * std::max(alpha_K_val, 0.05f) * H_K;
        const int   n_steps = 200;

        FILE* fh = csv_path.empty() ? nullptr : std::fopen(csv_path.c_str(), "w");
        if (fh) std::fprintf(fh, "H_kA_per_m,M_over_Ms\n");

        std::vector<float> H_hist, M_hist;
        for (int k = 0; k <= n_steps; ++k) {
            float H = H_max * (1.f - 2.f * k / n_steps);
            mm.relax_to_equilibrium(H, theta_H, 0.f, 2.0e-14f, 300);
            float M = mm.mean_reduced_magnetization(theta_H);
            H_hist.push_back(H); M_hist.push_back(M);
            if (fh) std::fprintf(fh, "%.4f,%.5f\n", H/1e3, M);
        }
        const size_t n_descending = H_hist.size();
        for (int k = 0; k <= n_steps; ++k) {
            float H = -H_max * (1.f - 2.f * k / n_steps);
            mm.relax_to_equilibrium(H, theta_H, 0.f, 2.0e-14f, 300);
            float M = mm.mean_reduced_magnetization(theta_H);
            H_hist.push_back(H); M_hist.push_back(M);
            if (fh) std::fprintf(fh, "%.4f,%.5f\n", H/1e3, M);
        }
        if (fh) std::fclose(fh);

        size_t i0 = 0; float best = 1e30f;
        for (size_t k = 0; k < n_descending; ++k)
            if (std::fabs(H_hist[k]) < best) { best = std::fabs(H_hist[k]); i0 = k; }
        Mr_out = M_hist[i0];

        Hc_out = 0.f;
        for (size_t k = 1; k < n_descending; ++k) {
            if (M_hist[k-1] > 0.f && M_hist[k] <= 0.f) {
                float t = M_hist[k-1] / (M_hist[k-1] - M_hist[k]);
                Hc_out = std::fabs(H_hist[k-1] + t * (H_hist[k] - H_hist[k-1]));
                break;
            }
        }
    };

    for (int m = 0; m < N_MATERIALS; ++m) {
        const MaterialSpec& mat = MATERIALS[m];
        const float theta_H = 0.f;
        const std::string outdir = outroot + "/" + mat.name;
        mkdir_p(outdir);

        // Pass 1: alpha_K=1 -- intrinsic, defect-free upper bound.
        float Mr_intrinsic, Hc_intrinsic;
        run_sweep(mat, 1.0f, theta_H, outdir + "/hysteresis_intrinsic.csv", Mr_intrinsic, Hc_intrinsic);

        // Pass 2: alpha_K = literature-informed value (independently sourced,
        // see MaterialSpec::alpha_K_source -- NOT back-solved from Hc_lit).
        float Mr_cal, Hc_cal;
        run_sweep(mat, mat.alpha_K, theta_H, outdir + "/hysteresis_calibrated.csv", Mr_cal, Hc_cal);
        // Also keep the plotting script's expected default filename pointing
        // at the calibrated (physically meaningful) loop.
        run_sweep(mat, mat.alpha_K, theta_H, outdir + "/hysteresis.csv", Mr_cal, Hc_cal);

        // Domain-map bookkeeping (independent of alpha_K -- gating only uses R_sd).
        MicromagneticExtension mm_map(mat.Ku, mat.Ms, mat.A_exch, seed);
        mm_map.init_from_grid(grid);
        int n_single = 0, n_multi = 0;
        const float rsd = mm_map.R_sd();
        for (auto& kv : mm_map.states())
            (kv.second.n_domains == 1) ? ++n_single : ++n_multi;
        const float frac_single = (float)n_single / (float)(n_single + n_multi);
        const float H_K = 2.f * mat.Ku / (MicromagneticExtension::MU0 * mat.Ms);

        FILE* fg = std::fopen((outdir + "/grains.csv").c_str(), "w");
        std::fprintf(fg, "grain_id,R_grain_nm,n_domains,easy_axis_deg\n");
        for (auto& kv : mm_map.states())
            std::fprintf(fg, "%d,%.3f,%d,%.3f\n", kv.first, kv.second.R_grain*1e9,
                         kv.second.n_domains, kv.second.g_easy * 180.0 / M_PI);
        std::fclose(fg);

        const float lit_mid = 0.5f*(mat.Hc_lit_lo + mat.Hc_lit_hi);
        const float ratio_intrinsic  = (lit_mid > 0.f) ? (Hc_intrinsic/1e3) / lit_mid : 0.f;
        const float ratio_calibrated = (lit_mid > 0.f) ? (Hc_cal/1e3)       / lit_mid : 0.f;

        printf("--- %s (%s) ---\n", mat.name, mat.mechanism);
        printf("  Source: %s\n", mat.source);
        printf("  Ku=%.3e J/m^3  Ms=%.3e A/m  A=%.3e J/m\n", (double)mat.Ku, (double)mat.Ms, (double)mat.A_exch);
        printf("  R_sd = %.2f nm  ->  %d/%d grains single-domain (%.1f%%)\n",
               rsd*1e9, n_single, n_single+n_multi, frac_single*100.0);
        printf("  H_K (intrinsic anisotropy field) = %.1f kA/m (mu0*H_K = %.3f T)\n",
               H_K/1e3, MicromagneticExtension::MU0*H_K);
        printf("  Hc(alpha_K=1, defect-free)  = %.1f kA/m   ratio vs lit. midpoint = %.2fx\n",
               Hc_intrinsic/1e3, ratio_intrinsic);
        printf("  Hc(alpha_K=%.3f, calibrated) = %.1f kA/m   ratio vs lit. midpoint = %.2fx\n",
               (double)mat.alpha_K, Hc_cal/1e3, ratio_calibrated);
        printf("  alpha_K source: %s\n", mat.alpha_K_source);
        printf("  Hc (literature, achieved) ~ %.0f-%.0f kA/m\n\n", mat.Hc_lit_lo, mat.Hc_lit_hi);

        std::fprintf(summary, "%s,%.4e,%.4e,%.4e,%.3f,%d,%d,%d,%.4f,%.2f,%.4f,%.4f,%.3f,%.4f,%.1f,%.1f,%.3f,%.3f,%s\n",
                     mat.name, mat.Ku, mat.Ms, mat.A_exch, rsd*1e9, n_single+n_multi,
                     n_single, n_multi, frac_single, H_K/1e3, Mr_cal,
                     Hc_intrinsic/1e3, mat.alpha_K, Hc_cal/1e3,
                     mat.Hc_lit_lo, mat.Hc_lit_hi, ratio_intrinsic, ratio_calibrated, mat.mechanism);
    }
    std::fclose(summary);

    // Dump the shared grain-id/orientation map once (same for every material).
    FILE* fmap = std::fopen((outroot + "/microstructure.csv").c_str(), "w");
    std::fprintf(fmap, "# Ny=%d Nx=%d dx_nm=%.4f\n", Ny, Nx, dx*1e9);
    std::fprintf(fmap, "x_nm,y_nm,grain_id,easy_axis_deg\n");
    for (int y = 0; y < Ny; ++y)
        for (int x = 0; x < Nx; ++x) {
            int i = grid.idx(0, y, x);
            std::fprintf(fmap, "%.3f,%.3f,%d,%.3f\n", x*dx*1e9, y*dx*1e9, grid.phi[i],
                         grid.g[i] * 180.0 / M_PI);
        }
    std::fclose(fmap);

    printf("Wrote per-material hysteresis loops + domain maps to %s/\n", outroot.c_str());
    return 0;
}
