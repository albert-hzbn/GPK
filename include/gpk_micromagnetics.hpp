#pragma once
/*
 * gpk_micromagnetics.hpp — Optional sub-grid domain-wall extension (3-D)
 *
 * A per-grain (and, for grains above the
 * single-domain radius, per-sub-domain) macrospin reduction with GENUINE
 * Landau-Lifshitz-Gilbert dynamics (precession + damping on a continuous
 * unit vector, not a discrete two-state flip), uniaxial OR cubic
 * anisotropy, a mechanistically distinct pinning-type coercivity model
 * alongside the original nucleation-type (Stoner-Wohlfarth) one, an
 * approximate long-range dipolar grain-grain coupling, and Curie-
 * temperature Ms(T).
 *
 * THIS IS A SEPARATE, OPT-IN MODULE — it is NOT included by gpk_core.hpp or
 * gpk_thermo.hpp, is not touched by GPKGrid::step(), and costs nothing
 * unless a caller explicitly constructs a MicromagneticExtension and calls
 * its methods.
 *
 * What changed from the first version of this module, and why: the first version reduced each
 * grain to a DISCRETE two-state macrospin (s=+-1, jumping instantaneously
 * between two fixed positions on the easy axis) with no field-driven
 * canting, no true dynamics, uniaxial-only anisotropy, and unconditionally
 * EXCLUDED any grain above the single-domain radius. Each of those was a
 * real, named limitation, not a rounding error, and each is addressed here:
 *   - continuous m(t)
 *   - genuine LLG precession+damping
 *   - cubic anisotropy
 *   - multi-domain sub-spin partition (no more blanket exclusion)
 *   - approximate long-range dipolar coupling
 *   - mechanistically distinct pinning-type coercivity
 *   - Ms(T)
 *
 * What is STILL not here, deliberately, because it cannot be done within a
 * per-grain/per-sub-spin reduction at all:
 *   - no resolved spatial magnetization TEXTURE within a sub-domain (a
 *     sub-spin is still a single macrospin, not a field) -- vortices,
 *     domain-wall profiles, and Néel/Bloch wall character remain out of
 *     reach; only aggregate multi-domain-like reversal statistics are
 *     approximated;
 *   - no spin-transfer-torque/current-driven domain-wall dynamics -- this
 *     needs the wall's internal structure as the object being moved, which
 *     does not exist in this framework at any level of refinement;
 *   - the Villari-effect multi-variant magnetostriction extension remains
 *     unimplemented (still no known simple closed form for the general
 *     case) -- though the cubic-anisotropy multi-well landscape added here
 *     is the machinery a future implementation would build on.
 */

#include "gpk_core.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <unordered_map>
#include <vector>

namespace gpk {

// =========================================================================
// Minimal 3-vector helper (kept local to this header -- no new dependency)
// =========================================================================
struct Vec3 {
    float x = 0.f, y = 0.f, z = 0.f;
    Vec3() = default;
    Vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
    Vec3 operator+(const Vec3& o) const { return {x+o.x, y+o.y, z+o.z}; }
    Vec3 operator-(const Vec3& o) const { return {x-o.x, y-o.y, z-o.z}; }
    Vec3 operator*(float s) const { return {x*s, y*s, z*s}; }
    float dot(const Vec3& o) const { return x*o.x + y*o.y + z*o.z; }
    Vec3 cross(const Vec3& o) const { return {y*o.z - z*o.y, z*o.x - x*o.z, x*o.y - y*o.x}; }
    float norm() const { return std::sqrt(std::max(dot(*this), 0.f)); }
    Vec3 normalized() const { float n = norm(); return n > 1e-12f ? (*this)*(1.f/n) : Vec3(0,0,1); }
};

enum class AnisotropyType { UNIAXIAL, CUBIC };
enum class CoercivityMechanism { NUCLEATION, PINNING };

// =========================================================================
// SubSpin — one macrospin. A single-domain grain owns exactly one; a
// multi-domain grain (R_grain > R_sd) owns several, each representing a
// fraction of the grain's volume.
// =========================================================================
struct SubSpin {
    Vec3  m{0.f, 0.f, 1.f};   // continuous unit magnetization vector (lab frame)
    float weight = 1.f;        // volume fraction of the grain this sub-spin represents
};

struct GrainMagneticState {
    std::vector<SubSpin> spins;
    float g_easy   = 0.f;      // easy-axis angle [rad] (== grid.g of the grain)
    float R_grain  = 0.f;      // equivalent grain radius [m]
    int   n_domains = 1;
    float alpha_K  = 1.f;      // Kronmuller quality factor (NUCLEATION mechanism)
    Vec3  centroid{0.f, 0.f, 0.f}; // grain center [m], for dipolar coupling
};

// =========================================================================
// MicromagneticExtension
// =========================================================================
class MicromagneticExtension {
public:
    // ---- anisotropy ----
    AnisotropyType anisotropy = AnisotropyType::UNIAXIAL;
    float Ku = 0.f;   // [J/m^3] uniaxial constant (UNIAXIAL)
    float K1 = 0.f, K2 = 0.f;  // [J/m^3] cubic constants (CUBIC)

    // ---- magnetization / Curie temperature ----
    float Ms0 = 0.f;     // [A/m] saturation magnetization at T=0
    float Tc  = 0.f;      // [K] Curie temperature; Tc<=0 => Ms(T)=Ms0 always
    float ms_beta = 0.5f; // mean-field/Bloch-law exponent

    float A_exch = 0.f;   // [J/m] exchange stiffness

    // ---- coercivity mechanism ----
    CoercivityMechanism mechanism = CoercivityMechanism::NUCLEATION;
    // NUCLEATION: Kronmuller quality factor.
    float alpha0   = 1.f;
    float k_defect = 0.f;
    // PINNING: Kersten-type pinning field set by a
    // microstructural pinning-site spacing d_pin, NOT a fraction of H_K --
    // a mechanistically distinct formula, not a relabeled alpha_K.
    float d_pin = 1.0e-9f;   // [m] characteristic pinning-site spacing
    float k_pin = 1.0f;      // O(1) geometric factor

    // ---- LLG dynamics ----
    float gamma_e        = 1.76e11f; // [rad/(s.T)] electron gyromagnetic ratio
    float alpha_damping   = 0.5f;     // Gilbert damping

    // ---- legacy thermally-activated switching ----
    float f0         = 1.0e10f;
    float sharrock_n = 1.5f;

    // ---- domain-count (Kittel 1-D domain-width estimate) ----
    float domain_wall_prefactor = 1.0f; // O(1) uncertainty factor, flagged honestly

    // ---- dipolar grain-grain coupling ----
    float dipolar_cutoff = 0.f; // [m]; <=0 => disabled (default off, matches old behavior)

    bool couple_to_driving_force = false;

    static constexpr float MU0 = 4.0e-7f * 3.14159265358979323846f;
    static constexpr float KB  = 1.380649e-23f;

    MicromagneticExtension(float Ku_val, float Ms_val, float A_exch_val, uint32_t seed = 1234)
        : Ku(Ku_val), Ms0(Ms_val), A_exch(A_exch_val), rng_(seed) {}

    // ---- Curie-temperature-dependent saturation magnetization ----
    float Ms(float T) const {
        if (Tc <= 0.f || T <= 0.f) return Ms0;
        if (T >= Tc) return 0.f;
        return Ms0 * std::pow(1.f - T/Tc, ms_beta);
    }

    float effective_K() const {
        return (anisotropy == AnisotropyType::CUBIC) ? std::fabs(K1) : Ku;
    }

    // Domain-wall energy density sigma_w = 4*sqrt(A*K) (standard 1-D Ising-
    // wall estimate; same expression already implicit in R_sd below).
    float wall_energy() const { return 4.f * std::sqrt(std::max(A_exch*effective_K(), 0.f)); }

    // Kittel single-domain radius.
    float R_sd() const {
        float Ms_ = std::max(Ms0, 1e-9f);
        return 9.f * std::sqrt(A_exch * effective_K()) / (MU0 * Ms_ * Ms_);
    }

    // Kittel 1-D domain width estimate for a grain of radius R_grain: d ~
    // sqrt(sigma_w*R/(mu0*Ms^2)) (domain width grows as sqrt(grain size) --
    // standard classical domain theory, prefactor uncertain by an O(1)
    // factor depending on grain shape, hence domain_wall_prefactor).
    float domain_width(float R_grain) const {
        float Ms_ = std::max(Ms0, 1e-9f);
        return domain_wall_prefactor * std::sqrt(wall_energy() * R_grain / (MU0 * Ms_ * Ms_));
    }

    // n_domains from the ratio of grain size to Kittel domain
    // width, replacing the previous ad hoc (R/R_sd)^p scaling with p
    // unconstrained. Still an order-of-magnitude estimate (flagged
    // honestly), but now tied to an actual named classical result.
    int domain_count(float R_grain) const {
        float rsd = R_sd();
        if (R_grain < rsd) return 1;
        float dw = domain_width(R_grain);
        return std::max(1, (int)std::ceil(R_grain / std::max(dw, 1e-12f)));
    }

    float H_K() const { return 2.f * effective_K() / (MU0 * std::max(Ms0, 1e-9f)); }

    // ---- Coercivity/switching field (mechanism-dependent) ----
    // NUCLEATION: Stoner-Wohlfarth astroid, reduced by alpha_K.
    // PINNING: Kersten-type estimate H_pin = k_pin *
    // sigma_w/(2*mu0*Ms*d_pin) -- depends on a MICROSTRUCTURAL length scale
    // (pinning-site spacing), not a fraction of H_K, so it is mechanistically
    // distinct from the nucleation formula rather than the same formula
    // with a smaller fitted prefactor.
    float H_switch(float theta, float alpha_K_val) const {
        if (mechanism == CoercivityMechanism::PINNING) {
            float Ms_ = std::max(Ms0, 1e-9f);
            return k_pin * wall_energy() / (2.f * MU0 * Ms_ * std::max(d_pin, 1e-12f));
        }
        const float HK = H_K();
        float ct = std::pow(std::fabs(std::cos(theta)), 2.f / 3.f);
        float st = std::pow(std::fabs(std::sin(theta)), 2.f / 3.f);
        float denom = std::pow(ct + st, 1.5f);
        return alpha_K_val * HK / std::max(denom, 1e-6f);
    }

    // ---- Anisotropy energy density for a unit vector m ----
    // UNIAXIAL: E = -Ku*(m.easy)^2, easy axis in-plane at angle g_easy.
    // CUBIC: standard E = K1*(a1^2 a2^2 + a2^2 a3^2 + a3^2 a1^2) + K2*a1^2 a2^2 a3^2,
    // with (a1,a2,a3) = crystal-frame direction cosines of m, obtained by
    // rotating the lab-frame m by -g_easy about z (same rotation convention
    // already used for 3-D cubic INTERFACIAL energy elsewhere in GPK).
    // alpha_scale (default 1) applies the Kronmuller microstructural quality
    // factor (per-grain st.alpha_K) DIRECTLY to the
    // anisotropy energy the continuous dynamics sees -- physically, defects
    // reduce the effective local anisotropy barrier a real magnetization
    // encounters, not just a separate legacy threshold check. Earlier in
    // this round alpha_K was threaded through H_switch() (used only by the
    // legacy discrete/thermal path) but NOT through here, so the continuous
    // LLG dynamics silently ignored it entirely -- calibrated alpha_K<1 runs
    // used the same effective anisotropy as alpha_K=1, and the (narrower,
    // alpha_K-scaled) field sweep range then never reached the field
    // actually needed to switch, reporting a spurious Hc=0. Found via the
    // multi-material study regressing after this refactor.
    float anisotropy_energy(const Vec3& m, float g_easy, float alpha_scale = 1.f) const {
        if (anisotropy == AnisotropyType::UNIAXIAL) {
            float c = m.x*std::cos(g_easy) + m.y*std::sin(g_easy);
            return -alpha_scale * Ku * c * c;
        }
        const float cg = std::cos(g_easy), sg = std::sin(g_easy);
        const float a1 =  m.x*cg + m.y*sg;
        const float a2 = -m.x*sg + m.y*cg;
        const float a3 =  m.z;
        const float a1a2 = a1*a1*a2*a2, a2a3 = a2*a2*a3*a3, a3a1 = a3*a3*a1*a1;
        return alpha_scale * (K1*(a1a2 + a2a3 + a3a1) + K2*(a1*a1*a2*a2*a3*a3));
    }

    // Effective field from anisotropy: h_eff = -(1/(mu0*Ms)) dE/dm.
    // UNIAXIAL has a clean analytic gradient; CUBIC uses a robust numerical
    // tangent-plane gradient (finite difference) rather than a hand-derived
    // analytic one, to avoid a transcription error in a less-common formula.
    Vec3 anisotropy_field(const Vec3& m, float g_easy, float Ms_val, float alpha_scale = 1.f) const {
        const float invMuMs = 1.f / (MU0 * std::max(Ms_val, 1e-9f));
        if (anisotropy == AnisotropyType::UNIAXIAL) {
            Vec3 easy(std::cos(g_easy), std::sin(g_easy), 0.f);
            float c = m.dot(easy);
            Vec3 dEdm = easy * (-2.f * alpha_scale * Ku * c);
            return dEdm * (-invMuMs);
        }
        // Numerical gradient in the tangent plane at m.
        Vec3 ref = (std::fabs(m.z) < 0.9f) ? Vec3(0,0,1) : Vec3(1,0,0);
        Vec3 t1 = m.cross(ref).normalized();
        Vec3 t2 = m.cross(t1).normalized();
        const float h = 1.0e-3f;
        float E0 = anisotropy_energy(m, g_easy, alpha_scale);
        float E1 = anisotropy_energy((m + t1*h).normalized(), g_easy, alpha_scale);
        float E2 = anisotropy_energy((m + t2*h).normalized(), g_easy, alpha_scale);
        Vec3 dEdm = t1*((E1-E0)/h) + t2*((E2-E0)/h);
        return dEdm * (-invMuMs);
    }

    // ---- Continuous cubic magnetostriction (a working
    // Villari-effect coupling, made tractable by the continuous m this
    // version tracks). The FIRST version of this module (and the base
    // E_mag(g)/eigenstrain machinery in gpk_core.hpp) assumed magnetization
    // sits fixed at the easy axis, so a Villari coupling needed a discrete
    // multi-variant switching machinery that did not exist. With a genuine
    // continuous m (now evolved by real LLG dynamics), the magnetostriction
    // ALONG the current magnetization direction is just the standard
    // polycrystalline cubic formula (e.g. Chikazumi, "Physics of
    // Ferromagnetism"), evaluated at whatever direction m currently is --
    // no discrete variant list needed:
    //   lambda(m) = lambda100 + 3*(lambda111-lambda100)*(a1^2 a2^2 + a2^2 a3^2 + a3^2 a1^2)
    // with (a1,a2,a3) the crystal-frame direction cosines of m (same
    // rotation-by-g_easy convention as anisotropy_energy). Self-consistency:
    // this reduces to exactly lambda100 at <100> and lambda111 at <111> --
    // checked directly in validation.
    // As m rotates under an applied field (via step_dynamics/
    // relax_to_equilibrium), lambda(m) traces out a REAL, continuous strain
    // response -- a genuine Villari-effect signature, not a discrete jump
    // between two fixed eigenstrain values. This is a magnetostriction
    // FORMULA, not itself coupled into gpk_core's elastic-coupling pipeline
    // -- a caller wanting that coupling would feed this value in
    // as ThermoModel::eigenstrain_coeff-like input, analogous to how Sec
    // 11.8's static cos^2(g-theta_H) term is used today.
    float cubic_magnetostriction(const Vec3& m, float g_easy, float lambda100, float lambda111) const {
        const float cg = std::cos(g_easy), sg = std::sin(g_easy);
        const float a1 =  m.x*cg + m.y*sg;
        const float a2 = -m.x*sg + m.y*cg;
        const float a3 =  m.z;
        const float sum_sq = a1*a1*a2*a2 + a2*a2*a3*a3 + a3*a3*a1*a1;
        return lambda100 + 3.f*(lambda111 - lambda100)*sum_sq;
    }

    // ---- Zeeman field is just H_ext itself (already in A/m). ----

    // ---- Dipolar field at a grain from all others within dipolar_cutoff
    //: point-dipole approximation, mu_j = Ms*V_j*<m_j>,
    // standard dipole field formula, summed over neighbours within a cutoff
    // to keep cost bounded (O(N_grains x N_neighbours), not O(N_grains^2)
    // for large ensembles). Off by default (dipolar_cutoff<=0).
    Vec3 dipolar_field(int32_t self_id, float T_val) const {
        Vec3 h(0,0,0);
        if (dipolar_cutoff <= 0.f) return h;
        auto it_self = states_.find(self_id);
        if (it_self == states_.end()) return h;
        const Vec3 r0 = it_self->second.centroid;
        const float Ms_ = Ms(T_val);
        for (auto& kv : states_) {
            if (kv.first == self_id) continue;
            Vec3 r = kv.second.centroid - r0;
            float dist = r.norm();
            if (dist < 1e-12f || dist > dipolar_cutoff) continue;
            Vec3 rhat = r * (1.f/dist);
            // Grain volume from R_grain (sphere in 3-D, disk*unit-thickness in 2-D
            // is not separable here, so this is left as an amplitude the caller
            // can rescale via k_pin-like external calibration if needed --
            // documented as approximate).
            float Vj = (4.f/3.f) * (float)M_PI * kv.second.R_grain*kv.second.R_grain*kv.second.R_grain;
            Vec3 mj(0,0,0);
            for (auto& s : kv.second.spins) mj = mj + s.m * (s.weight * Vj);
            Vec3 mu_j = mj * Ms_;
            float r3 = dist*dist*dist;
            h = h + ( rhat*(3.f*mu_j.dot(rhat)) - mu_j ) * (1.f/(4.f*(float)M_PI*r3));
        }
        return h;
    }

    // ---- Build per-grain state from the grid (unchanged interface) ----
    void init_from_grid(const GPKGrid& grid) {
        states_.clear();
        std::unordered_map<int32_t, long> counts;
        std::unordered_map<int32_t, float> orient;
        std::unordered_map<int32_t, double> sx, sy, sz;
        const int N = grid.N();
        for (int i = 0; i < N; ++i) {
            int32_t p = grid.phi[i];
            counts[p]++;
            orient.emplace(p, grid.g[i]);
            int z,y,x; grid.decode(i, z, y, x);
            sx[p] += x*grid.dx; sy[p] += y*grid.dx; sz[p] += z*grid.dx;
        }
        const float rsd = R_sd();
        for (auto& kv : counts) {
            GrainMagneticState st;
            st.g_easy = orient[kv.first];
            st.R_grain = (grid.Nz > 1)
                ? std::cbrt(3.0 * kv.second / (4.0 * M_PI)) * grid.dx
                : std::sqrt((double)kv.second / M_PI) * grid.dx;
            st.n_domains = domain_count(st.R_grain);
            st.alpha_K = alpha0;
            st.centroid = Vec3((float)(sx[kv.first]/kv.second), (float)(sy[kv.first]/kv.second),
                               (float)(sz[kv.first]/kv.second));
            st.spins.resize(st.n_domains);
            for (int k = 0; k < st.n_domains; ++k) {
                st.spins[k].weight = 1.f / st.n_domains;
                st.spins[k].m = Vec3(std::cos(st.g_easy), std::sin(st.g_easy), 0.f);
            }
            states_[kv.first] = std::move(st);
        }
        (void)rsd;
    }

    void update_alpha_from_composition(const GPKGrid& grid) {
        std::unordered_map<int32_t, double> csum;
        std::unordered_map<int32_t, long> cnt;
        const int N = grid.N();
        for (int i = 0; i < N; ++i) {
            if (!grid.mask[i]) continue;
            csum[grid.phi[i]] += grid.c[i];
            cnt[grid.phi[i]]++;
        }
        for (auto& kv : states_) {
            auto it = cnt.find(kv.first);
            if (it == cnt.end() || it->second == 0) continue;
            double cbar = csum[kv.first] / (double)it->second;
            kv.second.alpha_K = std::clamp(
                alpha0 - k_defect / (float)std::max(cbar, 1e-6), 0.01f, 1.0f);
        }
    }

    // ---- Single explicit LLG (Gilbert form) step for one sub-spin, RK2 (Heun) ----
    // dm/dt = -gamma*(m x B_eff) - alpha*gamma*(m x (m x B_eff)),  B_eff = mu0*H_eff
    // Renormalizes |m|=1 after stepping (standard practice for explicit
    // integrators, which otherwise drift off the unit sphere).
    Vec3 llg_rhs(const Vec3& m, const Vec3& H_eff) const {
        Vec3 B = H_eff * MU0;
        Vec3 mxB = m.cross(B);
        Vec3 mxmxB = m.cross(mxB);
        return (mxB*(-gamma_e)) - (mxmxB*(gamma_e*alpha_damping));
    }

    void llg_step(Vec3& m, const Vec3& H_eff, float dt) const {
        Vec3 k1 = llg_rhs(m, H_eff);
        Vec3 mid = (m + k1*(0.5f*dt)).normalized();
        Vec3 k2 = llg_rhs(mid, H_eff);
        m = (m + k2*dt).normalized();
    }

    // ---- Advance all sub-spins by dt under an applied field vector H_ext_vec,
    // integrating genuine LLG dynamics, for grains/sub-spins
    // gated as NUCLEATION-mechanism candidates by the caller's field regime.
    // This REPLACES the old discrete flip with continuous relaxation --
    // repeated calls with the SAME external field converge to the local
    // energy minimum (quasi-static limit) as would a Stoner-Wohlfarth
    // equilibrium solve; a single large-alpha_damping call already takes
    // most of the way there for typical field/anisotropy ratios.
    void step_dynamics(float H_ext, float theta_H, float T, float dt, int n_substeps = 1) {
        Vec3 Hext_vec(H_ext*std::cos(theta_H), H_ext*std::sin(theta_H), 0.f);
        const float Ms_ = Ms(T);
        for (auto& kv : states_) {
            GrainMagneticState& st = kv.second;
            Vec3 Hdip = dipolar_field(kv.first, T);
            for (auto& sp : st.spins) {
                for (int sub = 0; sub < n_substeps; ++sub) {
                    Vec3 Han = anisotropy_field(sp.m, st.g_easy, Ms_, st.alpha_K);
                    Vec3 Heff = Han + Hext_vec + Hdip;
                    llg_step(sp.m, Heff, dt / n_substeps);
                }
            }
        }
    }

    // Relax all sub-spins to their local energy minimum under a fixed field
    // (quasi-static/hysteresis-loop use case): repeatedly integrate LLG
    // with strong effective damping until the maximum |dm/dt| falls below a
    // tolerance or max_iters is reached. This is what replaces the old
    // discrete "did H exceed H_sw" threshold check for the NUCLEATION
    // mechanism's continuous version, and is the quantity compared against
    // the analytic Stoner-Wohlfarth equilibrium in validation.
    int relax_to_equilibrium(float H_ext, float theta_H, float T,
                             float dt, int max_iters = 200, float tol = 1e-5f) {
        Vec3 Hext_vec(H_ext*std::cos(theta_H), H_ext*std::sin(theta_H), 0.f);
        const float Ms_ = Ms(T);
        int iters = 0;
        for (; iters < max_iters; ++iters) {
            float max_rate = 0.f;
            for (auto& kv : states_) {
                GrainMagneticState& st = kv.second;
                Vec3 Hdip = dipolar_field(kv.first, T);
                for (auto& sp : st.spins) {
                    Vec3 Han = anisotropy_field(sp.m, st.g_easy, Ms_, st.alpha_K);
                    Vec3 Heff = Han + Hext_vec + Hdip;
                    Vec3 rate = llg_rhs(sp.m, Heff);
                    max_rate = std::max(max_rate, rate.norm());
                    llg_step(sp.m, Heff, dt);
                }
            }
            if (max_rate < tol) break;
        }
        return iters;
    }

    // ---- Legacy Poisson-process thermally-activated switching, now
    // operating on whether a sub-spin is in the field-favoured half-space
    // rather than a discrete s variable (unchanged mechanism).
    int stochastic_switch_check(float H_ext, float theta_H, float T, float dt) {
        int n_flipped = 0;
        const float H_mag = std::fabs(H_ext);
        const float theta_H_eff = (H_ext >= 0.f) ? theta_H : (theta_H + (float)M_PI);
        for (auto& kv : states_) {
            GrainMagneticState& st = kv.second;
            const float theta = st.g_easy - theta_H_eff;
            const float Hsw = H_switch(theta, st.alpha_K);
            for (auto& sp : st.spins) {
                bool favorable = sp.m.dot(Vec3(std::cos(theta_H_eff), std::sin(theta_H_eff), 0.f)) >= 0.f;
                if (favorable || T <= 0.f) continue;
                if (H_mag >= Hsw) continue; // handled by deterministic relax already
                const float ratio = std::clamp(H_mag / std::max(Hsw, 1e-9f), 0.f, 0.999f);
                const float Vgrain = (4.f/3.f) * (float)M_PI * st.R_grain*st.R_grain*st.R_grain * sp.weight;
                const float barrier = effective_K() * Vgrain * std::pow(1.f - ratio, sharrock_n);
                const float rate = f0 * std::exp(-barrier / (KB * std::max(T, 1.f)));
                const float P = 1.f - std::exp(-rate * dt);
                if (uni_(rng_) < P) {
                    sp.m = (sp.m * (-1.f)); // flip through the easy axis
                    ++n_flipped;
                }
            }
        }
        return n_flipped;
    }

    // ---- Ensemble-averaged reduced magnetization along theta_H, in [-1,1],
    // now volume-weighted over sub-spins (single- or multi-domain grains
    // handled uniformly).
    float mean_reduced_magnetization(float theta_H) const {
        if (states_.empty()) return 0.f;
        Vec3 dir(std::cos(theta_H), std::sin(theta_H), 0.f);
        double sum = 0.0;
        for (auto& kv : states_)
            for (auto& sp : kv.second.spins)
                sum += sp.weight * sp.m.dot(dir);
        return (float)(sum / states_.size());
    }

    void demag_energy_density(const GPKGrid& grid, float N_shape,
                              std::vector<float>& E_demag_out,
                              float gaussian_sigma = 2.0f) const {
        const int N = grid.N();
        E_demag_out.assign(N, 0.f);
        std::vector<float> mu(N, 0.f), mu_smooth(N), tmp1(N), tmp2(N);
        for (int i = 0; i < N; ++i) {
            auto it = states_.find(grid.phi[i]);
            if (it == states_.end()) continue;
            Vec3 mbar(0,0,0);
            for (auto& sp : it->second.spins) mbar = mbar + sp.m*sp.weight;
            float phi_n = std::atan2(grid.ny_f[i], grid.nx_f[i]);
            mu[i] = mbar.x*std::cos(phi_n) + mbar.y*std::sin(phi_n);
        }
        GaussKernel K(gaussian_sigma);
        gaussian_filter_3d(mu.data(), mu_smooth.data(), tmp1.data(), tmp2.data(),
                          grid.Nz, grid.Ny, grid.Nx, K);
        const float Ms_ = Ms0;
        for (int i = 0; i < N; ++i)
            E_demag_out[i] = -0.5f * MU0 * Ms_ * Ms_ * N_shape * mu[i] * mu_smooth[i];
    }

    void saturate(float s_val = 1.f) {
        for (auto& kv : states_)
            for (auto& sp : kv.second.spins)
                sp.m = Vec3(std::cos(kv.second.g_easy), std::sin(kv.second.g_easy), 0.f) * s_val;
    }

    void saturate_along(float theta_H) {
        Vec3 dir(std::cos(theta_H), std::sin(theta_H), 0.f);
        for (auto& kv : states_) {
            Vec3 easy(std::cos(kv.second.g_easy), std::sin(kv.second.g_easy), 0.f);
            float s = (easy.dot(dir) >= 0.f) ? 1.f : -1.f;
            for (auto& sp : kv.second.spins) sp.m = easy * s;
        }
    }

    std::unordered_map<int32_t, GrainMagneticState>& states() { return states_; }
    const std::unordered_map<int32_t, GrainMagneticState>& states() const { return states_; }

private:
    std::unordered_map<int32_t, GrainMagneticState> states_;
    std::mt19937 rng_;
    std::uniform_real_distribution<float> uni_{0.f, 1.f};
};

} // namespace gpk
