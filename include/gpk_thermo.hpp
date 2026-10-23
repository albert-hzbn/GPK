#pragma once
/*
 * Thermodynamic model specialisations for GPK (3-D).
 * Each model overrides ThermoModel virtual functions.
 *
 * interfacial_energy / mobility now receive the full 3-D normal (nx, ny, nz).
 */
#include "gpk_core.hpp"
#include <cmath>
#include <stdexcept>
#include <vector>

namespace gpk {

// =========================================================================
// Isotropic grain growth — capillarity only, ΔG_chem = 0.
// =========================================================================
struct IsotropicGrowthThermo : ThermoModel {
    explicit IsotropicGrowthThermo(float M0_val, float sigma0_val = 0.5f) {
        M0     = M0_val;
        sigma0 = sigma0_val;
        eps4   = 0.f;
        grain_network = true;
    }
    float bulk_driving_force(int, int, float, float) const override { return 0.f; }
    float mobility(float, float, float, float, float, float) const override { return M0; }
};

// =========================================================================
// Anisotropic single-grain growth — 3-D cubic Wulff shape.
// σ = σ₀(1 + ε₄(nx⁴ + ny⁴ + nz⁴ − 3/5))
// =========================================================================
struct AnisotropicGrowthThermo : ThermoModel {
    explicit AnisotropicGrowthThermo(float M0_val, float sigma0_val,
                                     float eps4_val)
    {
        M0     = M0_val;
        sigma0 = sigma0_val;
        eps4   = eps4_val;
        grain_network = true;
    }
    float bulk_driving_force(int, int, float, float) const override { return 0.f; }
    float mobility(float, float, float, float, float, float) const override { return M0; }
    // interfacial_energy: inherits base-class crystal-frame rotation + cubic formula.
};

// =========================================================================
// Binary alloy solidification.
//   Phase 0 = liquid,  Phase 1 = solid.
//   4-fold anisotropy → dendritic morphology (via eps4).
// =========================================================================
struct SolidificationThermo : ThermoModel {
    float m_liq;    // liquidus slope [K / (mol frac)]
    float k_part;   // equilibrium partition coefficient
    float T_melt;   // melting point of pure solvent [K]
    float L;        // latent heat [J/m³]

    SolidificationThermo(float M0_val   = 5e-8f,
                         float sigma0_v = 0.2f,
                         float eps4_v   = 0.04f,
                         float m_liq_v  = -500.f,
                         float k_part_v = 0.3f,
                         float T_melt_v = 933.f,
                         float L_v      = 1e8f)
        : m_liq(m_liq_v), k_part(k_part_v), T_melt(T_melt_v), L(L_v) {
        M0     = M0_val;
        sigma0 = sigma0_v;
        eps4   = eps4_v;
    }

    float bulk_driving_force(int phi_parent, int phi_product,
                             float c_val, float T_val) const override {
        float T_liq = T_melt + m_liq * c_val;
        float dT    = T_liq - T_val;
        float dG    = (L / T_melt) * dT;   // > 0 when undercooled
        // liquid→solid: ΔG = G_solid − G_liquid = −L·ΔT/Tm < 0  (spontaneous)
        if (phi_parent == 0 && phi_product == 1) return -dG;
        // solid→liquid: ΔG = G_liquid − G_solid = +L·ΔT/Tm > 0  (opposed)
        // Gives v_n = M(+dG − σ/R): positive for R > R* (solid stable), negative for R < R* (melts)
        if (phi_parent == 1 && phi_product == 0) return +dG;
        return 0.f;
    }

    float mobility(float, float, float, float, float, float) const override { return M0; }
};

// =========================================================================
// Recrystallisation — stored dislocation energy drives grain growth.
// =========================================================================
struct RecrystallisationThermo : ThermoModel {
    float E_stored;

    explicit RecrystallisationThermo(float M0_val, float sigma0_v,
                                     float E_stored_v)
        : E_stored(E_stored_v) {
        M0     = M0_val;
        sigma0 = sigma0_v;
        eps4   = 0.f;
    }

    float bulk_driving_force(int phi_parent, int phi_product,
                             float, float) const override {
        // deformed→rex: ΔG = G_rex − G_deformed = −E_stored < 0  (spontaneous)
        if (phi_parent == 0 && phi_product >= 1) return -E_stored;
        // rex→deformed: ΔG = G_deformed − G_rex = +E_stored > 0  (opposed)
        // Gives v_n = M(+E_stored − σκ): positive for large rex grains → no capillary erosion
        if (phi_parent >= 1 && phi_product == 0) return +E_stored;
        return 0.f;
    }

    float mobility(float, float, float, float, float, float) const override { return M0; }
};

// =========================================================================
// Dislocation stored-energy driven boundary migration + recovery.
//
// Generalises RecrystallisationThermo from a single scalar E_stored to a full
// spatial dislocation-density field rho(x) carried by GPKGrid. The stored
// energy density follows the Taylor relation
//
//     E_rho = 0.5 * alpha_T * mu * b^2 * rho          [J/m^3]
//
// and enters the driving force through the field coupling in step_velocities:
//
//     Delta G_stored = E_rho(product) - E_rho(parent) = disloc_coeff*(rho_q - rho_p),
//
// so a boundary migrates toward the higher-rho (more deformed) side. A moving
// front leaves recrystallised material at density rho_rex behind it (sweeping),
// and static recovery relaxes rho everywhere via d(rho)/dt = -recovery_rate*rho.
//
// A stored-energy nucleus grows only above the critical radius R* = sigma0/E_rho
// (Gibbs-Thomson balance E_rho = sigma*kappa), exactly analogous to the
// solidification / precipitation critical-radius tests.
// =========================================================================
struct DislocationThermo : ThermoModel {
    float mu, b, alphaT;   // shear modulus [Pa], Burgers vector [m], Taylor factor

    DislocationThermo(float M0_val, float sigma0_val,
                      float mu_val, float b_val, float alphaT_val = 0.5f,
                      float recovery_rate_val = 0.f, float rho_rex_val = 0.f)
        : mu(mu_val), b(b_val), alphaT(alphaT_val)
    {
        M0     = M0_val;
        sigma0 = sigma0_val;
        eps4   = 0.f;
        dislocation_enabled = true;
        disloc_coeff  = 0.5f * alphaT * mu * b * b;   // E_rho = disloc_coeff * rho
        recovery_rate = recovery_rate_val;
        rho_rex       = rho_rex_val;
    }

    // Stored energy is applied via the rho-field coupling in step_velocities,
    // so there is no separate chemical driving force here.
    float bulk_driving_force(int, int, float, float) const override { return 0.f; }

    // Constant intrinsic mobility. (Read-Shockley mobility would return zero for
    // equal orientations and freeze a recrystallisation front; a real ReX front
    // is a high-angle boundary that migrates.)
    float mobility(float, float, float, float, float, float) const override { return M0; }

    // Helpers used by drivers / validation.
    inline float stored_energy(float rho_val)  const { return disloc_coeff * rho_val; }
    inline float critical_radius(float rho_val) const {
        const float E = disloc_coeff * rho_val;
        return (E > 0.f) ? sigma0 / E : 1e30f;
    }
};

// =========================================================================
// Ostwald ripening — composition-coupled particle coarsening.
// =========================================================================
struct OstwaldThermo : ThermoModel {
    float c_sat;
    float Omega;

    OstwaldThermo(float M0_val, float sigma0_v,
                  float c_sat_v = 0.15f, float Omega_v = 5e8f) {
        M0     = M0_val;
        sigma0 = sigma0_v;
        eps4   = 0.f;
        c_sat  = c_sat_v;
        Omega  = Omega_v;
    }

    float bulk_driving_force(int phi_parent, int /*phi_product*/,
                             float c_val, float) const override {
        // matrix→particle: ΔG = G_particle − G_matrix = Ω(c_sat − c)  (<0 when supersaturated)
        if (phi_parent == 0) return Omega * (c_sat - c_val);
        // particle→matrix: ΔG = G_matrix − G_particle = Ω(c − c_sat)
        // Gives v_n = M(Ω(c − c_sat) − σκ): clips at 1 for large R in supersaturated matrix,
        // drives dissolution for small R or undersaturated matrix
        if (phi_parent >= 1) return Omega * (c_val - c_sat);
        return 0.f;
    }

    float interface_composition(int phi_parent, int phi_product,
                                float c_parent, float c_product) const override {
        // The Gibbs-Thomson supersaturation is defined in the matrix.  Using
        // the particle-side c (normally c_particle) suppresses dissolution
        // entirely and freezes the size distribution.
        if (phi_parent == 0) return c_parent;
        if (phi_product == 0) return c_product;
        return c_parent;
    }

    float mobility(float, float, float, float, float, float) const override { return M0; }
};

// =========================================================================
// Tensor anisotropy — gradient-energy tensor (tetragonal / uniaxial).
//
// Sharp-interface equivalent of c.cpp's b_ij/2 gradient energy term.
// In c.cpp the anisotropic Allen-Cahn PDE carries an extra term:
//
//   ∂η/∂t = −L[f′(η) − ε²∇²η − (bxx·∂²η/∂x² + byy·∂²η/∂y² + bzz·∂²η/∂z²)]
//
// The total gradient-energy tensor is K = diag(ε²+bxx, ε²+byy, ε²+bzz).
// In the sharp-interface limit σ ∝ √(K_ij n_i n_j):
//
//   σ(n̂) = σ₀ √(bxx·nc_x² + byy·nc_y² + bzz·nc_z²)
//
// where nc = R(g)⁻¹ n̂ is the crystal-frame normal.
//
// Parameter guide:
//   bxx = byy = bzz = 1  →  isotropic (σ = σ₀  for |n̂|=1)
//   bxx = byy = 1, bzz = 4  →  tetragonal, σ(n̂‖z) = 2σ₀
//   bxx ≠ byy, bzz = 1      →  2-D orthorhombic; strong in-plane anisotropy
// =========================================================================
struct TensorAnisotropyThermo : ThermoModel {
    float bxx, byy, bzz;

    // Optional volume-conservation bias (J/m³).  A driver may update this each
    // step with a proportional controller to hold the precipitate volume fixed
    // so that only its SHAPE evolves.  v_n = M(ΔG_bias − σκ): flat faces follow
    // the bias and stay put, while high-curvature groove roots still recede,
    // enabling pinch-off at (near) constant volume.  Mutable so a const thermo
    // reference can be retuned between steps.
    mutable float dG_bias = 0.f;

    TensorAnisotropyThermo(float M0_val, float sigma0_val,
                           float bxx_val = 1.f,
                           float byy_val = 1.f,
                           float bzz_val = 1.f)
        : bxx(bxx_val), byy(byy_val), bzz(bzz_val)
    {
        M0     = M0_val;
        sigma0 = sigma0_val;
        eps4   = 0.f;
    }

    float bulk_driving_force(int phi_parent, int, float, float) const override {
        if (dG_bias == 0.f) return 0.f;
        return (phi_parent != 0) ? dG_bias : -dG_bias;  // particle grows for bias>0
    }
    float mobility(float, float, float, float, float, float) const override { return M0; }

    float interfacial_energy(float nx, float ny, float nz,
                             float g_val) const override {
        const float cg = std::cos(g_val), sg = std::sin(g_val);
        const float nc_x =  nx * cg + ny * sg;
        const float nc_y = -nx * sg + ny * cg;
        const float nc_z = nz;
        const float K = bxx * nc_x * nc_x
                      + byy * nc_y * nc_y
                      + bzz * nc_z * nc_z;
        return sigma0 * std::sqrt(std::max(K, 1e-12f));
    }
};

// =========================================================================
// Anisotropic Ostwald — tensor interfacial energy + volume-conserving
// composition coupling.  Used for precipitate shape instabilities
// (plate breakup / cylinderisation) where the precipitate must stay
// bulk-stable so only its SHAPE evolves at (near) constant volume.
//
//   σ(n̂)   = σ₀ √(bxx·nc_x² + byy·nc_y² + bzz·nc_z²)   (broad/side/GB)
//   ΔG_chem = Ω(c_sat − c)  matrix→particle ;  Ω(c − c_sat) particle→matrix
//
// Curvature pumps solute from high-curvature regions (groove roots) into
// the matrix; diffusion (D>0) redeposits it on flat faces, conserving the
// precipitate volume while the groove deepens — the correct driving force
// for grain-boundary-groove-induced plate breakup.
// =========================================================================
struct AnisoOstwaldThermo : ThermoModel {
    float bxx, byy, bzz;
    float c_sat, Omega;

    AnisoOstwaldThermo(float M0_val, float sigma0_val,
                       float bxx_val, float byy_val, float bzz_val,
                       float c_sat_val, float Omega_val)
        : bxx(bxx_val), byy(byy_val), bzz(bzz_val),
          c_sat(c_sat_val), Omega(Omega_val)
    {
        M0     = M0_val;
        sigma0 = sigma0_val;
        eps4   = 0.f;
    }

    float bulk_driving_force(int phi_parent, int /*phi_product*/,
                             float c_val, float) const override {
        if (phi_parent == 0) return Omega * (c_sat - c_val);  // matrix→particle
        return Omega * (c_val - c_sat);                       // particle→matrix
    }

    float interface_composition(int phi_parent, int phi_product,
                                float c_parent, float c_product) const override {
        if (phi_parent == 0) return c_parent;
        if (phi_product == 0) return c_product;
        return c_parent;
    }

    float mobility(float, float, float, float, float, float) const override { return M0; }

    float interfacial_energy(float nx, float ny, float nz,
                             float g_val) const override {
        const float cg = std::cos(g_val), sg = std::sin(g_val);
        const float nc_x =  nx * cg + ny * sg;
        const float nc_y = -nx * sg + ny * cg;
        const float nc_z = nz;
        const float K = bxx * nc_x * nc_x
                      + byy * nc_y * nc_y
                      + bzz * nc_z * nc_z;
        return sigma0 * std::sqrt(std::max(K, 1e-12f));
    }
};

// =========================================================================
// Multi-phase precipitation with polynomial Gibbs energies.
//   G_q(c,T) = a_q + b_q*c + c_q*c^2 + d_q*(T-T_ref)
// The product phase is chosen by local Gibbs minimisation over neighboring
// candidate phases, matching the phase-selection formulation.
// =========================================================================
struct PrecipitationThermo : ThermoModel {
    int n_phases = 0;
    float T_ref = 900.f;
    std::vector<float> a, b, c2, dT;

    PrecipitationThermo(float M0_val,
                        float sigma0_val,
                        int n_phases_val,
                        const std::vector<float>& a_val,
                        const std::vector<float>& b_val,
                        const std::vector<float>& c2_val,
                        const std::vector<float>& dT_val,
                        float T_ref_val = 900.f)
        : n_phases(n_phases_val), T_ref(T_ref_val),
          a(a_val), b(b_val), c2(c2_val), dT(dT_val)
    {
        M0 = M0_val;
        sigma0 = sigma0_val;
        eps4 = 0.f;
        phase_selection = true;
        if ((int)a.size() != n_phases || (int)b.size() != n_phases ||
            (int)c2.size() != n_phases || (int)dT.size() != n_phases) {
            throw std::runtime_error("PrecipitationThermo: coefficient size mismatch");
        }
    }

    inline float G(int q, float c_val, float T_val) const {
        return a[q] + b[q] * c_val + c2[q] * c_val * c_val + dT[q] * (T_val - T_ref);
    }

    float bulk_driving_force(int phi_parent, int phi_product,
                             float c_val, float T_val) const override {
        if (phi_parent < 0 || phi_parent >= n_phases ||
            phi_product < 0 || phi_product >= n_phases) {
            return 0.f;
        }
        return G(phi_product, c_val, T_val) - G(phi_parent, c_val, T_val);
    }

    int select_product_phase(int phi_parent,
                             float c_val, float T_val,
                             const int32_t* candidates, int n_cands) const override {
        float best = 1e30f;
        int best_q = -1;
        for (int k = 0; k < n_cands; ++k) {
            int32_t q = candidates[k];
            if (q < 0 || q >= n_phases || q == phi_parent) continue;
            float dG = G(q, c_val, T_val) - G(phi_parent, c_val, T_val);
            if (dG < best) { best = dG; best_q = q; }
        }
        return best_q;
    }

    float mobility(float, float, float, float, float, float) const override { return M0; }
};

// =========================================================================
// Thermal/grain-boundary grooving — phase-PAIR-dependent interfacial energy.
//   Convention: phi == VAPOR_LABEL is the free surface (vacuum/vapour/pore);
//   any other phi label is a solid grain.
//   sigma(parent,product) = sigma_gb   if BOTH labels are solid grains
//                          = sigma_sv  if EITHER label is the vapor phase
// This is the model needed for the classical Mullins thermal-grooving /
// grain-boundary-groove dihedral-angle problem: sigma_gb = 2*sigma_sv*cos(psi/2)
// at the equilibrium groove root (used for real-material Fe in validate_real_materials).
// =========================================================================
struct GroovingThermo : ThermoModel {
    static constexpr int32_t VAPOR_LABEL = 0;
    float sigma_gb, sigma_sv;

    GroovingThermo(float M0_val, float sigma_gb_val, float sigma_sv_val)
        : sigma_gb(sigma_gb_val), sigma_sv(sigma_sv_val) {
        M0     = M0_val;
        sigma0 = sigma_gb_val;   // fallback for any orientation-only caller
        eps4   = 0.f;
    }
    float bulk_driving_force(int, int, float, float) const override { return 0.f; }
    float mobility(float, float, float, float, float, float) const override { return M0; }
    float interfacial_energy_pair(float, float, float, float,
                                  int32_t phi_parent, int32_t phi_product) const override {
        bool either_vapor = (phi_parent == VAPOR_LABEL) || (phi_product == VAPOR_LABEL);
        return either_vapor ? sigma_sv : sigma_gb;
    }
};

// =========================================================================
// Hydrogen reduction of a mixed metal oxide — moving oxide/metal interface.
//   Reaction (as written for iron-oxide/H2 systems): MO + H2 -> M + H2O.
//   Driving force from the imposed gas-phase ratio relative to the real
//   equilibrium ratio at temperature T:
//     dG_chem = dG0(T) + R T ln(pH2O/pH2)        [J/mol]  ->  convert to J/m^3
//   dG0(T) is taken as a single calibration point anchored to a literature-
//   verified equilibrium: at T0=1073 K (800 C), pH2O/pH2 = 0.493
//   gives equilibrium (dG_chem=0), so dG0(T0) = -R*T0*ln(0.493).  A linear
//   T-dependence about T0 is NOT assumed beyond that one verified anchor;
//   dG0 is held constant at its T0 value unless a second anchor is supplied.
//   phi_parent==0 is the oxide (parent); phi_parent>=1 is reduced metal.
// =========================================================================
struct ReductionThermo : ThermoModel {
    float pH2_over_pH2O;     // imposed process gas ratio (>1 favours reduction)
    float dG0_J_per_mol;     // calibration intercept, J/mol (see header note)
    float V_molar_metal;     // m^3/mol, for J/mol -> J/m^3 conversion

    ReductionThermo(float M0_val, float sigma0_val,
                    float pH2_over_pH2O_val, float dG0_J_per_mol_val,
                    float V_molar_metal_val)
        : pH2_over_pH2O(pH2_over_pH2O_val), dG0_J_per_mol(dG0_J_per_mol_val),
          V_molar_metal(V_molar_metal_val) {
        M0     = M0_val;
        sigma0 = sigma0_val;
        eps4   = 0.f;
    }

    float bulk_driving_force(int phi_parent, int phi_product,
                             float, float T_val) const override {
        const float Rgas = 8.314f;
        // dG_reaction = dG0 + RT ln(pH2O/pH2) = dG0 - RT ln(pH2_over_pH2O)
        float dG_mol = dG0_J_per_mol - Rgas * T_val * std::log(pH2_over_pH2O);
        float dG_vol = dG_mol / V_molar_metal;   // J/mol -> J/m^3
        // oxide(0) -> metal(>=1): favourable (negative) when dG_vol<0, i.e.
        // when the process gas is more reducing than equilibrium.
        if (phi_parent == 0 && phi_product >= 1) return dG_vol;
        if (phi_parent >= 1 && phi_product == 0) return -dG_vol;
        return 0.f;
    }
    float mobility(float, float, float, float, float, float) const override { return M0; }
};

// =========================================================================
// Magnetic-field-assisted / abnormal grain growth — uniaxial magnetocrystalline
// anisotropy + Zeeman energy as a per-grain driving-force term.
//   Phases here ARE grains (phi = grain id); g = crystal orientation angle,
//   reused as the in-plane easy-axis direction (uniaxial/tetragonal magnets,
//   e.g. Nd2Fe14B). No chemical driving force; capillarity is isotropic so
//   the ONLY new physics exercised is E_mag(g_product) - E_mag(g_parent),
//   wired generically by the core via magnetic_enabled (gpk_core.hpp/.cpp).
//   Reference values (Nd2Fe14B, room temperature): Ku ~ 4.3e6 J/m^3,
//   Ms ~ 1.28e6 A/m.
// =========================================================================
struct MagneticAnnealingThermo : ThermoModel {
    MagneticAnnealingThermo(float M0_val, float sigma0_val,
                            float Ku_val, float Ms_val,
                            float H_ext_val, float H_theta_val,
                            float g_easy_ref_val = 0.f)
    {
        M0     = M0_val;
        sigma0 = sigma0_val;
        eps4   = 0.f;
        magnetic_enabled = true;
        Ku       = Ku_val;
        Ms       = Ms_val;
        H_ext    = H_ext_val;
        H_theta  = H_theta_val;
        g_easy_ref = g_easy_ref_val;
    }
    float bulk_driving_force(int, int, float, float) const override { return 0.f; }
    float mobility(float, float, float, float, float, float) const override { return M0; }
    // interfacial_energy(): inherits the isotropic base-class default
    // (sigma0, curvature-only capillarity) — deliberately not anisotropic,
    // so grain competition is driven purely by E_mag, isolating the magnetic
    // effect for validation.
};

// =========================================================================
// Magnetoelastic (magnetostriction) coupling.
// Reuses the existing eigenstrain/Eshelby elastic pipeline (step_elastic()):
// eps* gains an isotropic-magnetostriction term (3/2)*lambda_s*(cos^2(g-H_theta)-1/3)
// on top of (or instead of, if eigenstrain_coeff=0) the composition-driven term.
// Exact for cubic-isotropic magnetostrictive materials (e.g. Fe-Ga/Galfenol,
// lambda_s ~ 2-4e-4); only qualitatively representative for materials whose
// magnetostriction is dominated by twin-boundary motion (Ni-Mn-Ga).
// Optionally also enables the magnetic anisotropy+Zeeman term (Ku/Ms
// nonzero), demonstrating the two mechanisms compose orthogonally through
// the same ThermoModel base.
// =========================================================================
struct MagnetoelasticThermo : ThermoModel {
    MagnetoelasticThermo(float M0_val, float sigma0_val,
                         float elastic_modulus_val, float lambda_s_val,
                         float H_theta_val, float H_ext_val = 0.f,
                         float Ku_val = 0.f, float Ms_val = 0.f)
    {
        M0     = M0_val;
        sigma0 = sigma0_val;
        eps4   = 0.f;
        elastic_enabled   = true;
        elastic_modulus   = elastic_modulus_val;
        eigenstrain_coeff = 0.f;   // pure magnetostriction, no composition eigenstrain
        c_ref             = 0.f;
        lambda_s          = lambda_s_val;
        H_theta           = H_theta_val;
        if (Ku_val != 0.f || Ms_val != 0.f) {
            magnetic_enabled = true;
            Ku    = Ku_val;
            Ms    = Ms_val;
            H_ext = H_ext_val;
        }
    }
    float bulk_driving_force(int, int, float, float) const override { return 0.f; }
    float mobility(float, float, float, float, float, float) const override { return M0; }
    // Elastic driving force (E_prod - E_el[i]) and, if enabled, the magnetic
    // driving force are both applied automatically by step_velocities()'s
    // existing elastic_enabled/magnetic_enabled branches; no override needed.
};

// =========================================================================
// Nd-Fe-B grain-boundary (GB) wetting phase.
//   Convention: phi == GBPHASE_LABEL (0) is the Nd-rich intergranular
//   liquid/amorphous phase; any other phi label is a T1 (Nd2Fe14B) grain.
//   sigma(T1/GBphase) decreases with local Cu content c (better wetting),
//   sigma(T1/T1) stays constant — mirrors GroovingThermo's either_vapor
//   pattern with GBPHASE_LABEL replacing VAPOR_LABEL and a composition-
//   dependent wetting energy replacing the constant sigma_sv. GB-phase
//   growth is driven by local Nd/Cu supersaturation, reusing OstwaldThermo's
//   exact bulk_driving_force pattern.
// =========================================================================
struct NdFeBGrainBoundaryThermo : ThermoModel {
    static constexpr int32_t GBPHASE_LABEL = 0;
    float sigma_gb;         // T1/T1 grain-grain interfacial energy [J/m^2]
    float sigma_gbphase_0;  // T1/GBphase energy at c=0 (no Cu) [J/m^2]
    float k_Cu;             // wetting-improvement coefficient (per unit composition)
    float c_sat, Omega;     // GB-phase growth driving force (Ostwald/Precipitation pattern)

    NdFeBGrainBoundaryThermo(float M0_val, float sigma_gb_val,
                             float sigma_gbphase_0_val, float k_Cu_val,
                             float c_sat_val, float Omega_val)
        : sigma_gb(sigma_gb_val), sigma_gbphase_0(sigma_gbphase_0_val),
          k_Cu(k_Cu_val), c_sat(c_sat_val), Omega(Omega_val)
    {
        M0     = M0_val;
        sigma0 = sigma_gb_val;   // fallback for any orientation-only caller
        eps4   = 0.f;
    }

    // sigma(T1/GBphase) = sigma_gbphase_0 * (1 - k_Cu*c), clamped to stay
    // physical: adding Cu can only improve wetting (lower sigma), down to the
    // complete-wetting limit of 0, never exceeding the no-Cu baseline.
    inline float sigma_wetting(float c_val) const {
        float s = sigma_gbphase_0 * (1.f - k_Cu * c_val);
        return std::clamp(s, 0.f, sigma_gbphase_0);
    }

    float bulk_driving_force(int phi_parent, int /*phi_product*/,
                             float c_val, float) const override {
        // T1 grain -> GB-phase: Omega*(c_sat - c) < 0 when locally supersaturated
        // in Nd/Cu (reuses OstwaldThermo's exact pattern, see above).
        if (phi_parent != GBPHASE_LABEL) return Omega * (c_sat - c_val);
        return Omega * (c_val - c_sat);
    }
    float mobility(float, float, float, float, float, float) const override { return M0; }

    float interfacial_energy_pair_comp(float, float, float, float,
                                       int32_t phi_parent, int32_t phi_product,
                                       float c_val) const override {
        bool either_gbphase = (phi_parent == GBPHASE_LABEL) || (phi_product == GBPHASE_LABEL);
        return either_gbphase ? sigma_wetting(c_val) : sigma_gb;
    }
};

} // namespace gpk
