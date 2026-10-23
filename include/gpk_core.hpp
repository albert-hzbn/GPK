#pragma once
/*
 * Gaussian-Projected Kinetics (GPK) — C++ core header  (3-D)
 *
 * Kinetic equation (sharp-interface limit of Allen-Cahn):
 *   v_n = M(n̂, g) · [ΔG_chem(c,T) − σ(n̂,g)·κ]
 *
 * Sign convention:
 *   v_n < 0  → parent phase loses volume fraction α → converts to competitor
 *   κ = ∇·n̂ > 0  for a convex parent grain (outward-normal convention)
 *
 * Grid layout: index i = z*Ny*Nx + y*Nx + x
 *   z — slowest index (depth);  y — rows;  x — columns (fastest)
 *   For a 2-D simulation set Nz = 1.
 *
 * Parallelisation: OpenMP over independent cell and compact-interface loops.
 * Full-domain work buffers are preallocated. Rare topology transactions use
 * sparse temporary containers whose size follows active labels/events.
 */

#include <vector>
#include <cmath>
#include <algorithm>
#include <functional>
#include <memory>
#include <cstdint>
#include <cstddef>
#include <numeric>

namespace gpk {

class GPKGrid;

// =========================================================================
// Gaussian kernel (precomputed, separable — shared by all 3 passes)
// =========================================================================
struct GaussKernel {
    std::vector<float> w;
    int radius;
    float sigma_cells;

    explicit GaussKernel(float sigma = 2.0f) : sigma_cells(std::max(sigma, 0.5f)) {
        sigma = sigma_cells;
        radius = static_cast<int>(std::ceil(3.5f * sigma));
        int sz  = 2 * radius + 1;
        w.resize(sz);
        float sum = 0.f;
        for (int i = 0; i < sz; ++i) {
            float x = float(i - radius);
            w[i] = std::exp(-x * x / (2.f * sigma * sigma));
            sum += w[i];
        }
        for (auto& v : w) v /= sum;
    }
};

/*
 * Separable 3-D Gaussian filter with periodic boundary conditions.
 * tmp1, tmp2 must each have size >= Nz*Ny*Nx.
 * When Nz == 1 the z-pass is a no-op and the function is equivalent to 2-D.
 */
void gaussian_filter_3d(const float* __restrict__ in,
                        float*       __restrict__ out,
                        float*       __restrict__ tmp1,
                        float*       __restrict__ tmp2,
                        int Nz, int Ny, int Nx,
                        const GaussKernel& K);

// =========================================================================
// Thermodynamic model (override to plug in CALPHAD or custom models)
// =========================================================================
struct ThermoModel {
    float sigma0 = 0.5f;    // J/m²  baseline interfacial energy
    float eps4   = 0.0f;    // 4-fold / cubic anisotropy strength
    float M0     = 1e-8f;   // m³/(J·s)  baseline mobility
    float Q      = 0.0f;    // activation energy (J/mol)
    float Rgas   = 8.314f;
    bool  phase_selection = false;
    bool  grain_network   = false; // many-grain topology with extinction cleanup

    // Stochastic kinetics controls.
    bool  stochastic   = false;
    float noise_scale  = 1.0f;
    float kB           = 1.380649e-23f;

    // Solute partitioning at a growing solid (opt-in). When enabled, a liquid
    // (label 0) volume that solidifies keeps k_partition * c_L and rejects the
    // remainder (1 - k_partition) * c_L to neighbouring liquid cells, and
    // diffusion acts on the liquid composition through face fluxes weighted by
    // the liquid fraction. The total solute content is conserved exactly.
    bool  solute_rejection = false;
    float k_partition      = 1.0f;

    // Elastic coupling controls.
    bool  elastic_enabled    = false;
    float elastic_modulus    = 0.0f;   // [J/m^3]
    float eigenstrain_coeff  = 0.0f;   // eps* = coeff * (c - c_ref)
    float c_ref              = 0.0f;
    int   elastic_smooth_iters = 8;

    // Stored-energy (dislocation) coupling and recovery.
    // Stored energy density follows the Taylor relation E_rho = disloc_coeff * rho,
    // with disloc_coeff = 0.5 * alpha_T * mu * b^2. It enters the driving force as
    // Delta G_stored = E_rho(product) - E_rho(parent), exactly like the elastic
    // energy. A swept boundary leaves material at density rho_rex behind it, and
    // static recovery relaxes rho by d(rho)/dt = -recovery_rate * rho.
    bool  dislocation_enabled = false;
    float disloc_coeff        = 0.0f;  // [J m^-2] so that E_rho = disloc_coeff * rho [J/m^3]
    float rho_rex             = 0.0f;  // [m^-2] density left behind a swept boundary
    float recovery_rate       = 0.0f;  // [1/s] static-recovery rate

    // Magnetic coupling controls.
    bool  magnetic_enabled = false;
    float Ku         = 0.f;   // [J/m^3] uniaxial magnetocrystalline anisotropy constant
    float Ms         = 0.f;   // [A/m] saturation magnetization
    float H_ext      = 0.f;   // [A/m] applied field magnitude
    float H_theta    = 0.f;   // [rad] applied field direction (lab frame; also reused
                               //       by the magnetostriction eigenstrain below)
    float g_easy_ref = 0.f;   // [rad] crystal-frame easy-axis offset (easy axis == g by default)
    static constexpr float MU0 = 4.0e-7f * 3.14159265358979323846f;  // [T*m/A] vacuum permeability

    // Magnetostrictive eigenstrain addition, reuses the elastic pipeline.
    float lambda_s = 0.f;     // saturation magnetostriction constant (dimensionless)

    // ΔG_chem = G_product − G_parent  [J/m³]
    virtual float bulk_driving_force(int, int, float, float) const {
        return 0.f;
    }

    // Composition sampled for interfacial thermodynamics.  Most models use
    // the parent cell value; two-phase coarsening models override this to use
    // the matrix-side concentration on either side of the same interface.
    virtual float interface_composition(int, int,
                                        float c_parent, float) const {
        return c_parent;
    }

    // Optional thermodynamic phase selector for multi-phase systems.
    // Returns >=0 to override geometric competitor, or -1 to keep geometric choice.
    // candidates[0..n_cands-1] are the neighbouring phase labels (stack array, no heap).
    virtual int select_product_phase(int phi_parent,
                                     float c_val, float T_val,
                                     const int32_t* candidates, int n_cands) const {
        (void)phi_parent; (void)c_val; (void)T_val;
        (void)candidates; (void)n_cands;
        return -1;
    }

    // σ(n̂, g) — 3-D cubic anisotropy: σ = σ₀(1 + ε₄(nx⁴+ny⁴+nz⁴ − 3/5))
    virtual float interfacial_energy(float nx, float ny, float nz,
                                     float g_val) const;

    // Phase-PAIR-aware interfacial energy — e.g. grain-boundary energy when
    // both phi_parent and phi_product are solid grains, vs. a much larger
    // free-surface (solid-vapor) energy when one of them is a void/vapour
    // label. Needed for thermal-grooving and sintering models, where sigma
    // is NOT just a function of (n_hat, g) but of which two phases meet.
    // Default delegates to the orientation-only interfacial_energy() above,
    // so every existing ThermoModel subclass is unaffected unless it opts in
    // by overriding this method directly.
    virtual float interfacial_energy_pair(float nx, float ny, float nz, float g_val,
                                          int32_t /*phi_parent*/, int32_t /*phi_product*/) const {
        return interfacial_energy(nx, ny, nz, g_val);
    }

    // Composition-aware phase-pair interfacial energy — generalises
    // interfacial_energy_pair() for cases where sigma depends on local
    // composition (e.g. Cu-content-dependent grain-boundary-phase wetting).
    // Default delegates to interfacial_energy_pair(), so every
    // pre-existing thermodynamic model is unaffected unless it opts in by
    // overriding this method directly.
    virtual float interfacial_energy_pair_comp(float nx, float ny, float nz, float g_val,
                                               int32_t phi_parent, int32_t phi_product,
                                               float /*c_val*/) const {
        return interfacial_energy_pair(nx, ny, nz, g_val, phi_parent, phi_product);
    }

    // E_mag(g) — uniaxial magnetocrystalline anisotropy + Zeeman energy density.
    // Quasi-static approximation: the local magnetization is assumed pinned to
    // the grain's easy axis and instantaneously equilibrated relative to the
    // (much slower) grain-boundary migration timescale, so no LLG/TDGL PDE is
    // solved — only an algebraic energy density (valid when Ku >> mu0*Ms*H_ext).
    // g_val is interpreted exactly as elsewhere in
    // GPK: the grain orientation angle (rotation about z), doubling as the
    // in-plane easy-axis direction for uniaxial (tetragonal) magnets such as
    // Nd2Fe14B.
    virtual float E_mag(float g_val) const {
        if (!magnetic_enabled) return 0.f;
        float c1 = std::cos(g_val - g_easy_ref);
        float c2 = std::cos(g_val - H_theta);
        return -Ku * c1 * c1 - MU0 * Ms * H_ext * c2;
    }

    // Read-Shockley misorientation-dependent mobility (uses g as proxy angle).
    virtual float mobility(float nx, float ny, float nz,
                           float g_parent, float g_product,
                           float T_val) const;

    virtual ~ThermoModel() = default;
};

// =========================================================================
// GPKGrid — main simulation domain (3-D, degenerates to 2-D when Nz=1)
// =========================================================================
class GPKGrid {
public:
    const int   Nz, Ny, Nx;  // grid dimensions (z=depth, y=rows, x=cols)
    const float dx;           // isotropic cell size [m]
    double      time = 0.0;

    // ---- primary state fields ----
    std::vector<int32_t> phi;   // phase / grain labels
    std::vector<float>   alpha; // sub-cell volume fraction [0,1]
    std::vector<float>   c;     // composition
    std::vector<float>   T;     // temperature [K]
    std::vector<float>   g;     // crystal orientation [rad]
    std::vector<float>   rho;   // dislocation density [m^-2] (stored-energy field)

    // ---- work arrays (reused every step) ----
    std::vector<float>   nx_f, ny_f, nz_f; // interface outward normals
    std::vector<float>   kappa_f;           // mean curvature [m⁻¹]
    std::vector<float>   vn_f;              // interface velocity [m/s]
    std::vector<float>   E_el;              // elastic energy density [J/m^3]
    std::vector<int32_t> phi_comp;          // competing grain label
    std::vector<float>   g_comp;            // competing grain's orientation [rad] —
                                              // synced into g[i] on phase switch (step_alpha)
                                              // so g never goes stale after a cell changes owner.
    std::vector<uint8_t> mask;              // 1 = interface cell, 0 = bulk

    // Diagnostics for the topology-aware transition stage in the last step.
    // A deferred crossing retains its sub-cell displacement and is tested
    // again after neighbouring transitions change the local configuration.
    uint64_t topology_accepted_last = 0;
    uint64_t topology_deferred_last = 0;
    uint64_t gaussian_remap_exchanges_last = 0;

    GPKGrid(int Nz, int Ny, int Nx, float dx,
            std::shared_ptr<ThermoModel> thermo,
            float D = 0.f,
            float gaussian_sigma = 2.0f);

    // Advance by one timestep.  dt < 0 → auto (stable) timestep.
    float step(float dt = -1.f, float dt_cap = 1.0e30f);

    // Run until time >= t_end.  Returns number of steps taken.
    int run(double t_end, float dt = -1.f,
            std::function<void(const GPKGrid&, int)> cb = nullptr,
            int cb_interval = 100);

    float  max_dt()             const;
    double phase_volume(int32_t label) const;   // sub-cell volume in units of cells
    double mean_grain_radius()  const;   // characteristic mean-volume radius [m]
    int    n_grains()           const;
    float  solid_fraction()     const;
    float  interface_fraction() const;

    // Total capacity of arrays owned by this grid, including compact-interface
    // indices and fused geometry kernels. This excludes allocator metadata,
    // the ThermoModel object and transient library/runtime storage.
    std::size_t allocated_array_bytes() const;

    // Read-only diagnostic of the generic p->q digital-topology certificate.
    // It applies the same dual-connectivity simple-point test used by step().
    bool topology_simple_move(int cell, int32_t new_label) const;

    int N() const { return Nz * Ny * Nx; }
    inline int idx(int z, int y, int x) const { return z*Ny*Nx + y*Nx + x; }

    // Decode a flat cell index into (z, y, x) grid coordinates.
    inline void decode(int i, int& z, int& y, int& x) const {
        x = i % Nx;
        y = (i / Nx) % Ny;
        z = i / (Ny * Nx);
    }

private:
    std::shared_ptr<ThermoModel> thermo_;
    float D_;
    GaussKernel gauss_;
    GaussKernel velocity_gauss_;
    GaussKernel remap_gauss_;

    // Compact interface representation. Cells are stored in increasing flat
    // index order, preserving deterministic topology-transaction order.
    std::vector<int> active_cells_;
    std::vector<int> active_block_counts_;
    std::vector<int> active_block_offsets_;

    // Exact convolution of the Gaussian with the fourth-order centred
    // derivative stencils. Geometry is evaluated only at interface cells.
    std::vector<float> geom_w0_, geom_w1_, geom_w2_;
    int geom_radius_ = 0;
    bool geom_use_global_separable_ = false;
    std::vector<float> geom_I_, geom_smooth_, geom_tmp1_, geom_tmp2_;

    std::vector<float>   alpha_new_;
    std::vector<float>   c_new_;
    std::vector<float>   eps_star_;
    std::vector<float>   eps_smooth_;
    std::vector<float>   eps_tmp_;
    uint64_t             step_id_ = 0;

    void build_geometry_kernels();
    void step_geometry_global_separable();

    void step_interface_mask();
    void step_geometry();
    void step_elastic();
    void step_velocities();
    void step_velocity_projection();
    void step_alpha(float dt);
    void step_balanced_gaussian_remap();
    void step_complete_unresolved_grains();
    bool topology_simple_move_counted(int cell, int32_t new_label,
                                      int old_label_count,
                                      int new_label_count,
                                      bool local_only = false) const;
    void step_diffusion(float dt);
    void step_recovery(float dt);

    // Opt-in solute partitioning (ThermoModel::solute_rejection).
    std::vector<float> solid_old_;
    void cell_solid_fraction(std::vector<float>& s) const;
    void step_solute_rejection();
    void step_diffusion_partitioned(float dt);
};

} // namespace gpk
