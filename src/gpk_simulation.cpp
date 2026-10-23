/* Timestep control, run orchestration, and public diagnostics. */
#include "gpk_core.hpp"

#include <omp.h>
#include <algorithm>
#include <cmath>
#include <unordered_set>

namespace gpk {

float GPKGrid::max_dt() const {
    float vmax = 1e-30f;
    #pragma omp parallel for reduction(max:vmax) schedule(static)
    for (int a = 0; a < static_cast<int>(active_cells_.size()); ++a) {
        const int i = active_cells_[a];
        vmax = std::max(vmax, std::fabs(vn_f[i]));
    }

    const float neighbors = (Nz > 1) ? 6.f : 4.f;
    // C_I=1/8 makes a constant-speed planar front traverse a cell in exactly
    // eight steps and limits fractional-crossing truncation on curved fronts.
    const float dt_iface  = 0.125f * dx / vmax;
    const float dt_diff   = (D_ > 0.f) ? 0.4f * dx * dx / (neighbors * D_) : 1e30f;
    return std::min(dt_iface, dt_diff);
}

// =========================================================================
// Advance one complete timestep.
// =========================================================================

float GPKGrid::step(float dt, float dt_cap) {
    step_interface_mask();
    step_geometry();
    step_elastic();
    step_velocities();
    step_velocity_projection();
    const float dt_stable = max_dt();
    if (dt < 0.f) dt = dt_stable;
    else          dt = std::min(dt, dt_stable);
    dt = std::min(dt, dt_cap);
    const bool partition = thermo_->solute_rejection;
    if (partition) cell_solid_fraction(solid_old_);
    step_alpha(dt);
    if (partition) {
        step_solute_rejection();
        step_diffusion_partitioned(dt);
    } else {
        step_diffusion(dt);
    }
    step_recovery(dt);
    time += dt;
    ++step_id_;
    return dt;
}

// =========================================================================
// run()
// =========================================================================

int GPKGrid::run(double t_end, float dt,
                 std::function<void(const GPKGrid&, int)> cb,
                 int cb_interval) {
    int step_count = 0;
    while (time < t_end) {
        const float remaining = static_cast<float>(t_end - time);
        step(dt, remaining);
        ++step_count;
        if (cb && step_count % cb_interval == 0)
            cb(*this, step_count);
    }
    return step_count;
}

// =========================================================================
// Statistics
// =========================================================================

double GPKGrid::phase_volume(int32_t label) const {
    double volume = 0.0;
    for (int i = 0; i < N(); ++i) {
        if (phi[i] == label) volume += alpha[i];
        if (mask[i] && phi_comp[i] == label && phi_comp[i] != phi[i])
            volume += 1.0 - alpha[i];
    }
    return volume;
}

double GPKGrid::mean_grain_radius() const {
    // A label tessellation is an exact partition of the grid.  Alpha is a
    // lineal interface-position coordinate, not an isotropic cut-cell volume;
    // summing it around curved/multi-junction boundaries biases the total
    // material volume and can even make an apparent mean size decrease.
    // Use the exact partition volume for the characteristic mean-volume
    // radius.  Sub-cell phase_volume() remains available for planar-front
    // diagnostics where the lineal reconstruction is geometrically exact.
    std::vector<int32_t> labels(phi);
    std::sort(labels.begin(), labels.end());
    const int ng = static_cast<int>(
        std::unique(labels.begin(), labels.end()) - labels.begin());
    if (ng == 0) return 0.0;
    const double mean_volume = static_cast<double>(N()) / ng;
    const double R = (Nz > 1)
        ? std::cbrt(3.0 * mean_volume / (4.0 * M_PI))
        : std::sqrt(mean_volume / M_PI);
    return R * dx;
}

int GPKGrid::n_grains() const {
    std::vector<int32_t> tmp(phi);
    std::sort(tmp.begin(), tmp.end());
    return static_cast<int>(
        std::unique(tmp.begin(), tmp.end()) - tmp.begin());
}

float GPKGrid::solid_fraction() const {
    return static_cast<float>(phase_volume(1) / static_cast<double>(N()));
}

float GPKGrid::interface_fraction() const {
    int cnt = 0;
    for (auto v : mask) if (v) ++cnt;
    return static_cast<float>(cnt) / static_cast<float>(N());
}

std::size_t GPKGrid::allocated_array_bytes() const {
    auto bytes = [](const auto& v) -> std::size_t {
        using value_type = typename std::decay_t<decltype(v)>::value_type;
        return v.capacity() * sizeof(value_type);
    };

    std::size_t total = 0;
    total += bytes(phi) + bytes(alpha) + bytes(c) + bytes(T) + bytes(g) + bytes(rho);
    total += bytes(nx_f) + bytes(ny_f) + bytes(nz_f) + bytes(kappa_f) + bytes(vn_f);
    total += bytes(E_el) + bytes(phi_comp) + bytes(g_comp) + bytes(mask);
    total += bytes(alpha_new_) + bytes(c_new_) + bytes(eps_star_)
           + bytes(eps_smooth_) + bytes(eps_tmp_);
    total += bytes(gauss_.w) + bytes(velocity_gauss_.w) + bytes(remap_gauss_.w);
    total += bytes(active_cells_) + bytes(active_block_counts_)
           + bytes(active_block_offsets_);
    total += bytes(geom_w0_) + bytes(geom_w1_) + bytes(geom_w2_);
    total += bytes(geom_I_) + bytes(geom_smooth_) + bytes(geom_tmp1_)
           + bytes(geom_tmp2_);
    return total;
}

} // namespace gpk

