/* GPKGrid construction and workspace allocation. */
#include "gpk_core.hpp"

#include <algorithm>
#include <utility>

namespace gpk {

GPKGrid::GPKGrid(int Nz_, int Ny_, int Nx_, float dx_,
                 std::shared_ptr<ThermoModel> thermo,
                 float D,
                 float gaussian_sigma)
    : Nz(Nz_), Ny(Ny_), Nx(Nx_), dx(dx_),
      phi(Nz_*Ny_*Nx_, 0),   alpha(Nz_*Ny_*Nx_, 1.f),
      c(Nz_*Ny_*Nx_, 0.f),   T(Nz_*Ny_*Nx_, 800.f),
      g(Nz_*Ny_*Nx_, 0.f),   rho(Nz_*Ny_*Nx_, 0.f),
      nx_f(Nz_*Ny_*Nx_, 0.f), ny_f(Nz_*Ny_*Nx_, 0.f),
      nz_f(Nz_*Ny_*Nx_, 0.f),
      kappa_f(Nz_*Ny_*Nx_, 0.f), vn_f(Nz_*Ny_*Nx_, 0.f),
      E_el(Nz_*Ny_*Nx_, 0.f),
      phi_comp(Nz_*Ny_*Nx_, 0), g_comp(Nz_*Ny_*Nx_, 0.f), mask(Nz_*Ny_*Nx_, false),
      thermo_(std::move(thermo)), D_(D),
      gauss_(gaussian_sigma),
      velocity_gauss_(Nz_ > 1
          ? std::clamp(0.375f * gaussian_sigma, 0.85f, 3.00f) : 0.85f),
      remap_gauss_(Nz_ > 1
          ? std::clamp(0.125f * gaussian_sigma, 0.70f, 1.00f) : 0.70f),
      alpha_new_(Nz_*Ny_*Nx_), c_new_(Nz_*Ny_*Nx_),
      eps_star_(Nz_*Ny_*Nx_, 0.f), eps_smooth_(Nz_*Ny_*Nx_, 0.f),
      eps_tmp_(Nz_*Ny_*Nx_, 0.f)
{
    const int N_ = Nz_ * Ny_ * Nx_;
    // Capacity depends only on the grid; no thread-local full-domain buffers
    // are required by the interface-local geometry reconstruction.
    active_cells_.reserve(N_);
    constexpr int block_size = 4096;
    const int nblocks = std::max(1, (N_ + block_size - 1) / block_size);
    active_block_counts_.resize(nblocks, 0);
    active_block_offsets_.resize(nblocks + 1, 0);
    build_geometry_kernels();
    // A broad three-dimensional tensor stencil grows cubically.  In that
    // regime a single shared separable workspace is faster and remains
    // independent of the OpenMP thread count.
    geom_use_global_separable_ = Nz_ > 1 && geom_radius_ > 18;
    if (geom_use_global_separable_) {
        geom_I_.resize(N_); geom_smooth_.resize(N_);
        geom_tmp1_.resize(N_); geom_tmp2_.resize(N_);
    }
}

} // namespace gpk
