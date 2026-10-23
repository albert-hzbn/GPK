/* Composition diffusion and dislocation-density recovery. */
#include "gpk_core.hpp"

#include <omp.h>
#include <algorithm>
#include <cmath>
#include <utility>

namespace gpk {

void GPKGrid::step_diffusion(float dt) {
    if (D_ <= 0.f) return;

    const float* cm  = c.data();
    const float  dx2 = dx * dx;

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N(); ++i) {
        int z, y, x; decode(i, z, y, x);
        const int xm = (x==0    ? Nx-1 : x-1), xp = (x==Nx-1 ? 0 : x+1);
        const int ym = (y==0    ? Ny-1 : y-1), yp = (y==Ny-1 ? 0 : y+1);
        const int zm = (z==0    ? Nz-1 : z-1), zp = (z==Nz-1 ? 0 : z+1);

        float lap = cm[idx(z,y,xp)] + cm[idx(z,y,xm)]
                  + cm[idx(z,yp,x)] + cm[idx(z,ym,x)];
        if (Nz > 1)
            lap += cm[idx(zp,y,x)] + cm[idx(zm,y,x)] - 6.f * cm[i];
        else
            lap -= 4.f * cm[i];

        float D_eff  = mask[i] ? D_ * alpha[i] : D_;
        c_new_[i]    = cm[i] + dt * D_eff * lap / dx2;
    }
    std::swap(c, c_new_);
}

// =========================================================================
// Opt-in solute partitioning at a growing solid.
//   Solid fraction of a cell: s = alpha for a solid resident (label != 0) and
//   s = 1 - alpha for a liquid resident (label 0).  With the cell-average
//   composition c and partition coefficient k, the liquid composition is
//   c_L = c / (1 - s (1 - k)).  When s increases by ds during a step, the new
//   solid keeps k c_L and the excess (1 - k) c_L ds is passed to neighbouring
//   liquid-resident cells in proportion to their liquid fraction.  Removal and
//   addition are equal, so the total solute content is conserved exactly.
// =========================================================================

void GPKGrid::cell_solid_fraction(std::vector<float>& s) const {
    const int N_ = N();
    s.resize(N_);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N_; ++i)
        s[i] = (phi[i] != 0) ? alpha[i] : 1.f - alpha[i];
}

void GPKGrid::step_solute_rejection() {
    const float k = thermo_->k_partition;
    if (k >= 1.f || solid_old_.size() != static_cast<size_t>(N())) return;
    const int N_ = N();
    std::vector<float> s_new;
    cell_solid_fraction(s_new);
    std::vector<float> delta(N_, 0.f);

    for (int i = 0; i < N_; ++i) {
        const float ds = s_new[i] - solid_old_[i];
        if (ds <= 0.f) continue;                       // only growth rejects solute
        const float s0 = std::clamp(solid_old_[i], 0.f, 1.f);
        const float cL = c[i] / std::max(1.f - s0 * (1.f - k), 1e-6f);
        float q = (1.f - k) * cL * std::min(ds, 1.f);
        q = std::min(q, c[i]);
        if (q <= 0.f) continue;

        int z, y, x; decode(i, z, y, x);
        int nb[6]; int nn = 0;
        nb[nn++] = idx(z, y, x == Nx - 1 ? 0 : x + 1);
        nb[nn++] = idx(z, y, x == 0 ? Nx - 1 : x - 1);
        nb[nn++] = idx(z, y == Ny - 1 ? 0 : y + 1, x);
        nb[nn++] = idx(z, y == 0 ? Ny - 1 : y - 1, x);
        if (Nz > 1) {
            nb[nn++] = idx(z == Nz - 1 ? 0 : z + 1, y, x);
            nb[nn++] = idx(z == 0 ? Nz - 1 : z - 1, y, x);
        }
        float wsum = 0.f, w[6];
        for (int a = 0; a < nn; ++a) {
            const int j = nb[a];
            w[a] = (phi[j] == 0) ? std::max(1.f - s_new[j], 0.f) : 0.f;
            wsum += w[a];
        }
        if (wsum <= 0.f) continue;                     // no liquid neighbour: keep solute
        delta[i] -= q;
        for (int a = 0; a < nn; ++a)
            if (w[a] > 0.f) delta[nb[a]] += q * w[a] / wsum;
    }
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N_; ++i)
        c[i] += delta[i];
}

// Conservative face-flux diffusion of the liquid composition.  The face
// diffusivity is D times the smaller liquid fraction of the two cells, so no
// solute flows through fully solid cells.
void GPKGrid::step_diffusion_partitioned(float dt) {
    if (D_ <= 0.f) return;
    const float k = thermo_->k_partition;
    const int N_ = N();
    std::vector<float> s, cL(N_);
    cell_solid_fraction(s);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N_; ++i)
        cL[i] = c[i] / std::max(1.f - s[i] * (1.f - k), 1e-6f);

    const float coef = dt * D_ / (dx * dx);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N_; ++i) {
        int z, y, x; decode(i, z, y, x);
        int nb[6]; int nn = 0;
        nb[nn++] = idx(z, y, x == Nx - 1 ? 0 : x + 1);
        nb[nn++] = idx(z, y, x == 0 ? Nx - 1 : x - 1);
        nb[nn++] = idx(z, y == Ny - 1 ? 0 : y + 1, x);
        nb[nn++] = idx(z, y == 0 ? Ny - 1 : y - 1, x);
        if (Nz > 1) {
            nb[nn++] = idx(z == Nz - 1 ? 0 : z + 1, y, x);
            nb[nn++] = idx(z == 0 ? Nz - 1 : z - 1, y, x);
        }
        const float li = 1.f - s[i];
        float flux = 0.f;
        for (int a = 0; a < nn; ++a) {
            const int j = nb[a];
            const float lf = std::min(li, 1.f - s[j]);
            if (lf > 0.f) flux += lf * (cL[j] - cL[i]);
        }
        c_new_[i] = c[i] + coef * flux;
    }
    std::swap(c, c_new_);
}

// =========================================================================
// Static recovery of the dislocation-density field.
//   d(rho)/dt = -recovery_rate * rho   ->   rho(t+dt) = rho(t) * exp(-r dt)
// The exponential update is unconditionally stable for any dt and r >= 0.
// =========================================================================

void GPKGrid::step_recovery(float dt) {
    const ThermoModel* th = thermo_.get();
    if (!th->dislocation_enabled || th->recovery_rate <= 0.f || dt <= 0.f) return;

    const float decay = std::exp(-th->recovery_rate * dt);
    const int   N_    = N();
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N_; ++i)
        rho[i] *= decay;
}


} // namespace gpk


