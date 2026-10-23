/* Elastic, thermodynamic, capillary, and projected interface velocities. */
#include "gpk_core.hpp"

#include <omp.h>
#include <algorithm>
#include <cmath>

namespace gpk {

void GPKGrid::step_elastic() {
    const ThermoModel* th = thermo_.get();
    if (!th->elastic_enabled || th->elastic_modulus <= 0.f ||
        (th->eigenstrain_coeff == 0.f && th->lambda_s == 0.f)) {
        std::fill(E_el.begin(), E_el.end(), 0.f);
        return;
    }

    const int N_ = N();
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N_; ++i) {
        float eps = th->eigenstrain_coeff * (c[i] - th->c_ref);
        if (th->lambda_s != 0.f) {
            // Isotropic magnetostriction eigenstrain: superposed
            // on the composition-driven term so existing composition-only and
            // new orientation-only models both reduce to their own single term.
            float cg = std::cos(g[i] - th->H_theta);
            eps += 1.5f * th->lambda_s * (cg * cg - (1.f / 3.f));
        }
        eps_star_[i]   = eps;
        eps_smooth_[i] = eps;
    }

    const int iters = std::max(1, th->elastic_smooth_iters);
    for (int it = 0; it < iters; ++it) {
        // Use raw pointers so -O3 cannot cache eps_smooth_.data() across the swap.
        const float* __restrict__ src = eps_smooth_.data();
        float*       __restrict__ dst = eps_tmp_.data();
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < N_; ++i) {
            int z, y, x; decode(i, z, y, x);
            const int xm = (x==0    ? Nx-1 : x-1), xp = (x==Nx-1 ? 0 : x+1);
            const int ym = (y==0    ? Ny-1 : y-1), yp = (y==Ny-1 ? 0 : y+1);
            const int zm = (z==0    ? Nz-1 : z-1), zp = (z==Nz-1 ? 0 : z+1);

            float avg = src[idx(z,y,xm)] + src[idx(z,y,xp)]
                      + src[idx(z,ym,x)] + src[idx(z,yp,x)];
            if (Nz > 1) {
                avg += src[idx(zm,y,x)] + src[idx(zp,y,x)];
                avg /= 6.f;
            } else {
                avg /= 4.f;
            }
            dst[i] = 0.5f * src[i] + 0.5f * avg;
        }
        eps_smooth_.swap(eps_tmp_);
    }

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N_; ++i) {
        float d  = eps_star_[i] - eps_smooth_[i];
        E_el[i]  = 0.5f * th->elastic_modulus * d * d;
    }
}

// Evaluate the common constitutive law on the reconstructed interface.

void GPKGrid::step_velocities() {
    const ThermoModel* th = thermo_.get();
    std::fill(vn_f.begin(), vn_f.end(), 0.f);

    #pragma omp parallel for schedule(static)
    for (int a = 0; a < static_cast<int>(active_cells_.size()); ++a) {
        const int i = active_cells_[a];

        int z, y, x; decode(i, z, y, x);
        const int xm = (x==0    ? Nx-1 : x-1), xp = (x==Nx-1 ? 0 : x+1);
        const int ym = (y==0    ? Ny-1 : y-1), yp = (y==Ny-1 ? 0 : y+1);
        const int zm = (z==0    ? Nz-1 : z-1), zp = (z==Nz-1 ? 0 : z+1);
        // Phase selection: collect unique neighbouring labels on the stack.
        if (th->phase_selection) {
            const int32_t ngh[6] = {
                phi[idx(z,y,xm)], phi[idx(z,y,xp)], phi[idx(z,ym,x)],
                phi[idx(z,yp,x)], phi[idx(zm,y,x)], phi[idx(zp,y,x)]
            };
            int32_t cand[6]; int ncand = 0;
            for (int k = 0; k < 6; ++k) {
                int32_t q = ngh[k];
                if (q == phi[i]) continue;
                bool seen = false;
                for (int c = 0; c < ncand; ++c) if (cand[c] == q) { seen = true; break; }
                if (!seen) cand[ncand++] = q;
            }
            int sel = th->select_product_phase(phi[i], c[i], T[i], cand, ncand);
            if (sel >= 0) phi_comp[i] = sel;
        }

        const int32_t qid = phi_comp[i];

        // Find orientation of the competing grain
        float g_p = g[i], g_q = g_p;
        float c_q = c[i];
        const int nei[6] = {
            idx(z,y,xm), idx(z,y,xp), idx(z,ym,x), idx(z,yp,x), idx(zm,y,x), idx(zp,y,x)
        };
        for (int k = 0; k < 6; ++k)
            if (phi[nei[k]] == qid) {
                g_q = g[nei[k]];
                c_q = c[nei[k]];
                break;
            }
        // Persist g_q so step_alpha() can sync g[i] to it if this cell
        // converts to grain qid this step -- otherwise g[i] would go stale
        // (keep the OLD grain's orientation forever), silently corrupting
        // every orientation-dependent term (interfacial anisotropy,
        // Read-Shockley mobility, magnetic anisotropy/Zeeman energy) for any
        // cell that has ever changed owner.
        g_comp[i] = g_q;

        const float c_iface = th->interface_composition(phi[i], qid, c[i], c_q);
        float dG  = th->bulk_driving_force(phi[i], qid, c_iface, T[i]);
        float sig = th->interfacial_energy_pair_comp(nx_f[i], ny_f[i], nz_f[i], g_p, phi[i], qid, c[i]);
        float mob = th->mobility(nx_f[i], ny_f[i], nz_f[i], g_p, g_q, T[i]);

        if (th->elastic_enabled) {
            float E_prod = E_el[i];
            for (int k = 0; k < 6; ++k)
                if (phi[nei[k]] == qid) { E_prod = E_el[nei[k]]; break; }
            dG += (E_prod - E_el[i]);
        }

        // Stored dislocation energy: E_rho = disloc_coeff * rho (Taylor relation).
        // The boundary migrates toward the higher-rho side, so the parent grain's
        // driving force gains E_rho(product) - E_rho(parent). Same structure as the
        // elastic term above.
        if (th->dislocation_enabled) {
            float rho_prod = rho[i];
            for (int k = 0; k < 6; ++k)
                if (phi[nei[k]] == qid) { rho_prod = rho[nei[k]]; break; }
            dG += th->disloc_coeff * (rho_prod - rho[i]);
        }

        if (th->magnetic_enabled) {
            dG += th->E_mag(g_q) - th->E_mag(g_p);
        }

        vn_f[i] = mob * (dG - sig * kappa_f[i]);
    }
}

// Project velocity only among cells belonging to the same label pair.

void GPKGrid::step_velocity_projection() {
    // A constitutive normal velocity cannot be resolved tangentially below
    // one cell. Project it on each resident/candidate pair before integrating
    // alpha; unrelated interfaces at a junction are never mixed.  Evaluate
    // the compact Gaussian stencil only at interface cells.  This is O(N_G r^d)
    // and, unlike one full-domain convolution per label pair, remains linear
    // when a high-resolution polycrystal contains many grains.
    alpha_new_ = vn_f;
    const float velocity_sigma=velocity_gauss_.sigma_cells;
    const int radius = std::max(1, static_cast<int>(std::ceil(3.f*velocity_sigma)));
    const float inv_two_sigma2 = 0.5f/(velocity_sigma*velocity_sigma);
    #pragma omp parallel for schedule(static)
    for (int a=0; a<static_cast<int>(active_cells_.size()); ++a) {
        const int i=active_cells_[a];
        if (phi[i] == phi_comp[i]) continue;
        const int32_t low = std::min(phi[i],phi_comp[i]);
        const int32_t high = std::max(phi[i],phi_comp[i]);
        const float orientation_i=phi[i]==low?1.f:-1.f;
        int z0,y0,x0; decode(i,z0,y0,x0);
        double numerator = 0.0, denominator = 0.0;
        const int dz0 = Nz > 1 ? -radius : 0;
        const int dz1 = Nz > 1 ?  radius : 0;
        for (int dz=dz0; dz<=dz1; ++dz)
        for (int dy=-radius; dy<=radius; ++dy)
        for (int dxs=-radius; dxs<=radius; ++dxs) {
            int z=(z0+dz)%Nz; if(z<0) z+=Nz;
            int y=(y0+dy)%Ny; if(y<0) y+=Ny;
            int x=(x0+dxs)%Nx; if(x<0) x+=Nx;
            const int j=idx(z,y,x);
            if (!mask[j] || phi[j] == phi_comp[j] ||
                std::min(phi[j],phi_comp[j]) != low ||
                std::max(phi[j],phi_comp[j]) != high) continue;
            const float r2=static_cast<float>(dxs*dxs+dy*dy+dz*dz);
            const double weight=std::exp(-r2*inv_two_sigma2);
            const float orientation_j=phi[j]==low?1.f:-1.f;
            numerator += weight*orientation_j*vn_f[j]; denominator += weight;
        }
        if (denominator > 0.0)
            alpha_new_[i]=orientation_i*static_cast<float>(numerator/denominator);
    }
    vn_f.swap(alpha_new_);
}

} // namespace gpk


