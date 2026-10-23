/* Interface detection and Gaussian-Hessian geometry reconstruction. */
#include "gpk_core.hpp"

#include <omp.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <unordered_set>

namespace gpk {

void GPKGrid::build_geometry_kernels() {
    geom_radius_ = gauss_.radius + 2;
    const int width = 2 * geom_radius_ + 1;
    geom_w0_.assign(width, 0.f);
    geom_w1_.assign(width, 0.f);
    geom_w2_.assign(width, 0.f);

    const int first_offset[4] = {-2, -1, 1, 2};
    const float first_coeff[4] = {1.f/12.f, -8.f/12.f, 8.f/12.f, -1.f/12.f};
    const int second_offset[5] = {-2, -1, 0, 1, 2};
    const float second_coeff[5] = {-1.f/12.f, 16.f/12.f, -30.f/12.f,
                                   16.f/12.f, -1.f/12.f};
    auto gaussian_at = [&](int offset) {
        if (offset < -gauss_.radius || offset > gauss_.radius) return 0.f;
        return gauss_.w[offset + gauss_.radius];
    };
    for (int o = -geom_radius_; o <= geom_radius_; ++o) {
        const int j = o + geom_radius_;
        geom_w0_[j] = gaussian_at(o);
        for (int k = 0; k < 4; ++k)
            geom_w1_[j] += first_coeff[k] * gaussian_at(o - first_offset[k]);
        for (int k = 0; k < 5; ++k)
            geom_w2_[j] += second_coeff[k] * gaussian_at(o - second_offset[k]);
    }
}

// =========================================================================
// Optional elastic coupling proxy (nonlocal coherent strain smoothing)
// =========================================================================

void GPKGrid::step_interface_mask() {
    const auto* p = phi.data();
    constexpr int block_size = 4096;
    const int nblocks = static_cast<int>(active_block_counts_.size());

    #pragma omp parallel for schedule(static)
    for (int block = 0; block < nblocks; ++block) {
        const int begin = block * block_size;
        const int end = std::min(begin + block_size, N());
        int count = 0;
        for (int i = begin; i < end; ++i) {
            int z, y, x; decode(i, z, y, x);
            const int xm = (x==0 ? Nx-1 : x-1), xp = (x==Nx-1 ? 0 : x+1);
            const int ym = (y==0 ? Ny-1 : y-1), yp = (y==Ny-1 ? 0 : y+1);
            const int zm = (z==0 ? Nz-1 : z-1), zp = (z==Nz-1 ? 0 : z+1);
            const int32_t v = p[i];
            const uint8_t active = static_cast<uint8_t>(
                (v != p[idx(z,y,xm)]) | (v != p[idx(z,y,xp)]) |
                (v != p[idx(z,ym,x)]) | (v != p[idx(z,yp,x)]) |
                (v != p[idx(zm,y,x)]) | (v != p[idx(zp,y,x)]));
            mask[i] = active;
            count += active;
        }
        active_block_counts_[block] = count;
    }

    active_block_offsets_[0] = 0;
    for (int block = 0; block < nblocks; ++block)
        active_block_offsets_[block + 1] = active_block_offsets_[block]
                                                + active_block_counts_[block];
    active_cells_.resize(active_block_offsets_[nblocks]);

    #pragma omp parallel for schedule(static)
    for (int block = 0; block < nblocks; ++block) {
        const int begin = block * block_size;
        const int end = std::min(begin + block_size, N());
        int out = active_block_offsets_[block];
        for (int i = begin; i < end; ++i)
            if (mask[i]) active_cells_[out++] = i;
    }
}

// =========================================================================
// Gaussian-Hessian geometry: indicator -> normal -> curvature.
// =========================================================================

void GPKGrid::step_geometry() {
    if (geom_use_global_separable_) {
        step_geometry_global_separable();
        return;
    }
    std::fill(nx_f.begin(), nx_f.end(), 0.f);
    std::fill(ny_f.begin(), ny_f.end(), 0.f);
    std::fill(nz_f.begin(), nz_f.end(), 0.f);
    std::fill(kappa_f.begin(), kappa_f.end(), 0.f);

    const int radius = geom_radius_;
    const float inv_dx = 1.f / dx;
    const float inv_dx2 = inv_dx * inv_dx;
    const float kappa_limit = 1.f / (gauss_.sigma_cells * dx);

    #pragma omp parallel for schedule(static)
    for (int a = 0; a < static_cast<int>(active_cells_.size()); ++a) {
        const int i = active_cells_[a];
        const int32_t resident = phi[i];
        int z0, y0, x0; decode(i, z0, y0, x0);
        float Ix=0.f, Iy=0.f, Iz=0.f, Ixx=0.f, Iyy=0.f, Izz=0.f;
        float Ixy=0.f, Ixz=0.f, Iyz=0.f;
        const int dz_begin = Nz > 1 ? -radius : 0;
        const int dz_end = Nz > 1 ? radius : 0;
        for (int dz=dz_begin; dz<=dz_end; ++dz) {
            int z=z0+dz; while(z<0)z+=Nz; while(z>=Nz)z-=Nz;
            const int jz=dz+radius;
            const float gz0=Nz>1?geom_w0_[jz]:1.f;
            const float gz1=Nz>1?geom_w1_[jz]:0.f;
            const float gz2=Nz>1?geom_w2_[jz]:0.f;
            for (int dy=-radius; dy<=radius; ++dy) {
                int y=y0+dy; while(y<0)y+=Ny; while(y>=Ny)y-=Ny;
                const int jy=dy+radius;
                const float gy0=geom_w0_[jy], gy1=geom_w1_[jy], gy2=geom_w2_[jy];
                for (int dxs=-radius; dxs<=radius; ++dxs) {
                    int x=x0+dxs; while(x<0)x+=Nx; while(x>=Nx)x-=Nx;
                    if (phi[idx(z,y,x)] != resident) continue;
                    const int jx=dxs+radius;
                    const float gx0=geom_w0_[jx], gx1=geom_w1_[jx], gx2=geom_w2_[jx];
                    Ix+=gx1*gy0*gz0; Iy+=gx0*gy1*gz0;
                    Ixx+=gx2*gy0*gz0; Iyy+=gx0*gy2*gz0; Ixy+=gx1*gy1*gz0;
                    if(Nz>1) {
                        Iz+=gx0*gy0*gz1; Izz+=gx0*gy0*gz2;
                        Ixz+=gx1*gy0*gz1; Iyz+=gx0*gy1*gz1;
                    }
                }
            }
        }
        Ix*=inv_dx; Iy*=inv_dx; Iz*=inv_dx;
        Ixx*=inv_dx2; Iyy*=inv_dx2; Izz*=inv_dx2;
        Ixy*=inv_dx2; Ixz*=inv_dx2; Iyz*=inv_dx2;
        const float grad_sq=Ix*Ix+Iy*Iy+Iz*Iz;
        if(grad_sq*dx*dx>1e-8f) {
            const float grad=std::sqrt(grad_sq);
            nx_f[i]=-Ix/grad;
            ny_f[i]=-Iy/grad;
            nz_f[i]=-Iz/grad;
            const float trace_H=Ixx+Iyy+Izz;
            const float gHg=Ix*Ix*Ixx+Iy*Iy*Iyy+Iz*Iz*Izz
                +2.f*(Ix*Iy*Ixy+Ix*Iz*Ixz+Iy*Iz*Iyz);
            float curvature=(gHg-grad_sq*trace_H)/(grad_sq*grad);
            if(std::fabs(curvature*dx)<1e-6f) curvature=0.f;
            kappa_f[i]=std::clamp(curvature,-kappa_limit,kappa_limit);
        }
    }

    #pragma omp parallel for schedule(static)
    for (int a=0; a<static_cast<int>(active_cells_.size()); ++a) {
        const int i=active_cells_[a];
        int z,y,x; decode(i,z,y,x);
        const int xm=(x==0?Nx-1:x-1),xp=(x==Nx-1?0:x+1);
        const int ym=(y==0?Ny-1:y-1),yp=(y==Ny-1?0:y+1);
        const int zm=(z==0?Nz-1:z-1),zp=(z==Nz-1?0:z+1);
        const int nei[6]={idx(z,y,xm),idx(z,y,xp),idx(z,ym,x),idx(z,yp,x),idx(zm,y,x),idx(zp,y,x)};
        const float score[6]={-nx_f[i],nx_f[i],-ny_f[i],ny_f[i],-nz_f[i],nz_f[i]};
        const int nface=Nz>1?6:4;
        int best=-1,best_support=-1; float best_score=-2.f;
        for(int k=0;k<nface;++k) {
            const int32_t q=phi[nei[k]]; if(q==phi[i]) continue;
            int support=0;
            for(int dz=(Nz>1?-1:0);dz<=(Nz>1?1:0);++dz)
            for(int dy=-1;dy<=1;++dy)
            for(int dxs=-1;dxs<=1;++dxs) {
                if(dxs==0&&dy==0&&dz==0) continue;
                int xx=x+dxs;if(xx<0)xx+=Nx;else if(xx>=Nx)xx-=Nx;
                int yy=y+dy;if(yy<0)yy+=Ny;else if(yy>=Ny)yy-=Ny;
                int zz=z+dz;if(zz<0)zz+=Nz;else if(zz>=Nz)zz-=Nz;
                support+=phi[idx(zz,yy,xx)]==q;
            }
            if(support>best_support||(support==best_support&&score[k]>best_score)) {
                best=k;best_support=support;best_score=score[k];
            }
        }
        phi_comp[i]=(best>=0)?phi[nei[best]]:phi[i];
    }
}

void GPKGrid::step_geometry_global_separable() {
    const int   N_   = N();
    const float idx12 = 1.f / (12.f * dx);
    const float idx_sq = 1.f / (dx * dx);
    const float idx12_sq = idx_sq / 12.f;
    const float idx144_sq = idx_sq / 144.f;

    // Collect unique grain labels present at the interface.
    // sort+unique is branch-free and avoids hash-map overhead for typical Ng.
    std::vector<int32_t> iface_grains;
    iface_grains.reserve(256);
    for (int i = 0; i < N_; ++i)
        if (mask[i]) iface_grains.push_back(phi[i]);
    std::sort(iface_grains.begin(), iface_grains.end());
    iface_grains.erase(std::unique(iface_grains.begin(), iface_grains.end()),
                       iface_grains.end());
    const int Ng = (int)iface_grains.size();

    std::fill(nx_f.begin(),    nx_f.end(),    0.f);
    std::fill(ny_f.begin(),    ny_f.end(),    0.f);
    std::fill(nz_f.begin(),    nz_f.end(),    0.f);
    std::fill(kappa_f.begin(), kappa_f.end(), 0.f);

    for (int gi = 0; gi < Ng; ++gi) {
        const int32_t p   = iface_grains[gi];
        float* I_buf    = geom_I_.data();
        float* I_smooth = geom_smooth_.data();
        float* t1       = geom_tmp1_.data();
        float* t2       = geom_tmp2_.data();

        // Binary indicator for grain p
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < N_; ++i)
            I_buf[i] = (phi[i] == p) ? 1.f : 0.f;

        // 3-D separable Gaussian smooth in one shared workspace.
        const int    krad = gauss_.radius;
        const float* kw   = gauss_.w.data();
        // x pass: indicator -> first workspace
        #pragma omp parallel for collapse(2) schedule(static)
        for (int z = 0; z < Nz; ++z)
        for (int y = 0; y < Ny; ++y)
        for (int x = 0; x < Nx; ++x) {
            float acc = 0.f;
            for (int k = 0, ki = -krad; ki <= krad; ++k, ++ki) {
                int xx = x+ki; if (xx<0) xx+=Nx; else if (xx>=Nx) xx-=Nx;
                acc += kw[k] * I_buf[(z*Ny+y)*Nx + xx];
            }
            t1[(z*Ny+y)*Nx + x] = acc;
        }
        // y pass: first workspace -> second workspace
        #pragma omp parallel for collapse(2) schedule(static)
        for (int z = 0; z < Nz; ++z)
        for (int y = 0; y < Ny; ++y)
        for (int x = 0; x < Nx; ++x) {
            float acc = 0.f;
            for (int k = 0, ki = -krad; ki <= krad; ++k, ++ki) {
                int yy = y+ki; if (yy<0) yy+=Ny; else if (yy>=Ny) yy-=Ny;
                acc += kw[k] * t1[(z*Ny+yy)*Nx + x];
            }
            t2[(z*Ny+y)*Nx + x] = acc;
        }
        // z pass: second workspace -> smoothed indicator
        if (Nz == 1) {
            std::memcpy(I_smooth, t2, sizeof(float) * N_);
        } else {
            #pragma omp parallel for collapse(2) schedule(static)
            for (int z = 0; z < Nz; ++z)
            for (int y = 0; y < Ny; ++y)
            for (int x = 0; x < Nx; ++x) {
                float acc = 0.f;
                for (int k = 0, ki = -krad; ki <= krad; ++k, ++ki) {
                    int zz = z+ki; if (zz<0) zz+=Nz; else if (zz>=Nz) zz-=Nz;
                    acc += kw[k] * t2[(zz*Ny+y)*Nx + x];
                }
                I_smooth[(z*Ny+y)*Nx + x] = acc;
            }
        }

        // Evaluate normal and curvature directly from the gradient and Hessian
        // of the smooth indicator.  This is algebraically div(-grad(I)/|grad(I)|)
        // but avoids differentiating an ill-conditioned normalized-gradient
        // field in the nearly constant Gaussian tails.
        #pragma omp parallel for collapse(2) schedule(static)
        for (int z = 0; z < Nz; ++z)
        for (int y = 0; y < Ny; ++y)
        for (int x = 0; x < Nx; ++x) {
            int i = idx(z, y, x);
            if (!mask[i] || phi[i] != p) continue;

            int xm = (x==0    ? Nx-1 : x-1), xp = (x==Nx-1 ? 0 : x+1);
            int ym = (y==0    ? Ny-1 : y-1), yp = (y==Ny-1 ? 0 : y+1);
            int zm = (z==0    ? Nz-1 : z-1), zp = (z==Nz-1 ? 0 : z+1);
            int xmm = (x < 2 ? x + Nx - 2 : x - 2), xpp = (x >= Nx-2 ? x - Nx + 2 : x + 2);
            int ymm = (y < 2 ? y + Ny - 2 : y - 2), ypp = (y >= Ny-2 ? y - Ny + 2 : y + 2);
            int zmm = (z < 2 ? z + Nz - 2 : z - 2), zpp = (z >= Nz-2 ? z - Nz + 2 : z + 2);

            const float I0 = I_smooth[i];
            const float Ix = (I_smooth[idx(z,y,xmm)] - 8.f*I_smooth[idx(z,y,xm)]
                            + 8.f*I_smooth[idx(z,y,xp)] - I_smooth[idx(z,y,xpp)]) * idx12;
            const float Iy = (I_smooth[idx(z,ymm,x)] - 8.f*I_smooth[idx(z,ym,x)]
                            + 8.f*I_smooth[idx(z,yp,x)] - I_smooth[idx(z,ypp,x)]) * idx12;
            const float Iz = (Nz > 1)
                           ? (I_smooth[idx(zmm,y,x)] - 8.f*I_smooth[idx(zm,y,x)]
                            + 8.f*I_smooth[idx(zp,y,x)] - I_smooth[idx(zpp,y,x)]) * idx12
                           : 0.f;

            const float Ixx = (-I_smooth[idx(z,y,xpp)] + 16.f*I_smooth[idx(z,y,xp)]
                              - 30.f*I0 + 16.f*I_smooth[idx(z,y,xm)]
                              - I_smooth[idx(z,y,xmm)]) * idx12_sq;
            const float Iyy = (-I_smooth[idx(z,ypp,x)] + 16.f*I_smooth[idx(z,yp,x)]
                              - 30.f*I0 + 16.f*I_smooth[idx(z,ym,x)]
                              - I_smooth[idx(z,ymm,x)]) * idx12_sq;
            const float Izz = (Nz > 1)
                            ? (-I_smooth[idx(zpp,y,x)] + 16.f*I_smooth[idx(zp,y,x)]
                              - 30.f*I0 + 16.f*I_smooth[idx(zm,y,x)]
                              - I_smooth[idx(zmm,y,x)]) * idx12_sq
                            : 0.f;
            // Tensor-product fourth-order centered mixed derivatives.  Using
            // second-order cross terms with fourth-order axial terms produced
            // a direction-dependent Hessian error on voxelised 3-D spheres.
            const int xo[4] = {xmm, xm, xp, xpp};
            const int yo[4] = {ymm, ym, yp, ypp};
            const int zo[4] = {zmm, zm, zp, zpp};
            const float dc[4] = {1.f, -8.f, 8.f, -1.f};
            float Ixy_sum = 0.f, Ixz_sum = 0.f, Iyz_sum = 0.f;
            for (int a = 0; a < 4; ++a)
            for (int b = 0; b < 4; ++b) {
                const float w = dc[a] * dc[b];
                Ixy_sum += w * I_smooth[idx(z, yo[b], xo[a])];
                if (Nz > 1) {
                    Ixz_sum += w * I_smooth[idx(zo[b], y, xo[a])];
                    Iyz_sum += w * I_smooth[idx(zo[b], yo[a], x)];
                }
            }
            const float Ixy = Ixy_sum * idx144_sq;
            const float Ixz = (Nz > 1) ? Ixz_sum * idx144_sq : 0.f;
            const float Iyz = (Nz > 1) ? Iyz_sum * idx144_sq : 0.f;

            const float grad_sq = Ix*Ix + Iy*Iy + Iz*Iz;
            // The dimensionless threshold rejects cancellation points at
            // multi-junctions where the Hessian quotient is ill-conditioned.
            // A fixed SI-unit threshold changes meaning when dx is refined.
            if (grad_sq * dx * dx > 1e-8f) {
                const float inv_grad = 1.f / std::sqrt(grad_sq);
                nx_f[i] = -Ix * inv_grad;
                ny_f[i] = -Iy * inv_grad;
                nz_f[i] = -Iz * inv_grad;

                const float trace_H = Ixx + Iyy + Izz;
                const float gHg = Ix*Ix*Ixx + Iy*Iy*Iyy + Iz*Iz*Izz
                                + 2.f*(Ix*Iy*Ixy + Ix*Iz*Ixz + Iy*Iz*Iyz);
                kappa_f[i] = (gHg - grad_sq*trace_H)
                           / (grad_sq * std::sqrt(grad_sq));
                if (std::fabs(kappa_f[i] * dx) < 1e-6f) kappa_f[i] = 0.f;
                // Curvature above 1/ell_G belongs to geometry smaller than the
                // Gaussian measurement scale and should not control kinetics
                // or the global timestep.
                const float kappa_limit = 1.f / (gauss_.sigma_cells * dx);
                kappa_f[i] = std::clamp(kappa_f[i], -kappa_limit, kappa_limit);
            }
        }
    }

    // Among unlike face neighbours, prefer the locally supported label and
    // use outward-normal alignment as the tie-breaker. This suppresses
    // checkerboard interpenetration at moving multi-junctions; unresolved
    // terminal labels are completed separately after the switching pass.
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N_; ++i) {
        if (!mask[i]) continue;
        int z, y, x; decode(i, z, y, x);
        const int xm = (x==0    ? Nx-1 : x-1), xp = (x==Nx-1 ? 0 : x+1);
        const int ym = (y==0    ? Ny-1 : y-1), yp = (y==Ny-1 ? 0 : y+1);
        const int zm = (z==0    ? Nz-1 : z-1), zp = (z==Nz-1 ? 0 : z+1);
        const int nei[6] = {
            idx(z,y,xm), idx(z,y,xp), idx(z,ym,x),
            idx(z,yp,x), idx(zm,y,x), idx(zp,y,x)
        };
        const float score[6] = {
            -nx_f[i], nx_f[i], -ny_f[i], ny_f[i], -nz_f[i], nz_f[i]
        };
        int best = -1, best_support = -1;
        float best_score = -2.f;
        const int nface = (Nz > 1) ? 6 : 4;
        for (int k = 0; k < nface; ++k) {
            const int32_t q = phi[nei[k]];
            if (q == phi[i]) continue;
            int support = 0;
            for (int dz = (Nz > 1 ? -1 : 0); dz <= (Nz > 1 ? 1 : 0); ++dz)
            for (int dy = -1; dy <= 1; ++dy)
            for (int dx_ = -1; dx_ <= 1; ++dx_) {
                if (dx_ == 0 && dy == 0 && dz == 0) continue;
                int xx = x + dx_; if (xx < 0) xx += Nx; else if (xx >= Nx) xx -= Nx;
                int yy = y + dy;  if (yy < 0) yy += Ny; else if (yy >= Ny) yy -= Ny;
                int zz = z + dz;  if (zz < 0) zz += Nz; else if (zz >= Nz) zz -= Nz;
                support += (phi[idx(zz,yy,xx)] == q);
            }
            if (support > best_support ||
                (support == best_support && score[k] > best_score)) {
                best = k; best_support = support; best_score = score[k];
            }
        }
        phi_comp[i] = (best >= 0) ? phi[nei[best]] : phi[i];
    }
}


} // namespace gpk


