/* Periodic separable Gaussian filtering. */
#include "gpk_core.hpp"

#include <omp.h>
#include <cstring>

namespace gpk {

void gaussian_filter_3d(const float* __restrict__ in,
                        float*       __restrict__ out,
                        float*       __restrict__ tmp1,
                        float*       __restrict__ tmp2,
                        int Nz, int Ny, int Nx,
                        const GaussKernel& K) {
    const int    krad = K.radius;
    const float* kw   = K.w.data();
    // x pass: input -> tmp1
    #pragma omp parallel for schedule(static)
    for (int z = 0; z < Nz; ++z)
    for (int y = 0; y < Ny; ++y) {
        const float* row = in   + (z*Ny + y) * Nx;
        float*       dst = tmp1 + (z*Ny + y) * Nx;
        for (int x = 0; x < Nx; ++x) {
            float acc = 0.f;
            for (int k = 0, ki = -krad; ki <= krad; ++k, ++ki) {
                int xx = x + ki;
                if (xx < 0) xx += Nx; else if (xx >= Nx) xx -= Nx;
                acc += kw[k] * row[xx];
            }
            dst[x] = acc;
        }
    }
    // y pass: tmp1 -> tmp2
    #pragma omp parallel for schedule(static)
    for (int z = 0; z < Nz; ++z)
    for (int y = 0; y < Ny; ++y) {
        float* dst = tmp2 + (z*Ny + y) * Nx;
        for (int x = 0; x < Nx; ++x) {
            float acc = 0.f;
            for (int k = 0, ki = -krad; ki <= krad; ++k, ++ki) {
                int yy = y + ki;
                if (yy < 0) yy += Ny; else if (yy >= Ny) yy -= Ny;
                acc += kw[k] * tmp1[(z*Ny + yy) * Nx + x];
            }
            dst[x] = acc;
        }
    }
    // z pass: tmp2 -> output (no-op in 2-D)
    if (Nz == 1) { std::memcpy(out, tmp2, sizeof(float) * Ny * Nx); return; }

    #pragma omp parallel for schedule(static)
    for (int z = 0; z < Nz; ++z)
    for (int y = 0; y < Ny; ++y) {
        float* dst = out + (z*Ny + y) * Nx;
        for (int x = 0; x < Nx; ++x) {
            float acc = 0.f;
            for (int k = 0, ki = -krad; ki <= krad; ++k, ++ki) {
                int zz = z + ki;
                if (zz < 0) zz += Nz; else if (zz >= Nz) zz -= Nz;
                acc += kw[k] * tmp2[(zz*Ny + y) * Nx + x];
            }
            dst[x] = acc;
        }
    }
}

// =========================================================================
// GPKGrid constructor
// =========================================================================

} // namespace gpk

