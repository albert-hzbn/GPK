/* Default constitutive behavior for ThermoModel. */
#include "gpk_core.hpp"

#include <algorithm>
#include <cmath>

namespace gpk {

float ThermoModel::interfacial_energy(float nx, float ny, float nz,
                                       float g_val) const {
    if (eps4 == 0.f) return sigma0;

    // Rotate lab-frame normal into the crystal frame.
    // For 2-D simulations g is a rotation about z; nz stays exactly 0.
    const float cg = std::cos(g_val), sg = std::sin(g_val);
    const float nc_x =  nx * cg + ny * sg;
    const float nc_y = -nx * sg + ny * cg;
    const float nc_z = nz;

    float aniso;
    if (nc_z == 0.f) {
        // 2-D: sigma = sigma0 * (1 + eps4*cos(4*theta_c)).
        aniso = std::cos(4.f * std::atan2(nc_y, nc_x));
    } else {
        // 3-D cubic: sigma = sigma0 * (1 + eps4*(nx^4+ny^4+nz^4-3/5)).
        aniso = nc_x*nc_x*nc_x*nc_x + nc_y*nc_y*nc_y*nc_y
              + nc_z*nc_z*nc_z*nc_z - 0.6f;
    }
    return sigma0 * (1.f + eps4 * aniso);
}

float ThermoModel::mobility(float /*nx*/, float /*ny*/, float /*nz*/,
                             float g_parent, float g_product,
                             float T_val) const {
    float dg = std::fabs(g_parent - g_product);
    dg = std::fmod(dg + (float)M_PI * 0.5f, (float)M_PI * 0.5f);
    if (dg > (float)M_PI * 0.25f) dg = (float)M_PI * 0.5f - dg;

    const float theta_m = 15.f * (float)M_PI / 180.f;
    float mis;
    if      (dg < 1e-6f)    mis = 0.f;
    else if (dg >= theta_m) mis = 1.f;
    else { float frac = dg / theta_m; mis = frac * (1.f - std::log(frac)); }

    float arr = (Q > 0.f) ? std::exp(-Q / (Rgas * std::max(T_val, 1.f))) : 1.f;
    return M0 * mis * arr;
}

// =========================================================================
// 3-D separable Gaussian filter (periodic BC)
// =========================================================================

} // namespace gpk

