#pragma once
/*
 * Lightweight PPM image writer — no external dependencies.
 * Writes P6 (binary) PPM suitable for viewing in most image tools.
 *
 * All render_* functions that accept alpha + phi_comp produce smooth,
 * anti-aliased boundaries: each interface cell is blended between its
 * own grain colour (weight = alpha) and its competitor's colour
 * (weight = 1-alpha), eliminating the staircase-pixelated boundary that
 * would appear if the raw integer phi field were rendered directly.
 */
#include <fstream>
#include <vector>
#include <cstdint>
#include <string>
#include <algorithm>
#include <cmath>

namespace gpk {

// ── basic helpers ──────────────────────────────────────────────────────────

inline uint8_t to_u8(float v) {
    return static_cast<uint8_t>(std::clamp(v, 0.f, 1.f) * 255.f + 0.5f);
}

inline void write_ppm(const std::string& path, int width, int height,
                      const std::vector<uint8_t>& rgb) {
    std::ofstream f(path, std::ios::binary);
    f << "P6\n" << width << " " << height << "\n255\n";
    f.write(reinterpret_cast<const char*>(rgb.data()),
            static_cast<std::streamsize>(rgb.size()));
}

// Deterministic, always-bright colour from a grain label (Knuth hash).
inline void grain_rgb(int32_t label, uint8_t& r, uint8_t& g, uint8_t& b) {
    auto h = static_cast<uint32_t>(label) * 2654435761u;
    r = 80 + (h & 0x7F);
    g = 80 + ((h >>  8) & 0x7F);
    b = 80 + ((h >> 16) & 0x7F);
}

// Linear blend of two RGB colours, t = weight of first colour.
inline void blend_rgb(float t,
                      uint8_t r1, uint8_t g1, uint8_t b1,
                      uint8_t r2, uint8_t g2, uint8_t b2,
                      uint8_t& r, uint8_t& g, uint8_t& b) {
    r = to_u8(t * (r1 / 255.f) + (1.f - t) * (r2 / 255.f));
    g = to_u8(t * (g1 / 255.f) + (1.f - t) * (g2 / 255.f));
    b = to_u8(t * (b1 / 255.f) + (1.f - t) * (b2 / 255.f));
}

// Viridis-like colormap (dark purple → teal → yellow) for scalar fields.
inline void viridis(float t, uint8_t& r, uint8_t& g, uint8_t& b) {
    t = std::clamp(t, 0.f, 1.f);
    float rr = t < 0.5f ? (0.27f + 0.1f * t)         : (0.2f + 1.6f * (t - 0.5f));
    float gg = t < 0.5f ? (0.4f * t)                  : (0.2f + 1.2f * (t - 0.5f));
    float bb = t < 0.5f ? (0.5f - t)                  : (0.5f - (t - 0.5f));
    r = to_u8(rr); g = to_u8(gg); b = to_u8(bb);
}

// ── render functions ───────────────────────────────────────────────────────

/*
 * render_grains
 * Grain-label image with smooth anti-aliased boundaries.
 *
 * At every interface cell the pixel is blended:
 *   colour = alpha[i] * grain_colour(phi[i])
 *          + (1 - alpha[i]) * grain_colour(phi_comp[i])
 *
 * This converts the staircase step-change between integer labels into a
 * smooth gradient, exactly representing the sub-cell boundary position
 * that alpha encodes.
 */
inline void render_grains(const std::vector<int32_t>& phi,
                           const std::vector<float>&   alpha,
                           const std::vector<int32_t>& phi_comp,
                           const std::vector<uint8_t>& mask,
                           int Ny, int Nx,
                           const std::string& path) {
    std::vector<uint8_t> rgb(Ny * Nx * 3);
    for (int i = 0; i < Ny * Nx; ++i) {
        uint8_t r1, g1, b1;
        grain_rgb(phi[i], r1, g1, b1);

        if (mask[i] && phi_comp[i] != phi[i]) {
            uint8_t r2, g2, b2;
            grain_rgb(phi_comp[i], r2, g2, b2);
            blend_rgb(std::clamp(alpha[i], 0.f, 1.f),
                      r1, g1, b1, r2, g2, b2,
                      rgb[3*i], rgb[3*i+1], rgb[3*i+2]);
        } else {
            rgb[3*i] = r1; rgb[3*i+1] = g1; rgb[3*i+2] = b1;
        }
    }
    write_ppm(path, Nx, Ny, rgb);
}

// Legacy overload — no alpha blending (kept for callers that don't have alpha).
inline void render_grains(const std::vector<int32_t>& phi,
                           int Ny, int Nx,
                           const std::string& path) {
    std::vector<uint8_t> rgb(Ny * Nx * 3);
    for (int i = 0; i < Ny * Nx; ++i)
        grain_rgb(phi[i], rgb[3*i], rgb[3*i+1], rgb[3*i+2]);
    write_ppm(path, Nx, Ny, rgb);
}

/*
 * render_phase
 * Binary phase image (0 = liquid/matrix, 1 = solid/precipitate) with smooth
 * interface: interface cells are blended between white (liquid) and blue (solid).
 */
inline void render_phase(const std::vector<int32_t>& phi,
                          const std::vector<float>&   alpha,
                          const std::vector<uint8_t>& mask,
                          int Ny, int Nx,
                          const std::string& path) {
    // liquid colour: (230, 230, 230)   solid colour: (30, 100, 200)
    const uint8_t lr = 230, lg = 230, lb = 230;
    const uint8_t sr = 30,  sg = 100, sb = 200;

    std::vector<uint8_t> rgb(Ny * Nx * 3);
    for (int i = 0; i < Ny * Nx; ++i) {
        if (phi[i] == 1) {
            if (mask[i]) {
                // Interface cell of a solid grain: blend solid ↔ liquid using alpha.
                float a = std::clamp(alpha[i], 0.f, 1.f);
                blend_rgb(a, sr, sg, sb, lr, lg, lb,
                          rgb[3*i], rgb[3*i+1], rgb[3*i+2]);
            } else {
                rgb[3*i] = sr; rgb[3*i+1] = sg; rgb[3*i+2] = sb;
            }
        } else {
            if (mask[i]) {
                // Interface cell of the liquid — competing with solid.
                float a = std::clamp(alpha[i], 0.f, 1.f);
                blend_rgb(a, lr, lg, lb, sr, sg, sb,
                          rgb[3*i], rgb[3*i+1], rgb[3*i+2]);
            } else {
                rgb[3*i] = lr; rgb[3*i+1] = lg; rgb[3*i+2] = lb;
            }
        }
    }
    write_ppm(path, Nx, Ny, rgb);
}

// Legacy overload — no blending.
inline void render_phase(const std::vector<int32_t>& phi,
                          int Ny, int Nx,
                          const std::string& path) {
    std::vector<uint8_t> rgb(Ny * Nx * 3);
    for (int i = 0; i < Ny * Nx; ++i) {
        if (phi[i] == 1) { rgb[3*i]=30;  rgb[3*i+1]=100; rgb[3*i+2]=200; }
        else             { rgb[3*i]=230; rgb[3*i+1]=230; rgb[3*i+2]=230; }
    }
    write_ppm(path, Nx, Ny, rgb);
}

/*
 * render_scalar
 * Scalar field (composition, temperature, …) rendered with viridis colormap.
 * No blending needed — the scalar field is already continuous.
 */
inline void render_scalar(const std::vector<float>& field,
                           int Ny, int Nx,
                           float vmin, float vmax,
                           const std::string& path) {
    std::vector<uint8_t> rgb(Ny * Nx * 3);
    float range = (vmax > vmin) ? (vmax - vmin) : 1.f;
    for (int i = 0; i < Ny * Nx; ++i) {
        float t = (field[i] - vmin) / range;
        viridis(t, rgb[3*i], rgb[3*i+1], rgb[3*i+2]);
    }
    write_ppm(path, Nx, Ny, rgb);
}

/*
 * render_rex
 * Recrystallisation image: deformed cells = grey, rex grains = coloured,
 * with smooth alpha-blended boundaries between deformed and rex regions.
 */
inline void render_rex(const std::vector<int32_t>& phi,
                        const std::vector<float>&   alpha,
                        const std::vector<int32_t>& phi_comp,
                        const std::vector<uint8_t>& mask,
                        int Ny, int Nx,
                        const std::string& path) {
    const uint8_t dr = 120, dg = 120, db = 120;   // deformed: grey
    std::vector<uint8_t> rgb(Ny * Nx * 3);

    for (int i = 0; i < Ny * Nx; ++i) {
        if (phi[i] == 0) {
            // Deformed matrix cell
            if (mask[i] && phi_comp[i] != 0) {
                // Blending with incoming rex grain colour
                uint8_t rr, gr, br;
                grain_rgb(phi_comp[i], rr, gr, br);
                float a = std::clamp(alpha[i], 0.f, 1.f);
                blend_rgb(a, dr, dg, db, rr, gr, br,
                          rgb[3*i], rgb[3*i+1], rgb[3*i+2]);
            } else {
                rgb[3*i] = dr; rgb[3*i+1] = dg; rgb[3*i+2] = db;
            }
        } else {
            uint8_t r1, g1, b1;
            grain_rgb(phi[i], r1, g1, b1);
            if (mask[i] && phi_comp[i] != phi[i]) {
                uint8_t r2, g2, b2;
                if (phi_comp[i] == 0) { r2 = dr; g2 = dg; b2 = db; }
                else                  grain_rgb(phi_comp[i], r2, g2, b2);
                float a = std::clamp(alpha[i], 0.f, 1.f);
                blend_rgb(a, r1, g1, b1, r2, g2, b2,
                          rgb[3*i], rgb[3*i+1], rgb[3*i+2]);
            } else {
                rgb[3*i] = r1; rgb[3*i+1] = g1; rgb[3*i+2] = b1;
            }
        }
    }
    write_ppm(path, Nx, Ny, rgb);
}

// Legacy overload — no blending.
inline void render_rex(const std::vector<int32_t>& phi,
                        int Ny, int Nx,
                        const std::string& path) {
    std::vector<uint8_t> rgb(Ny * Nx * 3);
    for (int i = 0; i < Ny * Nx; ++i) {
        if (phi[i] == 0) { rgb[3*i]=120; rgb[3*i+1]=120; rgb[3*i+2]=120; }
        else { grain_rgb(phi[i], rgb[3*i], rgb[3*i+1], rgb[3*i+2]); }
    }
    write_ppm(path, Nx, Ny, rgb);
}

} // namespace gpk
