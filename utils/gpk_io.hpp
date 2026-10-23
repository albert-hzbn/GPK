#pragma once
/*
 * gpk_io.hpp — Unified output dispatcher for GPK simulations.
 *
 * OutputWriter reads [output] config keys and writes snapshots in the
 * requested format(s):
 *
 *   format = vtk    → VTK Legacy ASCII (.vtk)
 *   format = csv    → Column-wise ASCII (.csv)
 *   format = ppm    → PPM image slice at z = Nz/2 (2-D view)
 *   format = all    → all three formats
 *
 * Each snapshot is written to:
 *   <dir>/<prefix>_<snap_index>.vtk   (VTK)
 *   <dir>/<prefix>_<snap_index>.csv   (CSV)
 *   <dir>/<prefix>_<snap_index>_<field>.ppm   (PPM per field)
 *
 * The output directory is created automatically.
 */
#include "vtk_writer.hpp"
#include "csv_writer.hpp"
#include "ppm_writer.hpp"
#include "input_parser.hpp"

#include <string>
#include <stdexcept>
#include <cstdio>
#ifdef _WIN32
#  include <direct.h>
#  define gpk_mkdir(p) ::_mkdir(p)
#else
#  include <sys/stat.h>
#  define gpk_mkdir(p) ::mkdir(p, 0755)
#endif

namespace gpk {

inline void mkdir_p(const std::string& path) {
    std::string cur;
    for (char c : path + "/") {
        if (c == '/' || c == '\\') {
            if (!cur.empty())
                gpk_mkdir(cur.c_str());
        }
        cur += c;
    }
}

class OutputWriter {
public:
    explicit OutputWriter(const InputParser& cfg) {
        format_  = cfg.get_str("output", "format", "vtk");
        dir_     = cfg.get_str("output", "dir",    "output");
        prefix_  = cfg.get_str("output", "prefix", "snap");
        compact_ = cfg.get_bool("output", "compact", false);
        mkdir_p(dir_);
    }

    void write(const GPKGrid& grid, int snap, double sim_time) const {
        bool do_vtk = format_ == "vtk" || format_ == "all";
        bool do_csv = format_ == "csv" || format_ == "all";
        bool do_ppm = format_ == "ppm" || format_ == "all";

        char tag[64];
        std::snprintf(tag, sizeof(tag), "%04d", snap);
        std::string base = dir_ + "/" + prefix_ + "_" + tag;

        if (do_vtk) write_vtk(grid, base + ".vtk", sim_time, snap);
        if (do_csv) write_csv(grid, base + ".csv", sim_time, snap, compact_);
        if (do_ppm) write_ppm_slice(grid, base, sim_time);
    }

private:
    std::string format_, dir_, prefix_;
    bool compact_;

    static void write_ppm_slice(const GPKGrid& grid,
                                const std::string& base,
                                double /*sim_time*/) {
        const int Ny   = grid.Ny;
        const int Nx   = grid.Nx;
        const int z0   = grid.Nz / 2;   // mid-plane slice
        const int slab = z0 * Ny * Nx;

        // Extract 2-D slices at z = z0
        std::vector<int32_t> phi2d  (grid.phi     .begin()+slab, grid.phi     .begin()+slab+Ny*Nx);
        std::vector<float>   alpha2d(grid.alpha   .begin()+slab, grid.alpha   .begin()+slab+Ny*Nx);
        std::vector<int32_t> comp2d (grid.phi_comp.begin()+slab, grid.phi_comp.begin()+slab+Ny*Nx);
        std::vector<uint8_t> mask2d (grid.mask    .begin()+slab, grid.mask    .begin()+slab+Ny*Nx);

        // Grain image — smooth alpha-blended boundaries
        render_grains(phi2d, alpha2d, comp2d, mask2d, Ny, Nx, base + "_grains.ppm");

        // Composition slice
        std::vector<float> c2d(grid.c.begin()+slab, grid.c.begin()+slab+Ny*Nx);
        float cmin = *std::min_element(c2d.begin(), c2d.end());
        float cmax = *std::max_element(c2d.begin(), c2d.end());
        if (cmax > cmin)
            render_scalar(c2d, Ny, Nx, cmin, cmax, base + "_comp.ppm");
    }
};

} // namespace gpk
