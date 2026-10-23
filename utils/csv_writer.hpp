#pragma once
/*
 * csv_writer.hpp — Write GPK grid state as column-wise ASCII CSV.
 *
 * Format: space-separated columns, one row per cell.
 * Header lines start with '#'.
 * Columns: x[m]  y[m]  z[m]  grain_id  alpha  composition  temperature  orientation
 *
 * This format is convenient for post-processing with numpy/pandas/matplotlib.
 * Only interface + non-bulk cells are written by default (compact=true).
 * Pass compact=false to write every cell.
 */
#include "gpk_core.hpp"
#include <fstream>
#include <string>
#include <cstdio>

namespace gpk {

inline void write_csv(const GPKGrid& grid,
                      const std::string& path,
                      double sim_time,
                      int    snap_index,
                      bool   compact = false) {
    std::ofstream f(path);
    if (!f) throw std::runtime_error("CSV: cannot open " + path);

    const float dx = grid.dx;

    // ---- Header ----
    f << "# GPK Microstructure CSV\n";
    f << "# snap=" << snap_index << "  time=" << sim_time << "\n";
    f << "# Nz=" << grid.Nz << "  Ny=" << grid.Ny << "  Nx=" << grid.Nx << "\n";
    f << "# dx=" << dx << " [m]\n";
    if (compact)
        f << "# compact=true (only interface + changed cells written)\n";
    f << "# cols: x[m] y[m] z[m] grain_id alpha composition temperature orientation\n";

    char row[256];
    for (int iz = 0; iz < grid.Nz; ++iz)
    for (int iy = 0; iy < grid.Ny; ++iy)
    for (int ix = 0; ix < grid.Nx; ++ix) {
        int i = grid.idx(iz, iy, ix);
        if (compact && !grid.mask[i]) continue;

        std::snprintf(row, sizeof(row),
            "%.6e %.6e %.6e %d %.6f %.6f %.2f %.6f\n",
            ix * dx, iy * dx, iz * dx,
            grid.phi[i],
            grid.alpha[i],
            grid.c[i],
            grid.T[i],
            grid.g[i]);
        f << row;
    }
}

} // namespace gpk
