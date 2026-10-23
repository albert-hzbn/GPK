#pragma once
/*
 * vtk_writer.hpp — Write GPK grid state as VTK Legacy ASCII files.
 *
 * Format: STRUCTURED_POINTS (VTK 3.0 legacy, ASCII).
 * Readable directly by ParaView, VisIt, and any VTK-based tool.
 *
 * One .vtk file contains all fields (grain_id, alpha, composition,
 * temperature, orientation) as SCALARS datasets.
 */
#include "gpk_core.hpp"
#include <fstream>
#include <string>
#include <cstdio>

namespace gpk {

inline void write_vtk(const GPKGrid& grid,
                      const std::string& path,
                      double sim_time,
                      int    snap_index) {
    std::ofstream f(path);
    if (!f) throw std::runtime_error("VTK: cannot open " + path);

    const int Nz = grid.Nz, Ny = grid.Ny, Nx = grid.Nx;
    const float dx = grid.dx;
    const int   N  = grid.N();

    // ---- Header ----
    f << "# vtk DataFile Version 3.0\n";
    f << "GPK Microstructure snap=" << snap_index
      << " time=" << sim_time << "\n";
    f << "ASCII\n";
    f << "DATASET STRUCTURED_POINTS\n";
    f << "DIMENSIONS " << Nx << " " << Ny << " " << Nz << "\n";
    f << "ORIGIN 0 0 0\n";
    f << "SPACING " << dx << " " << dx << " " << dx << "\n";
    f << "POINT_DATA " << N << "\n";

    auto write_int_field = [&](const char* name,
                                const std::vector<int32_t>& data) {
        f << "SCALARS " << name << " int 1\nLOOKUP_TABLE default\n";
        for (int i = 0; i < N; ++i) {
            f << data[i];
            f << ((i % 9 == 8) ? '\n' : ' ');
        }
        f << '\n';
    };

    auto write_float_field = [&](const char* name,
                                  const std::vector<float>& data) {
        f << "SCALARS " << name << " float 1\nLOOKUP_TABLE default\n";
        char buf[32];
        for (int i = 0; i < N; ++i) {
            std::snprintf(buf, sizeof(buf), "%.6g", data[i]);
            f << buf;
            f << ((i % 6 == 5) ? '\n' : ' ');
        }
        f << '\n';
    };

    write_int_field  ("grain_id",    grid.phi);
    write_float_field("alpha",       grid.alpha);
    write_float_field("composition", grid.c);
    write_float_field("temperature", grid.T);
    write_float_field("orientation", grid.g);
}

} // namespace gpk
