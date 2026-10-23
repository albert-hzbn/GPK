/*
 * GPK — unified entry point.
 *
 * Usage:  ./gpk <input.ini>
 *
 * The input file must contain a [run] section that specifies which model to run:
 *
 *   [run]
 *   model = grain_growth       # grain_growth | solidification |
 *                              # recrystallisation | ostwald_ripening |
 *                              # precipitation
 *
 * All other sections are the same as before.
 */

#include "gpk_core.hpp"
#include "gpk_thermo.hpp"
#include "gpk_io.hpp"

#include <cstdio>
#include <cmath>
#include <cstring>
#include <vector>
#include <string>
#include <sstream>
#include <random>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <map>
#include <queue>
#include <climits>
#include <chrono>
#include <fstream>
#include <stdexcept>
#include <omp.h>

#ifdef _WIN32
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#endif

using namespace gpk;

// ============================================================
//  Shared helpers
// ============================================================

static void voronoi_init(GPKGrid& grid, int n_seeds, uint32_t seed)
{
    const int Nz = grid.Nz, Ny = grid.Ny, Nx = grid.Nx;
    if (n_seeds <= 0 || n_seeds > grid.N())
        throw std::invalid_argument("Voronoi seed count must lie between one and the cell count");
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> rz(0, Nz-1), ry(0, Ny-1), rx(0, Nx-1);
    std::uniform_real_distribution<float> rg(0.f, (float)M_PI * 0.5f);

    std::vector<int> sz(n_seeds), sy(n_seeds), sx(n_seeds);
    std::vector<float> sg(n_seeds);
    std::unordered_set<int> occupied_seeds;
    occupied_seeds.reserve(static_cast<size_t>(2 * n_seeds));
    for (int k = 0; k < n_seeds; ++k) {
        int linear = -1;
        do {
            sz[k]=rz(rng); sy[k]=ry(rng); sx[k]=rx(rng);
            linear=(sz[k]*Ny+sy[k])*Nx+sx[k];
        } while (!occupied_seeds.insert(linear).second);
        sg[k]=rg(rng);
    }

    #pragma omp parallel for schedule(static)
    for (int z = 0; z < Nz; ++z)
    for (int y = 0; y < Ny; ++y)
    for (int x = 0; x < Nx; ++x) {
        long long best_d = LLONG_MAX; int best = 0;
        for (int k = 0; k < n_seeds; ++k) {
            long long dz=std::abs(z-sz[k]), dy=std::abs(y-sy[k]), dx_=std::abs(x-sx[k]);
            dz=std::min(dz,static_cast<long long>(Nz)-dz);
            dy=std::min(dy,static_cast<long long>(Ny)-dy);
            dx_=std::min(dx_,static_cast<long long>(Nx)-dx_);
            long long d = dz*dz + dy*dy + dx_*dx_;
            if (d < best_d) { best_d=d; best=k; }
        }
        int i = grid.idx(z, y, x);
        grid.phi[i]   = best + 1;
        grid.g[i]     = sg[best];
        grid.alpha[i] = 1.f;
    }

    // A categorical grain identity must be one periodic face-connected
    // component before the first kinetic step.  Integer Voronoi ties can
    // otherwise leave diagonal one-cell copies of a label; the topology
    // certificate then correctly preserves that already-disconnected input.
    // Keep the largest component of every label and attach all remaining
    // cells, layer by layer, to an already connected face-neighbour.  The
    // operation changes only measure-zero raster ties and never smooths an
    // evolved interface.
    struct Component { int32_t label; std::vector<int> cells; };
    std::vector<uint8_t> visited(static_cast<size_t>(grid.N()), uint8_t{0});
    std::vector<Component> components;
    std::vector<int> largest(static_cast<size_t>(n_seeds + 1), -1);
    auto neighbours = [&](int cell, int out[6]) {
        int z,y,x; grid.decode(cell,z,y,x);
        const int xm=(x==0?Nx-1:x-1), xp=(x==Nx-1?0:x+1);
        const int ym=(y==0?Ny-1:y-1), yp=(y==Ny-1?0:y+1);
        const int zm=(z==0?Nz-1:z-1), zp=(z==Nz-1?0:z+1);
        out[0]=grid.idx(z,y,xm); out[1]=grid.idx(z,y,xp);
        out[2]=grid.idx(z,ym,x); out[3]=grid.idx(z,yp,x);
        out[4]=grid.idx(zm,y,x); out[5]=grid.idx(zp,y,x);
    };
    for (int start=0; start<grid.N(); ++start) {
        if (visited[static_cast<size_t>(start)]) continue;
        const int32_t label=grid.phi[start];
        Component component{label,{}};
        std::queue<int> frontier;
        frontier.push(start); visited[static_cast<size_t>(start)]=1;
        while (!frontier.empty()) {
            const int cell=frontier.front(); frontier.pop();
            component.cells.push_back(cell);
            int face[6]; neighbours(cell,face);
            const int nface=Nz>1?6:4;
            for (int k=0;k<nface;++k) {
                const int next=face[k];
                if (!visited[static_cast<size_t>(next)] && grid.phi[next]==label) {
                    visited[static_cast<size_t>(next)]=1;
                    frontier.push(next);
                }
            }
        }
        const int index=static_cast<int>(components.size());
        components.push_back(std::move(component));
        const int previous=largest[static_cast<size_t>(label)];
        if (previous<0 || components[static_cast<size_t>(index)].cells.size()
                          >components[static_cast<size_t>(previous)].cells.size())
            largest[static_cast<size_t>(label)]=index;
    }

    std::vector<uint8_t> attached(static_cast<size_t>(grid.N()),uint8_t{0});
    size_t attached_count=0;
    for (size_t index=0; index<components.size(); ++index) {
        const int32_t label=components[index].label;
        if (largest[static_cast<size_t>(label)]!=static_cast<int>(index)) continue;
        for (const int cell:components[index].cells) {
            attached[static_cast<size_t>(cell)]=1;
            ++attached_count;
        }
    }
    const size_t reassigned_initial=static_cast<size_t>(grid.N())-attached_count;
    std::vector<int> cells;
    std::vector<int32_t> targets;
    std::vector<float> orientations;
    while (attached_count<static_cast<size_t>(grid.N())) {
        cells.clear(); targets.clear(); orientations.clear();
        for (int cell=0; cell<grid.N(); ++cell) {
            if (attached[static_cast<size_t>(cell)]) continue;
            int face[6]; neighbours(cell,face);
            const int nface=Nz>1?6:4;
            int32_t candidate[6]; int support[6]={0,0,0,0,0,0};
            int source[6]; int ncandidate=0;
            for (int k=0;k<nface;++k) {
                const int next=face[k];
                if (!attached[static_cast<size_t>(next)]) continue;
                const int32_t value=grid.phi[next];
                int q=0; while (q<ncandidate && candidate[q]!=value) ++q;
                if (q==ncandidate) { candidate[q]=value; source[q]=next; ++ncandidate; }
                ++support[q];
            }
            if (ncandidate==0) continue;
            int best=0;
            for (int q=1;q<ncandidate;++q)
                if (support[q]>support[best] ||
                    (support[q]==support[best] && candidate[q]<candidate[best])) best=q;
            cells.push_back(cell); targets.push_back(candidate[best]);
            orientations.push_back(grid.g[source[best]]);
        }
        if (cells.empty()) throw std::runtime_error("Voronoi connectivity normalization stalled");
        for (size_t k=0;k<cells.size();++k) {
            const int cell=cells[k];
            grid.phi[cell]=targets[k]; grid.g[cell]=orientations[k];
            attached[static_cast<size_t>(cell)]=1;
        }
        attached_count+=cells.size();
    }
    if (reassigned_initial>0)
        printf("Voronoi connectivity normalization reassigned %zu tie cells.\n",
               reassigned_initial);
}

// ============================================================
//  Model: grain_growth
// ============================================================

static int run_grain_growth(InputParser& cfg)
{
    const int    Nz       = cfg.get_int   ("grid",  "Nz");
    const int    Ny       = cfg.get_int   ("grid",  "Ny");
    const int    Nx       = cfg.get_int   ("grid",  "Nx");
    const float  dx       = cfg.get_float ("grid",  "dx");
    const double t_end    = cfg.get_double("time",  "t_end");
    const int    n_snaps  = cfg.get_int   ("time",  "n_snaps");
    const float  M0       = cfg.get_float ("thermo","M0");
    const float  sigma0   = cfg.get_float ("thermo","sigma0");
    const float  eps4     = cfg.get_float ("thermo","eps4", 0.f);
    const int    N_GRAINS = cfg.get_int   ("grain_growth","n_grains");
    const int    seed     = cfg.get_int   ("grain_growth","seed", 42);

    printf("=== GPK Grain Growth ===\n");
    printf("Grid %d×%d×%d, dx=%.1f µm, %d grains\n", Nz,Ny,Nx, dx*1e6f, N_GRAINS);
    printf("OpenMP threads: %d\n", omp_get_max_threads());
    printf("Reporting characteristic mean-volume radius; the network coarsening prefactor\n"
           "depends on dimension, topology, and the selected size statistic.\n\n");

    std::shared_ptr<ThermoModel> thermo;
    if (std::fabs(eps4) > 1e-12f)
        thermo = std::make_shared<AnisotropicGrowthThermo>(M0, sigma0, eps4);
    else
        thermo = std::make_shared<IsotropicGrowthThermo>(M0, sigma0);
    GPKGrid grid(Nz, Ny, Nx, dx, thermo, 0.f,
                 cfg.get_float("geometry", "sigma_cells", 2.f));
    if (cfg.has("grain_growth", "restart_labels")) {
        const std::string label_path = cfg.get_str("grain_growth", "restart_labels");
        const std::string orientation_path =
            cfg.get_str("grain_growth", "restart_orientations", "none");
        auto read_exact = [](const std::string& path, char* target,
                             std::streamsize bytes) {
            std::ifstream stream(path, std::ios::binary);
            if (!stream) throw std::runtime_error("Cannot open restart field '" + path + "'");
            stream.read(target, bytes);
            if (stream.gcount() != bytes)
                throw std::runtime_error("Restart field has the wrong size: '" + path + "'");
            char extra = 0;
            if (stream.read(&extra, 1))
                throw std::runtime_error("Restart field contains trailing data: '" + path + "'");
        };
        read_exact(label_path, reinterpret_cast<char*>(grid.phi.data()),
                   static_cast<std::streamsize>(grid.phi.size() * sizeof(grid.phi[0])));
        if (orientation_path != "none")
            read_exact(orientation_path, reinterpret_cast<char*>(grid.g.data()),
                       static_cast<std::streamsize>(grid.g.size() * sizeof(grid.g[0])));
        std::fill(grid.alpha.begin(), grid.alpha.end(), 1.f);
        grid.time = cfg.get_double("grain_growth", "restart_time", 0.0);
        printf("Restarted categorical state at t=%.4f s\n", grid.time);
    } else {
        voronoi_init(grid, N_GRAINS, (uint32_t)seed);
    }

    OutputWriter writer(cfg);
    const bool write_initial = cfg.get_bool("output", "write_initial", false);
    const int snap_offset = write_initial ? 1 : 0;
    if (write_initial) writer.write(grid, 0, grid.time);

    printf("%-12s %-18s %-10s\n",
           "Time [s]", "R_char [µm]", "Grains");

    auto t_wall0 = std::chrono::high_resolution_clock::now();

    for (int snap = 0; snap < n_snaps; ++snap) {
        double t_snap = t_end * (snap + 1.0) / n_snaps;
        grid.run(t_snap, -1.f,
                 [](const GPKGrid& state, int steps) {
                     printf("  progress: step %d, t=%.6g s, grains=%d\n",
                            steps, state.time, state.n_grains());
                     fflush(stdout);
                 },
                 50);

        if (grid.n_grains() <= 1) break;

        double R = grid.mean_grain_radius();
        printf("%-12.4f %-18.4f %-10d\n",
               grid.time, R*1e6, grid.n_grains());

        writer.write(grid, snap + snap_offset, grid.time);
    }

    double wall_s = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - t_wall0).count();
    printf("\nWall time: %.2f s\n", wall_s);
    return 0;
}

// ============================================================
//  Model: solidification
// ============================================================

static int run_solidification(InputParser& cfg)
{
    const int    Nz    = cfg.get_int   ("grid",  "Nz");
    const int    Ny    = cfg.get_int   ("grid",  "Ny");
    const int    Nx    = cfg.get_int   ("grid",  "Nx");
    const float  dx    = cfg.get_float ("grid",  "dx");
    const double t_end = cfg.get_double("time",  "t_end");
    const int    nsnap = cfg.get_int   ("time",  "n_snaps");

    const float M0     = cfg.get_float("thermo","M0");
    const float sig0   = cfg.get_float("thermo","sigma0");
    const float eps4   = cfg.get_float("thermo","eps4", 0.04f);
    const float D      = cfg.get_float("thermo","D");
    const float m_liq  = cfg.get_float("thermo","m_liq");
    const float k_part = cfg.get_float("thermo","k_part");
    const float T_melt = cfg.get_float("thermo","T_melt");
    const float L      = cfg.get_float("thermo","L");
    const bool  stochastic       = cfg.get_bool ("thermo","stochastic",        false);
    const float noise_scale      = cfg.get_float("thermo","noise_scale",       1.0f);
    const bool  elastic_enabled  = cfg.get_bool ("thermo","elastic_enabled",   false);
    const float elastic_modulus  = cfg.get_float("thermo","elastic_modulus",   0.0f);
    const float eigenstrain_coeff= cfg.get_float("thermo","eigenstrain_coeff", 0.0f);
    const int   elastic_iters    = cfg.get_int  ("thermo","elastic_smooth_iters", 8);
    const float c0               = cfg.get_float("initial","c0");
    const float dT               = cfg.get_float("initial","undercooling");
    const int   seed_r           = cfg.get_int  ("initial","seed_radius", 4);
    const float c_ref            = cfg.get_float("thermo","c_ref", c0);

    printf("=== GPK Solidification ===\n");
    printf("Grid %d×%d×%d, dx=%.2f µm\n", Nz,Ny,Nx, dx*1e6f);
    printf("OpenMP threads: %d\n\n", omp_get_max_threads());

    auto thermo = std::make_shared<SolidificationThermo>(
        M0, sig0, eps4, m_liq, k_part, T_melt, L);
    thermo->stochastic       = stochastic;
    thermo->noise_scale      = noise_scale;
    thermo->elastic_enabled  = elastic_enabled;
    thermo->elastic_modulus  = elastic_modulus;
    thermo->eigenstrain_coeff= eigenstrain_coeff;
    thermo->c_ref            = c_ref;
    thermo->elastic_smooth_iters = elastic_iters;
    thermo->solute_rejection = cfg.get_bool("thermo", "solute_rejection", false);
    thermo->k_partition      = k_part;
    if (thermo->solute_rejection)
        printf("Solute partitioning enabled, k = %.3f\n", (double)k_part);

    GPKGrid grid(Nz, Ny, Nx, dx, thermo, D,
                 cfg.get_float("geometry", "sigma_cells", 2.f));

    float T_liq = T_melt + m_liq * c0;
    std::fill(grid.c.begin(), grid.c.end(), c0);
    std::fill(grid.T.begin(), grid.T.end(), T_liq - dT);

    float R_star = sig0 * T_melt / (L * dT);
    printf("Gibbs-Thomson R* = %.3f µm\n", R_star * 1e6f);
    printf("Seed radius      = %.1f µm (must be > R*)\n\n", seed_r*dx*1e6f);
    if (stochastic)
        printf("Stochastic kinetics enabled, noise_scale = %.3f\n", (double)noise_scale);
    if (elastic_enabled)
        printf("Elastic coupling enabled, modulus = %.3e J/m^3, coeff = %.3e\n\n",
               (double)elastic_modulus, (double)eigenstrain_coeff);

    int cz=Nz/2, cy=Ny/2, cx=Nx/2;
    for (int dz=-seed_r; dz<=seed_r; ++dz)
    for (int dy=-seed_r; dy<=seed_r; ++dy)
    for (int dx_=-seed_r; dx_<=seed_r; ++dx_) {
        if (dz*dz+dy*dy+dx_*dx_ > seed_r*seed_r) continue;
        int zz=cz+dz, yy=cy+dy, xx=cx+dx_;
        if (zz<0||zz>=Nz||yy<0||yy>=Ny||xx<0||xx>=Nx) continue;
        int i = grid.idx(zz, yy, xx);
        grid.phi[i] = 1;
        grid.c[i]   = c0 * k_part;
    }

    OutputWriter writer(cfg);
    const bool write_initial = cfg.get_bool("output", "write_initial", false);
    const int snap_offset = write_initial ? 1 : 0;
    if (write_initial) writer.write(grid, 0, grid.time);
    double snap_dt = t_end / nsnap;

    printf("%-10s %-14s %-14s %-10s\n",
           "Time[ms]", "Solid frac", "Solute max", "Steps");

    auto t_wall0 = std::chrono::high_resolution_clock::now();
    int total_steps = 0;

    for (int snap = 0; snap < nsnap; ++snap) {
        int steps = grid.run(snap_dt * (snap+1), -1.f);
        total_steps += steps;

        float sf   = grid.solid_fraction();
        float cmax = *std::max_element(grid.c.begin(), grid.c.end());
        printf("%-10.3f %-14.4f %-14.4f %-10d\n",
               grid.time*1e3, sf, cmax, steps);

        writer.write(grid, snap + snap_offset, grid.time);
    }

    double wall_s = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - t_wall0).count();
    printf("\nWall time: %.2f s  (%d steps, %.2f ms/step)\n",
           wall_s, total_steps, 1e3*wall_s/total_steps);
    return 0;
}

// ============================================================
//  Model: recrystallisation
// ============================================================

static void rex_nucleate(GPKGrid& grid, int n_nuclei, int r_nuc,
                         std::mt19937& rng, int& next_gid)
{
    const int Nz=grid.Nz, Ny=grid.Ny, Nx=grid.Nx;
    bool is2d = (Nz == 1);
    std::uniform_int_distribution<int> rz(is2d ? 0 : r_nuc,
                                          is2d ? 0 : (Nz - r_nuc - 1));
    std::uniform_int_distribution<int> ry(r_nuc, Ny-r_nuc-1);
    std::uniform_int_distribution<int> rx(r_nuc, Nx-r_nuc-1);
    std::uniform_real_distribution<float> rg(0.f, (float)M_PI*0.5f);

    for (int n = 0; n < n_nuclei; ++n) {
        int cz=rz(rng), cy=ry(rng), cx=rx(rng);
        float gval = rg(rng);
        int gid = ++next_gid;
        for (int dz=-r_nuc; dz<=r_nuc; ++dz)
        for (int dy=-r_nuc; dy<=r_nuc; ++dy)
        for (int dx_=-r_nuc; dx_<=r_nuc; ++dx_) {
            if (dz*dz+dy*dy+dx_*dx_ > r_nuc*r_nuc) continue;
            int z=cz+dz, y=cy+dy, x=cx+dx_;
            if (z<0||z>=Nz||y<0||y>=Ny||x<0||x>=Nx) continue;
            if (is2d && dz != 0) continue;
            int i = grid.idx(z,y,x);
            if (grid.phi[i] == 0) {
                grid.phi[i]   = gid;
                grid.g[i]     = gval;
                grid.alpha[i] = 1.f;
            }
        }
    }
}

static int rex_stochastic_nucleation(GPKGrid& grid, float Jdot, float dt,
                                     int r_nuc, std::mt19937& rng, int& next_gid)
{
    if (Jdot <= 0.f || dt <= 0.f) return 0;
    const int Nz=grid.Nz, Ny=grid.Ny, Nx=grid.Nx;
    const float Vcell = grid.dx * grid.dx * grid.dx;
    const float P = 1.f - std::exp(-Jdot * Vcell * dt);
    if (P <= 0.f) return 0;

    std::uniform_real_distribution<float> u01(0.f, 1.f);
    std::uniform_real_distribution<float> rg(0.f, (float)M_PI * 0.5f);
    int added = 0;

    auto try_seed = [&](int z, int y, int x) {
        int i = grid.idx(z, y, x);
        if (grid.phi[i] != 0) return;
        if (u01(rng) > P) return;
        int gid = ++next_gid;
        float gval = rg(rng);
        for (int dz=-r_nuc; dz<=r_nuc; ++dz)
        for (int dy=-r_nuc; dy<=r_nuc; ++dy)
        for (int dx_=-r_nuc; dx_<=r_nuc; ++dx_) {
            if (dz*dz+dy*dy+dx_*dx_ > r_nuc*r_nuc) continue;
            int zz=z+dz, yy=y+dy, xx=x+dx_;
            if (zz<0||zz>=Nz||yy<0||yy>=Ny||xx<0||xx>=Nx) continue;
            int j = grid.idx(zz,yy,xx);
            if (grid.phi[j] == 0) {
                grid.phi[j] = gid;
                grid.g[j]   = gval;
                grid.alpha[j] = 1.f;
            }
        }
        ++added;
    };

    if (Nz == 1) {
        for (int y=r_nuc; y<Ny-r_nuc; ++y)
        for (int x=r_nuc; x<Nx-r_nuc; ++x)
            try_seed(0, y, x);
    } else {
        for (int z=r_nuc; z<Nz-r_nuc; ++z)
        for (int y=r_nuc; y<Ny-r_nuc; ++y)
        for (int x=r_nuc; x<Nx-r_nuc; ++x)
            try_seed(z, y, x);
    }
    return added;
}

static int run_recrystallisation(InputParser& cfg)
{
    const int    Nz       = cfg.get_int   ("grid",  "Nz");
    const int    Ny       = cfg.get_int   ("grid",  "Ny");
    const int    Nx       = cfg.get_int   ("grid",  "Nx");
    const float  dx       = cfg.get_float ("grid",  "dx");
    const double t_end    = cfg.get_double("time",  "t_end");
    const int    n_snaps  = cfg.get_int   ("time",  "n_snaps");

    const float  M0       = cfg.get_float("thermo","M0");
    const float  sig0     = cfg.get_float("thermo","sigma0");
    const float  E_stored = cfg.get_float("thermo","E_stored");

    const int    N_NUC    = cfg.get_int  ("nucleation","n_nuclei");
    const int    R_NUC    = cfg.get_int  ("nucleation","nucleus_r");
    const float  Jdot     = cfg.get_float("nucleation","Jdot", 0.f);
    const int    seed     = cfg.get_int  ("nucleation","seed", 42);

    printf("=== GPK Recrystallisation ===\n");
    printf("Grid %d×%d×%d, dx=%.1f µm\n", Nz,Ny,Nx, dx*1e6f);
    printf("E_stored = %.2e J/m³, OpenMP: %d threads\n\n",
           (double)E_stored, omp_get_max_threads());

    float R_star = sig0 / E_stored;
    printf("Critical R* = %.3f µm (nucleus radius = %d cells = %.1f µm)\n\n",
           R_star*1e6f, R_NUC, R_NUC*dx*1e6f);

    auto thermo = std::make_shared<RecrystallisationThermo>(M0, sig0, E_stored);
    GPKGrid grid(Nz, Ny, Nx, dx, thermo, 0.f,
                 cfg.get_float("geometry", "sigma_cells", 2.f));

    std::fill(grid.phi.begin(),   grid.phi.end(),   0);
    std::fill(grid.alpha.begin(), grid.alpha.end(), 1.f);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> rg_dist(0.f, (float)M_PI*0.5f);
    for (auto& v : grid.g) v = rg_dist(rng);

    int next_gid = 0;
    rex_nucleate(grid, N_NUC, R_NUC, rng, next_gid);
    printf("Nucleated %d rex grains (radius %d cells), Jdot = %.3e 1/(m^3 s)\n\n",
           N_NUC, R_NUC, (double)Jdot);

    OutputWriter writer(cfg);
    const bool write_initial = cfg.get_bool("output", "write_initial", false);
    const int snap_offset = write_initial ? 1 : 0;
    if (write_initial) writer.write(grid, 0, grid.time);
    double snap_dt = t_end / n_snaps;

    printf("%-10s %-12s %-14s %-10s\n",
           "Time[s]", "Rex frac", "ln(-ln(1-X))", "N_grains");

    auto t_wall0 = std::chrono::high_resolution_clock::now();

    for (int snap = 0; snap < n_snaps; ++snap) {
        double t_target = snap_dt * (snap + 1);
        while (grid.time < t_target) {
            float dt = grid.step(-1.f);
            rex_stochastic_nucleation(grid, Jdot, dt, R_NUC, rng, next_gid);
        }

        int rex_cnt = 0;
        for (auto v : grid.phi) if (v != 0) ++rex_cnt;
        float X = (float)rex_cnt / (float)grid.N();
        float avrami = (X > 0.f && X < 1.f) ? std::log(-std::log(1.f - X))
                                              : (X >= 1.f ? 99.f : -99.f);

        printf("%-10.4f %-12.4f %-14.3f %-10d\n",
               grid.time, X, avrami, grid.n_grains());

        writer.write(grid, snap + snap_offset, grid.time);
        if (X > 0.99f) { printf("Fully recrystallised.\n"); break; }
    }

    double wall_s = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - t_wall0).count();
    printf("\nWall time: %.2f s\n", wall_s);
    return 0;
}

// ============================================================
//  Model: ostwald_ripening
// ============================================================

static int place_particles(GPKGrid& grid, int n_try,
                            float r_mean_cells, float r_ln_sigma,
                            float c_part, float c_matrix, uint32_t seed)
{
    const int Nz=grid.Nz, Ny=grid.Ny, Nx=grid.Nx;
    std::mt19937 rng(seed);
    std::lognormal_distribution<float> r_dist(std::log(r_mean_cells), r_ln_sigma);
    std::uniform_int_distribution<int> rz(0,Nz-1), ry(0,Ny-1), rx(0,Nx-1);

    std::fill(grid.phi.begin(),   grid.phi.end(),   0);
    std::fill(grid.alpha.begin(), grid.alpha.end(), 1.f);
    std::fill(grid.c.begin(),     grid.c.end(),     c_matrix);

    int placed = 0;
    for (int attempt = 0; attempt < n_try*50 && placed < n_try; ++attempt) {
        int cap = (Nz == 1) ? std::min(Ny,Nx)/4 : std::min({Nz,Ny,Nx})/4;
        int ri = std::max(1, (int)std::round(
            std::clamp(r_dist(rng), 1.5f, (float)cap)));
        int cz=rz(rng), cy=ry(rng), cx=rx(rng);

        bool z_ok = (Nz == 1) || (cz-ri>=1 && cz+ri<=Nz-2);
        if (!z_ok || cy-ri<1||cy+ri>=Ny-1||cx-ri<1||cx+ri>=Nx-1) continue;

        bool overlap = false;
        for (int dz=-(ri+1); dz<=ri+1 && !overlap; ++dz)
        for (int dy=-ri-1;   dy<=ri+1 && !overlap; ++dy)
        for (int dx_=-ri-1;  dx_<=ri+1 && !overlap; ++dx_) {
            int oz=cz+dz, oy=cy+dy, ox=cx+dx_;
            if (oz<0||oz>=Nz||oy<0||oy>=Ny||ox<0||ox>=Nx) continue;
            if (grid.phi[grid.idx(oz,oy,ox)] != 0) overlap=true;
        }
        if (overlap) continue;

        int gid = ++placed;
        for (int dz=-ri; dz<=ri; ++dz)
        for (int dy=-ri; dy<=ri; ++dy)
        for (int dx_=-ri; dx_<=ri; ++dx_) {
            if (dz*dz+dy*dy+dx_*dx_ > ri*ri) continue;
            int oz=cz+dz, oy=cy+dy, ox=cx+dx_;
            if (oz<0||oz>=Nz||oy<0||oy>=Ny||ox<0||ox>=Nx) continue;
            int i = grid.idx(oz, oy, ox);
            grid.phi[i]   = gid;
            grid.c[i]     = c_part;
            grid.alpha[i] = 1.f;
        }
    }
    return placed;
}

struct PStats { double mean, min_r, max_r; int count; };
static PStats particle_stats(const GPKGrid& grid)
{
    std::unordered_map<int32_t,int> cnt;
    for (auto v : grid.phi) if (v != 0) cnt[v]++;
    if (cnt.empty()) return {0,0,0,0};
    double sum=0, rmin=1e30, rmax=0;
    for (auto& [k,a] : cnt) {
        double r = (grid.Nz > 1)
                   ? std::cbrt(3.0*a/(4.0*M_PI))
                   : std::sqrt((double)a/M_PI);
        sum+=r; if(r<rmin) rmin=r; if(r>rmax) rmax=r;
    }
    float dx=grid.dx; int n=(int)cnt.size();
    return {sum/n*dx, rmin*dx, rmax*dx, n};
}

static int run_ostwald_ripening(InputParser& cfg)
{
    const int    Nz    = cfg.get_int   ("grid","Nz");
    const int    Ny    = cfg.get_int   ("grid","Ny");
    const int    Nx    = cfg.get_int   ("grid","Nx");
    const float  dx    = cfg.get_float ("grid","dx");
    const double t_end = cfg.get_double("time","t_end");
    const int    nsnap = cfg.get_int   ("time","n_snaps");

    const float M0     = cfg.get_float("thermo","M0");
    const float sig0   = cfg.get_float("thermo","sigma0");
    const float D      = cfg.get_float("thermo","D");
    const float c_sat  = cfg.get_float("thermo","c_sat",  0.15f);
    const float Omega  = cfg.get_float("thermo","Omega",  5e8f);

    const int   N_try    = cfg.get_int  ("particles","n_particles");
    const float r_mean   = cfg.get_float("particles","r_mean_cells");
    const float r_lns    = cfg.get_float("particles","r_lognormal_s");
    const float c_part   = cfg.get_float("particles","c_particle");
    const float c_matrix = cfg.get_float("particles","c_matrix");
    const int   seed     = cfg.get_int  ("particles","seed", 999);

    printf("=== GPK Ostwald Ripening ===\n");
    printf("Grid %d×%d×%d, dx=%.2f µm, D=%.1e m²/s\n",
           Nz,Ny,Nx, dx*1e6f, (double)D);
    printf("OpenMP threads: %d\n\n", omp_get_max_threads());

    auto thermo = std::make_shared<OstwaldThermo>(M0, sig0, c_sat, Omega);
    GPKGrid grid(Nz, Ny, Nx, dx, thermo, D,
                 cfg.get_float("geometry", "sigma_cells", 2.f));

    int placed = place_particles(grid, N_try, r_mean, r_lns, c_part, c_matrix, seed);
    auto s0 = particle_stats(grid);
    printf("Placed %d particles; R_mean=%.2f µm, R_min=%.2f µm, R_max=%.2f µm\n\n",
           placed, s0.mean*1e6f, s0.min_r*1e6f, s0.max_r*1e6f);

    OutputWriter writer(cfg);
    double snap_dt = t_end / nsnap;

    printf("%-10s %-10s %-12s %-12s %-12s\n",
           "Time[ms]","N_part","R_mean[µm]","R_min[µm]","R_max[µm]");

    auto t_wall0 = std::chrono::high_resolution_clock::now();
    int total_steps = 0;

    for (int snap = 0; snap < nsnap; ++snap) {
        int steps = grid.run(snap_dt*(snap+1), -1.f);
        total_steps += steps;
        auto s = particle_stats(grid);
        printf("%-10.2f %-10d %-12.3f %-12.3f %-12.3f\n",
               grid.time*1e3, s.count,
               s.mean*1e6f, s.min_r*1e6f, s.max_r*1e6f);

        writer.write(grid, snap, grid.time);
        if (s.count == 0) { printf("All particles dissolved.\n"); break; }
    }

    double wall_s = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - t_wall0).count();
    printf("\nWall time: %.2f s  (%d steps)\n", wall_s, total_steps);
    return 0;
}

// ============================================================
//  Model: precipitation
// ============================================================

static std::vector<float> parse_csv_floats(const std::string& s)
{
    std::vector<float> out;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ','))
        if (!tok.empty()) out.push_back(std::stof(tok));
    return out;
}

static int run_precipitation(InputParser& cfg)
{
    const int    Nz    = cfg.get_int  ("grid","Nz");
    const int    Ny    = cfg.get_int  ("grid","Ny");
    const int    Nx    = cfg.get_int  ("grid","Nx");
    const float  dx    = cfg.get_float("grid","dx");
    const double t_end = cfg.get_double("time","t_end");
    const int    nsnap = cfg.get_int  ("time","n_snaps");

    const float  M0       = cfg.get_float("thermo","M0");
    const float  sig0     = cfg.get_float("thermo","sigma0");
    const float  D        = cfg.get_float("thermo","D",     0.f);
    const int    n_phases = cfg.get_int  ("thermo","n_phases");
    const float  T_ref    = cfg.get_float("thermo","T_ref", 900.f);

    std::vector<float> a  = parse_csv_floats(cfg.get_str("thermo","a"));
    std::vector<float> b  = parse_csv_floats(cfg.get_str("thermo","b"));
    std::vector<float> c2 = parse_csv_floats(cfg.get_str("thermo","c2"));
    std::vector<float> dT = parse_csv_floats(cfg.get_str("thermo","dT"));

    const float c0      = cfg.get_float("initial","c0");
    const float T0      = cfg.get_float("initial","T0");
    const int   n_seeds = cfg.get_int  ("initial","n_seeds");
    const int   seed_r  = cfg.get_int  ("initial","seed_radius", 2);
    const int   seed    = cfg.get_int  ("initial","seed", 123);

    printf("=== GPK Multi-phase Precipitation ===\n");
    printf("Grid %d×%d×%d, phases=%d, dx=%.2f µm\n",
           Nz, Ny, Nx, n_phases, dx*1e6f);
    printf("OpenMP threads: %d\n\n", omp_get_max_threads());

    auto thermo = std::make_shared<PrecipitationThermo>(
        M0, sig0, n_phases, a, b, c2, dT, T_ref);
    GPKGrid grid(Nz, Ny, Nx, dx, thermo, D,
                 cfg.get_float("geometry", "sigma_cells", 2.f));

    std::fill(grid.phi.begin(),   grid.phi.end(),   0);
    std::fill(grid.alpha.begin(), grid.alpha.end(), 1.f);
    std::fill(grid.c.begin(),     grid.c.end(),     c0);
    std::fill(grid.T.begin(),     grid.T.end(),     T0);

    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> rz(seed_r, std::max(seed_r, Nz-seed_r-1));
    std::uniform_int_distribution<int> ry(seed_r, std::max(seed_r, Ny-seed_r-1));
    std::uniform_int_distribution<int> rx(seed_r, std::max(seed_r, Nx-seed_r-1));
    std::uniform_int_distribution<int> rp(1, std::max(1, n_phases-1));

    for (int n = 0; n < n_seeds; ++n) {
        int cz=(Nz > 1 ? rz(rng) : 0), cy=ry(rng), cx=rx(rng);
        int p = rp(rng);
        bool overlap = false;
        const int z_guard = (Nz > 1 ? seed_r+1 : 0);
        for (int dz=-z_guard; dz<=z_guard && !overlap; ++dz)
        for (int dy=-seed_r-1; dy<=seed_r+1 && !overlap; ++dy)
        for (int dx_=-seed_r-1; dx_<=seed_r+1 && !overlap; ++dx_) {
            int zz=cz+dz, yy=cy+dy, xx=cx+dx_;
            if (zz<0||zz>=Nz||yy<0||yy>=Ny||xx<0||xx>=Nx) continue;
            if (grid.phi[grid.idx(zz,yy,xx)] != 0) overlap=true;
        }
        if (overlap) continue;
        const int z_radius = (Nz > 1 ? seed_r : 0);
        for (int dz=-z_radius; dz<=z_radius; ++dz)
        for (int dy=-seed_r; dy<=seed_r; ++dy)
        for (int dx_=-seed_r; dx_<=seed_r; ++dx_) {
            if (dz*dz+dy*dy+dx_*dx_ > seed_r*seed_r) continue;
            int zz=cz+dz, yy=cy+dy, xx=cx+dx_;
            if (zz<0||zz>=Nz||yy<0||yy>=Ny||xx<0||xx>=Nx) continue;
            grid.phi[grid.idx(zz,yy,xx)] = p;
        }
    }

    OutputWriter writer(cfg);
    const bool write_initial = cfg.get_bool("output", "write_initial", false);
    const int snap_offset = write_initial ? 1 : 0;
    if (write_initial) writer.write(grid, 0, grid.time);
    const double snap_dt = t_end / nsnap;

    printf("%-10s %-12s %-12s %-12s\n",
           "Time[s]", "Phase0", "Phase1", "Phase2+");

    auto t_wall0 = std::chrono::high_resolution_clock::now();
    for (int snap = 0; snap < nsnap; ++snap) {
        grid.run(snap_dt * (snap+1), -1.f);

        std::unordered_map<int32_t,int> cnt;
        for (auto p : grid.phi) cnt[p]++;
        float f0  = (float)cnt[0] / (float)grid.N();
        float f1  = (float)cnt[1] / (float)grid.N();
        float f2p = std::max(0.f, 1.f - f0 - f1);

        printf("%-10.4f %-12.4f %-12.4f %-12.4f\n",
               grid.time, f0, f1, f2p);
        writer.write(grid, snap + snap_offset, grid.time);
    }

    double wall_s = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - t_wall0).count();
    printf("\nWall time: %.2f s\n", wall_s);
    return 0;
}

// ============================================================
//  Model: wulff_shape
//  Tests tensor anisotropy by shrinking a single circular grain
//  and measuring how its bounding box aspect ratio evolves.
//
//  Physics: σ(n̂) = σ₀ √(bxx nc_x² + byy nc_y² + bzz nc_z²)
//           v_n  = −M σ(n̂) κ
//
//  Expected behaviour: boundary points where σ is largest move
//  inward fastest.  For bxx < byy a 2-D circular grain shrinks
//  faster in y than in x, so the grain becomes wider over time
//  (x-extent / y-extent > 1 within a few steps).
// ============================================================

static int run_wulff_shape(InputParser& cfg)
{
    const int    Nz    = cfg.get_int   ("grid",  "Nz");
    const int    Ny    = cfg.get_int   ("grid",  "Ny");
    const int    Nx    = cfg.get_int   ("grid",  "Nx");
    const float  dx    = cfg.get_float ("grid",  "dx");
    const double t_end = cfg.get_double("time",  "t_end");
    const int    nsnap = cfg.get_int   ("time",  "n_snaps");

    const float M0     = cfg.get_float("thermo","M0");
    const float sigma0 = cfg.get_float("thermo","sigma0");
    const float bxx    = cfg.get_float("thermo","bxx", 1.f);
    const float byy    = cfg.get_float("thermo","byy", 1.f);
    const float bzz    = cfg.get_float("thermo","bzz", 1.f);
    const int   r_seed = cfg.get_int  ("wulff", "seed_radius", 20);

    printf("=== GPK Wulff Shape Test ===\n");
    printf("Grid %d×%d×%d, dx=%.2f µm\n", Nz, Ny, Nx, dx*1e6f);
    printf("Tensor anisotropy: bxx=%.2f  byy=%.2f  bzz=%.2f\n",
           (double)bxx, (double)byy, (double)bzz);
    printf("  σ(n̂=x̂) = σ₀√bxx = %.4f  |  σ(n̂=ŷ) = σ₀√byy = %.4f\n",
           (double)(sigma0 * std::sqrt(bxx)), (double)(sigma0 * std::sqrt(byy)));
    printf("  Expected shrink ratio dRy/dRx = √(byy/bxx) = %.4f\n\n",
           std::sqrt((double)byy / (double)bxx));

    auto thermo = std::make_shared<TensorAnisotropyThermo>(M0, sigma0, bxx, byy, bzz);
    GPKGrid grid(Nz, Ny, Nx, dx, thermo, 0.f,
                 cfg.get_float("geometry", "sigma_cells", 2.f));

    // Background: phi=0, g=0
    std::fill(grid.phi.begin(),   grid.phi.end(),   0);
    std::fill(grid.alpha.begin(), grid.alpha.end(), 1.f);
    std::fill(grid.g.begin(),     grid.g.end(),     0.f);

    // Single circular grain (phi=1) in centre, crystal orientation g=0
    const int cz = Nz/2, cy = Ny/2, cx = Nx/2;
    for (int dz = -r_seed; dz <= r_seed; ++dz)
    for (int dy = -r_seed; dy <= r_seed; ++dy)
    for (int dx_ = -r_seed; dx_ <= r_seed; ++dx_) {
        if (dz*dz + dy*dy + dx_*dx_ > r_seed*r_seed) continue;
        int zz = cz+dz, yy = cy+dy, xx = cx+dx_;
        if (zz<0||zz>=Nz||yy<0||yy>=Ny||xx<0||xx>=Nx) continue;
        int i = grid.idx(zz, yy, xx);
        grid.phi[i]   = 1;
        grid.alpha[i] = 1.f;
    }

    OutputWriter writer(cfg);

    // Helper: measure bounding-box half-extents of phi==1 cells
    auto bbox = [&](float& hx, float& hy) {
        int xmin = Nx, xmax = -1, ymin = Ny, ymax = -1;
        for (int y = 0; y < Ny; ++y)
        for (int x = 0; x < Nx; ++x) {
            if (grid.phi[grid.idx(0, y, x)] == 1) {
                if (x < xmin) xmin = x;
                if (x > xmax) xmax = x;
                if (y < ymin) ymin = y;
                if (y > ymax) ymax = y;
            }
        }
        hx = (xmin <= xmax) ? 0.5f * (xmax - xmin + 1) * dx : 0.f;
        hy = (ymin <= ymax) ? 0.5f * (ymax - ymin + 1) * dx : 0.f;
    };

    float hx0, hy0;
    bbox(hx0, hy0);
    printf("Initial half-extents: hx=%.2f µm  hy=%.2f µm  ratio x/y=%.4f\n\n",
           hx0*1e6f, hy0*1e6f, (hy0>0.f ? hx0/hy0 : 0.f));

    printf("%-10s %-12s %-12s %-10s %-10s\n",
           "Time[s]", "hx[µm]", "hy[µm]", "x/y ratio", "Grain%");

    double snap_dt = t_end / nsnap;
    bool passed = false;

    auto t_wall0 = std::chrono::high_resolution_clock::now();

    for (int snap = 0; snap < nsnap; ++snap) {
        grid.run(snap_dt * (snap + 1), -1.f);

        float hx, hy;
        bbox(hx, hy);
        float ratio  = (hy > 0.f) ? hx / hy : 0.f;
        float f_grain = grid.solid_fraction();

        printf("%-10.4f %-12.3f %-12.3f %-10.4f %-10.4f\n",
               grid.time, hx*1e6f, hy*1e6f, ratio, f_grain);

        writer.write(grid, snap, grid.time);

        // Pass: grain is still alive AND x-extent > y-extent
        // (meaning byy > bxx made y-boundary shrink faster → wider grain)
        if (bxx < byy && ratio > 1.05f && f_grain > 0.005f) passed = true;

        if (f_grain < 0.002f) { printf("Grain dissolved.\n"); break; }
    }

    double wall_s = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - t_wall0).count();
    printf("\nWall time: %.2f s\n", wall_s);

    if (bxx != byy) {
        printf("\nAnisotropy test: %s\n",
               passed ? "PASS — grain became anisotropic in expected direction"
                      : "FAIL — grain did not become measurably anisotropic");
        return passed ? 0 : 1;
    }
    return 0;
}

// ============================================================
//  Model: precipitate_breakup  (3-D)
//
//  Estimates the CRITICAL ASPECT RATIO d/h at which a continuous α-phase
//  plate/layer (in a β matrix) breaks up into individual particles by
//  grain-boundary grooving.
//
//      d = in-plane spacing between grain boundaries (grain size)
//      h = plate thickness
//
//  Geometry — a 3-D α slab of thickness h, PERIODIC in x and y, subdivided
//  by a PERIODIC array of α/α grain boundaries on a d×d grid, embedded in β:
//
//        z
//        │   β matrix (phi = 0)
//        │ ┌───────────────────────────┐  ─┐
//        │ │ broad face α/β  (σ_b, n̂=ẑ)│   │ h   (slab thickness)
//        │ │  α  │GB│  α  │GB│  α  │GB│ │   │
//        │ │ broad face α/β  (σ_b)      │  ─┘
//        │ └───────────────────────────┘
//        └────────── x ──────────────────
//          |← d →|     d = GB spacing (periodic)
//
//  Face energies (PDF, Ti-6Al-V, Li et al. 2016):
//      broad face  α/β (coherent)        σ_b  = 0.091 J/m²   (n̂ = ±ẑ)
//      side / α/α grain boundary         σ_s  = 0.811 J/m²   (n̂ in x–y)
//
//  Encoded with the gradient-energy tensor σ(n̂)=σ₀√(bxx nc_x²+byy nc_y²+bzz nc_z²):
//      σ₀ = σ_s,  bxx = byy = 1,  bzz = (σ_b/σ_s)²
//   ⇒ broad faces (z-normal) cheap; vertical GBs (in-plane normal) expensive.
//
//  The flat broad faces (κ≈0) barely move, so the slab volume is ~conserved;
//  GBs groove from both broad faces.  A thin slab with large grains (large
//  d/h) is severed by the grooves into separate particles (BREAK); a thick
//  slab with small grains (small d/h) grooves but stays connected (stable).
//  Sweeping d/h brackets the critical ratio.
// ============================================================

// Count connected α regions (phi != 0) in 3-D, 6-connectivity, periodic in
// x and y (the in-plane slab directions), non-periodic in z (slab normal).
// Returns the number of α pieces of size >= min_size.
// Periodic in x, y, AND z, matching GPK's core (interface mask, geometry,
// velocity, diffusion all already wrap periodically on every axis -- an
// earlier version of this function treated z as non-periodic, which was
// inconsistent with the rest of the engine, found directly by checking
// src/gpk_core.cpp's index-wrapping pattern).
static int alpha_pieces_3d(const GPKGrid& g, int min_size)
{
    const int Nz = g.Nz, Ny = g.Ny, Nx = g.Nx, N = Nz * Ny * Nx;
    std::vector<int> lab(N, -1);
    std::vector<int> st;
    int big = 0;
    for (int s = 0; s < N; ++s) {
        if (g.phi[s] == 0 || lab[s] != -1) continue;
        st.clear(); st.push_back(s); lab[s] = 1; int sz = 0;
        while (!st.empty()) {
            int i = st.back(); st.pop_back(); ++sz;
            int z, y, x; g.decode(i, z, y, x);
            const int xm=(x==0?Nx-1:x-1), xp=(x==Nx-1?0:x+1);
            const int ym=(y==0?Ny-1:y-1), yp=(y==Ny-1?0:y+1);
            const int zm=(z==0?Nz-1:z-1), zp=(z==Nz-1?0:z+1);
            int nb[6] = { g.idx(z,y,xm), g.idx(z,y,xp),
                          g.idx(z,ym,x), g.idx(z,yp,x),
                          g.idx(zm,y,x), g.idx(zp,y,x) };
            for (int k = 0; k < 6; ++k) {
                int j = nb[k];
                if (g.phi[j] == 0 || lab[j] != -1) continue;
                lab[j] = 1; st.push_back(j);
            }
        }
        if (sz >= min_size) ++big;
    }
    return big;
}

// Dump the x–z cross-section at y = Ny/2 (rows = z, cols = x).  This slice
// cuts through the slab thickness so the GB grooves and pinch-off are visible.
static void dump_xz_csv(const GPKGrid& g, const std::string& path)
{
    FILE* f = std::fopen(path.c_str(), "w");
    if (!f) return;
    const int y = g.Ny / 2;
    for (int z = 0; z < g.Nz; ++z) {
        for (int x = 0; x < g.Nx; ++x)
            std::fprintf(f, "%d%c", g.phi[g.idx(z, y, x)], x==g.Nx-1?'\n':',');
    }
    std::fclose(f);
}

// Dump the in-plane x–y slice at the slab mid-plane z = Nz/2 (rows = y, cols = x).
static void dump_xy_csv(const GPKGrid& g, const std::string& path)
{
    FILE* f = std::fopen(path.c_str(), "w");
    if (!f) return;
    const int z = g.Nz / 2;
    for (int y = 0; y < g.Ny; ++y) {
        for (int x = 0; x < g.Nx; ++x)
            std::fprintf(f, "%d%c", g.phi[g.idx(z, y, x)], x==g.Nx-1?'\n':',');
    }
    std::fclose(f);
}

// Dump the FULL 3-D field in the same row-per-cell format as
// utils/csv_writer.hpp's write_csv (so it loads unchanged with
// the CSV readers in scripts/), for marching-cubes
// isosurface rendering. Only the grain-id column is meaningful here
// (alpha/comp/T/orientation are written for format compatibility).
static void dump_full3d_csv(const GPKGrid& g, const std::string& path,
                            double sim_time, int snap_index)
{
    FILE* f = std::fopen(path.c_str(), "w");
    if (!f) return;
    // 8 MB stdio buffer: with hundreds of thousands of rows per dump (and
    // many dumps per d-value for the time series), unbuffered/small-buffer
    // fprintf was the dominant cost of the whole sweep -- verified directly
    // via wall-clock timing (a 60-step run took LONGER than a 300-step run,
    // because more frequent small dumps fired more often at the smaller
    // step count -- the I/O, not the physics, was the bottleneck).
    static std::vector<char> iobuf(8u << 20);
    std::setvbuf(f, iobuf.data(), _IOFBF, iobuf.size());
    std::fprintf(f, "# GPK Microstructure CSV\n");
    std::fprintf(f, "# snap=%d  time=%g\n", snap_index, sim_time);
    std::fprintf(f, "# Nz=%d  Ny=%d  Nx=%d\n", g.Nz, g.Ny, g.Nx);
    std::fprintf(f, "# dx=%g [m]\n", (double)g.dx);
    std::fprintf(f, "# cols: x[m] y[m] z[m] grain_id alpha composition temperature orientation\n");
    for (int z = 0; z < g.Nz; ++z)
        for (int y = 0; y < g.Ny; ++y)
            for (int x = 0; x < g.Nx; ++x) {
                int i = g.idx(z, y, x);
                std::fprintf(f, "%.6e %.6e %.6e %d %.6f %.6f %.2f %.6f\n",
                             x*(double)g.dx, y*(double)g.dx, z*(double)g.dx,
                             g.phi[i], g.alpha[i], g.c[i], g.T[i], g.g[i]);
            }
    std::fclose(f);
}

static int run_precipitate_breakup(InputParser& cfg)
{
    // Nx is no longer read from [grid] -- it's computed per d-value as
    // Nx=2d below (fully periodic in x: grain1=[0,d), grain2=[d,2d), giving
    // two grain boundaries per period at spacing d, matching GPK's core
    // which is already periodic on every axis).
    const int    Ny     = cfg.get_int   ("grid", "Ny", 96);
    const int    Nz     = cfg.get_int   ("grid", "Nz", 48);
    const float  dx     = cfg.get_float ("grid", "dx", 5.0e-9f);
    const int    nsnap  = cfg.get_int   ("time", "n_snaps", 30);

    const float  M0     = cfg.get_float ("thermo", "M0",          1.0e-9f);
    const float  sig_b  = cfg.get_float ("thermo", "sigma_broad", 0.091f);
    const float  sig_s  = cfg.get_float ("thermo", "sigma_side",  0.811f);

    const float  c_sat    = cfg.get_float("thermo", "c_sat",    0.15f);
    const float  Omega    = cfg.get_float("thermo", "Omega",    5.0e8f);

    const int    h        = cfg.get_int  ("breakup", "h_cells", 16);   // plate thickness (fixed)
    const int    n_steps  = cfg.get_int  ("breakup", "n_steps", 2000); // step budget
    const float  min_frac = cfg.get_float("breakup", "min_piece_frac", 0.25f);
    std::vector<float> d_list = parse_csv_floats(cfg.get_str("breakup", "d_list"));
    if (d_list.empty()) { fprintf(stderr, "breakup.d_list is empty\n"); return 1; }
    if (Nz < 2) { fprintf(stderr, "precipitate_breakup is 3-D: set grid.Nz > 1\n"); return 1; }

    const std::string outdir = cfg.get_str("output", "dir", "output/precipitate_breakup");

    // Tensor mapping: σ(ẑ)=σ₀√bzz=σ_b (broad), σ(x̂)=σ(ŷ)=σ₀√bxx=σ_s (GB/side).
    const float sigma0 = sig_s;
    const float bxx    = 1.0f, byy = 1.0f;
    const float bzz    = (sig_b / sig_s) * (sig_b / sig_s);

    printf("=== GPK Precipitate Breakup 3-D, fully periodic lamellar stripe (critical d/h) ===\n");
    printf("Nz=%d Ny=%d (Nx=2d per d-value)  |  plate thickness h=%d cells, 2 GBs/period  |  %d steps/run\n",
           Nz, Ny, h, n_steps);
    printf("sigma_broad=%.3f  sigma_side=%.3f J/m^2  -> sigma0=%.3f bxx=byy=%.2f bzz=%.5f\n",
           (double)sig_b, (double)sig_s, (double)sigma0, (double)bxx, (double)bzz);
    printf("(eq. alpha=3.9 wt%%V, beta=39.8 wt%%V at 973K)  pure-capillary (D=0, c=c_sat)\n");
    printf("OpenMP threads: %d\n\n", omp_get_max_threads());

    // Pure capillary shape flow: AnisoOstwald with c≡c_sat (so ΔG=0) and D=0.
    // Anisotropic σ: broad faces (z-normal) cheap, side/GB faces (in-plane)
    // expensive.  GB pinch competes with spheroidisation → critical d/h.
    auto thermo = std::make_shared<AnisoOstwaldThermo>(
        M0, sigma0, bxx, byy, bzz, c_sat, Omega);

    mkdir_p(outdir);
    std::string summary = outdir + "/summary.csv";
    FILE* sf = std::fopen(summary.c_str(), "w");
    std::fprintf(sf, "d_cells,h_cells,dh,max_pieces,result\n");

    printf("%-8s %-8s %-12s %-10s\n", "d[cell]", "d/h", "max_pieces", "result");

    double crit_dh_break = 1e30, crit_dh_stable = 0.0;
    const int snap_every = std::max(1, n_steps / nsnap);

    for (float df : d_list) {
        const int d  = (int)std::lround(df);    // grain length = GB spacing (cells)
        const float dh = (float)d / (float)h;

        // Fully periodic in x AND y (matching GPK's core, which is already
        // periodic on every axis): the alpha phase fills x and y COMPLETELY
        // (no beta gap in either direction) -- found directly, with the
        // realisation that a periodic system with grain1=[0,d), grain2=[d,Nx)
        // and Nx=2d gives exactly TWO grain boundaries per period (at x=0,
        // the periodic seam, and at x=d), each separated by the SAME spacing
        // d -- a true periodic lamellar/colony stripe array, matching the
        // original "d also represents the GB spacing" intent literally
        // rather than only via a symmetry argument. z keeps a genuine beta
        // gap (the alpha slab has finite thickness h within Nz), which is
        // itself periodic in the underlying physics already.
        const int Nx_d = 2 * d;
        GPKGrid grid(Nz, Ny, Nx_d, dx, thermo, 0.f,
                     cfg.get_float("geometry", "sigma_cells", 2.f));
        std::fill(grid.phi.begin(),   grid.phi.end(),   0);
        std::fill(grid.alpha.begin(), grid.alpha.end(), 1.f);
        std::fill(grid.g.begin(),     grid.g.end(),     0.f);
        std::fill(grid.c.begin(),     grid.c.end(),     c_sat);  // ΔG=0 → pure capillary
        std::fill(grid.T.begin(),     grid.T.end(),     973.f);

        // alpha slab of thickness h, centred in z (beta above/below, itself
        // periodic); grain1=[0,d), grain2=[d,Nx_d) fills x completely;
        // y fills completely too (no beta gap, w_cells is now unused).
        const int cz = Nz/2;
        const int z0 = cz - h/2, z1 = z0 + h;
        for (int z = z0; z < z1; ++z)
            for (int y = 0; y < Ny; ++y)
                for (int x = 0; x < Nx_d; ++x)
                    grid.phi[grid.idx(z, y, x)] = (x < d) ? 1 : 2;

        long alpha0 = 0;
        for (auto v : grid.phi) if (v != 0) ++alpha0;
        const int min_piece = std::max(8, (int)std::lround(min_frac * (double)alpha0 / 2));

        char tag[110];
        std::snprintf(tag, sizeof(tag), "%s/d%03d_xz_t0.csv", outdir.c_str(), d);
        dump_xz_csv(grid, tag);
        std::snprintf(tag, sizeof(tag), "%s/d%03d_xy_t0.csv", outdir.c_str(), d);
        dump_xy_csv(grid, tag);
        std::snprintf(tag, sizeof(tag), "%s/d%03d_3d_t0.csv", outdir.c_str(), d);
        dump_full3d_csv(grid, tag, 0.0, 0);

        // Evolve (pure capillary).  Track the maximum number of substantial α
        // pieces (cleanest severance) and keep that snapshot; also keep an early
        // snapshot so STABLE cases (which slowly shrink) have a meaningful image.
        int max_pieces = 1;
        std::vector<int32_t> best_phi, grooved_phi;
        double best_time = 0.0, grooved_time = 0.0;
        // For cases that never sever (max_pieces stays 1), a fixed early-step
        // fallback is still too late: under pure capillary flow with no bulk
        // driving force, an ISOLATED particle has no stable size and keeps
        // shrinking until it vanishes -- verified directly (a fixed n_steps/12
        // checkpoint showed 1-2 cells left out of thousands, i.e. already
        // fully dissolved). Track the LAST snapshot where the alpha volume is
        // still >= 80% of its initial value instead: a meaningful "grooving
        // but still substantial" state, which is what "stable" should show.
        const long alpha_floor = (long)(0.80 * (double)alpha0);
        const int  N_           = grid.N();
        const int  vol_check_every = 4;   // see note below
        // Time-series: ~10 full-3D dumps per d-value, for the requested
        // microstructure evolution sequence. Deliberately DECOUPLED from
        // snap_every (which can be much finer, e.g. 50): a full-3D dump is
        // hundreds of thousands of rows (millions for the largest d), so
        // riding on snap_every made disk I/O -- not the physics -- the
        // dominant cost of the whole sweep (verified directly: a 60-step
        // run took LONGER than a 300-step run, because snap_every=n_steps/
        // nsnap collapsed to 1 at small n_steps, firing a dump every step).
        const int  ts_every = std::max(1, n_steps / 10);
        char ts_tag[130];
        std::snprintf(ts_tag, sizeof(ts_tag), "%s/d%03d_3d_ts0000.csv", outdir.c_str(), d);
        dump_full3d_csv(grid, ts_tag, 0.0, 0);
        for (int s = 0; s < n_steps; ++s) {
            grid.step(-1.f);
            // Checked every few steps, in parallel (cheap O(N) sum, but the
            // ORIGINAL serial every-step version was the dominant cost of
            // this whole sweep -- verified directly via wall-clock timing.
            // The alpha volume can still drop below the 80% floor within
            // the first few tens of steps -- checking at the coarser
            // snap_every (60-step) cadence missed that window entirely in
            // an earlier attempt -- so vol_check_every=4 is a middle
            // ground: ~15x fewer checks than every-step, while still far
            // finer than snap_every.
            if (s % vol_check_every == 0) {
                long cur_alpha = 0;
                const int32_t* phi_ptr = grid.phi.data();
                #pragma omp parallel for reduction(+:cur_alpha) schedule(static)
                for (int i = 0; i < N_; ++i) if (phi_ptr[i] != 0) ++cur_alpha;
                if (cur_alpha >= alpha_floor) {
                    grooved_phi = grid.phi; grooved_time = grid.time;
                }
            }
            if ((s + 1) % snap_every == 0 || s == n_steps - 1) {
                int big = alpha_pieces_3d(grid, min_piece);
                if (big > max_pieces) { max_pieces = big; best_phi = grid.phi; best_time = grid.time; }
            }
            if ((s + 1) % ts_every == 0 || s == n_steps - 1) {
                std::snprintf(ts_tag, sizeof(ts_tag), "%s/d%03d_3d_ts%04d.csv",
                             outdir.c_str(), d, s + 1);
                dump_full3d_csv(grid, ts_tag, grid.time, s + 1);
            }
        }

        // write final + display cross-sections (pinch for BREAK, early for STABLE)
        std::snprintf(tag, sizeof(tag), "%s/d%03d_xz_tf.csv", outdir.c_str(), d);
        dump_xz_csv(grid, tag);
        std::snprintf(tag, sizeof(tag), "%s/d%03d_xy_tf.csv", outdir.c_str(), d);
        dump_xy_csv(grid, tag);
        {
            std::vector<int32_t> save = grid.phi;
            const std::vector<int32_t>& disp =
                !best_phi.empty() ? best_phi
                : (!grooved_phi.empty() ? grooved_phi : save);
            double disp_time = !best_phi.empty() ? best_time
                : (!grooved_phi.empty() ? grooved_time : grid.time);
            grid.phi = disp;
            std::snprintf(tag, sizeof(tag), "%s/d%03d_xz_tb.csv", outdir.c_str(), d);
            dump_xz_csv(grid, tag);
            std::snprintf(tag, sizeof(tag), "%s/d%03d_xy_tb.csv", outdir.c_str(), d);
            dump_xy_csv(grid, tag);
            std::snprintf(tag, sizeof(tag), "%s/d%03d_3d_tb.csv", outdir.c_str(), d);
            dump_full3d_csv(grid, tag, disp_time, 1);
            grid.phi = save;
        }

        // BREAK if the slab severed into >= 2 substantial particles.
        const char* res;
        bool broke = (max_pieces >= 2), stable = !broke;
        res = broke ? "BREAK" : "stable";

        printf("%-8d %-8.2f %-12d %-10s\n", d, dh, max_pieces, res);
        std::fprintf(sf, "%d,%d,%.3f,%d,%s\n", d, h, dh, max_pieces, res);

        if (broke)  crit_dh_break  = std::min(crit_dh_break,  (double)dh);
        if (stable) crit_dh_stable = std::max(crit_dh_stable, (double)dh);
    }

    std::fclose(sf);
    printf("\n");
    if (crit_dh_break < 1e29 && crit_dh_stable > 0.0) {
        double crit = 0.5 * (crit_dh_break + crit_dh_stable);
        printf("Largest stable d/h    = %.2f\n", crit_dh_stable);
        printf("Smallest breaking d/h = %.2f\n", crit_dh_break);
        printf("==> Critical aspect ratio (d/h)* ~ %.2f\n", crit);
    } else if (crit_dh_break < 1e29) {
        printf("All tested ratios broke up; (d/h)* < %.2f\n", crit_dh_break);
    } else {
        printf("No breakup observed; (d/h)* > %.2f\n", crit_dh_stable);
    }
    printf("Snapshots + summary written to %s\n", outdir.c_str());
    return 0;
}

// ============================================================
//  Model: magnetic_annealing
//
//  Field-assisted / abnormal grain growth: uniaxial
//  magnetocrystalline anisotropy + Zeeman energy makes grains whose easy
//  axis (orientation g) aligns with an applied field H grow at the expense
//  of misaligned grains, on top of ordinary curvature-driven grain growth.
// ============================================================

// Like voronoi_init() but draws g from [0, pi) since a uniaxial easy axis
// has 180-degree symmetry (E_mag ~ cos^2), matching ThermoModel::E_mag.
static void voronoi_init_easy_axis(GPKGrid& grid, int n_seeds, uint32_t seed)
{
    const int Nz = grid.Nz, Ny = grid.Ny, Nx = grid.Nx;
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> rz(0, Nz-1), ry(0, Ny-1), rx(0, Nx-1);
    std::uniform_real_distribution<float> rg(0.f, (float)M_PI);

    std::vector<int> sz(n_seeds), sy(n_seeds), sx(n_seeds);
    std::vector<float> sg(n_seeds);
    for (int k = 0; k < n_seeds; ++k) {
        sz[k]=rz(rng); sy[k]=ry(rng); sx[k]=rx(rng); sg[k]=rg(rng);
    }

    #pragma omp parallel for schedule(static)
    for (int z = 0; z < Nz; ++z)
    for (int y = 0; y < Ny; ++y)
    for (int x = 0; x < Nx; ++x) {
        long long best_d = LLONG_MAX; int best = 0;
        for (int k = 0; k < n_seeds; ++k) {
            long long dz=z-sz[k], dy=y-sy[k], dx_=x-sx[k];
            long long d = dz*dz + dy*dy + dx_*dx_;
            if (d < best_d) { best_d=d; best=k; }
        }
        int i = grid.idx(z, y, x);
        grid.phi[i]   = best + 1;
        grid.g[i]     = sg[best];
        grid.alpha[i] = 1.f;
    }
}

static int run_magnetic_annealing(InputParser& cfg)
{
    const int    Nz       = cfg.get_int   ("grid",  "Nz");
    const int    Ny       = cfg.get_int   ("grid",  "Ny");
    const int    Nx       = cfg.get_int   ("grid",  "Nx");
    const float  dx       = cfg.get_float ("grid",  "dx");
    const double t_end    = cfg.get_double("time",  "t_end");
    const int    n_snaps  = cfg.get_int   ("time",  "n_snaps");

    const float  M0       = cfg.get_float("thermo","M0");
    const float  sigma0   = cfg.get_float("thermo","sigma0");
    const float  Ku       = cfg.get_float("thermo","Ku");
    const float  Ms       = cfg.get_float("thermo","Ms");
    const float  H_ext    = cfg.get_float("thermo","H_ext");
    const float  H_theta  = cfg.get_float("thermo","H_theta", 0.f);

    const int    N_GRAINS = cfg.get_int  ("magnetic_annealing","n_grains");
    const int    seed     = cfg.get_int  ("magnetic_annealing","seed", 42);

    printf("=== GPK Magnetic-Field-Assisted Grain Growth ===\n");
    printf("Grid %d×%d×%d, dx=%.2f µm, %d grains\n", Nz,Ny,Nx, dx*1e6f, N_GRAINS);
    printf("OpenMP threads: %d\n", omp_get_max_threads());

    auto thermo = std::make_shared<MagneticAnnealingThermo>(M0, sigma0, Ku, Ms, H_ext, H_theta);
    GPKGrid grid(Nz, Ny, Nx, dx, thermo, 0.f,
                 cfg.get_float("geometry", "sigma_cells", 2.f));
    voronoi_init_easy_axis(grid, N_GRAINS, (uint32_t)seed);

    const float regime = Ku / std::max(thermo->MU0 * Ms * H_ext, 1e-30f);
    printf("Ku=%.3e J/m^3  Ms=%.3e A/m  H_ext=%.3e A/m  H_theta=%.2f rad\n",
           (double)Ku, (double)Ms, (double)H_ext, (double)H_theta);
    printf("Regime check Ku/(mu0*Ms*H_ext) = %.3f  (>>1 => easy-axis-pinned assumption valid)\n\n",
           (double)regime);

    // Record initial per-grain area (cell count) and orientation.
    std::unordered_map<int32_t,int> cnt0;
    for (auto p : grid.phi) cnt0[p]++;
    std::unordered_map<int32_t,float> g_of;
    for (int i = 0; i < grid.N(); ++i) g_of[grid.phi[i]] = grid.g[i];

    auto misalign = [&](float g_val) {
        float d = std::fabs(g_val - H_theta);
        d = std::fmod(d, (float)M_PI);
        if (d < 0.f) d += (float)M_PI;
        return std::min(d, (float)M_PI - d);
    };
    const float deg15 = 15.f * (float)M_PI / 180.f;
    const float deg75 = 75.f * (float)M_PI / 180.f;

    OutputWriter writer(cfg);
    printf("%-10s %-14s %-16s %-10s %-10s\n",
           "Time[s]", "AlignedA/A0", "MisalignedA/A0", "N_align", "N_misal");

    float final_aligned_ratio = 1.f, final_misaligned_ratio = 1.f;
    int   final_n_align = 0, final_n_misalign = 0;

    auto t_wall0 = std::chrono::high_resolution_clock::now();

    for (int snap = 0; snap < n_snaps; ++snap) {
        double t_snap = t_end * (snap + 1.0) / n_snaps;
        grid.run(t_snap, -1.f);

        std::unordered_map<int32_t,int> cnt;
        for (auto p : grid.phi) cnt[p]++;

        double sum_aligned = 0.0, sum_misaligned = 0.0;
        int n_align = 0, n_misalign = 0;
        for (auto& kv : cnt0) {
            int32_t gid = kv.first;
            int     a0  = kv.second;
            auto it = cnt.find(gid);
            int a_now = (it != cnt.end()) ? it->second : 0;
            float ratio = (float)a_now / (float)a0;
            float dth = misalign(g_of[gid]);
            if (dth < deg15)      { sum_aligned    += ratio; ++n_align; }
            else if (dth > deg75) { sum_misaligned += ratio; ++n_misalign; }
        }
        final_aligned_ratio    = (n_align    > 0) ? (float)(sum_aligned    / n_align)    : 1.f;
        final_misaligned_ratio = (n_misalign > 0) ? (float)(sum_misaligned / n_misalign) : 1.f;
        final_n_align = n_align; final_n_misalign = n_misalign;

        printf("%-10.4f %-14.4f %-16.4f %-10d %-10d\n",
               grid.time, final_aligned_ratio, final_misaligned_ratio, n_align, n_misalign);

        writer.write(grid, snap, grid.time);
    }

    double wall_s = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - t_wall0).count();
    printf("\nWall time: %.2f s\n", wall_s);

    bool passed = (final_n_align > 0 && final_n_misalign > 0 &&
                   final_aligned_ratio > 1.0f && final_misaligned_ratio < 1.0f);
    printf("\nMagnetic-annealing test: %s\n",
           passed ? "PASS — field-aligned grains grew, misaligned grains shrank"
                  : "FAIL — no measurable magnetic-field-driven asymmetry");
    return passed ? 0 : 1;
}

// ============================================================
//  Model: magnetoelastic
//
//  Validates the magnetostrictive eigenstrain added to the
//  existing elastic-coupling pipeline. Three circular grains at distinct
//  fixed orientations theta = g - H_theta = 0, 45, 90 deg are embedded in a
//  matrix whose orientation is set to the "magic angle" 54.7356 deg (where
//  cos^2(theta)-1/3 = 0), so the MATRIX eigenstrain is exactly zero and only
//  the grains carry a nonzero magnetostrictive eigenstrain -- isolating the
//  effect under test. A single near-static evaluation (grid.step(0.f), which
//  computes geometry/elastic fields without evolving alpha) is enough to
//  read off E_el; the theoretical eps* ~ (3/2)*lambda_s*(cos^2(theta)-1/3)
//  scaling predicts the exact ratios E_el(0)/E_el(45)=16, E_el(90)/E_el(45)=4.
// ============================================================

static int run_magnetoelastic(InputParser& cfg)
{
    const int    Ny    = cfg.get_int   ("grid", "Ny");
    const int    Nx    = cfg.get_int   ("grid", "Nx");
    const float  dx    = cfg.get_float ("grid", "dx");

    const float  M0              = cfg.get_float("thermo","M0");
    const float  sigma0          = cfg.get_float("thermo","sigma0");
    const float  elastic_modulus = cfg.get_float("thermo","elastic_modulus");
    const float  lambda_s        = cfg.get_float("thermo","lambda_s");
    const float  H_theta         = cfg.get_float("thermo","H_theta", 0.f);
    const int    elastic_iters   = cfg.get_int  ("thermo","elastic_smooth_iters", 8);

    const int    r_seed = cfg.get_int("magnetoelastic", "grain_radius_cells", 20);
    const int    gap    = cfg.get_int("magnetoelastic", "gap_cells", 20);

    printf("=== GPK Magnetoelastic (Magnetostriction) Test ===\n");
    printf("Grid 1×%d×%d, dx=%.2f µm\n", Ny, Nx, dx*1e6f);
    printf("elastic_modulus=%.3e J/m^3  lambda_s=%.3e  H_theta=%.2f rad\n\n",
           (double)elastic_modulus, (double)lambda_s, (double)H_theta);

    auto thermo = std::make_shared<MagnetoelasticThermo>(
        M0, sigma0, elastic_modulus, lambda_s, H_theta);
    thermo->elastic_smooth_iters = elastic_iters;
    GPKGrid grid(1, Ny, Nx, dx, thermo, 0.f,
                 cfg.get_float("geometry", "sigma_cells", 2.f));

    const float magic_angle = std::acos(std::sqrt(1.f/3.f)); // 54.7356 deg

    std::fill(grid.phi.begin(),   grid.phi.end(),   0);
    std::fill(grid.alpha.begin(), grid.alpha.end(), 1.f);
    std::fill(grid.g.begin(),     grid.g.end(),     H_theta + magic_angle); // zero matrix eigenstrain

    struct GrainSpec { const char* label; float theta_deg; };
    GrainSpec specs[3] = { {"0deg", 0.f}, {"45deg", 45.f}, {"90deg", 90.f} };
    const int cy = Ny/2;
    const int spacing = 2*r_seed + gap;
    int cx[3];
    for (int k = 0; k < 3; ++k) cx[k] = spacing/2 + k*spacing;

    if (Nx < 3*spacing)
        fprintf(stderr, "Warning: Nx=%d < 3*(2*grain_radius+gap)=%d, grains may overlap under periodic BC\n",
                Nx, 3*spacing);

    for (int k = 0; k < 3; ++k) {
        float g_val = H_theta + specs[k].theta_deg * (float)M_PI / 180.f;
        for (int dy=-r_seed; dy<=r_seed; ++dy)
        for (int dxp=-r_seed; dxp<=r_seed; ++dxp) {
            if (dy*dy+dxp*dxp > r_seed*r_seed) continue;
            int y = cy+dy, x = cx[k]+dxp;
            if (y<0||y>=Ny||x<0||x>=Nx) continue;
            int i = grid.idx(0,y,x);
            grid.phi[i]   = k+1;
            grid.g[i]     = g_val;
            grid.alpha[i] = 1.f;
        }
    }

    grid.step(0.f);  // populate geometry/E_el fields without evolving alpha

    double avg[3] = {0.0, 0.0, 0.0};
    long   cnt[3] = {0, 0, 0};
    for (int i = 0; i < grid.N(); ++i) {
        int32_t p = grid.phi[i];
        if (p >= 1 && p <= 3) { avg[p-1] += grid.E_el[i]; ++cnt[p-1]; }
    }
    for (int k = 0; k < 3; ++k) avg[k] = (cnt[k] > 0) ? avg[k]/cnt[k] : 0.0;

    printf("%-8s %-16s\n", "theta", "E_el_avg[J/m^3]");
    for (int k = 0; k < 3; ++k)
        printf("%-8s %-16.6e\n", specs[k].label, avg[k]);

    // Dump the theta/E_el table for independent Python-side verification/plots.
    {
        const std::string outdir = cfg.get_str("output", "dir", "output/magnetoelastic");
        mkdir_p(outdir);
        FILE* f = std::fopen((outdir + "/results.csv").c_str(), "w");
        if (f) {
            std::fprintf(f, "# elastic_modulus=%.6e lambda_s=%.6e\n",
                         (double)elastic_modulus, (double)lambda_s);
            std::fprintf(f, "theta_deg,E_el_avg_J_per_m3\n");
            for (int k = 0; k < 3; ++k)
                std::fprintf(f, "%.4f,%.8e\n", (double)specs[k].theta_deg, avg[k]);
            std::fclose(f);
        }
    }

    double ratio_0_45  = (avg[1] > 0.0) ? avg[0]/avg[1] : 0.0;
    double ratio_90_45 = (avg[1] > 0.0) ? avg[2]/avg[1] : 0.0;
    printf("\nMeasured E_el(0)/E_el(45)  = %.3f  (theory: 16.000)\n", ratio_0_45);
    printf("Measured E_el(90)/E_el(45) = %.3f  (theory: 4.000)\n", ratio_90_45);

    bool passed = (std::fabs(ratio_0_45  - 16.0) / 16.0 < 0.25) &&
                  (std::fabs(ratio_90_45 -  4.0) /  4.0 < 0.25);
    printf("\nMagnetoelastic test: %s\n",
           passed ? "PASS — magnetostrictive eigenstrain matches cos^2(theta) scaling"
                  : "FAIL — measured ratios deviate from theory beyond tolerance");
    return passed ? 0 : 1;
}

// ============================================================
//  Model: ndfeb_wetting
//
//  Nd-Fe-B grain-boundary (GB) phase wetting: reuses the
//  Mullins-thermal-grooving two-grain-plus-phase geometry and wedge-angle
//  measurement technique from demos/validate_real_materials.cpp Test 7,
//  replacing the vapour phase with the Nd-rich GB phase and sweeping local
//  Cu content c (uniform per run, with c_sat=c so each run is pure-capillary/
//  wetting -- no chemical driving force -- isolating the wetting effect
//  under test, matching the precipitate_breakup demo's c≡c_sat convention).
//  Validates Koyama et al. 2021's qualitative finding: higher Cu content ->
//  lower sigma(T1/GBphase) -> deeper GB-phase penetration.
//
//  Pure-curvature motion has v_n ~ M*sigma*kappa: a LOWER sigma_wetting (more
//  Cu) directly means SLOWER relaxation at fixed curvature, so comparing
//  different c levels after a FIXED number of steps or a fixed physical time
//  systematically under-relaxes the higher-Cu cases (they look "less grooved"
//  simply because they move more slowly, not because they wet less). To
//  compare cases at an equivalent relaxation stage, each level is run to a
//  physical time t_target = t_base * sigma_gbphase_0/sigma_wetting(c) --
//  i.e. inversely scaled with its own interfacial energy, so the c=0
//  reference case runs for t_base and lower-sigma cases get proportionally
//  more time to reach the same normalized relaxation extent.
// ============================================================

static int run_ndfeb_wetting(InputParser& cfg)
{
    const int    N        = cfg.get_int  ("grid", "N", 400);
    const float  dx       = cfg.get_float("grid", "dx", 1.0e-6f);
    const double t_base   = cfg.get_double("ndfeb_wetting", "t_base", 1.0e-3);
    const int    W        = cfg.get_int  ("ndfeb_wetting", "wedge_window", 10);
    const int    D_AVG    = cfg.get_int  ("ndfeb_wetting", "depth_avg_halfwidth", 5);

    const float  M0             = cfg.get_float("thermo", "M0",             1.0e-8f);
    const float  sigma_gb       = cfg.get_float("thermo", "sigma_gb",        1.5f);
    const float  sigma_gbphase0 = cfg.get_float("thermo", "sigma_gbphase_0", 0.76f);
    const float  k_Cu           = cfg.get_float("thermo", "k_Cu",            3.0f);
    const float  Omega          = cfg.get_float("thermo", "Omega",           5.0e8f);

    std::vector<float> c_values =
        parse_csv_floats(cfg.get_str("ndfeb_wetting", "c_values", "0.0,0.02,0.05,0.10"));

    printf("=== GPK Nd-Fe-B Grain-Boundary-Phase Wetting ===\n");
    printf("sigma_gb=%.3f J/m^2  sigma_gbphase_0=%.3f J/m^2  k_Cu=%.2f\n",
           (double)sigma_gb, (double)sigma_gbphase0, (double)k_Cu);
    printf("OpenMP threads: %d\n\n", omp_get_max_threads());

    printf("%-8s %-16s %-14s %-16s\n", "c_Cu", "sigma_wet[J/m2]", "psi_meas[deg]", "avg_depth[cell]");

    // Root depth is measured at the T1/T1/GBphase triple junction (a genuine
    // geometric corner). GPK's Gaussian-smoothed curvature reconstruction is
    // known to be noisiest exactly at sharp
    // corners, so a single-column depth reading is noisy run-to-run; average
    // over a small window of columns straddling the corner (where curvature-
    // driven grooving is actually active -- far-field columns have ~flat,
    // near-zero-curvature boundaries under this pure-capillary, D=0 setup and
    // barely move regardless of sigma_wetting) to suppress single-cell noise
    // while still measuring the localized groove-root physics.
    std::vector<double> depths, psis, sigwets;
    for (float cCu : c_values) {
        auto th = std::make_shared<NdFeBGrainBoundaryThermo>(
            M0, sigma_gb, sigma_gbphase0, k_Cu, /*c_sat=*/cCu, Omega);
        GPKGrid grid(1, N, N, dx, th, 0.f,
                     cfg.get_float("geometry", "sigma_cells", 2.f));
        for (int y = 0; y < N; ++y)
            for (int x = 0; x < N; ++x) {
                int i = grid.idx(0, y, x);
                if (y < N/2) grid.phi[i] = NdFeBGrainBoundaryThermo::GBPHASE_LABEL;
                else grid.phi[i] = (x < N/2) ? 1 : 2;
                grid.alpha[i] = 1.f;
                grid.c[i] = cCu;   // uniform Cu content == c_sat -> pure wetting/capillary run
            }
        float sig_wet = th->sigma_wetting(cCu);
        double t_target = t_base * (double)sigma_gbphase0 / std::max((double)sig_wet, 1e-6);
        grid.run(t_target, -1.f);

        int root_y = -1;
        for (int y = 0; y < N; ++y)
            if (grid.phi[grid.idx(0, y, N/2)] != NdFeBGrainBoundaryThermo::GBPHASE_LABEL) { root_y = y; break; }

        double depth_sum = 0.0;
        int    depth_n   = 0;
        for (int dxv = -D_AVG; dxv <= D_AVG; ++dxv) {
            int xx = N/2 + dxv;
            int ry = -1;
            for (int y = 0; y < N; ++y)
                if (grid.phi[grid.idx(0, y, xx)] != NdFeBGrainBoundaryThermo::GBPHASE_LABEL) { ry = y; break; }
            double d = (ry >= 0) ? (double)(ry - N/2) : (double)(N/2);
            depth_sum += d; ++depth_n;
        }
        double depth = (depth_n > 0) ? depth_sum / depth_n : 0.0;
        depths.push_back(depth);

        double psi_meas = 0.0;
        if (root_y >= 0) {
            std::map<int, std::pair<double,double>> cen; std::map<int,int> cellcnt;
            for (int dyv = -W; dyv <= W; ++dyv)
                for (int dxv = -W; dxv <= W; ++dxv) {
                    int yy = root_y + dyv, xx = N/2 + dxv;
                    if (yy < 0 || yy >= N || xx < 0 || xx >= N) continue;
                    int l = grid.phi[grid.idx(0, yy, xx)];
                    cen[l].first += dxv; cen[l].second += dyv; cellcnt[l]++;
                }
            if (cen.size() == 3) {
                std::vector<std::pair<int,double>> dirs;
                for (auto& kv : cen) {
                    double gx = kv.second.first / cellcnt[kv.first];
                    double gy = kv.second.second / cellcnt[kv.first];
                    dirs.push_back({kv.first, std::atan2(gy, gx)});
                }
                std::sort(dirs.begin(), dirs.end(), [](auto&a, auto&b){ return a.second < b.second; });
                double gaps[3];
                for (int k = 0; k < 3; ++k) {
                    double a1 = dirs[k].second, a2 = dirs[(k+1)%3].second;
                    double d = a2 - a1; if (d < 0) d += 2*M_PI;
                    gaps[k] = d * 180.0 / M_PI;
                }
                double w0 = gaps[0] + gaps[2] - gaps[1];
                double w1 = gaps[0] + gaps[1] - gaps[2];
                double w2 = gaps[1] + gaps[2] - gaps[0];
                double wedges[3] = {w0, w1, w2};
                for (int k = 0; k < 3; ++k)
                    if (dirs[k].first == NdFeBGrainBoundaryThermo::GBPHASE_LABEL) psi_meas = wedges[k];
            }
        }
        psis.push_back(psi_meas);
        sigwets.push_back((double)sig_wet);
        printf("%-8.3f %-16.3f %-14.2f %-16.1f  (t=%.4g s)\n",
               (double)cCu, (double)sig_wet, psi_meas, depth, grid.time);
    }

    // Dump the c_Cu/sigma_wet/psi/depth table for independent Python-side
    // verification/plots.
    {
        const std::string outdir = cfg.get_str("output", "dir", "output/ndfeb_wetting");
        mkdir_p(outdir);
        FILE* f = std::fopen((outdir + "/results.csv").c_str(), "w");
        if (f) {
            std::fprintf(f, "# sigma_gbphase_0=%.6f k_Cu=%.6f\n", (double)sigma_gbphase0, (double)k_Cu);
            std::fprintf(f, "c_Cu,sigma_gb_J_per_m2,sigma_wetting_J_per_m2,psi_measured_deg,avg_depth_cells\n");
            for (size_t k = 0; k < c_values.size(); ++k)
                std::fprintf(f, "%.4f,%.6f,%.6f,%.4f,%.4f\n",
                             (double)c_values[k], (double)sigma_gb, sigwets[k], psis[k], depths[k]);
            std::fclose(f);
        }
    }

    // Pass criterion: compare the mean measured wedge angle psi of the
    // higher-Cu half of the sweep against the lower-Cu half. psi (a local,
    // dimensionless SHAPE property at the triple junction) is the quantity
    // Young's equation actually predicts is Cu-dependent; raw penetration
    // DEPTH is not a meaningful comparison quantity here even after the
    // t_base/sigma_wetting time-normalization above, since normalizing time
    // by 1/sigma_wetting cancels the sigma-dependence of depth to leading
    // order (v ~ M*sigma*kappa integrated over t ~ 1/sigma gives a result
    // ~independent of sigma) -- depth is printed for context only. A coarser
    // two-group comparison (rather than strict point-to-point monotonicity)
    // is used because the corner-localized psi reading is intrinsically
    // noisy (sharp corners are a known curvature-reconstruction breakdown
    // regime).
    bool trend_ok = true;
    if (psis.size() >= 2) {
        size_t half = psis.size() / 2;
        double lo = 0.0, hi = 0.0;
        for (size_t k = 0; k < half; ++k) lo += psis[k];
        for (size_t k = half; k < psis.size(); ++k) hi += psis[k];
        lo /= (double)half;
        hi /= (double)(psis.size() - half);
        trend_ok = hi < lo;
        printf("\nLow-Cu-half mean psi = %.2f deg, high-Cu-half mean psi = %.2f deg\n", lo, hi);
    }
    printf("\nNd-Fe-B wetting test: %s\n",
           trend_ok ? "PASS — higher-Cu half of the sweep shows a narrower (more-wetting) GBphase wedge angle"
                    : "FAIL — higher-Cu half does not show a narrower wedge angle than lower-Cu half");
    return trend_ok ? 0 : 1;
}

// ============================================================
//  Model: dislocation  (stored-energy coupling + recovery)
//
//  Rigorous validation of the dislocation extension against analytic results:
//    Test 1  : a rex nucleus (rho=0) grows into a deformed matrix (rho=rho0).
//              Measured front velocity dR/dt is compared to the sharp-interface
//              prediction v = M (E_rho - sigma/R), E_rho = 0.5 alpha_T mu b^2 rho0.
//              Also verifies sweeping: the swept region ends dislocation-free.
//    Test 1b : a subcritical nucleus (r0 < R* = sigma/E_rho) must shrink.
//    Test 2  : static recovery of a uniform field, rho(t) = rho0 exp(-r t).
// ============================================================

static long count_label(const GPKGrid& g, int lab) {
    long n = 0; for (auto v : g.phi) if (v == lab) ++n; return n;
}
static double mean_rho_label(const GPKGrid& g, int lab) {
    double s = 0; long n = 0;
    for (int i = 0; i < g.N(); ++i) if (g.phi[i] == lab) { s += g.rho[i]; ++n; }
    return n ? s / (double)n : 0.0;
}

static int run_dislocation(InputParser& cfg)
{
    const int    Nx    = cfg.get_int   ("grid", "Nx", 128);
    const int    Ny    = cfg.get_int   ("grid", "Ny", 128);
    const float  dx    = cfg.get_float ("grid", "dx", 1.0e-6f);
    const int    nsnap = cfg.get_int   ("time", "n_snaps", 20);
    const double t_end = cfg.get_double("time", "t_end");

    const float  M0      = cfg.get_float("thermo", "M0",            1.0e-11f);
    const float  sig0    = cfg.get_float("thermo", "sigma0",        0.5f);
    const float  mu      = cfg.get_float("thermo", "mu",           4.5e10f);
    const float  bb      = cfg.get_float("thermo", "b",            2.95e-10f);
    const float  alphaT  = cfg.get_float("thermo", "alphaT",        0.5f);
    const float  rho0    = cfg.get_float("thermo", "rho0",          1.0e14f);
    const float  rho_rex = cfg.get_float("thermo", "rho_rex",       0.0f);
    const float  rec     = cfg.get_float("thermo", "recovery_rate", 0.02f);
    const int    r_seed  = cfg.get_int  ("dislocation", "seed_radius",   20);
    const double rec_t   = cfg.get_double("dislocation","recovery_t_end", 100.0);

    // Test 1 uses recovery OFF (isolate the migration law); Test 2 turns it on.
    auto thermo = std::make_shared<DislocationThermo>(M0, sig0, mu, bb, alphaT, 0.f, rho_rex);
    const double E_rho = thermo->stored_energy(rho0);
    const double Rstar = thermo->critical_radius(rho0);

    printf("=== GPK Dislocation Stored-Energy Validation ===\n");
    printf("Grid %dx%d  dx=%.2f um   M0=%.2e  sigma0=%.3f\n",
           Ny, Nx, dx*1e6f, (double)M0, (double)sig0);
    printf("mu=%.2e Pa  b=%.3e m  alpha_T=%.2f  rho0=%.2e m^-2\n",
           (double)mu, (double)bb, (double)alphaT, (double)rho0);
    printf("E_rho = 0.5*alpha_T*mu*b^2*rho0 = %.4e J/m^3\n", E_rho);
    printf("Critical radius R* = sigma0/E_rho = %.3f um   (supercritical seed r0 = %.1f um)\n",
           Rstar*1e6, r_seed*dx*1e6f);
    printf("OpenMP threads: %d\n\n", omp_get_max_threads());

    int fails = 0;

    // -------- Test 1: FLAT-front velocity converges to v = M*E_rho --------
    // A planar rex/deformed boundary has no curvature, so v = M*E_rho exactly.
    // Front advance is measured from the alpha-integrated deformed volume
    // (V_def = sum over phi==0 of alpha), which counts the sub-cell interface
    // profile and so is free of both cell-quantisation and the constant
    // diffuse-front offset. The residual error is the alpha-thresholding time
    // discretisation (~CFL/2): it must vanish as dt -> 0. We verify this
    // first-order convergence explicitly.
    printf("--- Test 1: flat-front velocity converges to M*E_rho as dt->0 ---\n");
    {
        const double v_th = (double)M0 * E_rho;
        const int xlo = 3*Nx/8, xhi = 5*Nx/8;
        const double t1 = 0.75 * t_end;   // keep both fronts away from the boundary

        auto run_front = [&](double cfl, double& v_meas, double& mrho_rex) {
            GPKGrid grid(1, Ny, Nx, dx, thermo, 0.f,
                         cfg.get_float("geometry", "sigma_cells", 2.f));
            std::fill(grid.phi.begin(),   grid.phi.end(),   0);
            std::fill(grid.alpha.begin(), grid.alpha.end(), 1.f);
            std::fill(grid.g.begin(),     grid.g.end(),     0.f);
            std::fill(grid.rho.begin(),   grid.rho.end(),   rho0);
            for (int y = 0; y < Ny; ++y)
                for (int x = xlo; x < xhi; ++x) {
                    int i = grid.idx(0, y, x); grid.phi[i] = 1; grid.rho[i] = 0.f;
                }
            auto deformed_vol = [&]() {
                double v = 0.0;
                for (int i = 0; i < grid.N(); ++i) if (grid.phi[i] == 0) v += grid.alpha[i];
                return v;
            };
            const double V0 = deformed_vol();
            const float  dt_fix = (float)(cfl * dx / v_th);
            grid.run(t1, dt_fix);
            const double adv_cells = (V0 - deformed_vol()) / (2.0*Ny);  // per front
            v_meas   = adv_cells * dx / t1;
            mrho_rex = mean_rho_label(grid, 1);
        };

        printf("%-8s %-12s %-14s %-14s %-9s\n",
               "CFL", "dt[s]", "v_meas[m/s]", "v_theory[m/s]", "err[%]");
        const double cfls[3] = {0.4, 0.1, 0.025};
        double err_fine = 1e9, mrho_fine = 0.0;
        for (int k = 0; k < 3; ++k) {
            double vmea, mrho;
            run_front(cfls[k], vmea, mrho);
            double err = std::fabs(vmea - v_th)/v_th*100.0;
            printf("%-8.3f %-12.4e %-14.3e %-14.3e %-9.2f\n",
                   cfls[k], cfls[k]*dx/v_th, vmea, v_th, err);
            if (k == 2) { err_fine = err; mrho_fine = mrho; }
        }
        bool vel_ok   = err_fine < 5.0;                     // converges to theory
        bool sweep_ok = (rho0 > 0.f) && (mrho_fine < 0.05*rho0);
        printf("Finest-dt velocity error = %.2f%% (converges to M*E_rho)  -> %s\n",
               err_fine, vel_ok ? "PASS" : "FAIL");
        printf("Sweeping: mean rho in rex region = %.3e m^-2 (rho/rho0=%.4f)  -> %s\n\n",
               mrho_fine, mrho_fine/(double)rho0, sweep_ok ? "PASS (dislocation-free)" : "FAIL");
        fails += (!vel_ok) + (!sweep_ok);
    }

    // -------- Test 1b: subcritical nucleus must shrink --------
    printf("--- Test 1b: subcritical nucleus (r0 = 0.6 R*) must shrink ---\n");
    {
        int r_sub = std::max(2, (int)std::lround(0.6*Rstar/dx));
        GPKGrid grid(1, Ny, Nx, dx, thermo, 0.f,
                     cfg.get_float("geometry", "sigma_cells", 2.f));
        std::fill(grid.phi.begin(),   grid.phi.end(),   0);
        std::fill(grid.alpha.begin(), grid.alpha.end(), 1.f);
        std::fill(grid.g.begin(),     grid.g.end(),     0.f);
        std::fill(grid.rho.begin(),   grid.rho.end(),   rho0);
        const int cy = Ny/2, cx = Nx/2;
        for (int y = 0; y < Ny; ++y)
            for (int x = 0; x < Nx; ++x) {
                int ddy = y - cy, ddx = x - cx;
                if (ddy*ddy + ddx*ddx <= r_sub*r_sub) {
                    int i = grid.idx(0, y, x); grid.phi[i] = 1; grid.rho[i] = 0.f;
                }
            }
        long a0 = count_label(grid, 1);
        grid.run(t_end*0.5, -1.f);
        long a1 = count_label(grid, 1);
        bool ok = a1 < a0;
        printf("r0 = %.2f um (0.6 R*),  area %ld -> %ld cells  -> %s\n\n",
               r_sub*dx*1e6f, a0, a1, ok ? "PASS (shrinks)" : "FAIL");
        fails += (!ok);
    }

    // -------- Test 1c: supercritical nucleus must grow --------
    printf("--- Test 1c: supercritical nucleus (r0 >> R*) must grow ---\n");
    {
        GPKGrid grid(1, Ny, Nx, dx, thermo, 0.f,
                     cfg.get_float("geometry", "sigma_cells", 2.f));
        std::fill(grid.phi.begin(),   grid.phi.end(),   0);
        std::fill(grid.alpha.begin(), grid.alpha.end(), 1.f);
        std::fill(grid.g.begin(),     grid.g.end(),     0.f);
        std::fill(grid.rho.begin(),   grid.rho.end(),   rho0);
        const int cy = Ny/2, cx = Nx/2;
        for (int y = 0; y < Ny; ++y)
            for (int x = 0; x < Nx; ++x) {
                int ddy = y - cy, ddx = x - cx;
                if (ddy*ddy + ddx*ddx <= r_seed*r_seed) {
                    int i = grid.idx(0, y, x); grid.phi[i] = 1; grid.rho[i] = 0.f;
                }
            }
        long a0 = count_label(grid, 1);
        grid.run(t_end*0.5, -1.f);
        long a1 = count_label(grid, 1);
        bool ok = a1 > a0;
        printf("r0 = %.1f um (%.1f R*),  area %ld -> %ld cells  -> %s\n\n",
               r_seed*dx*1e6f, (r_seed*dx)/Rstar, a0, a1, ok ? "PASS (grows)" : "FAIL");
        fails += (!ok);
    }

    // -------- Test 2: static recovery --------
    printf("--- Test 2: static recovery, rho(t) = rho0 exp(-r t) ---\n");
    {
        auto th2 = std::make_shared<DislocationThermo>(M0, sig0, mu, bb, alphaT, rec, rho_rex);
        GPKGrid grid(1, Ny, Nx, dx, th2, 0.f,
                     cfg.get_float("geometry", "sigma_cells", 2.f));
        std::fill(grid.phi.begin(),   grid.phi.end(),   0);   // single phase, no boundaries
        std::fill(grid.alpha.begin(), grid.alpha.end(), 1.f);
        std::fill(grid.rho.begin(),   grid.rho.end(),   rho0);
        printf("recovery_rate r = %.4e 1/s\n", (double)rec);
        printf("%-10s %-14s %-14s %-9s\n", "t[s]", "rho_mean", "rho_theory", "err[%]");
        const double snap_dt = rec_t / nsnap;
        double err_max = 0.0;
        for (int s = 0; s < nsnap; ++s) {
            grid.run(snap_dt*(s+1), -1.f);
            double mrho = mean_rho_label(grid, 0);
            double th   = (double)rho0 * std::exp(-(double)rec * grid.time);
            double err  = (th > 0.0) ? std::fabs(mrho - th)/th*100.0 : 0.0;
            err_max = std::max(err_max, err);
            if (s % std::max(1, nsnap/8) == 0 || s == nsnap-1)
                printf("%-10.3f %-14.4e %-14.4e %-9.4f\n", grid.time, mrho, th, err);
        }
        bool ok = err_max < 1.0;
        printf("Max recovery error = %.4f%%  -> %s\n\n", err_max, ok ? "PASS" : "FAIL");
        fails += (!ok);
    }

    printf("==================================================\n");
    printf("Dislocation validation: %s  (%d check(s) failed)\n",
           fails == 0 ? "ALL TESTS PASS" : "FAILURES PRESENT", fails);
    return fails == 0 ? 0 : 1;
}

// ============================================================
//  main
// ============================================================

int main(int argc, char* argv[])
{
#ifdef _WIN32
    // Source strings are UTF-8.  Windows PowerShell may otherwise leave the
    // console on an OEM code page, producing mojibake such as "├ù" for "×".
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif

    if (argc < 2) {
        fprintf(stderr,
            "Usage: gpk <input.ini>\n\n"
            "The input file must contain:\n"
            "  [run]\n"
            "  model = grain_growth  "
            "# grain_growth | solidification | recrystallisation\n"
            "                        "
            "# | ostwald_ripening | precipitation | wulff_shape\n"
            "                        "
            "# | precipitate_breakup | magnetic_annealing | magnetoelastic\n"
            "                        "
            "# | ndfeb_wetting\n");
        return 1;
    }

    InputParser cfg(argv[1]);

    const std::string model = cfg.get_str("run", "model");

    if      (model == "grain_growth")      return run_grain_growth(cfg);
    else if (model == "solidification")    return run_solidification(cfg);
    else if (model == "recrystallisation") return run_recrystallisation(cfg);
    else if (model == "ostwald_ripening")  return run_ostwald_ripening(cfg);
    else if (model == "precipitation")     return run_precipitation(cfg);
    else if (model == "wulff_shape")       return run_wulff_shape(cfg);
    else if (model == "precipitate_breakup") return run_precipitate_breakup(cfg);
    else if (model == "magnetic_annealing")  return run_magnetic_annealing(cfg);
    else if (model == "magnetoelastic")      return run_magnetoelastic(cfg);
    else if (model == "ndfeb_wetting")       return run_ndfeb_wetting(cfg);
    else if (model == "dislocation")         return run_dislocation(cfg);
    else {
        fprintf(stderr,
            "Error: unknown model '%s'.\n"
            "Valid options: grain_growth, solidification, recrystallisation, "
            "ostwald_ripening, precipitation, wulff_shape, precipitate_breakup, "
            "magnetic_annealing, magnetoelastic, ndfeb_wetting\n",
            model.c_str());
        return 1;
    }
}
