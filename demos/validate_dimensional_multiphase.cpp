/* Dimensional geometry and multiphase validation for the GPK manuscript. */
#include "gpk_core.hpp"
#include "gpk_thermo.hpp"
#include "gpk_io.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <string>
#include <vector>

using namespace gpk;

static const std::string OUTDIR = "output/validate_dimensional_multiphase/";
static int passed = 0, failed = 0;

static void check(const char* name, bool ok, const char* detail) {
    std::printf("  %-58s %s  (%s)\n", name, ok ? "PASS" : "FAIL", detail);
    ok ? ++passed : ++failed;
}

static void scalar(const char* name, double measured, double theory, double tol_pct) {
    const double err = theory != 0.0 ? 100.0*std::fabs(measured-theory)/std::fabs(theory)
                                     : 100.0*std::fabs(measured);
    const bool ok = err <= tol_pct;
    std::printf("  %-58s sim=%-12.6g theory=%-12.6g err=%6.2f%%  %s\n",
                name, measured, theory, err, ok ? "PASS" : "FAIL");
    ok ? ++passed : ++failed;
}

static void write_metrics(const std::vector<std::string>& rows) {
    std::FILE* f = std::fopen((OUTDIR + "dimensional_multiphase_metrics.csv").c_str(), "w");
    if (!f) return;
    std::fprintf(f, "metric,value\n");
    for (const auto& row : rows) std::fprintf(f, "%s\n", row.c_str());
    std::fclose(f);
}

struct CurvatureStats { double mean=0.0, cv=0.0, p95_rel=0.0; int n=0; };

static CurvatureStats geometry_stats(GPKGrid& grid, int label, double theory) {
    grid.step(0.f);
    std::vector<double> values;
    for (int i=0; i<grid.N(); ++i)
        if (grid.mask[i] && grid.phi[i] == label && grid.kappa_f[i] > 0.0f)
            values.push_back(grid.kappa_f[i]);
    CurvatureStats s;
    s.n = static_cast<int>(values.size());
    if (values.empty()) return s;
    s.mean = std::accumulate(values.begin(), values.end(), 0.0)/values.size();
    double var=0.0;
    std::vector<double> rel;
    rel.reserve(values.size());
    for (double v : values) { var += (v-s.mean)*(v-s.mean); rel.push_back(std::fabs(v-theory)/theory); }
    s.cv = std::sqrt(var/values.size())/std::fabs(s.mean);
    std::sort(rel.begin(), rel.end());
    s.p95_rel = rel[static_cast<size_t>(0.95*(rel.size()-1))];
    return s;
}

static CurvatureStats sphere_test(int N, int R, float sigma_cells) {
    auto th = std::make_shared<IsotropicGrowthThermo>(1.f, 1.f);
    GPKGrid grid(N,N,N,1.f,th,0.f,sigma_cells);
    const int c=N/2;
    for (int z=0;z<N;++z) for(int y=0;y<N;++y) for(int x=0;x<N;++x) {
        const int dz=z-c, dy=y-c, dx=x-c;
        grid.phi[grid.idx(z,y,x)] = (dx*dx+dy*dy+dz*dz <= R*R) ? 1 : 0;
    }
    return geometry_stats(grid,1,2.0/R);
}

static CurvatureStats cylinder_test(int N, int R, float sigma_cells) {
    auto th = std::make_shared<IsotropicGrowthThermo>(1.f, 1.f);
    GPKGrid grid(N,N,N,1.f,th,0.f,sigma_cells);
    const int c=N/2;
    for (int z=0;z<N;++z) for(int y=0;y<N;++y) for(int x=0;x<N;++x) {
        const int dy=y-c, dx=x-c;
        grid.phi[grid.idx(z,y,x)] = (dx*dx+dy*dy <= R*R) ? 1 : 0;
    }
    return geometry_stats(grid,1,1.0/R);
}

static double flat_curvature(int Nz) {
    auto th = std::make_shared<IsotropicGrowthThermo>(1.f, 1.f);
    GPKGrid grid(Nz,24,64,1.f,th,0.f,2.f);
    for (int z=0;z<Nz;++z) for(int y=0;y<24;++y) for(int x=0;x<64;++x)
        grid.phi[grid.idx(z,y,x)] = x<32 ? 1 : 0;
    grid.step(0.f);
    double maximum=0.0;
    for (int i=0;i<grid.N();++i) if(grid.mask[i]) maximum=std::max(maximum,std::fabs((double)grid.kappa_f[i]));
    return maximum;
}

struct FrontResult { double displacement=0.0, expected=0.0, favoured_fraction=0.0; };

static FrontResult flat_multiphase_front(int Nz) {
    const float M=1.0e-4f, dG=1000.f, dx=1.f;
    std::vector<float> a={0.f,-dG}, zero={0.f,0.f};
    auto th=std::make_shared<PrecipitationThermo>(M,0.2f,2,a,zero,zero,zero,900.f);
    const int Ny=20,Nx=96;
    GPKGrid grid(Nz,Ny,Nx,dx,th,0.f,2.f);
    for(int z=0;z<Nz;++z) for(int y=0;y<Ny;++y) for(int x=0;x<Nx;++x) {
        int i=grid.idx(z,y,x); grid.phi[i]=(x<Nx/2)?1:0; grid.T[i]=900.f;
    }
    const double initial=grid.phase_volume(1);
    for(int k=0;k<80;++k) grid.step(-1.f);
    const double area=2.0*Ny*Nz;
    FrontResult r;
    r.displacement=(grid.phase_volume(1)-initial)/area;
    r.expected=M*dG*grid.time/dx;
    r.favoured_fraction=grid.phase_volume(1)/grid.N();
    return r;
}

struct SymmetryResult { double imbalance=0.0, matrix_loss=0.0, invalid=0.0; };

static SymmetryResult three_phase_symmetry(int Nz) {
    const float M=1.0e-4f;
    std::vector<float> a={0.f,-800.f,-800.f}, zero={0.f,0.f,0.f};
    auto th=std::make_shared<PrecipitationThermo>(M,0.15f,3,a,zero,zero,zero,900.f);
    const int Ny=18,Nx=120;
    GPKGrid grid(Nz,Ny,Nx,1.f,th,0.f,2.f);
    for(int z=0;z<Nz;++z) for(int y=0;y<Ny;++y) for(int x=0;x<Nx;++x) {
        int label=0;
        if(x<20) label=1;
        else if(x>=100) label=2;
        int i=grid.idx(z,y,x); grid.phi[i]=label; grid.T[i]=900.f;
    }
    const double f0_initial=grid.phase_volume(0)/grid.N();
    for(int k=0;k<80;++k) grid.step(-1.f);
    const double f0=grid.phase_volume(0)/grid.N();
    const double f1=grid.phase_volume(1)/grid.N(), f2=grid.phase_volume(2)/grid.N();
    int invalid=0; for(int q:grid.phi) invalid += (q<0 || q>2);
    return {std::fabs(f1-f2),f0_initial-f0,(double)invalid/grid.N()};
}

int main() {
    mkdir_p(OUTDIR);
    std::vector<std::string> rows;
    std::printf("# GPK dimensional geometry and multiphase validation\n");

    std::printf("\n=== D1: Analytic 3-D geometry ===\n");
    // The manuscript's refined simulations use sigma_G=4 cells at this
    // resolution; validate the same measurement scale used in the figures.
    const auto sphere=sphere_test(72,18,4.f);
    const auto cylinder=cylinder_test(72,18,4.f);
    scalar("sphere mean summed curvature",sphere.mean,2.0/18.0,10.0);
    check("sphere directional coefficient of variation < 12%",sphere.cv<0.12,"tests voxel-grid isotropy, not only the mean");
    check("sphere 95th-percentile pointwise error < 25%",sphere.p95_rel<0.25,"guards against a smooth mean hiding rough local geometry");
    scalar("cylinder mean summed curvature",cylinder.mean,1.0/18.0,10.0);
    const double flat2=flat_curvature(1),flat3=flat_curvature(16);
    check("2-D planar interface has zero curvature",flat2<1e-6,"periodic two-front geometry");
    check("3-D planar interface has zero curvature",flat3<1e-6,"detects spurious out-of-plane curvature");

    std::printf("\n=== D2: Dimension-matched multiphase front ===\n");
    const auto front2=flat_multiphase_front(1),front3=flat_multiphase_front(12);
    scalar("2-D favoured-phase front displacement",front2.displacement,front2.expected,8.0);
    scalar("3-D favoured-phase front displacement",front3.displacement,front3.expected,8.0);
    scalar("3-D/2-D displacement ratio",front3.displacement/front2.displacement,1.0,3.0);
    check("favoured phase grows in both dimensions",front2.favoured_fraction>0.5 && front3.favoured_fraction>0.5,"known Gibbs-energy ranking");

    std::printf("\n=== D3: Three-phase symmetry and label integrity ===\n");
    const auto sym2=three_phase_symmetry(1),sym3=three_phase_symmetry(12);
    check("2-D equal-energy products remain symmetric",sym2.imbalance<1e-5,"mirror-symmetric initialization");
    check("3-D equal-energy products remain symmetric",sym3.imbalance<1e-5,"same test with out-of-plane neighbors");
    check("matrix is consumed in both dimensions",sym2.matrix_loss>0.02 && sym3.matrix_loss>0.02,"visual trend agrees with Gibbs ordering");
    check("no invalid phase labels are created",sym2.invalid==0.0 && sym3.invalid==0.0,"categorical topology invariant");

    rows.push_back("sphere_mean_kappa,"+std::to_string(sphere.mean));
    rows.push_back("sphere_theory_kappa,"+std::to_string(2.0/18.0));
    rows.push_back("sphere_directional_cv,"+std::to_string(sphere.cv));
    rows.push_back("sphere_p95_relative_error,"+std::to_string(sphere.p95_rel));
    rows.push_back("cylinder_mean_kappa,"+std::to_string(cylinder.mean));
    rows.push_back("flat_2d_max_abs_kappa,"+std::to_string(flat2));
    rows.push_back("flat_3d_max_abs_kappa,"+std::to_string(flat3));
    rows.push_back("front_2d_displacement_cells,"+std::to_string(front2.displacement));
    rows.push_back("front_3d_displacement_cells,"+std::to_string(front3.displacement));
    rows.push_back("front_3d_to_2d_ratio,"+std::to_string(front3.displacement/front2.displacement));
    rows.push_back("symmetry_2d_fraction_difference,"+std::to_string(sym2.imbalance));
    rows.push_back("symmetry_3d_fraction_difference,"+std::to_string(sym3.imbalance));
    write_metrics(rows);

    std::printf("\n=== Summary: %d passed, %d failed ===\n",passed,failed);
    return failed==0 ? 0 : 1;
}
