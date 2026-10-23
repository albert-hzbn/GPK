#include "gpk_core.hpp"
#include <omp.h>
#include <cmath>
#include <cstdio>
#include <memory>
#include <vector>

using namespace gpk;

struct ConstantMobility final : ThermoModel {
    ConstantMobility() { sigma0 = 0.5f; grain_network = true; }
    float mobility(float, float, float, float, float, float) const override {
        return 1.0f;
    }
};

static void initialise_periodic_labels(GPKGrid& grid, int grains) {
    std::vector<int> sx(grains), sy(grains);
    for (int q=0; q<grains; ++q) {
        sx[q]=(37*q+11)%grid.Nx;
        sy[q]=(53*q+7)%grid.Ny;
    }
    for (int y=0; y<grid.Ny; ++y)
    for (int x=0; x<grid.Nx; ++x) {
        int best=0, best_d=grid.Nx*grid.Nx+grid.Ny*grid.Ny;
        for (int q=0; q<grains; ++q) {
            int dx=std::abs(x-sx[q]); dx=std::min(dx,grid.Nx-dx);
            int dy=std::abs(y-sy[q]); dy=std::min(dy,grid.Ny-dy);
            const int d=dx*dx+dy*dy;
            if(d<best_d) { best_d=d; best=q; }
        }
        const int i=grid.idx(0,y,x);
        grid.phi[i]=best;
        grid.g[i]=0.071f*best;
    }
}

int main() {
    int failures=0;
    auto check=[&](bool value,const char* name) {
        std::printf("%-62s %s\n",name,value?"PASS":"FAIL");
        failures += !value;
    };

    auto model=std::make_shared<ConstantMobility>();
    GPKGrid memory_grid(1,256,256,1.f,model,0.f,2.f);
    const double bytes_per_cell=double(memory_grid.allocated_array_bytes())/memory_grid.N();
    std::printf("2-D allocated array storage: %.2f bytes/cell\n",bytes_per_cell);
    check(bytes_per_cell<110.0,"interface-local storage remains below 110 bytes/cell");

    GPKGrid serial(1,96,96,1.f,model,0.f,2.f);
    GPKGrid parallel(1,96,96,1.f,model,0.f,2.f);
    initialise_periodic_labels(serial,32);
    initialise_periodic_labels(parallel,32);
    omp_set_num_threads(1);
    for(int s=0;s<8;++s) serial.step();
    omp_set_num_threads(std::min(4,omp_get_num_procs()));
    for(int s=0;s<8;++s) parallel.step();
    check(serial.phi==parallel.phi,"categorical evolution is thread-count deterministic");
    float max_alpha=0.f,max_kappa=0.f;
    for(int i=0;i<serial.N();++i) {
        max_alpha=std::max(max_alpha,std::fabs(serial.alpha[i]-parallel.alpha[i]));
        max_kappa=std::max(max_kappa,std::fabs(serial.kappa_f[i]-parallel.kappa_f[i]));
    }
    check(max_alpha<1e-6f,"sub-cell coordinates are thread-count deterministic");
    check(max_kappa<1e-6f,"fused Gaussian curvature is thread-count deterministic");

    auto sphere_model=std::make_shared<ConstantMobility>();
    sphere_model->sigma0=0.f;
    GPKGrid sphere(40,40,40,1.f,sphere_model,0.f,2.f);
    const float centre=19.5f, radius=10.f;
    for(int z=0;z<sphere.Nz;++z)
    for(int y=0;y<sphere.Ny;++y)
    for(int x=0;x<sphere.Nx;++x) {
        const float dz=z-centre,dy=y-centre,dx=x-centre;
        sphere.phi[sphere.idx(z,y,x)]=(dx*dx+dy*dy+dz*dz<=radius*radius)?1:0;
    }
    sphere.step(1e-3f);
    double sum=0.0; int count=0;
    for(int i=0;i<sphere.N();++i)
        if(sphere.mask[i]&&sphere.phi[i]==1&&std::isfinite(sphere.kappa_f[i])) {
            sum+=sphere.kappa_f[i]; ++count;
        }
    const double mean=count?sum/count:0.0;
    std::printf("3-D sphere mean curvature: %.6f (target %.6f)\n",mean,2.0/radius);
    check(count>0&&std::fabs(mean-2.0/radius)/(2.0/radius)<0.20,
          "3-D fused curvature agrees with the spherical target within 20%");

    // Broad 3-D kernels use the adaptive shared separable workspace rather
    // than a cubic direct tensor stencil.  This is the production-width path.
    GPKGrid broad(56,56,56,1.f,sphere_model,0.f,5.f);
    const float broad_centre=27.5f, broad_radius=14.f;
    for(int z=0;z<broad.Nz;++z)
    for(int y=0;y<broad.Ny;++y)
    for(int x=0;x<broad.Nx;++x) {
        const float dz=z-broad_centre,dy=y-broad_centre,dx=x-broad_centre;
        broad.phi[broad.idx(z,y,x)]=(dx*dx+dy*dy+dz*dz<=broad_radius*broad_radius)?1:0;
    }
    broad.step(1e-3f);
    double broad_sum=0.0; int broad_count=0;
    for(int i=0;i<broad.N();++i)
        if(broad.mask[i]&&broad.phi[i]==1&&std::isfinite(broad.kappa_f[i])) {
            broad_sum+=broad.kappa_f[i]; ++broad_count;
        }
    const double broad_mean=broad_count?broad_sum/broad_count:0.0;
    const double broad_bytes=double(broad.allocated_array_bytes())/broad.N();
    std::printf("broad-kernel 3-D curvature: %.6f; storage %.2f bytes/cell\n",
                broad_mean,broad_bytes);
    check(broad_count>0&&std::fabs(broad_mean-2.0/broad_radius)/(2.0/broad_radius)<0.25,
          "broad 3-D separable path agrees with the spherical target within 25%");
    check(broad_bytes<115.0,"broad 3-D workspace remains thread-count independent");

    std::printf("Summary: %d failure(s)\n",failures);
    return failures?1:0;
}
