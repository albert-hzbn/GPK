/* Kinetic crossings, balanced remapping, and unresolved-grain extinction. */
#include "gpk_core.hpp"

#include <omp.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace gpk {

namespace {
static inline uint64_t mix64(uint64_t x) {
    x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27; x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return x;
}

static inline float u01_from_u64(uint64_t x) {
    const uint64_t m = (x >> 11) | 1ULL;
    return float((m & ((1ULL << 53) - 1)) * (1.0 / double(1ULL << 53)));
}

static inline float normal_from_cell_step(int i, uint64_t step_id, uint64_t salt = 0ULL) {
    uint64_t a = mix64((uint64_t(i) + 0x9e3779b97f4a7c15ULL) ^ (step_id << 1) ^ salt);
    uint64_t b = mix64((uint64_t(i) + 0xd2b74407b1ce6e93ULL) ^ (step_id << 3) ^ (salt + 17ULL));
    float u1 = std::max(1e-12f, u01_from_u64(a));
    float u2 = u01_from_u64(b);
    return std::sqrt(-2.f * std::log(u1)) * std::cos(2.f * float(M_PI) * u2);
}
} // namespace

void GPKGrid::step_balanced_gaussian_remap() {
    gaussian_remap_exchanges_last = 0;
    if (topology_accepted_last == 0) return;
    step_interface_mask();

    std::fill(eps_star_.begin(), eps_star_.end(),
              -std::numeric_limits<float>::infinity()); // resident score
    std::fill(eps_smooth_.begin(), eps_smooth_.end(),
              -std::numeric_limits<float>::infinity()); // best candidate score
    std::copy(phi.begin(), phi.end(), phi_comp.begin());
    // Compact direct projection: cost scales with interface cells rather than
    // with (number of labels x domain cells).
    const float remap_sigma=remap_gauss_.sigma_cells;
    const int radius=std::max(1,static_cast<int>(std::ceil(3.f*remap_sigma)));
    const float inv_two_sigma2=0.5f/(remap_sigma*remap_sigma);
    #pragma omp parallel for schedule(static)
    for (int a=0;a<static_cast<int>(active_cells_.size());++a) {
        const int i=active_cells_[a];
        int z0,y0,x0; decode(i,z0,y0,x0);
        const int xm=(x0==0?Nx-1:x0-1), xp=(x0==Nx-1?0:x0+1);
        const int ym=(y0==0?Ny-1:y0-1), yp=(y0==Ny-1?0:y0+1);
        const int zm=(z0==0?Nz-1:z0-1), zp=(z0==Nz-1?0:z0+1);
        const int face[6]={idx(z0,y0,xm),idx(z0,y0,xp),idx(z0,ym,x0),
                           idx(z0,yp,x0),idx(zm,y0,x0),idx(zp,y0,x0)};
        const int nface=Nz>1?6:4;
        int32_t candidate[6]; int ncandidate=0;
        for(int k=0;k<nface;++k) {
            const int32_t value=phi[face[k]];
            if(value==phi[i]) continue;
            bool known=false; for(int q=0;q<ncandidate;++q) known|=candidate[q]==value;
            if(!known) candidate[ncandidate++]=value;
        }
        double resident=0.0; double best=-1.0; int32_t best_label=phi[i];
        double support[6]={0.,0.,0.,0.,0.,0.};
        const int dz0=Nz>1?-radius:0, dz1=Nz>1?radius:0;
        for(int dz=dz0;dz<=dz1;++dz)
        for(int dy=-radius;dy<=radius;++dy)
        for(int dxs=-radius;dxs<=radius;++dxs) {
            int z=(z0+dz)%Nz;if(z<0)z+=Nz;
            int y=(y0+dy)%Ny;if(y<0)y+=Ny;
            int x=(x0+dxs)%Nx;if(x<0)x+=Nx;
            const int32_t value=phi[idx(z,y,x)];
            const float r2=static_cast<float>(dxs*dxs+dy*dy+dz*dz);
            const double weight=std::exp(-r2*inv_two_sigma2);
            if(value==phi[i]) resident+=weight;
            for(int q=0;q<ncandidate;++q) if(value==candidate[q]) support[q]+=weight;
        }
        eps_star_[i]=static_cast<float>(resident);
        for(int q=0;q<ncandidate;++q) if(support[q]>best) {
            best=support[q]; best_label=candidate[q];
        }
        eps_smooth_[i]=static_cast<float>(best);
        phi_comp[i]=best_label;
    }

    struct Proposal { float gain; int cell; };
    struct PairLists { std::vector<Proposal> low_to_high, high_to_low; };
    std::map<std::pair<int32_t,int32_t>, PairLists> grouped;
    for (const int i : active_cells_) {
        const int32_t p = phi[i], q = phi_comp[i];
        const float gain = eps_smooth_[i] - eps_star_[i];
        if (p == q || !(gain > 1e-7f)) continue;
        const auto key = std::make_pair(std::min(p,q), std::max(p,q));
        Proposal proposal{gain, i};
        if (p == key.first) grouped[key].low_to_high.push_back(proposal);
        else                grouped[key].high_to_low.push_back(proposal);
    }

    auto rank = [](const Proposal& a, const Proposal& b) {
        if (a.gain != b.gain) return a.gain > b.gain;
        return a.cell < b.cell;
    };
    std::unordered_map<int32_t,int> label_count;
    label_count.reserve(64);
    for (const int32_t value : phi) ++label_count[value];
    const bool sweep_rho = thermo_->dislocation_enabled;
    auto target_orientation = [&](int cell, int32_t target) {
        int z,y,x; decode(cell,z,y,x);
        const int xm=(x==0?Nx-1:x-1), xp=(x==Nx-1?0:x+1);
        const int ym=(y==0?Ny-1:y-1), yp=(y==Ny-1?0:y+1);
        const int zm=(z==0?Nz-1:z-1), zp=(z==Nz-1?0:z+1);
        const int nei[6]={idx(z,y,xm),idx(z,y,xp),idx(z,ym,x),
                          idx(z,yp,x),idx(zm,y,x),idx(zp,y,x)};
        const int nface=Nz>1?6:4;
        for(int k=0;k<nface;++k) if(phi[nei[k]]==target) return g[nei[k]];
        return g[cell];
    };
    // Count the discrete interfacial faces incident to one proposed cell.
    // If the paired cell is a face neighbour, use its proposed label so that
    // the exchange is assessed as one atomic transaction.  BGR is a
    // geometry-only, exactly volume-balanced operation, so accepting only a
    // strict decrease makes its anti-alias action an explicit Lyapunov step
    // for the categorical interface measure.
    auto mismatch_count = [&](int cell, int32_t proposed,
                              int paired_cell, int32_t paired_proposed) {
        int z,y,x; decode(cell,z,y,x);
        const int xm=(x==0?Nx-1:x-1), xp=(x==Nx-1?0:x+1);
        const int ym=(y==0?Ny-1:y-1), yp=(y==Ny-1?0:y+1);
        const int zm=(z==0?Nz-1:z-1), zp=(z==Nz-1?0:z+1);
        const int nei[6]={idx(z,y,xm),idx(z,y,xp),idx(z,ym,x),
                          idx(z,yp,x),idx(zm,y,x),idx(zp,y,x)};
        const int nface=Nz>1?6:4;
        int mismatch=0;
        for(int k=0;k<nface;++k) {
            const int32_t q=(nei[k]==paired_cell)?paired_proposed:phi[nei[k]];
            mismatch += q!=proposed;
        }
        return mismatch;
    };
    for (auto& item : grouped) {
        auto& forward = item.second.low_to_high;
        auto& reverse = item.second.high_to_low;
        std::sort(forward.begin(), forward.end(), rank);
        std::sort(reverse.begin(), reverse.end(), rank);
        const size_t count = std::min(forward.size(), reverse.size());
        for (size_t k=0; k<count; ++k) {
            const int a=forward[k].cell, b=reverse[k].cell;
            const int32_t low=item.first.first, high=item.first.second;
            if(phi[a]!=low || phi[b]!=high) continue;
            const float ga=target_orientation(a,high);
            const float gb=target_orientation(b,low);
            const int before=mismatch_count(a,low,b,high)
                            +mismatch_count(b,high,a,low);
            const int after=mismatch_count(a,high,b,low)
                           +mismatch_count(b,low,a,high);
            if(after>=before) continue;
            if(!topology_simple_move_counted(a,high,label_count[low],
                                             label_count[high],true)) continue;
            phi[a]=high;
            if(!topology_simple_move_counted(b,low,label_count[high]+1,
                                             label_count[low]-1,true)) {
                phi[a]=low;
                continue;
            }
            phi[b]=low; g[a]=ga; g[b]=gb; alpha[a]=alpha[b]=1.f;
            if(sweep_rho) rho[a]=rho[b]=thermo_->rho_rex;
            gaussian_remap_exchanges_last += 2;
        }
    }
    step_interface_mask();
}

void GPKGrid::step_complete_unresolved_grains() {
    if (!thermo_->grain_network) return;
    // Extinction is a rare, label-scale event. Checking every eighth step
    // retains sub-Gaussian timing accuracy while avoiding an O(N) inventory
    // pass on every curvature update.
    if ((step_id_ & 7ULL) != 0ULL) return;

    std::unordered_map<int32_t,int> label_count;
    label_count.reserve(128);
    for (const int32_t value : phi) ++label_count[value];

    // A region smaller than a d-dimensional ball of radius sigma_G cannot
    // supply a resolved interior to the Gaussian geometry operator.  Keeping
    // such a label alive solely to preserve its component count creates thin
    // tendrils and voxel islands.  Completing its extinction is a legitimate
    // topology event and is distinct from deleting a resolved grain.
    const double sigma = std::max(gauss_.sigma_cells, 1.f);
    const double support_volume = (Nz > 1)
        ? (4.0 * M_PI * sigma*sigma*sigma / 3.0)
        : (M_PI * sigma*sigma);
    const int min_resolved_cells = std::max(2, static_cast<int>(std::ceil(support_volume)));
    std::unordered_set<int32_t> unresolved;
    for (const auto& item : label_count)
        if (item.second < min_resolved_cells) unresolved.insert(item.first);
    if (unresolved.empty() || unresolved.size() == label_count.size()) return;

    const bool sweep_rho = thermo_->dislocation_enabled;
    // Erode every unresolved region from its boundary inward.  Candidates
    // must be resolved labels, so two disappearing remnants cannot sustain
    // each other.  Each layer is selected from one immutable state and then
    // committed together, making the result independent of OpenMP ordering.
    while (true) {
        std::vector<int> cells;
        std::vector<int32_t> targets;
        std::vector<float> target_g;
        for (int i=0; i<N(); ++i) {
            if (unresolved.find(phi[i]) == unresolved.end()) continue;
            int z,y,x; decode(i,z,y,x);
            const int xm=(x==0?Nx-1:x-1), xp=(x==Nx-1?0:x+1);
            const int ym=(y==0?Ny-1:y-1), yp=(y==Ny-1?0:y+1);
            const int zm=(z==0?Nz-1:z-1), zp=(z==Nz-1?0:z+1);
            const int face[6]={idx(z,y,xm),idx(z,y,xp),idx(z,ym,x),
                               idx(z,yp,x),idx(zm,y,x),idx(zp,y,x)};
            const int nface=Nz>1?6:4;
            int best=-1, best_support=-1;
            for (int k=0;k<nface;++k) {
                const int32_t q=phi[face[k]];
                if (unresolved.find(q) != unresolved.end()) continue;
                int support=0;
                for (int dz=(Nz>1?-1:0);dz<=(Nz>1?1:0);++dz)
                for (int dy=-1;dy<=1;++dy)
                for (int dxs=-1;dxs<=1;++dxs) {
                    int zz=(z+dz+Nz)%Nz, yy=(y+dy+Ny)%Ny, xx=(x+dxs+Nx)%Nx;
                    support += phi[idx(zz,yy,xx)] == q;
                }
                if (support>best_support) { best=k; best_support=support; }
            }
            if (best>=0) {
                cells.push_back(i); targets.push_back(phi[face[best]]);
                target_g.push_back(g[face[best]]);
            }
        }
        if (cells.empty()) break;
        for (size_t k=0;k<cells.size();++k) {
            const int i=cells[k]; phi[i]=targets[k]; g[i]=target_g[k]; alpha[i]=1.f;
            if (sweep_rho) rho[i]=thermo_->rho_rex;
        }
    }
    step_interface_mask();
}

void GPKGrid::step_alpha(float dt) {
    const float idt = dt / dx;
    const float cell_vol = dx * dx * dx;
    const float inv_dt = (dt > 0.f) ? (1.f / dt) : 0.f;
    const bool do_noise = thermo_->stochastic && dt > 0.f;
    const float noise_amp = do_noise
        ? thermo_->noise_scale
          * std::sqrt(2.f * thermo_->M0 * thermo_->kB * inv_dt
                      / std::max(cell_vol, 1e-30f))
        : 0.f;

    alpha_new_ = alpha;
    #pragma omp parallel for schedule(static)
    for (int a = 0; a < static_cast<int>(active_cells_.size()); ++a) {
        const int i = active_cells_[a];
        float vn_eff = vn_f[i];
        if (do_noise)
            vn_eff += noise_amp * std::sqrt(std::max(T[i], 1.f))
                      * normal_from_cell_step(i, step_id_, 0xabcddcbaULL);
        alpha_new_[i] = std::min(alpha[i] + vn_eff * idt, 1.f);
    }

    std::unordered_map<int32_t,int> label_count;
    label_count.reserve(64);
    for (const int32_t value : phi) ++label_count[value];

    topology_accepted_last = 0;
    topology_deferred_last = 0;
    const bool sweep_rho = thermo_->dislocation_enabled;
    const float rho_rex = thermo_->rho_rex;
    // Grain-network labels are persistent region identities and therefore
    // require component-preserving transactions. Phase-selection labels are
    // phase classes: disconnected particles may physically coalesce or split,
    // so a kinetic crossing must not be rejected merely for changing the
    // component count of the class. The geometric remap below remains strict.
    const bool tracks_region_identity = thermo_->grain_network;

    // Deterministic serial reference semantics.  Every certificate is tested
    // against the state produced by preceding transactions.  Implementations
    // may parallelise independent vertices of the radius-two conflict graph.
    for (const int i : active_cells_) {
        if (!(alpha_new_[i] <= 1e-6f && phi_comp[i] != phi[i])) {
            alpha[i] = alpha_new_[i];
            continue;
        }
        const int32_t old_label = phi[i];
        const int32_t new_label = phi_comp[i];
        const int old_count = label_count[old_label];
        const int new_count = label_count[new_label];
        if (!tracks_region_identity ||
            topology_simple_move_counted(i, new_label, old_count, new_count)) {
            phi[i] = new_label;
            g[i] = g_comp[i];
            alpha[i] = 1.f;
            --label_count[old_label];
            ++label_count[new_label];
            ++topology_accepted_last;
            if (sweep_rho) rho[i] = rho_rex;
        } else {
            alpha[i] = std::max(alpha_new_[i], 1e-6f);
            ++topology_deferred_last;
        }
    }
    step_balanced_gaussian_remap();
    // Three-dimensional interfaces have two independent tangential
    // directions, so a single balanced exchange sweep can leave orthogonal
    // one-cell ledges.  A second sweep removes those residual ledges while
    // preserving every label volume during remapping.  Both sweeps retain the
    // strict interface-area decrease and topology certificate enforced above.
    if (Nz > 1 && gaussian_remap_exchanges_last > 0) {
        const std::uint64_t first_sweep = gaussian_remap_exchanges_last;
        step_balanced_gaussian_remap();
        gaussian_remap_exchanges_last += first_sweep;
    }
    step_complete_unresolved_grains();
}

} // namespace gpk
