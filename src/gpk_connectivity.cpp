/* Digital-topology certificates for proposed label changes. */
#include "gpk_core.hpp"

#include <cmath>
#include <unordered_set>
#include <vector>

namespace gpk {

bool GPKGrid::topology_simple_move_counted(int cell, int32_t new_label,
                                           int old_label_count,
                                           int new_label_count,
                                           bool local_only) const {
    if (cell < 0 || cell >= N()) return false;
    const int32_t old_label = phi[cell];
    if (old_label == new_label) return false;

    int z0, y0, x0; decode(cell, z0, y0, x0);
    const int lz_count = (Nz > 1) ? 3 : 1;
    const int lz_centre = (Nz > 1) ? 1 : 0;
    const int centre = lz_centre * 9 + 4;

    auto topological_number = [&](int32_t label, bool equal,
                                  bool full_connectivity) {
        bool occupied[27] = {};
        bool seen[27] = {};
        for (int lz = 0; lz < lz_count; ++lz) {
            const int dz = (Nz > 1) ? lz - 1 : 0;
            int z = z0 + dz; if (z < 0) z += Nz; else if (z >= Nz) z -= Nz;
            for (int ly = 0; ly < 3; ++ly) {
                int y = y0 + ly - 1; if (y < 0) y += Ny; else if (y >= Ny) y -= Ny;
                for (int lx = 0; lx < 3; ++lx) {
                    int x = x0 + lx - 1; if (x < 0) x += Nx; else if (x >= Nx) x -= Nx;
                    const int local = lz * 9 + ly * 3 + lx;
                    const bool same = phi[idx(z,y,x)] == label;
                    occupied[local] = equal ? same : !same;
                }
            }
        }
        occupied[centre] = false;

        int component_count = 0;
        for (int seed = 0; seed < 27; ++seed) {
            const int sz = seed / 9;
            if (sz >= lz_count || !occupied[seed] || seen[seed]) continue;
            int queue[27]; int head = 0, tail = 0;
            queue[tail++] = seed; seen[seed] = true;
            bool touches_centre = false;
            while (head < tail) {
                const int current = queue[head++];
                const int cz = current / 9;
                const int rem = current % 9;
                const int cy = rem / 3;
                const int cx = rem % 3;
                const int manhattan = std::abs(cz-lz_centre)
                                      + std::abs(cy-1) + std::abs(cx-1);
                if (full_connectivity || manhattan == 1) touches_centre = true;
                for (int dz = -1; dz <= 1; ++dz)
                for (int dy = -1; dy <= 1; ++dy)
                for (int dxc = -1; dxc <= 1; ++dxc) {
                    if (dz == 0 && dy == 0 && dxc == 0) continue;
                    if (!full_connectivity &&
                        std::abs(dz)+std::abs(dy)+std::abs(dxc) != 1) continue;
                    const int nz = cz + dz, ny = cy + dy, nx = cx + dxc;
                    if (nz < 0 || nz >= lz_count || ny < 0 || ny >= 3 || nx < 0 || nx >= 3)
                        continue;
                    const int next = nz * 9 + ny * 3 + nx;
                    if (occupied[next] && !seen[next]) {
                        seen[next] = true;
                        queue[tail++] = next;
                    }
                }
            }
            if (touches_centre) ++component_count;
        }
        return component_count;
    };

    auto global_branches_connected = [&](int32_t label, bool exclude_centre) {
        int z, y, x; decode(cell, z, y, x);
        const int xm=(x==0?Nx-1:x-1), xp=(x==Nx-1?0:x+1);
        const int ym=(y==0?Ny-1:y-1), yp=(y==Ny-1?0:y+1);
        const int zm=(z==0?Nz-1:z-1), zp=(z==Nz-1?0:z+1);
        const int neighbours[6] = {idx(z,y,xm),idx(z,y,xp),idx(z,ym,x),
                                   idx(z,yp,x),idx(zm,y,x),idx(zp,y,x)};
        const int nface = Nz > 1 ? 6 : 4;
        std::vector<int> branches;
        for (int k=0; k<nface; ++k)
            if (phi[neighbours[k]] == label) branches.push_back(neighbours[k]);
        if (branches.size() <= 1) return true;
        std::unordered_set<int> visited;
        visited.reserve(256);
        std::unordered_set<int> targets(branches.begin()+1,branches.end());
        std::vector<int> stack;
        stack.reserve(256); stack.push_back(branches.front());
        visited.insert(branches.front());
        while (!stack.empty()) {
            const int current = stack.back(); stack.pop_back();
            int zz, yy, xx; decode(current, zz, yy, xx);
            const int xxm=(xx==0?Nx-1:xx-1), xxp=(xx==Nx-1?0:xx+1);
            const int yym=(yy==0?Ny-1:yy-1), yyp=(yy==Ny-1?0:yy+1);
            const int zzm=(zz==0?Nz-1:zz-1), zzp=(zz==Nz-1?0:zz+1);
            const int nexts[6] = {idx(zz,yy,xxm),idx(zz,yy,xxp),idx(zz,yym,xx),
                                  idx(zz,yyp,xx),idx(zzm,yy,xx),idx(zzp,yy,xx)};
            for (int k=0; k<nface; ++k) {
                const int next = nexts[k];
                if (visited.find(next)!=visited.end() || phi[next] != label ||
                    (exclude_centre && next == cell)) continue;
                visited.insert(next);
                if(targets.erase(next)>0 && targets.empty()) return true;
                stack.push_back(next);
            }
        }
        return targets.empty();
    };

    // Constant-cost connectivity proof for geometry-only remap proposals.
    // The 3^d simple-point test is deliberately conservative at an oblique
    // staircase: two arms of the same interface may reconnect just outside
    // that stencil.  A 5^d face-connected search proves such a reconnection
    // locally without the O(N) worst case of a full periodic flood fill.  A
    // failed bounded proof is deferred, so this relaxation cannot authorize a
    // split or merger whose connecting path lies outside the inspected box.
    auto bounded_branches_connected = [&](int32_t label, bool exclude_centre) {
        constexpr int radius=2, side=5, plane=25, capacity=125;
        const int zcount=Nz>1?side:1, zcentre=Nz>1?radius:0;
        const int local_centre=zcentre*plane+radius*side+radius;
        bool occupied[capacity]={};
        bool seen[capacity]={};
        for(int lz=0;lz<zcount;++lz) {
            const int dz=Nz>1?lz-radius:0;
            int z=z0+dz; while(z<0)z+=Nz; while(z>=Nz)z-=Nz;
            for(int ly=0;ly<side;++ly) {
                int y=y0+ly-radius; while(y<0)y+=Ny; while(y>=Ny)y-=Ny;
                for(int lx=0;lx<side;++lx) {
                    int x=x0+lx-radius; while(x<0)x+=Nx; while(x>=Nx)x-=Nx;
                    const int local=lz*plane+ly*side+lx;
                    occupied[local]=phi[idx(z,y,x)]==label;
                }
            }
        }
        if(exclude_centre) occupied[local_centre]=false;
        const int branch_local[6]={
            zcentre*plane+radius*side+(radius-1),
            zcentre*plane+radius*side+(radius+1),
            zcentre*plane+(radius-1)*side+radius,
            zcentre*plane+(radius+1)*side+radius,
            (zcentre-1)*plane+radius*side+radius,
            (zcentre+1)*plane+radius*side+radius};
        const int nface=Nz>1?6:4;
        int branches[6],nbranch=0;
        for(int k=0;k<nface;++k)
            if(occupied[branch_local[k]]) branches[nbranch++]=branch_local[k];
        if(nbranch<=1) return true;
        int queue[capacity],head=0,tail=0;
        queue[tail++]=branches[0]; seen[branches[0]]=true;
        while(head<tail) {
            const int current=queue[head++];
            const int cz=current/plane, rem=current%plane;
            const int cy=rem/side, cx=rem%side;
            const int dzv[6]={0,0,0,0,-1,1};
            const int dyv[6]={0,0,-1,1,0,0};
            const int dxv[6]={-1,1,0,0,0,0};
            for(int k=0;k<nface;++k) {
                const int zz=cz+dzv[k],yy=cy+dyv[k],xx=cx+dxv[k];
                if(zz<0||zz>=zcount||yy<0||yy>=side||xx<0||xx>=side) continue;
                const int next=zz*plane+yy*side+xx;
                if(occupied[next]&&!seen[next]) {seen[next]=true;queue[tail++]=next;}
            }
        }
        for(int k=1;k<nbranch;++k) if(!seen[branches[k]]) return false;
        return true;
    };

    // Material labels remain face connected. A multi-branch local event uses
    // the early-exit global certificate above; for an oblique stair its arms
    // reconnect after visiting only a few nearby cells, while a real bridge
    // or premature merger exhausts the relevant component and is deferred.
    const int old_object = topological_number(old_label, true, false);
    const bool old_extinction = old_label_count <= 1;
    if (old_object == 0 && !old_extinction) return false;
    if (old_object > 1 && !(local_only
        ? bounded_branches_connected(old_label,true)
        : global_branches_connected(old_label,true)))
        return false;

    const int new_object = topological_number(new_label, true, false);
    if (new_object == 0) return false;
    if (new_object > 1 && !(local_only
        ? bounded_branches_connected(new_label,false)
        : global_branches_connected(new_label,false)))
        return false;
    (void)new_label_count;
    return true;
}

bool GPKGrid::topology_simple_move(int cell, int32_t new_label) const {
    if (cell < 0 || cell >= N()) return false;
    const int32_t old_label = phi[cell];
    int old_count = 0, new_count = 0;
    for (const int32_t value : phi) {
        old_count += value == old_label;
        new_count += value == new_label;
    }
    return topology_simple_move_counted(cell, new_label, old_count, new_count);
}

} // namespace gpk
