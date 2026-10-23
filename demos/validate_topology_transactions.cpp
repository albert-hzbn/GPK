/* Model-independent regression tests for topology-aware GPK label changes. */
#include "gpk_core.hpp"

#include <cstdio>
#include <memory>

using namespace gpk;

namespace {

std::shared_ptr<ThermoModel> neutral_model() {
    return std::make_shared<ThermoModel>();
}

bool expect(const char* name, bool value, bool wanted) {
    const bool pass = value == wanted;
    std::printf("%-34s observed=%s expected=%s  %s\n", name,
                value ? "accept" : "defer", wanted ? "accept" : "defer",
                pass ? "PASS" : "FAIL");
    return pass;
}

} // namespace

int main() {
    bool ok = true;

    // A smooth boundary advance preserves both phases' connectivity.
    GPKGrid smooth(1, 9, 9, 1.f, neutral_model());
    for (int y=0; y<9; ++y)
        for (int x=0; x<9; ++x)
            smooth.phi[smooth.idx(0,y,x)] = (x <= 4) ? 1 : 0;
    ok &= expect("2D smooth boundary", smooth.topology_simple_move(
                 smooth.idx(0,4,4), 0), true);

    // Deleting the neck would split one material component.
    GPKGrid bridge2(1, 11, 11, 1.f, neutral_model());
    std::fill(bridge2.phi.begin(), bridge2.phi.end(), 0);
    for (int y=3; y<=7; ++y)
        for (int x=2; x<=4; ++x) bridge2.phi[bridge2.idx(0,y,x)] = 1;
    for (int y=3; y<=7; ++y)
        for (int x=6; x<=8; ++x) bridge2.phi[bridge2.idx(0,y,x)] = 1;
    bridge2.phi[bridge2.idx(0,5,5)] = 1;
    ok &= expect("2D component split", bridge2.topology_simple_move(
                 bridge2.idx(0,5,5), 0), false);

    // Adding the centre would merge two disconnected product components.
    GPKGrid merge2(1, 9, 9, 1.f, neutral_model());
    std::fill(merge2.phi.begin(), merge2.phi.end(), 0);
    merge2.phi[merge2.idx(0,4,3)] = 1;
    merge2.phi[merge2.idx(0,4,5)] = 1;
    ok &= expect("2D premature merger", merge2.topology_simple_move(
                 merge2.idx(0,4,4), 1), false);

    // A one-cell component is allowed to complete its physical extinction.
    GPKGrid extinct2(1, 9, 9, 1.f, neutral_model());
    std::fill(extinct2.phi.begin(), extinct2.phi.end(), 0);
    extinct2.phi[extinct2.idx(0,4,4)] = 1;
    ok &= expect("2D terminal extinction", extinct2.topology_simple_move(
                 extinct2.idx(0,4,4), 0), true);

    // Three-dimensional neck deletion is tested independently of any
    // thermodynamic, elastic or phase-selection module.
    GPKGrid bridge3(9, 9, 9, 1.f, neutral_model());
    std::fill(bridge3.phi.begin(), bridge3.phi.end(), 0);
    for (int z=3; z<=5; ++z)
        for (int y=3; y<=5; ++y)
            for (int x=1; x<=3; ++x) bridge3.phi[bridge3.idx(z,y,x)] = 2;
    for (int z=3; z<=5; ++z)
        for (int y=3; y<=5; ++y)
            for (int x=5; x<=7; ++x) bridge3.phi[bridge3.idx(z,y,x)] = 2;
    bridge3.phi[bridge3.idx(4,4,4)] = 2;
    ok &= expect("3D component split", bridge3.topology_simple_move(
                 bridge3.idx(4,4,4), 0), false);

    GPKGrid extinct3(7, 7, 7, 1.f, neutral_model());
    std::fill(extinct3.phi.begin(), extinct3.phi.end(), 0);
    extinct3.phi[extinct3.idx(3,3,3)] = 3;
    ok &= expect("3D terminal extinction", extinct3.topology_simple_move(
                 extinct3.idx(3,3,3), 0), true);

    std::printf("Topology transaction suite: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
