# GPK core architecture

This document describes code ownership and the runtime path of the C++ solver.
The public API is intentionally kept in `include/gpk_core.hpp`; the source files
are organized around independent reasons to change.

## Runtime flow

`GPKGrid::step()` coordinates one timestep in this order:

1. `step_interface_mask()` builds the sorted compact interface set.
2. `step_geometry()` reconstructs Gaussian-projected normals and curvature.
3. `step_elastic()` updates the optional mechanical energy field.
4. `step_velocities()` evaluates mobility, bulk driving force and capillarity.
5. `step_velocity_projection()` conditions velocity on each label pair.
6. `max_dt()` selects a stable interface/diffusion timestep.
7. `step_alpha()` integrates sub-cell motion and certifies label crossings.
8. `step_diffusion()` and `step_recovery()` update transported state fields.

The first five operations read the persistent state and populate temporary
geometry/kinetic fields. Only the topology stage changes label ownership.

## Source ownership

| File | Owns | Should not own |
|---|---|---|
| `gpk_thermo.cpp` | default `ThermoModel` energy and mobility laws | grid traversal |
| `gpk_gaussian.cpp` | reusable periodic separable filter | GPK state changes |
| `gpk_grid.cpp` | construction and workspace allocation | timestep policy |
| `gpk_geometry.cpp` | active interface, Gaussian derivatives, normal and curvature | material laws |
| `gpk_physics.cpp` | elastic proxy, constitutive velocity and pair projection | topology decisions |
| `gpk_connectivity.cpp` | local/global connectivity certificates | constitutive calibration or state mutation |
| `gpk_transitions.cpp` | crossings, balanced remap and unresolved-grain extinction | material-specific repair rules |
| `gpk_transport.cpp` | diffusion and dislocation recovery | interface geometry |
| `gpk_simulation.cpp` | step/run orchestration, stable timestep and diagnostics | process-specific equations |

## Core invariants

- `phi` is the persistent categorical grain/phase identity.
- `alpha` stores the resident interface position and is reset to one after an
  accepted ownership change.
- `active_cells_` is sorted by flat cell index. This makes serial topology
  transactions deterministic and independent of OpenMP scheduling.
- `phi_comp` and `g_comp` describe the same candidate owner; an accepted move
  updates both label and orientation.
- Geometry and velocity fields are temporary. They must be regenerated after
  label changes and must never be interpreted as persistent microstructure.
- Periodic indexing is used consistently by geometry, topology and transport.
- Grain-network labels represent connected identities and use strict topology
  certification. Phase-selection labels represent classes and may form more
  than one disconnected particle.

## Adding a constitutive model

Prefer deriving from `ThermoModel` (or using a model in `gpk_thermo.hpp`) and
override only the required callbacks:

- `bulk_driving_force()` for chemical or stored-energy differences;
- `interface_composition()` for matrix-side/interface sampling;
- `select_product_phase()` for multiphase competition;
- `interfacial_energy_pair_comp()` for orientation-, pair- or composition-
  dependent energy;
- `mobility()` for temperature, misorientation or orientation dependence.

Do not add process names or material-specific repair rules to geometry or
topology. New physics must reach the shared velocity law through these
constitutive callbacks.

## Validation expectations

After changing the core, build in Release mode and run the registered CTest
suite. At minimum, changes to geometry or topology must pass
`theory_foundations`, `dimensional_multiphase`, `topology_transactions`,
`interface_local_geometry`, `extended_base_models`, `real_materials`, and
`gpk_phase_field`. A refactor is complete only when these executables reproduce
their existing deterministic gates.
