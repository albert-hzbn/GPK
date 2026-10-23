# GPK: Gaussian-projected kinetics

[![build-and-test](https://github.com/albert-hzbn/GPK/actions/workflows/ci.yml/badge.svg)](https://github.com/albert-hzbn/GPK/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

GPK is a sharp-interface framework for simulating microstructure evolution in two and three dimensions. Every cell of a regular grid stores an integer grain or phase label. Near a boundary, a temporary Gaussian projection of the label map provides the interface normal and curvature; it is then discarded, so no diffuse interface is ever stored. Capillarity, stored energy, chemical driving forces, solute transport and elasticity move the boundaries through one topology-preserving update. Because the Gaussian measurement length is fixed in physical units, refining the mesh keeps a consistent capillary scale.

The same solver models:

* isotropic and anisotropic grain growth
* recrystallisation driven by stored energy
* multiphase precipitation and Ostwald ripening
* dendritic solidification, with or without elastic coupling
* dislocation stored-energy coupling and precipitate break-up
* magnetic-material extensions (magnetic annealing, magnetoelastic coupling, Nd–Fe–B grain-boundary wetting)

The method and its verification are described in the accompanying article (see [Citing GPK](#citing-gpk)).

## Contents

1. [Quick start](#quick-start)
2. [Requirements and build](#requirements-and-build)
3. [Examples](#examples)
4. [Input and output](#input-and-output)
5. [Tests and verification](#tests-and-verification)
6. [Reproducing the manuscript](#reproducing-the-manuscript)
7. [Repository layout](#repository-layout)
8. [Citing GPK](#citing-gpk)
9. [Licence and contact](#licence-and-contact)

## Quick start

```bash
git clone https://github.com/albert-hzbn/GPK.git
cd GPK
cmake -B build
cmake --build build --parallel
cd build
./gpk inputs/solidification.ini
```

This 2-D dendrite takes about a second. It prints the solid fraction and solute maximum over time and writes VTK snapshots to `build/output/solidification/`, which open directly in ParaView. Set `format = csv` in the `[output]` section of the deck for Python post-processing (see [Input and output](#input-and-output)). Next, try `./gpk inputs/grain_growth.ini` (3-D grain growth, about 4 minutes) or any deck in [Examples](#examples).

## Requirements and build

| Tool | Version |
|---|---|
| C++ compiler | C++17, for example GCC ≥ 9 or Clang ≥ 11 (tested with GCC 15.2 on Windows/MSYS2 and with GCC on Ubuntu in CI) |
| CMake | ≥ 3.16 |
| OpenMP | included with GCC |
| Python | ≥ 3.10, only for the Python tests, post-processing and figure scripts |

**Linux** (macOS should work with GCC or LLVM Clang plus OpenMP, but has not been tested)

```bash
cmake -B build                      # Release build by default
cmake --build build --parallel
```

**Windows.** Install [MSYS2](https://www.msys2.org/), add the UCRT64 toolchain (`pacman -S mingw-w64-ucrt-x86_64-gcc mingw-w64-ucrt-x86_64-cmake`), then in a shell where `C:\msys64\ucrt64\bin` is on the `PATH`:

```bash
cmake -B build -G "MinGW Makefiles"
cmake --build build --parallel
```

**Options.** `-DGPK_NATIVE_ARCH=ON` optimises for the build machine (`-march=native`). `-DCMAKE_BUILD_TYPE=Debug -DGPK_ENABLE_SANITIZERS=ON` builds with AddressSanitizer and UndefinedBehaviorSanitizer.

The build places the executables in `build/` (with `.exe` on Windows) and copies `inputs/` to `build/inputs/`:

| Executable | Purpose |
|---|---|
| `gpk` | Runs any model; the model is chosen in the input file |
| `validate_*` | Verification suites run by `ctest` |
| `benchmark`, `bench_detailed`, `bench_gpk_vs_mbo` | Performance and scaling reports |
| `micromagnetics_hysteresis`, `micromagnetics_materials_study` | Sub-grid micromagnetics studies |

The Python tools need `python -m pip install -r requirements.txt`.

## Examples

Every model has an example input deck in `inputs/`. Run them from the build folder:

```bash
cd build
./gpk inputs/solidification.ini
```

| Input deck | Model | What it shows | Run time* |
|---|---|---|---|
| `solidification.ini` | `solidification` | 2-D dendrite growing into an undercooled Al–Cu-like melt, with solute rejection | 1 s |
| `solidification_3d.ini` | `solidification` | The same dendrite on a 64³ grid | 12 s |
| `precipitation.ini` | `precipitation` | Multiphase precipitation: local Gibbs-energy selection among competing phases | 1 s |
| `recrystallisation.ini` | `recrystallisation` | 3-D recrystallisation driven by stored energy, with Avrami (JMAK) kinetics | 10 s |
| `grain_growth.ini` | `grain_growth` | 3-D isotropic grain growth compared with the parabolic coarsening law | 4 min |
| `ostwald_ripening.ini` | `ostwald_ripening` | 2-D Ostwald ripening: small particles dissolve, large ones grow | > 15 min |
| `wulff_shape.ini` | `wulff_shape` | A grain shrinking with tensorial interfacial-energy anisotropy | < 1 s |
| `dislocation.ini` | `dislocation` | Dislocation stored-energy coupling and recovery, checked against the analytical recovery law | 1 s |
| `precipitate_breakup.ini` | `precipitate_breakup` | Sweep of the critical spacing-to-thickness ratio at which a lamellar precipitate breaks up (Ti-6Al-V, 973 K) | > 15 min |
| `precipitate_breakup_refine.ini` | `precipitate_breakup` | Finer sweep around the critical ratio found by the previous deck | > 15 min |
| `magnetic_annealing.ini` | `magnetic_annealing` | Grain growth with an additional magnetic driving force | 2 s |
| `magnetoelastic.ini` | `magnetoelastic` | Magnetostrictive elastic energy of three grains, checked against the cos² θ law | < 1 s |
| `ndfeb_wetting.ini` | `ndfeb_wetting` | Wetting of an Nd–Fe–B grain boundary by the Nd-rich phase as a function of Cu content | 6 s |

\* Wall time with 24 OpenMP threads on a Windows workstation (GCC 15.2, Release build). Expect longer times on fewer cores; set the thread count with `OMP_NUM_THREADS`. "> 15 min" means the run had not finished after 15 minutes.

The subfolders of `inputs/` hold variants of the same models (for example `inputs/grain_growth/anisotropic_3d.ini` or `inputs/solidification/dendrite_3d.ini`). To run in 3-D, set `Nz` larger than 1 in the `[grid]` section. To write your own case, copy a deck and edit it; the keys are explained in the [user guide](docs/USER_GUIDE.md#5-parameter-reference).

`python/` contains a small pure-Python 2-D reference implementation with three demos (`demo_grain_growth.py`, `demo_solidification.py`, `demo_anisotropy.py`).

## Input and output

An input deck is an INI file:

```ini
[run]
model = grain_growth

[grid]
Nz    = 32          # depth (set to 1 for 2-D)
Ny    = 64
Nx    = 64
dx    = 1.0e-6      # cell size [m]

[time]
t_end   = 0.139     # simulation end time [s]
n_snaps = 10        # number of output snapshots

[thermo]
M0     = 1.0e-8     # boundary mobility [m^3/(J*s)]
sigma0 = 0.5        # interfacial energy [J/m^2]
eps4   = 0.0        # cubic anisotropy strength
D      = 0.0        # no solute diffusion

[output]
format  = csv       # vtk | csv | ppm | all
dir     = output/grain_growth

[grain_growth]
n_grains = 40
seed     = 42
```

This is `inputs/grain_growth.ini`, shortened. Output paths are relative to the folder you run `gpk` from.

Snapshots can be written as CSV (one row per cell: position, label, sub-cell fraction, composition, temperature, orientation), legacy VTK for ParaView or VisIt, or PPM images of the mid-plane. `scripts/post_process.py` measures grain size, shape, solute distribution and Avrami exponents from CSV output.

The [user guide](docs/USER_GUIDE.md) gives the full parameter reference, the output formats, a step-by-step tutorial and instructions for adding a new model. [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) describes how the solver is organised.

## Tests and verification

```bash
ctest --test-dir build --output-on-failure
```

| Test | What it checks |
|---|---|
| `theory_foundations` | Curvature accuracy, circle and sphere shrinkage, grain-growth laws and other results of curvature flow |
| `extended_base_models` | Flat and curved bicrystals, curvature crossover, grooving angles and precipitate shrinkage |
| `real_materials` | One case per physics model with literature material parameters, compared with theory or experiment |
| `dimensional_multiphase` | Geometry in 2-D and 3-D and multiphase boundary motion |
| `topology_transactions` | The topology-preserving label update (TCKT) and remap |
| `interface_local_geometry` | Local normals and curvature near boundaries and junctions |
| `gpk_phase_field` | Grain-by-grain comparison with a multi-order-parameter phase-field model, and cost at matched accuracy |
| `magnetic_extended`, `micromagnetics_v2` | Magnetic extensions and the sub-grid micromagnetics module |
| `implementation_regression` | Regression tests of the Python reference implementation (`test_implementation.py`) |
| `literature_regression` | Self-contained comparisons with published results (`scripts/validate_literature.py`) |

On the development machine (Windows, GCC 15.2), 10 of the 11 tests pass; the full suite takes about 11 minutes with four tests in parallel, most of it in `gpk_phase_field` and `theory_foundations`.

**Known failure.** `magnetic_extended` fails 2 of its 18 checks. In test M2, magnetic grains starting 1.16 and 1.94 times above the critical radius should grow but stay at about their initial size (the separate growth-rate check at 1.94 times the critical radius passes, with a 25% deviation from theory). The magnetic extensions are not used in the manuscript. The CI workflow skips this test until it is resolved.

The Python tests need NumPy and SciPy. If CMake does not find Python, pass `-DPython3_EXECUTABLE=/path/to/python` when configuring.

## Reproducing the manuscript

[`paper/README.md`](paper/README.md) maps every figure of the manuscript and its Supporting Information to the script that draws it, the data it reads and the run that produced the data. In short:

* `paper/inputs/` holds the input deck of every simulation shown;
* `paper/data/` holds the processed results, so most figures can be redrawn without rerunning anything;
* `paper/run_simulations.py` reruns the production simulations;
* [`paper/LARGE_DATA.md`](paper/LARGE_DATA.md) lists the large 3-D snapshot fields (about 29 GB) that do not fit in the repository, with checksums and the commands that regenerate them.

## Repository layout

```text
GPK/
├── CMakeLists.txt
├── include/            public API (gpk_core.hpp, gpk_thermo.hpp, gpk_micromagnetics.hpp)
├── src/                solver: Gaussian geometry, physics, topology, transport, time stepping
├── utils/              header-only input/output helpers
├── demos/              gpk driver, verification suites and benchmarks
├── inputs/             example input decks
├── scripts/            analysis, post-processing and Al–Cu workflow scripts
├── python/             2-D pure-Python reference implementation
├── gpu_tests/          optional CUDA phase-field/GPK benchmark test
├── paper/              everything needed to reproduce the manuscript figures
└── docs/               user guide and architecture notes
```

An optional CUDA benchmark compares a dense phase-field solver with GPU Gaussian-projected label kinetics in 3-D:

```bash
python -m pip install -r requirements-gpu.txt
python -m unittest gpu_tests/test_gpu_pf_gpk_3d.py
python scripts/gpu_pf_gpk_3d.py --sizes 24 32 48 --phases 8 --steps 12
```

## Citing GPK

If you use GPK, please cite the article and the software (see [`CITATION.cff`](CITATION.cff)):

> A. Linda and D. Raabe, *A general sharp-interface simulation framework for microstructure evolution*, submitted to npj Computational Materials (2026).

## Licence and contact

GPK is released under the [MIT licence](LICENSE). The Al–Cu thermodynamic data in `paper/calphad/COST507-modified.tdb` is the public COST 507 light-alloy database (Round II, 1999), as obtained from OpenCalphad, with edits that make it readable by pycalphad (noted as comments in the file). It remains under its original terms.

Questions and bug reports: open an issue on GitHub, or contact Albert Linda (a.linda@mpi-susmat.de) and Dierk Raabe (d.raabe@mpi-susmat.de), Max Planck Institute for Sustainable Materials, Düsseldorf.
