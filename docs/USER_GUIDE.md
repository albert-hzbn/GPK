# GPK user guide

This guide covers the input file format, output formats, post-processing,
a step-by-step tutorial, the parameter reference and how to add a model.
For installation and a first run, see the main [README](../README.md).

## 1. Input file format

Every simulation is controlled by an INI file.  Comments begin with `#`.
Sections are `[section_name]`.  Values accept scientific notation.

### General structure

Every INI file must begin with a `[run]` section that names the model:

```ini
[run]
model = solidification   # grain_growth | solidification | recrystallisation
                         # | ostwald_ripening | precipitation | precipitate_breakup
                         # | dislocation | wulff_shape | magnetic_annealing
                         # | magnetoelastic | ndfeb_wetting

[grid]
Nz = 1        # depth  (set to 1 for a 2-D run)
Ny = 128
Nx = 128
dx = 0.5e-6   # cell size [m]

[time]
t_end   = 2.0e-3   # end time [s]
n_snaps = 6        # number of output snapshots

[thermo]
M0      = 5.0e-8   # boundary mobility [m³/(J·s)]
sigma0  = 0.20     # interfacial energy [J/m²]
eps4    = 0.04     # 4-fold anisotropy strength (0 = isotropic)
D       = 3.0e-9   # solute diffusivity [m²/s]  (0 = no solute)

[output]
format  = csv      # vtk | csv | ppm | all
dir     = output/my_sim
prefix  = snap
compact = false    # true = interface cells only (smaller files)
```

### 2-D vs 3-D

Set `Nz = 1` for a 2-D (x–y plane) simulation.  All physics reduce correctly:
sphere → circle formula for grain radius, 3-D Laplacian → 2-D Laplacian, etc.

### Simulation-specific sections

| Model (`[run] model`) | Extra section | Key parameters |
|---|---|---|
| `grain_growth` | `[grain_growth]` | `n_grains`, `seed` |
| `solidification` | `[initial]` | `c0`, `undercooling`, `seed_radius` |
| `recrystallisation` | `[nucleation]` | `n_nuclei`, `nucleus_r`, `seed` |
| `precipitation` | `[initial]` | `n_seeds`, `seed_radius`, `c0`, `T0` |
| `ostwald_ripening` | `[particles]` | `n_particles`, `r_mean_cells`, `r_lognormal_s`, `c_particle`, `c_matrix` |
| other models | see the example deck | `inputs/<model>.ini` lists every key the model reads, with comments |

Full parameter reference: [§5](#5-parameter-reference).

---

## 2. Output formats

### CSV (recommended for post-processing)

One file per snapshot.  Space-separated columns, header lines start with `#`.

```
# GPK Microstructure CSV
# snap=0  time=0.0139125
# Nz=32  Ny=64  Nx=64
# dx=1e-06 [m]
# cols: x[m] y[m] z[m] grain_id alpha composition temperature orientation
0.000000e+00  0.000000e+00  0.000000e+00  3  0.973  0.000  0.000  1.047
...
```

Set `compact = true` to write only interface cells, dramatically reducing file
size for large 3-D grids.

### VTK (for ParaView / VisIt)

VTK Legacy ASCII `STRUCTURED_POINTS` format.  All fields (`grain_id`, `alpha`,
`composition`, `temperature`, `orientation`) are written as `SCALARS` datasets.

Open in ParaView:
- *File → Open* → select `snap_0000.vtk`
- Apply filter *Threshold* on `alpha` to extract the interface
- Colour by `grain_id` or `composition`

### PPM (quick visualisation)

A 2-D image slice at `z = Nz/2` for each field.  Readable by any image viewer
or `scripts/plot_ppm.py`.

### Switching format

Edit `format` in the `[output]` section of the INI file:

```ini
[output]
format = vtk      # ParaView
format = csv      # Python post-processing
format = all      # write every format simultaneously
```

---

## 3. Post-processing with Python

Install dependencies once:

```bash
python -m pip install -r requirements.txt
```

### `scripts/post_process.py` — grain property analysis

```bash
# Analyse all simulations → grain_analysis.png
python scripts/post_process.py

# Single simulation
python scripts/post_process.py --sim grain_growth

# Custom directory and output file
python scripts/post_process.py --dir build/output/solidification \
                                --out sol_analysis.png

# Control how many snapshots to load
python scripts/post_process.py --max-snaps 30
```

**What it measures per snapshot:**

| Metric | Description |
|---|---|
| Equivalent radius | $(3V/4\pi)^{1/3}$ (3-D) or $\sqrt{A/\pi}$ (2-D) |
| Aspect ratio | Largest / smallest inertia tensor eigenvalue |
| Grain boundary length / area | Interface face counting |
| Solute distribution | Mean, max, histogram per phase |
| Avrami exponent | JMAK linear fit in $\ln(t)$ vs $\ln(-\ln(1-X))$ space |

### `scripts/plot_microstructures.py` — overview figure

Re-runs all demos and plots metric time-series side-by-side.

```bash
python scripts/plot_microstructures.py
```

### `scripts/plot_ppm.py` — PPM gallery

Displays all PPM snapshots as a gallery (requires `format = ppm` or `all`).

```bash
python scripts/plot_ppm.py
python scripts/plot_ppm.py --pattern "solid_*"
python scripts/plot_ppm.py --explain
```

---

## 4. Tutorial — step-by-step example

### Step 1 — Build

```bash
cmake -B build && cmake --build build -j$(nproc)
```

### Step 2 — Edit the input file

Open `build/inputs/solidification.ini` and optionally tweak:

```ini
[run]
model = solidification   # already set — do not change for this demo

[thermo]
eps4 = 0.04   # → 0.0 for spherical, larger for more branching

[initial]
undercooling = 5.0   # increase for faster/denser dendrites
```

### Step 3 — Run

```bash
cd build && ./gpk inputs/solidification.ini && cd ..
```

Expected output:

```
=== GPK Solidification ===
Grid 1×128×128, dx=0.50 µm
OpenMP threads: 8

Gibbs-Thomson R* = 0.373 µm
Seed radius      = 2.0 µm (must be > R*)

Time[ms]   Solid frac    Solute max    Steps
0.333      0.0050        0.0200        ...
...
Wall time: 2.4 s
```

### Step 4 — Post-process

```bash
python scripts/post_process.py --sim solidification --out sol_tutorial.png
```

### Step 5 — Switch to 3-D

Edit `build/inputs/solidification.ini`:

```ini
[grid]
Nz = 64   # was 1
Ny = 64
Nx = 64
```

Re-run with the same command, then open
`build/output/solidification/snap_0005.vtk` in ParaView.

---

## 5. Parameter reference

### `[run]`

| Key | Values | Description |
|---|---|---|
| `model` | `grain_growth`, `solidification`, `recrystallisation`, `ostwald_ripening`, `precipitation`, `precipitate_breakup`, `dislocation`, `wulff_shape`, `magnetic_annealing`, `magnetoelastic`, `ndfeb_wetting` | Which simulation to run (required). Each model has an example deck in `inputs/` named after it. |

### `[grid]`

| Key | Type | Default | Description |
|---|---|---|---|
| `Nz` | int | 1 | Grid depth. `1` = 2-D simulation |
| `Ny` | int | — | Grid height |
| `Nx` | int | — | Grid width |
| `dx` | float | — | Cell size [m] |

### `[time]`

| Key | Type | Description |
|---|---|---|
| `t_end` | float | Simulation end time [s] |
| `n_snaps` | int | Number of output snapshots (evenly spaced) |

### `[thermo]`

| Key | Description | Typical range |
|---|---|---|
| `M0` | Boundary mobility [m³/(J·s)] | 1e-12 – 1e-7 |
| `sigma0` | Interfacial energy [J/m²] | 0.05 – 1.0 |
| `eps4` | Cubic anisotropy strength | 0.0 (isotropic) – 0.08 |
| `D` | Solute diffusivity [m²/s] | 0 (none) – 1e-8 |

Additional thermo keys per physics:

| Key | Physics | Description |
|---|---|---|
| `m_liq` | Solidification | Liquidus slope [K/(mol frac)] |
| `k_part` | Solidification | Partition coefficient |
| `T_melt` | Solidification | Melting point [K] |
| `L` | Solidification | Latent heat [J/m³] |
| `E_stored` | Recrystallisation | Stored dislocation energy [J/m³] |

### `[output]`

| Key | Values | Description |
|---|---|---|
| `format` | `vtk`, `csv`, `ppm`, `all` | Output format(s) |
| `dir` | path | Output directory (created if absent) |
| `prefix` | string | Filename prefix (default `snap`) |
| `compact` | `true`/`false` | Write interface cells only (CSV) |

### `[grain_growth]`

| Key | Description |
|---|---|
| `n_grains` | Number of initial grains |
| `seed` | Random seed |

### `[nucleation]` (recrystallisation)

| Key | Description |
|---|---|
| `n_nuclei` | Number of rex nuclei |
| `nucleus_r` | Nucleus radius [cells] |
| `seed` | Random seed |

### `[initial]` (solidification)

| Key | Description |
|---|---|
| `c0` | Initial solute [mol frac] |
| `undercooling` | Initial undercooling [K] |
| `seed_radius` | Solid seed radius [cells] |

### `[particles]` (Ostwald ripening)

| Key | Description |
|---|---|
| `n_particles` | Number of particles to place |
| `r_mean_cells` | Median radius [cells] |
| `r_lognormal_s` | Log-normal distribution width |
| `c_particle` | Particle composition [mol frac] |
| `c_matrix` | Matrix composition [mol frac] |
| `seed` | Random seed |

---

## 6. Extending GPK

### Adding a new model

1. In `include/gpk_thermo.hpp`, define a struct inheriting `ThermoModel`:

```cpp
struct MyThermo : ThermoModel {
    float my_param = 1.0f;

    float mobility(float, float, float,
                   float, float, float) const override {
        return M0;
    }
    float bulk_driving_force(int phi_parent, int phi_product,
                             float c, float T) const override {
        (void)phi_parent; (void)phi_product; (void)c;
        return my_param * T; // G_product - G_parent [J/m^3]
    }
};
```

2. Add a `run_my_model(InputParser& cfg)` function in `demos/gpk.cpp` following
   the same pattern as the existing ones.

3. Register it in `main()` inside `demos/gpk.cpp`:

```cpp
else if (model == "my_model") return run_my_model(cfg);
```

4. Create an INI file with `[run] model = my_model` and any extra sections your
   model needs.  No CMakeLists.txt change is required.

### Loading CSV output in Python

```python
import numpy as np, re
from pathlib import Path

def read_snap(path):
    meta = {}
    with open(path) as f:
        for line in f:
            if not line.startswith('#'): break
            for key in ('Nz','Ny','Nx'):
                m = re.search(rf'{key}=(\d+)', line)
                if m: meta[key] = int(m.group(1))
            m = re.search(r'time=([\d.e+\-]+)', line)
            if m: meta['time'] = float(m.group(1))
    data = np.loadtxt(path, comments='#')
    return data, meta

data, meta = read_snap('build/output/grain_growth/snap_0005.csv')
x, y, z  = data[:,0], data[:,1], data[:,2]
grain_id = data[:,3].astype(int)
alpha    = data[:,4]
```

### 2-D Python reference implementation

`python/` contains a pure-Python 2-D GPK implementation with three demos:

```bash
cd python
python demo_grain_growth.py
python demo_solidification.py
python demo_anisotropy.py
```
