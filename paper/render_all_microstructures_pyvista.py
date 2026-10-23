"""Render every manuscript/SI microstructure panel with VTK through PyVista.

Run this module with ParaView's ``pvpython``.  It emits only raster panel
assets; the publication-figure scripts then place those assets beside vector
text and quantitative plots without re-rendering the microstructure data.
"""

from __future__ import annotations

import csv
import os
import re
import sys
from pathlib import Path

import numpy as np

import pyvista as pv

from gpk_data import output_dir
from pyvista_rendering import (
    add_shaded_surface,
    configure_3d_camera,
    render_label_field_2d,
    render_scalar_field_2d,
)


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
ASSETS = HERE / "figures" / "pyvista_panels"
ASSETS.mkdir(parents=True, exist_ok=True)

BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
PURPLE = "#7B4FA3"
CYAN = "#56B4E9"


def load_csv_snapshot(directory: Path, index: int) -> dict:
    files = sorted(directory.glob("snap_*.csv"))
    if not files:
        raise FileNotFoundError(f"No snapshots in {directory}")
    path = files[index]
    meta: dict[str, float] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.startswith("#"):
                break
            for key in ("time", "dx", "Nz", "Ny", "Nx"):
                match = re.search(rf"{key}=([\d.e+\-]+)", line)
                if match:
                    meta[key] = float(match.group(1))
    data = np.loadtxt(path, comments="#", usecols=(3,), dtype=np.int64)
    nx = int(meta["Nx"])
    ny = int(meta["Ny"])
    nz = int(meta.get("Nz", 1))
    return {"labels": data.reshape(nz, ny, nx), "meta": meta, "path": path}


def load_bicrystal(path: Path) -> np.ndarray:
    header = path.read_text(encoding="utf-8", errors="replace").splitlines()[0]
    shape = {key: int(value) for key, value in re.findall(r"(Ny|Nx)=(\d+)", header)}
    values = np.loadtxt(path, comments="#", skiprows=2, dtype=np.int64)
    return values.reshape(shape["Ny"], shape["Nx"])


def load_pf_label(base: Path, kind: str, index: int) -> np.ndarray:
    data = np.loadtxt(base / f"{kind}_snap_{index}.csv", delimiter=",",
                      skiprows=3, dtype=np.int64)
    side = int(round(np.sqrt(data.shape[0])))
    return data[:, 2].reshape(side, side)


def render_qualitative_2d() -> None:
    root = output_dir("manuscript_large_domain")
    cases = [
        ("isotropic", root / "isotropic_2d_connected", None),
        ("anisotropic", root / "anisotropic_2d_connected", None),
        ("recrystallisation", root / "recrystallisation_2d", None),
        ("multiphase", root / "multiphase_2d",
         {0: "#E5E7EB", 1: ORANGE, 2: BLUE}),
        ("elastic", root / "elastic_2d",
         {0: "#DDE8F2", 1: PURPLE}),
    ]
    for name, directory, semantic in cases:
        for state, index in (("initial", 0), ("final", -1)):
            snap = load_csv_snapshot(directory, index)
            render_label_field_2d(
                snap["labels"][snap["labels"].shape[0] // 2],
                ASSETS / f"qualitative2d_{name}_{state}.png",
                semantic=semantic, curvature=False, show_boundaries=False,
                window_size=(1400, 1400),
            )
        print(f"2D qualitative: {name}", flush=True)


def render_phase_field_comparison() -> None:
    base = output_dir("gpk_pf_comparison")
    render_label_field_2d(load_pf_label(base, "gpk", 0),
                          ASSETS / "pf_common_0.png", sigma_cells=1.25,
                          window_size=(900, 900))
    for kind in ("gpk", "pf"):
        for index in (4, 8, 16):
            render_label_field_2d(load_pf_label(base, kind, index),
                                  ASSETS / f"pf_{kind}_{index}.png",
                                  sigma_cells=1.25, window_size=(900, 900))
    print("2D GPK/PF comparison", flush=True)


def render_bicrystals() -> None:
    base = output_dir("validate_extended_base_models")
    planar = {
        "g1": {1: CYAN, 2: PURPLE},
        "g2": {0: "#D8DEE8", 1: ORANGE},
        "g5": {0: "#B8D8EA", 1: ORANGE},
        "g6": {0: "#F7F9FB", 1: GREEN, 2: CYAN},
        "g7": {0: "#C96A5B", 1: "#B8C1C8"},
        "g8": {0: "#D8DEE8", 1: PURPLE},
        "g9": {1: BLUE, 2: PURPLE},
        "g10": {0: "#D8DEE8", 1: BLUE},
    }
    for key, semantic in planar.items():
        for source, state in (("before", "initial"), ("after", "final")):
            render_label_field_2d(load_bicrystal(base / f"{key}_grid_{source}.csv"),
                                  ASSETS / f"bicrystal_{key}_{state}.png",
                                  semantic=semantic, sigma_cells=1.0,
                                  window_size=(1050, 600))
    curved = {"g13": BLUE, "g14": GREEN, "g15": PURPLE}
    for key, colour in curved.items():
        for source, state in (("before", "initial"), ("after", "final")):
            render_label_field_2d(
                load_bicrystal(base / f"{key}_grid_{source}.csv"),
                ASSETS / f"curved_{key}_{state}.png",
                semantic={0: "#E4E9F0", 1: colour}, sigma_cells=1.1,
                crop=(65, 295, 65, 295), window_size=(900, 900),
            )
    print("2D planar and curved bicrystals", flush=True)


def _load_alcu(case: str, index: int) -> dict:
    root = output_dir("alcu_validation") / "precipitation"
    return dict(np.load(root / case / f"snap_{index:03d}.npz"))


def render_alcu_sections() -> None:
    snap = _load_alcu("three_dimensional_ultrafine", 3)
    labels = snap["labels"]
    z = int(np.argmax(np.count_nonzero(labels > 0, axis=(0, 1))))
    support = labels[:, :, z] > 0
    render_scalar_field_2d(
        snap["c_matrix"][:, :, z], ASSETS / "alcu_matrix_cu.png",
        vmin=0.004, vmax=0.023, ramp="blues", window_size=(1100, 1000),
    )
    render_scalar_field_2d(
        snap["elastic_energy"][:, :, z] / 1.0e6,
        ASSETS / "alcu_elastic_energy.png", ramp="magma",
        window_size=(1100, 1000),
    )
    for case, tag in (("three_dimensional_mesh_fine", "96"),
                      ("three_dimensional_mesh_ultrafine", "128")):
        state = _load_alcu(case, 3)
        mask = state["labels"] == 3
        coords = np.argwhere(mask)
        n_y = mask.shape[1]
        angles = 2.0 * np.pi * coords[:, 1] / n_y
        centre = int(np.rint((np.arctan2(np.sum(np.sin(angles)),
                                        np.sum(np.cos(angles))) % (2.0 * np.pi))
                            * n_y / (2.0 * np.pi))) % n_y
        render_label_field_2d(mask[:, centre, :].astype(np.int64),
                              ASSETS / f"alcu_mesh_{tag}.png",
                              semantic={0: "#F1F5F9", 1: GREEN},
                              sigma_cells=1.1, window_size=(1000, 850))
    print("Al-Cu scalar and mesh-audit sections", flush=True)


def _plate_mesh(position: np.ndarray, variant: np.ndarray,
                diameter: np.ndarray, thickness: np.ndarray,
                resolution: int = 16) -> pv.PolyData:
    """Build one indexed triangular mesh for a population of plate particles."""
    points: list[np.ndarray] = []
    faces: list[int] = []
    offset = 0
    angles = np.linspace(0.0, 2.0 * np.pi, resolution, endpoint=False)
    ring = np.column_stack((np.cos(angles), np.sin(angles)))
    for centre, habit, width, height in zip(position, variant, diameter, thickness):
        normal_axis = {1: 2, 2: 1, 3: 0}[int(habit)]
        radial_axes = [axis for axis in range(3) if axis != normal_axis]
        local = np.zeros((2 * resolution + 2, 3), dtype=float)
        for layer, sign in enumerate((-0.5, 0.5)):
            subset = slice(layer * resolution, (layer + 1) * resolution)
            local[subset, radial_axes[0]] = 0.5 * width * ring[:, 0]
            local[subset, radial_axes[1]] = 0.5 * width * ring[:, 1]
            local[subset, normal_axis] = sign * height
        local[-2, normal_axis] = -0.5 * height
        local[-1, normal_axis] = 0.5 * height
        points.append(local + centre)
        bottom_centre = offset + 2 * resolution
        top_centre = bottom_centre + 1
        for i in range(resolution):
            j = (i + 1) % resolution
            b0, b1 = offset + i, offset + j
            t0, t1 = offset + resolution + i, offset + resolution + j
            faces.extend((3, bottom_centre, b1, b0,
                          3, top_centre, t0, t1,
                          3, b0, b1, t1, 3, b0, t1, t0))
        offset += len(local)
    if not points:
        return pv.PolyData()
    return pv.PolyData(np.vstack(points), np.asarray(faces, dtype=np.int64))


def render_population(path: Path, output: Path) -> None:
    data = dict(np.load(path))
    domain = float(data["domain_nm"])
    # These panels occupy half of a full-width journal figure.  A 4k-class
    # native render preserves thin plate edges and specular gradients after
    # the composite PDF and manuscript page are reduced.
    plotter = pv.Plotter(off_screen=True, window_size=(3200, 2700))
    plotter.set_background("#F4F7FA", top="#FFFFFF")
    try:
        plotter.enable_anti_aliasing("ssaa")
    except Exception:
        pass
    floor = pv.Plane(center=(domain / 2, domain / 2, -18.0),
                     direction=(0, 0, 1), i_size=1.08 * domain,
                     j_size=1.08 * domain, i_resolution=1, j_resolution=1)
    plotter.add_mesh(floor, color="#D7E2EA", opacity=0.42, pbr=True,
                     metallic=0.0, roughness=0.68)
    # Camera bounds must include complete precipitates, not just the nominal
    # periodic cell.  Plate centres close to a periodic face can extend beyond
    # [0, domain]; framing only the cell clipped those legitimate surfaces.
    scene_bounds = np.asarray([0.0, domain, 0.0, domain, -18.0, domain], dtype=float)

    def include_mesh_bounds(mesh: pv.PolyData) -> None:
        if mesh.n_points == 0:
            return
        bounds = np.asarray(mesh.bounds, dtype=float)
        scene_bounds[0::2] = np.minimum(scene_bounds[0::2], bounds[0::2])
        scene_bounds[1::2] = np.maximum(scene_bounds[1::2], bounds[1::2])

    for habit, colour in ((1, BLUE), (2, ORANGE), (3, GREEN)):
        chosen = data["theta_prime_variant"] == habit
        mesh = _plate_mesh(data["theta_prime_position_nm"][chosen],
                           data["theta_prime_variant"][chosen],
                           data["theta_prime_diameter_nm"][chosen],
                           data["theta_prime_thickness_nm"][chosen], 32)
        include_mesh_bounds(mesh)
        add_shaded_surface(plotter, mesh, colour, metallic=0.04, roughness=0.48)
    # The numerous theta-double-prime plates are drawn as a single translucent
    # mesh, retaining particle count while keeping theta-prime legible.
    mesh = _plate_mesh(data["theta_double_prime_position_nm"],
                       data["theta_double_prime_variant"],
                       data["theta_double_prime_diameter_nm"],
                       data["theta_double_prime_thickness_nm"], 16)
    include_mesh_bounds(mesh)
    add_shaded_surface(plotter, mesh, "#7CC7E8", opacity=0.22,
                       metallic=0.01, roughness=0.55)
    plotter.add_mesh(pv.Box(bounds=(0, domain, 0, domain, 0, domain)).extract_all_edges(),
                     color="#536574", opacity=0.48, line_width=1.1,
                     lighting=False)
    plotter.add_light(pv.Light(position=(-0.8 * domain, -0.5 * domain, 2.5 * domain),
                               focal_point=(domain / 2,) * 3, intensity=0.72))
    plotter.add_light(pv.Light(position=(1.8 * domain, 1.4 * domain, 1.5 * domain),
                               focal_point=(domain / 2,) * 3, intensity=0.34))
    # A subdued rear light completes a three-point arrangement.  It separates
    # overlapping plate silhouettes without restoring the former high-gloss
    # appearance because the particle materials remain predominantly diffuse.
    plotter.add_light(pv.Light(position=(1.55 * domain, -0.75 * domain, 0.95 * domain),
                               focal_point=(domain / 2,) * 3, intensity=0.24))
    try:
        plotter.enable_shadows()
    except Exception:
        pass
    span = scene_bounds[1::2] - scene_bounds[0::2]
    # Frame the complete precipitate geometry, including plates that cross a
    # periodic domain face.  The former tight camera crop allowed those plates
    # to reach the raster edge and made the publication panel look clipped.
    padding = 0.095 * float(np.max(span))
    scene_bounds[0::2] -= padding
    scene_bounds[1::2] += padding
    configure_3d_camera(plotter, tuple(scene_bounds),
                        azimuth=-52.0, elevation=27.0, zoom=0.95)
    plotter.show(screenshot=str(output), auto_close=True)


def render_alcu_populations() -> None:
    root = output_dir("alcu_validation") / "realistic_population_3d"
    for hour in (120, 168):
        render_population(root / f"particles_{hour}h.npz",
                          ASSETS / f"alcu_population_{hour}h.png")
        print(f"Al-Cu 3D population: {hour} h", flush=True)


def main() -> None:
    pv.global_theme.allow_empty_mesh = True
    render_qualitative_2d()
    render_phase_field_comparison()
    render_bicrystals()
    render_alcu_sections()
    render_alcu_populations()
    print(f"PyVista panel assets written to {ASSETS}", flush=True)


if __name__ == "__main__":
    main()
