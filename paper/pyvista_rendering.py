"""Shared VTK/PyVista rendering primitives for manuscript microstructures.

This module deliberately contains no Matplotlib dependency.  Simulation
fields are converted to VTK image data, interfaces are reconstructed as
Gaussian-projected VTK contours, and the off-screen raster is produced by a
PyVista ``Plotter``.  Matplotlib is used elsewhere only to assemble these
rendered panels with vector text and quantitative plots.
"""

from __future__ import annotations

import colorsys
from collections import defaultdict, deque
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pyvista as pv


BACKGROUND = "#F7F9FC"
BOUNDARY = "#17212B"
BOUNDARY_HALO = "#FFFFFF"


def _hex_rgb(value: str) -> np.ndarray:
    value = value.lstrip("#")
    if len(value) != 6:
        raise ValueError(f"Expected #RRGGBB colour, got {value!r}")
    return np.asarray([int(value[i:i + 2], 16) for i in (0, 2, 4)], dtype=np.uint8)


def categorical_rgb(labels: np.ndarray,
                    semantic: Mapping[int, str] | None = None) -> np.ndarray:
    """Return a persistent, label-indexed uint8 RGB field."""
    labels = np.asarray(labels, dtype=np.int64)
    rgb = np.empty(labels.shape + (3,), dtype=np.uint8)
    for label in np.unique(labels):
        if semantic is not None:
            colour = semantic.get(int(label), "#CBD5E1")
            value = _hex_rgb(colour)
        else:
            hue = (float(label) * 0.61803398875) % 1.0
            # Use a print-safe categorical palette with enough chroma and
            # luminance contrast to survive reduction to journal-column size.
            # The older near-pastel palette became washed out on white pages.
            saturation = 0.60 + 0.055 * (int(label) % 3)
            brightness = 0.76 + 0.032 * (int(label) % 5)
            value = np.rint(255.0 * np.asarray(
                colorsys.hsv_to_rgb(hue, saturation, brightness)
            )).astype(np.uint8)
        rgb[labels == label] = value
    return rgb


def _periodic_gaussian(field: np.ndarray, sigma: float) -> np.ndarray:
    """Separable periodic Gaussian convolution implemented with NumPy rolls."""
    if sigma <= 0.0:
        return np.asarray(field, dtype=float)
    radius = max(1, int(np.ceil(3.5 * sigma)))
    offsets = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel /= kernel.sum()
    result = np.asarray(field, dtype=float)
    for axis in range(result.ndim):
        filtered = np.zeros_like(result)
        for offset, weight in zip(offsets, kernel):
            filtered += weight * np.roll(result, int(offset), axis=axis)
        result = filtered
    return result


def regularize_labels(labels: np.ndarray, passes: int = 2) -> np.ndarray:
    """Remove isolated display voxels with a conservative periodic modal pass.

    A cell changes only when at least five cells in its 3x3 neighbourhood
    support the same label.  The returned field is exclusively a rendering
    reconstruction; callers retain the original labels for every metric and
    topology audit.
    """
    result = np.asarray(labels, dtype=np.int64).copy()
    for _ in range(max(0, passes)):
        neighbourhood = np.stack([
            np.roll(np.roll(result, dy, axis=0), dx, axis=1)
            for dy in (-1, 0, 1) for dx in (-1, 0, 1)
        ])
        best_label = result.copy()
        best_count = np.ones(result.shape, dtype=np.uint8)
        for candidate in neighbourhood:
            count = np.count_nonzero(neighbourhood == candidate, axis=0).astype(np.uint8)
            update = count > best_count
            best_label[update] = candidate[update]
            best_count[update] = count[update]
        result = np.where(best_count >= 5, best_label, result)
    return result


def remove_disconnected_fragments(labels: np.ndarray,
                                  relative_cutoff: float = 0.02,
                                  absolute_cutoff: int = 12) -> np.ndarray:
    """Remove only small components that duplicate an existing grain label.

    Periodic connectivity is honoured.  A genuinely small grain has its own
    persistent label and is retained; only satellite components of a label
    whose principal component is much larger are replaced.  This prevents
    display of label fragments as grains embedded inside unrelated grains.
    """
    result = np.asarray(labels, dtype=np.int64).copy()
    ny, nx = result.shape
    visited = np.zeros((ny, nx), dtype=bool)
    components: dict[int, list[list[tuple[int, int]]]] = defaultdict(list)
    for y0 in range(ny):
        for x0 in range(nx):
            if visited[y0, x0]:
                continue
            label = int(result[y0, x0])
            queue: deque[tuple[int, int]] = deque([(y0, x0)])
            visited[y0, x0] = True
            component: list[tuple[int, int]] = []
            while queue:
                y, x = queue.popleft()
                component.append((y, x))
                for yn, xn in (((y - 1) % ny, x), ((y + 1) % ny, x),
                               (y, (x - 1) % nx), (y, (x + 1) % nx)):
                    if not visited[yn, xn] and int(result[yn, xn]) == label:
                        visited[yn, xn] = True
                        queue.append((yn, xn))
            components[label].append(component)

    orphan = np.zeros((ny, nx), dtype=bool)
    for label_components in components.values():
        if len(label_components) < 2:
            continue
        largest = max(len(component) for component in label_components)
        cutoff = max(absolute_cutoff, int(np.ceil(relative_cutoff * largest)))
        for component in label_components:
            if len(component) < cutoff:
                yy, xx = zip(*component)
                orphan[np.asarray(yy), np.asarray(xx)] = True

    for _ in range(6):
        if not np.any(orphan):
            break
        neighbourhood = np.stack([
            np.roll(np.roll(result, dy, axis=0), dx, axis=1)
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1),
                           (-1, -1), (-1, 1), (1, -1), (1, 1))
        ])
        valid = np.stack([
            ~np.roll(np.roll(orphan, dy, axis=0), dx, axis=1)
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1),
                           (-1, -1), (-1, 1), (1, -1), (1, 1))
        ])
        best_label = result.copy()
        best_count = np.zeros((ny, nx), dtype=np.uint8)
        for candidate in neighbourhood:
            count = np.count_nonzero((neighbourhood == candidate) & valid, axis=0)
            update = count > best_count
            best_label[update] = candidate[update]
            best_count[update] = count[update]
        fill = orphan & (best_count > 0)
        result[fill] = best_label[fill]
        orphan[fill] = False
    return result


def _image_plane(values: np.ndarray, name: str = "values") -> pv.ImageData:
    """Create a zero-thickness VTK image whose points coincide with cell centres."""
    ny, nx = values.shape[:2]
    image = pv.ImageData(dimensions=(nx, ny, 1), spacing=(1.0, 1.0, 1.0),
                         origin=(0.5, 0.5, 0.02))
    image.point_data[name] = np.asarray(values).reshape((-1,) + values.shape[2:],
                                                         order="C")
    return image


def _cell_plane(values: np.ndarray, name: str = "rgb") -> pv.ImageData:
    """Create a VTK image with one coloured cell per simulation cell."""
    ny, nx = values.shape[:2]
    image = pv.ImageData(dimensions=(nx + 1, ny + 1, 1),
                         spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0))
    image.cell_data[name] = np.asarray(values).reshape((-1,) + values.shape[2:],
                                                        order="C")
    return image


def _colour_ramp(values: np.ndarray, anchors: Sequence[tuple[float, str]]) -> np.ndarray:
    """Map normalized values to RGB without importing Matplotlib."""
    x = np.clip(np.asarray(values, dtype=float), 0.0, 1.0)
    positions = np.asarray([item[0] for item in anchors], dtype=float)
    colours = np.asarray([_hex_rgb(item[1]) for item in anchors], dtype=float)
    out = np.empty(x.shape + (3,), dtype=np.uint8)
    for channel in range(3):
        out[..., channel] = np.rint(np.interp(x, positions, colours[:, channel])).astype(np.uint8)
    return out


def scalar_rgb(values: np.ndarray, vmin: float | None = None,
               vmax: float | None = None, ramp: str = "viridis") -> np.ndarray:
    """Convert a scalar field to a perceptually ordered uint8 RGB field."""
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros(values.shape + (3,), dtype=np.uint8)
    lo = float(np.min(finite) if vmin is None else vmin)
    hi = float(np.max(finite) if vmax is None else vmax)
    scaled = np.clip((values - lo) / max(hi - lo, 1.0e-30), 0.0, 1.0)
    ramps = {
        "viridis": ((0.00, "#440154"), (0.25, "#3B528B"),
                    (0.50, "#21918C"), (0.75, "#5EC962"), (1.00, "#FDE725")),
        "plasma": ((0.00, "#0D0887"), (0.25, "#7E03A8"),
                   (0.50, "#CC4778"), (0.75, "#F89540"), (1.00, "#F0F921")),
        "blues": ((0.00, "#F7FBFF"), (0.35, "#C6DBEF"),
                  (0.70, "#4292C6"), (1.00, "#08306B")),
        "magma": ((0.00, "#000004"), (0.25, "#51127C"),
                  (0.50, "#B73779"), (0.75, "#FC8961"), (1.00, "#FCFDBF")),
        # Yellow-to-red ramp of the v3 palette (also defined in alcu_si_figures_v3.py).
        "yellowred": ((0.00, "#FFFBE6"), (0.35, "#FFD966"),
                      (0.70, "#F07A20"), (1.00, "#B01010")),
    }
    return _colour_ramp(scaled, ramps[ramp])


def _curvature_rgb(contour: pv.PolyData, projection: np.ndarray,
                   kappa_max: float) -> np.ndarray:
    """Sample Gaussian-Hessian curvature on a VTK contour and colour it."""
    gy, gx = np.gradient(projection, edge_order=2)
    gyy, gyx = np.gradient(gy, edge_order=2)
    gxy, gxx = np.gradient(gx, edge_order=2)
    gxy = 0.5 * (gxy + gyx)
    grad2 = gx * gx + gy * gy
    denom = np.maximum(grad2 * np.sqrt(grad2), 1.0e-12)
    kappa = np.abs((gx * gx * gyy - 2.0 * gx * gy * gxy + gy * gy * gxx) / denom)
    xy = np.rint(contour.points[:, :2] - 0.5).astype(int)
    xy[:, 0] = np.clip(xy[:, 0], 0, projection.shape[1] - 1)
    xy[:, 1] = np.clip(xy[:, 1], 0, projection.shape[0] - 1)
    return _colour_ramp(np.clip(kappa[xy[:, 1], xy[:, 0]] / kappa_max, 0.0, 1.0),
                        ((0.00, "#0D0887"), (0.25, "#7E03A8"),
                         (0.50, "#CC4778"), (0.75, "#F89540"),
                         (1.00, "#F0F921")))


def _configure_xy(plotter: pv.Plotter, nx: int, ny: int) -> None:
    """Fit the VTK cell edges exactly to the viewport without a background rim."""
    cx, cy = 0.5 * nx, 0.5 * ny
    plotter.camera_position = [(cx, cy, 2.5 * max(nx, ny)),
                               (cx, cy, 0.0), (0.0, 1.0, 0.0)]
    plotter.camera.parallel_projection = True
    aspect = plotter.window_size[0] / plotter.window_size[1]
    # The cell plane spans [0, nx] x [0, ny].  A factor of 0.5 maps those
    # physical edges to the viewport edges.  The former 0.505 overscan left a
    # visible six-pixel background rim in a 1400-pixel square panel.
    plotter.camera.parallel_scale = 0.5 * max(ny, nx / aspect)


def render_label_field_2d(labels: np.ndarray, output: Path | str,
                          semantic: Mapping[int, str] | None = None,
                          sigma_cells: float = 1.25,
                          curvature: bool = False,
                          kappa_max: float = 0.16,
                          crop: tuple[int, int, int, int] | None = None,
                          regularize_passes: int = 0,
                          remove_fragments: bool = False,
                          show_boundaries: bool = False,
                          window_size: tuple[int, int] = (900, 900),
                          background: str = BACKGROUND) -> Path:
    """Render a categorical 2D microstructure entirely through PyVista/VTK."""
    labels = np.asarray(labels, dtype=np.int64)
    if crop is not None:
        x0, x1, y0, y1 = crop
        labels = labels[y0:y1, x0:x1]
    if remove_fragments:
        labels = remove_disconnected_fragments(labels)
    if regularize_passes:
        labels = regularize_labels(labels, regularize_passes)
    ny, nx = labels.shape
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    plotter = pv.Plotter(off_screen=True, window_size=window_size)
    plotter.set_background(background)
    base = _cell_plane(categorical_rgb(labels, semantic))
    plotter.add_mesh(base, scalars="rgb", rgb=True, preference="cell",
                     show_scalar_bar=False, lighting=False)

    # Each resident label produces its own continuous 0.5 level set, matching
    # the Gaussian geometry estimator rather than exposing stair-stepped voxel
    # edges.  Duplicate coincident contours are harmless and improve coverage
    # at periodic junctions.
    if show_boundaries:
        contours: list[tuple[pv.PolyData, np.ndarray]] = []
        for label in np.unique(labels):
            projection = _periodic_gaussian((labels == label).astype(float), sigma_cells)
            if float(np.max(projection)) < 0.5:
                continue
            field = _image_plane(projection, "projection")
            contour = field.contour([0.5], scalars="projection")
            if contour.n_points < 2:
                continue
            contour.points[:, 2] = 0.08
            contours.append((contour, projection))

        for contour, _ in contours:
            plotter.add_mesh(contour, color=BOUNDARY_HALO, line_width=4.2,
                             show_scalar_bar=False, lighting=False)
        for contour, projection in contours:
            if curvature:
                contour.point_data["curvature_rgb"] = _curvature_rgb(
                    contour, projection, kappa_max
                )
                plotter.add_mesh(contour, scalars="curvature_rgb", rgb=True,
                                 line_width=2.2, show_scalar_bar=False, lighting=False)
            else:
                plotter.add_mesh(contour, color=BOUNDARY, line_width=2.3,
                                 show_scalar_bar=False, lighting=False)

    _configure_xy(plotter, nx, ny)
    plotter.show(screenshot=str(output), auto_close=True)
    return output


def render_scalar_field_2d(values: np.ndarray, output: Path | str,
                           vmin: float | None = None, vmax: float | None = None,
                           ramp: str = "viridis",
                           interface_mask: np.ndarray | None = None,
                           interface_colour: str = "#FFFFFF",
                           sigma_cells: float = 0.75,
                           window_size: tuple[int, int] = (1000, 900)) -> Path:
    """Render a continuous 2D field and optional sharp interface in PyVista."""
    values = np.asarray(values, dtype=float)
    ny, nx = values.shape
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    plotter = pv.Plotter(off_screen=True, window_size=window_size)
    plotter.set_background(BACKGROUND)
    plotter.add_mesh(_cell_plane(scalar_rgb(values, vmin, vmax, ramp)),
                     scalars="rgb", rgb=True, preference="cell",
                     show_scalar_bar=False, lighting=False)
    if interface_mask is not None:
        projection = _periodic_gaussian(np.asarray(interface_mask, dtype=float), sigma_cells)
        contour = _image_plane(projection, "projection").contour([0.45], scalars="projection")
        if contour.n_points:
            contour.points[:, 2] = 0.08
            plotter.add_mesh(contour, color=interface_colour, line_width=2.5,
                             show_scalar_bar=False, lighting=False)
    _configure_xy(plotter, nx, ny)
    plotter.show(screenshot=str(output), auto_close=True)
    return output


def image_surface(mask: np.ndarray, sigma_cells: float = 1.0,
                  level: float = 0.45) -> pv.PolyData:
    """Return a smooth PyVista isosurface from a categorical 3D mask."""
    mask = np.asarray(mask, dtype=float)
    nz, ny, nx = mask.shape
    volume = pv.ImageData(dimensions=(nx, ny, nz), spacing=(1.0, 1.0, 1.0),
                          origin=(0.5, 0.5, 0.5))
    # Periodic Gaussian projection is performed in NumPy so its numerical
    # convention matches the 2D renderer and the model's periodic domains.
    projection = _periodic_gaussian(mask, sigma_cells)
    volume.point_data["projection"] = projection.ravel(order="C")
    surface = volume.contour([level], scalars="projection")
    if surface.n_points:
        surface = surface.clean().triangulate()
        surface = surface.smooth(n_iter=24, relaxation_factor=0.08,
                                 boundary_smoothing=True,
                                 feature_smoothing=False)
        surface = surface.compute_normals(point_normals=True, cell_normals=False,
                                          auto_orient_normals=True)
    return surface


def add_shaded_surface(plotter: pv.Plotter, surface: pv.PolyData,
                       colour: str, opacity: float = 1.0,
                       metallic: float = 0.03, roughness: float = 0.48) -> None:
    """Add a smooth, depth-cued surface using PyVista's PBR material path."""
    if surface.n_points == 0:
        return
    plotter.add_mesh(surface, color=colour, opacity=opacity, smooth_shading=True,
                     pbr=True, metallic=metallic, roughness=roughness,
                     show_scalar_bar=False)


def configure_3d_camera(plotter: pv.Plotter, bounds: Sequence[float],
                        azimuth: float = -45.0, elevation: float = 25.0,
                        zoom: float = 1.10) -> None:
    """Apply a reproducible isometric, parallel-projection camera."""
    xmin, xmax, ymin, ymax, zmin, zmax = map(float, bounds)
    centre = np.asarray([(xmin + xmax) / 2.0, (ymin + ymax) / 2.0,
                         (zmin + zmax) / 2.0])
    radius = max(xmax - xmin, ymax - ymin, zmax - zmin)
    az, el = np.deg2rad([azimuth, elevation])
    direction = np.asarray([np.cos(el) * np.cos(az),
                            np.cos(el) * np.sin(az), np.sin(el)])
    plotter.camera_position = [centre + 3.2 * radius * direction,
                               centre, (0.0, 0.0, 1.0)]
    plotter.camera.parallel_projection = True
    plotter.camera.parallel_scale = 0.72 * radius / max(zoom, 1.0e-6)
