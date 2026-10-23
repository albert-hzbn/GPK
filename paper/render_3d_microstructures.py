"""Render true 3D GPK label volumes as publication-quality surface reconstructions.

Run with ParaView's pvpython. The output is a high-resolution raster rendering
wrapped in a PDF, because transparency, surface shading and depth ordering are
not reliably preserved by vector exporters.
"""

from __future__ import annotations

import colorsys
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

# ParaView bundles VTK but not PyVista.  The repository-local directory holds
# PyVista's pure-Python package and is intentionally placed before ParaView's
# older typing-extensions module.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / ".pyvista_pv"))
import pyvista as pv
from pyvista_rendering import categorical_rgb, configure_3d_camera
from gpk_data import data_roots, output_dir


ROOT = Path(__file__).resolve().parent
DATA_ROOTS = data_roots()
RENDER_DIR = Path(os.environ.get("TEMP", ".")) / "gpk_3d_render"
RENDER_DIR.mkdir(parents=True, exist_ok=True)
OUT_PNG = RENDER_DIR / "fig_qualitative_3d_curvature_longtime.png"
OUT_PDF = RENDER_DIR / "fig_qualitative_3d_curvature_longtime.pdf"
DEJAVU_REGULAR = Path(r"C:\Windows\Fonts\arial.ttf")
DEJAVU_BOLD = Path(r"C:\Windows\Fonts\arialbd.ttf")


def publication_fonts() -> tuple[ImageFont.FreeTypeFont, ImageFont.FreeTypeFont]:
    """Use the same DejaVu Sans family as the Matplotlib manuscript figures."""
    missing = [path for path in (DEJAVU_REGULAR, DEJAVU_BOLD) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required publication font missing: {missing[0]}")
    return (ImageFont.truetype(str(DEJAVU_REGULAR), 48),
            ImageFont.truetype(str(DEJAVU_BOLD), 54))


def read_snapshot(path: Path) -> dict:
    meta: dict[str, float] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.startswith("#"):
                break
            if "Nx=" in line:
                for token in line[1:].split():
                    key, value = token.split("=")
                    meta[key] = int(value)
            elif "time=" in line:
                for token in line[1:].split():
                    if "=" in token:
                        key, value = token.split("=", 1)
                        meta[key] = float(value)
            elif line.startswith("# dx="):
                meta["dx"] = float(line.split("=")[1].split()[0])

    nz, ny, nx = int(meta["Nz"]), int(meta["Ny"]), int(meta["Nx"])
    cached = Path(path).with_suffix(".npz")
    if cached.exists():
        # Binary copy of the same label and orientation columns, written by
        # cache_snapshot.py for snapshots too large to parse as text here.
        with np.load(cached) as store:
            return {
                "meta": meta,
                "labels": store["labels"].reshape(nz, ny, nx),
                "orientation": store["orientation"].astype(np.float64).reshape(nz, ny, nx),
            }
    data = np.loadtxt(path, comments="#")
    return {
        "meta": meta,
        "labels": data[:, 3].astype(np.int32).reshape(nz, ny, nx),
        "orientation": data[:, 7].reshape(nz, ny, nx),
    }


def categorical_color(label: int) -> tuple[float, float, float]:
    """Use the same bright categorical palette as the two-dimensional plate."""
    hue = (0.61803398875 * label) % 1.0
    saturation = 0.43 + 0.13 * (label % 3) / 2.0
    value = 0.88 + 0.08 * (label % 5) / 4.0
    return colorsys.hsv_to_rgb(hue, saturation, value)


def regularize_labels(labels: np.ndarray, passes: int = 2) -> np.ndarray:
    """Apply a local 3 x 3 x 3 modal reconstruction before surface extraction.

    This is a visualization-only analogue of the smoothing routinely applied
    when continuous grain surfaces are reconstructed from voxel maps. It
    removes isolated staircase voxels without interpolating new label values.
    """
    result = labels.copy()
    values = np.unique(labels)
    for _ in range(passes):
        padded = np.pad(result, 1, mode="edge")
        counts = np.zeros((len(values),) + result.shape, dtype=np.uint8)
        for dz in range(3):
            for dy in range(3):
                for dx in range(3):
                    neighborhood = padded[dz:dz + result.shape[0],
                                          dy:dy + result.shape[1],
                                          dx:dx + result.shape[2]]
                    for index, value in enumerate(values):
                        counts[index] += neighborhood == value
        result = values[np.argmax(counts, axis=0)]
    return result


KAPPA_MAX = 0.24


def binary_surface(mask: np.ndarray, spacing: float, smooth_iterations: int = 24,
                   gaussian_sigma: float = 0.75):
    padded = np.pad(mask.astype(np.uint8), 1, mode="constant")
    image = vtk.vtkImageData()
    image.SetDimensions(padded.shape[2], padded.shape[1], padded.shape[0])
    image.SetSpacing(spacing, spacing, spacing)
    image.SetOrigin(-spacing, -spacing, -spacing)
    scalars = numpy_to_vtk(padded.ravel(order="C"), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
    image.GetPointData().SetScalars(scalars)

    # Reconstruct the same kind of continuous geometry view used by GPK before
    # extracting an isosurface.  Contouring the raw binary voxels was the main
    # source of the faceted appearance in the previous figure.
    cast = vtk.vtkImageCast()
    cast.SetInputData(image)
    cast.SetOutputScalarTypeToFloat()
    cast.Update()
    gaussian = vtk.vtkImageGaussianSmooth()
    gaussian.SetInputConnection(cast.GetOutputPort())
    gaussian.SetDimensionality(3)
    gaussian.SetStandardDeviations(gaussian_sigma, gaussian_sigma, gaussian_sigma)
    gaussian.SetRadiusFactors(3.0, 3.0, 3.0)
    gaussian.Update()

    contour = vtk.vtkFlyingEdges3D()
    contour.SetInputConnection(gaussian.GetOutputPort())
    contour.SetValue(0, 0.50)
    contour.ComputeNormalsOff()
    contour.Update()

    smooth = vtk.vtkWindowedSincPolyDataFilter()
    smooth.SetInputConnection(contour.GetOutputPort())
    smooth.SetNumberOfIterations(smooth_iterations)
    smooth.SetPassBand(0.060)
    smooth.BoundarySmoothingOff()
    smooth.FeatureEdgeSmoothingOff()
    smooth.NonManifoldSmoothingOn()
    smooth.NormalizeCoordinatesOn()
    smooth.Update()

    normals = vtk.vtkPolyDataNormals()
    normals.SetInputConnection(smooth.GetOutputPort())
    normals.SetFeatureAngle(70.0)
    normals.ConsistencyOn()
    normals.AutoOrientNormalsOn()
    normals.SplittingOff()
    normals.Update()
    return normals.GetOutput()


def curvature_lut():
    lut = vtk.vtkLookupTable()
    lut.SetNumberOfTableValues(256)
    anchors = [
        (0.0, (0.050, 0.030, 0.528)),
        (0.35, (0.489, 0.010, 0.659)),
        (0.68, (0.928, 0.333, 0.251)),
        (1.0, (0.940, 0.975, 0.131)),
    ]
    for index in range(256):
        t = index / 255.0
        for (ta, ca), (tb, cb) in zip(anchors[:-1], anchors[1:]):
            if ta <= t <= tb:
                u = (t-ta)/(tb-ta)
                rgb = tuple((1-u)*a + u*b for a, b in zip(ca, cb))
                lut.SetTableValue(index, *rgb, 1.0)
                break
    lut.SetRange(0.0, KAPPA_MAX)
    lut.Build()
    return lut


CURVATURE_LUT = curvature_lut()


def curvature_polydata(polydata):
    if polydata is None or polydata.GetNumberOfPoints() == 0:
        return None
    curvatures = vtk.vtkCurvatures()
    curvatures.SetInputData(polydata)
    curvatures.SetCurvatureTypeToMean()
    curvatures.Update()
    attributes = vtk.vtkAttributeSmoothingFilter()
    attributes.SetInputConnection(curvatures.GetOutputPort())
    attributes.SetNumberOfIterations(8)
    attributes.SetRelaxationFactor(0.12)
    attributes.SetSmoothingStrategyToAllPoints()
    attributes.Update()
    output = vtk.vtkPolyData()
    output.DeepCopy(attributes.GetOutput())
    source = output.GetPointData().GetScalars()
    if source is None:
        return None
    values = np.abs(2.0 * vtk_to_numpy(source))  # VTK H -> summed |k1+k2|
    values = np.nan_to_num(values, nan=0.0, posinf=KAPPA_MAX, neginf=0.0)
    values = np.clip(values, 0.0, KAPPA_MAX)
    scalar = numpy_to_vtk(values.astype(np.float32), deep=True)
    scalar.SetName("CurvatureMagnitude")
    output.GetPointData().SetScalars(scalar)
    return output


def add_surface(renderer, polydata, color=None, opacity=1.0, show_curvature=True):
    mapper = vtk.vtkPolyDataMapper()
    if show_curvature:
        coloured = curvature_polydata(polydata)
        if coloured is None:
            return None
        mapper.SetInputData(coloured)
        mapper.SetLookupTable(CURVATURE_LUT)
        mapper.SetScalarRange(0.0, KAPPA_MAX)
        mapper.SetColorModeToMapScalars()
        mapper.ScalarVisibilityOn()
    else:
        mapper.SetInputData(polydata)
        mapper.ScalarVisibilityOff()
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    if color is not None and not show_curvature:
        actor.GetProperty().SetColor(*color)
    actor.GetProperty().SetOpacity(opacity)
    actor.GetProperty().SetInterpolationToPBR()
    actor.GetProperty().SetRoughness(0.52)
    actor.GetProperty().SetMetallic(0.0)
    renderer.AddActor(actor)
    return actor


def add_curvature_bar(renderer):
    bar = vtk.vtkScalarBarActor()
    bar.SetLookupTable(CURVATURE_LUT)
    # Draw the quantity label separately below the tick labels.  VTK's
    # horizontal scalar-bar title shares the tick-label baseline at large
    # publication fonts, which made the central 0.070 tick look doubled.
    bar.SetTitle("")
    bar.SetOrientationToHorizontal()
    bar.SetPosition(0.22, 0.170)
    bar.SetWidth(0.56)
    bar.SetHeight(0.075)
    bar.SetNumberOfLabels(3)
    bar.SetTextPositionToPrecedeScalarBar()
    for prop in (bar.GetTitleTextProperty(), bar.GetLabelTextProperty()):
        prop.SetFontFamilyToArial()
        prop.SetColor(0.10, 0.12, 0.15)
        prop.SetBold(False)
    bar.GetTitleTextProperty().SetFontSize(56)
    bar.GetLabelTextProperty().SetFontSize(52)
    bar.UnconstrainedFontSizeOn()
    renderer.AddViewProp(bar)


def add_outline(renderer, bounds, color=(0.28, 0.32, 0.37), opacity=0.55):
    cube = vtk.vtkCubeSource()
    cube.SetBounds(*bounds)
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(cube.GetOutputPort())
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetRepresentationToWireframe()
    actor.GetProperty().SetLineWidth(1.2)
    actor.GetProperty().SetColor(*color)
    actor.GetProperty().SetOpacity(opacity)
    renderer.AddActor(actor)


def add_text(renderer, text, x, y, size, bold=False, color=(0.08, 0.10, 0.12)):
    actor = vtk.vtkTextActor()
    actor.SetInput(text)
    actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
    actor.SetPosition(x, y)
    prop = actor.GetTextProperty()
    prop.SetFontFamilyToArial()
    prop.SetFontSize(size)
    prop.SetColor(*color)
    prop.SetBold(bold)
    prop.SetJustificationToLeft()
    prop.SetVerticalJustificationToTop()
    # VTK 9.7 removed the legacy AddActor2D alias; AddViewProp accepts the
    # same vtkTextActor and remains compatible with older VTK releases.
    renderer.AddViewProp(actor)


def configure_renderer(renderer, bounds, title, panel, state, annotation, show_bar):
    renderer.SetBackground(1.0, 1.0, 1.0)
    renderer.SetUseDepthPeeling(True)
    renderer.SetMaximumNumberOfPeels(100)
    renderer.SetOcclusionRatio(0.05)

    add_outline(renderer, bounds)
    if panel:
        add_text(renderer, panel, 0.025, 0.975, 82, bold=True)
        add_text(renderer, title, 0.115, 0.975, 68, bold=True)
    add_text(renderer, state, 0.75, 0.975, 58, bold=True, color=(0.24, 0.28, 0.33))
    add_text(renderer, annotation, 0.035, 0.078, 54, color=(0.18, 0.22, 0.27))
    if show_bar:
        add_text(renderer, "|kappa| dx", 0.385, 0.125, 58, color=(0.10, 0.12, 0.15))
        add_curvature_bar(renderer)

    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    center = np.array([(xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2])
    span = max(xmax - xmin, ymax - ymin, zmax - zmin)
    camera = renderer.GetActiveCamera()
    camera.SetFocalPoint(*center)
    camera.SetPosition(*(center + span * np.array([1.65, -2.05, 1.45])))
    camera.SetViewUp(0.0, 0.0, 1.0)
    camera.ParallelProjectionOn()
    camera_scale = 0.90 if title == "Elastic coupling" else 0.74
    camera.SetParallelScale(camera_scale * span)

    key = vtk.vtkLight()
    key.SetLightTypeToSceneLight()
    key.SetPosition(*(center + span * np.array([1.7, -1.8, 2.4])))
    key.SetFocalPoint(*center)
    key.SetIntensity(0.85)
    renderer.AddLight(key)
    fill = vtk.vtkLight()
    fill.SetLightTypeToSceneLight()
    fill.SetPosition(*(center + span * np.array([-1.2, 1.0, 0.8])))
    fill.SetFocalPoint(*center)
    fill.SetIntensity(0.45)
    renderer.AddLight(fill)


def regularize_face_labels(face: np.ndarray, passes: int = 2,
                           radius: int = 2) -> np.ndarray:
    """Suppress display-scale islands on an exterior categorical section.

    The operation is confined to the rendered specimen skin.  A 5 x 5 modal
    neighborhood removes isolated surface colour flecks while keeping the
    underlying three-dimensional labels and measurements untouched.
    """
    result = face.copy()
    values = np.unique(result)
    for _ in range(passes):
        padded = np.pad(result, radius, mode="edge")
        counts = np.zeros((len(values),) + result.shape, dtype=np.uint8)
        for dy in range(2 * radius + 1):
            for dx in range(2 * radius + 1):
                neighbour = padded[dy:dy + result.shape[0],
                                   dx:dx + result.shape[1]]
                for index, value in enumerate(values):
                    counts[index] += neighbour == value
        result = values[np.argmax(counts, axis=0)]
    return result


def add_exterior_label_faces(renderer, labels: np.ndarray):
    """Render the three camera-facing exterior grain maps without cutaways.

    Independent closed isosurfaces for every grain can turn small components
    touching the specimen boundary into apparent pits.  Exterior categorical
    faces show the simulated surface ownership directly and are the usual
    full-volume presentation for dense polycrystals.
    """
    nz, ny, nx = labels.shape
    points = vtk.vtkPoints()
    quads = vtk.vtkCellArray()
    rgba = vtk.vtkUnsignedCharArray()
    rgba.SetName("grain_rgba")
    rgba.SetNumberOfComponents(4)

    def cell(vertices, label):
        ids = []
        for vertex in vertices:
            ids.append(points.InsertNextPoint(*vertex))
        quad = vtk.vtkQuad()
        for i, point_id in enumerate(ids):
            quad.GetPointIds().SetId(i, point_id)
        quads.InsertNextCell(quad)
        rgb = categorical_color(int(label))
        rgba.InsertNextTuple4(*(int(255 * channel) for channel in rgb), 255)

    # Top (z=max), front (y=min) and right (x=max): exactly the faces visible
    # from the shared camera. A tiny offset prevents coincident-edge flicker.
    eps = 0.015
    top = regularize_face_labels(labels[-1])
    front = regularize_face_labels(labels[:, 0, :])
    right = regularize_face_labels(labels[:, :, -1])
    for y in range(ny):
        for x in range(nx):
            cell(((x, y, nz + eps), (x + 1, y, nz + eps),
                  (x + 1, y + 1, nz + eps), (x, y + 1, nz + eps)), top[y, x])
    for z in range(nz):
        for x in range(nx):
            cell(((x, -eps, z), (x + 1, -eps, z),
                  (x + 1, -eps, z + 1), (x, -eps, z + 1)), front[z, x])
    for z in range(nz):
        for y in range(ny):
            cell(((nx + eps, y, z), (nx + eps, y + 1, z),
                  (nx + eps, y + 1, z + 1), (nx + eps, y, z + 1)), right[z, y])

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetPolys(quads)
    polydata.GetCellData().SetScalars(rgba)
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)
    mapper.SetScalarModeToUseCellData()
    mapper.SetColorModeToDirectScalars()
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetInterpolationToPBR()
    actor.GetProperty().SetRoughness(0.64)
    actor.GetProperty().SetMetallic(0.0)
    renderer.AddActor(actor)

    # Reconstruct only true label transitions as continuous 0.5 contours of
    # the projected face indicators.  Drawing every changed pixel edge made a
    # smooth physical boundary look like a staircase after panel reduction.
    appended = vtk.vtkAppendPolyData()

    def add_face_contours(face, plane, fixed):
        height, width = face.shape
        for label in np.unique(face):
            indicator = (face == label).astype(np.float32)
            image = vtk.vtkImageData()
            image.SetDimensions(width, height, 1)
            image.SetSpacing(1.0, 1.0, 1.0)
            image.SetOrigin(0.5, 0.5, 0.0)
            scalar = numpy_to_vtk(indicator.ravel(order="C"), deep=True,
                                  array_type=vtk.VTK_FLOAT)
            image.GetPointData().SetScalars(scalar)
            gaussian = vtk.vtkImageGaussianSmooth()
            gaussian.SetInputData(image)
            gaussian.SetDimensionality(2)
            gaussian.SetStandardDeviations(0.90, 0.90, 0.0)
            gaussian.SetRadiusFactors(3.0, 3.0, 0.0)
            contour = vtk.vtkMarchingSquares()
            contour.SetInputConnection(gaussian.GetOutputPort())
            contour.SetValue(0, 0.5)
            contour.Update()
            source = contour.GetOutput()
            if source.GetNumberOfPoints() == 0:
                continue
            transformed = vtk.vtkPolyData()
            transformed.DeepCopy(source)
            points = transformed.GetPoints()
            for point_id in range(points.GetNumberOfPoints()):
                u, v, _ = points.GetPoint(point_id)
                if plane == "top":
                    point = (u, v, fixed)
                elif plane == "front":
                    point = (u, fixed, v)
                else:
                    point = (fixed, u, v)
                points.SetPoint(point_id, *point)
            appended.AddInputData(transformed)

    add_face_contours(top, "top", nz + 2.0 * eps)
    add_face_contours(front, "front", -2.0 * eps)
    add_face_contours(right, "right", nx + 2.0 * eps)
    appended.Update()
    clean = vtk.vtkCleanPolyData()
    clean.SetInputConnection(appended.GetOutputPort())
    clean.SetTolerance(0.01)
    clean.Update()
    boundaries = clean.GetOutput()
    boundary_mapper = vtk.vtkPolyDataMapper()
    boundary_mapper.SetInputData(boundaries)
    boundary_mapper.ScalarVisibilityOff()
    boundary_actor = vtk.vtkActor()
    boundary_actor.SetMapper(boundary_mapper)
    boundary_actor.GetProperty().SetColor(0.035, 0.045, 0.055)
    boundary_actor.GetProperty().SetOpacity(1.0)
    boundary_actor.GetProperty().SetLineWidth(2.35)
    boundary_actor.GetProperty().LightingOff()
    renderer.AddActor(boundary_actor)


def render_polycrystal(renderer, snap, show_curvature=True):
    # A dense, opaque polycrystal is presented as one complete specimen, not as
    # dozens of independently closed and overlapping grain objects.  The latter
    # composition can make a legitimate boundary intersection look like a
    # grain embedded in another grain.  Exterior ownership and its continuous
    # transition traces give an unambiguous literature-style full-volume view.
    add_exterior_label_faces(renderer, snap["labels"])


def render_solidification(renderer, snap):
    solid = snap["labels"] != 0
    add_surface(renderer, binary_surface(solid, 1.0, 16), (0.94, 0.47, 0.05))


def render_recrystallisation(renderer, snap, show_curvature=True):
    labels = snap["labels"]
    for label in np.unique(labels):
        if label == 0:
            continue
        mask = labels == label
        if np.count_nonzero(mask) < 8:
            continue
        add_surface(renderer, binary_surface(mask, 1.0, 30), categorical_color(int(label)),
                    show_curvature=show_curvature)
    if np.mean(labels != 0) > 0.95:
        add_exterior_label_faces(renderer, labels)


def render_multiphase(renderer, snap, show_curvature=True):
    labels = snap["labels"]
    phase_colors = {1: (0.94, 0.47, 0.05), 2: (0.16, 0.45, 0.73)}
    for label, color in phase_colors.items():
        mask = labels == label
        if np.count_nonzero(mask) >= 8:
            add_surface(renderer, binary_surface(mask, 1.0, 30), color,
                        show_curvature=show_curvature)


def render_elastic(renderer, snap, show_curvature=True):
    transformed = snap["labels"] == 1
    add_surface(renderer, binary_surface(transformed, 1.0, 30), (0.45, 0.30, 0.68),
                show_curvature=show_curvature)


def snapshot(dirname: str, index: int) -> dict:
    files = files_for(dirname)
    if not files:
        raise FileNotFoundError(dirname)
    return read_snapshot(files[index])


def files_for(dirname: str):
    """Return the most recently generated complete snapshot series.

    Both in-tree and build-tree output roots are supported. Selecting by the
    newest endpoint prevents an older duplicate directory from silently taking
    precedence after a production rerun.
    """
    candidates = []
    for root in DATA_ROOTS:
        files = sorted((root / dirname).glob("snap_*.csv"))
        if files:
            candidates.append(files)
    if not candidates:
        return []
    return max(candidates, key=lambda files: files[-1].stat().st_mtime)


def metric(snap, kind: str) -> str:
    labels = snap["labels"]
    time = snap["meta"].get("time", 0.0)
    if kind == "grain":
        value = f"grains = {len(np.unique(labels))}"
    elif kind == "rex":
        value = f"recrystallised = {np.mean(labels != 0):.3f}"
    elif kind == "multiphase":
        value = f"phase 1 = {np.mean(labels == 1):.3f}; phase 2 = {np.mean(labels == 2):.3f}"
    else:
        value = f"transformed = {np.mean(labels == 1):.3f}"
    # Domain sizes are stated in the adjacent manuscript text.  A compact,
    # single-line metric remains legible after the five-panel plate is scaled.
    return f"{value} | time = {time:.3g} s"


def main():
    cases = [
        ("a", "Isotropic grain growth", "manuscript_smallgrain_connected/coarse_isotropic_3d",
         "manuscript_smallgrain_connected/fine_isotropic_3d", "grain", render_polycrystal),
        ("b", "Anisotropic grain growth", "manuscript_smallgrain_connected/coarse_anisotropic_3d",
         "manuscript_smallgrain_connected/fine_anisotropic_3d", "grain", render_polycrystal),
        ("c", "Recrystallisation", "fig8_large/recrystallisation_3d",
         "fig8_large/recrystallisation_3d", "rex", render_recrystallisation),
        ("d", "Multiphase growth", "fig8_large/multiphase_3d",
         "fig8_large/multiphase_3d", "multiphase", render_multiphase),
        ("e", "Elastic-coupled dendritic growth", "fig8_large/dendrite_3d_512",
         "fig8_large/dendrite_3d_512", "elastic", render_elastic),
    ]
    # The dendrite is shown at the last saved state before its arms reach the
    # domain faces; GPK_DENDRITE_INDEX selects that snapshot.
    final_index = {"e": int(os.environ.get("GPK_DENDRITE_INDEX", "-1"))}

    panel_width, panel_height = 2100, 1120
    regular_font, bold_font = publication_fonts()
    canvas = Image.new("RGB", (2 * panel_width, 5 * panel_height), "#FFFFFF")
    for row, (panel, title, initial_dir, final_dir, kind, render_fn) in enumerate(cases):
        initial_files = files_for(initial_dir)
        final_files = files_for(final_dir)
        snaps = (read_snapshot(initial_files[0]), read_snapshot(final_files[final_index.get(panel, -1)]))
        for col, (snap, state) in enumerate(zip(snaps, ("initial", "final"))):
            plotter = pv.Plotter(off_screen=True,
                                 window_size=(panel_width, panel_height),
                                 border=True, border_color="#D7DCE2")
            plotter.set_background("#FFFFFF")
            try:
                plotter.enable_anti_aliasing("ssaa")
            except Exception:
                pass
            nz, ny, nx = snap["labels"].shape
            bounds = (0, nx, 0, ny, 0, nz)
            floor = pv.Plane(center=(0.5 * nx, 0.5 * ny, -0.055 * nz),
                             direction=(0.0, 0.0, 1.0),
                             i_size=1.16 * nx, j_size=1.16 * ny,
                             i_resolution=1, j_resolution=1)
            plotter.add_mesh(floor, color="#E6EBF0", opacity=0.28,
                             pbr=True, metallic=0.0, roughness=0.72,
                             show_scalar_bar=False)
            _render_raw_volume(plotter, snap["labels"], kind)
            plotter.add_mesh(pv.Box(bounds=bounds).extract_all_edges(),
                             color="#52606D", opacity=0.48, line_width=1.0,
                             lighting=False)
            configure_3d_camera(plotter, bounds, azimuth=-52.0,
                                elevation=27.0, zoom=1.06)
            plotter.add_light(pv.Light(
                position=(-0.80 * nx, -0.55 * ny, 2.55 * nz),
                focal_point=(0.5 * nx, 0.5 * ny, 0.5 * nz), intensity=0.72))
            plotter.add_light(pv.Light(
                position=(1.75 * nx, 1.35 * ny, 1.55 * nz),
                focal_point=(0.5 * nx, 0.5 * ny, 0.5 * nz), intensity=0.34))
            # Do not enable cast shadows on categorical specimen skins.  A
            # grain-coloured exterior is one contiguous box surface, so hard
            # shadow-map wedges can be mistaken for simulated grain patches.
            # The two scene lights and PBR roughness provide depth without
            # introducing additional categorical-looking regions.
            plotter.add_light(pv.Light(light_type="headlight", intensity=0.40))
            panel_path = RENDER_DIR / f"figure7_panel_{row}_{col}.png"
            plotter.show(screenshot=str(panel_path), auto_close=True)
            panel_image = Image.open(panel_path).convert("RGB")
            if panel_image.size != (panel_width, panel_height):
                panel_image = panel_image.resize((panel_width, panel_height), Image.Resampling.LANCZOS)
            # Composite typography after VTK rendering so the font family,
            # weight and rasterization match the DejaVu Sans used elsewhere.
            draw = ImageDraw.Draw(panel_image)
            if col == 0:
                draw.text((34, 18), f"{panel}  {title}", font=bold_font,
                          fill="#111827")
            state_label = state.capitalize()
            state_width = draw.textlength(state_label, font=regular_font)
            draw.text((panel_width - 34 - state_width, 22), state_label,
                      font=regular_font, fill="#4B5563")
            if kind == "elastic" and state == "final":
                _paste_section_inset(panel_image, snap["labels"], draw, regular_font)
            canvas.paste(panel_image, (col * panel_width, row * panel_height))
    image = canvas
    image.save(OUT_PNG, "PNG")
    image.save(OUT_PDF, "PDF", resolution=333.0, quality=95)
    print(f"Wrote {OUT_PDF}")


SECTION_HEIGHT = 0.383  # fraction of Nz; 30 um below the centre of the 512^3 box


def _paste_section_inset(panel_image, labels, draw, font) -> None:
    """Paste the raw horizontal label section through the dendrite arms.

    The inset is the stored label slice at z = SECTION_HEIGHT * Nz, drawn
    cell by cell (nearest neighbour), cropped to the crystal with a margin.
    """
    nz = labels.shape[0]
    section = labels[int(round(SECTION_HEIGHT * nz))] != 0
    ys, xs = np.nonzero(section)
    if ys.size == 0:
        return
    half = int(max(ys.max() - ys.min(), xs.max() - xs.min()) * 0.56) + 4
    cy, cx = (ys.max() + ys.min()) // 2, (xs.max() + xs.min()) // 2
    window = section[max(0, cy - half):cy + half, max(0, cx - half):cx + half]
    rgb = np.empty(window.shape + (3,), dtype=np.uint8)
    rgb[...] = (227, 227, 227)
    rgb[window] = (31, 79, 180)
    size = 400
    inset = Image.fromarray(rgb[::-1]).resize((size, size), Image.Resampling.NEAREST)
    x0, y0 = 40, panel_image.height - size - 70
    panel_image.paste(inset, (x0, y0))
    draw.rectangle((x0 - 1, y0 - 1, x0 + size, y0 + size), outline="#52606D", width=2)
    draw.text((x0, y0 + size + 10), "Section 30 µm below centre",
              font=font, fill="#4B5563")


def _raw_image(labels: np.ndarray) -> pv.ImageData:
    """Return one VTK cell for every stored 3D categorical label."""
    nz, ny, nx = labels.shape
    grid = pv.ImageData(dimensions=(nx + 1, ny + 1, nz + 1),
                        spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0))
    grid.cell_data["labels"] = labels.ravel(order="C").astype(np.int32)
    return grid


def _render_raw_polycrystal_skin(plotter: pv.Plotter,
                                 labels: np.ndarray) -> None:
    """Render the complete categorical exterior of a dense polycrystal.

    Rendering a sparse independent set of closed grains made a polycrystal
    resemble isolated particles.  The exterior skin instead retains every
    saved label on the complete specimen and exposes the contiguous grain
    mosaic on the three camera-facing faces.  It is extracted directly from
    the raw VTK image: no modal reconstruction, contour, boundary line or
    interpolation is introduced.
    """
    surface = _raw_image(labels).extract_surface()
    surface.cell_data["grain_rgb"] = categorical_rgb(
        np.asarray(surface.cell_data["labels"], dtype=np.int64)
    )
    plotter.add_mesh(surface, scalars="grain_rgb", rgb=True,
                     show_edges=False, smooth_shading=False,
                     pbr=True, metallic=0.0, roughness=0.68,
                     show_scalar_bar=False)


def _raw_phase_surface(labels: np.ndarray, phase: int) -> pv.PolyData:
    locations = np.where(labels == phase)
    if locations[0].size == 0:
        return pv.PolyData()
    # Restrict VTK extraction to the occupied bounding brick plus one raw-cell
    # guard layer.  This is exactly equivalent to thresholding the full image,
    # but avoids scanning an 8.4-million-cell production volume once for every
    # displayed grain.
    lower = [max(0, int(axis.min()) - 1) for axis in locations]
    upper = [min(labels.shape[i], int(locations[i].max()) + 2) for i in range(3)]
    subvolume = labels[lower[0]:upper[0], lower[1]:upper[1], lower[2]:upper[2]]
    nz, ny, nx = subvolume.shape
    grid = pv.ImageData(dimensions=(nx + 1, ny + 1, nz + 1),
                        spacing=(1.0, 1.0, 1.0),
                        origin=(float(lower[2]), float(lower[1]), float(lower[0])))
    grid.cell_data["phase"] = (subvolume.ravel(order="C") == phase).astype(np.uint8)
    selected = grid.threshold(0.5, scalars="phase", preference="cell")
    return selected.extract_surface().triangulate()


def _render_raw_volume(plotter: pv.Plotter, labels: np.ndarray, kind: str) -> None:
    """Render raw VTK image cells with no contour or line overlay."""
    if kind == "grain":
        _render_raw_polycrystal_skin(plotter, labels)
        return
    if kind == "rex":
        # At nucleation, show every recrystallised identity inside the parent
        # volume.  Once impingement is complete, show the complete categorical
        # specimen exterior.  The former communicates the actual 64-nucleus
        # starting population; the latter exposes the connected final grain
        # network instead of reducing it to a few isolated particle-like
        # surfaces.
        if np.mean(labels != 0) > 0.95:
            _render_raw_polycrystal_skin(plotter, labels)
            return
        values = [int(value) for value in np.unique(labels) if int(value) != 0]
        for label in values:
            if np.count_nonzero(labels == label) < 8:
                continue
            colour = (0.78 * categorical_rgb(
                np.asarray([[label]], dtype=np.int64))[0, 0] / 255.0)
            plotter.add_mesh(_raw_phase_surface(labels, int(label)), color=colour,
                             smooth_shading=True, pbr=True, metallic=0.02,
                             roughness=0.54, show_scalar_bar=False)
        return
    colours = {
        "multiphase": {1: "#C62020", 2: "#1F4FB4"},
        "elastic": {1: "#1F4FB4"},
    }[kind]
    if kind == "elastic" and np.mean(labels == 1) > 0.01:
        # Ambient occlusion darkens the grooves between side branches on the
        # final dendrite exterior.  It changes shading only, not geometry.
        try:
            plotter.enable_ssao(radius=6.0, bias=0.2, kernel_size=128)
        except Exception:
            pass
    for phase, colour in colours.items():
        surface = _raw_phase_surface(labels, phase)
        if surface.n_points:
            plotter.add_mesh(surface, color=colour, smooth_shading=True,
                             pbr=True, metallic=0.02, roughness=0.54,
                             show_scalar_bar=False)


if __name__ == "__main__":
    main()
