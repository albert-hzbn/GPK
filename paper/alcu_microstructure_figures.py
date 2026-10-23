"""v3 copy for main-text Figure 13 and Supporting Information Figures S9 and S10,
with the primary/secondary palette of the v3 manuscript.  The legacy main
figure function is not called.
"""
"""Publication figures from the audited Al--Cu GPK precipitation runs."""
from pathlib import Path
import csv

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=["#1F4FB4", "#D62020", "#7B2FA8",
                                                     "#E0A100", "#444444", "#7EA6E0"])
YELLOWRED = mpl.colors.LinearSegmentedColormap.from_list(
    "yellowred", [(0.00, "#FFFBE6"), (0.35, "#FFD966"), (0.70, "#F07A20"), (1.00, "#B01010")])
from matplotlib.colors import ListedColormap, TwoSlopeNorm
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import to_rgb
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from gpk_data import output_dir
try:
    from scipy.ndimage import gaussian_filter, label as connected_components
    from scipy.spatial import ConvexHull
except ImportError:  # si_figure uses only saved arrays and Matplotlib
    gaussian_filter = None
    connected_components = None
    ConvexHull = None
try:
    from skimage.measure import marching_cubes
except ImportError:  # pre-rendered PyVista panels do not require scikit-image
    marching_cubes = None
try:
    from run_alcu_gpk_microstructure import periodic_components
except ImportError:  # permits regeneration of the self-contained SI audit
    periodic_components = None


if periodic_components is None:
    def periodic_components(mask):
        """Periodic face-connected components without a SciPy dependency."""
        mask = np.asarray(mask, dtype=bool)
        labels = np.zeros(mask.shape, dtype=np.int32)
        component = 0
        for seed in np.argwhere(mask & (labels == 0)):
            seed_tuple = tuple(int(v) for v in seed)
            if labels[seed_tuple] != 0:
                continue
            component += 1
            labels[seed_tuple] = component
            stack = [seed_tuple]
            while stack:
                point = stack.pop()
                for axis, size in enumerate(mask.shape):
                    for offset in (-1, 1):
                        neighbour = list(point)
                        neighbour[axis] = (neighbour[axis] + offset) % size
                        neighbour = tuple(neighbour)
                        if mask[neighbour] and labels[neighbour] == 0:
                            labels[neighbour] = component
                            stack.append(neighbour)
        return labels, component


ROOT = Path(__file__).resolve().parents[1]
DATA = output_dir("alcu_validation") / "precipitation"
POP = output_dir("alcu_validation") / "realistic_population_3d"
FIG = ROOT / "paper"
PV_ASSETS = FIG / "figures" / "pyvista_panels"
COLORS = ["#1F4FB4", "#D62020", "#7B2FA8"]
MATRIX_COLOR = "#CFE0F7"

mpl.rcParams.update({
    "font.family": "Arial", "font.size": 9.2,
    "axes.labelsize": 9.2, "axes.titlesize": 9.8,
    "xtick.labelsize": 8.3, "ytick.labelsize": 8.3,
    "legend.fontsize": 8.0, "axes.linewidth": 0.7,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

SI_TEXT_BOX = dict(boxstyle="round,pad=0.18", fc="white", ec="#CBD5E1",
                   lw=0.55, alpha=0.97)
SI_PANEL_BOX = dict(boxstyle="square,pad=0.18", fc="white", ec="#CBD5E1",
                    lw=0.55, alpha=0.97)
SI_LEGEND = dict(frameon=True, facecolor="white", edgecolor="#CBD5E1",
                 framealpha=0.97, fancybox=False)


def load(case, index):
    return dict(np.load(DATA / case / f"snap_{index:03d}.npz"))


def metrics(case):
    with (DATA / case / "metrics.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {k: np.asarray([float(r[k]) for r in rows]) for k in rows[0]}


def resolved_aspects(case):
    with (DATA / case / "particle_statistics.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return np.asarray([float(r["aspect_ratio"]) for r in rows
                       if float(r["minor_dimension_nm"]) >= 2.0*float(load(case, 3)["dx_m"])*1e9])


def shape_history(case):
    times, medians = [], []
    for index in range(4):
        snap = load(case, index)
        labels = snap["labels"]
        dx_nm = float(snap["dx_m"])*1e9
        aspects = []
        for variant in (1, 2, 3):
            cc, ncc = periodic_components(labels == variant)
            for component in range(1, ncc+1):
                xyz = np.argwhere(cc == component)
                if len(xyz) < 5:
                    continue
                weights = snap["alpha"][cc == component]
                centered = np.empty_like(xyz, dtype=float)
                for axis, size in enumerate(labels.shape):
                    angle = 2.0*np.pi*xyz[:, axis]/size
                    mean_angle = np.arctan2(np.sum(weights*np.sin(angle)),
                                            np.sum(weights*np.cos(angle)))
                    centre = (mean_angle % (2.0*np.pi))*size/(2.0*np.pi)
                    centered[:, axis] = ((xyz[:, axis]-centre+0.5*size) % size)-0.5*size
                cov = (centered*weights[:, None]).T@centered/max(float(np.sum(weights)), 1e-30)
                ev = np.maximum(np.linalg.eigvalsh(np.atleast_2d(cov)), 1e-12)
                minor_nm = 4.0*np.sqrt(ev[0])*dx_nm
                if minor_nm >= 2.0*dx_nm:
                    aspects.append(float(np.sqrt(ev[-1]/ev[0])))
        times.append(float(snap["time_s"])/1000)
        medians.append(float(np.median(aspects)))
    return np.asarray(times), np.asarray(medians)


def micro2d(ax, snap, panel, label):
    dx = float(snap["dx_m"])*1e9
    display_sigma = 0.75/dx  # display-only support; raw statistics are unchanged
    shape = snap["labels"].shape
    extent = [0, shape[1]*dx, 0, shape[0]*dx]
    im = ax.imshow(np.zeros(shape), origin="lower", extent=extent,
                   cmap=ListedColormap([MATRIX_COLOR]), vmin=0, vmax=1,
                   interpolation="nearest")
    fields = np.asarray([(snap["labels"] == v).astype(float) for v in (1, 2, 3)])
    scores = np.asarray([gaussian_filter(fields[v-1], display_sigma, mode="nearest")
                         for v in (1, 2, 3)])
    overlay = np.argmax(scores, axis=0)+1
    overlay = np.ma.masked_where(np.max(scores, axis=0) < 0.45, overlay)
    ax.imshow(overlay, origin="lower", extent=extent,
              cmap=ListedColormap(COLORS), vmin=1, vmax=3, interpolation="nearest", alpha=0.92)
    # The 0.5 contour is the sharp interface; no diffuse thickness is implied.
    ax.contour(gaussian_filter(np.sum(fields, axis=0), display_sigma, mode="nearest"),
               levels=[0.45], origin="lower", extent=extent,
               colors="white", linewidths=0.55)
    ax.plot([6, 31], [7, 7], color="black", lw=2.2, solid_capstyle="butt")
    ax.text(18.5, 10, "25 nm", ha="center", va="bottom", fontsize=7)
    ax.set(xlabel="x (nm)", ylabel="y (nm)", title=label)
    ax.text(-0.15, 1.04, panel, transform=ax.transAxes, weight="bold", fontsize=10)
    return im


def categorical_surface(mask: np.ndarray, spacing: float,
                        smoothing_steps: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Triangulate exposed categorical voxel faces and relax display vertices."""
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    quads = []
    corner_signs = np.asarray([[-0.5, -0.5], [0.5, -0.5],
                               [0.5, 0.5], [-0.5, 0.5]])
    for axis in range(3):
        others = [a for a in range(3) if a != axis]
        for direction in (-1, 1):
            shifted = np.roll(padded, -direction, axis=axis)
            exposed = padded & ~shifted
            points = np.argwhere(exposed)-1
            for point in points:
                quad = np.tile(point.astype(float), (4, 1))
                quad[:, axis] += 0.5*direction
                quad[:, others[0]] += corner_signs[:, 0]
                quad[:, others[1]] += corner_signs[:, 1]
                if direction < 0:
                    quad = quad[[0, 3, 2, 1]]
                quads.append(quad)
    if not quads:
        return np.empty((0, 3)), np.empty((0, 3), dtype=int)
    quads = np.asarray(quads)
    vertices, inverse = np.unique(quads.reshape(-1, 3), axis=0,
                                  return_inverse=True)
    q = inverse.reshape(-1, 4)
    faces = np.vstack((q[:, [0, 1, 2]], q[:, [0, 2, 3]])).astype(int)
    edges = np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    for _ in range(smoothing_steps):
        sums = np.zeros_like(vertices)
        counts = np.zeros(len(vertices), dtype=float)
        np.add.at(sums, edges[:, 0], vertices[edges[:, 1]])
        np.add.at(sums, edges[:, 1], vertices[edges[:, 0]])
        np.add.at(counts, edges[:, 0], 1.0)
        np.add.at(counts, edges[:, 1], 1.0)
        mean = sums/np.maximum(counts[:, None], 1.0)
        vertices = 0.65*vertices+0.35*mean
    return vertices*spacing, faces


def periodic_unwrapped_surface(mask: np.ndarray, spacing: float):
    """Return one continuous minimum-image surface and its periodic centre."""
    if marching_cubes is None:
        raise RuntimeError("scikit-image is required only for legacy Matplotlib 3D rendering")
    coords = np.argwhere(mask)
    size = np.asarray(mask.shape, dtype=int)
    centre = np.empty(3, dtype=float)
    for axis, n in enumerate(size):
        angle = 2.0*np.pi*coords[:, axis]/n
        mean_angle = np.arctan2(np.sum(np.sin(angle)), np.sum(np.cos(angle)))
        centre[axis] = (mean_angle % (2.0*np.pi))*n/(2.0*np.pi)
    anchor = np.rint(centre).astype(int) % size
    delta = (coords-anchor+size//2) % size-size//2
    lower, upper = delta.min(axis=0), delta.max(axis=0)
    padding = 4
    local_shape = tuple((upper-lower+1+2*padding).tolist())
    local = np.zeros(local_shape, dtype=float)
    local_coords = delta-lower+padding
    local[tuple(local_coords.T)] = 1.0
    # Suppress grid-scale teeth in the display surface; all quantitative
    # morphology continues to be measured from the unsmoothed label mask.
    # Preserve the broad faces of physically thin theta-prime plates.  A
    # two-cell display blur can suppress the centre of a four-cell-thick slab
    # and falsely render a cavity even when the categorical particle is solid.
    smooth = gaussian_filter(local, 1.00, mode="constant")
    vertices, faces, _, _ = marching_cubes(
        smooth, level=0.45, spacing=(1.0, 1.0, 1.0),
        allow_degenerate=False)
    vertices += lower-padding
    return vertices*spacing, faces, anchor.astype(float)


def render3d(snap):
    labels = snap["labels"]
    dx = float(snap["dx_m"])*1e9
    fig = plt.Figure(figsize=(5.0, 4.05), dpi=210, facecolor="white")
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111, projection="3d")
    L = np.asarray(labels.shape)*dx
    x0, y0, z0 = 0.0, 0.0, 0.0
    x1, y1, z1 = L
    matrix_faces = [
        [(x0,y0,z0),(x1,y0,z0),(x1,y1,z0),(x0,y1,z0)],
        [(x0,y0,z1),(x1,y0,z1),(x1,y1,z1),(x0,y1,z1)],
        [(x0,y0,z0),(x1,y0,z0),(x1,y0,z1),(x0,y0,z1)],
        [(x0,y1,z0),(x1,y1,z0),(x1,y1,z1),(x0,y1,z1)],
        [(x0,y0,z0),(x0,y1,z0),(x0,y1,z1),(x0,y0,z1)],
        [(x1,y0,z0),(x1,y1,z0),(x1,y1,z1),(x1,y0,z1)],
    ]
    ax.add_collection3d(Poly3DCollection(
        matrix_faces, facecolors=MATRIX_COLOR, edgecolors="#7aa6b8",
        linewidths=0.35, alpha=0.075, antialiased=True))
    light = np.asarray([0.4, -0.6, 1.0]); light /= np.linalg.norm(light)
    surfaces = []
    for variant, color in zip((1, 2, 3), COLORS):
        # Render the same persistent label support used by the Gaussian
        # geometry audit. The sub-cell fraction remains a lineal interface
        # position and is not misrepresented as a diffuse phase field.
        raw = labels == variant
        if not np.any(raw):
            continue
        verts, faces, centre = periodic_unwrapped_surface(raw, dx)
        surfaces.append((variant, color, verts, faces, centre))

    # Place every full periodic object in one minimum-image ensemble.  Lattice
    # translations do not alter relative positions on the periodic torus.
    reference = surfaces[0][4]
    placed = []
    for variant, color, verts, faces, centre in surfaces:
        relative = (centre-reference+0.5*np.asarray(labels.shape)) % np.asarray(labels.shape) \
                   -0.5*np.asarray(labels.shape)
        placed.append((variant, color, verts+relative*dx, faces))
    all_vertices = np.vstack([item[2] for item in placed])
    translation = 0.5*L-0.5*(all_vertices.min(axis=0)+all_vertices.max(axis=0))

    for variant, color, verts, faces in placed:
        verts = verts+translation
        triangles = verts[faces]
        normal = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
        normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-30)
        intensity = 0.65+0.35*np.clip(normal@light, 0.0, 1.0)
        base = np.asarray(to_rgb(color))
        facecolors = np.clip(intensity[:, None]*base[None, :], 0.0, 1.0)
        mesh = Poly3DCollection(triangles, facecolors=facecolors, edgecolors="none",
                                linewidths=0.0, antialiased=True)
        ax.add_collection3d(mesh)
    corners = np.asarray([[x, y, z] for x in (0, L[0]) for y in (0, L[1]) for z in (0, L[2])])
    for i, p0 in enumerate(corners):
        for p1 in corners[i+1:]:
            if np.count_nonzero(p0 != p1) == 1:
                ax.plot(*zip(p0, p1), color="#777777", lw=0.55, alpha=0.38)
    ax.set(xlim=(0, L[0]), ylim=(0, L[1]), zlim=(0, L[2]))
    ax.set_box_aspect(L)
    ax.view_init(elev=24, azim=-55)
    ax.set_axis_off()
    fig.subplots_adjust(0, 0, 1, 1)
    canvas.draw()
    image = np.asarray(canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return image


def legacy_main_figure():
    """Three-dimensional Al--Cu result and transparent literature comparison."""
    d30, d33 = load("three_dimensional_ultrafine", 0), load("three_dimensional_ultrafine", 3)
    with (DATA / "three_dimensional_ultrafine" / "particle_statistics.csv").open(
            newline="", encoding="utf-8") as stream:
        particle_rows = list(csv.DictReader(stream))
    gpk_major = float(np.median([float(r["major_dimension_nm"]) for r in particle_rows]))
    gpk_minor = float(np.median([float(r["minor_dimension_nm"]) for r in particle_rows]))
    gpk_aspect = float(np.median([float(r["aspect_ratio"]) for r in particle_rows]))
    fig = plt.figure(figsize=(7.2, 5.8), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0])

    axa = fig.add_subplot(gs[0, 0])
    with (FIG / "calphad" / "al4cu_phase_fractions.csv").open(
            newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    temperature = np.asarray([float(r["temperature_K"]) for r in rows])
    fcc = np.asarray([float(r["stable_fcc"]) for r in rows])
    theta = np.asarray([float(r["stable_theta"]) for r in rows])
    axa.plot(temperature, fcc, color="#4c9ac6", label=r"$\alpha$-Al (FCC)")
    axa.plot(temperature, theta, color="#444444", label=r"$\theta$-Al$_2$Cu")
    axa.axvline(453.0, color="0.35", ls="--", lw=0.8)
    axa.set(xlabel="Temperature (K)", ylabel="Equilibrium phase fraction",
            ylim=(-0.02, 1.02), title="CALPHAD phase inventory")
    axa.legend(frameon=False, loc="center right", fontsize=6.8)
    axa.text(0.015, 0.985, "a", transform=axa.transAxes, weight="bold", fontsize=10,
             va="top", bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8})

    axb, axc = fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])
    axb.imshow(render3d(d30)); axc.imshow(render3d(d33))
    for ax, panel, title in ((axb, "b", "3D: initial"),
                             (axc, "c", f"3D: {float(d33['time_s']):.0f} s")):
        ax.set_axis_off(); ax.set_title(title, pad=2)
        ax.text(-0.03, 1.02, panel, transform=ax.transAxes, weight="bold", fontsize=10)

    axd = fig.add_subplot(gs[1, 0])
    t3, ar3 = shape_history("three_dimensional_ultrafine")
    keep3 = t3 <= float(d33["time_s"])/1000+1.0e-12
    t3, ar3 = t3[keep3], ar3[keep3]
    axd.plot(1000*t3, ar3, "o-", color="#D62020", lw=1.5, ms=3.5,
             label="aspect ratio")
    axd.set(xlabel="time (s)", ylabel="median aspect ratio",
            title="GPK morphology history")
    axd.grid(alpha=0.20)
    axd2 = axd.twinx()
    m3 = metrics("three_dimensional_ultrafine")
    keep_m3 = m3["time_s"] <= float(d33["time_s"])+1.0e-9
    m3 = {key: values[keep_m3] for key, values in m3.items()}
    axd2.plot(m3["time_s"], 100*m3["precipitate_fraction"],
              color="#D62020", alpha=0.38, lw=1.0)
    axd2.set_ylabel(r"$\theta'$ fraction (\%)")
    axd.legend(frameon=False, loc="upper left", fontsize=6.6)
    axd.text(-0.17, 1.05, "d", transform=axd.transAxes, weight="bold", fontsize=10)

    # Rodríguez-Veiga et al. (2018), Table 1: homogeneous theta-prime at
    # 120 and 168 h, 453 K. These are mature populations, not equal-time data.
    axe = fig.add_subplot(gs[1, 1])
    diam = np.asarray([gpk_major, 142.5, 342.5])
    thick = np.asarray([gpk_minor, 4.9, 9.0])
    labels_lit = ["GPK\n500 s", "experiment\n120 h", "experiment\n168 h"]
    colors_lit = ["#D62020", "#1F4FB4", "#1F4FB4"]
    for x, y, label, color in zip(diam, thick, labels_lit, colors_lit):
        axe.scatter(x, y, s=34, color=color, edgecolor="white", linewidth=0.5, zorder=3)
        axe.annotate(label, (x, y), xytext=(4, 4), textcoords="offset points", fontsize=6.2)
    axe.set(xlabel="diameter / major dimension (nm)",
            ylabel="thickness / minor dimension (nm)", title="Size versus experiment")
    axe.set_xlim(0, 385); axe.set_ylim(4.2, 9.7)
    axe.grid(alpha=0.18)
    axe.text(0.015, 0.985, "e", transform=axe.transAxes, weight="bold", fontsize=10,
             va="top", bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8})

    axf = fig.add_subplot(gs[1, 2])
    names = ["GPK\n500 s", "Exp.\n120 h", "Exp.\n168 h"]
    aspect = [gpk_aspect, 28.0, 39.0]
    bars = axf.barh(np.arange(3), aspect, color=["#D62020", "#1F4FB4", "#1F4FB4"])
    axf.set_yticks(np.arange(3), names)
    axf.set_ylim(2.6, -0.75)
    axf.set(xlabel="aspect ratio", title="Shape comparison")
    axf.grid(axis="x", alpha=0.18)
    for bar, value in zip(bars, aspect):
        axf.text(value+0.7, bar.get_y()+bar.get_height()/2, f"{value:.2g}",
                 va="center", fontsize=6.5)
    axf.set_xlim(0, 46)
    axf.text(0.02, 0.95, "Phase field: anisotropic mobility gives\n~3$\\times$ the isotropic-mobility aspect ratio",
             transform=axf.transAxes, fontsize=5.8, va="top")
    axf.text(-0.17, 1.05, "f", transform=axf.transAxes, weight="bold", fontsize=10)

    fig.savefig(FIG / "fig11_al4cu_material_validation.pdf", bbox_inches="tight",
                pad_inches=0.18)
    plt.close(fig)


def _particle_faces(center, variant, diameter, thickness, n=12,
                    triangulate_caps=False):
    angle = np.linspace(0.0, 2.0*np.pi, n, endpoint=False)
    radius = 0.5*diameter
    if variant == 1:
        ring = np.column_stack((np.zeros(n), radius*np.cos(angle), radius*np.sin(angle)))
        normal = np.asarray([0.5*thickness, 0.0, 0.0])
    elif variant == 2:
        ring = np.column_stack((radius*np.cos(angle), np.zeros(n), radius*np.sin(angle)))
        normal = np.asarray([0.0, 0.5*thickness, 0.0])
    else:
        ring = np.column_stack((radius*np.cos(angle), radius*np.sin(angle), np.zeros(n)))
        normal = np.asarray([0.0, 0.0, 0.5*thickness])
    top, bottom = center+ring+normal, center+ring-normal
    if triangulate_caps:
        top_centre, bottom_centre = center+normal, center-normal
        faces = []
        for i in range(n):
            j = (i+1) % n
            faces.append(np.asarray([top_centre, top[i], top[j]]))
            faces.append(np.asarray([bottom_centre, bottom[j], bottom[i]]))
    else:
        faces = [top, bottom[::-1]]
    for i in range(n):
        faces.append(np.asarray([top[i], top[(i+1) % n],
                                 bottom[(i+1) % n], bottom[i]]))
    return faces


def _lit_face_color(face, base_color, light, view, particle_centre, diameter):
    """Lambert plus Blinn-Phong shading for a two-sided precipitate face."""
    face = np.asarray(face, dtype=float)
    normal = np.cross(face[1]-face[0], face[2]-face[0])
    length = np.linalg.norm(normal)
    if length <= 1.0e-12:
        return (*to_rgb(base_color), 0.96)
    normal /= length
    if np.dot(normal, view) < 0.0:
        normal = -normal
    diffuse = max(0.0, float(np.dot(normal, light)))
    half_vector = light+view
    half_vector /= np.linalg.norm(half_vector)
    specular = max(0.0, float(np.dot(normal, half_vector)))**18
    offset = (np.mean(face, axis=0)-particle_centre)/max(0.5*diameter, 1.0e-12)
    offset_norm = np.linalg.norm(offset)
    if offset_norm > 1.0e-12:
        offset /= offset_norm
    positional_gloss = 0.35+0.65*max(0.0, float(np.dot(offset, light)))**3
    base = np.asarray(to_rgb(base_color))
    sheen = 0.14*positional_gloss*(0.30+0.70*diffuse)
    rgb = np.clip(base*(0.36+0.64*diffuse)
                  +0.62*specular*positional_gloss+sheen, 0.0, 1.0)
    return (*rgb, 0.96)


def _projected_shadow(faces, light, domain, expansion=1.0):
    vertices = np.vstack(faces)
    projected = vertices[:, :2]-vertices[:, 2, None]*light[:2]/light[2]
    projected = np.clip(projected, 0.0, domain)
    try:
        polygon = projected[ConvexHull(projected).vertices]
    except Exception:
        return None
    centre = np.mean(polygon, axis=0)
    polygon = centre+expansion*(polygon-centre)
    polygon = np.clip(polygon, 0.0, domain)
    return np.column_stack((polygon, np.full(len(polygon), 0.8)))


def render_population(path):
    snap = dict(np.load(path))
    # Render the population independently at high native size, then place the
    # tightly cropped image in the publication layout.  A large square render
    # prevents the thin plates and rounded cap edges from being lost when the
    # composite figure is reduced to journal-column width.
    fig = plt.Figure(figsize=(5.8, 5.35), dpi=320, facecolor="white")
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111, projection="3d")
    domain = float(snap["domain_nm"])
    ax.set(xlim=(0, domain), ylim=(0, domain), zlim=(0, domain))
    ax.set_autoscale_on(False)
    light = np.asarray([-0.22, -0.30, 1.0]); light /= np.linalg.norm(light)
    elev, azim = 25.0, -56.0
    er, ar = np.deg2rad(elev), np.deg2rad(azim)
    view = np.asarray([np.cos(er)*np.cos(ar), np.cos(er)*np.sin(ar), np.sin(er)])

    # A pale receiving plane and two nested projected polygons produce a soft
    # cast shadow.  This is a display operation only; particle coordinates and
    # sizes remain those in the NPZ snapshot.
    floor = np.asarray([[0, 0, 0], [domain, 0, 0],
                        [domain, domain, 0], [0, domain, 0]])
    ax.add_collection3d(Poly3DCollection(
        [floor], facecolor="#f3f6f8", edgecolor="none", alpha=0.72))

    qfaces = []
    for centre, variant, diameter in zip(
            snap["theta_double_prime_position_nm"],
            snap["theta_double_prime_variant"],
            snap["theta_double_prime_diameter_nm"]):
        qfaces.append(_particle_faces(centre, int(variant), float(diameter), 0.0, n=10)[0])
    if qfaces:
        ax.add_collection3d(Poly3DCollection(
            qfaces, facecolor="#77c9ed", edgecolor="none",
            linewidth=0.0, alpha=0.065, zsort="average"))

    solid_faces, solid_colors = [], []
    shadow_inner, shadow_outer, shadow_alpha = [], [], []
    for centre, variant, diameter, thickness in zip(
            snap["theta_prime_position_nm"], snap["theta_prime_variant"],
            snap["theta_prime_diameter_nm"], snap["theta_prime_thickness_nm"]):
        faces = _particle_faces(centre, int(variant), float(diameter),
                                float(thickness), n=36, triangulate_caps=False)
        base = COLORS[int(variant)-1]
        solid_faces.extend(faces)
        solid_colors.extend(_lit_face_color(
            face, base, light, view, np.asarray(centre), float(diameter))
            for face in faces)
        inner = _projected_shadow(faces, light, domain, expansion=1.0)
        outer = _projected_shadow(faces, light, domain, expansion=1.08)
        if inner is not None and outer is not None:
            shadow_inner.append(inner); shadow_outer.append(outer)
            shadow_alpha.append(0.035+0.055*np.exp(-float(centre[2])/domain))
    if shadow_outer:
        ax.add_collection3d(Poly3DCollection(
            shadow_outer, facecolors=[(0.05, 0.08, 0.11, 0.022)]*len(shadow_outer),
            edgecolor="none", zsort="min"))
        ax.add_collection3d(Poly3DCollection(
            shadow_inner,
            facecolors=[(0.05, 0.08, 0.11, alpha) for alpha in shadow_alpha],
            edgecolor="none", zsort="min"))
    if solid_faces:
        ax.add_collection3d(Poly3DCollection(
            solid_faces, facecolors=solid_colors,
            edgecolors=[(*tuple(0.46*np.asarray(color[:3])), 0.55)
                        for color in solid_colors],
            linewidth=0.16, zsort="average"))
    corners = np.asarray([[x, y, z] for x in (0, domain) for y in (0, domain)
                          for z in (0, domain)])
    for i, p0 in enumerate(corners):
        for p1 in corners[i+1:]:
            if np.count_nonzero(p0 != p1) == 1:
                ax.plot(*zip(p0, p1), color="#617984", lw=0.62, alpha=0.42)
    ax.set(xlim=(0, domain), ylim=(0, domain), zlim=(0, domain))
    ax.set_box_aspect((1, 1, 1)); ax.view_init(elev=elev, azim=azim); ax.set_axis_off()
    fig.subplots_adjust(0, 0, 1, 1); canvas.draw()
    image = np.asarray(canvas.buffer_rgba())[..., :3].copy(); plt.close(fig)
    nonwhite = np.min(image, axis=2) < 247
    if np.any(nonwhite):
        yy, xx = np.where(nonwhite)
        pad = 8
        y0, y1 = max(0, yy.min()-pad), min(image.shape[0], yy.max()+pad+1)
        x0, x1 = max(0, xx.min()-pad), min(image.shape[1], xx.max()+pad+1)
        image = image[y0:y1, x0:x1]
    return image


def main_figure():
    """Realistic 3D population ageing and comparison with experiment."""
    import json
    with (POP / "population_metrics.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    col = lambda key: np.asarray([float(row[key]) for row in rows])
    t = col("time_h")
    report = json.loads((POP / "validation_report.json").read_text(encoding="utf-8"))
    end = report["primary_realization"]
    interval = report["ensemble_95_percent_interval"]
    envelope = report["structural_sensitivity_envelope"]
    # The microstructures are the primary evidence in this figure.  Give them
    # the full-width top row; the quantitative checks follow as paired rows.
    # This is substantially more legible in the final 0.98-textwidth placement
    # than the former six equal cells, which reduced each 3-D volume to one
    # third of the page width.
    fig = plt.figure(figsize=(7.2, 6.9), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.42, 1.0, 1.0])

    for slot, hour, panel in ((gs[0, 0], 120, "a"), (gs[0, 1], 168, "b")):
        ax = fig.add_subplot(slot)
        pv_panel = PV_ASSETS / f"alcu_population_{hour}h.png"
        if not pv_panel.exists():
            raise RuntimeError(f"Missing PyVista rendering: {pv_panel}")
        panel_image = plt.imread(pv_panel)
        # Remove only the empty raster surround introduced by the conservative
        # PyVista camera.  Retain a measured white safety margin so periodic
        # plates remain complete while the microstructure fills its panel.
        rgb = panel_image[..., :3]
        occupied = np.min(rgb, axis=2) < 0.985
        if np.any(occupied):
            yy, xx = np.where(occupied)
            margin = int(0.045 * max(np.ptp(yy) + 1, np.ptp(xx) + 1))
            y0 = max(0, int(yy.min()) - margin)
            y1 = min(panel_image.shape[0], int(yy.max()) + margin + 1)
            x0 = max(0, int(xx.min()) - margin)
            x1 = min(panel_image.shape[1], int(xx.max()) + margin + 1)
            panel_image = panel_image[y0:y1, x0:x1]
        ax.imshow(panel_image, interpolation="lanczos", resample=True)
        ax.set_axis_off(); ax.set_title(f"3D population: {hour} h", pad=3,
                                       fontsize=10.4)
        ax.text(0.02, 0.98, panel, transform=ax.transAxes, weight="bold",
                fontsize=10, va="top", ha="left",
                bbox=dict(boxstyle="square,pad=0.13", fc="white", ec="0.75", alpha=0.96))

    axc = fig.add_subplot(gs[1, 0])
    axc.plot(t, 100*col("theta_prime_fraction"), "o-", color="#D62020",
             label=r"$\theta'$ prediction")
    axc.plot(t, 100*col("theta_double_prime_fraction"), "s-", color="#7EA6E0",
             label=r"$\theta''$ conditioned")
    axc.set(xlabel="ageing time (h)", ylabel="phase fraction (%)",
            title="Cu-inventory phase transfer")
    axc.legend(frameon=False, fontsize=7.6); axc.grid(alpha=0.2)
    axc.text(0.02, 0.98, "c", transform=axc.transAxes, weight="bold", fontsize=10,
             va="top", bbox=dict(boxstyle="square,pad=0.13", fc="white", ec="0.75", alpha=0.96))

    axd = fig.add_subplot(gs[1, 1])
    axd.plot(t, col("theta_prime_count_per_um3"), "o-", color="#D62020",
             label=r"$\theta'$")
    axd.plot(t, col("theta_double_prime_count_per_um3"), "s-",
             color="#7EA6E0", label=r"$\theta''$")
    axd.set_yscale("symlog", linthresh=1.0)
    axd.set(xlabel="ageing time (h)", ylabel=r"number density ($\mu$m$^{-3}$)",
            title="Endpoint-conditioned counts")
    axd.legend(frameon=False); axd.grid(alpha=0.2)
    axd.text(0.02, 0.98, "d", transform=axd.transAxes, weight="bold", fontsize=10,
             va="top", bbox=dict(boxstyle="square,pad=0.13", fc="white", ec="0.75", alpha=0.96))

    axe = fig.add_subplot(gs[2, 0])
    names = ["diameter", "thickness", "fraction"]
    pred = np.asarray([end["theta_prime_mean_diameter_nm"],
                       end["theta_prime_mean_thickness_nm"],
                       end["theta_prime_fraction"]])
    exp = np.asarray([342.5, 9.0, 0.010]); exp_sd = np.asarray([46.8, 0.6, 0.005])
    lo = np.asarray([envelope["theta_prime_mean_diameter_nm"]["minimum"],
                     envelope["theta_prime_mean_thickness_nm"]["minimum"],
                     envelope["theta_prime_fraction"]["minimum"]])
    hi = np.asarray([envelope["theta_prime_mean_diameter_nm"]["maximum"],
                     envelope["theta_prime_mean_thickness_nm"]["maximum"],
                     envelope["theta_prime_fraction"]["maximum"]])
    x = np.arange(3); width = 0.34
    axe.bar(x-width/2, np.ones(3), width, color="#1F4FB4", label="experiment")
    axe.errorbar(x-width/2, np.ones(3), yerr=exp_sd/exp, fmt="none",
                 ecolor="black", capsize=2, lw=0.8)
    axe.bar(x+width/2, pred/exp, width, color="#D62020", label="conditional GPK")
    axe.errorbar(x+width/2, pred/exp, yerr=np.vstack(((pred-lo)/exp, (hi-pred)/exp)),
                 fmt="none", ecolor="black", capsize=2, lw=0.8)
    axe.axhline(1.0, color="0.3", lw=0.7)
    axe.set_xticks(x, ["diameter", "thickness", "fraction"])
    axe.set_ylim(0.35, 1.86)
    axe.set(ylabel="calculation / experiment", title="168 h comparison with experiment")
    axe.legend(frameon=False, fontsize=7.8); axe.grid(axis="y", alpha=0.2)
    labels = [f"diameter\n{pred[0]:.1f} nm",
              f"thickness\n{pred[1]:.2f} nm",
              f"fraction\n{100*pred[2]:.2f}%"]
    # Put the values in a second tick-label line.  This keeps every number away
    # from the error-bar stems and leaves the legend in clear upper whitespace.
    axe.set_xticks(x, labels)
    axe.tick_params(axis="x", labelsize=7.4, pad=3)
    axe.text(0.02, 0.98, "e", transform=axe.transAxes, weight="bold", fontsize=10,
             va="top", bbox=dict(boxstyle="square,pad=0.13", fc="white", ec="0.75", alpha=0.96))

    axf = fig.add_subplot(gs[2, 1])
    aspect_pred = float(end["theta_prime_mean_aspect_ratio"])
    aspect_lo = float(envelope["theta_prime_mean_aspect_ratio"]["minimum"])
    aspect_hi = float(envelope["theta_prime_mean_aspect_ratio"]["maximum"])
    aspect = [aspect_pred, 39.0]
    bars = axf.barh([0, 1], aspect, color=["#D62020", "#1F4FB4"])
    axf.errorbar(aspect_pred, 0,
                 xerr=np.asarray([[aspect_pred-aspect_lo], [aspect_hi-aspect_pred]]),
                 fmt="none", ecolor="black", capsize=2, lw=0.8)
    axf.errorbar(39.0, 1, xerr=6.6, fmt="none", ecolor="black", capsize=2, lw=0.8)
    axf.set_yticks([0, 1], ["conditional GPK", "experiment"]); axf.invert_yaxis()
    axf.set(xlabel="aspect ratio", title="168 h aspect ratio", xlim=(0, 52))
    axf.grid(axis="x", alpha=0.2)
    for bar, value in zip(bars, aspect):
        # Pull the values farther inside the bars so the horizontal uncertainty
        # bars and caps remain visually separate from the text.
        axf.text(max(4.0, value-11.0), bar.get_y()+bar.get_height()/2, f"{value:.1f}",
                 ha="right", va="center", color="white", weight="bold", fontsize=8.0)
    # The phase-field context is already stated in the caption.  Keeping it out
    # of the plotting area prevents the note from obscuring the experiment bar.
    axf.text(0.02, 0.98, "f", transform=axf.transAxes, weight="bold", fontsize=10,
             va="top", bbox=dict(boxstyle="square,pad=0.13", fc="white", ec="0.75", alpha=0.96))

    fig.savefig(FIG / "fig11_al4cu_material_validation.pdf", bbox_inches="tight",
                pad_inches=0.15, dpi=600)
    plt.close(fig)


def si_figure():
    d3 = load("three_dimensional_ultrafine", 3)
    m3 = metrics("three_dimensional_ultrafine")
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8), constrained_layout=True)
    z = int(np.argmax(np.count_nonzero(d3["labels"] > 0, axis=(0, 1))))
    dx3 = float(d3["dx_m"])*1e9
    extent3 = [0, d3["labels"].shape[1]*dx3, 0, d3["labels"].shape[0]*dx3]
    # Central sections drawn cell by cell from the saved fields.  The matrix
    # Cu fraction is defined only in the matrix, so plate cells are filled with
    # their variant colour (as in main-text Figure 13) instead of showing the
    # undefined values.  The elastic energy is defined everywhere; a thin line
    # marks the plate boundaries.
    lab = d3["labels"][:, :, z]
    variant = d3["variant_field"][:, :, z]
    plate = (lab > 0) | (d3["precip_fraction"][:, :, z] > 0.5)
    c_m = np.ma.masked_where(plate | ~np.isfinite(d3["c_matrix"][:, :, z]),
                             d3["c_matrix"][:, :, z])
    energy = d3["elastic_energy"][:, :, z] / 1e6
    variant_colours = {1: "#1F4FB4", 2: "#D62020", 3: "#7B2FA8"}
    plate_rgba = np.zeros(lab.shape + (4,))
    for v, colour in variant_colours.items():
        plate_rgba[plate & (variant == v)] = mpl.colors.to_rgba(colour)
    plate_rgba[plate & ~np.isin(variant, list(variant_colours))] = mpl.colors.to_rgba("#555555")
    blues = mpl.colormaps["Blues"].copy()
    im0 = axes[0, 0].imshow(c_m, origin="lower", extent=extent3, cmap=blues,
                            norm=mpl.colors.Normalize(0.004, 0.023), interpolation="nearest")
    axes[0, 0].imshow(plate_rgba, origin="lower", extent=extent3, interpolation="nearest")
    im1 = axes[0, 1].imshow(energy, origin="lower", extent=extent3, cmap=YELLOWRED,
                            interpolation="nearest")
    yy = (np.arange(lab.shape[0]) + 0.5) * dx3
    xx = (np.arange(lab.shape[1]) + 0.5) * dx3
    axes[0, 1].contour(xx, yy, plate.astype(float), levels=[0.5], colors="#222222",
                       linewidths=0.6)
    for j, (im, title, label) in enumerate((
            (im0, "Matrix Cu fraction", "matrix Cu fraction"),
            (im1, "Elastic energy (MJ m$^{-3}$)", "elastic energy (MJ m$^{-3}$)"))):
        ax = axes[0, j]
        ax.set_title(title); ax.set_xlabel("y (nm)")
        if j == 0:
            ax.set_ylabel("x (nm)")
        ax.set_aspect("equal")
        fig.colorbar(im, ax=ax, shrink=0.78)
        ax.text(0.025, 0.965, chr(97+j), transform=ax.transAxes,
                va="top", weight="bold", bbox=SI_PANEL_BOX)
    # A clipped logarithmic line at 1e-18 concealed the fact that the saved
    # Cu-balance diagnostic is exactly zero.  Show the non-trivial solute
    # redistribution instead and state the exact audit result explicitly.
    axes[1, 0].plot(m3["time_s"], 100.0*m3["mean_matrix_cu"],
                    color="#1F4FB4", lw=1.7, label="matrix Cu")
    axes[1, 0].plot(m3["time_s"], 100.0*m3["precipitate_fraction"],
                    color="#D62020", lw=1.7, label=r"$\theta'$ fraction")
    axes[1, 0].set(xlabel="time (s)", ylabel="fraction (%)",
                   title="Cu redistribution and balance")
    axes[1, 0].legend(**SI_LEGEND, loc="upper left", bbox_to_anchor=(0.08, 0.82))
    axes[1, 0].text(
        0.97, 0.96,
        rf"max $|\Delta C_\mathrm{{tot}}/C_0|$ = {np.max(np.abs(m3['relative_mass_error'])):g}" "\n"
        f"{len(m3['time_s'])} audited records",
        transform=axes[1, 0].transAxes, ha="right", va="top", fontsize=7.6,
        bbox=SI_TEXT_BOX)
    axes[1, 0].grid(alpha=0.22)
    axes[1, 0].text(0.025, 0.965, "c", transform=axes[1, 0].transAxes, va="top", weight="bold",
                    bbox=SI_PANEL_BOX)
    # Resolve the small but non-zero variation instead of spanning six empty
    # logarithmic decades up to the acceptance limit.
    residual = m3["mechanical_equilibrium_residual"]
    residual_scaled = 1.0e16*residual
    axes[1, 1].plot(m3["time_s"], residual_scaled, color="#1F4FB4", lw=1.7)
    padding = max(0.004, 0.18*(float(np.max(residual_scaled))-
                               float(np.min(residual_scaled))))
    axes[1, 1].set_ylim(float(np.min(residual_scaled))-padding,
                        float(np.max(residual_scaled))+padding)
    axes[1, 1].set(xlabel="time (s)",
                   ylabel=r"normalized residual ($10^{-16}$)",
                   title="Mechanical equilibrium residual")
    max_residual = float(np.max(residual))
    axes[1, 1].text(
        0.05, 0.06,
        f"max = {max_residual:.3e}\n"
        f"limit = $10^{{-10}}$\n"
        f"margin = {1.0e-10/max_residual:.2e}",
        transform=axes[1, 1].transAxes, ha="left", va="bottom", fontsize=7.6,
        bbox=SI_TEXT_BOX)
    axes[1, 1].grid(alpha=0.22)
    axes[1, 1].text(0.025, 0.965, "d", transform=axes[1, 1].transAxes, va="top", weight="bold",
                    bbox=SI_PANEL_BOX)
    fig.savefig(FIG / "figS11_alcu_fields_and_audits.pdf", bbox_inches="tight")
    plt.close(fig)


def mesh_convergence_figure():
    """Matched-domain 3D overlay and endpoint convergence metrics."""
    b3 = load("three_dimensional_mesh_fine", 3)
    h3 = load("three_dimensional_mesh_ultrafine", 3)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.75), constrained_layout=True)
    raw_sections = []
    for tag in ("96", "128"):
        panel_path = PV_ASSETS / f"alcu_mesh_{tag}.png"
        if not panel_path.exists():
            raise RuntimeError(f"Missing PyVista rendering: {panel_path}")
        raw_sections.append(plt.imread(panel_path))
    axes[0].imshow(np.concatenate(raw_sections, axis=1), interpolation="nearest")
    axes[0].axvline(raw_sections[0].shape[1] - 0.5, color="white", lw=1.0)
    axes[0].text(0.25, 0.97, r"96$^3$", transform=axes[0].transAxes,
                 ha="center", va="top", fontsize=8.0,
                 bbox=SI_TEXT_BOX)
    axes[0].text(0.75, 0.97, r"128$^3$", transform=axes[0].transAxes,
                 ha="center", va="top", fontsize=8.0,
                 bbox=SI_TEXT_BOX)
    axes[0].set_title(r"Raw VTK habit-normal sections", fontsize=9.0, pad=4)
    axes[0].set_xticks([]); axes[0].set_yticks([])

    def endpoint(case):
        m = metrics(case)
        return m["precipitate_fraction"][-1], shape_history(case)[1][-1]

    vf3b, ar3b = endpoint("three_dimensional_mesh_fine")
    vf3h, ar3h = endpoint("three_dimensional_mesh_ultrafine")
    x = np.arange(2)
    labels = [r"96$^3$", r"128$^3$"]
    axes[1].bar(x, 100*np.asarray([vf3b, vf3h]), color=["#7EA6E0", "#E0A100"])
    axes[1].set_xticks(x, labels)
    axes[1].set(ylabel="precipitate fraction (%)", title="3D volume fraction at 500 s")
    axes[1].set_ylim(0, 100*max(vf3b, vf3h)*1.16)
    rel = lambda a, b: 100.0*abs(b-a)/abs(a)
    axes[1].text(0.5, 100*max(vf3b, vf3h)*1.04,
                 f"{rel(vf3b, vf3h):.2f}% diff.", ha="center", fontsize=8.2,
                 bbox=SI_TEXT_BOX)

    axes[2].bar(x, [ar3b, ar3h], color=["#7EA6E0", "#E0A100"])
    axes[2].set_xticks(x, labels)
    axes[2].set(ylabel="resolved aspect ratio", title="3D shape metric at 500 s")
    ar_max = max(ar3b, ar3h)
    axes[2].set_ylim(0, ar_max*1.18)
    axes[2].text(0.5, ar_max*1.035,
                 f"{rel(ar3b, ar3h):.2f}% diff.", ha="center", fontsize=8.2,
                 bbox=SI_TEXT_BOX)
    for panel, ax in zip("abc", axes.flat):
        ax.text(-0.14, 1.04, panel, transform=ax.transAxes, weight="bold", fontsize=10)
        ax.grid(alpha=0.18)
    fig.savefig(FIG / "figS12_alcu_mesh_convergence.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    # A figure whose 3-D fields are not in the repository is skipped with a
    # message (see paper/LARGE_DATA.md); the others are still drawn.
    for figure in (main_figure, si_figure, mesh_convergence_figure):
        try:
            figure()
        except (FileNotFoundError, RuntimeError) as exc:
            print(f"skipped {figure.__name__}: input data missing ({exc}); see paper/LARGE_DATA.md")
