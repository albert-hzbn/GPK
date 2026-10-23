"""Generate manuscript-specific, publication-quality vector figures.

The figures in this module are deliberately composed for A4 manuscript width:
large type, restrained colour, light backgrounds, and no magnetic-model content.
"""

from __future__ import annotations

import importlib.util
import math
import os
import re
from collections import namedtuple
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import hsv_to_rgb, Normalize
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np
from gpk_data import output_dir
try:
    from scipy import stats
    from scipy.ndimage import gaussian_filter
except ImportError:
    _LinregressResult = namedtuple(
        "LinregressResult", "slope intercept rvalue pvalue stderr"
    )

    class _StatsFallback:
        @staticmethod
        def linregress(x, y):
            slope, intercept = np.polyfit(x, y, 1)
            rval = np.corrcoef(x, y)[0, 1]
            return _LinregressResult(slope, intercept, rval, np.nan, np.nan)
    stats = _StatsFallback()
    gaussian_filter = None


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
OUT = Path(os.environ.get("GPK_FIGURE_OUT", ROOT / "figures"))
OUT.mkdir(parents=True, exist_ok=True)
PV_ASSETS = ROOT / "figures" / "pyvista_panels"

# Primary/secondary palette shared by all figures generated here: blue, red,
# green and yellow, with purple and a light blue as secondary colours.
BLUE = "#1F4FB4"
CYAN = "#7EA6E0"      # light blue
ORANGE = "#E0A100"    # golden yellow (legible as a line on white)
RED = "#D62020"
GREEN = "#7B2FA8"
PURPLE = "#444444"
GRAY = "#555555"
LIGHT = "#EDEDED"
MATRIX = "#EF7B7B"
PRODUCT = "#7EA6E0"
# Area fills for phases and labelled regions (same as the schematics).
FILL_BLUE, FILL_YELLOW, FILL_GREEN = "#7EA6E0", "#FFD966", "#B08AD6"
FILL_RED, FILL_PURPLE, FILL_GREY = "#EF7B7B", "#9E9E9E", "#E3E3E3"
FILL_LIGHTBLUE, FILL_WHITE = "#CFE0F7", "#F5F5F5"
# Sequential scale for magnitudes (curvature, energy): yellow to red.
YELLOWRED = matplotlib.colors.LinearSegmentedColormap.from_list(
    "yellowred", [(0.00, "#FFFBE6"), (0.35, "#FFD966"), (0.70, "#F07A20"), (1.00, "#B01010")])

# SI annotations drawn over axes use one neutral, nearly opaque treatment.
# This avoids transparent legends and coloured or grey patches that change
# contrast with the data beneath them after manuscript-width reduction.
SI_TEXT_BOX = dict(boxstyle="round,pad=0.16", fc="white", ec="#CBD5E1",
                   lw=0.55, alpha=0.97)
SI_LEGEND = dict(frameon=True, facecolor="white", edgecolor="#CBD5E1",
                 framealpha=0.97, fancybox=False)

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 9.5,
    "axes.titlesize": 10.2,
    "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.1,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.08,
    "axes.prop_cycle": matplotlib.cycler(color=[BLUE, RED, GREEN, ORANGE, PURPLE, CYAN]),
})


def clean(ax, panel: str, title: str, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_title(title, loc="left", fontweight="bold", pad=5)
    ax.text(-0.13, 1.08, panel, transform=ax.transAxes, fontweight="bold",
            fontsize=10, va="top")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, color="#D9DEE5", linewidth=0.6, alpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)


def signed_scientific_latex(value: float, digits: int = 2) -> str:
    """Format a signed, non-zero deviation without a long near-unity decimal."""
    if value == 0.0:
        return "0"
    exponent = int(math.floor(math.log10(abs(value))))
    mantissa = value / (10.0**exponent)
    return rf"{mantissa:+.{digits}f}\times10^{{{exponent}}}"


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    # PDF keeps text, lines and patches as vectors.  The high fallback DPI
    # protects any rasterized artist that may be introduced in later edits.
    fig.savefig(OUT / name, format="pdf", facecolor="white", dpi=600)
    plt.close(fig)


def _pyvista_panel(name: str, trim_background: bool = False) -> np.ndarray:
    """Load a panel rendered from raw VTK image data by PyVista.

    ``trim_background`` removes only a uniform outer renderer rim.  Pixel
    values inside the VTK viewport are returned unchanged.
    """
    path = PV_ASSETS / name
    if not path.exists():
        raise RuntimeError(
            f"Missing PyVista panel {path}. Run render_all_microstructures_pyvista.py "
            "with ParaView's pvpython first."
        )
    image = plt.imread(path)
    if not trim_background:
        return image
    rgb = image[..., :3]
    background = rgb[0, 0]
    foreground = np.any(np.abs(rgb - background) > 1.0e-7, axis=2)
    rows = np.flatnonzero(np.any(foreground, axis=1))
    cols = np.flatnonzero(np.any(foreground, axis=0))
    if rows.size == 0 or cols.size == 0:
        return image
    return image[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]


def _workflow_box(ax, xy, width, height, number, heading, body,
                  edge="#4B5563", face="#F7F8FA", accent=False,
                  heading_fontsize=8.0, body_fontsize=8.3):
    """Draw one restrained, consistently typeset workflow stage."""
    x, y = xy
    box = FancyBboxPatch(
        xy, width, height,
        boxstyle="square,pad=0.0",
        ec=edge, fc=face, lw=0.9, zorder=2,
    )
    ax.add_patch(box)

    # Plain step number in front of the heading (no badge).
    ax.text(
        x + 0.34, y + height - 0.38, f"{number}.",
        ha="center", va="center", fontsize=8.2, fontweight="bold",
        color="#000000", zorder=4,
    )
    ax.text(
        x + 0.64, y + height - 0.38, heading,
        ha="left", va="center", fontsize=heading_fontsize, fontweight="bold",
        color="#000000", linespacing=0.92, zorder=4,
    )
    ax.plot(
        [x + 0.18, x + width - 0.18],
        [y + height - 0.73, y + height - 0.73],
        color="#9E9E9E", lw=0.6, zorder=3,
    )
    ax.text(
        x + width / 2, y + (height - 0.73) / 2,
        body, ha="center", va="center", fontsize=body_fontsize,
        color="#000000", linespacing=1.32, zorder=4,
    )
    return box


def overview() -> None:
    """Educational workflow with a practical grid-level example."""
    fig, ax = plt.subplots(figsize=(7.2, 4.65))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6.5); ax.axis("off")

    # Primary/secondary scheme shared with the other schematic figures.
    graphite = "#222222"
    teal = "#D62020"          # accent role (red)
    neutral_fill = "#FFFFFF"
    teal_fill = "#DCE7F8"     # light blue for the temporary geometry stage
    p_fill, q_fill, grid_edge = "#7EA6E0", "#FFE38A", "#555555"

    _workflow_box(
        ax, (0.15, 3.72), 2.25, 2.35, 1, "DISCRETE\n\nSTATE",
        "label $g$\nsub-cell fraction $\\alpha$\nmaterial fields\n$c, T, \\rho$",
        edge=graphite, face=neutral_fill,
        heading_fontsize=7.4, body_fontsize=7.8,
    )
    _workflow_box(
        ax, (3.20, 3.51), 3.00, 2.77, 2, "GAUSSIAN\n\nGEOMETRY",
        "$\\widetilde I_g=K_{\\ell_K}*I_g$\n"
        "$\\nabla\\widetilde I_g,\\ H(\\widetilde I_g)$\n"
        "$\\Downarrow$\n$\\mathbf{n}_g,\\ \\kappa_g$",
        edge=teal, face=teal_fill, accent=True,
        heading_fontsize=7.2, body_fontsize=8.3,
    )
    _workflow_box(
        ax, (7.00, 3.72), 2.35, 2.35, 3, "KINETICS",
        "$\\Delta G,\\ M,\\ \\gamma$\n\n"
        "$v_n=M(\\Delta G-\\gamma\\kappa)$",
        edge=graphite, face=neutral_fill,
        heading_fontsize=7.8, body_fontsize=8.0,
    )
    _workflow_box(
        ax, (10.15, 3.72), 1.70, 2.35, 4, "UPDATE",
        "$\\alpha^{n+1}$\n\nTCKT switch",
        edge=graphite, face=neutral_fill,
        heading_fontsize=7.2, body_fontsize=7.8,
    )

    # Long, mechanically aligned connectors make the direction of the
    # algorithm explicit at both manuscript and standalone figure scale.
    for x1, x2 in [(2.52, 3.08), (6.32, 6.88), (9.47, 10.03)]:
        arrow = FancyArrowPatch(
            (x1, 4.895), (x2, 4.895),
            arrowstyle="-|>", mutation_scale=13.5,
            linewidth=1.45, color="#333333",
            shrinkA=0, shrinkB=0, capstyle="round", joinstyle="miter",
            zorder=5,
        )
        ax.add_patch(arrow)

    # A quiet bracket identifies the temporary field without introducing a
    # second competing colour or a floating decorative callout.
    ax.plot([3.62, 5.78], [3.35, 3.35], color=teal, lw=1.05)
    ax.plot([3.62, 3.62], [3.35, 3.49], color=teal, lw=1.05)
    ax.plot([5.78, 5.78], [3.35, 3.49], color=teal, lw=1.05)
    ax.text(
        4.70, 3.10, "temporary field; discarded after geometry evaluation",
        ha="center", color=teal, fontsize=7.7,
    )

    ax.text(
        10.88, 3.30, "TCKT: topology-certified\nkinetic transaction",
        ha="right", va="center", color=graphite, fontsize=7.4, linespacing=1.15,
    )

    # The accepted update becomes the persistent state of the next timestep.
    # Keep this feedback connector quieter than the three primary transfers,
    # but make the closed modelling cycle stated in the text unambiguous.
    ax.plot(
        [11.00, 11.00, 0.78], [3.72, 2.76, 2.76],
        color="#666666", lw=1.05, solid_capstyle="round", zorder=1,
    )
    ax.add_patch(FancyArrowPatch(
        (0.78, 2.76), (0.78, 3.71),
        arrowstyle="-|>", mutation_scale=11.5,
        linewidth=1.05, color="#666666",
        shrinkA=0, shrinkB=0, capstyle="round", joinstyle="miter", zorder=1,
    ))

    # Enlarged three-step example.  Only the stage headings remain inside the
    # panels; the manuscript caption carries the explanation.  This provides
    # generous clear space around every grid, arrow and symbol after reduction
    # to journal width.
    ax.text(0.15, 2.39, "EXAMPLE: ONE BOUNDARY UPDATE", ha="left", va="center",
            fontsize=8.5, fontweight="bold", color="#333333")
    panel_y, panel_h, panel_w = 0.22, 1.96, 3.46
    panel_x = [0.18, 4.27, 8.36]
    titles = ["1  STORED LABELS", "2  MEASURE SHAPE", "3  ACCEPT CHANGE"]
    for x, title in zip(panel_x, titles):
        ax.add_patch(FancyBboxPatch(
            (x, panel_y), panel_w, panel_h,
            boxstyle="square,pad=0.05",
            ec="#8C8C8C", fc="#FFFFFF", lw=0.8, zorder=0,
        ))
        ax.text(x + panel_w / 2, 1.94, title, ha="center", va="center",
                fontsize=8.4, fontweight="bold", color=teal)

    # Panel 1: the persistent piecewise-constant state.
    cell = 0.32
    grid1_x, grid_y = 0.95, 0.35
    split = [2, 2, 3, 3]
    for row in range(4):
        for col in range(6):
            color = p_fill if col < split[row] else q_fill
            ax.add_patch(Rectangle((grid1_x + col * cell, grid_y + row * cell),
                                   cell, cell, fc=color, ec=grid_edge, lw=0.35))

    # Panel 2: the smooth curve exists only for geometry measurement.
    grid2_x = 5.04
    for row in range(4):
        for col in range(6):
            color = "#C3D6F2" if col < split[row] else "#FFF3C4"
            ax.add_patch(Rectangle((grid2_x + col * cell, grid_y + row * cell),
                                   cell, cell, fc=color, ec="#8C8C8C", lw=0.35))
    yy = np.linspace(grid_y + 0.05, grid_y + 4 * cell - 0.05, 100)
    xx = grid2_x + 2.15 * cell + 0.33 * np.sin(
        (yy - grid_y) / (4 * cell) * np.pi - np.pi / 2
    )
    ax.plot(xx, yy, color=teal, lw=1.6, solid_capstyle="round", zorder=4)
    mid = 51
    ax.add_patch(FancyArrowPatch(
        (xx[mid], yy[mid]), (xx[mid] + 0.48, yy[mid] + 0.02),
        arrowstyle="-|>", mutation_scale=12.0, linewidth=1.5,
        color=teal, zorder=5,
    ))
    ax.text(xx[mid] + 0.53, yy[mid] + 0.03, r"$\mathbf{n}$",
            ha="left", va="center", fontsize=8.8, fontweight="bold", color=teal)
    # Keep curvature notation in the clear band above the grid.  Placing it on
    # the curve made the glyph merge with the interface after page reduction.
    ax.text(grid2_x + 0.06, grid_y + 4 * cell + 0.12, r"$\kappa$",
            ha="left", va="center", fontsize=8.8, color=teal)

    # Panel 3: before/after mini-grids make the accepted switch explicit.
    mini, mini_y = 0.245, 0.43
    before_x, after_x = 8.62, 10.46
    ax.text(before_x + 0.49, 1.56, "BEFORE", ha="center", va="center",
            fontsize=7.2, fontweight="bold", color="#555555")
    ax.text(after_x + 0.49, 1.56, "AFTER", ha="center", va="center",
            fontsize=7.2, fontweight="bold", color="#555555")
    for gx, changed in [(before_x, False), (after_x, True)]:
        for row in range(4):
            for col in range(4):
                is_blue = col < 2 or (changed and row == 2 and col == 2)
                color = p_fill if is_blue else q_fill
                edge = teal if (row == 2 and col == 2) else grid_edge
                lw = 1.7 if (row == 2 and col == 2) else 0.35
                ax.add_patch(Rectangle((gx + col * mini, mini_y + row * mini),
                                       mini, mini, fc=color, ec=edge, lw=lw))
    ax.add_patch(FancyArrowPatch(
        (9.73, 0.92), (10.30, 0.92), arrowstyle="-|>",
        mutation_scale=13, linewidth=1.55, color="#333333",
    ))
    ax.text(10.02, 1.13, r"$v_n$", ha="center", va="bottom",
            fontsize=8.4, fontweight="bold", color="#333333")

    # Connect the explanatory panels without placing text over the data.
    for x1, x2 in [(3.72, 4.18), (7.81, 8.27)]:
        ax.add_patch(FancyArrowPatch((x1, 1.14), (x2, 1.14),
                                    arrowstyle="-|>", mutation_scale=12,
                                    linewidth=1.35, color="#666666"))
    save(fig, "fig1_gpk_overview.pdf")


def profile() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.75), constrained_layout=True)
    s = np.linspace(-5.5, 5.5, 500)
    phi = 0.5*(1-np.tanh(s/2))
    dphi = -0.25/np.cosh(s/2)**2
    well = 0.5*phi**2*(1-phi)**2

    ax = axes[0]
    ax.plot(s, phi, color=BLUE, lw=2.2)
    ax.axvline(0, color=GRAY, ls="--", lw=0.8)
    ax.fill_betweenx([0,1], -1, 1, color="#FBDADA", alpha=0.8)
    ax.annotate("interface", (0,0.50), (1.3,0.70), arrowprops=dict(arrowstyle="->", color=RED), color=RED)
    clean(ax, "a", "Equilibrium profile", r"$s=x_\perp/\epsilon$", r"$\phi_0$")
    ax.set_ylim(-0.04,1.04)

    ax = axes[1]
    ax.plot(s, -dphi, color=GREEN, lw=2.2)
    ax.fill_between(s, -dphi, color="#ECE3F5", alpha=0.8)
    clean(ax, "b", "Localized interface weight", r"$s=x_\perp/\epsilon$", r"$-\phi_0'$ ")
    # Keep the normalization callout above the peak and away from both curve
    # shoulders by reserving headroom and anchoring it to the upper-right.
    ax.set_ylim(-0.005, 0.31)
    ax.text(0.97, 0.95, r"$\int(-\phi_0')\,ds=1$", transform=ax.transAxes,
            ha="right", va="top", color=GREEN, bbox=SI_TEXT_BOX)

    ax = axes[2]
    ax.plot(phi, well, color=ORANGE, lw=2.2)
    ax.scatter([0,1],[0,0], color=[BLUE,RED], zorder=3)
    clean(ax, "c", "Normalized double well", r"$\phi$", r"$W(\phi)$")
    # Reserve headroom and anchor the equation in the upper-right so the
    # callout clears both branches and the maximum of the double-well curve.
    ax.set_ylim(-0.001, 0.042)
    ax.text(0.97, 0.95, r"$W=\frac{1}{2}\phi^2(1-\phi)^2$",
            transform=ax.transAxes, ha="right", va="top", bbox=SI_TEXT_BOX)
    save(fig, "figS1_profile.pdf")


def geometry_pipeline() -> None:
    n = 64
    yy, xx = np.mgrid[0:n, 0:n]
    # A closed ellipse keeps the full interface away from the image boundary,
    # so the curvature panel visualises physical curvature rather than
    # one-sided derivative artefacts at the frame edge.  Its nonconstant
    # curvature also makes the Hessian output readable around the full contour.
    indicator = ((((xx-(n-1)/2)/18.0)**2 +
                  ((yy-(n-1)/2)/24.0)**2) <= 1.0).astype(float)
    try:
        from scipy.ndimage import gaussian_filter
        smooth = gaussian_filter(indicator, sigma=2.0, mode="nearest")
    except ImportError:
        radius = 7
        offsets = np.arange(-radius, radius + 1)
        kernel = np.exp(-0.5 * (offsets / 2.0) ** 2)
        kernel /= kernel.sum()
        padded = np.pad(indicator, ((0, 0), (radius, radius)), mode="edge")
        smooth_x = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="valid"), 1, padded)
        padded = np.pad(smooth_x, ((radius, radius), (0, 0)), mode="edge")
        smooth = np.apply_along_axis(lambda col: np.convolve(col, kernel, mode="valid"), 0, padded)
    gy, gx = np.gradient(smooth)
    gyy, gyx = np.gradient(gy)
    gxy, gxx = np.gradient(gx)
    fxy = 0.5*(gxy + gyx)
    mag = np.hypot(gx, gy)
    nx = -gx/np.maximum(mag, 1e-12); ny = -gy/np.maximum(mag, 1e-12)
    curvature = -(gxx + gyy)/np.maximum(mag, 1e-12) \
                + (gx*gx*gxx + 2*gx*gy*fxy + gy*gy*gyy)/np.maximum(mag, 1e-12)**3

    interface_band = mag > 0.08*mag.max()
    curvature_display = np.ma.masked_where(~interface_band, np.abs(curvature))
    curvature_limit = float(np.percentile(np.abs(curvature[interface_band]), 97))

    # Reserve explicit inter-panel gutters.  Arrows drawn outside axes are
    # otherwise reduced to clipped fragments when constrained_layout closes
    # those gutters at manuscript width.
    # All four outer frames have identical dimensions.  Panel d reserves a
    # compact internal column for its colour scale, so the scale never changes
    # the panel size or overlaps the measured interface.
    fig = plt.figure(figsize=(7.2, 2.78))
    grid = fig.add_gridspec(
        1, 4, width_ratios=[1.0, 1.0, 1.0, 1.0],
        left=0.022, right=0.988, bottom=0.035, top=0.86, wspace=0.18,
    )
    axes = [fig.add_subplot(grid[0, i]) for i in range(4)]
    for ax in axes:
        ax.set_box_aspect(1)
    curvature_ax = axes[3].inset_axes([0.025, 0.105, 0.735, 0.79])
    titles = ["Binary label", "Gaussian projection", "Outward normals", "Local curvature"]
    data = [indicator, smooth, smooth]
    cmaps = ["Blues", "Blues", "Blues"]
    images = []
    for i, (ax, title, dat, cmap) in enumerate(zip(axes[:3], titles[:3], data, cmaps)):
        kwargs = dict(origin="lower", cmap=cmap,
                      interpolation="nearest" if i == 0 else "bilinear")
        images.append(ax.imshow(dat, **kwargs))
        ax.set_xticks([]); ax.set_yticks([])
        # Keep the panel identifier and title in one aligned title block.
        # Separate artists collided for the longer "Gaussian projection"
        # heading after manuscript-width reduction.
        ax.set_title(f"{'abcd'[i]}   {title}", loc="left", fontsize=8.9,
                     fontweight="bold", pad=5)
    axes[3].set_facecolor("#F2F2F2")
    axes[3].set_xticks([]); axes[3].set_yticks([])
    axes[3].set_title("d   Local curvature", loc="left", fontsize=8.9,
                      fontweight="bold", pad=5)
    curvature_ax.set_facecolor("#F2F2F2")
    curvature_image = curvature_ax.imshow(
        curvature_display, origin="lower", cmap=YELLOWRED,
        interpolation="bilinear", vmin=0.0, vmax=curvature_limit,
    )
    curvature_ax.set_xticks([]); curvature_ax.set_yticks([])
    for spine in curvature_ax.spines.values():
        spine.set_visible(False)
    # Draw the process arrows in figure coordinates after layout is fixed.
    # This keeps complete shafts and arrowheads visible in every PDF viewer.
    fig.canvas.draw()
    for i in range(3):
        left_box = axes[i].get_position()
        right_box = axes[i + 1].get_position()
        arrow = FancyArrowPatch(
            (left_box.x1 + 0.003, 0.5 * (left_box.y0 + left_box.y1)),
            (right_box.x0 - 0.003, 0.5 * (right_box.y0 + right_box.y1)),
            transform=fig.transFigure,
            arrowstyle="-|>", mutation_scale=10.5,
            color=GRAY, lw=1.35, clip_on=False, zorder=20,
        )
        fig.add_artist(arrow)
    step = 8
    arrow_mask = mag[::step, ::step] > 0.08*mag.max()
    axes[2].quiver(xx[::step,::step][arrow_mask], yy[::step,::step][arrow_mask],
                   nx[::step,::step][arrow_mask], ny[::step,::step][arrow_mask],
                   color=RED, scale=10, width=0.011, pivot="mid")
    axes[2].contour(smooth, levels=[0.5], colors="white", linewidths=0.75,
                    origin="lower")
    curvature_ax.contour(smooth, levels=[0.5], colors="black", linewidths=0.65,
                         origin="lower", alpha=0.75)
    # A shorter vertical gradient occupies the reserved right-hand column.
    # Its ticks face the empty gutter, while the rotated label remains inside
    # the same-size outer panel frame.
    cax = axes[3].inset_axes([0.835, 0.28, 0.060, 0.44])
    colourbar = fig.colorbar(curvature_image, cax=cax, orientation="vertical",
                            ticks=[0.0, 0.05, 0.10], format="%.2f")
    colourbar.ax.yaxis.set_ticks_position("left")
    colourbar.ax.tick_params(labelsize=6.8, length=2.2, pad=1.5)
    colourbar.set_label(r"$|\kappa|\Delta x$", fontsize=7.2, labelpad=2.2)
    annotation_box = dict(boxstyle="round,pad=0.20", fc="#FFFFFF",
                          ec="#CCCCCC", lw=0.6, alpha=0.97)
    axes[0].text(0.06, 0.08, r"$I_g\in\{0,1\}$", transform=axes[0].transAxes,
                 fontsize=8.2, bbox=annotation_box)
    axes[1].text(0.06, 0.08, r"$\widetilde I_g=K_{\ell_K}*I_g$", transform=axes[1].transAxes,
                 fontsize=8.2, bbox=annotation_box)
    # Panel d has a deliberately blank interior because curvature is plotted
    # only in the accepted interface band.  Put the explanatory label in that
    # blank region rather than over the coloured curvature ring.
    # Keep one compact expression fully inside the unplotted ellipse interior;
    # this avoids both contour contact and line-to-line text collisions.
    curvature_note_box = dict(boxstyle="round,pad=0.12", fc="#FFFFFF",
                              ec="#CCCCCC", lw=0.55, alpha=1.0)
    curvature_ax.text(0.50, 0.50, "$\\kappa_g$",
                 transform=curvature_ax.transAxes, ha="center", va="center",
                 fontsize=6.6, bbox=curvature_note_box)
    save(fig, "figS2_geometry_pipeline.pdf")


def curvature_signs() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.75), constrained_layout=True)
    for i, ax in enumerate(axes):
        ax.set_xlim(-1.4,1.4); ax.set_ylim(-1.55,1.2); ax.set_aspect("equal"); ax.axis("off")
        ax.text(-0.10,1.05,"abc"[i], transform=ax.transAxes, fontweight="bold", fontsize=10)

    ax = axes[0]
    ax.add_patch(Circle((0,0),0.65,fc="#DCEAF8",ec=BLUE,lw=2))
    for th in np.linspace(0,2*np.pi,8,endpoint=False):
        ax.annotate("", xy=(0.95*np.cos(th),0.95*np.sin(th)), xytext=(0.64*np.cos(th),0.64*np.sin(th)),
                    arrowprops=dict(arrowstyle="->",color=RED,lw=1.2))
    ax.set_title("Resident inside", fontweight="bold")
    ax.text(0,-0.05,"resident",ha="center",color=BLUE)
    ax.text(0,-1.30,r"$\kappa_p=+1/R$" + "\ncapillarity shrinks",ha="center")

    ax = axes[1]
    ax.add_patch(Rectangle((-1.15,-0.90),2.3,0.90,fc="#DCEAF8",ec="none"))
    ax.add_patch(Rectangle((-1.15,0),2.3,0.90,fc="#ECE3F5",ec="none"))
    ax.plot([-1.15,1.15],[0,0],color=GREEN,lw=2)
    for x in [-0.8,-0.25,0.3,0.85]:
        ax.annotate("",xy=(x,0.55),xytext=(x,0.05),arrowprops=dict(arrowstyle="->",color=RED))
    ax.set_title("Flat interface", fontweight="bold")
    ax.text(0,-1.30,r"$\kappa_p=0$",ha="center")

    ax = axes[2]
    ax.add_patch(Circle((0,0),0.65,fc="#ECE3F5",ec=BLUE,lw=2,ls="--"))
    for th in np.linspace(0,2*np.pi,8,endpoint=False):
        ax.annotate("", xy=(0.40*np.cos(th),0.40*np.sin(th)), xytext=(0.72*np.cos(th),0.72*np.sin(th)),
                    arrowprops=dict(arrowstyle="->",color=RED,lw=1.2))
    ax.set_title("Resident outside", fontweight="bold")
    ax.text(0,-0.05,"candidate",ha="center",color=GREEN)
    ax.text(0,-1.30,r"$\kappa_p=-1/R$" + "\ncapillarity aids inclusion growth",ha="center")
    save(fig, "figS3_curvature_convention.pdf")


def gaussian_accuracy() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.05), constrained_layout=True)
    x = np.linspace(-10,10,500)
    ax = axes[0]
    for sig, col in [(1.5,ORANGE),(2.0,BLUE),(3.0,GREEN)]:
        g = np.exp(-0.5*(x/sig)**2)/(np.sqrt(2*np.pi)*sig)
        ax.plot(x,g,color=col,label=rf"$\sigma_K={sig:g}$ cells")
    # The +-2-cell band is the default kernel width; drawn behind the curves
    # with a visible tint and dashed edges so it reads as a region.
    ax.axvspan(-2,2,facecolor="#DCCCEE",edgecolor="#7B2FA8",ls="--",lw=0.8,
               alpha=0.6,zorder=0,label=r"default, $\pm2$ cells")
    clean(ax,"a","Kernel scale controls support","cell offset","normalized weight")
    ax.set_ylim(0,0.30)
    # The upper-left corner is empty for every curve, so the legend never
    # covers a peak or a flank.
    ax.legend(loc="upper left",frameon=True,facecolor="white",edgecolor="none",
              framealpha=0.95,fontsize=7.2,handlelength=1.2,handletextpad=0.5,borderaxespad=0.2)

    ax = axes[1]
    ratio = np.linspace(0.01,0.45,400)
    error = 100*ratio**2*2.8
    ax.plot(ratio,error,color=BLUE,lw=2.2,label=r"$O[(\ell_K/R)^2]$")
    ax.axvspan(0.02,0.20,color="#ECE3F5",alpha=0.8)
    ax.scatter([2/15],[3.51],s=55,color=RED,zorder=4,label="circle test: 3.51%")
    ax.annotate(r"default: $\sigma_K=2$, $R/\Delta x=15$",(2/15,3.51),(0.22,7.3),
                arrowprops=dict(arrowstyle="->",color=RED),fontsize=8,color=RED)
    clean(ax,"b","Resolved-geometry accuracy",r"smoothing ratio $\ell_K/R$","curvature error [%]")
    ax.set_xlim(0,0.45); ax.set_ylim(0,12)
    ax.text(0.04,0.88,"recommended\nresolved regime",transform=ax.transAxes,color=GREEN)
    ax.legend(frameon=False,loc="lower right")
    save(fig, "figS4_gaussian_accuracy_final.pdf")


def stability() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.40), constrained_layout=True)
    xi = np.linspace(0,np.pi,400)
    ax = axes[0]
    for r_, col, label in [(0.40,GREEN,"stable"),(0.50,ORANGE,"limit"),(0.60,RED,"unstable")]:
        amp = 1-4*r_*np.sin(xi/2)**2
        ax.plot(xi/np.pi,amp,color=col,label=rf"$r={r_:.2f}$: {label}")
    ax.axhspan(-1,1,color="#ECE3F5",alpha=0.5)
    ax.axhline(-1,color=GRAY,ls="--",lw=0.8); ax.axhline(1,color=GRAY,ls="--",lw=0.8)
    clean(ax,"a","Explicit-diffusion amplification",r"normalized wavenumber $\xi/\pi$",r"$G(\xi)$")
    # Keep the legend outside the data region; the previous upper-right legend
    # hid both the G=1 stability boundary and the initial portions of the three
    # amplification curves.
    ax.legend(**SI_LEGEND, loc="upper center", bbox_to_anchor=(0.5, -0.22),
              ncol=3, fontsize=7.2, handlelength=1.8, columnspacing=0.8)

    ax = axes[1]
    dx = np.logspace(-1,1,200)
    diffusion = 0.04*dx**2
    interface = 0.125*dx
    ax.loglog(dx,diffusion,color=BLUE,label=r"diffusion: $\Delta t\propto\Delta x^2$")
    ax.loglog(dx,interface,color=ORANGE,label=r"interface: $\Delta t\propto\Delta x$")
    ax.fill_between(dx,np.minimum(diffusion,interface),1e-4,color="#ECE3F5",alpha=0.6)
    clean(ax,"b","The active timestep is the minimum",r"grid spacing $\Delta x$ [$\mu$m]",r"stable $\Delta t$ [ms]")
    ax.legend(**SI_LEGEND)
    save(fig, "figS5_stability.pdf")


def topology_transactions() -> None:
    """Executable topology cases used to explain the generic TCKT rule."""
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.35), constrained_layout=True)
    cases = []

    smooth = np.zeros((7, 7), dtype=int)
    smooth[:, :4] = 1
    cases.append((smooth, (3, 3), "Smooth advance", "accept", GREEN))

    bridge = np.zeros((7, 7), dtype=int)
    bridge[1:6, 1:3] = 1; bridge[1:6, 4:6] = 1; bridge[3, 3] = 1
    cases.append((bridge, (3, 3), "Bridge deletion", "defer", RED))

    merge = np.zeros((7, 7), dtype=int)
    merge[3, 2] = 1; merge[3, 4] = 1
    cases.append((merge, (3, 3), "Premature merger", "defer", RED))

    extinct = np.zeros((7, 7), dtype=int)
    extinct[3, 3] = 1
    cases.append((extinct, (3, 3), "Terminal extinction", "accept", GREEN))

    cmap = matplotlib.colors.ListedColormap(["#EEF2F5", "#86B5E1"])
    for j, (field, centre, heading, result, colour) in enumerate(cases):
        ax = axes[j]
        ax.imshow(field, cmap=cmap, vmin=0, vmax=1, origin="lower",
                  interpolation="nearest")
        ax.set_xticks(np.arange(-0.5, 7, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 7, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.85)
        ax.tick_params(which="both", bottom=False, left=False,
                       labelbottom=False, labelleft=False)
        y, x = centre
        ax.add_patch(Rectangle((x-0.48, y-0.48), 0.96, 0.96,
                               fill=False, ec=colour, lw=2.4))
        ax.annotate("", xy=(x+0.15, y), xytext=(x-0.15, y),
                    arrowprops=dict(arrowstyle="->", color=colour, lw=1.6))
        ax.set_title(heading, fontsize=8.3, fontweight="bold", pad=4)
        ax.text(0.50, -0.12, result.upper(), transform=ax.transAxes,
                ha="center", va="top", color=colour, fontsize=8.2,
                fontweight="bold", bbox=SI_TEXT_BOX)
        ax.text(-0.07, 1.06, "abcd"[j], transform=ax.transAxes,
                fontweight="bold", fontsize=10)
    save(fig, "figS13_topology_transactions.pdf")


def gibbs_thomson() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.15), constrained_layout=True)
    ax = axes[0]
    ax.set_xlim(-1.6,1.6); ax.set_ylim(-1.2,1.2); ax.axis("off"); ax.set_aspect("equal")
    for cx, rad, outward, title, col in [(-0.75,0.34,False,r"$R<R_{\rm crit}$\nshrinks",RED),
                                         (0.75,0.62,True,r"$R>R_{\rm crit}$\ngrows",GREEN)]:
        ax.add_patch(Circle((cx,0),rad,fc="#DCEAF8",ec=BLUE,lw=2))
        for th in [0,np.pi/2,np.pi,3*np.pi/2]:
            start=rad*(0.88 if outward else 1.18); end=rad*(1.28 if outward else 0.72)
            ax.annotate("",xy=(cx+end*np.cos(th),end*np.sin(th)),xytext=(cx+start*np.cos(th),start*np.sin(th)),
                        arrowprops=dict(arrowstyle="->",color=col,lw=1.4))
        ax.text(cx,-1.00,title,ha="center",color=col,fontweight="bold")
    ax.set_title("Size selection",fontweight="bold")
    ax.text(-0.10,1.05,"a",transform=ax.transAxes,fontweight="bold",fontsize=10)

    ax = axes[1]
    rr = np.linspace(0.35,3.2,400)
    ax.plot(rr,1/rr,color=RED,lw=2.2,label=r"capillarity $\sigma/R$")
    ax.axhline(1,color=GREEN,lw=2,label=r"bulk reduction $\mathcal{D}$")
    ax.axvline(1,color=GRAY,ls="--",lw=1.2)
    ax.fill_between(rr,1/rr,1,where=rr<1,color="#FBE2E2",alpha=0.7)
    ax.fill_between(rr,1/rr,1,where=rr>1,color="#ECE3F5",alpha=0.7)
    ax.scatter([1],[1],color=ORANGE,zorder=4)
    ax.annotate(r"$R_{\rm crit}=\sigma/\mathcal{D}$",(1,1),(1.35,1.65),
                arrowprops=dict(arrowstyle="->",color=ORANGE),color=ORANGE)
    clean(ax,"b","Bulk-capillary balance",r"radius $R/R_{\rm crit}$","normalized driving magnitude")
    ax.set_xlim(0.35,3.2); ax.set_ylim(0,3.0)
    ax.legend(frameon=False,loc="upper right")
    save(fig, "fig7_gibbs_thomson.pdf")


def foundational_validation() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), constrained_layout=True)
    validation = output_dir("validate_theory_foundations")

    ax = axes[0, 0]
    curved = output_dir("validate_extended_base_models")
    shapes = ["circle", "ellipse", "square"]
    measured = []
    for key in ("g13", "g14", "g15"):
        history = np.loadtxt(curved / f"{key}_area_t.csv", delimiter=",", skiprows=1)
        exact = np.loadtxt(curved / f"{key}_area_t_theory.csv", delimiter=",", skiprows=1)
        measured.append(stats.linregress(history[:, 0], history[:, 1]).slope /
                        stats.linregress(exact[:, 0], exact[:, 1]).slope)
    x = np.arange(3)
    ax.bar(x, measured, width=0.56, color=[BLUE, GREEN, PURPLE], label="GPK")
    ax.axhline(1.0, color=ORANGE, linestyle="--", label="Gauss-Bonnet")
    ax.set_xticks(x, shapes)
    ax.set_ylim(0, 1.16)
    clean(ax, "a", "Shape-independent area loss", ylabel="measured / exact")
    ax.legend(frameon=True, loc="lower right", facecolor="white",
              edgecolor="#CCCCCC", framealpha=1.0, borderpad=0.45)

    ax = axes[0, 1]
    bins = np.loadtxt(validation / "t3_bins_mean.csv", delimiter=",", skiprows=1)
    counts = np.loadtxt(validation / "t3_bins_count.csv", delimiter=",", skiprows=1)
    scatter = np.loadtxt(validation / "t3_scatter.csv", delimiter=",", skiprows=1)
    n, measured = bins[:, 0], bins[:, 1] * 1e8
    count = counts[:, 1].astype(int)
    sem = np.zeros_like(measured)
    for j, nn in enumerate(n):
        values = scatter[np.isclose(scatter[:, 0] + 6, nn), 1] * 1e8
        sem[j] = np.std(values, ddof=1) / np.sqrt(len(values)) if len(values) > 1 else np.nan
    supported = count >= 8
    theory_n = np.arange(3, 12)
    theory = (np.pi / 3) * 1e-8 * 0.324 * (theory_n - 6) * 1e8
    ax.errorbar(n[supported], measured[supported], yerr=sem[supported], fmt="o-",
                color=BLUE, capsize=2.2, lw=1.5, label=r"GPK class mean $\pm$ SEM")
    if np.any(~supported):
        ax.scatter(n[~supported], measured[~supported], marker="x", s=28,
                   color=GRAY, label="rare class (fewer than 8 grains)")
    ax.plot(theory_n, theory, "--", color=ORANGE, label="exact prefactor")
    ax.axhline(0, color=GRAY, linewidth=0.8)
    ax.set_xticks(theory_n)
    clean(ax, "b", "von Neumann-Mullins topology", "neighbour count, $Z$",
          r"$\mathrm{d}A/\mathrm{d}t$  [$10^{-8}$ m$^2$ s$^{-1}$]")
    ax.legend(frameon=False)

    ax = axes[1, 0]
    decay = np.loadtxt(validation / "t4_perturbation_decay.csv", delimiter=",", skiprows=1)
    t, amp = decay[:, 0], decay[:, 1] / 0.25e-6
    tt = np.linspace(t.min(), t.max(), 200)
    ax.semilogy(t, amp, "o", color=BLUE, label="GPK")
    ax.semilogy(tt, amp[0] * np.exp(-7.807 * tt), "--", color=ORANGE,
                label="Mullins prediction")
    clean(ax, "c", "Perturbation decay", "time [s]", "mode amplitude [cells]")
    ax.legend(frameon=False)

    ax = axes[1, 1]
    curvature = np.loadtxt(validation / "t1_curvature_error.csv", delimiter=",", skiprows=1)
    resolution = curvature[:, 0]
    kappa_measured, kappa_exact = curvature[:, 1], curvature[:, 2]
    ax.plot(resolution, kappa_measured, "o-", color=GREEN, label="GPK")
    ax.plot(resolution, kappa_exact, "--", color=ORANGE, label=r"exact $1/R$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([10, 15, 20, 30, 40, 60], ["10", "15", "20", "30", "40", "60"])
    clean(ax, "d", r"Gaussian curvature convergence", r"circle resolution $R/\Delta x$",
          r"measured curvature $\kappa\,\Delta x$")
    ax.legend(frameon=False)

    save(fig, "fig3_foundational_validation.pdf")


def _pair_canvas(ax, title: str) -> None:
    ax.set_xlim(0, 2.2)
    ax.set_ylim(0, 1.35)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=9.5, fontweight="bold", pad=3)
    ax.text(0.45, 1.20, "initial", ha="center", color=GRAY, fontsize=8)
    ax.text(1.75, 1.20, "final", ha="center", color=GRAY, fontsize=8)
    ax.annotate("", xy=(1.28, 0.58), xytext=(0.92, 0.58),
                arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.5))


def _snapshot_plane(snap) -> np.ndarray:
    """Return the central xy label plane from a stored GPK snapshot."""
    nx = int(snap["meta"]["Nx"])
    ny = int(snap["meta"]["Ny"])
    nz = int(snap["meta"].get("Nz", 1))
    volume = snap["grain_id"].reshape(nz, ny, nx)
    return volume[nz // 2]


def _label_rgb(labels: np.ndarray, semantic: dict[int, str] | None = None) -> np.ndarray:
    """Render integer labels consistently while retaining one-cell interfaces."""
    if semantic is None:
        hue = np.mod(labels.astype(float) * 0.61803398875, 1.0)
        sat = 0.43 + 0.13 * np.mod(labels, 3) / 2.0
        val = 0.88 + 0.08 * np.mod(labels, 5) / 4.0
        rgb = hsv_to_rgb(np.stack((hue, sat, val), axis=-1))
    else:
        rgb = np.empty((*labels.shape, 3), dtype=float)
        for gid in np.unique(labels):
            rgb[labels == gid] = matplotlib.colors.to_rgb(semantic.get(int(gid), "#CBD5E1"))

    boundary = np.zeros(labels.shape, dtype=bool)
    boundary[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    boundary[1:, :] |= labels[1:, :] != labels[:-1, :]
    rgb[boundary] = matplotlib.colors.to_rgb("#263442")
    return rgb


def _simulation_pair(fig, slot, snaps, title: str, panel: str,
                     metric, semantic=None, scale_um: int = 50) -> None:
    """Place two real simulation states in a compact, shared-time panel."""
    sub = slot.subgridspec(2, 2, height_ratios=(0.14, 1), hspace=0.04, wspace=0.04)
    heading = fig.add_subplot(sub[0, :])
    heading.axis("off")
    heading.text(0.00, 0.55, panel, fontsize=10, fontweight="bold", va="center")
    heading.text(0.10, 0.55, title, fontsize=9.5, fontweight="bold", va="center")

    for col, (state, state_name) in enumerate(zip(snaps, ("early", "late"))):
        ax = fig.add_subplot(sub[1, col])
        plane = _snapshot_plane(state)
        ax.imshow(_label_rgb(plane, semantic), origin="lower", interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#52606D"); spine.set_linewidth(0.7)
        time = state["meta"].get("time", 0.0)
        ax.set_title(f"{state_name}:  $t={time:.3g}$ s", fontsize=8, pad=2)
        ax.text(0.03, 0.04, metric(state), transform=ax.transAxes,
                fontsize=7.2, color="white", va="bottom",
                bbox=dict(boxstyle="round,pad=0.20", fc="#17212B", ec="none", alpha=0.84))

        if col == 0:
            dx_um = state["meta"].get("dx", 1e-6) * 1e6
            length_px = scale_um / dx_um
            y = plane.shape[0] * 0.91
            x1 = plane.shape[1] * 0.07
            ax.plot([x1, x1 + length_px], [y, y], color="#17212B", lw=2.6,
                    solid_capstyle="butt")
            ax.text(x1 + length_px / 2, y - plane.shape[0] * 0.035,
                    rf"{scale_um} $\mu$m", color="#17212B", fontsize=6.8,
                    ha="center", va="top",
                    bbox=dict(boxstyle="square,pad=0.05", fc="white", ec="none", alpha=0.72))


def representative_microstructures() -> None:
    """Figure 8: direct, non-magnetic simulation snapshots from CSV output."""
    mod = _validation_module()
    def endpoints(name):
        snaps = mod.load_snaps(output_dir(name))
        if len(snaps) < 2:
            raise RuntimeError(f"Need at least two snapshots for {name}")
        return snaps[0], snaps[-1]

    gg = endpoints("validation_gg_2d")
    rex = endpoints("validation_rex_2d_full")
    precip = endpoints("vis_precip_2d")
    competing = endpoints("vis_precip_nodiff")

    fig = plt.figure(figsize=(7.2, 5.35), constrained_layout=True)
    outer = fig.add_gridspec(2, 2, wspace=0.06, hspace=0.10)
    _simulation_pair(fig, outer[0, 0], gg, "Normal grain growth", "a",
                     lambda s: rf"$N_g={len(np.unique(s['grain_id']))}$")
    _simulation_pair(fig, outer[0, 1], rex, "Recrystallisation", "b",
                     lambda s: rf"$X_{{\rm RX}}={np.mean(s['grain_id'] != 0):.3f}$")
    _simulation_pair(fig, outer[1, 0], precip, "Subcritical-particle dissolution", "c",
                     lambda s: rf"$f_p={np.mean(s['grain_id'] == 1):.3f}$",
                     semantic={1: ORANGE, 2: "#DCE7F2"}, scale_um=25)
    _simulation_pair(fig, outer[1, 1], competing, "Competitive particle evolution", "d",
                     lambda s: rf"$f_1={np.mean(s['grain_id'] == 1):.3f}$",
                     semantic={0: "#E5E7EB", 1: ORANGE, 2: BLUE}, scale_um=25)
    save(fig, "fig4_multiphysics_microstructures.pdf")


def _load_snapshot(directory: Path, index: int = -1):
    files = sorted(directory.glob("snap_*.csv"))
    if not files:
        raise RuntimeError(f"No CSV snapshots found in {directory}")
    path = files[index]
    meta = {}
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.startswith("#"):
                break
            for key in ("time", "dx", "Nz", "Ny", "Nx"):
                match = re.search(rf"{key}=([\d.e+\-]+)", line)
                if match:
                    meta[key] = float(match.group(1))
    data = np.loadtxt(path, comments="#")
    if data.ndim == 1:
        data = data[np.newaxis, :]
    return {
        "x": data[:, 0], "y": data[:, 1], "z": data[:, 2],
        "grain_id": data[:, 3].astype(int), "alpha": data[:, 4],
        "comp": data[:, 5], "meta": meta,
    }


def _characteristic_grain_series(directory: Path):
    """Return time, characteristic mean-volume radius and grain count."""
    files = sorted(directory.glob("snap_*.csv"))
    times, radii, counts = [], [], []
    for index, _ in enumerate(files):
        snap = _load_snapshot(directory, index)
        ng = len(np.unique(snap["grain_id"]))
        if ng == 0:
            continue
        dx_um = snap["meta"].get("dx", 1e-6) * 1e6
        nz = int(snap["meta"].get("Nz", 1))
        mean_cells = len(snap["grain_id"]) / ng
        radius = ((mean_cells / np.pi)**0.5 if nz == 1 else
                  (3*mean_cells/(4*np.pi))**(1/3)) * dx_um
        times.append(snap["meta"].get("time", 0.0))
        radii.append(radius)
        counts.append(ng)
    return np.asarray(times), np.asarray(radii), np.asarray(counts)


def _regularize_labels_2d(labels: np.ndarray, passes: int = 2) -> np.ndarray:
    """Display-only modal reconstruction for voxelated categorical maps."""
    result = labels.copy()
    values = np.unique(labels)
    for _ in range(passes):
        padded = np.pad(result, 1, mode="edge")
        counts = np.zeros((len(values),) + result.shape, dtype=np.uint8)
        for dy in range(3):
            for dx in range(3):
                neighbor = padded[dy:dy+result.shape[0], dx:dx+result.shape[1]]
                for index, value in enumerate(values):
                    counts[index] += neighbor == value
        result = values[np.argmax(counts, axis=0)]
    return result


def _overlay_gaussian_curvature_2d(ax, labels: np.ndarray,
                                   sigma_cells: float = 4.0,
                                   kappa_max: float = 0.16) -> None:
    """Draw smooth GPK level-set interfaces coloured by |kappa|*dx.

    The categorical raster remains the unmodified cell state.  The overlay is
    reconstructed from label-conditioned Gaussian projections, matching the
    geometry view used by the solver and avoiding a misleading pixel-edge
    contour.  Curvature magnitude is used because the outward sign reverses
    when the same shared interface is viewed from the adjacent label.
    """
    if gaussian_filter is None:
        raise RuntimeError("scipy.ndimage is required for curvature overlays")
    norm = Normalize(0.0, kappa_max, clip=True)
    segments, values = [], []
    for label in np.unique(labels):
        smooth = gaussian_filter((labels == label).astype(float), sigma_cells,
                                 mode="wrap", truncate=3.5)
        gy, gx = np.gradient(smooth, edge_order=2)
        gyy, gyx = np.gradient(gy, edge_order=2)
        gxy, gxx = np.gradient(gx, edge_order=2)
        gxy = 0.5 * (gxy + gyx)
        grad2 = gx*gx + gy*gy
        denom = np.maximum(grad2*np.sqrt(grad2), 1e-12)
        kappa = np.abs((gx*gx*gyy - 2.0*gx*gy*gxy + gy*gy*gxx) / denom)
        contour = ax.contour(smooth, levels=[0.5], colors="none", linewidths=0)
        paths = contour.allsegs[0]
        contour.remove()
        for vertices in paths:
            if len(vertices) < 2:
                continue
            ij = np.rint(vertices).astype(int)
            ij[:, 0] = np.clip(ij[:, 0], 0, labels.shape[1]-1)
            ij[:, 1] = np.clip(ij[:, 1], 0, labels.shape[0]-1)
            kv = kappa[ij[:, 1], ij[:, 0]]
            segments.extend(np.stack([vertices[:-1], vertices[1:]], axis=1))
            values.extend(0.5*(kv[:-1] + kv[1:]))
    if not segments:
        return
    # A pale halo separates the smooth sub-cell contour from the categorical
    # background; the coloured core makes local curvature unmistakable.
    ax.add_collection(LineCollection(segments, colors="#F8FAFC", linewidths=1.35,
                                     capstyle="round", joinstyle="round", zorder=5))
    lc = LineCollection(segments, cmap="plasma", norm=norm, linewidths=0.72,
                        capstyle="round", joinstyle="round", zorder=6)
    lc.set_array(np.asarray(values))
    ax.add_collection(lc)


def _overlay_gaussian_boundaries_2d(ax, labels: np.ndarray,
                                    sigma_cells: float = 1.25) -> None:
    """Draw a common display-only sub-cell boundary on categorical labels.

    The same reconstruction is used for GPK and phase-field argmax labels in
    the matched comparison.  A white halo separates neighbouring fill colours;
    the dark core is the 0.5 contour of a label-conditioned Gaussian field.
    Neither layer changes the labels used by any reported metric.
    """
    if gaussian_filter is None:
        raise RuntimeError("scipy.ndimage is required for boundary overlays")
    segments = []
    for label in np.unique(labels):
        smooth = gaussian_filter((labels == label).astype(float), sigma_cells,
                                 mode="wrap", truncate=3.5)
        if smooth.max() < 0.5:
            continue
        contour = ax.contour(smooth, levels=[0.5], colors="none", linewidths=0)
        for vertices in contour.allsegs[0]:
            if len(vertices) >= 2:
                segments.extend(np.stack([vertices[:-1], vertices[1:]], axis=1))
        contour.remove()
    if not segments:
        return
    ax.add_collection(LineCollection(segments, colors="#F8FAFC", linewidths=1.45,
                                     capstyle="round", joinstyle="round", zorder=5))
    ax.add_collection(LineCollection(segments, colors="#18232E", linewidths=0.78,
                                     capstyle="round", joinstyle="round", zorder=6))


def qualitative_2d_gallery() -> None:
    """Initial/final raw VTK cell fields rendered by PyVista."""
    cases = [
        ("a", "Isotropic grain growth", "isotropic",
         output_dir("manuscript_large_domain") / "isotropic_2d_connected",
         lambda s: rf"$N_g={len(np.unique(s['grain_id']))}$"),
        ("b", "Anisotropic grain growth", "anisotropic",
         output_dir("manuscript_large_domain") / "anisotropic_2d_connected",
         lambda s: rf"$N_g={len(np.unique(s['grain_id']))}$"),
        ("c", "Recrystallisation", "recrystallisation",
         output_dir("manuscript_large_domain") / "recrystallisation_2d",
         lambda s: rf"$X_{{\rm RX}}={np.mean(s['grain_id'] != 0):.3f}$"),
        ("d", "Multiphase growth", "multiphase",
         output_dir("manuscript_large_domain") / "multiphase_2d",
         lambda s: (rf"$f_1={np.mean(s['grain_id'] == 1):.2f}$, "
                    rf"$f_2={np.mean(s['grain_id'] == 2):.2f}$")),
        ("e", "Elastic-coupled dendrite", "elastic",
         output_dir("fig7_large") / "dendrite_2d",
         lambda s: rf"$f_1={np.mean(s['grain_id'] == 1):.3f}$"),
    ]

    fig = plt.figure(figsize=(7.2, 4.00), constrained_layout=False)
    grid = fig.add_gridspec(2, 6, left=0.025, right=0.99, top=0.975,
                            bottom=0.035, wspace=0.10, hspace=0.23)
    slots = [grid[0, 1:3], grid[0, 3:5], grid[1, 0:2], grid[1, 2:4], grid[1, 4:6]]
    axes = []
    for slot, (panel, title, asset_tag, directory, metric) in zip(slots, cases):
        files = sorted(directory.glob("snap_*.csv"))
        snaps = (_load_snapshot(directory, 0),
                 _load_snapshot(directory, len(files)-1))
        sub = slot.subgridspec(2, 2, height_ratios=(0.17, 1),
                               hspace=0.08, wspace=0.035)
        heading = fig.add_subplot(sub[0, :]); heading.axis("off")
        heading.text(0.0, 0.55, panel, fontsize=11.5, fontweight="bold", va="center")
        heading.text(0.12, 0.55, title, fontsize=10.2, fontweight="bold", va="center")
        for col, (snap, state) in enumerate(zip(snaps, ("initial", "final"))):
            ax = fig.add_subplot(sub[1, col]); axes.append(ax)
            # The dendrite domain is filled edge to edge by liquid, so trimming
            # would crop the panel to the crystal and hide the matrix margin.
            ax.imshow(_pyvista_panel(f"qualitative2d_{asset_tag}_{state}.png",
                                     trim_background=asset_tag != "elastic"),
                      interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color("#555555"); spine.set_linewidth(0.5)
            # Exact times and endpoint metrics are stated in the adjacent text
            # and run table.  State labels alone keep the image field clear and
            # prevent a second line from colliding with the row heading.
            ax.set_title(state.capitalize(), fontsize=8.8, pad=3.2)
    save(fig, "fig_qualitative_2d_gallery.pdf")


def _orthogonal_volume(ax, snap, semantic=None) -> None:
    """Render three central planes from a true 3D output volume in perspective."""
    nx = int(snap["meta"]["Nx"])
    ny = int(snap["meta"]["Ny"])
    nz = int(snap["meta"].get("Nz", 1))
    dx_um = snap["meta"].get("dx", 1e-6) * 1e6
    volume = snap["grain_id"].reshape(nz, ny, nx)
    xm, ym, zm = nx // 2, ny // 2, nz // 2
    x = np.arange(nx) * dx_um
    y = np.arange(ny) * dx_um
    z = np.arange(nz) * dx_um

    xx, yy = np.meshgrid(x, y)
    ax.plot_surface(xx, yy, np.full_like(xx, z[zm]),
                    facecolors=_label_rgb(volume[zm], semantic), shade=False,
                    linewidth=0, antialiased=False, rasterized=True)
    xx, zz = np.meshgrid(x, z)
    ax.plot_surface(xx, np.full_like(xx, y[ym]), zz,
                    facecolors=_label_rgb(volume[:, ym, :], semantic), shade=False,
                    linewidth=0, antialiased=False, rasterized=True)
    yy, zz = np.meshgrid(y, z)
    ax.plot_surface(np.full_like(yy, x[xm]), yy, zz,
                    facecolors=_label_rgb(volume[:, :, xm], semantic), shade=False,
                    linewidth=0, antialiased=False, rasterized=True)

    ax.set_xlim(0, x[-1]); ax.set_ylim(0, y[-1]); ax.set_zlim(0, z[-1])
    ax.set_box_aspect((nx, ny, nz))
    ax.view_init(elev=25, azim=-52)
    ax.set_xlabel(r"$x$ [$\mu$m]", labelpad=-4, fontsize=7)
    ax.set_ylabel(r"$y$ [$\mu$m]", labelpad=-4, fontsize=7)
    ax.set_zlabel(r"$z$ [$\mu$m]", labelpad=-5, fontsize=7)
    ax.tick_params(labelsize=6, pad=-2)
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((0.97, 0.98, 0.99, 1.0))
        axis.pane.set_edgecolor("#CBD5E1")


def qualitative_3d_gallery() -> None:
    """Orthogonal cutaways from newly generated 3D CSV volumes."""
    output = output_dir("manuscript_gallery")
    cases = [
        ("a", "Isotropic grain growth", "grain_growth_isotropic_3d", 0, None,
         lambda s: rf"$N_g={len(np.unique(s['grain_id']))}$"),
        ("b", "Anisotropic grain growth", "grain_growth_anisotropic_3d", 0, None,
         lambda s: rf"$N_g={len(np.unique(s['grain_id']))}$"),
        ("c", "Anisotropic solidification", "solidification_dendrite_3d", 1,
         {0: "#DDE8F2", 1: ORANGE},
         lambda s: rf"$f_s={np.mean(s['grain_id'] != 0):.3f}$"),
        ("d", "Recrystallisation", "recrystallisation_standard_3d", 2, None,
         lambda s: rf"$X_{{\rm RX}}={np.mean(s['grain_id'] != 0):.3f}$"),
    ]

    fig = plt.figure(figsize=(7.2, 5.65), constrained_layout=True)
    axes = [fig.add_subplot(2, 2, i + 1, projection="3d") for i in range(4)]
    for ax, (panel, title, dirname, index, semantic, metric) in zip(axes, cases):
        snap = _load_snapshot(output / dirname, index)
        _orthogonal_volume(ax, snap, semantic)
        ax.set_title(title, fontsize=9.2, fontweight="bold", pad=2)
        ax.text2D(-0.03, 0.98, panel, transform=ax.transAxes, fontsize=10,
                  fontweight="bold", va="top")
        ax.text2D(0.03, 0.05,
                  metric(snap) + rf";  $t={snap['meta'].get('time', 0.0):.3g}$ s",
                  transform=ax.transAxes, fontsize=7.2, color="white",
                  bbox=dict(boxstyle="round,pad=0.20", fc="#17212B", ec="none", alpha=0.86))
    save(fig, "fig_qualitative_3d_gallery.pdf")


def extended_validation() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), constrained_layout=True)

    ax = axes[0, 0]
    names = ["rex", "solid.", "reduction", "precip."]
    ratios = [0.835, 0.834, 0.835, 0.835]
    ax.bar(names, ratios, color=[BLUE, CYAN, GREEN, PURPLE])
    ax.axhline(1, ls="--", color=ORANGE, label="prescribed velocity")
    ax.set_ylim(0.72, 1.04)
    clean(ax, "a", "Flat-front velocity audit", ylabel=r"$v_{\rm GPK}/v_{\rm input}$")
    ax.legend(frameon=False, loc="lower right")

    ax = axes[0, 1]
    rr = np.array([0.4, 0.7, 1.3, 2.0])
    dr = np.array([-0.95, -0.42, 0.48, 1.00])
    ax.axvspan(0.25, 1.0, color="#FBE2E2", alpha=0.8)
    ax.axvspan(1.0, 2.15, color="#ECE3F5", alpha=0.8)
    ax.axvline(1, color=GRAY, ls="--")
    ax.axhline(0, color=GRAY, lw=0.8)
    ax.scatter(rr, dr, s=42, c=[RED, RED, GREEN, GREEN], zorder=3)
    ax.text(0.48, 0.90, "shrink", transform=ax.transAxes, color=RED, ha="center")
    ax.text(0.78, 0.90, "grow", transform=ax.transAxes, color=GREEN, ha="center")
    clean(ax, "b", "Gibbs-Thomson crossover", r"$R_0/R_{\rm crit}$",
          "normalized radius change")

    ax = axes[1, 0]
    t = np.array([0, 7, 16, 25, 34, 42, 50, 57])
    psi = np.array([177, 154, 155, 165, 160, 130, 148, 145])
    ax.plot(t, psi, "o-", color=BLUE)
    ax.axhline(145, color=ORANGE, ls="--", label="force-balance target")
    clean(ax, "c", "Groove-angle relaxation", "time [ms]", r"$\psi$ [deg]")
    ax.legend(frameon=False)

    ax = axes[1, 1]
    t = np.array([0, 2, 4, 6, 8, 10.5])
    r = np.array([200, 204, 209, 214, 219, 229])
    ax.plot(t, r, "o-", color=GREEN, label="GPK")
    ax.plot([0, 10.5], [200, 228], "--", color=ORANGE, label="local theory")
    clean(ax, "d", "Supercritical particle growth", r"time [$\mu$s]", "radius [nm]")
    ax.legend(frameon=False)

    save(fig, "fig5_extended_validation.pdf")


def benchmark() -> None:
    cost_dir = output_dir("gpk_pf_comparison")
    resolution = np.genfromtxt(cost_dir / "accuracy_matched_resolution.csv",
                               delimiter=",", names=True, dtype=None,
                               encoding="utf-8")
    scaling = np.genfromtxt(cost_dir / "accuracy_matched_cost_scaling.csv",
                            delimiter=",", names=True)
    gmask = resolution["method"] == "gpk"
    pmask = resolution["method"] == "phase_field"
    gr = resolution[gmask]
    pr = resolution[pmask]
    problem_area = scaling["domain_um"]**2
    runtime_fits = {
        key: np.polyfit(np.log(problem_area), np.log(scaling[key]), 1)
        for key in ("gpk_median_seconds", "pf_median_seconds")
    }

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.25), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(gr["grid_n"], gr["label_agreement_vs_384"], "o-", color=BLUE,
            label="GPK")
    ax.plot(pr["grid_n"], pr["label_agreement_vs_384"], "s-", color=ORANGE,
            label="phase field")
    ax.axhline(0.95, color="black", lw=0.8, ls=":", label="required agreement")
    ax.axvline(192, color=BLUE, lw=0.9, ls="--")
    ax.set_ylim(0.92, 1.004)
    clean(ax, "a", "Morphology convergence", "grid side, $N$",
          "label agreement with $384^2$")
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")

    ax = axes[0, 1]
    energy_error = 100 * pr["pf_surface_energy_relative_error"]
    ax.semilogy(pr["grid_n"], energy_error, "s-", color=ORANGE,
                label="discrete PF profile")
    ax.axhline(1.0, color="black", lw=0.8, ls=":", label="1% energy requirement")
    ax.axvline(96, color=GRAY, lw=0.9, ls="--")
    ax.axvline(256, color=PURPLE, lw=1.0, ls="--")
    # Put both resolution annotations in whitespace and give them an opaque
    # backing so they cannot collide visually with the orange data trace.
    note_box = dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.92)
    ax.text(112, 1.28, "morphology\nonly", fontsize=7.2, color=GRAY,
            va="bottom", bbox=note_box)
    ax.text(249, 3.2, "full PF\naccuracy", fontsize=7.2, color=PURPLE,
            ha="right", va="top", bbox=note_box)
    clean(ax, "b", "Diffuse-interface resolution", "PF grid side, $N$",
          "surface-energy error [%]")
    ax.legend(frameon=False, fontsize=7.3, loc="upper right")

    ax = axes[1, 0]
    for median_key, min_key, max_key, color, marker, label in [
        ("gpk_median_seconds", "gpk_min_seconds", "gpk_max_seconds",
         BLUE, "o", "GPK"),
        ("pf_median_seconds", "pf_min_seconds", "pf_max_seconds",
         ORANGE, "s", "phase field"),
    ]:
        slope, _ = runtime_fits[median_key]
        ax.fill_between(problem_area, scaling[min_key], scaling[max_key],
                        color=color, alpha=0.16, linewidth=0)
        ax.loglog(problem_area, scaling[median_key], marker + "-", color=color,
                  label=rf"{label}, $\beta_t={slope:.2f}$")
    clean(ax, "c", "Accuracy-matched cost", r"domain area [$\mu$m$^2$]",
          "kernel wall time [s]")
    ax.legend(frameon=False, fontsize=7.7)

    ax = axes[1, 1]
    ax.plot(scaling["domain_um"], scaling["pf_over_gpk_time"], "o-",
            color=PURPLE, label="wall time")
    ax.plot(scaling["domain_um"], scaling["pf_over_gpk_memory"], "s-",
            color=GREEN, label="array capacity")
    ax.axhline(1, color="black", lw=0.8, ls=":")
    ax.set_yscale("log")
    ax.set_ylim(0.16, 1.25 * max(np.max(scaling["pf_over_gpk_time"]),
                                np.max(scaling["pf_over_gpk_memory"])))
    clean(ax, "d", "Measured cost ratios", r"domain side [$\mu$m]",
          "phase field / GPK")
    ax.legend(frameon=False, fontsize=7.7)
    save(fig, "fig6_benchmark.pdf")


def _validation_module():
    path = REPO / "scripts" / "validate_grain_rex.py"
    spec = importlib.util.spec_from_file_location("gpk_validation", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def grain_growth() -> None:
    mod = _validation_module()
    sim_dir = output_dir("validation_grain_growth_dense_2d")
    snaps = mod.load_snaps(sim_dir)
    times, radii, counts = _characteristic_grain_series(sim_dir)
    r2 = radii**2
    # Use the complete, predeclared saved-state series.  The present trajectory
    # is more nearly linear over the full interval than over the formerly used
    # late-time subset, and using every state avoids a fit-window choice made
    # after inspecting the endpoint fluctuations caused by grain extinctions.
    fit_mask = np.ones(times.shape, dtype=bool)
    slope, intercept, rval, *_ = stats.linregress(times, r2)
    tfit = np.linspace(times.min(), times.max(), 200)

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), constrained_layout=True)
    ax = axes[0, 0]
    ax.scatter(times, r2, s=24, color=BLUE, label="GPK")
    ax.plot(tfit, intercept + slope*tfit, color=GREEN,
            label=rf"all-state fit: {slope:.3f} $\mu$m$^2$/s")
    clean(ax, "a", "Parabolic coarsening", "time [s]", r"$R_{\rm char}^2$ [$\mu$m$^2$]")
    ax.legend(**SI_LEGEND)
    ax.text(0.04, 0.89, f"$R^2={rval**2:.3f}$", transform=ax.transAxes,
            color=GREEN, bbox=SI_TEXT_BOX)

    ax = axes[0, 1]
    fit_r2 = intercept + slope*times
    residual = (r2-fit_r2)/np.maximum(fit_r2, 1e-12)*100
    ax.plot(times, residual, "o-", color=BLUE)
    ax.axhline(0, color=GRAY, lw=0.8)
    clean(ax, "b", "Fit residual", "time [s]", "relative residual [%]")

    ax = axes[1, 0]
    r0 = mod.radius_distribution(snaps[0]); rf = mod.radius_distribution(snaps[-1])
    bins = np.linspace(0, max(r0.max(), rf.max())*1.05, 15)
    ax.hist(r0, bins=bins, alpha=0.65, color=CYAN, label="initial")
    ax.hist(rf, bins=bins, alpha=0.65, color=PURPLE, label="final")
    clean(ax, "c", "Grain-size distribution", r"equivalent radius [$\mu$m]", "count")
    ax.legend(**SI_LEGEND)

    ax = axes[1, 1]
    ax.plot(times, counts, "o-", color=PURPLE)
    clean(ax, "d", "Topology evolution", "time [s]", "number of grains")
    save(fig, "figS6_grain_growth.pdf")


def recrystallisation() -> None:
    mod = _validation_module()
    snaps2 = mod.load_snaps(output_dir("validation_rex_current_2d"))
    snaps3 = mod.load_snaps(output_dir("validation_rex_current_3d"))
    lnt2, y2, t2, x2 = mod.avrami_data(snaps2)
    lnt3, y3, t3, x3 = mod.avrami_data(snaps3)
    s2, b2, r2, *_ = stats.linregress(lnt2, y2)
    s3, b3, r3, *_ = stats.linregress(lnt3, y3)

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), constrained_layout=True)
    for ax, lnt, yy, slope, intercept, rr, dim, color, panel in [
        (axes[0,0], lnt2, y2, s2, b2, r2, 2, BLUE, "a"),
        (axes[0,1], lnt3, y3, s3, b3, r3, 3, PURPLE, "b")]:
        xx = np.linspace(lnt.min(), lnt.max(), 150)
        ax.scatter(lnt, yy, s=23, color=color, label="GPK")
        ax.plot(xx, slope*xx+intercept, color=GREEN,
                label=f"fit $n={slope:.2f}$")
        cx, cy = lnt.mean(), yy.mean()
        ax.plot(xx, dim*(xx-cx)+cy, "--", color=ORANGE,
                label=f"theory $n={dim}$")
        clean(ax, panel, f"{dim}D Avrami plot", r"$\ln t$", r"$\ln[-\ln(1-X)]$")
        ax.legend(**SI_LEGEND)
        ax.text(0.04, 0.88, f"$R^2={rr**2:.3f}$", transform=ax.transAxes,
                bbox=SI_TEXT_BOX)

    for ax, tt, xxdata, slope, intercept, dim, color, panel in [
        (axes[1,0], t2, x2, s2, b2, 2, BLUE, "c"),
        (axes[1,1], t3, x3, s3, b3, 3, PURPLE, "d")]:
        mask = tt > 0
        fine = np.linspace(tt[mask].min(), tt[mask].max(), 240)
        fit = 1-np.exp(-np.exp(slope*np.log(fine)+intercept))
        ax.scatter(tt[mask], 100*xxdata[mask], s=21, color=color, label="GPK")
        ax.plot(fine, 100*fit, color=GREEN, label="JMAK fit")
        clean(ax, panel, f"{dim}D transformed fraction", "time [s]", "$X$ [%]")
        ax.set_ylim(0, 104)
        ax.legend(**SI_LEGEND)
    save(fig, "figS7_recrystallisation.pdf")


def _bicrystal_grid(path: Path) -> np.ndarray:
    header = path.read_text(encoding="utf-8", errors="replace").splitlines()[0]
    fields = {key: int(value) for key, value in re.findall(r"(Ny|Nx)=(\d+)", header)}
    values = np.loadtxt(path, comments="#", skiprows=2, dtype=int)
    return values.reshape(fields["Ny"], fields["Nx"])


def bicrystal_validation() -> None:
    """Current-solver bicrystal velocity, stationarity and field audits."""
    base = output_dir("validate_extended_base_models")
    fig = plt.figure(figsize=(7.2, 8.30), constrained_layout=False)
    outer = fig.add_gridspec(
        5, 2, left=0.065, right=0.99, top=0.975, bottom=0.025,
        height_ratios=(1.18, 1.0, 1.0, 1.0, 1.0), hspace=0.34, wspace=0.22)

    ax = fig.add_subplot(outer[0, 0])
    names, ratios = [], []
    ratio_by_key = {}
    for key, label in [("g2", "stored energy"), ("g5", "solidification"),
                       ("g7", "reduction"), ("g8", "Ostwald"),
                       ("g10", "multiphase")]:
        sim = np.loadtxt(next(base.glob(f"{key}_area_t.csv")), delimiter=",", skiprows=1)
        theory = np.loadtxt(next(base.glob(f"{key}_area_t_theory.csv")), delimiter=",", skiprows=1)
        slope_sim = stats.linregress(sim[:, 0], sim[:, 1]).slope
        slope_exact = (theory[-1, 1] - theory[0, 1]) / (theory[-1, 0] - theory[0, 0])
        ratio = slope_sim / slope_exact
        names.append(label); ratios.append(ratio); ratio_by_key[key] = ratio
    xpos = np.arange(len(names))
    point_colours = [BLUE, ORANGE, GREEN, PURPLE, CYAN]
    fractional_errors = np.asarray(ratios) - 1.0
    velocity_error_ppm = 1.0e6 * fractional_errors
    ax.axhline(0.0, color="black", lw=0.9, ls="--", label="exact input velocity")
    ax.vlines(xpos, 0.0, velocity_error_ppm, color=point_colours, lw=1.2, alpha=0.8)
    ax.scatter(xpos, velocity_error_ppm, s=58, c=point_colours, edgecolors="white",
               linewidths=0.9, zorder=3)
    # Two-line labels remain inside the top row; rotated labels previously
    # intruded into the headings of panels c and d.
    display_names = ["stored\nenergy", "solidification", "reduction",
                     "Ostwald", "multiphase"]
    ax.set_xticks(xpos, display_names)
    ax.tick_params(axis="x", labelsize=7.1, pad=2)
    ax.set_ylim(-0.145, 0.075)
    for xval, fractional_error, error, colour in zip(
            xpos, fractional_errors, velocity_error_ppm, point_colours):
        offset = 0.014 if error >= 0.0 else -0.014
        ax.text(xval, error + offset,
                rf"${signed_scientific_latex(float(fractional_error))}$", ha="center",
                va="bottom" if error >= 0.0 else "top", fontsize=6.3,
                color=colour, fontweight="bold", bbox=SI_TEXT_BOX)
    clean(ax, "a", "Driven flat-front bicrystals", ylabel="velocity error [ppm]")
    ax.set_yticks([-0.10, -0.05, 0.0, 0.05], ["-0.10", "-0.05", "0", "+0.05"])
    ax.legend(**SI_LEGEND, fontsize=7.2, loc="lower left")

    ax = fig.add_subplot(outer[0, 1])
    stationary = []
    stationary_counts = {}
    for key in ("g1", "g6", "g9"):
        before = _bicrystal_grid(base / f"{key}_grid_before.csv")
        after = _bicrystal_grid(base / f"{key}_grid_after.csv")
        changed = int(np.count_nonzero(before != after))
        stationary.append(changed)
        stationary_counts[key] = (changed, int(before.size))
    xstationary = np.arange(len(stationary))
    colours = [BLUE, GREEN, PURPLE]
    ax.axhspan(-0.12, 0.12, color="#ECE3F5", zorder=0)
    ax.axhline(0.0, color="#222222", lw=0.9, zorder=1)
    ax.scatter(xstationary, stationary, s=78, c=colours, edgecolors="white",
               linewidths=0.9, zorder=3)
    ax.set_xticks(xstationary,
                  ["isotropic", "phase-pair\n$\\sigma$", "anisotropic\n$\\sigma$"])
    for xpos, (key, value) in enumerate(zip(("g1", "g6", "g9"), stationary)):
        alignment = "left" if xpos == 0 else "right" if xpos == 2 else "center"
        offset = 0.035 if xpos == 0 else -0.035 if xpos == 2 else 0.0
        total = stationary_counts[key][1]
        ax.text(xpos + offset, 0.22, f"{value:,d} of {total:,d}", ha=alignment, va="bottom",
                fontsize=8.2, fontweight="bold", color=colours[xpos],
                bbox=SI_TEXT_BOX)
    clean(ax, "b", "Zero-force planar stationarity", ylabel="changed cells [count]")
    # Keep the panel letter clear of the upper y tick at the compact SI width.
    ax.texts[-1].set_position((-0.13, 1.15))
    ax.tick_params(axis="x", labelsize=7.1, pad=2)
    ax.set_xlim(-0.22, 2.22)
    ax.set_ylim(-0.16, 1.05)
    ax.set_yticks([0, 1])

    def crop_uniform_canvas(panel_image: np.ndarray) -> np.ndarray:
        """Remove only uniform PyVista canvas above and below raw cell data."""
        rgb = panel_image[..., :3]
        background = rgb[0, 0]
        non_background = np.max(np.abs(rgb - background), axis=2) > 0.025
        occupied_rows = np.flatnonzero(np.any(non_background, axis=1))
        if occupied_rows.size == 0:
            return panel_image
        pad = max(3, int(round(0.02 * panel_image.shape[0])))
        row0 = max(0, int(occupied_rows[0]) - pad)
        row1 = min(panel_image.shape[0], int(occupied_rows[-1]) + pad + 1)
        return panel_image[row0:row1]

    cases = [
        ("g1", "Isotropic, zero force",
         f"stationary: {stationary_counts['g1'][0]} of {stationary_counts['g1'][1]:,d} cells",
         {1: FILL_BLUE, 2: FILL_YELLOW}),
        ("g2", "Stored-energy driving",
         rf"translation: $v/v_*-1={signed_scientific_latex(ratio_by_key['g2']-1.0)}$",
         {0: FILL_GREY, 1: FILL_RED}),
        ("g5", "Undercooling driving",
         rf"translation: $v/v_*-1={signed_scientific_latex(ratio_by_key['g5']-1.0)}$",
         {0: FILL_LIGHTBLUE, 1: FILL_RED}),
        ("g6", "Phase-pair energy, zero force",
         f"stationary: {stationary_counts['g6'][0]} of {stationary_counts['g6'][1]:,d} cells",
         {0: FILL_WHITE, 1: FILL_GREEN, 2: FILL_BLUE}),
        ("g7", "Reduction driving",
         rf"translation: $v/v_*-1={signed_scientific_latex(ratio_by_key['g7']-1.0)}$",
         {0: FILL_RED, 1: FILL_GREY}),
        ("g8", "Supersaturation driving",
         rf"translation: $v/v_*-1={signed_scientific_latex(ratio_by_key['g8']-1.0)}$",
         {0: FILL_GREY, 1: FILL_PURPLE}),
        ("g9", "Anisotropic energy, zero force",
         f"stationary: {stationary_counts['g9'][0]} of {stationary_counts['g9'][1]:,d} cells",
         {1: FILL_BLUE, 2: FILL_YELLOW}),
        ("g10", "Multiphase Gibbs driving",
         rf"translation: $v/v_*-1={signed_scientific_latex(ratio_by_key['g10']-1.0)}$",
         {0: FILL_GREY, 1: FILL_GREEN}),
    ]
    for index, (key, title, result, semantic) in enumerate(cases):
        slot = outer[1 + index//2, index % 2]
        panel = chr(ord("c") + index)
        sub = slot.subgridspec(4, 2, height_ratios=(0.24, 0.18, 0.16, 1),
                               hspace=0.02, wspace=0.045)
        heading = fig.add_subplot(sub[0, :]); heading.axis("off")
        heading.text(0.0, 0.55, panel, fontweight="bold", fontsize=9.5, va="center")
        heading.text(0.09, 0.55, title, fontweight="bold", fontsize=8.2, va="center")
        result_ax = fig.add_subplot(sub[1, :]); result_ax.axis("off")
        result_ax.text(0.99, 0.55, result, color=GRAY, fontsize=7.8,
                       ha="right", va="center")
        state_ax = fig.add_subplot(sub[2, :]); state_ax.axis("off")
        state_ax.text(0.25, 0.45, "initial", fontsize=7.9, ha="center", va="center")
        state_ax.text(0.75, 0.45, "final", fontsize=7.9, ha="center", va="center")
        for col, state in enumerate(("before", "after")):
            iax = fig.add_subplot(sub[3, col])
            asset_state = "initial" if state == "before" else "final"
            panel_image = crop_uniform_canvas(
                _pyvista_panel(f"bicrystal_{key}_{asset_state}.png")
            )
            iax.imshow(panel_image, interpolation="nearest", aspect="auto")
            iax.set_xticks([]); iax.set_yticks([])
            for spine in iax.spines.values():
                spine.set_linewidth(0.5); spine.set_color("#555555")

    save(fig, "figS9_bicrystal_validation.pdf")


def _shape_descriptors(labels: np.ndarray) -> tuple[float, float]:
    """Second-moment aspect ratio and boundary radial coefficient of variation."""
    mask = labels == 1
    yy, xx = np.nonzero(mask)
    centred = np.column_stack((xx - xx.mean(), yy - yy.mean()))
    eigenvalues = np.linalg.eigvalsh(centred.T @ centred / len(centred))
    aspect = float(np.sqrt(eigenvalues[-1] / eigenvalues[0]))
    boundary = mask & (
        (mask != np.roll(mask, 1, axis=0)) |
        (mask != np.roll(mask, -1, axis=0)) |
        (mask != np.roll(mask, 1, axis=1)) |
        (mask != np.roll(mask, -1, axis=1)))
    by, bx = np.nonzero(boundary)
    radius = np.hypot(bx - xx.mean(), by - yy.mean())
    radial_cv = float(radius.std() / radius.mean())
    return aspect, radial_cv


def curved_bicrystal_validation() -> None:
    """Circular, elliptical and square two-grain capillary regressions."""
    base = output_dir("validate_extended_base_models")
    fig = plt.figure(figsize=(7.2, 5.15), constrained_layout=False)
    outer = fig.add_gridspec(2, 1, left=0.065, right=0.99, top=0.975, bottom=0.045,
                             height_ratios=(1.0, 1.18), hspace=0.30)
    top = outer[0].subgridspec(1, 2, wspace=0.27)

    ax = fig.add_subplot(top[0, 0])
    circle = np.loadtxt(base / "g13_area_t.csv", delimiter=",", skiprows=1)
    circle_exact = np.loadtxt(base / "g13_area_t_theory.csv", delimiter=",", skiprows=1)
    ax.plot(circle[:, 0]*1e3, circle[:, 1]/np.pi*1e12, "o", color=BLUE,
            label="GPK")
    ax.plot(circle_exact[:, 0]*1e3, circle_exact[:, 1]/np.pi*1e12, "--",
            color=ORANGE, label=r"$R_0^2-2M\sigma t$")
    clean(ax, "a", "Circular bicrystal shrinkage", "time [ms]", r"$R^2$ [$\mu$m$^2$]")
    ax.legend(**SI_LEGEND)

    ax = fig.add_subplot(top[0, 1])
    ratios, errors = [], []
    for key in ("g13", "g14", "g15"):
        history = np.loadtxt(base / f"{key}_area_t.csv", delimiter=",", skiprows=1)
        exact = np.loadtxt(base / f"{key}_area_t_theory.csv", delimiter=",", skiprows=1)
        ratio = (stats.linregress(history[:, 0], history[:, 1]).slope /
                 stats.linregress(exact[:, 0], exact[:, 1]).slope)
        ratios.append(ratio); errors.append(100*abs(ratio-1))
    xpos = np.arange(3)
    ax.bar(xpos, ratios, color=[BLUE, GREEN, PURPLE], width=0.58)
    ax.axhline(1.0, color=ORANGE, ls="--", lw=1.2, label="Gauss-Bonnet")
    for x, ratio, error in zip(xpos, ratios, errors):
        ax.text(x, ratio+0.016, f"{error:.2f}%", ha="center", va="bottom",
                fontsize=8.1, bbox=SI_TEXT_BOX)
    ax.set_xticks(xpos, ["circle", "ellipse", "square"])
    ax.set_ylim(0.90, 1.07)
    clean(ax, "b", "Shape-independent area loss", ylabel="measured / exact")
    ax.legend(**SI_LEGEND, loc="lower right")

    gallery = outer[1].subgridspec(1, 3, wspace=0.15)
    cases = [
        ("g13", "Circle", BLUE, "circle remains circular"),
        ("g14", "Ellipse", GREEN, "aspect ratio"),
        ("g15", "Square", PURPLE, "radial variation"),
    ]
    for idx, (key, title, colour, metric_name) in enumerate(cases):
        sub = gallery[0, idx].subgridspec(2, 2, height_ratios=(0.34, 1),
                                         hspace=0.05, wspace=0.035)
        before = _bicrystal_grid(base / f"{key}_grid_before.csv")
        after = _bicrystal_grid(base / f"{key}_grid_after.csv")
        ar0, cv0 = _shape_descriptors(before)
        ar1, cv1 = _shape_descriptors(after)
        heading = fig.add_subplot(sub[0, :]); heading.axis("off")
        heading.text(0.0, 0.72, f"{chr(ord('c') + idx)}   {title}",
                     fontweight="bold", fontsize=8.2, va="center")
        if metric_name == "aspect ratio":
            outcome = f"AR {ar0:.2f} -> {ar1:.2f}"
        elif metric_name == "radial variation":
            outcome = f"radial CV {cv0:.2f} -> {cv1:.2f}"
        else:
            outcome = f"radial CV {cv0:.3f} -> {cv1:.3f}"
        heading.text(0.0, 0.16, outcome, ha="left", va="center",
                     color=GRAY, fontsize=7.2)
        for col, (state, labels) in enumerate((("initial", before), ("final", after))):
            iax = fig.add_subplot(sub[1, col])
            iax.imshow(_pyvista_panel(f"curved_{key}_{state}.png"),
                       interpolation="nearest")
            iax.set_aspect("equal")
            iax.set_title(state, fontsize=8.0, pad=1.5)
            iax.set_xticks([]); iax.set_yticks([])
            for spine in iax.spines.values():
                spine.set_linewidth(0.5); spine.set_color("#555555")

    save(fig, "figS14_curved_bicrystal_validation.pdf")


def _mini_state(ax, x0: float, kind: str, final: bool) -> None:
    y0, w, h = 0.19, 0.36, 0.43
    if kind in {"particle_shrink", "particle_grow", "pore"}:
        ax.add_patch(Rectangle((x0, y0), w, h, color=MATRIX if kind != "pore" else PRODUCT,
                               ec="0.45", lw=0.5))
        radius = {"particle_shrink": (0.11, 0.035), "particle_grow": (0.08, 0.145),
                  "pore": (0.12, 0.045)}[kind][int(final)]
        ax.add_patch(Circle((x0+w/2, y0+h/2), radius,
                            color="white" if kind == "pore" else PRODUCT, ec="0.4", lw=0.4))
    elif kind == "groove":
        ax.add_patch(Rectangle((x0, y0), w, h, color=MATRIX, ec="0.45", lw=0.5))
        if final:
            ax.plot([x0, x0+0.12, x0+0.18, x0+0.24, x0+w],
                    [y0+h, y0+h*0.96, y0+h*0.72, y0+h*0.96, y0+h], color=PRODUCT, lw=3)
        else:
            ax.plot([x0, x0+w], [y0+h, y0+h], color=PRODUCT, lw=3)
        ax.plot([x0+w/2, x0+w/2], [y0, y0+h*(0.72 if final else 1)], color="white", lw=1.2)
    else:
        frac0, frac1 = {"static": (0.50,0.50), "front": (0.50,0.72),
                        "anisotropic": (0.50,0.50)}[kind]
        frac = frac1 if final else frac0
        ax.add_patch(Rectangle((x0, y0), w, h, color=MATRIX, ec="0.45", lw=0.5))
        ax.add_patch(Rectangle((x0, y0), w*frac, h, color=PRODUCT, ec="none"))


def extended_microstructures() -> None:
    cases = [
        ("G1", "zero-force flat front", "static"),
        ("G2", "recrystallisation", "front"),
        ("G3", "subcritical nucleus", "particle_shrink"),
        ("G4", "critical-radius growth", "particle_grow"),
        ("G5", "solidification front", "front"),
        ("G6", "zero-force groove pair", "static"),
        ("G7", "reduction front", "front"),
        ("G8", "coarsening front", "front"),
        ("G9", "anisotropic flat front", "anisotropic"),
        ("G10", "precipitation front", "front"),
        ("G11", "groove relaxation", "groove"),
        ("G12", "supercritical particle", "particle_grow"),
    ]
    fig, axes = plt.subplots(4, 3, figsize=(7.2, 8.4), constrained_layout=True)
    for ax, (gid, title, kind) in zip(axes.flat, cases):
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
        ax.set_title(f"{gid}  {title}", loc="left", fontsize=8.5, fontweight="bold", pad=1)
        _mini_state(ax, 0.04, kind, False)
        _mini_state(ax, 0.60, kind, True)
        ax.annotate("", xy=(0.57, 0.41), xytext=(0.43, 0.41),
                    arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.1))
        ax.text(0.22, 0.08, "initial", ha="center", color=GRAY, fontsize=7)
        ax.text(0.78, 0.08, "final", ha="center", color=GRAY, fontsize=7)
    save(fig, "figS8_extended_microstructures.pdf")


def timestep_schematic() -> None:
    """Eight operations of one GPK timestep, with the symbols used in the text."""
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    ax.set_xlim(0, 17.0); ax.set_ylim(-0.1, 8.0); ax.axis("off")
    w, h = 3.8, 3.0
    xs = [0.1, 4.4, 8.7, 13.0]
    top_y, bottom_y = 4.3, 0.1
    # Boxes are tinted by role: stored state white, geometry blue, material
    # physics yellow and the label change red.
    geometry = dict(edge="#222222", face="#DCE7F8")   # light blue
    physics = dict(edge="#222222", face="#FFF0B3")    # light yellow
    update = dict(edge="#222222", face="#F8CFCF")     # light red
    state = dict(edge="#222222", face="#FFFFFF")
    steps = [
        ("ACTIVE CELLS", "cell $i$ is active when\na neighbour has\na different label", state),
        ("INDICATOR", "$I_{g_i}=1$ in cells\nwith label $g_i$,\n$I_{g_i}=0$ elsewhere", geometry),
        ("DERIVATIVES", r"gradient $\nabla\widetilde{I}_{g_i}$" "\n" r"Hessian $H(\widetilde{I}_{g_i})$", geometry),
        ("GEOMETRY", r"outward normal $\mathbf{n}_{g_i}$" "\n" r"curvature $\kappa_{g_i}$", geometry),
        ("PAIR PHYSICS", "resident $p$\ncandidate $q$\n" r"$M_{pq}$, $\Delta G_{q-p}$, $\gamma_{pq}$", physics),
        ("VELOCITY", r"$v_{n,i}=M_{pq}(\Delta G_{q-p}$" "\n" r"$-\,\gamma_{pq}\kappa_p)$" "\n" r"pair average, $\Delta t$", physics),
        ("TOPOLOGY", r"trial fraction $\alpha^*$" "\nTCKT check\nbalanced remap", update),
        ("BULK FIELDS", r"advance $c$, $T$, $\rho$" "\n" r"write state $S^{n+1}$", state),
    ]
    positions = [(xs[k], top_y) for k in range(4)] + [(xs[3 - k], bottom_y) for k in range(4)]
    for k, ((head, body, style), xy) in enumerate(zip(steps, positions), start=1):
        _workflow_box(ax, xy, w, h, k, head, body, heading_fontsize=7.6,
                      body_fontsize=7.4, **style)
    arrow = dict(arrowstyle="-|>", color="#333333", lw=1.0, mutation_scale=10)
    for k in range(3):
        ax.add_patch(FancyArrowPatch((xs[k] + w + 0.04, top_y + h / 2),
                                     (xs[k + 1] - 0.04, top_y + h / 2), **arrow))
        ax.add_patch(FancyArrowPatch((xs[3 - k] - 0.04, bottom_y + h / 2),
                                     (xs[2 - k] + w + 0.04, bottom_y + h / 2), **arrow))
    ax.add_patch(FancyArrowPatch((xs[3] + w / 2, top_y - 0.04),
                                 (xs[3] + w / 2, bottom_y + h + 0.04), **arrow))
    ax.add_patch(FancyArrowPatch((xs[0] + w / 2, bottom_y + h + 0.04),
                                 (xs[0] + w / 2, top_y - 0.04),
                                 linestyle=(0, (3, 2)), **arrow))
    ax.text(xs[0] + w / 2 + 0.15, (top_y + bottom_y + h) / 2,
            r"next timestep $n\rightarrow n+1$", ha="left", va="center",
            fontsize=7.4, color="#333333")
    ax.text(0.1, 7.95, r"ONE GPK TIMESTEP: FROM STATE $S^{n}$ TO STATE $S^{n+1}$",
            ha="left", va="top", fontsize=8.4, fontweight="bold", color="#333333")
    save(fig, "fig2_timestep_operations.pdf")


def gpk_phase_field_comparison() -> None:
    """Matched categorical microstructures from a common Voronoi state."""
    base = output_dir("gpk_pf_comparison")
    metrics = np.genfromtxt(base / "snapshot_metrics.csv", delimiter=",", names=True)
    times = np.atleast_1d(metrics["time"])

    def load_label(kind: str, index: int) -> np.ndarray:
        data = np.loadtxt(base / f"{kind}_snap_{index}.csv", delimiter=",", skiprows=3,
                          dtype=int)
        side = int(round(np.sqrt(data.shape[0])))
        if side * side != data.shape[0]:
            raise ValueError(f"Expected a square label grid, got {data.shape[0]} cells")
        return data[:, 2].reshape(side, side)

    def draw_microstructure(ax, asset_name: str) -> None:
        ax.imshow(_pyvista_panel(asset_name), interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.55); spine.set_color("#555555")

    fig = plt.figure(figsize=(7.2, 3.45), constrained_layout=False)
    gs = fig.add_gridspec(2, 4, left=0.040, right=0.995, top=0.90, bottom=0.045,
                          hspace=0.10, wspace=0.13)

    # Panel a is the one common initial condition, constructed directly by a
    # periodic Voronoi tessellation and passed unchanged to both solvers.
    ax0 = fig.add_subplot(gs[:, 0])
    draw_microstructure(ax0, "pf_common_0.png")
    ax0.set_title("common periodic\nVoronoi state\n$t=0$ s; GPK = PF",
                  fontsize=9.0, fontweight="bold", pad=3)
    ax0.text(-0.14, 1.08, "a", transform=ax0.transAxes,
             fontweight="bold", fontsize=10, va="top")

    chosen = [4, 8, 16]  # 1, 2 and 4 s for the 17-state output
    for col, idx in enumerate(chosen, start=1):
        top = fig.add_subplot(gs[0, col])
        bottom = fig.add_subplot(gs[1, col])
        draw_microstructure(top, f"pf_gpk_{idx}.png")
        draw_microstructure(bottom, f"pf_pf_{idx}.png")
        top.set_title(rf"$t={times[idx]:.0f}$ s", fontsize=9.0,
                      fontweight="bold", pad=2)
        top.text(-0.13, 1.13, chr(ord("a") + col), transform=top.transAxes,
                 fontweight="bold", fontsize=10, va="top")
        if col == 1:
            top.text(-0.10, 0.50, "GPK", transform=top.transAxes, rotation=90,
                     ha="right", va="center", fontsize=8.4, fontweight="bold")
            bottom.text(-0.10, 0.50, "phase field", transform=bottom.transAxes,
                        rotation=90, ha="right", va="center", fontsize=8.4,
                        fontweight="bold")

    save(fig, "fig10_gpk_phase_field_comparison.pdf")


def main() -> None:
    """Draw every figure whose input data is available.

    A figure whose simulation fields are not in the repository (see
    paper/LARGE_DATA.md) is skipped with a message instead of stopping the run.
    """
    figures = [overview, profile, geometry_pipeline, curvature_signs, gaussian_accuracy,
               stability, topology_transactions, gibbs_thomson, foundational_validation,
               qualitative_2d_gallery, extended_validation, benchmark, grain_growth,
               recrystallisation, bicrystal_validation, curved_bicrystal_validation,
               gpk_phase_field_comparison, timestep_schematic]
    skipped = []
    for figure in figures:
        try:
            figure()
        except (FileNotFoundError, RuntimeError, ValueError, IndexError, KeyError) as exc:
            skipped.append(figure.__name__)
            print(f"skipped {figure.__name__}: input data missing ({exc}); see paper/LARGE_DATA.md")
    # The 3D gallery is drawn separately with ParaView's PyVista/VTK by
    # render_3d_microstructures.py so surface shading and depth ordering are kept.
    if not (OUT / "fig_qualitative_3d_curvature_longtime.pdf").exists():
        print("note: run render_3d_microstructures.py with pvpython for the 3D gallery")
    print(f"Generated {len(figures) - len(skipped)} of {len(figures)} figures in {OUT}.")


if __name__ == "__main__":
    main()
