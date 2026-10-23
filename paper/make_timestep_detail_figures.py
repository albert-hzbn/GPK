"""Schematics for the timestep details of Section 2.2.

fig_velocity_subcell.pdf
    (a) pair-conditioned velocity average at a triple junction,
    (b) Gaussian weights of the velocity average and the remap score,
    (c) sub-cell fraction update and crossing.
fig_tckt_bgr.pdf
    (a)-(d) topology-certified kinetic transaction (TCKT) cases,
    (e)-(f) balanced Gaussian remap (BGR) of a protrusion/indentation pair.

The drawings use primary and secondary colours (blue, yellow, green and red)
with dark grey line work, shared with Figures 1 and 2.  All counts
shown (pieces P_p and P_q, label counts and the boundary-face count) are
computed from the drawn label arrays with the rules stated in the manuscript.
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = Path(os.environ.get("GPK_FIGURE_OUT", ROOT / "figures"))

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 8.0,
    "axes.titlesize": 8.4,
    "axes.labelsize": 8.0,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "pdf.fonttype": 42,
    "mathtext.fontset": "dejavusans",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04,
    "text.color": "#222222",
    "axes.labelcolor": "#222222",
    "axes.edgecolor": "#222222",
    "xtick.color": "#222222",
    "ytick.color": "#222222",
})

# Label fills: resident p, candidate q, third grain r.
# Blue resident p, yellow candidate q, purple third grain r, red for the cell
# under test, blue/purple for secondary curves (no red-green pairs, npj).
P_FILL, Q_FILL, R_FILL = "#7EA6E0", "#FFE38A", "#B08AD6"
LINE = "#222222"
GRID = "#555555"
ACCENT = "#D62020"     # red
OLIVE = "#7B2FA8"      # purple (accepted, remap curve)
INDIGO = "#1F4FB4"     # blue (3-D velocity curve)
P, Q, R = 1, 0, 2
FILLS = {P: P_FILL, Q: Q_FILL, R: R_FILL}


def panel_label(ax, letter, title):
    ax.set_title(title, loc="left", pad=3)
    ax.text(0.0, 1.0, letter, transform=ax.transAxes, fontweight="bold",
            fontsize=9.5, va="bottom", ha="right",
            # offset left of the title so the two never touch
            )


def draw_cells(ax, field, lw=0.35):
    ny, nx = field.shape
    for y in range(ny):
        for x in range(nx):
            ax.add_patch(Rectangle((x, y), 1, 1, fc=FILLS[int(field[y, x])],
                                   ec=GRID, lw=lw))
    ax.set_xlim(-0.1, nx + 0.1)
    ax.set_ylim(-0.1, ny + 0.1)
    ax.set_aspect("equal")
    ax.axis("off")


def text_label(ax, x, y, s, size=9):
    ax.text(x, y, s, ha="center", va="center", fontsize=size, style="italic",
            color=LINE)


# --------------------------------------------------------------------------
# Connectivity helpers (face connectivity, as in the TCKT rules)
# --------------------------------------------------------------------------
def components(mask):
    lab = np.zeros(mask.shape, dtype=int)
    n = 0
    for start in zip(*np.nonzero(mask)):
        if lab[start]:
            continue
        n += 1
        stack = [start]
        lab[start] = n
        while stack:
            y, x = stack.pop()
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                yy, xx = y + dy, x + dx
                if (0 <= yy < mask.shape[0] and 0 <= xx < mask.shape[1]
                        and mask[yy, xx] and not lab[yy, xx]):
                    lab[yy, xx] = n
                    stack.append((yy, xx))
    return lab, n


def local_pieces(field, centre, label):
    y, x = centre
    block = (field[y - 1:y + 2, x - 1:x + 2] == label).copy()
    block[1, 1] = False
    _, n = components(block)
    return n, block


def tckt(field, centre, p, q):
    """Return (P_p, P_q, accepted) following the resident and candidate rules."""
    y, x = centre
    pp, block_p = local_pieces(field, centre, p)
    pq, block_q = local_pieces(field, centre, q)

    def reconnect(label, block):
        mask = field == label
        mask[y, x] = False
        lab, _ = components(mask)
        ids = {lab[y - 1 + j, x - 1 + i] for j in range(3) for i in range(3)
               if block[j, i]}
        return len(ids) <= 1

    if pp == 1:
        resident_ok = True
    elif pp == 0:
        resident_ok = np.count_nonzero(field == p) == 1
    else:
        resident_ok = reconnect(p, block_p)
    candidate_ok = pq == 1 or (pq > 1 and reconnect(q, block_q))
    return pp, pq, resident_ok and candidate_ok


def boundary_faces(field):
    return int(np.count_nonzero(field[1:, :] != field[:-1, :])
               + np.count_nonzero(field[:, 1:] != field[:, :-1]))


# --------------------------------------------------------------------------
# Figure: velocity average and sub-cell update
# --------------------------------------------------------------------------
def velocity_subcell():
    fig = plt.figure(figsize=(7.2, 2.55))
    gs = fig.add_gridspec(1, 3, width_ratios=(1.0, 1.05, 1.15), wspace=0.30,
                          left=0.02, right=0.99, top=0.88, bottom=0.15)

    # (a) Pair-conditioned average at a T-junction: p left, q upper right,
    # r lower right.
    ax = fig.add_subplot(gs[0])
    n = 11
    field = np.full((n, n), P)
    field[5:, 5:] = Q
    field[:5, 5:] = R
    draw_cells(ax, field)

    def pair_of(y, x):
        lab = field[y, x]
        if lab == P and x == 4:
            return "pq" if field[y, 5] == Q else "pr"
        if lab == Q and x == 5:
            return "pq"
        if lab == R and x == 5:
            return "pr"
        if lab == Q and y == 5:
            return "qr"
        if lab == R and y == 4:
            return "qr"
        return None

    cy, cx = 7, 4
    sigma_v, r_v = 0.85, 3
    for y in range(n):
        for x in range(n):
            pair = pair_of(y, x)
            r = np.hypot(y - cy, x - cx)
            if pair is None or r > r_v + 1e-9 or (y, x) == (cy, cx):
                continue
            xc, yc = x + 0.5, y + 0.5
            if pair == "pq":
                w = np.exp(-r * r / (2 * sigma_v ** 2))
                rad = 0.07 + 0.30 * np.sqrt(w)
                filled = field[y, x] == P            # s_j = +1 on the p side
                ax.add_patch(Circle((xc, yc), rad, fc=ACCENT if filled else "white",
                                    ec=ACCENT, lw=0.8, zorder=3))
            else:
                d = 0.17
                ax.plot([xc - d, xc + d], [yc - d, yc + d], color=LINE, lw=0.8, zorder=3)
                ax.plot([xc - d, xc + d], [yc + d, yc - d], color=LINE, lw=0.8, zorder=3)
    # Centre cell i: thick outline and a full-weight marker.
    ax.add_patch(Rectangle((cx, cy), 1, 1, fill=False, ec=LINE, lw=1.7, zorder=5))
    ax.add_patch(Circle((cx + 0.5, cy + 0.5), 0.37, fc=ACCENT, ec=ACCENT, lw=0.8, zorder=4))
    ax.add_patch(Circle((cx + 0.5, cy + 0.5), r_v + 0.5, fill=False, ec=LINE,
                        lw=0.7, ls=(0, (4, 2.5)), zorder=5))
    text_label(ax, 0.75, 10.25, "$p$")
    text_label(ax, 9.5, 9.5, "$q$")
    text_label(ax, 9.5, 1.5, "$r$")
    ax.annotate("$i$", xy=(cx, cy + 0.5), xytext=(2.3, 8.7), fontsize=8.5,
                ha="center", va="center",
                arrowprops=dict(arrowstyle="-", lw=0.6, color=LINE,
                                shrinkA=1.5, shrinkB=0))
    panel_label(ax, "a", "  Pair-conditioned average")

    # (b) Gaussian weights.
    ax = fig.add_subplot(gs[1])
    r = np.linspace(0, 6, 400)
    styles = {0.85: dict(ls="-", lw=1.3), 3.00: dict(ls=(0, (5, 2.5)), lw=1.2),
              0.70: dict(ls=(0, (1.2, 1.6)), lw=1.4)}
    curve_col = {0.85: ACCENT, 3.00: INDIGO, 0.70: OLIVE}
    for s, st in styles.items():
        ax.plot(r, np.exp(-r * r / (2 * s * s)), color=curve_col[s], **st)
    for d in (1.0, np.sqrt(2.0), 2.0):
        ax.axvline(d, color="#b3b3b3", lw=0.5, ls=(0, (2, 2)), zorder=0)
    ax.plot([1.0], [0.5], "o", ms=3.4, mfc="white", mec=ACCENT, mew=0.8, zorder=4)
    # Labels sit in empty parts of the plot, each joined to its curve: the
    # 3-D label above the dashed curve, the two 2-D labels below it.
    ax.annotate(r"$\sigma_v=3.00$, 3-D", xy=(3.9, np.exp(-3.9**2 / 18)),
                xytext=(3.55, 0.74), fontsize=7.5, va="center", color=INDIGO,
                arrowprops=dict(arrowstyle="-", lw=0.5, color=INDIGO))
    ax.annotate(r"$\sigma_v=0.85$, 2-D", xy=(1.0, 0.5), xytext=(2.05, 0.30),
                fontsize=7.5, va="center", color=ACCENT,
                arrowprops=dict(arrowstyle="-", lw=0.5, color=ACCENT, shrinkB=2.5))
    ax.annotate(r"$\sigma_r=0.70$, remap", xy=(1.25, np.exp(-1.25**2 / 0.98)),
                xytext=(2.05, 0.12), fontsize=7.5, va="center", color=OLIVE,
                arrowprops=dict(arrowstyle="-", lw=0.5, color=OLIVE))
    ax.set_xlim(0, 6); ax.set_ylim(0, 1.05)
    ax.set_xticks(range(7))
    ax.set_xlabel("distance $r$ (cells)")
    ax.set_ylabel("Gaussian weight $w$")
    ax.spines[["top", "right"]].set_visible(False)
    panel_label(ax, "b", "  Gaussian weights")

    # (c) Sub-cell fraction along the normal while p recedes.
    ax = fig.add_subplot(gs[2])
    ax.set_xlim(-0.05, 4.55); ax.set_ylim(-0.55, 3.75); ax.axis("off")
    rows = [
        (2.55, 0.60, None, r"step $n$:  $\alpha^n=0.60$", "resident $p$"),
        (1.30, 0.35, 0.60, r"$\alpha^*=\alpha^n+v_n\Delta t/\Delta x=0.35$", "resident $p$"),
        (0.05, 0.00, 0.35, r"$\alpha^*\leq0$", "TCKT test"),
    ]
    h = 0.55
    for y0, alpha, previous, label, verdict in rows:
        ax.add_patch(Rectangle((0, y0), 1, h, fc=P_FILL, ec=GRID, lw=0.35))
        ax.add_patch(Rectangle((2, y0), 1, h, fc=Q_FILL, ec=GRID, lw=0.35))
        ax.add_patch(Rectangle((1, y0), alpha, h, fc=P_FILL, ec="none"))
        ax.add_patch(Rectangle((1 + alpha, y0), 1 - alpha, h, fc=Q_FILL, ec="none"))
        ax.add_patch(Rectangle((1, y0), 1, h, fill=False, ec=LINE, lw=1.3))
        ax.plot([1 + alpha, 1 + alpha], [y0, y0 + h], color=ACCENT, lw=1.6)
        if previous is not None:
            # Dashed line at the previous boundary position, and an arrow from
            # it to the new position.  A small head and no end clearance keep
            # the tail visible even for the quarter-cell move of the middle row.
            ax.plot([1 + previous, 1 + previous], [y0, y0 + h], color=ACCENT,
                    lw=0.8, ls=(0, (2, 1.5)))
            ax.add_patch(FancyArrowPatch((1 + previous, y0 + h / 2),
                                         (1 + alpha, y0 + h / 2),
                                         arrowstyle="-|>,head_length=0.35,head_width=0.18",
                                         mutation_scale=10, lw=0.9, color=LINE,
                                         shrinkA=0, shrinkB=0))
        ax.text(0.0, y0 + h + 0.12, label, ha="left", va="bottom", fontsize=7.5)
        ax.text(3.15, y0 + h / 2, verdict, ha="left", va="center", fontsize=7.5)
    text_label(ax, 0.5, 2.55 + h / 2, "$p$", size=8.5)
    text_label(ax, 2.5, 2.55 + h / 2, "$q$", size=8.5)
    ax.annotate("", xy=(2.0, -0.25), xytext=(1.0, -0.25),
                arrowprops=dict(arrowstyle="<->", color=LINE, lw=0.6,
                                shrinkA=0, shrinkB=0))
    ax.text(1.5, -0.33, r"$\Delta x$", ha="center", va="top", fontsize=7.5)
    panel_label(ax, "c", "  Sub-cell update")

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig_velocity_subcell.pdf", facecolor="white")
    plt.close(fig)


# --------------------------------------------------------------------------
# Figure: TCKT and BGR
# --------------------------------------------------------------------------
def tckt_cases():
    """Four 7x7 windows; the centre cell (row 3, column 3) is a proposed p -> q crossing."""
    cases = []
    smooth = np.full((7, 7), Q)
    smooth[:, :4] = P
    cases.append(("Smooth advance", smooth))

    bridge = np.full((7, 7), Q)
    bridge[1:6, 1:3] = P
    bridge[1:6, 4:6] = P
    bridge[3, 3] = P
    cases.append(("Bridge (split of $p$)", bridge))

    ring = np.full((7, 7), Q)
    ring[3, 1:6] = P
    ring[6, 1:6] = P
    ring[3:7, 1] = P
    ring[3:7, 5] = P
    ring[4:6, 2:5] = R
    cases.append(("Arms reconnect", ring))

    merge = np.full((7, 7), P)
    merge[3, 1:3] = Q
    merge[3, 4:6] = Q
    cases.append(("Merger of $q$", merge))
    return cases


def tckt_bgr():
    fig = plt.figure(figsize=(7.2, 3.75))
    top = fig.add_gridspec(1, 4, left=0.02, right=0.99, top=0.92, bottom=0.50,
                           wspace=0.22)
    bottom = fig.add_gridspec(1, 3, left=0.02, right=0.99, top=0.40, bottom=0.02,
                              wspace=0.08, width_ratios=(1, 0.30, 1))
    for k, (title, field) in enumerate(tckt_cases()):
        ax = fig.add_subplot(top[k])
        draw_cells(ax, field)
        if np.any(field == R):
            text_label(ax, 3.5, 5.5, "$r$", size=8.5)
        pp, pq, ok = tckt(field, (3, 3), P, Q)
        ax.add_patch(Rectangle((2, 2), 3, 3, fill=False, ec=LINE, lw=0.8,
                               ls=(0, (3, 2)), zorder=4))
        ax.add_patch(Rectangle((3, 3), 1, 1, fill=False, ec=ACCENT, lw=1.8, zorder=5))
        text_label(ax, 3.5, 3.5, "$i$", size=8.5)
        ax.text(3.5, -0.45, rf"$P_p={pp}$, $P_q={pq}$", ha="center", va="top",
                fontsize=7.5)
        ax.text(3.5, -1.45, "accepted" if ok else "deferred", ha="center",
                va="top", fontsize=7.5, style="italic", color=OLIVE if ok else ACCENT)
        ax.set_ylim(-2.2, 7.1)
        panel_label(ax, "abcd"[k], "  " + title)
        print(f"TCKT {title}: P_p={pp} P_q={pq} -> {'accepted' if ok else 'deferred'}")

    before = np.full((6, 12), Q)
    before[:, :6] = P
    before[2, 6] = P        # protrusion of p into q
    before[4, 5] = Q        # indentation of q into p
    after = before.copy()
    after[2, 6] = Q
    after[4, 5] = P
    nf_before = boundary_faces(before)
    for k, (field, title, letter) in enumerate(((before, "Before remap", "e"),
                                                 (after, "After remap", "f"))):
        ax = fig.add_subplot(bottom[0 if k == 0 else 2])
        draw_cells(ax, field)
        npc, nqc, nf = (int(np.count_nonzero(field == P)),
                        int(np.count_nonzero(field == Q)), boundary_faces(field))
        if k == 0:
            for (y, x) in ((2, 6), (4, 5)):
                ax.add_patch(Rectangle((x, y), 1, 1, fill=False, ec=ACCENT, lw=1.8))
            ax.annotate("protrusion", xy=(6.5, 3.0), xytext=(8.6, 6.65),
                        fontsize=7.5, va="center", ha="center",
                        arrowprops=dict(arrowstyle="-", lw=0.5, color=LINE))
            ax.annotate("indentation", xy=(5.5, 5.0), xytext=(3.4, 6.65),
                        fontsize=7.5, va="center", ha="center",
                        arrowprops=dict(arrowstyle="-", lw=0.5, color=LINE))
        text_label(ax, 1.5, 1.5, "$p$", size=8.5)
        text_label(ax, 10.5, 1.5, "$q$", size=8.5)
        ax.text(6.0, -0.4, rf"$N_p={npc}$,  $N_q={nqc}$,  $N_{{\rm face}}={nf}$",
                ha="center", va="top", fontsize=7.5)
        if k == 1:
            ax.text(6.0, -1.3, rf"$\Delta N_{{\rm face}}={nf - nf_before}$: exchange committed",
                    ha="center", va="top", fontsize=7.5, style="italic", color=OLIVE)
        ax.set_ylim(-2.0, 7.1)
        panel_label(ax, letter, "  " + title)
        print(f"BGR {title}: N_p={npc} N_q={nqc} N_face={nf}")
    ax = fig.add_subplot(bottom[1]); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.add_patch(FancyArrowPatch((0.12, 0.60), (0.88, 0.60), arrowstyle="-|>",
                                 mutation_scale=9, lw=0.9, color=LINE))
    ax.text(0.5, 0.68, "balanced\nexchange", ha="center", va="bottom", fontsize=7.5)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig_tckt_bgr.pdf", facecolor="white")
    plt.close(fig)


def si_topology_transactions():
    """Supplementary Figure S2: the four executable TCKT unit-test cases."""
    cases = []
    smooth = np.full((7, 7), Q)
    smooth[:, :4] = P
    cases.append(("Smooth advance", smooth))
    bridge = np.full((7, 7), Q)
    bridge[1:6, 1:3] = P
    bridge[1:6, 4:6] = P
    bridge[3, 3] = P
    cases.append(("Bridge deletion", bridge))
    gap = np.full((7, 7), P)
    gap[3, 2] = Q
    gap[3, 4] = Q
    cases.append(("Gap insertion", gap))
    extinct = np.full((7, 7), Q)
    extinct[3, 3] = P
    cases.append(("Terminal extinction", extinct))

    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.25))
    fig.subplots_adjust(left=0.02, right=0.99, top=0.86, bottom=0.03, wspace=0.22)
    for k, (title, field) in enumerate(cases):
        ax = axes[k]
        draw_cells(ax, field)
        pp, pq, ok = tckt(field, (3, 3), P, Q)
        ax.add_patch(Rectangle((2, 2), 3, 3, fill=False, ec=LINE, lw=0.8,
                               ls=(0, (3, 2)), zorder=4))
        ax.add_patch(Rectangle((3, 3), 1, 1, fill=False, ec=ACCENT, lw=1.8, zorder=5))
        text_label(ax, 3.5, 3.5, "$i$", size=8.5)
        ax.text(3.5, -0.45, rf"$P_p={pp}$, $P_q={pq}$", ha="center", va="top",
                fontsize=7.5)
        ax.text(3.5, -1.45, "accepted" if ok else "deferred", ha="center",
                va="top", fontsize=7.5, style="italic", color=OLIVE if ok else ACCENT)
        ax.set_ylim(-2.2, 7.1)
        panel_label(ax, "abcd"[k], "  " + title)
        print(f"SI TCKT {title}: P_p={pp} P_q={pq} -> {'accepted' if ok else 'deferred'}")
    fig.savefig(OUT / "figS13_topology_transactions.pdf", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    velocity_subcell()
    tckt_bgr()
    si_topology_transactions()
