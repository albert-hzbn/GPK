"""Re-render the Supporting Information bicrystal panels with the v3 palette.

Run with ParaView's pvpython.  Reads the raw label grids written by the
extended base-model validation and writes bicrystal_*.png and curved_*.png
into figures/pyvista_panels.  The phase colours are the primary/secondary
fills used throughout the v3 figures (see generate_publication_figures.py).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / ".pyvista_pv"))
import pyvista as pv  # noqa: E402

from pyvista_rendering import render_label_field_2d, render_scalar_field_2d  # noqa: E402
from gpk_data import output_dir

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
ASSETS = HERE / "figures" / "pyvista_panels"
BASE = output_dir("validate_extended_base_models")

FILL_BLUE, FILL_YELLOW, FILL_GREEN = "#7EA6E0", "#FFD966", "#B08AD6"
FILL_RED, FILL_PURPLE, FILL_GREY = "#EF7B7B", "#9E9E9E", "#E3E3E3"
FILL_LIGHTBLUE, FILL_WHITE = "#CFE0F7", "#F5F5F5"


def load_bicrystal(path: Path) -> np.ndarray:
    header = path.read_text(encoding="utf-8", errors="replace").splitlines()[0]
    shape = {key: int(value) for key, value in re.findall(r"(Ny|Nx)=(\d+)", header)}
    values = np.loadtxt(path, comments="#", skiprows=2, dtype=np.int64)
    return values.reshape(shape["Ny"], shape["Nx"])


def main() -> None:
    pv.global_theme.allow_empty_mesh = True
    planar = {
        "g1": {1: FILL_BLUE, 2: FILL_YELLOW},
        "g2": {0: FILL_GREY, 1: FILL_RED},
        "g5": {0: FILL_LIGHTBLUE, 1: FILL_RED},
        "g6": {0: FILL_WHITE, 1: FILL_GREEN, 2: FILL_BLUE},
        "g7": {0: FILL_RED, 1: FILL_GREY},
        "g8": {0: FILL_GREY, 1: FILL_PURPLE},
        "g9": {1: FILL_BLUE, 2: FILL_YELLOW},
        "g10": {0: FILL_GREY, 1: FILL_GREEN},
    }
    for key, semantic in planar.items():
        for source, state in (("before", "initial"), ("after", "final")):
            render_label_field_2d(load_bicrystal(BASE / f"{key}_grid_{source}.csv"),
                                  ASSETS / f"bicrystal_{key}_{state}.png",
                                  semantic=semantic, sigma_cells=1.0,
                                  window_size=(1050, 600))
    curved = {"g13": FILL_BLUE, "g14": FILL_GREEN, "g15": FILL_PURPLE}
    for key, colour in curved.items():
        for source, state in (("before", "initial"), ("after", "final")):
            render_label_field_2d(load_bicrystal(BASE / f"{key}_grid_{source}.csv"),
                                  ASSETS / f"curved_{key}_{state}.png",
                                  semantic={0: FILL_GREY, 1: colour}, sigma_cells=1.1,
                                  crop=(65, 295, 65, 295), window_size=(900, 900))
    print("bicrystal panels written", flush=True)
    render_alcu_sections()


def _load_alcu(case: str, index: int) -> dict:
    root = output_dir("alcu_validation") / "precipitation"
    return dict(np.load(root / case / f"snap_{index:03d}.npz"))


def render_alcu_sections() -> None:
    """Al-Cu field sections and mesh-audit sections for Figures S9 and S10."""
    snap = _load_alcu("three_dimensional_ultrafine", 3)
    labels = snap["labels"]
    z = int(np.argmax(np.count_nonzero(labels > 0, axis=(0, 1))))
    render_scalar_field_2d(snap["c_matrix"][:, :, z], ASSETS / "alcu_matrix_cu.png",
                           vmin=0.004, vmax=0.023, ramp="blues", window_size=(1100, 1000))
    render_scalar_field_2d(snap["elastic_energy"][:, :, z] / 1.0e6,
                           ASSETS / "alcu_elastic_energy.png", ramp="yellowred",
                           window_size=(1100, 1000))
    for case, tag in (("three_dimensional_mesh_fine", "96"),
                      ("three_dimensional_mesh_ultrafine", "128")):
        state = _load_alcu(case, 3)
        mask = state["labels"] == 3
        coords = np.argwhere(mask)
        n_y = mask.shape[1]
        angles = 2.0 * np.pi * coords[:, 1] / n_y
        centre = int(np.rint((np.arctan2(np.sum(np.sin(angles)), np.sum(np.cos(angles)))
                              % (2.0 * np.pi)) * n_y / (2.0 * np.pi))) % n_y
        render_label_field_2d(mask[:, centre, :].astype(np.int64),
                              ASSETS / f"alcu_mesh_{tag}.png",
                              semantic={0: FILL_GREY, 1: "#7B2FA8"},
                              sigma_cells=1.1, window_size=(1000, 850))
    print("Al-Cu sections written", flush=True)


if __name__ == "__main__":
    main()
