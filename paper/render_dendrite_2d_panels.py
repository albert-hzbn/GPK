"""Render the 2-D elastic-coupled dendrite panels (Figure 7e) with PyVista.

Run with ParaView's pvpython.  Reads output/fig7_large/dendrite_2d and writes
qualitative2d_elastic_{initial,final}.png into figures/pyvista_panels.  The
final state is the snapshot selected by GPK_DENDRITE2D_INDEX (default: last).
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / ".pyvista_pv"))
import pyvista as pv  # noqa: E402

from pyvista_rendering import render_label_field_2d  # noqa: E402
from gpk_data import output_dir

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
ASSETS = HERE / "figures" / "pyvista_panels"
SOURCE = output_dir("fig7_large") / "dendrite_2d"
SOLID = "#1F4FB4"   # dendrite (blue, v3 palette)


def load(index: int) -> tuple[np.ndarray, float]:
    files = sorted(SOURCE.glob("snap_*.csv"))
    path = files[index]
    meta = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.startswith("#"):
                break
            for key in ("time", "Nz", "Ny", "Nx"):
                match = re.search(rf"{key}=([\d.e+\-]+)", line)
                if match:
                    meta[key] = float(match.group(1))
    labels = np.loadtxt(path, comments="#", usecols=(3,), dtype=np.int64)
    labels = labels.reshape(int(meta.get("Nz", 1)), int(meta["Ny"]), int(meta["Nx"]))
    return labels[labels.shape[0] // 2], meta["time"]


def main() -> None:
    pv.global_theme.allow_empty_mesh = True
    final_index = int(os.environ.get("GPK_DENDRITE2D_INDEX", "-1"))
    for state, index in (("initial", 0), ("final", final_index)):
        labels, time = load(index)
        # Central 800 x 800 cells (400 um); the crystal stays well inside it.
        ny, nx = labels.shape
        h = min(400, ny // 2, nx // 2)
        crop = (nx // 2 - h, nx // 2 + h, ny // 2 - h, ny // 2 + h)
        render_label_field_2d(labels, crop=crop, output=ASSETS / f"qualitative2d_elastic_{state}.png",
                              semantic={0: "#E3E3E3", 1: SOLID}, curvature=False,
                              show_boundaries=False, window_size=(1400, 1400))
        print(f"{state}: t = {time:.6g} s, solid fraction = {np.mean(labels != 0):.4f}")


if __name__ == "__main__":
    main()
