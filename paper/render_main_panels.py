"""Re-render the 2-D multiphase gallery panels (Figure 9(d)) with the v3 palette.

Run with ParaView's pvpython.  Phase 1 is red and phase 2 blue on a light grey
matrix, the same phase colours as the three-dimensional multiphase row of
Figure 10.  Grain-growth and recrystallisation panels keep their categorical
grain colours, because hundreds of neighbouring grains need distinct colours.
"""
from __future__ import annotations

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
SOURCE = output_dir("manuscript_large_domain") / "multiphase_2d"


def load(index: int) -> np.ndarray:
    files = sorted(SOURCE.glob("snap_*.csv"))
    meta = {}
    with files[index].open(encoding="utf-8") as stream:
        for line in stream:
            if not line.startswith("#"):
                break
            for key in ("Nz", "Ny", "Nx"):
                match = re.search(rf"{key}=(\d+)", line)
                if match:
                    meta[key] = int(match.group(1))
    labels = np.loadtxt(files[index], comments="#", usecols=(3,), dtype=np.int64)
    labels = labels.reshape(meta.get("Nz", 1), meta["Ny"], meta["Nx"])
    return labels[labels.shape[0] // 2]


def main() -> None:
    pv.global_theme.allow_empty_mesh = True
    for state, index in (("initial", 0), ("final", -1)):
        render_label_field_2d(load(index), ASSETS / f"qualitative2d_multiphase_{state}.png",
                              semantic={0: "#E3E3E3", 1: "#C62020", 2: "#1F4FB4"},
                              curvature=False, show_boundaries=False, window_size=(1400, 1400))
    print("2-D multiphase panels written", flush=True)
    render_alcu_populations()


def render_alcu_populations() -> None:
    """Figure 13(a,b): theta' variants in blue, red and green (v3 palette)."""
    import importlib.util
    path = REPO / "paper" / "render_all_microstructures_pyvista.py"
    spec = importlib.util.spec_from_file_location("gpk_v1_renderer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.BLUE, module.ORANGE, module.GREEN = "#1F4FB4", "#C62020", "#7B2FA8"
    root = output_dir("alcu_validation") / "realistic_population_3d"
    for hour in (120, 168):
        module.render_population(root / f"particles_{hour}h.npz",
                                 ASSETS / f"alcu_population_{hour}h.png")
        print(f"Al-Cu population {hour} h written", flush=True)


if __name__ == "__main__":
    main()
