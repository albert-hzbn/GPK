"""Audit that elastic-coupling gallery endpoints do not reach the domain boundary."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CASES = (
    ("2d", ROOT / "build/output/manuscript_high_resolution/elastic_2d/snap_0001.csv", 8.0),
    ("3d", ROOT / "build/output/manuscript_high_resolution/elastic_3d/snap_0001.csv", 5.3333333),
)
OUTPUT = ROOT / "paper/elastic_endpoint_boundary_audit.json"


def load(path: Path) -> tuple[np.ndarray, float]:
    header: list[str] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.startswith("#"):
                break
            header.append(line)
    text = "".join(header)
    shape_match = re.search(r"Nz=(\d+)\s+Ny=(\d+)\s+Nx=(\d+)", text)
    time_match = re.search(r"time=([0-9.eE+-]+)", text)
    if not shape_match or not time_match:
        raise ValueError(f"Incomplete snapshot header: {path}")
    nz, ny, nx = map(int, shape_match.groups())
    labels = np.loadtxt(path, comments="#", usecols=(3,), dtype=np.int8)
    return labels.reshape(nz, ny, nx), float(time_match.group(1))


def audit(path: Path, sigma_cells: float) -> dict[str, object]:
    labels, time_s = load(path)
    z, y, x = np.where(labels == 1)
    if x.size == 0:
        raise RuntimeError(f"No transformed cells in {path}")
    nz, ny, nx = labels.shape
    margins = [int(x.min()), int(nx - 1 - x.max()),
               int(y.min()), int(ny - 1 - y.max())]
    names = ["x_negative", "x_positive", "y_negative", "y_positive"]
    if nz > 1:
        margins.extend((int(z.min()), int(nz - 1 - z.max())))
        names.extend(("z_negative", "z_positive"))
    minimum = min(margins)
    return {
        "snapshot": str(path.relative_to(ROOT)).replace("\\", "/"),
        "grid": [int(nz), int(ny), int(nx)],
        "time_s": time_s,
        "transformed_fraction": float(np.mean(labels == 1)),
        "sigma_G_cells": sigma_cells,
        "face_margins_cells": dict(zip(names, margins, strict=True)),
        "minimum_margin_cells": minimum,
        "touches_boundary": minimum == 0,
        "margin_exceeds_sigma_G": minimum > sigma_cells,
    }


def main() -> None:
    report = {name: audit(path, sigma) for name, path, sigma in CASES}
    for name, result in report.items():
        if result["touches_boundary"] or not result["margin_exceeds_sigma_G"]:
            raise RuntimeError(f"{name} elastic endpoint lacks boundary clearance: {result}")
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
