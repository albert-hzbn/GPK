"""Quantitative and topology audit of the manuscript's 2-D/3-D multiphase CSVs."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy import ndimage


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "build" / "output" / "manuscript_high_resolution"
OUT = ROOT / "build" / "output" / "validate_dimensional_multiphase" / "multiphase_visual_metrics.csv"


def load(path: Path):
    meta = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.startswith("#"):
                break
            for token in line[1:].split():
                if "=" in token:
                    key, value = token.split("=", 1)
                    try:
                        meta[key] = float(value)
                    except ValueError:
                        pass
    raw = np.loadtxt(path, comments="#")
    shape = (int(meta["Nz"]), int(meta["Ny"]), int(meta["Nx"]))
    return meta, raw[:, 3].astype(np.int16).reshape(shape)


def interface_faces(mask: np.ndarray) -> int:
    axes = (1, 2) if mask.shape[0] == 1 else (0, 1, 2)
    return sum(int(np.count_nonzero(mask != np.roll(mask, -1, axis=axis))) for axis in axes)


def audit(dirname: str, dimension: int):
    files = sorted((BASE / dirname).glob("snap_*.csv"))
    if len(files) < 2:
        raise RuntimeError(f"Need initial and final snapshots in {BASE / dirname}")
    snapshots = [load(path) for path in files]
    fractions = np.asarray([[np.mean(labels == q) for q in range(3)] for _, labels in snapshots])
    selected_meta, selected = snapshots[-1]
    structure = ndimage.generate_binary_structure(dimension, 1)
    sizes, unsupported, serrations = [], [], []
    for phase in (1, 2):
        mask = selected == phase
        plane_or_volume = mask[0] if dimension == 2 else mask
        components, count = ndimage.label(plane_or_volume, structure=structure)
        component_sizes = np.bincount(components.ravel())[1:]
        sizes.extend(component_sizes.tolist())

        axes = (1, 2) if dimension == 2 else (0, 1, 2)
        support = sum(np.roll(mask, shift, axis=axis) for axis in axes for shift in (-1, 1))
        unsupported.append(np.count_nonzero(mask & (support == 0)) / max(np.count_nonzero(mask), 1))

        faces = interface_faces(mask)
        if dimension == 2:
            ideal = np.sum(2.0*np.sqrt(np.pi*component_sizes))
        else:
            ideal = np.sum(4.0*np.pi*(3.0*component_sizes/(4.0*np.pi))**(2.0/3.0))
        serrations.append(faces/max(ideal, 1e-12))

    return {
        "dimension": dimension,
        "time": selected_meta["time"],
        "fraction_sum_error": float(np.max(np.abs(fractions.sum(axis=1)-1.0))),
        "phase1_monotone": bool(np.all(np.diff(fractions[:, 1]) >= -1e-12)),
        "phase2_monotone": bool(np.all(np.diff(fractions[:, 2]) >= -1e-12)),
        "invalid_label_fraction": float(np.mean(~np.isin(selected, (0, 1, 2)))),
        "minimum_product_component_cells": int(min(sizes)) if sizes else 0,
        "isolated_product_fraction": float(max(unsupported)),
        "isolated_product_tolerance": 2.0e-5 if dimension == 2 else 1.0e-6,
        "grid_serration_index": float(max(serrations)),
        "f0": float(fractions[-1, 0]), "f1": float(fractions[-1, 1]), "f2": float(fractions[-1, 2]),
    }


def main():
    rows = [audit("multiphase_2d", 2), audit("multiphase_3d", 3)]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    all_passed = True
    for row in rows:
        passed = (row["fraction_sum_error"] < 1e-12 and row["phase1_monotone"] and
                  row["phase2_monotone"] and row["invalid_label_fraction"] == 0.0 and
                  row["isolated_product_fraction"] <= row["isolated_product_tolerance"])
        all_passed = all_passed and passed
        print(f"{row['dimension']}D: {'PASS' if passed else 'FAIL'}; "
              f"f=({row['f0']:.4f},{row['f1']:.4f},{row['f2']:.4f}), "
              f"min component={row['minimum_product_component_cells']} cells, "
              f"isolated={row['isolated_product_fraction']:.3g} "
              f"(limit {row['isolated_product_tolerance']:.1g}), "
              f"raw-grid serration={row['grid_serration_index']:.3f}")
    if not all_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
