"""Prolong a categorical GPK CSV snapshot onto a finer periodic cell grid.

The operation is nearest-cell injection: it does not invent labels or smooth
interfaces.  The fine-grid solver subsequently performs the physical
Gaussian/topology evolution, so all reported endpoint diagnostics are measured
after the relaxation stage rather than on the prolonged field.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


def metadata(path: Path) -> dict[str, float]:
    text = ""
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.startswith("#"):
                break
            text += line
    values: dict[str, float] = {}
    for key in ("Nz", "Ny", "Nx", "time", "dx"):
        match = re.search(rf"{key}=([\d.e+\-]+)", text)
        if match:
            values[key] = float(match.group(1))
    return values


def prolong(source: Path, destination: Path, shape: tuple[int, int, int]) -> None:
    meta = metadata(source)
    old_shape = tuple(int(meta[key]) for key in ("Nz", "Ny", "Nx"))
    values = np.loadtxt(source, comments="#", usecols=(3, 7))
    labels = values[:, 0].astype(np.int32).reshape(old_shape)
    orientations = values[:, 1].astype(np.float32).reshape(old_shape)
    indices = [np.floor(np.arange(new) * old / new).astype(np.int64)
               for old, new in zip(old_shape, shape)]
    labels_fine = labels[np.ix_(*indices)]
    orientations_fine = orientations[np.ix_(*indices)]

    destination.mkdir(parents=True, exist_ok=True)
    label_path = destination / "labels_int32.bin"
    orientation_path = destination / "orientations_float32.bin"
    labels_fine.astype("<i4", copy=False).tofile(label_path)
    orientations_fine.astype("<f4", copy=False).tofile(orientation_path)
    report = {
        "source": str(source),
        "source_shape": old_shape,
        "target_shape": shape,
        "source_time_s": meta.get("time"),
        "source_dx_m": meta.get("dx"),
        "labels": int(np.unique(labels_fine).size),
        "operation": "nearest-cell categorical prolongation; no filtering",
        "label_binary": str(label_path),
        "orientation_binary": str(orientation_path),
    }
    (destination / "prolongation_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--shape", type=int, nargs=3, default=(128, 256, 256),
                        metavar=("NZ", "NY", "NX"))
    args = parser.parse_args()
    prolong(args.source, args.destination, tuple(args.shape))


if __name__ == "__main__":
    main()
