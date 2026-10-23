"""Audit raw three-dimensional grain-label quality without render-time smoothing.

The metrics intentionally operate on the categorical solver output.  They detect
voxel-scale boundary roughness, fragmented exterior grain traces, and grains that
are adjacent to only one enclosing label.  This keeps geometric validation
separate from visualization.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import ndimage


def load_labels(path: Path) -> np.ndarray:
    header = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith("#"):
                break
            header.append(line)
    match = re.search(r"Nz=(\d+)\s+Ny=(\d+)\s+Nx=(\d+)", "".join(header))
    if not match:
        raise ValueError(f"Grid dimensions are absent from {path}")
    nz, ny, nx = map(int, match.groups())
    labels = np.loadtxt(path, comments="#", usecols=(3,), dtype=np.int32)
    return labels.reshape(nz, ny, nx)


def exterior_patch_metrics(labels: np.ndarray) -> tuple[int, int]:
    faces = (
        labels[0], labels[-1], labels[:, 0, :], labels[:, -1, :],
        labels[:, :, 0], labels[:, :, -1],
    )
    patches = 0
    small_patches = 0
    connectivity = np.ones((3, 3), dtype=np.uint8)
    for face in faces:
        for grain in np.unique(face):
            components, count = ndimage.label(face == grain, structure=connectivity)
            sizes = np.bincount(components.ravel())[1:]
            patches += int(count)
            small_patches += int(np.count_nonzero(sizes < 25))
    return patches, small_patches


def audit(labels: np.ndarray) -> dict[str, int | float]:
    same_neighbours = np.zeros(labels.shape, dtype=np.uint8)
    interface_faces = 0
    adjacency: dict[int, set[int]] = defaultdict(set)
    for axis in range(3):
        shifted = np.roll(labels, -1, axis=axis)
        changed = labels != shifted
        interface_faces += int(np.count_nonzero(changed))
        same_neighbours += labels == shifted
        same_neighbours += labels == np.roll(labels, 1, axis=axis)
        # Collapse the millions of unlike faces to the unique unordered label
        # pairs before updating the Python adjacency sets.  This leaves the
        # audit definition unchanged while making production-volume checks
        # practical and deterministic.
        pairs = np.stack((labels[changed], shifted[changed]), axis=1)
        pairs = np.unique(np.sort(pairs, axis=1), axis=0)
        for p, q in pairs:
            adjacency[int(p)].add(int(q))
            adjacency[int(q)].add(int(p))

    interface_cells = same_neighbours < 6
    patches, small_patches = exterior_patch_metrics(labels)
    volumes = np.bincount(labels.ravel())
    present = np.flatnonzero(volumes)
    neighbour_counts = [len(adjacency[int(grain)]) for grain in present]
    return {
        "grains": int(present.size),
        "minimum_grain_voxels": int(volumes[present].min()),
        "periodic_interfacial_faces": interface_faces,
        "interface_cells": int(np.count_nonzero(interface_cells)),
        "low_support_interface_cells": int(
            np.count_nonzero(interface_cells & (same_neighbours <= 2))
        ),
        "exterior_face_patches": patches,
        "exterior_face_patches_under_25_pixels": small_patches,
        "grains_with_one_or_fewer_neighbours": int(
            np.count_nonzero(np.asarray(neighbour_counts) <= 1)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshots", nargs="+", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    result = {str(path): audit(load_labels(path)) for path in args.snapshots}
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.json:
        args.json.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
