"""Audit periodic face connectivity before and after a grain-network run."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def read_labels(path: Path) -> np.ndarray:
    meta: dict[str, int] = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.startswith("#"):
                break
            for token in line[1:].replace(",", " ").split():
                if "=" in token:
                    key, value = token.split("=", 1)
                    if key in {"Nx", "Ny", "Nz"}:
                        meta[key] = int(float(value))
    labels = np.loadtxt(path, comments="#", usecols=(3,), dtype=np.int32)
    return labels.reshape(meta.get("Nz", 1), meta["Ny"], meta["Nx"])


def periodic_component_sizes(labels: np.ndarray, value: int) -> list[int]:
    occupied = {tuple(int(v) for v in point)
                for point in np.argwhere(labels == value)}
    sizes: list[int] = []
    while occupied:
        seed = occupied.pop()
        stack = [seed]
        size = 1
        while stack:
            point = stack.pop()
            for axis in range(labels.ndim):
                for shift in (-1, 1):
                    neighbour = list(point)
                    neighbour[axis] = (neighbour[axis]+shift) % labels.shape[axis]
                    neighbour = tuple(neighbour)
                    if neighbour in occupied:
                        occupied.remove(neighbour)
                        stack.append(neighbour)
                        size += 1
        sizes.append(size)
    return sizes


def summary(labels: np.ndarray) -> dict[str, int]:
    values = np.unique(labels)
    sizes = [size for value in values
             for size in periodic_component_sizes(labels, int(value))]
    return {
        "labels": int(len(values)),
        "components": int(len(sizes)),
        "component_excess": int(len(sizes)-len(values)),
        "components_le_2_cells": int(sum(size <= 2 for size in sizes)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", nargs="?", type=Path,
                        default=ROOT / "output" / "topology_network_smoke")
    args = parser.parse_args()
    files = sorted(args.directory.glob("snap_*.csv"))
    if len(files) < 2:
        raise FileNotFoundError(f"Need initial and final snapshots in {args.directory}")
    initial = summary(read_labels(files[0]))
    final = summary(read_labels(files[-1]))
    if final["component_excess"] > initial["component_excess"]:
        raise RuntimeError(f"Topology split detected: {initial} -> {final}")
    print("initial", initial)
    print("final  ", final)
    print("Periodic grain-network topology audit: PASS")


if __name__ == "__main__":
    main()
