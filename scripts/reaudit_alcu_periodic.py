"""Regenerate Al--Cu particle/topology audits with periodic connectivity."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from run_alcu_gpk_microstructure import (
    Parameters, neighbor_competitor, particle_statistics,
    periodic_components, topology_metrics,
)


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "output" / "alcu_validation" / "precipitation"


def reaudit(case_name: str) -> None:
    case = DATA / case_name
    paths = sorted(case.glob("snap_*.npz"))
    if len(paths) < 2:
        raise RuntimeError(f"missing snapshots for {case_name}")
    initial = dict(np.load(paths[0]))
    final = dict(np.load(paths[-1]))
    p = Parameters(dx=float(final["dx_m"]))
    labels, alpha = final["labels"], final["alpha"]
    _, competitor = neighbor_competitor(labels)

    stats = particle_statistics(labels, alpha, competitor, p)
    with (case / "particle_statistics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["variant", "fractional_voxel_count", "aspect_ratio",
                         "major_dimension_nm", "minor_dimension_nm"])
        writer.writerows(stats)

    initial_geometry = np.asarray([(initial["labels"] == v).astype(float)
                                   for v in (1, 2, 3)])
    final_geometry = np.asarray([(labels == v).astype(float)
                                 for v in (1, 2, 3)])
    initial_topology = topology_metrics(initial_geometry)
    final_topology = topology_metrics(final_geometry)
    with (case / "topology_audit.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["state", "variant", "components", "enclosed_cavities",
                         "threshold_cells", "exposed_grid_faces",
                         "noncontiguous_habit_lines"])
        for state, rows in (("initial", initial_topology), ("final", final_topology)):
            for row in rows:
                writer.writerow([state, *row])

    initial_present = {v for v in (1, 2, 3) if np.any(initial["labels"] == v)}
    final_present = {v for v in (1, 2, 3) if np.any(labels == v)}
    if not initial_present.issubset(final_present):
        raise RuntimeError(f"lost variants: {initial_present-final_present}")
    if any(row[2] for row in final_topology):
        raise RuntimeError(f"periodic cavity audit failed: {final_topology}")
    if any(row[1] > initial_topology[i][1]
           for i, row in enumerate(final_topology)):
        raise RuntimeError(f"periodic component audit failed: {final_topology}")
    print(case_name, "periodic topology", final_topology, "particles", stats)


def regression() -> None:
    mask = np.zeros((8, 8, 8), dtype=bool)
    mask[0, 3:5, 3:5] = True
    mask[-1, 3:5, 3:5] = True
    _, count = periodic_components(mask)
    if count != 1:
        raise RuntimeError(f"periodic boundary regression failed: {count}")
    print("periodic boundary regression: PASS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cases", nargs="+", help="Al--Cu case directories")
    args = parser.parse_args()
    regression()
    for name in args.cases:
        reaudit(name)
