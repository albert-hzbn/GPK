"""Write the audited Al--Cu baseline/refined mesh comparison."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "output" / "alcu_validation" / "precipitation"
OUT = DATA / "mesh_convergence.csv"


CASES = (
    ("3D fine, 0.5 ks", "three_dimensional_mesh_fine"),
    ("3D ultrafine, 0.5 ks", "three_dimensional_mesh_ultrafine"),
    ("3D production, 0.5 ks", "three_dimensional_ultrafine"),
)


def rows(path: Path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def summarize(label: str, case: str):
    metric = rows(DATA / case / "metrics.csv")
    particles = rows(DATA / case / "particle_statistics.csv")
    params = dict((r["parameter"], r["value"])
                  for r in rows(DATA / case / "parameters.csv"))
    final = metric[-1]
    aspects = np.asarray([float(r["aspect_ratio"]) for r in particles])
    majors = np.asarray([float(r["major_dimension_nm"]) for r in particles])
    minors = np.asarray([float(r["minor_dimension_nm"]) for r in particles])
    deferred = sum(float(r["topology_deferred_cells"]) for r in metric)
    remapped = sum(float(r.get("gaussian_remap_exchanges", 0.0)) for r in metric)
    interface_exposure = sum(float(r["interface_cells"]) for r in metric)
    return {
        "label": label,
        "case": case,
        "shape": params["shape"],
        "dx_nm": 1e9*float(params["dx"]),
        "end_time_s": float(final["time_s"]),
        "precipitate_fraction": float(final["precipitate_fraction"]),
        "median_aspect_ratio": float(np.median(aspects)),
        "median_major_nm": float(np.median(majors)),
        "median_minor_nm": float(np.median(minors)),
        "mass_error": float(final["relative_mass_error"]),
        "equilibrium_residual": float(final["mechanical_equilibrium_residual"]),
        "max_interface_cfl": max(float(r["max_interface_displacement_cells"])
                                 for r in metric),
        "topology_deferred_cells": deferred,
        "topology_deferred_per_interface_step": deferred/max(interface_exposure, 1.0),
        "gaussian_remap_exchanges": remapped,
        "gaussian_remap_per_interface_step": remapped/max(interface_exposure, 1.0),
    }


def main():
    result = [summarize(*item) for item in CASES]
    with OUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=result[0].keys())
        writer.writeheader()
        writer.writerows(result)
    for row in result:
        print(row)


if __name__ == "__main__":
    main()
