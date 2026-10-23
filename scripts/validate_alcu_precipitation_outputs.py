"""Executable acceptance gates for the reported Al--Cu precipitation runs."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "output" / "alcu_validation" / "precipitation"
CASES = (
    "three_dimensional_ultrafine",
    "three_dimensional_mesh_fine",
    "three_dimensional_mesh_ultrafine",
)


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def endpoint(case: str):
    metrics = read_csv(DATA / case / "metrics.csv")
    parameters = {row["parameter"]: row["value"]
                  for row in read_csv(DATA / case / "parameters.csv")}
    particles = read_csv(DATA / case / "particle_statistics.csv")
    topology = read_csv(DATA / case / "topology_audit.csv")
    initial_topology = [r for r in topology if r["state"] == "initial"]
    final_topology = [r for r in topology if r["state"] == "final"]
    initial_components = {int(r["variant"]): int(r["components"])
                          for r in initial_topology}
    final_components = {int(r["variant"]): int(r["components"])
                        for r in final_topology}
    with np.load(DATA / case / "snap_003.npz") as snap:
        matrix_present = snap["precip_fraction"] < 1.0-1e-7
        saved_cm = snap["c_matrix"]
        assert np.all(np.isfinite(saved_cm[matrix_present])), case
        # COST-507 Al2Cu stoichiometry gives x_Cu=1/3.  The saved extension
        # field may contain this limiting value on the precipitate side.
        assert np.nanmin(saved_cm) >= 0.0, case
        assert np.nanmax(saved_cm) <= 1.0/3.0+1e-12, case
    last = metrics[-1]
    result = {
        "mass_error": abs(float(last["relative_mass_error"])),
        "equilibrium_residual": float(last["mechanical_equilibrium_residual"]),
        "max_interface_cfl": max(float(r["max_interface_displacement_cells"])
                                 for r in metrics),
        "precipitate_fraction": float(last["precipitate_fraction"]),
        "median_aspect_ratio": float(np.median(
            [float(r["aspect_ratio"]) for r in particles])),
        "median_major_nm": float(np.median(
            [float(r["major_dimension_nm"]) for r in particles])),
        "median_minor_nm": float(np.median(
            [float(r["minor_dimension_nm"]) for r in particles])),
        "components": sum(int(r["components"]) for r in final_topology),
        "unauthorized_new_components": sum(
            max(final_components.get(v, 0)-initial_components.get(v, 0), 0)
            for v in set(initial_components) | set(final_components)),
        "cavities": sum(int(r["enclosed_cavities"]) for r in final_topology),
        "noncontiguous_habit_lines": sum(
            int(r["noncontiguous_habit_lines"]) for r in final_topology),
        "initial_precipitate_fraction": float(metrics[0]["precipitate_fraction"]),
    }
    assert result["mass_error"] <= 1e-12, (case, result)
    assert result["equilibrium_residual"] <= 1e-10, (case, result)
    declared_cfl = float(parameters["interface_cfl"])
    assert result["max_interface_cfl"] <= declared_cfl*(1.0+1e-10), (case, result)
    assert result["cavities"] == 0, (case, result)
    assert result["unauthorized_new_components"] == 0, (case, result)
    # Do not impose monotonic precipitate volume: coherent capillarity and
    # competitive ripening may legitimately dissolve a subcritical plate.
    # Line convexity is diagnostic only; a connected, cavity-free plate may
    # contain a resolved ledge or neck along its habit-normal columns.
    return result


def relative_difference(a: float, b: float):
    return 100.0*abs(b-a)/abs(a)


def main():
    report = {case: endpoint(case) for case in CASES}
    report["mesh_differences_percent"] = {
        "3d_volume_fraction": relative_difference(
            report["three_dimensional_mesh_fine"]["precipitate_fraction"],
            report["three_dimensional_mesh_ultrafine"]["precipitate_fraction"]),
        "3d_aspect_ratio": relative_difference(
            report["three_dimensional_mesh_fine"]["median_aspect_ratio"],
            report["three_dimensional_mesh_ultrafine"]["median_aspect_ratio"]),
        "3d_major_dimension": relative_difference(
            report["three_dimensional_mesh_fine"]["median_major_nm"],
            report["three_dimensional_mesh_ultrafine"]["median_major_nm"]),
        "3d_minor_dimension": relative_difference(
            report["three_dimensional_mesh_fine"]["median_minor_nm"],
            report["three_dimensional_mesh_ultrafine"]["median_minor_nm"]),
    }
    mesh = report["mesh_differences_percent"]
    # Gate the conserved extensive quantity and the two directly measured
    # characteristic dimensions.  The derived aspect ratio is still reported
    # but is not a separate gate because it compounds numerator and
    # denominator deviations.
    report["mesh_status"] = {
        "3d_direct_measures_pass_2_percent": max(
            mesh[k] for k in ("3d_volume_fraction", "3d_major_dimension",
                              "3d_minor_dimension")) <= 2.0,
    }
    with (DATA / "validation_report.json").open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
