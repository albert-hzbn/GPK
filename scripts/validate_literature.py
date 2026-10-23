#!/usr/bin/env python3
"""
Quick literature-facing validation for GPK Section 11 features.

Checks performed:
1) Recrystallisation: Avrami transform linearity ln(-ln(1-X)) vs ln(t).
2) Ostwald ripening: Gibbs-Thomson critical-radius sorting in 2-D.
3) Solidification: Gibbs-Thomson critical-radius behavior (subcritical melts, supercritical grows).
4) Precipitation: thermodynamic phase selection favors lower-G phase fraction over time.
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


@dataclass
class Fit:
    slope: float
    intercept: float
    r2: float


def linfit(xs: List[float], ys: List[float]) -> Fit:
    n = len(xs)
    if n < 2:
        return Fit(0.0, 0.0, 0.0)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx <= 1e-30:
        return Fit(0.0, my, 0.0)
    m = sxy / sxx
    b = my - m * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (m * x + b)) ** 2 for x, y in zip(xs, ys))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-30 else 0.0
    return Fit(m, b, r2)


def parse_csv(path: Path) -> Tuple[float, float, List[Tuple[int, float, float, float]]]:
    time = 0.0
    dx = 1.0
    rows: List[Tuple[int, float, float, float]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                if "time=" in line:
                    parts = line.replace("#", "").split()
                    for p in parts:
                        if p.startswith("time="):
                            time = float(p.split("=", 1)[1])
                if line.startswith("# dx="):
                    dx = float(line.split("=")[1].split()[0])
                continue
            if not line.strip():
                continue
            x, y, z, gid, alpha, comp, temp, ori = line.split()
            rows.append((int(gid), float(alpha), float(comp), float(temp)))
    return time, dx, rows


def list_snaps(out_dir: Path) -> List[Path]:
    return sorted(out_dir.glob("snap_*.csv"))


def run(cmd: List[str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=str(cwd), check=True)


def gpk_executable(build_dir: Path) -> Path:
    for candidate in (build_dir / "gpk.exe", build_dir / "gpk"):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"GPK executable not found in {build_dir}; build the project first")


def write(path: Path, txt: str) -> None:
    path.write_text(txt, encoding="utf-8")


def validate_recrystallisation(build_dir: Path, work_dir: Path) -> Dict[str, float]:
    out_dir = work_dir / "rex_output"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    ini = work_dir / "recrystallisation_lit.ini"
    write(
        ini,
        f"""
[grid]
Nz = 1
Ny = 96
Nx = 96
dx = 1.0e-6

[time]
t_end = 0.05
n_snaps = 8

[thermo]
M0 = 5.0e-10
sigma0 = 0.3
eps4 = 0.0
D = 0.0
E_stored = 1.0e6

[nucleation]
n_nuclei = 10
nucleus_r = 2
Jdot = 3.0e13
seed = 42

[output]
format = csv
dir = {out_dir}
prefix = snap
compact = false
""".strip() + "\n",
    )
    text = ini.read_text(encoding="utf-8")
    ini.write_text("[run]\nmodel = recrystallisation\n\n" + text, encoding="utf-8")
    run([str(gpk_executable(build_dir)), str(ini)], cwd=build_dir)

    xs: List[float] = []
    ys: List[float] = []
    for fp in list_snaps(out_dir):
        t, _dx, rows = parse_csv(fp)
        n = len(rows)
        if n == 0 or t <= 0:
            continue
        x = sum(1 for gid, _a, _c, _tt in rows if gid != 0) / n
        if 1e-4 < x < 1 - 1e-4:
            xs.append(math.log(t))
            ys.append(math.log(-math.log(1 - x)))
    fit = linfit(xs, ys)
    return {"avrami_n": fit.slope, "avrami_r2": fit.r2}


def validate_ostwald(build_dir: Path, work_dir: Path) -> Dict[str, float]:
    sigma = 0.10
    omega = 5.0e8
    dx = 0.5e-6
    delta_c = 1.0e-4
    r_star = sigma / (omega * delta_c)

    def run_case(name: str, radius_cells: int) -> Tuple[float, float]:
        out_dir = work_dir / f"ostwald_{name}_output"
        if out_dir.exists():
            shutil.rmtree(out_dir)
        ini = work_dir / f"ostwald_{name}.ini"
        write(
            ini,
            f"""
[run]
model = ostwald_ripening

[grid]
Nz = 1
Ny = 48
Nx = 48
dx = {dx}

[time]
t_end = 0.008
n_snaps = 4

[thermo]
M0 = 5.0e-9
sigma0 = {sigma}
eps4 = 0.0
D = 0.0
c_sat = 0.15
Omega = {omega}

[particles]
n_particles = 1
r_mean_cells = {radius_cells}
r_lognormal_s = 0.0
c_particle = {0.15 + delta_c}
c_matrix = {0.15 + delta_c}
seed = 999

[output]
format = csv
dir = {out_dir}
prefix = snap
compact = false
""".strip() + "\n",
        )
        run([str(gpk_executable(build_dir)), str(ini)], cwd=build_dir)

        snapshots = list_snaps(out_dir)
        radii: List[float] = []
        for fp in (snapshots[0], snapshots[-1]):
            _t, snap_dx, rows = parse_csv(fp)
            area = sum(1 for gid, _a, _c, _tt in rows if gid != 0)
            radii.append(math.sqrt(area / math.pi) * snap_dx if area else 0.0)
        return radii[0], radii[1]

    sub0, sub1 = run_case("subcritical", 2)
    sup0, sup1 = run_case("supercritical", 8)
    return {
        "critical_radius_cells": r_star / dx,
        "sub_radius_ratio": sub1 / sub0 if sub0 > 0 else 0.0,
        "sup_radius_ratio": sup1 / sup0 if sup0 > 0 else 0.0,
    }


def validate_solidification(build_dir: Path, work_dir: Path) -> Dict[str, float]:
    def make_ini(name: str, seed_r: int) -> Path:
        out_dir = work_dir / f"solid_{name}_output"
        if out_dir.exists():
            shutil.rmtree(out_dir)
        ini = work_dir / f"solid_{name}.ini"
        write(
            ini,
            f"""
[grid]
Nz = 1
Ny = 96
Nx = 96
dx = 0.5e-6

[time]
t_end = 6.0e-4
n_snaps = 4

[thermo]
M0 = 5.0e-8
sigma0 = 0.20
eps4 = 0.04
D = 3.0e-9
m_liq = -500.0
k_part = 0.3
T_melt = 933.0
L = 1.0e8
stochastic = false
elastic_enabled = false

[initial]
c0 = 0.02
undercooling = 5.0
seed_radius = {seed_r}

[output]
format = csv
dir = {out_dir}
prefix = snap
compact = false
""".strip() + "\n",
        )
        return ini

    ini_sub = make_ini("sub", 1)
    ini_sup = make_ini("sup", 4)

    for ini in (ini_sub, ini_sup):
        text = ini.read_text(encoding="utf-8")
        ini.write_text("[run]\nmodel = solidification\n\n" + text, encoding="utf-8")
        run([str(gpk_executable(build_dir)), str(ini)], cwd=build_dir)

    def sf_last(out_dir: Path) -> float:
        snaps = list_snaps(out_dir)
        t, dx, rows = parse_csv(snaps[-1])
        n = len(rows)
        return sum(1 for gid, _a, _c, _tt in rows if gid == 1) / max(1, n)

    sf_sub = sf_last(work_dir / "solid_sub_output")
    sf_sup = sf_last(work_dir / "solid_sup_output")
    return {"sf_subcritical": sf_sub, "sf_supercritical": sf_sup, "delta_sf": sf_sup - sf_sub}


def validate_precipitation(build_dir: Path, work_dir: Path) -> Dict[str, float]:
    out_dir = work_dir / "precip_output"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    ini = work_dir / "precip_lit.ini"
    write(
        ini,
        f"""
[grid]
Nz = 1
Ny = 96
Nx = 96
dx = 0.5e-6

[time]
t_end = 4.0e-3
n_snaps = 8

[thermo]
M0 = 8.0e-10
sigma0 = 0.18
D = 8.0e-10
n_phases = 3
T_ref = 900.0
a = 0.0, -8.0e6, -7.5e6
b = 0.0, 2.0e7, -1.0e7
c2 = 0.0, 2.5e8, 1.8e8
dT = 0.0, 1.0e4, 5.0e3

[initial]
c0 = 0.06
T0 = 880.0
n_seeds = 8
seed_radius = 1
seed = 123

[output]
format = csv
dir = {out_dir}
prefix = snap
compact = false
""".strip() + "\n",
    )
    text = ini.read_text(encoding="utf-8")
    ini.write_text("[run]\nmodel = precipitation\n\n" + text, encoding="utf-8")
    run([str(gpk_executable(build_dir)), str(ini)], cwd=build_dir)

    snaps = list_snaps(out_dir)
    _, _, rows0 = parse_csv(snaps[0])
    _, _, rows1 = parse_csv(snaps[-1])

    def frac(rows: List[Tuple[int, float, float, float]], p: int) -> float:
        n = len(rows)
        return sum(1 for gid, _a, _c, _tt in rows if gid == p) / max(1, n)

    p1_0, p2_0 = frac(rows0, 1), frac(rows0, 2)
    p1_1, p2_1 = frac(rows1, 1), frac(rows1, 2)
    return {
        "phase1_start": p1_0,
        "phase1_end": p1_1,
        "phase2_start": p2_0,
        "phase2_end": p2_1,
        "phase2_gain": p2_1 - p2_0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--build-dir", default=None,
                    help="folder containing the gpk executable (default: <workspace>/build)")
    args = ap.parse_args()

    ws = Path(args.workspace).resolve()
    build_dir = Path(args.build_dir).resolve() if args.build_dir else ws / "build"
    work_dir = build_dir / "test" / "literature_validation"
    work_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    results["recrystallisation"] = validate_recrystallisation(build_dir, work_dir)
    results["ostwald"] = validate_ostwald(build_dir, work_dir)
    results["solidification"] = validate_solidification(build_dir, work_dir)
    results["precipitation"] = validate_precipitation(build_dir, work_dir)

    print("=== Literature-facing validation summary ===")
    for k, vals in results.items():
        print(f"[{k}]")
        for kk, vv in vals.items():
            print(f"  {kk}: {vv:.6g}")

    checks = {
        "recrystallisation Avrami fit": (
            1.5 <= results["recrystallisation"]["avrami_n"] <= 4.5
            and results["recrystallisation"]["avrami_r2"] >= 0.90
        ),
        "Ostwald Gibbs-Thomson radius sorting": (
            results["ostwald"]["sub_radius_ratio"] < 0.8
            and results["ostwald"]["sup_radius_ratio"] > 1.05
        ),
        "solidification critical-radius ordering": (
            results["solidification"]["delta_sf"] > 0.01
        ),
        "precipitation lower-energy phase gains fraction": (
            results["precipitation"]["phase2_gain"] > 0
        ),
    }
    print("\n=== Gate results ===")
    for label, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
