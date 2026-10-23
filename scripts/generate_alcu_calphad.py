"""Generate the reproducible CALPHAD baseline for the Al-4 wt.% Cu GPK case.

The script uses the literature COST-507 thermodynamic database through
pycalphad.  It deliberately performs thermodynamic preprocessing only; no new
GPK state variable or evolution equation is introduced.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
import numpy as np
from pycalphad import Database, calculate, equilibrium, variables as v
from scipy.interpolate import UnivariateSpline

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
TDB = PAPER / "calphad" / "COST507-modified.tdb"
CSV_OUT = PAPER / "calphad" / "al4cu_phase_fractions.csv"
DRIVING_OUT = PAPER / "calphad" / "alcu_theta_prime_driving_453K.csv"
PDF_OUT = PAPER / "fig11_al4cu_calphad.pdf"

WT_CU = 0.04
M_AL = 26.9815385
M_CU = 63.546
X_CU = (WT_CU / M_CU) / ((WT_CU / M_CU) + ((1.0 - WT_CU) / M_AL))
COMPONENTS = ["AL", "CU", "VA"]
MOLAR_VOLUME = 1.0e-5


def phase_fractions(db: Database, phases: list[str], temperature: float) -> dict[str, float]:
    """Return nonzero equilibrium phase fractions at fixed T and bulk X(Cu)."""
    result = equilibrium(
        db,
        COMPONENTS,
        phases,
        {v.P: 101325.0, v.T: temperature, v.X("CU"): X_CU},
    )
    names = np.asarray(result.Phase.values).reshape(-1)
    fractions = np.asarray(result.NP.values).reshape(-1)
    output = {phase: 0.0 for phase in phases}
    for name, fraction in zip(names, fractions):
        phase = str(name)
        if phase in output and np.isfinite(fraction) and fraction > 0.0:
            output[phase] += float(fraction)
    return output


def main() -> None:
    db = Database(str(TDB))
    required = {"FCC_A1", "ALCU_THETA", "ALCU_PRIME"}
    missing = required.difference(db.phases)
    if missing:
        raise RuntimeError(f"COST-507 database is missing phases: {sorted(missing)}")

    temperatures = np.linspace(423.15, 823.15, 81)
    rows: list[dict[str, float]] = []
    for temperature in temperatures:
        stable = phase_fractions(db, ["FCC_A1", "ALCU_THETA"], float(temperature))
        metastable = phase_fractions(db, ["FCC_A1", "ALCU_PRIME"], float(temperature))
        rows.append(
            {
                "temperature_K": float(temperature),
                "x_Cu": X_CU,
                "stable_fcc": stable["FCC_A1"],
                "stable_theta": stable["ALCU_THETA"],
                "metastable_fcc": metastable["FCC_A1"],
                "metastable_theta_prime": metastable["ALCU_PRIME"],
            }
        )

    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    with CSV_OUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # Homogeneous metastable FCC branch and theta-prime grand-potential
    # difference.  Equilibrium() cannot be used for the matrix branch here
    # because it convexifies the low-temperature FCC free energy.
    fcc_points = calculate(
        db, COMPONENTS, ["FCC_A1"], T=453.0, P=101325.0,
        output="GM", pdens=15000,
    )
    compositions = np.asarray(fcc_points.X.values).reshape(-1, 2)[:, 1]
    gibbs = np.asarray(fcc_points.GM.values).reshape(-1)
    valid = (np.isfinite(compositions) & np.isfinite(gibbs) &
             (compositions >= 1.0e-5) & (compositions <= 0.05))
    order = np.argsort(compositions[valid])
    compositions = compositions[valid][order]
    gibbs = gibbs[valid][order]
    unique_x, indices = np.unique(np.round(compositions, 12), return_index=True)
    fcc_spline = UnivariateSpline(unique_x, gibbs[indices], s=0, k=3)
    prime = calculate(db, COMPONENTS, ["ALCU_PRIME"], T=453.0,
                      P=101325.0, output="GM")
    prime_x = float(np.asarray(prime.X.values).reshape(-1, 2)[0, 1])
    prime_g = float(np.asarray(prime.GM.values).reshape(-1)[0])
    driving_rows = []
    for composition in np.linspace(1.0e-5, 0.05, 300):
        matrix_g = float(fcc_spline(composition))
        derivative = float(fcc_spline.derivative()(composition))
        driving_molar = matrix_g + (prime_x-composition)*derivative-prime_g
        driving_rows.append({
            "temperature_K": 453.0,
            "matrix_x_Cu": float(composition),
            "theta_prime_x_Cu": prime_x,
            "driving_J_per_mol": driving_molar,
            "driving_J_per_m3": driving_molar/MOLAR_VOLUME,
        })
    with DRIVING_OUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(driving_rows[0]))
        writer.writeheader(); writer.writerows(driving_rows)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.0,
            "axes.titlesize": 10.0,
            "axes.labelsize": 9.5,
            "legend.fontsize": 8.2,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.15), sharey=True)
    t_celsius = temperatures - 273.15
    stable_fcc = np.array([row["stable_fcc"] for row in rows])
    stable_theta = np.array([row["stable_theta"] for row in rows])
    meta_fcc = np.array([row["metastable_fcc"] for row in rows])
    meta_prime = np.array([row["metastable_theta_prime"] for row in rows])

    panels = [
        (axes[0], stable_fcc, stable_theta, r"stable $\alpha$-Al", r"stable $\theta$-Al$_2$Cu", "Stable equilibrium"),
        (axes[1], meta_fcc, meta_prime, r"$\alpha$-Al", r"metastable $\theta'$-Al$_2$Cu", r"Metastable $\theta'$ constraint"),
    ]
    for label, (axis, matrix, product, matrix_name, product_name, title) in zip("ab", panels):
        axis.plot(t_celsius, matrix, color="#2868a9", linewidth=2.1, label=matrix_name)
        axis.plot(t_celsius, product, color="#d65f2e", linewidth=2.1, label=product_name)
        axis.fill_between(t_celsius, 0.0, product, color="#d65f2e", alpha=0.13)
        axis.axvline(520.0, color="#4f5965", linewidth=1.0, linestyle="--")
        axis.text(515.0, 0.56, "solution\ntreatment", color="#4f5965", va="center", ha="right")
        axis.set_title(title, loc="left", fontweight="bold", pad=7)
        axis.text(-0.12, 1.04, label, transform=axis.transAxes, fontweight="bold", fontsize=11)
        axis.set_xlabel(r"temperature [$^\circ$C]")
        axis.set_xlim(150.0, 550.0)
        axis.set_ylim(0.0, 1.02)
        axis.grid(True, color="#d7dee8", linewidth=0.65)
        axis.legend(loc="center left", frameon=False)
    axes[0].set_ylabel("equilibrium phase fraction")
    fig.text(
        0.5,
        0.005,
        r"Al-4 wt.%% Cu ($X_{Cu}=%.5f$); COST-507 database evaluated with pycalphad" % X_CU,
        ha="center",
        fontsize=8.2,
        color="#4f5965",
    )
    fig.subplots_adjust(left=0.095, right=0.985, bottom=0.20, top=0.86, wspace=0.16)
    fig.savefig(PDF_OUT, format="pdf", bbox_inches="tight")
    plt.close(fig)

    for temperature in (473.15, 673.15, 793.15):
        stable = phase_fractions(db, ["FCC_A1", "ALCU_THETA"], temperature)
        print(f"T={temperature:.2f} K: {stable}")
    print(f"wrote {CSV_OUT}")
    print(f"wrote {DRIVING_OUT}")
    print(f"wrote {PDF_OUT}")


if __name__ == "__main__":
    main()
