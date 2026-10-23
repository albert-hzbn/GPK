"""Population-scale 3D Al-Cu ageing audit from 120 h to 168 h.

The dense voxel GPK solver resolves local interface geometry, but a one-micron
representative volume at sub-nanometre spacing is too large for a defensible
workstation calculation.  This script therefore advances a dilute population
of explicit oblate sharp interfaces.  The 120 h TEM population initializes the
calculation.  Number-density loss and theta-double-prime disappearance are
endpoint conditioning observables; the 168 h theta-prime fraction, diameter,
thickness and aspect ratio are conditional challenge observables.

The population closure is deliberately transparent:
  * capillarity ranks particles by equivalent volume, so the smallest dissolve;
  * the measured 120 h diameter, thickness and aspect-ratio scatter determine
    a correlated bivariate population rather than an arbitrary zero covariance;
  * the independently verified local GPK anisotropic response partitions
    volume growth between the in-plane directions and broad-face normal;
  * theta-double-prime dissolution supplies theta-prime volume through a
    stoichiometric Cu-inventory balance; and
  * theta-double-prime particles dissolve over the same 120--168 h interval.

It is a population-scale closure around the GPK interface law, not a replacement
for the local Gaussian geometry solver.  No 168 h shape or phase-fraction
measurement enters the evolution.  Because the measured 168 h count affects
the volume assigned to each survivor, the morphology comparison is conditional,
not an independent prospective validation.  The script records that dependence
explicitly and never converts target agreement into an executable pass gate.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "alcu_validation" / "realistic_population_3d"
DOMAIN_NM = 1000.0
DOMAIN_VOLUME_NM3 = DOMAIN_NM**3
TIMES_H = np.asarray([120.0, 132.0, 144.0, 156.0, 168.0])
CU_FRACTION_THETA_PRIME = 1.0/3.0       # Al2Cu
CU_FRACTION_THETA_DOUBLE_PRIME = 1.0/4.0  # Al3Cu
# Independent local 128^3 GPK verification grows an aspect-8 spherical-start
# plate to 10.4170478785 under the same ledge mobility law.  Its dimensionless
# response transfers local anisotropic kinetics to the population closure; no
# 168 h experimental dimension enters this value.
LOCAL_GPK_INITIAL_ASPECT = 8.0
LOCAL_GPK_FINAL_ASPECT = 10.417047878530138
LOCAL_GPK_ASPECT_RESPONSE = LOCAL_GPK_FINAL_ASPECT/LOCAL_GPK_INITIAL_ASPECT

# Rodriguez-Veiga et al. (2018), Table 1, Al-4 wt.% Cu at 453 K.
EXPERIMENT = {
    "gp_18": {
        "count": 92997, "count_sd": 9543, "fraction": 0.004,
        "fraction_sd": 0.001, "diameter_nm": 14.6,
        "diameter_sd_nm": 0.8, "thickness_nm": 0.256,
        "thickness_sd_nm": None, "aspect": 57.1, "aspect_sd": 3.3,
    },
    "theta_double_prime_18": {
        "count": 13298, "count_sd": 1793, "fraction": 0.003,
        "fraction_sd": 0.001, "diameter_nm": 13.1,
        "diameter_sd_nm": 0.6, "thickness_nm": 1.4,
        "thickness_sd_nm": 0.1, "aspect": 9.3, "aspect_sd": 0.8,
    },
    "theta_double_prime_30": {
        "count": 15229, "count_sd": 1183, "fraction": 0.016,
        "fraction_sd": 0.004, "diameter_nm": 31.3,
        "diameter_sd_nm": 1.4, "thickness_nm": 1.6,
        "thickness_sd_nm": 0.2, "aspect": 19.4, "aspect_sd": 2.8,
    },
    "theta_prime_30": {
        "count": 86, "count_sd": 11, "fraction": 0.006,
        "fraction_sd": 0.002, "diameter_nm": 171.9,
        "diameter_sd_nm": 13.5, "thickness_nm": 2.9,
        "thickness_sd_nm": 0.2, "aspect": 59.5, "aspect_sd": 8.6,
        "population_note": "heterogeneous nucleation at dislocations and grain boundaries",
    },
    "theta_prime_120": {
        "count": 131, "count_sd": 63, "fraction": 0.008,
        "fraction_sd": 0.001, "diameter_nm": 142.5,
        "diameter_sd_nm": 25.1, "thickness_nm": 4.9,
        "thickness_sd_nm": 0.6, "aspect": 28.0, "aspect_sd": 2.4,
    },
    "theta_prime_168": {
        "count": 17, "count_sd": 6, "fraction": 0.010,
        "fraction_sd": 0.005, "diameter_nm": 342.5,
        "diameter_sd_nm": 46.8, "thickness_nm": 9.0,
        "thickness_sd_nm": 0.6, "aspect": 39.0, "aspect_sd": 6.6,
    },
    "theta_double_prime_120": {
        "count": 2883, "count_sd": 170, "fraction": 0.010,
        "fraction_sd": 0.002, "diameter_nm": 45.1,
        "diameter_sd_nm": 2.1, "thickness_nm": 2.1,
        "thickness_sd_nm": 0.2, "aspect": 21.6, "aspect_sd": 2.7,
    },
}


def disk_volume(diameter_nm: np.ndarray, thickness_nm: np.ndarray) -> np.ndarray:
    return 0.25*np.pi*diameter_nm**2*thickness_nm


def constrained_population(rng: np.random.Generator, count: int,
                           diameter: float, diameter_sd: float,
                           thickness: float, thickness_sd: float,
                           aspect: float, aspect_sd: float,
                           target_fraction: float):
    """Draw positive correlated marginals and enforce measured inventory.

    The covariance follows from first-order propagation of the independently
    reported diameter, thickness and aspect-ratio standard deviations.  A
    single common length factor then enforces the reported phase fraction
    without changing aspect ratios or rank ordering.
    """
    covariance = 0.5*thickness**3/diameter * (
        (diameter_sd/thickness)**2
        +(diameter*thickness_sd/thickness**2)**2-aspect_sd**2)
    correlation = np.clip(covariance/(diameter_sd*thickness_sd), -0.95, 0.95)
    covariance_matrix = np.asarray([
        [diameter_sd**2, correlation*diameter_sd*thickness_sd],
        [correlation*diameter_sd*thickness_sd, thickness_sd**2],
    ])
    samples = rng.multivariate_normal(
        [diameter, thickness], covariance_matrix, size=count)
    d = np.clip(samples[:, 0], 0.35*diameter, None)
    h = np.clip(samples[:, 1], 0.35*thickness, None)
    v = disk_volume(d, h)
    scale = (target_fraction*DOMAIN_VOLUME_NM3/np.sum(v))**(1.0/3.0)
    return d*scale, h*scale


def random_metadata(rng: np.random.Generator, count: int):
    positions = rng.uniform(0.0, DOMAIN_NM, size=(count, 3))
    variants = np.resize(np.asarray([1, 2, 3], dtype=int), count)
    rng.shuffle(variants)
    return positions, variants


def target_count(time_h: float, initial: int, final: int) -> int:
    # One effective ledge/coarsening hazard is calibrated only to the measured
    # number-density decline.  This avoids using any 168 h shape observable.
    fraction = (time_h-TIMES_H[0])/(TIMES_H[-1]-TIMES_H[0])
    hazard = np.log(initial/final)
    return max(final, int(np.rint(initial*np.exp(-hazard*fraction))))


def theta_double_prime_fraction(time_h: float) -> float:
    x = np.clip((time_h-TIMES_H[0])/(TIMES_H[-1]-TIMES_H[0]), 0.0, 1.0)
    return EXPERIMENT["theta_double_prime_120"]["fraction"]*(1.0-x)**2


def theta_prime_fraction(time_h: float, capture_efficiency: float = 1.0) -> float:
    """Cu-inventory prediction from Al3Cu dissolution into Al2Cu plates.

    Comparable atomic volumes are assumed.  The 168 h theta-prime fraction is
    therefore a validation observable and is not used in this function.
    """
    dissolved = (EXPERIMENT["theta_double_prime_120"]["fraction"]
                 - theta_double_prime_fraction(time_h))
    transfer = CU_FRACTION_THETA_DOUBLE_PRIME/CU_FRACTION_THETA_PRIME
    return (EXPERIMENT["theta_prime_120"]["fraction"]
            + capture_efficiency*transfer*dissolved)


def simulate(seed: int = 453120, write_particles: bool = True,
             final_theta_prime_count: int | None = None,
             capture_efficiency: float = 1.0,
             local_aspect_response: float = LOCAL_GPK_ASPECT_RESPONSE):
    rng = np.random.default_rng(seed)
    p120 = EXPERIMENT["theta_prime_120"]
    p168 = EXPERIMENT["theta_prime_168"]
    q120 = EXPERIMENT["theta_double_prime_120"]
    if final_theta_prime_count is None:
        final_theta_prime_count = p168["count"]

    d0, h0 = constrained_population(
        rng, p120["count"], p120["diameter_nm"], p120["diameter_sd_nm"],
        p120["thickness_nm"], p120["thickness_sd_nm"], p120["aspect"],
        p120["aspect_sd"], p120["fraction"])
    pos0, var0 = random_metadata(rng, p120["count"])
    volume0 = disk_volume(d0, h0)
    # Gibbs-Thomson/ledge-controlled ripening removes the smallest equivalent
    # volumes first.  Stable sorting makes the realization reproducible.
    survival_order = np.argsort(volume0, kind="stable")[::-1]
    # Variant-independent capillarity must not create a spurious orientation
    # bias in a finite sample.  Stratify variants along the survival ranking so
    # every population checkpoint remains balanced to within one particle.
    var0[survival_order] = np.resize(np.asarray([1, 2, 3], dtype=int), p120["count"])

    qd0, qh0 = constrained_population(
        rng, q120["count"], q120["diameter_nm"], q120["diameter_sd_nm"],
        q120["thickness_nm"], q120["thickness_sd_nm"], q120["aspect"],
        q120["aspect_sd"], q120["fraction"])
    qpos0, qvar0 = random_metadata(rng, q120["count"])
    qorder = np.argsort(disk_volume(qd0, qh0), kind="stable")[::-1]
    qvar0[qorder] = np.resize(np.asarray([1, 2, 3], dtype=int), q120["count"])

    rows = []
    snapshots = {}
    for time_h in TIMES_H:
        count = target_count(time_h, p120["count"], final_theta_prime_count)
        idx = survival_order[:count]
        phase_fraction = theta_prime_fraction(time_h, capture_efficiency)
        volume_ratio = phase_fraction*DOMAIN_VOLUME_NM3/np.sum(volume0[idx])
        progress = (time_h-TIMES_H[0])/(TIMES_H[-1]-TIMES_H[0])
        shape_response = local_aspect_response**progress
        isotropic_scale = volume_ratio**(1.0/3.0)
        # The q^(1/3), q^(-2/3) factors change d/h by q while their
        # contribution to d^2*h is exactly one.
        radial_scale = isotropic_scale*shape_response**(1.0/3.0)
        normal_scale = isotropic_scale*shape_response**(-2.0/3.0)
        d = d0[idx]*radial_scale
        h = h0[idx]*normal_scale

        qfraction = theta_double_prime_fraction(time_h)
        qcount = int(np.rint(q120["count"]*qfraction/q120["fraction"]))
        qidx = qorder[:qcount]
        if qcount:
            qscale = (qfraction*DOMAIN_VOLUME_NM3 /
                      np.sum(disk_volume(qd0[qidx], qh0[qidx])))**(1.0/3.0)
            qd, qh = qd0[qidx]*qscale, qh0[qidx]*qscale
        else:
            qd, qh = np.empty(0), np.empty(0)

        rows.append({
            "time_h": time_h,
            "theta_prime_count_per_um3": count,
            "theta_prime_fraction": phase_fraction,
            "theta_prime_mean_diameter_nm": float(np.mean(d)),
            "theta_prime_sd_diameter_nm": float(np.std(d, ddof=1)) if count > 1 else 0.0,
            "theta_prime_mean_thickness_nm": float(np.mean(h)),
            "theta_prime_sd_thickness_nm": float(np.std(h, ddof=1)) if count > 1 else 0.0,
            "theta_prime_mean_aspect_ratio": float(np.mean(d/h)),
            "theta_prime_sd_aspect_ratio": float(np.std(d/h, ddof=1)) if count > 1 else 0.0,
            "theta_double_prime_count_per_um3": qcount,
            "theta_double_prime_fraction": qfraction,
        })
        snapshots[float(time_h)] = {
            "theta_prime": (pos0[idx], var0[idx], d, h),
            "theta_double_prime": (qpos0[qidx], qvar0[qidx], qd, qh),
        }

    if write_particles:
        OUT.mkdir(parents=True, exist_ok=True)
        with (OUT / "population_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader(); writer.writerows(rows)
        for time_h in (120.0, 168.0):
            snap = snapshots[time_h]
            with (OUT / f"particles_{int(time_h)}h.npz").open("wb") as stream:
                np.savez_compressed(
                    stream, domain_nm=DOMAIN_NM, time_h=time_h,
                    theta_prime_position_nm=snap["theta_prime"][0],
                    theta_prime_variant=snap["theta_prime"][1],
                    theta_prime_diameter_nm=snap["theta_prime"][2],
                    theta_prime_thickness_nm=snap["theta_prime"][3],
                    theta_double_prime_position_nm=snap["theta_double_prime"][0],
                    theta_double_prime_variant=snap["theta_double_prime"][1],
                    theta_double_prime_diameter_nm=snap["theta_double_prime"][2],
                    theta_double_prime_thickness_nm=snap["theta_double_prime"][3],
                )
    return rows


def main():
    rows = simulate()
    endpoint_samples = []
    for seed in range(453120, 453320):
        endpoint_samples.append(simulate(seed, write_particles=False)[-1])
    keys = ("theta_prime_mean_diameter_nm", "theta_prime_mean_thickness_nm",
            "theta_prime_mean_aspect_ratio")
    uncertainty = {}
    for key in keys:
        values = np.asarray([row[key] for row in endpoint_samples])
        uncertainty[key] = {
            "mean": float(np.mean(values)),
            "p2_5": float(np.percentile(values, 2.5)),
            "p97_5": float(np.percentile(values, 97.5)),
        }
    # Sensitivity is deliberately wider than finite-population sampling.  The
    # endpoint count spans the reported +/-1 SD interval, the fraction of
    # dissolved theta-double-prime Cu captured by theta-prime spans 0.5--1.0,
    # and the transferred local anisotropic response is perturbed by +/-10%.
    sensitivity_rows = []
    for final_count in (11, 14, 17, 20, 23):
        for capture_efficiency in (0.50, 0.75, 1.00):
            for response_factor in (0.90, 1.00, 1.10):
                response = LOCAL_GPK_ASPECT_RESPONSE*response_factor
                endpoint = simulate(
                    453120, write_particles=False,
                    final_theta_prime_count=final_count,
                    capture_efficiency=capture_efficiency,
                    local_aspect_response=response)[-1]
                sensitivity_rows.append({
                    "final_theta_prime_count_per_um3": final_count,
                    "theta_double_prime_cu_capture_efficiency": capture_efficiency,
                    "local_aspect_response_factor": response_factor,
                    **endpoint,
                })
    sensitivity_keys = (
        "theta_prime_fraction", "theta_prime_mean_diameter_nm",
        "theta_prime_mean_thickness_nm", "theta_prime_mean_aspect_ratio")
    sensitivity_envelope = {
        key: {
            "minimum": float(min(row[key] for row in sensitivity_rows)),
            "maximum": float(max(row[key] for row in sensitivity_rows)),
        } for key in sensitivity_keys
    }

    validation = {
        "model_version": "GPK population closure audit 2.0",
        "assessment_type": "retrospective endpoint-conditioned challenge",
        "domain_um": 1.0,
        "initial_particles": 3014,
        "final_particles": 17,
        "conditioning_observables": [
            "120 h theta-prime and theta-double-prime summary populations",
            "168 h theta-prime number density",
            "168 h theta-double-prime non-detection",
        ],
        "transferred_non_tem_observable": [
            "independent local 128^3 GPK anisotropic shape response"],
        "conditional_168h_challenge_observables": [
            "theta-prime fraction", "diameter", "thickness", "aspect ratio"],
        "independence_warning": (
            "The 168 h count is conditioned, so per-particle dimensions are not "
            "statistically independent of the endpoint data. No prospective or "
            "parameter-free alloy prediction is claimed."),
        "local_gpk_aspect_response": LOCAL_GPK_ASPECT_RESPONSE,
        "primary_realization": rows[-1],
        "ensemble_95_percent_interval": uncertainty,
        "structural_sensitivity_envelope": sensitivity_envelope,
        "experiment_168h": EXPERIMENT["theta_prime_168"],
    }

    # Reproducibility gates cover only internal invariants and finite output.
    # Experimental agreement is report-only: it cannot make the program pass.
    first, last = rows[0], rows[-1]
    assert first["theta_prime_count_per_um3"] == 131
    assert first["theta_double_prime_count_per_um3"] == 2883
    assert last["theta_prime_count_per_um3"] == 17
    assert last["theta_double_prime_count_per_um3"] == 0
    assert np.isclose(first["theta_prime_fraction"], 0.008)
    assert np.isclose(last["theta_prime_fraction"], 0.0155)
    assert np.isclose(first["theta_double_prime_fraction"], 0.010)
    assert np.isclose(last["theta_double_prime_fraction"], 0.0)
    comparisons = (
        ("theta_prime_fraction", last["theta_prime_fraction"],
         EXPERIMENT["theta_prime_168"]["fraction"],
         EXPERIMENT["theta_prime_168"]["fraction_sd"]),
        ("diameter", last["theta_prime_mean_diameter_nm"],
         EXPERIMENT["theta_prime_168"]["diameter_nm"],
         EXPERIMENT["theta_prime_168"]["diameter_sd_nm"]),
        ("thickness", last["theta_prime_mean_thickness_nm"],
         EXPERIMENT["theta_prime_168"]["thickness_nm"],
         EXPERIMENT["theta_prime_168"]["thickness_sd_nm"]),
        ("aspect_ratio", last["theta_prime_mean_aspect_ratio"],
         EXPERIMENT["theta_prime_168"]["aspect"],
         EXPERIMENT["theta_prime_168"]["aspect_sd"]),
    )
    validation["conditional_standardized_residuals"] = {
        name: float((prediction-observation)/standard_deviation)
        for name, prediction, observation, standard_deviation in comparisons
    }
    assert all(np.isfinite(value) for value in
               validation["conditional_standardized_residuals"].values())
    validation["target_agreement_is_acceptance_gate"] = False
    assert all(np.isfinite(value) for value in
               validation["conditional_standardized_residuals"].values())

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "parameter_sensitivity.csv").open(
            "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=sensitivity_rows[0].keys())
        writer.writeheader(); writer.writerows(sensitivity_rows)
    measurement_rows = []
    for key, record in EXPERIMENT.items():
        measurement_rows.append({"state": key, **record})
    measurement_fields = sorted({field for row in measurement_rows for field in row})
    with (OUT / "literature_measurements.csv").open(
            "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=measurement_fields)
        writer.writeheader(); writer.writerows(measurement_rows)

    evidence_ledger = [
        {"evidence": "120 h theta-prime count/fraction/d/h/aspect",
         "role": "initialization", "source": "Rodriguez-Veiga 2018 Table 1",
         "enters_equations": "yes"},
        {"evidence": "120 h theta-double-prime count/fraction/d/h/aspect",
         "role": "initialization", "source": "Rodriguez-Veiga 2018 Table 1",
         "enters_equations": "yes"},
        {"evidence": "168 h theta-prime number density",
         "role": "endpoint conditioning", "source": "Rodriguez-Veiga 2018 Table 1",
         "enters_equations": "yes"},
        {"evidence": "168 h theta-double-prime non-detection",
         "role": "endpoint conditioning", "source": "Rodriguez-Veiga 2018 Table 1",
         "enters_equations": "yes"},
        {"evidence": "local anisotropic aspect response",
         "role": "transferred numerical response", "source": "independent 128^3 GPK run",
         "enters_equations": "yes"},
        {"evidence": "168 h theta-prime phase fraction",
         "role": "conditional challenge", "source": "Rodriguez-Veiga 2018 Table 1",
         "enters_equations": "no"},
        {"evidence": "168 h theta-prime mean diameter",
         "role": "conditional challenge", "source": "Rodriguez-Veiga 2018 Table 1",
         "enters_equations": "no"},
        {"evidence": "168 h theta-prime mean thickness",
         "role": "conditional challenge", "source": "Rodriguez-Veiga 2018 Table 1",
         "enters_equations": "no"},
        {"evidence": "168 h theta-prime mean aspect ratio",
         "role": "conditional challenge", "source": "Rodriguez-Veiga 2018 Table 1",
         "enters_equations": "no"},
        {"evidence": "anisotropic/isotropic phase-field aspect-ratio ratio",
         "role": "external mechanistic comparison", "source": "Ta 2021",
         "enters_equations": "no"},
    ]
    with (OUT / "evidence_ledger.csv").open(
            "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=evidence_ledger[0].keys())
        writer.writeheader(); writer.writerows(evidence_ledger)

    protocol = {
        "status": "post-hoc audit protocol; not preregistered",
        "source": "Rodriguez-Veiga et al. (2018), Table 1",
        "condition": "Al-4 wt.% Cu, 453 K, 1 micrometre cubed representative volume",
        "conditioning": validation["conditioning_observables"],
        "challenge": validation["conditional_168h_challenge_observables"],
        "reported_metrics": [
            "signed residual", "standardized residual using reported experimental SD",
            "finite-population 95% interval", "structural sensitivity envelope"],
        "acceptance_policy": (
            "No experimental target threshold is asserted in code. Only conservation, "
            "finite-output, endpoint-conditioning and reproducibility invariants are gates."),
        "raw_distribution_limit": (
            "Particle-level TEM pairs are unavailable; KS, Wasserstein and coverage "
            "tests are therefore not computed from reconstructed pseudo-data."),
        "claim": (
            "Given the observed endpoint count/non-detection and a transferred local GPK "
            "shape response, the closure reconstructs the reported mean theta-prime "
            "fraction and dimensions. It does not validate nucleation or coarsening clocks."),
    }
    with (OUT / "validation_protocol.json").open("w", encoding="utf-8") as stream:
        json.dump(protocol, stream, indent=2)
    with (OUT / "validation_report.json").open("w", encoding="utf-8") as stream:
        json.dump(validation, stream, indent=2)

    provenance_files = [
        Path(__file__), OUT / "population_metrics.csv",
        OUT / "parameter_sensitivity.csv", OUT / "literature_measurements.csv",
        OUT / "evidence_ledger.csv", OUT / "validation_protocol.json"]
    provenance = {
        str(path.relative_to(ROOT)).replace("\\", "/"):
            hashlib.sha256(path.read_bytes()).hexdigest()
        for path in provenance_files
    }
    with (OUT / "sha256_manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(provenance, stream, indent=2)
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
