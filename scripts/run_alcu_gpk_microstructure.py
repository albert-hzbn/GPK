"""Al-4 wt.% Cu sharp-interface precipitation with material physics.

This material module retains the GPK state/geometry/update split while adding
two closures required by theta-prime precipitation:

1. conservative face-flux Cu transport with a stoichiometric precipitate and
   an equilibrium matrix composition; and
2. a periodic Fourier microelasticity solve for cubic Al containing the three
   tetragonal theta-prime eigenstrain variants.
3. orientation-dependent ledge-controlled interface mobility and a
   Cahn--Hoffman anisotropic-capillarity closure for theta-prime plates.

The calculation is a mesoscale morphology demonstration, not a nucleation
model. Initial nuclei are prescribed, as in the comparison phase-field study.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import argparse
import csv
import itertools
import os
import time

import numpy as np
from scipy import fft as spfft
from scipy.ndimage import (binary_dilation, gaussian_filter,
                           label as connected_components)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "alcu_validation" / "precipitation"
CALPHAD_DRIVING = (ROOT / "paper" / "calphad" /
                   "alcu_theta_prime_driving_453K.csv")
_DRIVING_DATA = np.genfromtxt(CALPHAD_DRIVING, delimiter=",", names=True)
_DRIVING_X = np.asarray(_DRIVING_DATA["matrix_x_Cu"], dtype=float)
_DRIVING_JM3 = np.asarray(_DRIVING_DATA["driving_J_per_m3"], dtype=float)


@dataclass
class Parameters:
    dx: float = 1.5e-9
    dt: float = 10.0
    temperature: float = 453.0
    diffusivity: float = 1.66e-20
    mobility: float = 1.0e-19
    mobility_anisotropy_beta: float = 2000.0
    mobility_ledge_exponent: float = 12.0
    c0: float = 0.0169
    # COST-507 ALCU_PRIME is stoichiometric Al2Cu (x_Cu=1/3); its
    # homogeneous-FCC grand-potential crossing at 453 K is x_Cu=3.719e-4.
    c_matrix_eq: float = 3.7189506575706577e-4
    c_theta: float = 1.0/3.0
    molar_volume: float = 1.0e-5
    # Values used by the matched reference phase-field parameterisation.  The
    # paper also discusses larger microscopy estimates, but does not use them
    # in the reported simulations.
    sigma_coherent: float = 0.10
    sigma_rim: float = 0.245
    gaussian_sigma_cells: float = 2.0
    velocity_sigma_cells: float = 0.85
    remap_sigma_cells: float = 0.70
    composition_extension_sigma_cells: float = 0.75
    c11: float = 107.07e9
    c12: float = 63.08e9
    c44: float = 28.52e9
    eigen_small: float = 0.00746
    eigen_large: float = -0.051
    # Resolved theta-prime plates.  Reported radii retain at least four cells
    # through the smallest dimension even at this 8:1 design ratio.
    embryo_aspect_ratio: float = 8.0
    interface_cfl: float = 0.125
    elastic_refresh_steps_2d: int = 10
    elastic_refresh_steps_3d: int = 20
    kernel_workers: int = 8


def roll_grad(a: np.ndarray, dx: float, axis: int) -> np.ndarray:
    """Fourth-order periodic first derivative."""
    return (-np.roll(a, -2, axis=axis) + 8.0*np.roll(a, -1, axis=axis)
            - 8.0*np.roll(a, 1, axis=axis) + np.roll(a, 2, axis=axis)) / (12.0*dx)


def roll_second(a: np.ndarray, dx: float, axis: int) -> np.ndarray:
    """Fourth-order periodic pure second derivative."""
    return (-np.roll(a, -2, axis=axis) + 16.0*np.roll(a, -1, axis=axis)
            - 30.0*a + 16.0*np.roll(a, 1, axis=axis)
            - np.roll(a, 2, axis=axis)) / (12.0*dx*dx)


def variant_eigenstrains(p: Parameters) -> np.ndarray:
    # Rows are x-, y- and z-normal theta-prime plate variants; columns xx,yy,zz.
    s, q = p.eigen_small, p.eigen_large
    return np.asarray([[q, s, s], [s, q, s], [s, s, q]], dtype=float)


def spectral_elasticity(variant_fractions: np.ndarray, p: Parameters):
    """Solve div(sigma)=0 for homogeneous cubic elasticity and eigenstrain.

    Returns elastic driving derivatives for each variant, energy density,
    hydrostatic stress and a dimensionless Fourier equilibrium residual.
    """
    ndim = variant_fractions.ndim - 1
    shape = variant_fractions.shape[1:]
    eig = variant_eigenstrains(p)
    eps0 = np.zeros((3,) + shape, dtype=float)
    for variant in (1, 2, 3):
        ind = variant_fractions[variant-1]
        for j in range(3):
            eps0[j] += eig[variant - 1, j] * ind

    k_axes = [2.0 * np.pi * np.fft.fftfreq(n, d=p.dx) for n in shape]
    mesh = np.meshgrid(*k_axes, indexing="ij", sparse=False)
    k = [np.zeros(shape) for _ in range(3)]
    for a in range(ndim):
        k[a] = mesh[a]
    kx, ky, kz = k

    e0h = [spfft.fftn(eps0[j], workers=p.kernel_workers) for j in range(3)]
    c11, c12, c44 = p.c11, p.c12, p.c44
    # Acoustic tensor A_il = C_ijkl k_j k_k for a cubic solid.
    A = np.empty(shape + (3, 3), dtype=float)
    A[..., 0, 0] = c11*kx*kx + c44*(ky*ky + kz*kz)
    A[..., 1, 1] = c11*ky*ky + c44*(kx*kx + kz*kz)
    A[..., 2, 2] = c11*kz*kz + c44*(kx*kx + ky*ky)
    A[..., 0, 1] = A[..., 1, 0] = (c12 + c44)*kx*ky
    A[..., 0, 2] = A[..., 2, 0] = (c12 + c44)*kx*kz
    A[..., 1, 2] = A[..., 2, 1] = (c12 + c44)*ky*kz
    zero = (kx == 0) & (ky == 0) & (kz == 0)
    A[zero] = np.eye(3)

    tr0h = e0h[0] + e0h[1] + e0h[2]
    b = np.empty(shape + (3,), dtype=complex)
    b[..., 0] = kx*(c11*e0h[0] + c12*(e0h[1] + e0h[2]))
    b[..., 1] = ky*(c11*e0h[1] + c12*(e0h[0] + e0h[2]))
    b[..., 2] = kz*(c11*e0h[2] + c12*(e0h[0] + e0h[1]))
    b[zero] = 0.0
    uh = -1j * np.linalg.solve(A, b[..., None])[..., 0]

    eh = np.empty(shape + (6,), dtype=complex)  # xx,yy,zz,xy,xz,yz
    eh[..., 0] = 1j*kx*uh[..., 0]
    eh[..., 1] = 1j*ky*uh[..., 1]
    eh[..., 2] = 1j*kz*uh[..., 2]
    eh[..., 3] = 0.5j*(kx*uh[..., 1] + ky*uh[..., 0])
    eh[..., 4] = 0.5j*(kx*uh[..., 2] + kz*uh[..., 0])
    eh[..., 5] = 0.5j*(ky*uh[..., 2] + kz*uh[..., 1])
    # Normal-expansion boundary condition: the unconstrained body follows the
    # volume-average transformation strain, so the k=0 macrostress vanishes.
    eh[..., 0][zero] = e0h[0][zero]
    eh[..., 1][zero] = e0h[1][zero]
    eh[..., 2][zero] = e0h[2][zero]
    strain = np.moveaxis(np.asarray([
        spfft.ifftn(eh[..., j], workers=p.kernel_workers).real for j in range(6)
    ]), 0, -1)

    de = strain.copy()
    de[..., 0] -= eps0[0]
    de[..., 1] -= eps0[1]
    de[..., 2] -= eps0[2]
    stress = np.empty_like(de)
    stress[..., 0] = c11*de[..., 0] + c12*(de[..., 1] + de[..., 2])
    stress[..., 1] = c11*de[..., 1] + c12*(de[..., 0] + de[..., 2])
    stress[..., 2] = c11*de[..., 2] + c12*(de[..., 0] + de[..., 1])
    stress[..., 3:] = 2.0*c44*de[..., 3:]
    energy = 0.5*(stress[..., 0]*de[..., 0] + stress[..., 1]*de[..., 1]
                  + stress[..., 2]*de[..., 2] + 2.0*np.sum(stress[..., 3:]*de[..., 3:], axis=-1))
    driving = np.empty((3,) + shape, dtype=float)
    for v in range(3):
        # Variational derivative dF_el/d eta_v = -sigma:epsilon*_v.
        driving[v] = -(stress[..., 0]*eig[v, 0] + stress[..., 1]*eig[v, 1]
                       + stress[..., 2]*eig[v, 2])

    dh0, dh1, dh2 = eh[..., 0]-e0h[0], eh[..., 1]-e0h[1], eh[..., 2]-e0h[2]
    sh = [None]*6
    sh[0] = c11*dh0 + c12*(dh1+dh2)
    sh[1] = c11*dh1 + c12*(dh0+dh2)
    sh[2] = c11*dh2 + c12*(dh0+dh1)
    sh[3] = 2.0*c44*eh[..., 3]
    sh[4] = 2.0*c44*eh[..., 4]
    sh[5] = 2.0*c44*eh[..., 5]
    rx = kx*sh[0] + ky*sh[3] + kz*sh[4]
    ry = kx*sh[3] + ky*sh[1] + kz*sh[5]
    rz = kx*sh[4] + ky*sh[5] + kz*sh[2]
    residual = np.sqrt(np.mean(np.abs(rx)**2 + np.abs(ry)**2 + np.abs(rz)**2))
    scale = np.sqrt(np.mean(np.sum(np.abs(b)**2, axis=-1)))
    residual = float(residual / max(scale, 1e-30))
    hydro = (stress[..., 0] + stress[..., 1] + stress[..., 2]) / 3.0
    return driving, energy, hydro, residual


def initial_state(shape: tuple[int, ...], n_seeds: int, radius: float,
                  seed: int, p: Parameters,
                  seed_layout: list[tuple] | None = None,
                  initial_shape: str = "plate"):
    if initial_shape not in {"plate", "sphere"}:
        raise ValueError("initial_shape must be 'plate' or 'sphere'")
    rng = np.random.default_rng(seed)
    labels = np.zeros(shape, dtype=np.int8)
    coords = np.indices(shape)
    placed = []
    if seed_layout is not None:
        if len(seed_layout) != n_seeds:
            raise ValueError("seed_layout must contain exactly n_seeds entries")
        for entry in seed_layout:
            if len(entry) == 2:
                center_tuple, variant = entry
                radius_scale = 1.0
            elif len(entry) == 3:
                center_tuple, variant, radius_scale = entry
            else:
                raise ValueError("seed entries are (centre, variant[, radius_scale])")
            center = np.asarray(center_tuple, dtype=int)
            if center.shape != (len(shape),) or np.any(center < 0) or np.any(center >= shape):
                raise ValueError(f"invalid seed center {center_tuple} for shape {shape}")
            local_radius = radius*float(radius_scale)
            axes = np.full(len(shape), local_radius, dtype=float)
            if initial_shape == "plate" and variant-1 < len(shape):
                axes[variant-1] = local_radius/p.embryo_aspect_ratio
            # Use minimum-image distances because the simulation is periodic.
            d2 = 0.0
            for a in range(len(shape)):
                delta = np.abs(coords[a]-center[a])
                delta = np.minimum(delta, shape[a]-delta)
                d2 += (delta/axes[a])**2
            labels[d2 <= 1.0] = variant
            placed.append((center, variant))
    else:
        # Random layouts are retained for auxiliary studies, but the material
        # validation cases below use explicit interior layouts so no particle
        # is cut by a periodic face during the reported time window.
        minimum_margin = int(np.ceil(radius + 4.0*p.gaussian_sigma_cells))
        if any(2*minimum_margin >= n for n in shape):
            raise ValueError("domain is too small for the requested seed margin")
        attempts = 0
        while len(placed) < n_seeds and attempts < 10000:
            attempts += 1
            center = np.asarray([rng.integers(minimum_margin, n-minimum_margin) for n in shape])
            if any(np.linalg.norm(center-q) < 4.5*radius for q, _ in placed):
                continue
            variant = int(rng.integers(1, 3 if len(shape) == 2 else 4))
            axes = np.full(len(shape), radius, dtype=float)
            if initial_shape == "plate" and variant-1 < len(shape):
                axes[variant-1] = radius/p.embryo_aspect_ratio
            d2 = sum(((coords[a]-center[a])/axes[a])**2 for a in range(len(shape)))
            labels[d2 <= 1.0] = variant
            placed.append((center, variant))
    alpha = np.ones(shape, dtype=np.float64)
    precip = labels > 0
    f0 = float(np.mean(precip))
    cm0 = (p.c0 - f0*p.c_theta) / (1.0-f0)
    if cm0 <= p.c_matrix_eq:
        raise RuntimeError("initial nuclei consume too much Cu")
    ctot = np.full(shape, cm0, dtype=np.float64)
    ctot[precip] = p.c_theta
    return labels, alpha, ctot, placed


def neighbor_competitor(labels: np.ndarray):
    ndim = labels.ndim
    mask = np.zeros(labels.shape, dtype=bool)
    support = np.zeros((3,) + labels.shape, dtype=np.int16)
    matrix_neighbor = np.zeros(labels.shape, dtype=bool)
    for axis in range(ndim):
        for shift in (-1, 1):
            nb = np.roll(labels, shift, axis=axis)
            mask |= nb != labels
            matrix_neighbor |= nb == 0
            for v in (1, 2, 3):
                support[v-1] += nb == v
    competitor = np.argmax(support, axis=0).astype(np.int8) + 1
    competitor[(labels > 0) & matrix_neighbor] = 0
    competitor[~mask] = labels[~mask]
    return mask, competitor


def variant_fraction_fields(labels: np.ndarray, alpha: np.ndarray,
                            competitor: np.ndarray) -> np.ndarray:
    """Return one continuous precipitate-fraction field per habit variant."""
    fraction = precip_fraction(labels, alpha, competitor)
    variant = np.where(labels > 0, labels, competitor)
    return np.asarray([fraction*(variant == v) for v in (1, 2, 3)])


def geometry(labels: np.ndarray, competitor: np.ndarray,
             geometry_fields: np.ndarray, p: Parameters,
             return_variant_normals: bool = False):
    """Label-conditioned Gaussian normals and Hessian curvature.

    The differentiated field is the temporary Gaussian projection of each
    persistent label support.  Sub-cell position remains in ``alpha`` and does
    not become a diffuse state field. Fourth-order axial and mixed derivatives
    match the production GPK geometry pipeline.
    """
    ndim = labels.ndim
    normals = np.zeros((ndim,) + labels.shape)
    curvature = np.zeros(labels.shape)

    def project_variant(v: int):
        active = (competitor == v) | (labels == v)
        if not np.any(active):
            return v, active, None, None
        field = gaussian_filter(geometry_fields[v-1], p.gaussian_sigma_cells, mode="wrap")
        grad = np.asarray([roll_grad(field, p.dx, a) for a in range(ndim)])
        mag2 = np.sum(grad*grad, axis=0)
        resolved = mag2 > (1e-4/p.dx)**2
        mag = np.sqrt(np.maximum(mag2, 1e-60))
        nout = np.zeros_like(grad)
        nout[:, resolved] = -grad[:, resolved]/mag[resolved]
        hessian = np.empty((ndim, ndim) + labels.shape, dtype=float)
        for a in range(ndim):
            hessian[a, a] = roll_second(field, p.dx, a)
            for b in range(a+1, ndim):
                mixed = roll_grad(roll_grad(field, p.dx, a), p.dx, b)
                hessian[a, b] = hessian[b, a] = mixed
        trace_h = sum(hessian[a, a] for a in range(ndim))
        ghg = np.einsum("i...,ij...,j...->...", grad, hessian, grad, optimize=True)
        kap = np.zeros_like(field)
        kap[resolved] = -(mag2[resolved]*trace_h[resolved]-ghg[resolved]) / mag[resolved]**3
        kap = np.clip(kap, -1.0/(p.gaussian_sigma_cells*p.dx),
                      1.0/(p.gaussian_sigma_cells*p.dx))
        return v, active, nout, kap

    # Gaussian filtering and Hessian construction dominate the local geometry
    # stage and are independent for the three theta-prime variants.  SciPy's
    # compiled filters release the GIL, so a small thread pool lowers elapsed
    # time without duplicating the full simulation state in memory.
    with ThreadPoolExecutor(max_workers=min(3, p.kernel_workers)) as pool:
        results = list(pool.map(project_variant, (1, 2, 3)))
    full_variant_normals = np.zeros((3, ndim) + labels.shape, dtype=float)
    for v, active, nout, kap in results:
        if nout is None:
            continue
        full_variant_normals[v-1] = nout
        for a in range(ndim):
            normals[a][active] = nout[a][active]
        curvature[active] = kap[active]
    if return_variant_normals:
        return normals, curvature, full_variant_normals
    return normals, curvature


def project_pair_velocity(labels: np.ndarray, competitor: np.ndarray,
                          mask: np.ndarray, velocity: np.ndarray,
                          p: Parameters) -> np.ndarray:
    """Remove unresolved tangential velocity oscillations without shape repair.

    The constitutive velocity is projected separately for every unordered
    resident--candidate pair and normalized by the projected pair support.
    This material module expresses velocity in one precipitate-outward frame
    on both sides of an interface (positive means precipitation), rather than
    in each resident label's outward frame.  The projector must therefore
    preserve the same physical sign on both sides; applying a resident-label
    sign here would cancel the chemical driving force.  Unrelated phases at a
    junction still cannot contaminate one another.
    """
    projected = velocity.copy()
    pairs = sorted({tuple(sorted((int(a), int(b))))
                    for a, b in zip(labels[mask], competitor[mask]) if a != b})
    def project_pair(pair):
        first, second = pair
        pair_mask = mask & (((labels == first) & (competitor == second)) |
                            ((labels == second) & (competitor == first)))
        if not np.any(pair_mask):
            return pair_mask, None
        support = gaussian_filter(pair_mask.astype(float),
                                  p.velocity_sigma_cells, mode="wrap")
        numerator = gaussian_filter(
            np.where(pair_mask, velocity, 0.0),
                                    p.velocity_sigma_cells, mode="wrap")
        values = numerator[pair_mask] / np.maximum(support[pair_mask], 1.0e-12)
        return pair_mask, values

    with ThreadPoolExecutor(max_workers=min(len(pairs), p.kernel_workers) or 1) as pool:
        pair_results = list(pool.map(project_pair, pairs))
    for pair_mask, values in pair_results:
        if values is not None:
            projected[pair_mask] = values
    return projected


def _periodic_patch(labels: np.ndarray, index: tuple[int, ...]) -> np.ndarray:
    axes = [np.asarray([(i-1) % n, i, (i+1) % n])
            for i, n in zip(index, labels.shape)]
    return labels[np.ix_(*axes)]


def _topological_number(mask: np.ndarray, full_connectivity: bool) -> int:
    """Digital topological number in a 3^d neighbourhood.

    With face connectivity, only components incident on a face neighbour of
    the centre enter the topological number.  With full connectivity every
    non-central voxel in the 3^d patch is adjacent to the centre and is
    therefore counted.  The centre must already be cleared by the caller.
    """
    occupied = {tuple(int(v) for v in coordinate)
                for coordinate in np.argwhere(mask)}
    if not occupied:
        return 0
    if full_connectivity:
        offsets = [offset for offset in itertools.product((-1, 0, 1),
                                                           repeat=mask.ndim)
                   if any(offset)]
        incident = occupied
    else:
        offsets = []
        incident = set()
        centre = (1,)*mask.ndim
        for axis in range(mask.ndim):
            for shift in (-1, 1):
                offset = [0]*mask.ndim; offset[axis] = shift
                offsets.append(tuple(offset))
                position = list(centre); position[axis] += shift
                incident.add(tuple(position))
    components = 0
    unseen = set(occupied)
    while unseen:
        seed = unseen.pop(); stack = [seed]; touches = seed in incident
        while stack:
            point = stack.pop()
            for offset in offsets:
                neighbour = tuple(a+b for a, b in zip(point, offset))
                if neighbour in unseen:
                    unseen.remove(neighbour); stack.append(neighbour)
                    touches |= neighbour in incident
        components += int(touches)
    return components


def periodic_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Face-connected components on a periodic Cartesian grid."""
    labels, count = connected_components(mask, structure=None)
    if count == 0:
        return labels, 0
    parent = np.arange(count+1, dtype=np.int32)

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = int(parent[value])
        return value

    def union(first: int, second: int) -> None:
        if first == 0 or second == 0:
            return
        first, second = find(first), find(second)
        if first != second:
            parent[second] = first

    for axis in range(mask.ndim):
        lower = np.take(labels, 0, axis=axis).ravel()
        upper = np.take(labels, -1, axis=axis).ravel()
        for first, second in zip(lower, upper):
            union(int(first), int(second))
    roots = np.asarray([find(i) for i in range(count+1)], dtype=np.int32)
    unique = sorted(set(int(x) for x in roots[1:]))
    compact = np.zeros(count+1, dtype=np.int32)
    for new_label, root in enumerate(unique, 1):
        compact[roots == root] = new_label
    return compact[labels], len(unique)


def topology_simple_relabel(labels: np.ndarray, index: tuple[int, ...],
                            old_label: int, new_label: int,
                            label_counts: dict[int, int],
                            local_only: bool = False) -> bool:
    """Dual-connectivity simple-point test for a generic p->q relabel.

    Material labels use face connectivity. A local multi-branch event triggers
    a periodic global component check: genuine
    resident splitting and disconnected-product merging are deferred, whereas
    a locally critical cavity/tunnel or T1 event may proceed when its branches
    are already connected globally.  A one-cell component may disappear as a
    resolution-certified extinction event.
    """
    if old_label == new_label:
        return False
    patch = _periodic_patch(labels, index)
    centre = (1,)*labels.ndim

    old_object = patch == old_label
    old_object[centre] = False
    old_components = _topological_number(old_object, full_connectivity=False)
    extinction = label_counts.get(old_label, 0) <= 1
    if old_components == 0 and not extinction:
        return False
    if old_components > 1:
        if local_only:
            return False
        old_mask = labels == old_label
        old_mask[index] = False
        cc, _ = periodic_components(old_mask)
        neighbour_ids = set()
        for axis in range(labels.ndim):
            for step in (-1, 1):
                offset = [0]*labels.ndim; offset[axis] = step
                neighbour = tuple((i+d) % n for i, d, n in
                                  zip(index, offset, labels.shape))
                if labels[neighbour] == old_label:
                    neighbour_ids.add(int(cc[neighbour]))
        if len(neighbour_ids-{0}) > 1:
            return False
    new_object = patch == new_label
    new_object[centre] = False
    new_components = _topological_number(new_object, full_connectivity=False)
    if new_components == 0:
        return False
    if new_components > 1:
        if local_only:
            return False
        cc, _ = periodic_components(labels == new_label)
        neighbour_ids = set()
        for axis in range(labels.ndim):
            for step in (-1, 1):
                offset = [0]*labels.ndim; offset[axis] = step
                neighbour = tuple((i+d) % n for i, d, n in
                                  zip(index, offset, labels.shape))
                if labels[neighbour] == new_label:
                    neighbour_ids.add(int(cc[neighbour]))
        if len(neighbour_ids-{0}) > 1:
            return False
    return True


def commit_topology_transactions(labels: np.ndarray, alpha: np.ndarray,
                                 competitor: np.ndarray,
                                 crossing: np.ndarray) -> int:
    """Commit topology-safe relabels as deterministic local transactions.

    Candidates are visited in lexicographic order and each certificate is
    evaluated against the already committed state.  This is the conservative
    serial reference for the model-independent topology rule; production
    implementations may parallelise a proper distance-two conflict coloring.
    Deferred cells retain their accumulated sub-cell displacement and are
    reconsidered after neighbouring events.  No material- or
    morphology-specific repair occurs.
    """
    if not np.any(crossing):
        return 0
    label_counts = {int(v): int(np.count_nonzero(labels == v))
                    for v in np.unique(labels)}
    deferred = 0
    for coordinate in np.argwhere(crossing):
        index = tuple(int(x) for x in coordinate)
        old_label = int(labels[index]); new_label = int(competitor[index])
        # Prescribed theta-prime nuclei carry particle identity.  Use the
        # strict local certificate so a growing rim cannot hollow its parent
        # nucleus into a tunnel or split it into disconnected fragments.
        if topology_simple_relabel(labels, index, old_label, new_label,
                                   label_counts, local_only=True):
            labels[index] = new_label
            alpha[index] = 1.0
            label_counts[old_label] = label_counts.get(old_label, 0)-1
            label_counts[new_label] = label_counts.get(new_label, 0)+1
        else:
            deferred += 1
            alpha[index] = max(float(alpha[index]), 1.0e-6)
    return deferred


def conservative_gaussian_remap(labels: np.ndarray, alpha: np.ndarray,
                                sigma_cells: float) -> int:
    """Remove grid-scale interface modes by a volume-conserving Gaussian map.

    Each face-connected component is replaced by the equal-volume upper level
    set of its own narrow Gaussian projection inside a one-cell admissible
    band.  The operation is isotropic, label conditioned, component preserving
    and independent of material, phase identity or crystallographic axes.
    """
    original = labels.copy()
    remapped = labels.copy()
    structure = np.ones((3,)*labels.ndim, dtype=bool)
    for value in (int(v) for v in np.unique(original) if v != 0):
        cc, count = connected_components(original == value, structure=None)
        for component in range(1, count+1):
            support = cc == component
            target = int(np.count_nonzero(support))
            if target <= 1:
                continue
            admissible = binary_dilation(support, structure=structure,
                                         iterations=1, border_value=0)
            admissible &= (remapped == 0) | support
            score = gaussian_filter(support.astype(float), sigma_cells,
                                    mode="wrap")
            candidates = np.flatnonzero(admissible)
            order = np.lexsort((candidates, -score.ravel()[candidates]))
            selected = candidates[order[:target]]
            remapped[support] = 0
            remapped.ravel()[selected] = value
    changed = remapped != labels
    labels[:] = remapped
    alpha[changed] = 1.0
    return int(np.count_nonzero(changed))


def balanced_gaussian_remap(labels: np.ndarray, alpha: np.ndarray,
                            sigma_cells: float) -> int:
    """Multilabel, exactly volume-balanced anti-alias projection.

    Cells propose the neighbouring label with the larger narrow-Gaussian
    support. For every unordered label pair, equal numbers of the strongest
    proposals are exchanged in opposite directions. Each label count is
    therefore invariant, while one-cell protrusions trade places with
    one-cell indentations. Exchanges are restricted to labels in the same
    chemical phase class: otherwise a volume-balanced matrix--precipitate
    exchange would cancel the phase change just committed by the kinetic law.
    """
    values = np.unique(labels)
    def score(value):
        return int(value), gaussian_filter((labels == value).astype(float),
                                           sigma_cells, mode="wrap")
    with ThreadPoolExecutor(max_workers=min(len(values), 4)) as pool:
        scores = dict(pool.map(score, values))
    interface = np.zeros(labels.shape, dtype=bool)
    candidates = {int(value): np.zeros(labels.shape, dtype=bool)
                  for value in values}
    for axis in range(labels.ndim):
        for shift in (-1, 1):
            neighbour = np.roll(labels, shift, axis=axis)
            interface |= neighbour != labels
            for value in values:
                candidates[int(value)] |= neighbour == value
    proposals: dict[tuple[int, int], list[tuple[float, int]]] = {}
    flat_labels = labels.ravel()
    for flat in np.flatnonzero(interface):
        old = int(flat_labels[flat]); old_score = float(scores[old].ravel()[flat])
        best = old; best_gain = 0.0
        for value in values:
            value = int(value)
            if value == old or not candidates[value].ravel()[flat]:
                continue
            gain = float(scores[value].ravel()[flat])-old_score
            if gain > best_gain:
                best = value; best_gain = gain
        if best != old:
            proposals.setdefault((old, best), []).append((best_gain, int(flat)))

    changed = np.zeros(labels.size, dtype=bool)
    label_counts = {int(v): int(np.count_nonzero(labels == v)) for v in values}
    for ia, first in enumerate(values):
        for second in values[ia+1:]:
            first = int(first); second = int(second)
            # In this material module, zero is alpha-Al and positive labels
            # are theta-prime orientation variants.  BGR regularizes identity
            # boundaries within a phase; it must never alter phase fractions.
            if (first == 0) != (second == 0):
                continue
            forward = sorted(proposals.get((first, second), ()), reverse=True)
            reverse = sorted(proposals.get((second, first), ()), reverse=True)
            count = min(len(forward), len(reverse))
            for (_, flat_f), (_, flat_r) in zip(forward[:count], reverse[:count]):
                index_f = np.unravel_index(flat_f, labels.shape)
                index_r = np.unravel_index(flat_r, labels.shape)
                if (int(flat_labels[flat_f]) != first or
                        int(flat_labels[flat_r]) != second):
                    continue
                if not topology_simple_relabel(labels, index_f, first, second,
                                               label_counts, local_only=True):
                    continue
                flat_labels[flat_f] = second
                label_counts[first] -= 1; label_counts[second] += 1
                if not topology_simple_relabel(labels, index_r, second, first,
                                               label_counts, local_only=True):
                    flat_labels[flat_f] = first
                    label_counts[first] += 1; label_counts[second] -= 1
                    continue
                flat_labels[flat_r] = first
                label_counts[second] -= 1; label_counts[first] += 1
                changed[flat_f] = True; changed[flat_r] = True
    alpha.ravel()[changed] = 1.0
    return int(np.count_nonzero(changed))


def precip_fraction(labels, alpha, competitor):
    return np.where(labels > 0, alpha, np.where(competitor > 0, 1.0-alpha, 0.0))


def diffuse_conservative(ctot, fprec, p: Parameters, dt: float):
    matrix = np.maximum(1.0-fprec, 1e-8)
    cm = np.where(matrix > 1e-7, (ctot-fprec*p.c_theta)/matrix, p.c_matrix_eq)
    cm = np.clip(cm, 0.0, p.c_theta)
    dc = np.zeros_like(ctot)
    for axis in range(ctot.ndim):
        mf = 0.5*(matrix + np.roll(matrix, -1, axis=axis))
        flux_plus = -p.diffusivity*mf*(np.roll(cm, -1, axis=axis)-cm)/p.dx
        dc -= (flux_plus - np.roll(flux_plus, 1, axis=axis))/p.dx
    return ctot + dt*dc, cm


def diffuse_conservative_subcycled(ctot, fprec, p: Parameters, dt: float):
    """Advance conservative Cu transport at its own explicit stability scale."""
    stable_dt = 0.4*p.dx*p.dx/(2.0*ctot.ndim*p.diffusivity)
    substeps = max(1, int(np.ceil(dt/stable_dt)))
    sub_dt = dt/substeps
    cm = None
    for _ in range(substeps):
        ctot, cm = diffuse_conservative(ctot, fprec, p, sub_dt)
    return ctot, cm, substeps


def conservative_partition_transfer(ctot: np.ndarray, old_fraction: np.ndarray,
                                    new_fraction: np.ndarray,
                                    matrix_composition: np.ndarray,
                                    p: Parameters) -> None:
    """Apply the local Stefan partition transfer without changing total Cu.

    Creating theta-prime requires ``(c_theta-c_matrix) df`` Cu locally.  That
    amount is drawn from the surrounding matrix through a normalized Gaussian
    neighbourhood; dissolution performs the reverse transfer.  This replaces
    the incorrect cell-local lever-rule gate, which could freeze an otherwise
    admissible sharp-interface crossing.
    """
    delta_fraction = np.clip(new_fraction, 0.0, 1.0) - old_fraction
    transfer = delta_fraction*(p.c_theta-matrix_composition)
    total_transfer = float(np.sum(transfer))
    ctot += transfer
    if abs(total_transfer) <= 1.0e-30:
        return
    matrix = np.clip(1.0-new_fraction, 0.0, 1.0)
    weight = gaussian_filter(np.abs(transfer),
                             p.composition_extension_sigma_cells,
                             mode="wrap")*matrix
    correction = -total_transfer
    if correction < 0.0:
        # Draw only from matrix Cu above the equilibrium inventory.
        capacity = np.maximum(ctot-p.c_matrix_eq*matrix, 0.0)
        weight *= capacity
        if float(np.sum(capacity)) < -correction*(1.0-1.0e-12):
            raise RuntimeError("insufficient matrix Cu for partition transfer")
    weight_sum = float(np.sum(weight))
    if weight_sum <= 0.0:
        raise RuntimeError("empty matrix neighbourhood for partition transfer")
    ctot += correction*weight/weight_sum
    # Remove the final floating-point summation residue deterministically.
    residue = total_transfer + float(np.sum(correction*weight/weight_sum))
    ctot.ravel()[int(np.argmax(weight))] -= residue


def interface_matrix_composition(cm: np.ndarray, fprec: np.ndarray,
                                 p: Parameters) -> np.ndarray:
    """Extend the one-sided matrix composition to the sharp interface.

    ``c_m`` has no thermodynamic meaning in a pure theta-prime cell.  Using an
    arbitrary equilibrium value there biases the two-sided pair projection
    toward dissolution.  A normalized narrow Gaussian projection supplies the
    trace of the surrounding alpha-Al composition on both sides of the sharp
    interface without changing either the conserved field or phase labels.
    """
    matrix_fraction = np.clip(1.0-fprec, 0.0, 1.0)
    sigma = p.composition_extension_sigma_cells
    numerator = gaussian_filter(cm*matrix_fraction, sigma, mode="wrap")
    denominator = gaussian_filter(matrix_fraction, sigma, mode="wrap")
    trace = numerator/np.maximum(denominator, 1.0e-12)
    return np.clip(trace, 0.0, p.c_theta)


def calphad_chemical_driving(cm_trace: np.ndarray) -> np.ndarray:
    """COST-507 alpha/Al2Cu grand-potential difference at 453 K."""
    return np.interp(cm_trace, _DRIVING_X, _DRIVING_JM3,
                     left=_DRIVING_JM3[0], right=_DRIVING_JM3[-1])


def cached_label_geometry(labels: np.ndarray, competitor: np.ndarray,
                          p: Parameters):
    """Geometry terms that are invariant between categorical label events."""
    geometry_fields = np.asarray([(labels == v).astype(float) for v in (1, 2, 3)])
    normals, kappa, variant_normals = geometry(
        labels, competitor, geometry_fields, p, return_variant_normals=True)
    variant = np.where(labels > 0, labels, competitor)
    capillary = np.zeros_like(labels, dtype=float)
    normal_to_face = np.zeros_like(capillary)
    for v in (1, 2, 3):
        nv = variant_normals[v-1]
        coeff = np.asarray([
            p.sigma_coherent**2 if a == v-1 else p.sigma_rim**2
            for a in range(labels.ndim)
        ])
        gamma = np.sqrt(np.maximum(np.sum(
            coeff.reshape((-1,) + (1,)*labels.ndim)*nv*nv, axis=0), 1e-30))
        xi = [coeff[a]*nv[a]/gamma for a in range(labels.ndim)]
        cap_v = sum(roll_grad(xi[a], p.dx, a) for a in range(labels.ndim))
        capillary[variant == v] = cap_v[variant == v]
        axis = v-1
        if axis < labels.ndim:
            normal_to_face[variant == v] = nv[axis][variant == v]**2
    ledge_availability = (np.maximum(1.0-normal_to_face, 0.0)
                          ** p.mobility_ledge_exponent)
    mobility_factor = ((1.0+p.mobility_anisotropy_beta*ledge_availability) /
                       (1.0+p.mobility_anisotropy_beta))
    return normals, kappa, variant_normals, variant, capillary, mobility_factor


def particle_statistics(labels: np.ndarray, alpha: np.ndarray,
                        competitor: np.ndarray, p: Parameters):
    rows = []
    for variant in (1, 2, 3):
        cc, ncc = periodic_components(labels == variant)
        for component in range(1, ncc+1):
            xyz = np.argwhere(cc == component)
            if len(xyz) < 5: continue
            weights = alpha[cc == component]
            centered = np.empty_like(xyz, dtype=float)
            # Minimum-image coordinates prevent a periodic plate cut by a box
            # face from appearing as several tiny particles or an object with
            # the diameter of the simulation cell.
            for axis, size in enumerate(labels.shape):
                angle = 2.0*np.pi*xyz[:, axis]/size
                mean_angle = np.arctan2(np.sum(weights*np.sin(angle)),
                                        np.sum(weights*np.cos(angle)))
                centre = (mean_angle % (2.0*np.pi))*size/(2.0*np.pi)
                centered[:, axis] = ((xyz[:, axis]-centre+0.5*size) % size)-0.5*size
            cov = (centered*weights[:, None]).T@centered / max(float(np.sum(weights)), 1e-30)
            ev = np.maximum(np.linalg.eigvalsh(np.atleast_2d(cov)), 1e-12)
            aspect = float(np.sqrt(ev[-1]/ev[0]))
            extents = 4.0*np.sqrt(ev)*p.dx*1e9  # approximately two-sigma diameters
            rows.append((variant, float(np.sum(weights)), aspect,
                         float(extents[-1]), float(extents[0])))
    return rows


def topology_metrics(variant_fractions: np.ndarray, threshold: float = 0.5):
    """Face-connected component, cavity and grid-boundary audit per variant."""
    rows = []
    ndim = variant_fractions.ndim-1
    for variant in (1, 2, 3):
        mask = variant_fractions[variant-1] >= threshold
        _, components = periodic_components(mask)
        _, n_background = periodic_components(~mask)
        cavities = max(n_background-1, 0)
        exposed = 0
        for axis in range(ndim):
            exposed += int(np.count_nonzero(mask != np.roll(mask, 1, axis=axis)))
        noncontiguous_lines = 0
        habit_axis = variant-1
        if habit_axis < ndim:
            moved = np.moveaxis(mask, habit_axis, -1).reshape(-1, mask.shape[habit_axis])
            for line in moved:
                occupied = np.flatnonzero(line)
                if len(occupied) and len(occupied) != occupied[-1]-occupied[0]+1:
                    noncontiguous_lines += 1
        rows.append((variant, components, cavities, int(np.count_nonzero(mask)),
                     exposed, noncontiguous_lines))
    return rows


def run_case(name: str, shape: tuple[int, ...], n_seeds: int, radius: float,
             t_end: float, snapshot_times: list[float], seed: int,
             dx: float | None = None,
             seed_layout: list[tuple] | None = None,
             gaussian_sigma_cells: float | None = None,
             interface_cfl: float | None = None,
             mobility_anisotropy_beta: float | None = None,
             mobility_ledge_exponent: float | None = None,
             remap_sigma_cells: float | None = None,
             initial_shape: str = "plate"):
    p = Parameters()
    if dx is not None: p.dx = dx
    if gaussian_sigma_cells is not None:
        p.gaussian_sigma_cells = gaussian_sigma_cells
    if interface_cfl is not None:
        p.interface_cfl = interface_cfl
    if mobility_anisotropy_beta is not None:
        p.mobility_anisotropy_beta = mobility_anisotropy_beta
    if mobility_ledge_exponent is not None:
        p.mobility_ledge_exponent = mobility_ledge_exponent
    if remap_sigma_cells is not None:
        p.remap_sigma_cells = remap_sigma_cells
    case = OUT / name
    case.mkdir(parents=True, exist_ok=True)
    labels, alpha, ctot, seeds = initial_state(
        shape, n_seeds, radius, seed, p, seed_layout=seed_layout,
        initial_shape=initial_shape
    )
    initial_mass = float(np.sum(ctot))
    initial_precipitate_fraction = float(np.mean(labels > 0))
    initial_component_counts = {
        v: periodic_components(labels == v)[1]
        for v in (1, 2, 3)
    }
    mask, comp = neighbor_competitor(labels)
    variant_fractions = variant_fraction_fields(labels, alpha, comp)
    elastic_drive, eel, hydro, residual = spectral_elasticity(variant_fractions, p)
    elastic_every = (p.elastic_refresh_steps_2d if labels.ndim == 2
                     else p.elastic_refresh_steps_3d)
    elastic_reference = variant_fractions.copy()
    next_snap = 0
    metrics = []
    t = 0.0
    start = time.perf_counter()

    def save(snap_idx):
        mask, comp = neighbor_competitor(labels)
        fp = precip_fraction(labels, alpha, comp)
        vf_fields = variant_fraction_fields(labels, alpha, comp)
        _, save_eel, save_hydro, save_residual = spectral_elasticity(vf_fields, p)
        matrix = np.maximum(1.0-fp, 1e-8)
        cm = np.where(matrix > 1e-7, (ctot-fp*p.c_theta)/matrix, p.c_matrix_eq)
        cm = np.clip(cm, 0.0, p.c_theta)
        # Matrix composition is undefined in a pure precipitate cell.  Store
        # NaN there instead of amplifying roundoff by division through the
        # matrix-fraction floor; this field is diagnostic and never reloaded
        # by the evolution solver.
        cm[fp >= 1.0-1e-7] = np.nan
        np.savez_compressed(case / f"snap_{snap_idx:03d}.npz", labels=labels,
                            alpha=alpha, c_total=ctot, c_matrix=cm,
                            precip_fraction=fp,
                            variant_field=np.where(labels > 0, labels, comp),
                            variant_fractions=vf_fields,
                            elastic_energy=save_eel,
                            hydrostatic_stress=save_hydro, time_s=t, dx_m=p.dx,
                            mechanical_equilibrium_residual=save_residual)

    save(0); next_snap = 1
    step = 0
    geometry_dirty = True
    while t < t_end-1e-9:
        mask, comp = neighbor_competitor(labels)
        fp = precip_fraction(labels, alpha, comp)
        variant_fractions = variant_fraction_fields(labels, alpha, comp)
        matrix = np.maximum(1.0-fp, 1e-8)
        cm = np.where(matrix > 1e-7, (ctot-fp*p.c_theta)/matrix, p.c_matrix_eq)
        cm = np.clip(cm, 0.0, p.c_theta)
        if step % elastic_every == 0:
            elastic_drive, eel, hydro, residual = spectral_elasticity(variant_fractions, p)
            elastic_reference = variant_fractions.copy()

        if geometry_dirty:
            (normals, kappa, variant_normals, variant, capillary,
             mobility_factor) = cached_label_geometry(labels, comp, p)
            geometry_dirty = False
        cm_interface = interface_matrix_composition(cm, fp, p)
        chem = calphad_chemical_driving(cm_interface)
        gel = np.zeros_like(alpha)
        for v in (1, 2, 3): gel[variant == v] = elastic_drive[v-1][variant == v]
        # theta-prime broad faces (normal parallel to the tetragonal axis)
        # advance by ledge nucleation and are much slower than the rim.  Ta
        # et al. report beta_mu=2000, i.e. mu_face/mu_rim=1/2001.  This smooth
        # tensor form preserves that measured limiting ratio without adding a
        # grid-aligned cusp to the sharp-interface update.
        velocity = p.mobility*mobility_factor*(chem-gel-capillary)
        velocity[~mask] = 0.0
        velocity = project_pair_velocity(labels, comp, mask, velocity, p)
        vmax = float(np.max(np.abs(velocity[mask]))) if np.any(mask) else 0.0
        interface_dt = p.interface_cfl*p.dx/max(vmax, 1e-30)
        dt = min(p.dt, interface_dt, t_end-t)
        max_displacement = vmax*dt/p.dx
        ctot[:], cm, diffusion_substeps = diffuse_conservative_subcycled(
            ctot, fp, p, dt)
        da = np.zeros_like(alpha)
        grow = (labels == 0) & (comp > 0) & (velocity > 0)
        # The complete variational driving force decides local dissolution.
        # A flat-interface c_m<c_eq gate is incorrect because Gibbs--Thomson
        # and elastic terms can make a high-curvature segment retract even in
        # a globally supersaturated matrix.
        shrink = (labels > 0) & (comp == 0) & (velocity < 0)
        da[grow] = -velocity[grow]*dt/p.dx
        da[shrink] = velocity[shrink]*dt/p.dx  # velocity<0 lowers resident alpha
        old_fp = fp.copy()
        alpha += da
        trial_fp = np.clip(precip_fraction(labels, alpha, comp), 0.0, 1.0)
        conservative_partition_transfer(ctot, old_fp, trial_fp, cm_interface, p)

        crossing = (grow | shrink) & (alpha <= 0.0)
        topology_deferred_cells = commit_topology_transactions(
            labels, alpha, comp, crossing)
        gaussian_remap_exchanges = 0
        if np.any(crossing):
            gaussian_remap_exchanges = conservative_gaussian_remap(
                labels, alpha, p.remap_sigma_cells)
            geometry_dirty = True
        t += dt; step += 1

        mass_err = (float(np.sum(ctot))-initial_mass)/initial_mass
        vf = float(np.mean(precip_fraction(labels, alpha, neighbor_competitor(labels)[1])))
        metrics.append((t, vf, float(np.mean(cm[labels == 0])), mass_err,
                        residual, float(np.max(np.abs(hydro))), int(np.sum(mask)),
                        dt, max_displacement, topology_deferred_cells,
                        gaussian_remap_exchanges))
        while next_snap < len(snapshot_times) and t >= snapshot_times[next_snap]-0.5*dt:
            save(next_snap); next_snap += 1

    with (case / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["time_s", "precipitate_fraction", "mean_matrix_cu", "relative_mass_error",
                    "mechanical_equilibrium_residual", "max_abs_hydrostatic_stress_Pa",
                    "interface_cells", "dt_s", "max_interface_displacement_cells",
                    "topology_deferred_cells", "gaussian_remap_exchanges"])
        w.writerows(metrics)
    with (case / "parameters.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["parameter", "value"])
        for key, val in vars(p).items(): w.writerow([key, val])
        w.writerow(["shape", "x".join(map(str, shape))]); w.writerow(["n_seeds", len(seeds)])
        w.writerow(["wall_time_s", time.perf_counter()-start])
    mask, comp = neighbor_competitor(labels)
    stats = particle_statistics(labels, alpha, comp, p)
    with (case / "particle_statistics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["variant", "fractional_voxel_count", "aspect_ratio",
                    "major_dimension_nm", "minor_dimension_nm"])
        w.writerows(stats)
    final_fields = variant_fraction_fields(labels, alpha, comp)
    final_counts = {
        v: periodic_components(labels == v)[1]
        for v in (1, 2, 3)
    }
    if abs(metrics[-1][3]) > 1e-12 or metrics[-1][4] > 1e-10:
        raise RuntimeError("Al-Cu material audit failed conservation or mechanical equilibrium")
    if max(row[8] for row in metrics) > p.interface_cfl*(1.0+1e-10):
        raise RuntimeError("Al-Cu material audit failed interface CFL")
    # The chemical lever rule is not a monotonicity constraint once coherent
    # elastic and Gibbs--Thomson terms enter the sharp-interface equilibrium.
    # Competitive ripening may lower total precipitate fraction while larger
    # plates grow.  Gate the actual invariants and retain every represented
    # orientation variant instead of enforcing an unphysical growth sign.
    initial_variants = {v for v, count in initial_component_counts.items() if count > 0}
    missing_variants = sorted(v for v in initial_variants if final_counts[v] == 0)
    if missing_variants:
        raise RuntimeError(f"Al-Cu run lost represented variants: {missing_variants}")
    # This material module contains no nucleation law, so a variant may not
    # acquire a new disconnected component.  Complete dissolution is a valid
    # sharp-interface extinction event, and coalescence of same-variant
    # precipitates is also physical; both can reduce the component count.
    if any(final_counts[v] > initial_component_counts[v] for v in (1, 2, 3)):
        raise RuntimeError(
            f"Al-Cu topology created an unauthorized component: "
            f"{initial_component_counts} -> {final_counts}")
    initial_labels = dict(np.load(case / "snap_000.npz"))["labels"]
    initial_geometry = np.asarray([(initial_labels == v).astype(float) for v in (1, 2, 3)])
    final_geometry = np.asarray([(labels == v).astype(float) for v in (1, 2, 3)])
    initial_topology = topology_metrics(initial_geometry)
    final_topology = topology_metrics(final_geometry)
    with (case / "topology_audit.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["state", "variant", "components", "enclosed_cavities",
                    "threshold_cells", "exposed_grid_faces",
                    "noncontiguous_habit_lines"])
        for state, rows in (("initial", initial_topology), ("final", final_topology)):
            for row in rows:
                w.writerow([state, *row])
    if any(row[2] != 0 for row in final_topology):
        raise RuntimeError(f"Al-Cu topology audit found enclosed cavities: {final_topology}")
    # Line convexity along a crystallographic axis is retained as a diagnostic,
    # not a validity gate: a physically admissible plate may contain a ledge or
    # neck without changing its component or cavity topology.
    print(name, "vf=", metrics[-1][1], "mass error=", metrics[-1][3],
          "equilibrium residual=", metrics[-1][4],
          "median aspect=", np.median([x[2] for x in stats]) if stats else np.nan,
          "wall_s=", time.perf_counter()-start)


def publication_cases():
    """Resolved, multiparticle Al--Cu cases used by the paper.

    Every 3-D state contains alpha-Al plus all three theta-prime orientation
    variants.  The 2-D sections likewise include the two edge-on variants and
    the broad-face section of the out-of-plane variant.
    """
    # Unequal, supercritical embryos trigger competitive growth without the
    # unphysical collective dissolution caused by seeding many subcritical
    # particles.  All crystallographic variants remain represented.
    layout_2d_hi = [
        ((48, 52), 1, 1.05), ((202, 48), 2, 0.95),
        ((54, 207), 2, 1.08), ((204, 202), 1, 0.92),
    ]
    layout_2d_lo = [
        ((36, 39), 1, 1.05), ((152, 36), 2, 0.95),
        ((41, 155), 2, 1.08), ((153, 152), 1, 0.92),
    ]
    layout_3d_80 = [
        ((20, 20, 20), 1, 1.05), ((60, 22, 56), 2, 0.95),
        ((24, 60, 58), 3, 1.00),
    ]
    layout_3d_64 = [
        ((16, 16, 16), 1, 1.05), ((48, 18, 45), 2, 0.95),
        ((19, 48, 46), 3, 1.00),
    ]
    layout_3d_96 = [
        ((24, 24, 24), 1, 1.05), ((72, 27, 68), 2, 0.95),
        ((29, 72, 69), 3, 1.00),
    ]
    layout_3d_128 = [
        ((32, 32, 32), 1, 1.05), ((96, 36, 91), 2, 0.95),
        ((39, 96, 92), 3, 1.00),
    ]
    return {
        "two_dimensional": dict(
            shape=(192, 192), n_seeds=4, radius=6.375, t_end=30000.0,
            snapshot_times=[0.0, 6000.0, 15000.0, 30000.0], seed=24051,
            dx=0.50e-9, seed_layout=layout_2d_lo, initial_shape="sphere",
        ),
        "two_dimensional_high_mesh": dict(
            shape=(256, 256), n_seeds=4, radius=8.5, t_end=30000.0,
            snapshot_times=[0.0, 6000.0, 15000.0, 30000.0], seed=24051,
            dx=0.375e-9, seed_layout=layout_2d_hi,
            gaussian_sigma_cells=8.0/3.0, initial_shape="sphere",
        ),
        "three_dimensional": dict(
            shape=(96, 96, 96), n_seeds=3, radius=6.0, t_end=10000.0,
            snapshot_times=[0.0, 2000.0, 5000.0, 10000.0], seed=24052,
            dx=(2.0/3.0)*1e-9, seed_layout=layout_3d_96,
            gaussian_sigma_cells=3.0, initial_shape="sphere",
        ),
        "three_dimensional_mesh_reference": dict(
            shape=(64, 64, 64), n_seeds=3, radius=6.4, t_end=1000.0,
            snapshot_times=[0.0, 250.0, 500.0, 1000.0], seed=24052,
            dx=1.00e-9, seed_layout=layout_3d_64,
            interface_cfl=0.35, mobility_anisotropy_beta=50.0,
            mobility_ledge_exponent=4.0, remap_sigma_cells=1.25,
            initial_shape="sphere",
        ),
        "three_dimensional_high_mesh": dict(
            shape=(96, 96, 96), n_seeds=3, radius=9.6, t_end=1000.0,
            snapshot_times=[0.0, 250.0, 500.0, 1000.0], seed=24052,
            dx=(2.0/3.0)*1e-9, seed_layout=layout_3d_96,
            gaussian_sigma_cells=3.0, interface_cfl=0.35,
            mobility_anisotropy_beta=50.0, mobility_ledge_exponent=4.0,
            remap_sigma_cells=1.25, initial_shape="sphere",
        ),
        "three_dimensional_mesh_fine": dict(
            shape=(96, 96, 96), n_seeds=3, radius=9.6, t_end=500.0,
            snapshot_times=[0.0, 125.0, 250.0, 500.0], seed=24052,
            dx=(2.0/3.0)*1e-9, seed_layout=layout_3d_96,
            gaussian_sigma_cells=3.0, interface_cfl=0.35,
            mobility_anisotropy_beta=50.0, mobility_ledge_exponent=4.0,
            remap_sigma_cells=1.25, initial_shape="sphere",
        ),
        "three_dimensional_mesh_ultrafine": dict(
            shape=(128, 128, 128), n_seeds=3, radius=12.8, t_end=500.0,
            snapshot_times=[0.0, 125.0, 250.0, 500.0], seed=24052,
            dx=0.50e-9, seed_layout=layout_3d_128,
            gaussian_sigma_cells=4.0, interface_cfl=0.35,
            mobility_anisotropy_beta=50.0, mobility_ledge_exponent=4.0,
            remap_sigma_cells=1.25, initial_shape="sphere",
        ),
        "three_dimensional_ultrafine": dict(
            shape=(128, 128, 128), n_seeds=3, radius=8.0, t_end=500.0,
            snapshot_times=[0.0, 100.0, 250.0, 500.0], seed=24052,
            dx=0.50e-9, seed_layout=layout_3d_128,
            gaussian_sigma_cells=4.0, interface_cfl=0.35,
            mobility_anisotropy_beta=50.0, mobility_ledge_exponent=4.0,
            remap_sigma_cells=1.25,
            initial_shape="sphere",
        ),
        "two_dimensional_sphere_pilot": dict(
            shape=(256, 256), n_seeds=4, radius=8.5, t_end=6000.0,
            snapshot_times=[0.0, 1000.0, 3000.0, 6000.0], seed=24051,
            dx=0.375e-9, seed_layout=layout_2d_hi,
            gaussian_sigma_cells=8.0/3.0, initial_shape="sphere",
        ),
        "three_dimensional_sphere_pilot": dict(
            shape=(80, 80, 80), n_seeds=3, radius=8.0, t_end=2000.0,
            snapshot_times=[0.0, 500.0, 1000.0, 2000.0], seed=24052,
            dx=0.80e-9, seed_layout=layout_3d_80,
            gaussian_sigma_cells=2.5, initial_shape="sphere",
        ),
    }


def _run_named_case(item):
    name, kwargs = item
    run_case(name=name, **kwargs)
    return name


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="*", default=None,
                        help="case names (default: all publication cases)")
    parser.add_argument("--case-workers", type=int, default=3,
                        help="independent cases to execute concurrently")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cases = publication_cases()
    selected = args.cases or list(cases)
    unknown = sorted(set(selected)-set(cases))
    if unknown:
        raise SystemExit(f"unknown cases: {', '.join(unknown)}")
    work = [(name, cases[name]) for name in selected]
    workers = max(1, min(args.case_workers, len(work), os.cpu_count() or 1))
    if workers == 1:
        for item in work:
            _run_named_case(item)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for name in pool.map(_run_named_case, work):
                print("completed", name, flush=True)
