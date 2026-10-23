#!/usr/bin/env python3
"""CUDA benchmark for 3-D phase-field and Gaussian-projected kinetics.

This is a deliberately self-contained research benchmark.  Both methods run
on the same GPU, use periodic boundaries, start from identical labels, and are
timed with CUDA events after warm-up.  GPK stores one sharp label and one
sub-cell fraction per voxel.  PF stores Q diffuse order parameters and evolves
the standard Fan--Chen multi-order-parameter Allen--Cahn equation.

The GPK kernel performs local Gaussian projection over candidate labels from
the six-neighbour stencil.  Its fractional state integrates the projected
curvature drive before a label transaction.  This is the GPU analogue of the
geometry/projection/transaction path in the C++ implementation; it is not a
wrapper around the CPU solver.

Outputs: benchmark.csv, results.json, benchmark.png, and report.html.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

# Keep NVRTC artifacts inside the workspace; this also works in restricted
# runners where the user-profile cache is read-only.
os.environ.setdefault("CUPY_CACHE_DIR", str(Path(__file__).parents[1] / "output" / ".cupy_cache"))

try:
    import cupy as cp
except ImportError as exc:  # pragma: no cover - exercised by CLI error path
    raise SystemExit(
        "CuPy is required. Install cupy-cuda13x (or the package matching your CUDA runtime)."
    ) from exc


PF_STEP = cp.RawKernel(r"""
extern "C" __global__
void pf_step(const float* eta, float* next, int n, int qn, int N,
             float dt, float kappa, float gamma) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int total = n * qn;
    if (tid >= total) return;
    int q = tid / n, i = tid - q * n;
    int x = i % N, y = (i / N) % N, z = i / (N * N);
    int xm = x ? x - 1 : N - 1, xp = x + 1 == N ? 0 : x + 1;
    int ym = y ? y - 1 : N - 1, yp = y + 1 == N ? 0 : y + 1;
    int zm = z ? z - 1 : N - 1, zp = z + 1 == N ? 0 : z + 1;
    int yz = z * N * N, ymz = z * N * N + ym * N;
    int ypz = z * N * N + yp * N;
    int base = q * n;
    float c = eta[base + i];
    float lap = eta[base + yz + y * N + xm] + eta[base + yz + y * N + xp]
              + eta[base + ymz + x] + eta[base + ypz + x]
              + eta[base + zm * N * N + y * N + x]
              + eta[base + zp * N * N + y * N + x] - 6.0f * c;
    float ss = 0.0f;
    for (int p = 0; p < qn; ++p) {
        float v = eta[p * n + i]; ss += v * v;
    }
    float df = -c + c*c*c + 2.0f * gamma * c * (ss - c*c);
    next[base + i] = c - dt * (df - kappa * lap);
}
""", "pf_step")


GPK_STEP = cp.RawKernel(r"""
extern "C" __global__
void gpk_step(const int* labels, const float* alpha, int* out_labels,
              float* out_alpha, int n, int N, float rate) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    int x = i % N, y = (i / N) % N, z = i / (N * N);
    int cand[7];
    cand[0] = labels[i];
    int xn[6] = {x ? x-1 : N-1, x+1==N ? 0 : x+1, x, x, x, x};
    int yn[6] = {y, y, y ? y-1 : N-1, y+1==N ? 0 : y+1, y, y};
    int zn[6] = {z, z, z, z, z ? z-1 : N-1, z+1==N ? 0 : z+1};
    int nc = 1;
    for (int k = 0; k < 6; ++k) {
        int v = labels[zn[k]*N*N + yn[k]*N + xn[k]], seen = 0;
        for (int j = 0; j < nc; ++j) seen |= cand[j] == v;
        if (!seen) cand[nc++] = v;
    }
    if (nc == 1) { out_labels[i] = cand[0]; out_alpha[i] = 1.0f; return; }

    // Tensor-product [1,2,1]^3 Gaussian, normalized by 64.
    float score[7] = {0,0,0,0,0,0,0};
    for (int dz = -1; dz <= 1; ++dz) {
        int zz = (z + dz + N) % N, wz = dz == 0 ? 2 : 1;
        for (int dy = -1; dy <= 1; ++dy) {
            int yy = (y + dy + N) % N, wy = dy == 0 ? 2 : 1;
            for (int dx = -1; dx <= 1; ++dx) {
                int xx = (x + dx + N) % N, wx = dx == 0 ? 2 : 1;
                int v = labels[zz*N*N + yy*N + xx];
                float w = (float)(wx * wy * wz) * (1.0f/64.0f);
                for (int j = 0; j < nc; ++j) if (cand[j] == v) score[j] += w;
            }
        }
    }
    int best = 1;
    for (int j = 2; j < nc; ++j)
        if (score[j] > score[best] || (score[j] == score[best] && cand[j] < cand[best])) best = j;
    float drive = score[best] - score[0];
    float a = alpha[i] - rate * fmaxf(drive, 0.0f);
    int owner = cand[0];
    if (a <= 0.0f && drive > 0.0f) { owner = cand[best]; a = 1.0f; }
    out_labels[i] = owner; out_alpha[i] = fminf(fmaxf(a, 0.0f), 1.0f);
}
""", "gpk_step")


@dataclass
class Record:
    case: str
    method: str
    N: int
    phases: int
    steps: int
    milliseconds: float
    timing_std_ms: float
    voxel_steps_per_second: float
    allocated_mib: float
    interface_fraction: float
    agreement_with_initial: float


def periodic_voronoi(N: int, phases: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, N, size=(phases, 3), endpoint=False)
    z, y, x = np.indices((N, N, N), sparse=True)
    best = np.full((N, N, N), np.iinfo(np.int32).max, dtype=np.int32)
    labels = np.zeros((N, N, N), dtype=np.int32)
    for q, (sz, sy, sx) in enumerate(seeds):
        dz = np.minimum(abs(z - sz), N - abs(z - sz))
        dy = np.minimum(abs(y - sy), N - abs(y - sy))
        dx = np.minimum(abs(x - sx), N - abs(x - sx))
        d2 = (dz * dz + dy * dy + dx * dx).astype(np.int32)
        take = d2 < best
        labels[take] = q
        best[take] = d2[take]
    return labels


def spherical_inclusions(N: int, phases: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    z, y, x = np.indices((N, N, N))
    labels = np.zeros((N, N, N), dtype=np.int32)
    radius = max(2, N // 9)
    for q in range(1, phases):
        cz, cy, cx = rng.integers(radius, N - radius, size=3)
        r = radius * (0.75 + 0.5 * rng.random())
        labels[(z-cz)**2 + (y-cy)**2 + (x-cx)**2 <= r*r] = q
    return labels


def lamellar(N: int, phases: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    z, y, x = np.indices((N, N, N))
    wave = x + 0.17 * N * np.sin(2*np.pi*y/N) + 0.11 * N * np.sin(2*np.pi*z/N)
    wave += rng.normal(0.0, 0.025 * N, size=(N, N, N))
    return (np.floor(wave * phases / N).astype(np.int32) % phases)


GENERATORS = {
    "voronoi": periodic_voronoi,
    "inclusions": spherical_inclusions,
    "lamellar": lamellar,
}


def interface_fraction(labels: cp.ndarray) -> float:
    inter = cp.zeros(labels.shape, dtype=cp.bool_)
    for axis in range(3):
        inter |= labels != cp.roll(labels, 1, axis=axis)
        inter |= labels != cp.roll(labels, -1, axis=axis)
    return float(cp.mean(inter).get())


def init_pf(labels: np.ndarray, phases: int) -> cp.ndarray:
    lab = cp.asarray(labels.ravel())
    return (cp.arange(phases, dtype=cp.int32)[:, None] == lab[None, :]).astype(cp.float32)


def labels_from_pf(eta: cp.ndarray, N: int) -> cp.ndarray:
    return cp.argmax(eta, axis=0).astype(cp.int32).reshape(N, N, N)


def run_pf(labels: np.ndarray, phases: int, steps: int, timed: bool = True):
    N = labels.shape[0]; n = labels.size
    eta = init_pf(labels, phases); nxt = cp.empty_like(eta)
    block = 256; grid = ((n * phases + block - 1) // block,)
    args0 = (eta, nxt, n, phases, N, np.float32(0.035), np.float32(1.0), np.float32(1.5))
    PF_STEP(grid, (block,), args0); eta, nxt = nxt, eta
    cp.cuda.Stream.null.synchronize()
    start, end = cp.cuda.Event(), cp.cuda.Event(); start.record()
    for _ in range(steps):
        PF_STEP(grid, (block,), (eta, nxt, n, phases, N, np.float32(0.035),
                                 np.float32(1.0), np.float32(1.5)))
        eta, nxt = nxt, eta
    end.record(); end.synchronize()
    ms = float(cp.cuda.get_elapsed_time(start, end)) if timed else 0.0
    out = labels_from_pf(eta, N)
    return out, ms, int(eta.nbytes + nxt.nbytes)


def run_gpk(labels: np.ndarray, steps: int, timed: bool = True):
    N = labels.shape[0]; n = labels.size
    lab = cp.asarray(labels.ravel(), dtype=cp.int32); out = cp.empty_like(lab)
    alpha = cp.ones(n, dtype=cp.float32); alpha_out = cp.empty_like(alpha)
    block = 256; grid = ((n + block - 1) // block,)
    GPK_STEP(grid, (block,), (lab, alpha, out, alpha_out, n, N, np.float32(3.0)))
    lab, out, alpha, alpha_out = out, lab, alpha_out, alpha
    cp.cuda.Stream.null.synchronize()
    start, end = cp.cuda.Event(), cp.cuda.Event(); start.record()
    for _ in range(steps):
        GPK_STEP(grid, (block,), (lab, alpha, out, alpha_out, n, N, np.float32(3.0)))
        lab, out, alpha, alpha_out = out, lab, alpha_out, alpha
    end.record(); end.synchronize()
    ms = float(cp.cuda.get_elapsed_time(start, end)) if timed else 0.0
    return lab.reshape(N, N, N), ms, int(lab.nbytes + out.nbytes + alpha.nbytes + alpha_out.nbytes)


def boundary_f1(a: cp.ndarray, b: cp.ndarray, tolerance: int = 1) -> float:
    def boundary(x):
        out = cp.zeros(x.shape, dtype=cp.bool_)
        for ax in range(3):
            out |= x != cp.roll(x, 1, axis=ax)
            out |= x != cp.roll(x, -1, axis=ax)
        return out
    ba, bb = boundary(a), boundary(b)
    da, db = ba.copy(), bb.copy()
    for _ in range(tolerance):
        for ax in range(3):
            da |= cp.roll(da, 1, ax) | cp.roll(da, -1, ax)
            db |= cp.roll(db, 1, ax) | cp.roll(db, -1, ax)
    precision = cp.sum(ba & db) / cp.maximum(cp.sum(ba), 1)
    recall = cp.sum(bb & da) / cp.maximum(cp.sum(bb), 1)
    return float((2*precision*recall/cp.maximum(precision+recall, 1e-12)).get())


def benchmark_case(case: str, N: int, phases: int, steps: int, seed: int,
                   repeats: int = 5):
    initial = GENERATORS[case](N, phases, seed)
    initial_gpu = cp.asarray(initial)
    g_runs = [run_gpk(initial, steps) for _ in range(repeats)]
    p_runs = [run_pf(initial, phases, steps) for _ in range(repeats)]
    gl, _, gb = g_runs[0]; pl, _, pb = p_runs[0]
    gt = np.asarray([r[1] for r in g_runs]); pt = np.asarray([r[1] for r in p_runs])
    gms, pms = float(np.median(gt)), float(np.median(pt))
    records = []
    for method, lab, ms, spread, mem in (("GPK", gl, gms, float(np.std(gt)), gb),
                                         ("PF", pl, pms, float(np.std(pt)), pb)):
        records.append(Record(
            case, method, N, phases, steps, ms, spread,
            N**3 * steps / (ms * 1e-3), mem / 2**20,
            interface_fraction(lab), float(cp.mean(lab == initial_gpu).get())))
    metrics = {
        "case": case, "N": N, "phases": phases,
        "label_agreement": float(cp.mean(gl == pl).get()),
        "boundary_f1_1voxel": boundary_f1(gl, pl, 1),
        "gpk_speedup_same_grid": pms / gms,
        "pf_over_gpk_memory": pb / gb,
    }
    return records, metrics


def cpu_gpk_step(labels: np.ndarray, alpha: np.ndarray, rate: float = 3.0):
    N = labels.shape[0]; ol = labels.copy(); oa = alpha.copy()
    weights = (1, 2, 1)
    for z in range(N):
        for y in range(N):
            for x in range(N):
                candidates = [int(labels[z, y, x])]
                for zz, yy, xx in ((z,y,x-1),(z,y,x+1),(z,y-1,x),(z,y+1,x),(z-1,y,x),(z+1,y,x)):
                    v = int(labels[zz%N, yy%N, xx%N])
                    if v not in candidates: candidates.append(v)
                if len(candidates) == 1:
                    oa[z,y,x] = 1.0; continue
                score = [0.0] * len(candidates)
                for dz in (-1,0,1):
                    for dy in (-1,0,1):
                        for dx in (-1,0,1):
                            v = int(labels[(z+dz)%N,(y+dy)%N,(x+dx)%N])
                            if v in candidates:
                                score[candidates.index(v)] += weights[dz+1]*weights[dy+1]*weights[dx+1]/64
                best = max(range(1,len(candidates)), key=lambda j:(score[j],-candidates[j]))
                drive = score[best]-score[0]
                a = float(alpha[z,y,x])-rate*max(drive,0)
                if a <= 0 and drive > 0: ol[z,y,x], a = candidates[best], 1.0
                oa[z,y,x] = min(max(a,0),1)
    return ol, oa


def self_test() -> dict:
    # Exact transaction parity on a deliberately irregular, small 3-D field.
    labels = periodic_voronoi(8, 5, 991)
    alpha = np.ones_like(labels, dtype=np.float32)
    cl, ca = cpu_gpk_step(labels, alpha)
    n = labels.size; N = 8
    dl = cp.asarray(labels.ravel()); da = cp.asarray(alpha.ravel())
    ol, oa = cp.empty_like(dl), cp.empty_like(da)
    GPK_STEP(((n+255)//256,), (256,), (dl, da, ol, oa, n, N, np.float32(3.0)))
    cp.cuda.Stream.null.synchronize()
    gpk_labels_ok = np.array_equal(cl.ravel(), cp.asnumpy(ol))
    gpk_alpha_error = float(np.max(np.abs(ca.ravel()-cp.asnumpy(oa))))

    # PF parity against NumPy for one explicit Euler step.
    phases = 5; eta = np.eye(phases, dtype=np.float32)[labels.ravel()].T.copy()
    ss = np.sum(eta*eta, axis=0)
    ref = np.empty_like(eta)
    for q in range(phases):
        e = eta[q].reshape((N,N,N))
        lap = sum(np.roll(e,s,a) for a in range(3) for s in (-1,1))-6*e
        c = eta[q]
        df = -c+c**3+3.0*c*(ss-c*c)
        ref[q] = c-0.035*(df-lap.ravel())
    de, dn = cp.asarray(eta), cp.empty_like(cp.asarray(eta))
    PF_STEP(((n*phases+255)//256,), (256,),
            (de,dn,n,phases,N,np.float32(.035),np.float32(1),np.float32(1.5)))
    cp.cuda.Stream.null.synchronize()
    pf_error = float(np.max(np.abs(ref-cp.asnumpy(dn))))
    result = {"gpk_label_parity": gpk_labels_ok,
              "gpk_alpha_max_abs_error": gpk_alpha_error,
              "pf_max_abs_error": pf_error,
              "passed": gpk_labels_ok and gpk_alpha_error < 1e-6 and pf_error < 2e-6}
    if not result["passed"]:
        raise AssertionError(f"CPU/GPU parity failed: {result}")
    return result


def write_outputs(out: Path, records: list[Record], metrics: list[dict], tests: dict,
                  metadata: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with (out/"benchmark.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(records[0])))
        w.writeheader(); w.writerows(asdict(r) for r in records)
    payload = {"metadata": metadata, "self_test": tests,
               "records": [asdict(r) for r in records], "comparisons": metrics}
    (out/"results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cases = list(dict.fromkeys(r.case for r in records))
    sizes = sorted(set(r.N for r in records))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    colors = {"GPK":"#2563eb", "PF":"#f97316"}
    styles = {case: style for case, style in zip(cases, ("-", "--", ":", "-."))}
    markers = {case: marker for case, marker in zip(cases, ("o", "s", "^", "D"))}
    for method in ("GPK","PF"):
        for case in cases:
            rr=[r for r in records if r.method==method and r.case==case]
            axes[0].plot([r.N for r in rr],[r.voxel_steps_per_second/1e6 for r in rr],
                         marker=markers[case], linestyle=styles[case],
                         color=colors[method], alpha=.72,
                         label=f"{method} {case}")
    axes[0].set(xlabel="linear grid N",ylabel="M voxel-steps/s",title="GPU throughput")
    axes[0].grid(alpha=.25); axes[0].legend(fontsize=7,ncol=2)
    largest=max(sizes); rr=[r for r in records if r.N==largest]
    x=np.arange(len(cases)); width=.36
    gm=[next(r for r in rr if r.case==c and r.method=="GPK").allocated_mib for c in cases]
    pm=[next(r for r in rr if r.case==c and r.method=="PF").allocated_mib for c in cases]
    axes[1].bar(x-width/2,gm,width,label="GPK",color=colors["GPK"])
    axes[1].bar(x+width/2,pm,width,label="PF",color=colors["PF"])
    axes[1].set_xticks(x,cases,rotation=15); axes[1].set(ylabel="MiB",title=f"State/work memory, {largest}³")
    axes[1].legend(); axes[1].grid(axis="y",alpha=.25)
    for case in cases:
        mm=[m for m in metrics if m["case"]==case]
        axes[2].plot([m["N"] for m in mm],[m["gpk_speedup_same_grid"] for m in mm],
                     marker=markers[case],linestyle=styles[case],label=case)
    axes[2].axhline(1,color="#64748b",lw=1)
    axes[2].set(xlabel="linear grid N",ylabel="PF time / GPK time",
                title="Same-grid GPK speedup")
    axes[2].legend(fontsize=8); axes[2].grid(alpha=.25)
    fig.suptitle("3-D CUDA PF vs Gaussian-Projected Kinetics")
    fig.tight_layout(); fig.savefig(out/"benchmark.png",dpi=180); plt.close(fig)

    rows="\n".join(
        "<tr>"+"".join(f"<td>{html.escape(str(v))}</td>" for v in (
            r.case,r.method,f"{r.N}³",r.phases,r.steps,f"{r.milliseconds:.3f}",
            f"{r.timing_std_ms:.3f}",
            f"{r.voxel_steps_per_second/1e6:.2f}",f"{r.allocated_mib:.2f}",
            f"{r.interface_fraction:.3f}"))+"</tr>" for r in records)
    cmp_rows="\n".join(
        f"<tr><td>{m['case']}</td><td>{m['N']}³</td><td>{m['label_agreement']:.3f}</td>"
        f"<td>{m['boundary_f1_1voxel']:.3f}</td><td>{m['gpk_speedup_same_grid']:.2f}×</td>"
        f"<td>{m['pf_over_gpk_memory']:.1f}×</td></tr>" for m in metrics)
    report=f"""<!doctype html><meta charset='utf-8'><title>3-D GPU PF/GPK benchmark</title>
<style>body{{font:15px system-ui;max-width:1180px;margin:36px auto;color:#182230}}h1,h2{{color:#0f3f73}}
table{{border-collapse:collapse;width:100%;margin:14px 0 28px}}th,td{{padding:7px 9px;border:1px solid #ccd5df;text-align:right}}th{{background:#eaf1f8}}td:first-child,th:first-child{{text-align:left}}
.ok{{padding:12px;background:#dcfce7;border-left:5px solid #16a34a}}.note{{background:#fff7ed;padding:12px;border-left:5px solid #f97316}}img{{max-width:100%}}</style>
<h1>3-D GPU Phase Field vs Gaussian-Projected Kinetics</h1>
<p><b>Generated:</b> {metadata['generated_at']} &nbsp; <b>GPU:</b> {html.escape(metadata['gpu'])} &nbsp; <b>CuPy:</b> {metadata['cupy']}</p>
<p class='ok'><b>Verification passed.</b> GPU kernels agree with independent CPU one-step references. PF max error {tests['pf_max_abs_error']:.2e}; GPK alpha max error {tests['gpk_alpha_max_abs_error']:.2e}.</p>
<h2>Method and scope</h2><p>Both solvers use periodic cubic domains and identical seeded labels. PF is the explicit Fan–Chen multi-order-parameter Allen–Cahn model (Q dense float fields, 7-point Laplacian). GPU GPK keeps one integer label and one fractional interface state per voxel, evaluates a separable 3×3×3 Gaussian projection for local candidate phases, and performs deterministic fractional label transactions.</p>
<p class='note'><b>Interpretation:</b> these are same-grid kernel measurements, not a claim of physical-time speedup. PF normally needs additional voxels across each diffuse interface and smaller timesteps; resolution-matched physical speedup must be reported separately. Boundary F1 measures short-horizon morphological agreement and is not full model validation.</p>
<img src='benchmark.png' alt='benchmark plots'>
<h2>Raw timings</h2><table><tr><th>Case</th><th>Method</th><th>Grid</th><th>Phases</th><th>Steps</th><th>Median GPU ms</th><th>Std. dev. ms</th><th>M voxel-steps/s</th><th>Allocated MiB</th><th>Interface fraction</th></tr>{rows}</table>
<h2>Direct comparisons</h2><table><tr><th>Case</th><th>Grid</th><th>Label agreement</th><th>Boundary F1 (±1 voxel)</th><th>PF/GPK time</th><th>PF/GPK memory</th></tr>{cmp_rows}</table>
<h2>Reproduction</h2><pre>{html.escape(metadata['command'])}</pre><p>Machine-readable data: <a href='benchmark.csv'>benchmark.csv</a> and <a href='results.json'>results.json</a>.</p>"""
    (out/"report.html").write_text(report,encoding="utf-8")


def parse_args(argv: Iterable[str] | None = None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sizes",nargs="+",type=int,default=[24,32,48])
    p.add_argument("--cases",nargs="+",choices=GENERATORS,default=list(GENERATORS))
    p.add_argument("--phases",type=int,default=8)
    p.add_argument("--steps",type=int,default=12)
    p.add_argument("--repeats",type=int,default=5)
    p.add_argument("--seed",type=int,default=20260924)
    p.add_argument("--output",type=Path,default=Path("output/gpu_pf_gpk_3d"))
    p.add_argument("--self-test-only",action="store_true")
    return p.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args=parse_args(argv)
    device=cp.cuda.Device(); props=cp.cuda.runtime.getDeviceProperties(device.id)
    gpu=props["name"].decode() if isinstance(props["name"],bytes) else str(props["name"])
    print(f"GPU: {gpu}; CuPy {cp.__version__}; CUDA runtime {cp.cuda.runtime.runtimeGetVersion()}")
    tests=self_test(); print(f"CPU/GPU parity: PASS {tests}")
    if args.self_test_only: return 0
    if any(n < 8 for n in args.sizes) or args.phases < 2 or args.steps < 1 or args.repeats < 1:
        raise SystemExit("sizes must be >=8, phases >=2, steps >=1, and repeats >=1")
    records=[]; metrics=[]
    for N in args.sizes:
        for ci,case in enumerate(args.cases):
            print(f"Benchmarking {case:10s} {N}^3 Q={args.phases} ...",flush=True)
            rr,mm=benchmark_case(case,N,args.phases,args.steps,args.seed+ci,args.repeats)
            records.extend(rr); metrics.append(mm)
            print(f"  GPK {rr[0].milliseconds:.3f} ms | PF {rr[1].milliseconds:.3f} ms | boundary F1 {mm['boundary_f1_1voxel']:.3f}")
    metadata={"generated_at":time.strftime("%Y-%m-%d %H:%M:%S %z"),"gpu":gpu,
              "cupy":cp.__version__,"cuda_runtime":cp.cuda.runtime.runtimeGetVersion(),
              "python":platform.python_version(),"platform":platform.platform(),
              "command":" ".join([Path(sys.executable).name,Path(__file__).name,*sys.argv[1:]])}
    write_outputs(args.output,records,metrics,tests,metadata)
    print(f"Report: {(args.output/'report.html').resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
