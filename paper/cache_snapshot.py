"""Convert large GPK CSV snapshots to compact .npz caches for rendering.

Usage: python cache_snapshot.py snap_0000.csv [snap_0012.csv ...]

Each cache stores the header metadata, the grain labels (int32) and the
orientation column (float32) exactly as written in the CSV.  The values are
not resampled or filtered; render_3d_microstructures.read_snapshot loads the
cache instead of the CSV when it exists.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def read_meta(path: Path) -> tuple[dict, int]:
    meta: dict = {}
    n_header = 0
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.startswith("#"):
                break
            n_header += 1
            for token in line[1:].split():
                if "=" in token:
                    key, value = token.split("=", 1)
                    try:
                        meta[key] = float(value)
                    except ValueError:
                        pass
    return meta, n_header


def cache(path: Path) -> Path:
    meta, n_header = read_meta(path)
    n = int(meta["Nz"]) * int(meta["Ny"]) * int(meta["Nx"])
    labels = np.empty(n, dtype=np.int32)
    orientation = np.empty(n, dtype=np.float32)
    pos = 0
    reader = pd.read_csv(path, sep=r"\s+", header=None, skiprows=n_header,
                         usecols=[3, 7], comment="#", chunksize=8_000_000,
                         engine="c")
    for chunk in reader:
        k = len(chunk)
        labels[pos:pos + k] = chunk[3].to_numpy(np.int32)
        orientation[pos:pos + k] = chunk[7].to_numpy(np.float32)
        pos += k
    if pos != n:
        raise RuntimeError(f"{path}: expected {n} rows, read {pos}")
    out = path.with_suffix(".npz")
    np.savez(out, labels=labels, orientation=orientation, meta=json.dumps(meta))
    return out


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        print(cache(Path(arg)))
