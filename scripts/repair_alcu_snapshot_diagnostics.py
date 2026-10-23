"""Reconstruct bounded matrix-composition diagnostics in existing snapshots.

The conserved state is ``c_total``.  ``c_matrix`` is derived for plotting and
is undefined where the matrix fraction vanishes; old snapshots divided by a
small floor in those cells and therefore contained meaningless outliers.
"""
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "output" / "alcu_validation" / "precipitation"
C_THETA = 1.0/3.0


def main():
    repaired = 0
    for path in DATA.glob("*/snap_*.npz"):
        with np.load(path) as archive:
            fields = {key: archive[key] for key in archive.files}
        if "c_total" not in fields or "precip_fraction" not in fields:
            continue
        matrix = 1.0-fields["precip_fraction"]
        defined = matrix > 1e-7
        cm = np.full(matrix.shape, np.nan, dtype=float)
        cm[defined] = ((fields["c_total"][defined]
                        - fields["precip_fraction"][defined]*C_THETA)
                       / matrix[defined])
        cm[defined] = np.clip(cm[defined], 0.0, C_THETA)
        fields["c_matrix"] = cm
        np.savez_compressed(path, **fields)
        repaired += 1
    print(f"repaired {repaired} snapshot diagnostics")


if __name__ == "__main__":
    main()
