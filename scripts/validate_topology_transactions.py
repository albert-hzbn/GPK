"""Cross-check the Python Al--Cu TCKT against canonical 2-D/3-D cases."""
from __future__ import annotations

import numpy as np

from run_alcu_gpk_microstructure import topology_simple_relabel


def check(name: str, labels: np.ndarray, index: tuple[int, ...],
          new_label: int, expected: bool) -> bool:
    old_label = int(labels[index])
    counts = {int(v): int(np.count_nonzero(labels == v)) for v in np.unique(labels)}
    observed = topology_simple_relabel(labels, index, old_label, new_label, counts)
    passed = observed == expected
    print(f"{name:28s} observed={'accept' if observed else 'defer':6s} "
          f"expected={'accept' if expected else 'defer':6s} "
          f"{'PASS' if passed else 'FAIL'}")
    return passed


def main() -> None:
    cases = []
    smooth = np.zeros((9, 9), np.int8); smooth[:, :5] = 1
    cases.append(("2D smooth boundary", smooth, (4, 4), 0, True))

    bridge = np.zeros((11, 11), np.int8)
    bridge[3:8, 2:5] = 1; bridge[3:8, 6:9] = 1; bridge[5, 5] = 1
    cases.append(("2D component split", bridge, (5, 5), 0, False))

    merge = np.zeros((9, 9), np.int8); merge[4, 3] = merge[4, 5] = 1
    cases.append(("2D premature merger", merge, (4, 4), 1, False))

    extinct = np.zeros((9, 9), np.int8); extinct[4, 4] = 1
    cases.append(("2D terminal extinction", extinct, (4, 4), 0, True))

    bridge3 = np.zeros((9, 9, 9), np.int8)
    bridge3[3:6, 3:6, 1:4] = 2; bridge3[3:6, 3:6, 5:8] = 2
    bridge3[4, 4, 4] = 2
    cases.append(("3D component split", bridge3, (4, 4, 4), 0, False))

    extinct3 = np.zeros((7, 7, 7), np.int8); extinct3[3, 3, 3] = 3
    cases.append(("3D terminal extinction", extinct3, (3, 3, 3), 0, True))

    if not all(check(*case) for case in cases):
        raise SystemExit("Topology transaction validation failed")
    print("Python topology transaction suite: PASS")


if __name__ == "__main__":
    main()
