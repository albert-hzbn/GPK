"""Locate the simulation output used by the figure scripts.

The figure scripts look for every result folder in these roots, in order:

1. ``$GPK_DATA``, if set;
2. ``output/`` and ``build/output/`` in the repository, where your own runs land
   (see ``paper/run_simulations.py`` and ``ctest``);
3. ``paper/data/``, the processed data shipped with the repository.

A folder in ``paper/data`` that contains ``LARGE_FILES_OMITTED.txt`` lacks the
large snapshot fields (see ``paper/LARGE_DATA.md``), so a regenerated copy in
one of the other roots is preferred when it exists.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BUNDLED = REPO / "paper" / "data"
OMITTED = "LARGE_FILES_OMITTED.txt"


def data_roots() -> list[Path]:
    roots = [Path(os.environ["GPK_DATA"])] if os.environ.get("GPK_DATA") else []
    return roots + [REPO / "output", REPO / "build" / "output", BUNDLED]


def output_dir(name: str) -> Path:
    """Return the folder ``name`` from the first root that holds a complete copy."""
    fallback = None
    for root in data_roots():
        candidate = root / name
        if candidate.is_dir():
            if not (candidate / OMITTED).exists() and not any(candidate.rglob(OMITTED)):
                return candidate
            fallback = fallback or candidate
    return fallback or (REPO / "output" / name)
