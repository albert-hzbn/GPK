"""Re-run the simulations behind the manuscript figures.

Every run writes to ``output/`` in the repository root, where the figure
scripts find it before the bundled data in ``paper/data`` (see gpk_data.py).
The verification data (Figures 8, 11 and 12 and Supplementary Figures S6 and
S7) comes from the test executables instead; run ``ctest`` in the build
directory for those.

Examples::

    python paper/run_simulations.py --list
    python paper/run_simulations.py --only gallery2d
    python paper/run_simulations.py --only grain_rex --threads 8
    python paper/run_simulations.py --dry-run          # print the commands only

Run times range from minutes (2-D cases) to many hours (the 512^3 dendrite).
"""
from __future__ import annotations

import argparse
import configparser
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
INPUTS = REPO / "paper" / "inputs"
SCRIPTS = REPO / "scripts"

# group -> ordered steps. A step is ("gpk", deck), ("restart", coarse_dir, restart_dir, fine_deck)
# or ("python", script, *args). Decks write to the [output] dir they name.
GROUPS: dict[str, list[tuple]] = {
    "gallery2d": [  # Figure 9 (two-dimensional gallery)
        ("gpk", "large_domain_isotropic_2d.ini"),
        ("gpk", "large_domain_anisotropic_2d.ini"),
        ("gpk", "large_domain_recrystallisation_2d.ini"),
        ("gpk", "large_domain_multiphase_2d.ini"),
        ("gpk", "large_domain_elastic_2d.ini"),
        ("gpk", "fig7_dendrite_2d.ini"),
    ],
    "gallery3d": [  # Figure 10 (three-dimensional gallery)
        ("gpk", "smallgrain_coarse_isotropic_3d.ini"),
        ("restart", "output/manuscript_smallgrain_connected/coarse_isotropic_3d",
         "output/manuscript_smallgrain_connected/restart_isotropic", "smallgrain_fine_isotropic_3d.ini"),
        ("gpk", "smallgrain_fine_isotropic_3d.ini"),
        ("gpk", "smallgrain_coarse_anisotropic_3d.ini"),
        ("restart", "output/manuscript_smallgrain_connected/coarse_anisotropic_3d",
         "output/manuscript_smallgrain_connected/restart_anisotropic", "smallgrain_fine_anisotropic_3d.ini"),
        ("gpk", "smallgrain_fine_anisotropic_3d.ini"),
        ("gpk", "fig8_recrystallisation_3d.ini"),
        ("gpk", "fig8_multiphase_3d.ini"),
        ("gpk", "fig8_dendrite_3d_512.ini"),
    ],
    "grain_rex": [  # Supplementary Figures S4 and S5
        ("gpk", "validation_grain_growth_dense_2d.ini"),
        ("gpk", "validation_rex_current_2d.ini"),
        ("gpk", "validation_rex_current_3d.ini"),
    ],
    "alcu": [  # Figure 13 and Supplementary Figures S8-S10
        ("python", "generate_alcu_calphad.py"),
        ("gpk", "al4cu_recrystallisation_2d.ini"),
        ("python", "run_alcu_gpk_microstructure.py"),
        ("python", "run_alcu_realistic_population.py"),
    ],
}


def gpk_executable(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    for name in ("gpk.exe", "gpk"):
        candidate = REPO / "build" / name
        if candidate.exists():
            return candidate
    sys.exit("gpk executable not found: build the project first (see README) or pass --gpk")


def fine_shape(deck: str) -> list[str]:
    ini = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
    ini.read(INPUTS / deck)
    return [ini["grid"][k].strip() for k in ("Nz", "Ny", "Nx")]


def command(step: tuple, gpk: Path) -> list[str]:
    kind = step[0]
    if kind == "gpk":
        return [str(gpk), str(INPUTS / step[1])]
    if kind == "restart":
        _, coarse, restart, deck = step
        snaps = sorted((REPO / coarse).glob("snap_*.csv"))
        last = str(snaps[-1]) if snaps else str(REPO / coarse / "snap_<last>.csv")
        return [sys.executable, str(SCRIPTS / "create_multiresolution_restart.py"), last,
                str(REPO / restart), "--shape", *fine_shape(deck)]
    if kind == "python":
        return [sys.executable, str(SCRIPTS / step[1]), *step[2:]]
    raise ValueError(kind)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="*", choices=sorted(GROUPS), help="groups to run (default: all)")
    parser.add_argument("--list", action="store_true", help="list the groups and their steps")
    parser.add_argument("--dry-run", action="store_true", help="print the commands without running them")
    parser.add_argument("--gpk", help="path to the gpk executable (default: build/gpk)")
    parser.add_argument("--threads", type=int, help="OpenMP threads (sets OMP_NUM_THREADS)")
    args = parser.parse_args()

    if args.list:
        for name, steps in GROUPS.items():
            print(name)
            for step in steps:
                print("   ", " ".join(str(s) for s in step))
        return
    gpk = Path(args.gpk) if args.gpk else (Path("build/gpk") if args.dry_run else gpk_executable(None))
    env = dict(os.environ)
    if args.threads:
        env["OMP_NUM_THREADS"] = str(args.threads)
    for name in args.only or list(GROUPS):
        for step in GROUPS[name]:
            cmd = command(step, gpk)
            print(f"[{name}]", " ".join(cmd), flush=True)
            if not args.dry_run:
                subprocess.run(cmd, cwd=REPO, env=env, check=True)


if __name__ == "__main__":
    main()
