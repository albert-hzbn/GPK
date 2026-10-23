"""
plot_ppm.py
===========
Reads and plots .ppm image files (P6 binary or P3 ASCII).

Usage:
    python3 plot_ppm.py                         # auto-discover all PPMs in build/
    python3 plot_ppm.py file1.ppm file2.ppm     # specific files
    python3 plot_ppm.py --pattern "solid_*"     # glob pattern inside build/
"""

import sys
import struct
import argparse
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

BUILD = Path(__file__).parent.parent / "build"


# ── PPM parser ─────────────────────────────────────────────────────────────

def _skip_comments(data: bytes, pos: int) -> int:
    """Advance past whitespace and # comment lines."""
    while pos < len(data):
        if data[pos:pos+1] in (b' ', b'\t', b'\n', b'\r'):
            pos += 1
        elif data[pos:pos+1] == b'#':
            while pos < len(data) and data[pos:pos+1] != b'\n':
                pos += 1
        else:
            break
    return pos


def _read_token(data: bytes, pos: int) -> tuple[str, int]:
    """Read one whitespace-delimited token and return (token, new_pos)."""
    pos = _skip_comments(data, pos)
    start = pos
    while pos < len(data) and data[pos:pos+1] not in (b' ', b'\t', b'\n', b'\r'):
        pos += 1
    return data[start:pos].decode(), pos


def read_ppm(path) -> np.ndarray:
    """
    Parse a PPM file (P3 or P6) and return an (H, W, 3) uint8 array.

    Supports:
      P6 — binary RGB (most common, used by the C++ demos)
      P3 — ASCII RGB
    """
    data = Path(path).read_bytes()

    magic, pos = _read_token(data, 0)
    if magic not in ("P3", "P6"):
        raise ValueError(f"{path}: unsupported format '{magic}' (need P3 or P6)")

    width,  pos = _read_token(data, pos)
    height, pos = _read_token(data, pos)
    maxval, pos = _read_token(data, pos)

    W, H, maxv = int(width), int(height), int(maxval)

    # One mandatory whitespace byte separates header from pixel data (P6)
    pos = _skip_comments(data, pos)
    if data[pos:pos+1] in (b' ', b'\t', b'\n', b'\r'):
        pos += 1

    n_pixels = H * W * 3

    if magic == "P6":
        if maxv <= 255:
            raw = np.frombuffer(data[pos:pos + n_pixels], dtype=np.uint8)
        else:
            # 16-bit P6 (big-endian)
            raw = np.frombuffer(data[pos:pos + n_pixels * 2], dtype=">u2")
            raw = (raw / maxv * 255).astype(np.uint8)
    else:  # P3 ASCII
        tokens = data[pos:].split()
        raw = np.array(tokens[:n_pixels], dtype=np.uint16)
        raw = (raw * 255 // maxv).astype(np.uint8)

    return raw.reshape(H, W, 3)


# ── plotting ───────────────────────────────────────────────────────────────

def group_ppm_files(paths: list[Path]) -> dict[str, list[Path]]:
    """
    Group PPM files by logical simulation type based on filename prefixes.
    Returns an ordered dict: {group_name: [path, ...]}
    """
    groups: dict[str, list] = {}
    for p in sorted(paths):
        name = p.stem
        # strip trailing _phase / _comp / _N
        base = re.sub(r"_\d+$", "", name)
        base = re.sub(r"_(phase|comp)$", "", base)
        groups.setdefault(base, []).append(p)
    return groups


def plot_ppm_files(paths: list[Path], out: Path | None = None,
                   cols: int = 5, title: str = "PPM Viewer"):
    """
    Plot a list of PPM files in a grid.  Each image shows its filename.
    Saves to *out* if provided, otherwise displays interactively.
    """
    n = len(paths)
    if n == 0:
        print("No PPM files found.")
        return

    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols,
                              figsize=(cols * 3.2, rows * 3.2 + 0.6))
    fig.patch.set_facecolor("#0d0d0d")
    fig.suptitle(title, color="white", fontsize=14, fontweight="bold", y=1.01)

    axes_flat = np.array(axes).flatten() if n > 1 else [axes]

    for i, ax in enumerate(axes_flat):
        ax.set_facecolor("#111111")
        if i < n:
            img = read_ppm(paths[i])
            ax.imshow(img)
            ax.set_title(paths[i].name, color="#cccccc", fontsize=7.5, pad=3)
        ax.axis("off")

    plt.tight_layout()

    if out:
        fig.savefig(out, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"Saved → {out}")
    else:
        matplotlib.use("TkAgg")
        plt.show()


def plot_ppm_grouped(groups: dict[str, list[Path]],
                     out: Path | None = None):
    """
    Lay out PPM files in named row-groups (one row per simulation type).
    """
    group_names = list(groups.keys())
    n_groups = len(group_names)

    # Determine max images per group for column count
    max_cols = max(len(v) for v in groups.values())

    fig = plt.figure(figsize=(max_cols * 3.0, n_groups * 3.2 + 0.6))
    fig.patch.set_facecolor("#0d0d0d")
    fig.suptitle("PPM Microstructure Gallery", color="white",
                 fontsize=14, fontweight="bold")

    gs = fig.add_gridspec(n_groups, max_cols, hspace=0.4, wspace=0.05)

    for row, gname in enumerate(group_names):
        files = groups[gname]
        for col in range(max_cols):
            ax = fig.add_subplot(gs[row, col])
            ax.set_facecolor("#111111")
            if col < len(files):
                img = read_ppm(files[col])
                ax.imshow(img)
                label = files[col].stem
                ax.set_title(label, color="#cccccc", fontsize=7, pad=2)
                if col == 0:
                    ax.set_ylabel(gname.replace("_", " ").title(),
                                  color="#ffcc44", fontsize=9,
                                  rotation=90, labelpad=4)
            ax.axis("off")

    if out:
        fig.savefig(out, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"Saved → {out}")
    else:
        matplotlib.use("TkAgg")
        plt.show()


# ── demo: show PPM format internals ────────────────────────────────────────

def explain_ppm(path: Path):
    """Print a human-readable breakdown of a PPM file's header."""
    data = path.read_bytes()
    print(f"\n{'─'*50}")
    print(f"File : {path}")
    print(f"Size : {len(data):,} bytes")
    # Print header as text
    header_end = 0
    fields = 0
    while fields < 4 and header_end < min(200, len(data)):
        c = data[header_end:header_end+1]
        header_end += 1
        if c in (b' ', b'\t', b'\n', b'\r') and data[header_end-2:header_end-1] not in (b' ', b'\t', b'\n', b'\r'):
            fields += 1
    header_text = data[:header_end].decode(errors="replace").strip()
    lines = header_text.split('\n')
    magic  = lines[0]
    wh     = lines[1] if len(lines) > 1 else "?"
    maxval = lines[2] if len(lines) > 2 else "?"
    W, H = (int(x) for x in wh.split())
    print(f"\nHeader:")
    print(f"  Magic  : {magic!r}  ({'binary' if magic=='P6' else 'ASCII'} RGB)")
    print(f"  Size   : {W} × {H} pixels")
    print(f"  Maxval : {maxval}")
    print(f"  Pixel data: {W*H*3:,} bytes  ({W}×{H}×3 channels)")
    print(f"{'─'*50}\n")


# ── main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Plot .ppm files")
    parser.add_argument("files", nargs="*",
                        help="PPM files to plot (default: all in build/)")
    parser.add_argument("--pattern", default="*.ppm",
                        help="Glob pattern inside build/ (default: *.ppm)")
    parser.add_argument("--out", default=None,
                        help="Output PNG path (default: ppm_gallery.png)")
    parser.add_argument("--explain", action="store_true",
                        help="Print PPM header info for each file")
    parser.add_argument("--flat", action="store_true",
                        help="Flat grid layout instead of grouped rows")
    args = parser.parse_args()

    if args.files:
        paths = [Path(f) for f in args.files]
    else:
        paths = sorted(BUILD.glob(args.pattern))

    if not paths:
        print(f"No PPM files found (pattern={args.pattern}, build={BUILD})")
        sys.exit(1)

    print(f"Found {len(paths)} PPM file(s)")

    if args.explain:
        for p in paths:
            explain_ppm(p)

    out_path = Path(args.out) if args.out else \
               Path(__file__).parent / "ppm_gallery.png"

    if args.flat:
        plot_ppm_files(paths, out=out_path)
    else:
        groups = group_ppm_files(paths)
        plot_ppm_grouped(groups, out=out_path)


if __name__ == "__main__":
    main()
