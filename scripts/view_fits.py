#!/usr/bin/env python3
"""
view_fits.py

Visualize one or more FITS files from the stpsf data directory.
Automatically detects the data type (OPD, pupil, segment map) and
applies an appropriate colormap and scale.

Usage
-----
# Single file:
python scripts/view_fits.py ~/data/stpsf-data/JWST_OTE_OPD_cycle1_example_2022-07-30.fits

# All files in a directory:
python scripts/view_fits.py ~/data/stpsf-data/

# Save to a specific output directory instead of displaying:
python scripts/view_fits.py ~/data/stpsf-data/ --output-dir ./results/raw_fits

# Show only OPD files:
python scripts/view_fits.py ~/data/stpsf-data/ --filter opd
"""

import argparse
import os
from pathlib import Path

import astropy.io.fits as fits
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


# ---------------------------------------------------------------------------
# Data type detection
# ---------------------------------------------------------------------------

def detect_type(header, data):
    """Guess what kind of FITS data this is from header and data stats."""
    bunit = header.get("BUNIT", "").lower()
    filename = header.get("FILENAME", "").lower()
    comment = " ".join(str(v) for v in header.get("COMMENT", [])).lower()

    if "micron" in bunit or "meter" in bunit or "nm" in bunit or "opd" in filename:
        return "opd"
    if data.dtype == np.uint8 or (np.issubdtype(data.dtype, np.integer) and data.max() < 50):
        return "segments"
    if data.min() >= 0 and data.max() <= 1 and np.unique(data).size < 10:
        return "pupil_binary"
    if data.min() >= 0 and data.max() <= 1:
        return "pupil"
    return "opd"  # default — signed float → treat as wavefront


# ---------------------------------------------------------------------------
# Per-type plot configuration
# ---------------------------------------------------------------------------

def plot_opd(ax, data, header):
    bunit = header.get("BUNIT", "microns")
    vmax = np.nanpercentile(np.abs(data[data != 0]), 99) if np.any(data != 0) else 1.0
    im = ax.imshow(data, cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="lower")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, label=bunit)
    rms = np.std(data[data != 0]) if np.any(data != 0) else 0.0
    ax.set_title(f"OPD  (RMS={rms:.4f} {bunit})", fontsize=11)


def plot_pupil(ax, data, header):
    im = ax.imshow(data, cmap="gray", vmin=0, vmax=1, origin="lower")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, label="transmission")
    ax.set_title(f"Pupil mask  ({data.shape[0]}×{data.shape[1]} px)", fontsize=11)


def plot_segments(ax, data, header):
    n_segments = int(data.max())
    # Use a discrete colormap — one colour per segment, black for gaps
    cmap = plt.get_cmap("tab20", n_segments + 1)
    cmap.set_under("black")
    im = ax.imshow(data, cmap=cmap, vmin=0.5, vmax=n_segments + 0.5, origin="lower")
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, ticks=range(1, n_segments + 1))
    cbar.set_label("Segment ID")
    ax.set_title(f"Segment map  ({n_segments} segments)", fontsize=11)


PLOT_FN = {
    "opd": plot_opd,
    "pupil": plot_pupil,
    "pupil_binary": plot_pupil,
    "segments": plot_segments,
}


# ---------------------------------------------------------------------------
# Per-file rendering
# ---------------------------------------------------------------------------

def render_fits(fits_path, output_dir=None, show=False):
    path = Path(fits_path).expanduser()

    try:
        with fits.open(path) as hdul:
            # Find the first extension with 2D image data
            ext = next(
                (e for e in hdul if e.data is not None and e.data.ndim == 2),
                None,
            )
            if ext is None:
                print(f"  skip {path.name} — no 2D image data found")
                return

            raw = ext.data.copy()        # copy while file is still open
            header = ext.header.copy()

    except Exception as e:
        print(f"  skip {path.name} — could not open: {e}")
        return

    data = raw.astype(np.float64)
    dtype = detect_type(header, raw)
    print(f"  {path.name}  [{dtype}]  shape={data.shape}  "
          f"min={data.min():.4g}  max={data.max():.4g}")

    fig, ax = plt.subplots(figsize=(7, 7))
    PLOT_FN[dtype](ax, data, header)
    ax.set_xlabel("pixel x")
    ax.set_ylabel("pixel y")
    fig.suptitle(path.name, fontsize=10, y=0.98)
    plt.tight_layout()

    if output_dir is not None:
        out_path = Path(output_dir) / (path.stem.replace(".fits", "") + "_view.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"    → {out_path}")
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Visualize stpsf FITS data files.")
    parser.add_argument(
        "input",
        type=Path,
        help="Path to a single .fits file or a directory of .fits files.",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=None,
        help="Directory to save PNG outputs. If omitted, images are displayed interactively.",
    )
    parser.add_argument(
        "--filter", "-f",
        choices=["opd", "pupil", "segments", "all"],
        default="all",
        help="Only show a specific data type (default: all).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display images interactively (requires a display).",
    )
    args = parser.parse_args()

    input_path = args.input.expanduser()

    # Collect files
    if input_path.is_dir():
        exts = ["*.fits", "*.fits.gz"]
        files = []
        for pattern in exts:
            files.extend(sorted(input_path.glob(pattern)))
    elif input_path.is_file():
        files = [input_path]
    else:
        print(f"Error: {input_path} is not a file or directory.")
        return

    if not files:
        print(f"No .fits files found at {input_path}")
        return

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        matplotlib.use("Agg")
    elif args.show:
        matplotlib.use("TkAgg")

    print(f"Found {len(files)} file(s)\n")

    for f in files:
        render_fits(f, output_dir=args.output_dir, show=args.show)


if __name__ == "__main__":
    main()
