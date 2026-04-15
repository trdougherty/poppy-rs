#!/usr/bin/env python3
"""
compare_engines.py

Runs a PSF simulation through both the numpy and Rust backends, saves
comparison images, and writes a stats report.

If --input-dir is provided, each .fits file in the directory is treated as
an OPD (wavefront error) map and used as an optical element in the system.
Without --input-dir, a synthetic circular aperture is used instead.

Usage examples
--------------
# Synthetic aperture only:
python compare_engines.py --output-dir ./results

# With a directory of OPD FITS files:
python compare_engines.py --input-dir ./opds --output-dir ./results

# Custom wavelength, field of view, oversampling:
python compare_engines.py --input-dir ./opds --output-dir ./results \
    --wavelength 3.5e-6 --fov 5.0 --oversample 4 --nlambda 9
"""

import argparse
import gc
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless — no display required
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import poppy
import poppy_rs  # noqa — confirms it is importable before we start


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_optical_system(opd_path=None, oversample=4, fov_arcsec=3.0):
    """Return a poppy OpticalSystem, optionally with an OPD loaded from file."""
    osys = poppy.OpticalSystem(oversample=oversample)
    osys.add_pupil(poppy.CircularAperture(radius=3.25))  # 6.5 m JWST-like primary
    if opd_path is not None:
        osys.add_pupil(poppy.FITSOpticalElement(opd=str(opd_path), opdunits="meter"))
    osys.add_detector(pixelscale=0.063, fov_arcsec=fov_arcsec)
    return osys


def run_backend(osys, wavelengths, use_rust):
    """Run calc_psf with the chosen backend. Returns (psf_hdul, elapsed_seconds).

    wavelengths: scalar float for monochromatic, or list of floats for broadband.
    """
    poppy.conf.use_rust = use_rust
    poppy.accel_math.update_math_settings()

    t0 = time.perf_counter()
    psf = osys.calc_psf(wavelength=wavelengths)
    elapsed = time.perf_counter() - t0

    return psf, elapsed


def compute_stats(numpy_data, rust_data, t_numpy, t_rust, label):
    """Return a dict of comparison statistics."""
    diff = numpy_data - rust_data
    abs_diff = np.abs(diff)
    peak = numpy_data.max()

    return {
        "label": label,
        "peak_numpy": float(peak),
        "peak_rust": float(rust_data.max()),
        "total_flux_numpy": float(numpy_data.sum()),
        "total_flux_rust": float(rust_data.sum()),
        "max_abs_diff": float(abs_diff.max()),
        "mean_abs_diff": float(abs_diff.mean()),
        "rel_diff_at_peak": float(abs_diff.max() / peak) if peak > 0 else 0.0,
        "time_numpy_s": round(t_numpy, 4),
        "time_rust_s": round(t_rust, 4),
        "speedup": round(t_numpy / t_rust, 3) if t_rust > 0 else None,
    }


def save_comparison_image(numpy_data, rust_data, stats, output_path):
    """Save a 3-panel comparison PNG: numpy | rust | |diff|."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    peak = numpy_data.max()
    norm = mcolors.LogNorm(vmin=peak * 1e-6, vmax=peak)

    for ax, data, title in [
        (axes[0], numpy_data, f"numpy  ({stats['time_numpy_s']*1000:.1f} ms)"),
        (axes[1], rust_data,  f"rust   ({stats['time_rust_s']*1000:.1f} ms)"),
    ]:
        im = ax.imshow(data, norm=norm, cmap="inferno", origin="lower")
        ax.set_title(title, fontsize=12)
        plt.colorbar(im, ax=ax, fraction=0.046)

    diff = np.abs(numpy_data - rust_data)
    diff_max = diff.max()
    if diff_max > 0:
        diff_norm = mcolors.LogNorm(vmin=max(diff_max * 1e-4, 1e-20), vmax=diff_max)
    else:
        diff_norm = mcolors.Normalize(vmin=0, vmax=1)
    im2 = axes[2].imshow(diff, norm=diff_norm, cmap="magma", origin="lower")
    axes[2].set_title(
        f"|numpy − rust|  max={stats['max_abs_diff']:.1e}  "
        f"rel={stats['rel_diff_at_peak']:.1e}",
        fontsize=11,
    )
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    fig.suptitle(
        f"{stats['label']}  —  speedup {stats['speedup']}x",
        fontsize=13,
        y=1.01,
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_profile_image(numpy_data, rust_data, stats, output_path):
    """Save a horizontal-slice profile comparison."""
    cy, cx = np.array(numpy_data.shape) // 2
    profile_n = numpy_data[cy, :]
    profile_r = rust_data[cy, :]
    pixels = np.arange(len(profile_n)) - cx

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    ax1.semilogy(pixels, profile_n, label="numpy", lw=2)
    ax1.semilogy(pixels, profile_r, label="rust", lw=1.5, ls="--", color="tomato")
    ax1.set_ylabel("Intensity")
    ax1.set_title(f"{stats['label']} — horizontal PSF profile")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(pixels, profile_n - profile_r, color="purple", lw=1)
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_ylabel("numpy − rust")
    ax2.set_xlabel("Pixel offset from center")
    ax2.set_title("Residual")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare poppy numpy vs Rust FFT backends on PSF simulation."
    )
    parser.add_argument(
        "--input-dir", "-i",
        type=Path,
        default=None,
        help="Directory of OPD .fits files to use as inputs. "
             "If omitted, a synthetic circular aperture is used.",
    )
    parser.add_argument(
        "--input-file", "-f",
        type=Path,
        default=None,
        help="Single OPD .fits file to use as input.",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=Path("./results"),
        help="Directory to write output images and stats (default: ./results).",
    )
    parser.add_argument(
        "--wavelength", "-w",
        type=float,
        default=2e-6,
        help="Central wavelength in metres (default: 2e-6).",
    )
    parser.add_argument(
        "--nlambda", "-n",
        type=int,
        default=1,
        help="Number of wavelength samples for broadband PSF (default: 1, monochromatic).",
    )
    parser.add_argument(
        "--fov",
        type=float,
        default=3.0,
        help="Field of view in arcseconds (default: 3.0).",
    )
    parser.add_argument(
        "--oversample",
        type=int,
        default=4,
        help="Oversampling factor (default: 4).",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Collect input files or fall back to synthetic
    if args.input_file is not None:
        inputs = [args.input_file]
    elif args.input_dir is not None:
        inputs = sorted(args.input_dir.glob("*.fits"))
        if not inputs:
            print(f"No .fits files found in {args.input_dir}, using synthetic aperture.")
            inputs = [None]
    else:
        inputs = [None]

    all_stats = []

    for opd_path in inputs:
        label = opd_path.stem if opd_path is not None else "synthetic_aperture"
        print(f"\n{'='*60}")
        print(f"Input: {label}")
        print(f"{'='*60}")

        osys = build_optical_system(
            opd_path=opd_path,
            oversample=args.oversample,
            fov_arcsec=args.fov,
        )

        # Build wavelength list: single value or linear spread across ±10% of central wavelength
        if args.nlambda == 1:
            wavelengths = args.wavelength
        else:
            wavelengths = np.linspace(args.wavelength * 0.9, args.wavelength * 1.1, args.nlambda)

        print(f"  Running numpy backend (nlambda={args.nlambda})...")
        psf_numpy, t_numpy = run_backend(osys, wavelengths, use_rust=False)
        print(f"    {t_numpy*1000:.1f} ms")

        print(f"  Running rust backend  (nlambda={args.nlambda})...")
        psf_rust, t_rust = run_backend(osys, wavelengths, use_rust=True)
        print(f"    {t_rust*1000:.1f} ms")

        numpy_data = psf_numpy[0].data
        rust_data  = psf_rust[0].data

        stats = compute_stats(numpy_data, rust_data, t_numpy, t_rust, label)
        all_stats.append(stats)

        print(f"  speedup      : {stats['speedup']}x")
        print(f"  max |diff|   : {stats['max_abs_diff']:.2e}")
        print(f"  rel diff     : {stats['rel_diff_at_peak']:.2e}")

        # Save outputs
        stem = args.output_dir / label
        save_comparison_image(numpy_data, rust_data, stats, f"{stem}_comparison.png")
        save_profile_image(numpy_data, rust_data, stats, f"{stem}_profile.png")
        print(f"  Saved: {stem}_comparison.png")
        print(f"  Saved: {stem}_profile.png")

    # Write stats report
    stats_path = args.output_dir / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"\nStats written to: {stats_path}")

    # Print summary table
    print(f"\n{'label':<30} {'numpy':>10} {'rust':>10} {'speedup':>10} {'max|diff|':>12} {'rel diff':>10}")
    print("-" * 86)
    for s in all_stats:
        print(
            f"{s['label']:<30} "
            f"{s['time_numpy_s']*1000:>9.1f}ms "
            f"{s['time_rust_s']*1000:>9.1f}ms "
            f"{s['speedup']:>9.2f}x "
            f"{s['max_abs_diff']:>12.2e} "
            f"{s['rel_diff_at_peak']:>10.2e}"
        )

    # Leave rust enabled after the script
    poppy.conf.use_rust = True
    poppy.accel_math.update_math_settings()


if __name__ == "__main__":
    main()
