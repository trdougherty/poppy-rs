"""
rebuild_star_field.py

Download a real JWST NIRCam cal.fits image of the LMC calibration field,
detect stars, build a PSF model grid using stpsf + the Rust backend, then
forward-model the field and show how well the model matches the data.

Outputs (written to --output-dir):
  star_field_panel.png   — data | model | residuals (3-panel)
  star_field_psf_grid.png — sampled PSF grid used for modelling

Usage:
    source ../venv/bin/activate
    maturin develop --release
    python scripts/rebuild_star_field.py [--input path/to/cal.fits]
                                         [--output-dir ./results/star_field]
                                         [--cutout-size 512]
                                         [--backend rust|numpy]
"""

import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.stats import sigma_clipped_stats
from astropy.visualization import simple_norm
from astropy.wcs import WCS
from photutils.detection import DAOStarFinder
from photutils.psf import GriddedPSFModel, PSFPhotometry, SourceGrouper
import poppy
import stpsf

STPSF_DATA = Path.home() / "data" / "stpsf-data"
DEFAULT_INPUT = (
    Path.home()
    / "data" / "jwst_lmc"
    / "mastDownload" / "JWST"
    / "jw01476006004_06101_00001_nrca1"
    / "jw01476006004_06101_00001_nrca1_cal.fits"
)


# ---------------------------------------------------------------------------
# PSF grid
# ---------------------------------------------------------------------------

def build_psf_grid(instrument_name, filter_name, opd_file, detector,
                   grid_step=512, oversampling=4):
    """Compute a stpsf PSF grid across the detector using the active backend."""
    inst = getattr(stpsf, instrument_name)()
    inst.filter   = filter_name
    inst.detector = detector
    inst.pupilopd = str(opd_file)

    t0 = time.perf_counter()
    grid = inst.psf_grid(
        num_psfs=9,          # 3x3 grid across the detector
        all_detectors=False,
        oversample=oversampling,
        fov_pixels=64,       # size of each PSF stamp
        verbose=False,
    )
    elapsed = time.perf_counter() - t0
    return grid, elapsed


# ---------------------------------------------------------------------------
# Image loading + cutout
# ---------------------------------------------------------------------------

def load_cutout(fits_path, size):
    with fits.open(fits_path) as hdul:
        sci   = hdul["SCI"].data.astype(np.float64)
        hdr   = hdul["SCI"].header
        wcs   = WCS(hdr)

    cy, cx = np.array(sci.shape) // 2
    cutout = Cutout2D(sci, (cx, cy), size, wcs=wcs)
    return cutout.data, cutout.wcs, hdr


# ---------------------------------------------------------------------------
# Star detection
# ---------------------------------------------------------------------------

def detect_stars(data, fwhm=2.5, threshold_sigma=5.0, max_stars=80):
    _, median, std = sigma_clipped_stats(data, sigma=3.0)
    finder = DAOStarFinder(fwhm=fwhm, threshold=threshold_sigma * std,
                           brightest=max_stars)
    sources = finder(data - median)
    return sources, median


# ---------------------------------------------------------------------------
# Forward model
# ---------------------------------------------------------------------------

def forward_model(data, sources, psf_grid_model):
    """Fit PSF photometry to detected sources; return model image."""
    grouper  = SourceGrouper(min_separation=10.0)
    photometry = PSFPhotometry(
        psf_model=psf_grid_model,
        fit_shape=(21, 21),
        grouper=grouper,
        aperture_radius=5,
    )
    result   = photometry(data, init_params=sources)
    residual = photometry.make_residual_image(data, psf_shape=(21, 21))
    model    = data - residual
    return model, residual, result


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def save_panel(data, model, residual, output_path, title):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    norm_data = simple_norm(data, stretch="log", percent=99.5)
    norm_res  = simple_norm(np.abs(residual), stretch="linear", percent=99)

    for ax, img, norm, label in [
        (axes[0], data,     norm_data, "Data"),
        (axes[1], model,    norm_data, "PSF Model"),
        (axes[2], residual, norm_res,  "Residual"),
    ]:
        ax.imshow(img, norm=norm, cmap="inferno", origin="lower")
        ax.set_title(label, fontsize=13)
        ax.axis("off")

    fig.suptitle(title, fontsize=14, y=1.01)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def save_psf_grid(psf_grid_model, output_path):
    data   = psf_grid_model.data          # (n_psfs, ny, nx) or (n, 1, ny, nx)
    stamps = data.reshape(-1, data.shape[-2], data.shape[-1])
    n      = int(np.ceil(np.sqrt(len(stamps))))

    fig, axes = plt.subplots(n, n, figsize=(n * 2.5, n * 2.5))
    axes = np.array(axes).ravel()
    for i, (ax, stamp) in enumerate(zip(axes, stamps)):
        ax.imshow(stamp, norm=simple_norm(stamp, stretch="log", percent=99),
                  cmap="viridis", origin="lower")
        ax.set_title(f"PSF {i}", fontsize=8)
        ax.axis("off")
    for ax in axes[len(stamps):]:
        ax.axis("off")

    plt.suptitle("stpsf PSF grid (Rust backend)", fontsize=12)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",       "-i", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir",  "-o", type=Path, default=Path("results/star_field"))
    parser.add_argument("--cutout-size", "-c", type=int,  default=512)
    parser.add_argument(
        "--backend", "-b", choices=["rust", "numpy"], default="rust",
        help="Which poppy backend to use for PSF grid (default: rust)",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"ERROR: input file not found:\n  {args.input}")
        print("\nDownload it with:")
        print("  python -c \"from astroquery.mast import Observations; ...\"")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)

    use_rust = args.backend == "rust"
    poppy.conf.use_rust = use_rust
    poppy.accel_math.update_math_settings()

    # --- Load image ---
    print(f"\nLoading {args.input.name} ...")
    data, wcs, hdr = load_cutout(args.input, args.cutout_size)
    _instrume_map = {"NIRCAM": "NIRCam", "NIRISS": "NIRISS", "MIRI": "MIRI",
                     "NIRSPEC": "NIRSpec", "FGS": "FGS"}
    instrument  = _instrume_map.get(hdr.get("INSTRUME", "NIRCAM"), "NIRCam")
    filter_name = hdr.get("FILTER",   "F200W")
    detector    = hdr.get("DETECTOR", "NRCA1").upper()
    print(f"  Instrument : {instrument}  Filter : {filter_name}  Detector : {detector}")
    print(f"  Cutout     : {data.shape[0]}×{data.shape[1]} px")

    # --- Detect stars ---
    print("\nDetecting stars ...")
    sources, sky_median = detect_stars(data)
    print(f"  Found {len(sources)} sources")

    # --- Build PSF grid ---
    print(f"\nBuilding PSF grid ({args.backend} backend) ...")
    opd_file = STPSF_DATA / "JWST_OTE_OPD_cycle1_example_2022-07-30.fits"
    psf_grid, t_psf = build_psf_grid(instrument, filter_name, opd_file, detector)
    print(f"  PSF grid computed in {t_psf*1000:.0f} ms")

    save_psf_grid(psf_grid, args.output_dir / "star_field_psf_grid.png")

    # --- Forward model ---
    print("\nFitting PSF model to sources ...")
    t0 = time.perf_counter()
    model, residual, phot_table = forward_model(data - sky_median, sources, psf_grid)
    t_fit = time.perf_counter() - t0
    print(f"  PSF fit completed in {t_fit*1000:.0f} ms  ({len(phot_table)} sources fit)")

    # --- Residual stats ---
    _, med_res, std_res = sigma_clipped_stats(residual, sigma=3.0)
    _, med_data, std_data = sigma_clipped_stats(data, sigma=3.0)
    print(f"\n  Background std (data)     : {std_data:.4f} MJy/sr")
    print(f"  Residual  std             : {std_res:.4f} MJy/sr")
    print(f"  Frac residual / data std  : {std_res / std_data:.3f}")

    # --- Save panel ---
    title = (
        f"JWST {instrument} {filter_name}  —  LMC Calibration Field  "
        f"({args.backend} backend, {len(phot_table)} stars)"
    )
    save_panel(data - sky_median, model, residual,
               args.output_dir / "star_field_panel.png", title)

    print("\nDone.")


if __name__ == "__main__":
    main()
