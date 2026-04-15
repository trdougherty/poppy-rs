"""
integration_test.py

Full end-to-end PSF integration test using stpsf + a real JWST OPD.
Exercises the complete poppy propagation stack (FFT + MFT) with an actual
instrument model and filter bandpass, comparing numpy vs Rust backends.

Usage:
    cd /home/thomas/Work/JWST/poppy-rs
    source ../venv/bin/activate
    maturin develop --release
    python tests/integration_test.py
"""

import json
import sys
import time
from pathlib import Path
import numpy as np
import poppy
import stpsf

STPSF_DATA = "/home/thomas/data/stpsf-data"
OPD_FILE   = f"{STPSF_DATA}/JWST_OTE_OPD_cycle1_example_2022-07-30.fits"

# Instrument + filter combinations to test.
# nlambda=None lets stpsf pick the correct number for the filter width.
CONFIGS = [
    dict(instrument="NIRCam", filter="F200W", fov_arcsec=3.0, nlambda=None),
    dict(instrument="NIRCam", filter="F444W", fov_arcsec=3.0, nlambda=None),
    dict(instrument="MIRI",   filter="F560W", fov_arcsec=3.0, nlambda=None),
]


def make_instrument(instrument_name, filter_name, opd_file):
    inst = getattr(stpsf, instrument_name)()
    inst.filter   = filter_name
    inst.pupilopd = opd_file
    return inst


def run_backend(inst, fov_arcsec, nlambda, use_rust):
    poppy.conf.use_rust = use_rust
    poppy.accel_math.update_math_settings()

    kwargs = dict(fov_arcsec=fov_arcsec, oversample=2)
    if nlambda is not None:
        kwargs["nlambda"] = nlambda

    t0 = time.perf_counter()
    psf = inst.calc_psf(**kwargs)
    elapsed = time.perf_counter() - t0

    return psf[0].data, elapsed


def check(label, numpy_data, rust_data, t_numpy, t_rust):
    diff     = np.abs(numpy_data - rust_data)
    peak     = numpy_data.max()
    max_diff = float(diff.max())
    rel_diff = max_diff / peak if peak > 0 else 0.0
    speedup  = round(t_numpy / t_rust, 3) if t_rust > 0 else float("inf")

    tol = 1e-10
    passed = rel_diff < tol

    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {label}")
    print(f"         numpy       = {t_numpy*1000:.1f} ms")
    print(f"         rust        = {t_rust*1000:.1f} ms")
    print(f"         speedup     = {speedup:.2f}x")
    print(f"         max |diff|  = {max_diff:.2e}  (rel {rel_diff:.2e}, tol {tol:.0e})")
    print()

    return passed, {
        "label":        label,
        "time_numpy_s": round(t_numpy, 4),
        "time_rust_s":  round(t_rust, 4),
        "speedup":      speedup,
        "max_abs_diff": max_diff,
        "rel_diff":     rel_diff,
        "passed":       bool(passed),
    }


def main():
    try:
        import poppy_rs  # noqa — confirms it is importable
    except ImportError:
        print("ERROR: poppy_rs not found. Run: maturin develop --release")
        sys.exit(1)

    all_pass   = True
    all_stats  = []
    output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(exist_ok=True)

    for cfg in CONFIGS:
        inst_name  = cfg["instrument"]
        filt       = cfg["filter"]
        fov        = cfg["fov_arcsec"]
        nlambda    = cfg["nlambda"]
        label      = f"{inst_name} {filt}"

        print("=" * 60)
        print(label)
        print("=" * 60)

        inst = make_instrument(inst_name, filt, OPD_FILE)

        numpy_data, t_numpy = run_backend(inst, fov, nlambda, use_rust=False)
        rust_data,  t_rust  = run_backend(inst, fov, nlambda, use_rust=True)

        passed, stats = check(label, numpy_data, rust_data, t_numpy, t_rust)
        all_pass  = all_pass and passed
        all_stats.append(stats)

    stats_path = output_dir / "integration_stats.json"
    with open(stats_path, "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"Results written to {stats_path}")

    print("=" * 60)
    print("SUMMARY:", "ALL PASS" if all_pass else "FAILURES")
    print("=" * 60)
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
