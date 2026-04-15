"""
Comparison test suite for poppy_rs vs poppy (numpy).

Validates numerical correctness and reports timing.

Usage:
    cd /home/thomas/Work/JWST/poppy-rs
    source ../venv/bin/activate
    maturin develop --release
    python tests/compare.py
"""

import numpy as np
import time
import sys

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_circular_pupil(npix, fill=0.95):
    """Create a circular aperture filling `fill` fraction of the array."""
    y, x = np.mgrid[-1:1:complex(0, npix), -1:1:complex(0, npix)]
    r = np.sqrt(x**2 + y**2)
    pupil = np.zeros((npix, npix), dtype=np.complex128)
    pupil[r <= fill] = 1.0
    pupil /= np.sqrt(np.sum(np.abs(pupil)**2))
    return pupil


def report(name, match, max_diff, time_numpy, time_rust):
    status = "PASS" if match else "FAIL"
    speedup = time_numpy / time_rust if time_rust > 0 else float('inf')
    print(f"  [{status}] {name}")
    print(f"         max |diff|  = {max_diff:.2e}")
    print(f"         numpy       = {time_numpy*1000:.1f} ms")
    print(f"         rust        = {time_rust*1000:.1f} ms")
    print(f"         speedup     = {speedup:.2f}x")
    print()
    return match


# ---------------------------------------------------------------------------
# FFT tests
# ---------------------------------------------------------------------------

def test_fft(poppy_accel, poppy_rs, sizes=(256, 512, 1024, 2048)):
    """Compare fft_2d output for multiple array sizes."""
    print("=" * 60)
    print("FFT_2D COMPARISON")
    print("=" * 60)

    all_pass = True

    for n in sizes:
        arr = np.random.rand(n, n).astype(np.complex128) + \
              1j * np.random.rand(n, n)

        for forward in [True, False]:
            label = f"{'fwd' if forward else 'inv'} {n}x{n}"

            # numpy (poppy) reference
            t0 = time.perf_counter()
            ref = poppy_accel.fft_2d(arr, forward=forward)
            t_numpy = time.perf_counter() - t0

            # rust
            t0 = time.perf_counter()
            rust_out = poppy_rs.fft_2d(arr, forward=forward)
            t_rust = time.perf_counter() - t0

            max_diff = np.max(np.abs(ref - rust_out))
            # Scale tolerance by array size (FFT accumulates floating point error)
            tol = n * 1e-10
            passed = report(label, max_diff < tol, max_diff, t_numpy, t_rust)
            all_pass = all_pass and passed

    return all_pass


# ---------------------------------------------------------------------------
# Matrix DFT tests
# ---------------------------------------------------------------------------

def test_matrix_dft(poppy_mft_module, poppy_rs):
    """Compare matrix_dft output across centering modes and sizes."""
    print("=" * 60)
    print("MATRIX_DFT COMPARISON")
    print("=" * 60)

    configs = [
        (256, 64,  16.0, "256->64"),
        (256, 256, 16.0, "256->256"),
        (512, 128, 16.0, "512->128"),
        (1024, 256, 16.0, "1024->256"),
    ]

    centerings = ["FFTSTYLE", "SYMMETRIC", "ADJUSTABLE"]
    all_pass = True

    for npup, npix, nlam_d, desc in configs:
        pupil = make_circular_pupil(npup)

        for centering in centerings:
            label = f"{desc} {centering}"

            # numpy (poppy) reference
            t0 = time.perf_counter()
            ref = poppy_mft_module.matrix_dft(
                pupil, nlam_d, npix,
                centering=centering,
            )
            t_numpy = time.perf_counter() - t0

            # rust — accepts (nlamDY, nlamDX) tuple and (npixY, npixX) tuple
            t0 = time.perf_counter()
            rust_out = poppy_rs.matrix_dft(
                pupil,
                (nlam_d, nlam_d),
                (npix, npix),
                centering=centering,
            )
            t_rust = time.perf_counter() - t0

            max_diff = np.max(np.abs(ref - rust_out))
            tol = 1e-10
            passed = report(label, max_diff < tol, max_diff, t_numpy, t_rust)
            all_pass = all_pass and passed

    return all_pass


# ---------------------------------------------------------------------------
# Roundtrip test: forward MFT then inverse should recover the pupil
# ---------------------------------------------------------------------------

def test_mft_roundtrip(poppy_mft_module, poppy_rs):
    """Test forward -> inverse MFT roundtrip for both backends."""
    print("=" * 60)
    print("MFT ROUNDTRIP (forward -> inverse)")
    print("=" * 60)

    npup = 300
    npix = 2000
    nlam_d = 200.0
    pupil = make_circular_pupil(npup)

    all_pass = True

    for backend_name, mft_fn, imft_fn in [
        ("numpy",
         lambda p: poppy_mft_module.matrix_dft(p, nlam_d, npix, centering="ADJUSTABLE"),
         lambda img: poppy_mft_module.matrix_dft(img, nlam_d, npup, inverse=True, centering="ADJUSTABLE")),
        ("rust",
         lambda p: poppy_rs.matrix_dft(p, (nlam_d, nlam_d), (npix, npix), centering="ADJUSTABLE"),
         lambda img: poppy_rs.matrix_dft(img, (nlam_d, nlam_d), (npup, npup), inverse=True, centering="ADJUSTABLE")),
    ]:
        t0 = time.perf_counter()
        image = mft_fn(pupil)
        recovered = imft_fn(image)
        elapsed = time.perf_counter() - t0

        recovered_intensity = np.abs(recovered)**2
        pupil_intensity = np.abs(pupil)**2

        # Normalize for comparison
        recovered_intensity /= recovered_intensity.max()
        pupil_intensity /= pupil_intensity.max()

        max_diff = np.max(np.abs(pupil_intensity - recovered_intensity))
        tol = 1e-4  # roundtrip accumulates error
        passed = max_diff < tol

        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {backend_name} roundtrip")
        print(f"         max |diff|  = {max_diff:.2e}")
        print(f"         time        = {elapsed*1000:.1f} ms")
        print()
        all_pass = all_pass and passed

    return all_pass


# ---------------------------------------------------------------------------
# MFT-FFT equivalence (from poppy's own test suite)
# ---------------------------------------------------------------------------

def test_mft_fft_equivalence(poppy_rs):
    """The MFT should match the FFT when computed on the same sampling."""
    print("=" * 60)
    print("MFT-FFT EQUIVALENCE (rust backend)")
    print("=" * 60)

    npix = 256
    pupil = make_circular_pupil(npix)
    nlam_d = float(npix)

    # MFT with FFTSTYLE centering at native sampling = FFT
    mft_out = poppy_rs.matrix_dft(
        pupil, (nlam_d, nlam_d), (npix, npix), centering="FFTSTYLE"
    )

    # Equivalent FFT (poppy sign convention: forward propagation = inverse FFT)
    fft_out = np.fft.fftshift(
        np.fft.ifft2(np.fft.fftshift(pupil))
    ) * np.sqrt(npix * npix)

    max_diff = np.max(np.abs(mft_out - fft_out))
    norm = np.abs(mft_out).sum()
    rel_diff = np.max(np.abs(mft_out - fft_out)) / norm

    tol = 1e-10
    passed = rel_diff < tol

    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] MFT == FFT (FFTSTYLE, {npix}x{npix})")
    print(f"         max |diff| / norm = {rel_diff:.2e}")
    print()
    return passed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        import poppy_rs
    except ImportError:
        print("ERROR: poppy_rs not found. Run: maturin develop --release")
        sys.exit(1)

    try:
        import poppy
        from poppy import accel_math as poppy_accel
        from poppy import matrixDFT as poppy_mft
    except ImportError:
        print("ERROR: poppy not found. Run: pip install -e ../poppy")
        sys.exit(1)

    # Force numpy for all poppy reference calls so we are always comparing
    # Rust against numpy, not Rust against Rust.
    poppy.conf.use_rust = False
    poppy.accel_math.update_math_settings()

    results = []

    results.append(("FFT correctness", test_fft(poppy_accel, poppy_rs)))
    results.append(("MFT correctness", test_matrix_dft(poppy_mft, poppy_rs)))
    results.append(("MFT roundtrip", test_mft_roundtrip(poppy_mft, poppy_rs)))
    results.append(("MFT-FFT equivalence", test_mft_fft_equivalence(poppy_rs)))

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_pass = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
        all_pass = all_pass and passed

    print()
    sys.exit(0 if all_pass else 1)
