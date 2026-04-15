# The Problem: Python Memory and Buffer Management in poppy's FFT Hot Path

## Background

[poppy](https://github.com/spacetelescope/poppy) is the physical optics propagation library
used by the James Webb Space Telescope's PSF simulation tools (stpsf, formerly WebbPSF).
Its inner loop — computing a PSF across N wavelengths — repeatedly calls two primitives:

1. `accel_math.fft_2d` — a 2D complex FFT used to propagate wavefronts between optical planes
2. `matrixDFT.matrix_dft` — a matrix discrete Fourier transform (Soummer et al. 2007) used
   to resample the wavefront onto the detector at arbitrary magnification

For a typical broadband NIRCam F200W PSF (21 wavelengths, 1024×1024 pupil plane, 4× oversample),
these functions are called **hundreds of times per `calc_psf()` invocation**.

## The Memory Problem

### 1. Temporary array explosion in `fft_2d`

poppy's numpy FFT path does:

```python
wavefront = np.fft.ifftshift(wavefront)   # allocation #1
wavefront = np.fft.fft2(wavefront)        # allocation #2
wavefront = np.fft.fftshift(wavefront)    # allocation #3
wavefront *= normalization                # in-place, OK
```

Each `fftshift` / `ifftshift` / `fft2` call allocates a **full new array**.
For a 4096×4096 complex128 wavefront that is **268 MB per temporary**, yielding
~800 MB of peak live memory per wavelength just for shift + FFT overhead.

At 21 wavelengths computed sequentially, Python's GC does reclaim these between calls,
but each call still causes a large allocation spike. On machines with limited RAM or
when running many wavelengths in parallel this causes OOM kills.

### 2. The column-FFT striding problem

numpy's `fft2` internally handles both axes via its pocketfft backend, which transposes
the data to get contiguous column access. The old rustfft-based implementation in poppy-rs
did this naively — copying each column element-by-element into a temporary buffer:

```rust
let mut col_buf = vec![Complex64::new(0.0, 0.0); rows];
for c in 0..cols {
    for r in 0..rows { col_buf[r] = buf[[r, c]]; }  // 4096 cache misses per column
    col_fft.process(&mut col_buf);
    for r in 0..rows { buf[[r, c]] = col_buf[r]; }
}
```

For a 4096×4096 array this is 4096 separate column passes, each causing a full cache
thrash. Benchmarks showed this was **slower than numpy** (0.64× speedup) despite being
compiled native code.

### 3. Quadratic serial allocation in `matrix_dft`

The matrix DFT builds two exponential matrices before the triple dot product:

```python
# expYV[v, y] = exp(2πi · Vs[v] · Ys[y])
expYV = np.exp(2j * np.pi * np.outer(Vs, Ys))   # (npix, npup) allocation
expXU = np.exp(2j * np.pi * np.outer(Xs, Us))   # (npup, npix) allocation
```

For a coronagraphic mode (1024→512 detector), each matrix is ~8 MB and contains
~500k `exp()` evaluations computed **serially**. At 20 wavelengths this is 10M
serial transcendental function calls just to build the matrices, before any BLAS
work begins. The `numexpr` path partially mitigates this but does not parallelize
across rows.

### 4. No dispatch path for Rust in `matrix_dft`

`accel_math.fft_2d` had been given a Rust dispatch hook, but `matrixDFT.matrix_dft`
had no equivalent — the Rust backend was silently ignored for all MFT operations,
meaning even with `poppy.conf.use_rust = True` only half the hot path was accelerated.

## Observed Impact

| Workload | Peak temp alloc / wavelength | Sequential calls |
|---|---|---|
| NIRCam F200W broadband (21 wl) | ~800 MB | ~63 FFT + 21 MFT |
| NIRCam coronagraphy (20 wl, 4096² FFT) | ~2.4 GB | ~60 FFT + 20 MFT |
| Gridded PSF library (144 wl) | ~800 MB × 144 | 432 FFT + 144 MFT |

Running `compare_engines.py` on a 1024 px JWST pupil FITS file against both backends
caused a full system OOM crash on a 16 GB machine before this work.
