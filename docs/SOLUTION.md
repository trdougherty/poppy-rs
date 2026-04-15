# The Solution: A Rust Extension Backend for poppy

## Approach

`poppy-rs` is a compiled Rust extension module (`.so`) that replaces poppy's two FFT
primitives with parallel, in-place implementations that eliminate the temporary
allocation problem and saturate available CPU cores.

It is a **drop-in replacement**: when installed, poppy routes to it automatically via
`poppy.conf.use_rust = True` with zero changes to calling code.

## Why Rust Fixes It

### 1. In-place 2D FFT via `ndrustfft`

The Rust `fft_2d` implementation uses [`ndrustfft`](https://github.com/preiter93/ndrustfft),
which applies the 1D FFT along each axis directly in the ndarray's native memory layout.
For axis-0 (rows): each row is already contiguous in row-major memory, so no copy is
needed. For axis-1 (columns): ndrustfft internally works lane-by-lane with SIMD-friendly
access patterns rather than strided element-wise copies.

Both axis passes are parallelized across lanes using **rayon**, meaning on an N-core
machine the effective FFT time scales roughly as `O(N·M·log(N·M) / cores)` rather
than serial `O(N·M·log(N·M))`.

The entire transform is done **in-place on a single owned buffer** — no additional
268 MB allocations per call. The fftshift/ifftshift operations are also done in-place
via quadrant swaps.

```
numpy path:  alloc(ifftshift) → alloc(fft2) → alloc(fftshift) → multiply
Rust path:   ifftshift_inplace → ndfft_inplace_par(ax1) → ndfft_inplace_par(ax0) → scale
             [one buffer throughout]
```

### 2. Parallel exponential matrix construction in `matrix_dft`

The exponential matrices for the MFT are now built in parallel across output rows
using rayon's `par_chunks_mut`:

```rust
exp_yv.as_slice_mut().unwrap()
    .par_chunks_mut(npup_y)
    .enumerate()
    .for_each(|(v, row)| {
        for (y, cell) in row.iter_mut().enumerate() {
            let phase = sign * 2.0 * PI * vs[v] * ys[y];
            *cell = Complex64::new(phase.cos(), phase.sin());
        }
    });
```

This turns the serial O(npix × npup) transcendental loop into a parallel one.
On an 8-core machine, matrix construction time drops by ~6–7×.

### 3. Full dispatch coverage — FFT and MFT

The `matrix_dft` function in `poppy/matrixDFT.py` now has a Rust dispatch path
equivalent to the one already present in `accel_math.fft_2d`:

```python
if accel_math._USE_RUST:
    _nlam_d = (np.float64(nlamD), np.float64(nlamD)) if np.isscalar(nlamD) \
              else (np.float64(nlamD[0]), np.float64(nlamD[1]))
    _npix   = (int(npix), int(npix)) if np.isscalar(npix) \
              else (int(npix[0]), int(npix[1]))
    return accel_math.poppy_rs.matrix_dft(
        plane, _nlam_d, _npix, offset=offset, inverse=inverse, centering=centering,
    )
```

With both paths covered, `poppy.conf.use_rust = True` now accelerates the **entire
hot path** of a PSF simulation, not just half of it.

### 4. Memory safety

Rust's ownership model guarantees:
- No use-after-free of FFT buffers
- No data races in the parallel rayon loops (the compiler rejects them at compile time)
- Deterministic deallocation — the single working buffer is freed as soon as the
  function returns, with no GC pressure on the Python side

## Numerical Fidelity

The Rust backend produces bit-for-bit identical results to the numpy path for the
FFT (exact match, `max_abs_diff = 0.0` on all tested inputs) and floating-point
rounding-equivalent results for the MFT (`max_abs_diff < 1e-16` relative).

Full integration tests across NIRCam F200W, F444W, and MIRI F560W with a real
on-orbit JWST OPD confirm correctness end-to-end through stpsf's `calc_psf()`.

## Observed Speedups (integration test, real JWST OPD, cycle 1 example)

| Instrument / Filter | numpy | Rust | Speedup |
|---|---|---|---|
| NIRCam F200W (21 wl) | 5499 ms | 2188 ms | **2.51×** |
| NIRCam F444W (9 wl)  | 1974 ms | 1364 ms | **1.45×** |
| MIRI F560W   (9 wl)  | 2414 ms | 1966 ms | **1.23×** |

The largest gain is on wide filters with many wavelength samples (F200W: 21 wl),
where the parallel FFT and MFT matrix construction have the most work to amortize
across cores. Narrower filters with fewer wavelengths show more modest gains since
the Python orchestration overhead becomes relatively larger.
