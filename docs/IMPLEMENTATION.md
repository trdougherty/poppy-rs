# Implementation Details

## How the `.so` Extension Works

`poppy-rs` is built with [maturin](https://github.com/PyO3/maturin) and
[PyO3](https://github.com/PyO3/pyo3), which compile Rust into a CPython extension
module (`poppy_rs.cpython-3xx-x86_64-linux-gnu.so`).

### Extension entry point

```rust
#[pymodule]
fn poppy_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fft_2d, m)?)?;
    m.add_function(wrap_pyfunction!(matrix_dft, m)?)?;
    Ok(())
}
```

Two functions are exported. Python sees them as ordinary callables:

```python
import poppy_rs
poppy_rs.fft_2d(wavefront, forward=True, normalization=None, fftshift=True)
poppy_rs.matrix_dft(plane, nlam_d, npix, offset=None, inverse=False, centering="FFTSTYLE")
```

### Buffer passing — zero copy from Python to Rust

PyO3's `numpy` crate accepts a `PyReadonlyArray2<Complex64>`, which is a **borrowed
view** of the numpy array's existing memory — no copy on entry. The array data is then
`.to_owned()` once into a Rust `ndarray::Array2<Complex64>` (one allocation, the
working buffer), all transforms happen in-place on that buffer, and the final result
is handed back to Python via `.into_pyarray(py)` which transfers ownership of the
Rust allocation directly into a numpy array without copying.

Total copies across the round-trip: **one** (the input, so we don't mutate the
caller's array). The output is zero-copy.

### Dependency tree

```
poppy_rs (.so)
├── pyo3          — Python/Rust FFI and GIL management
├── numpy (PyO3)  — zero-copy array interop
├── ndarray       — n-dimensional array operations
├── ndrustfft     — axis-wise FFT on ndarrays (wraps rustfft)
│   └── rustfft   — high-performance FFT planning and execution
└── rayon         — data-parallel iterators (work-stealing thread pool)
```

### GIL release

The Rust FFT and matrix-build code runs **outside the Python GIL**. PyO3's
`py.allow_threads()` releases the GIL during the compute-intensive sections,
allowing other Python threads to run concurrently. Rayon's thread pool is
separate from Python's thread pool and does not hold the GIL.

### Memory layout

Both `fft_2d` and `matrix_dft` operate on C-contiguous (row-major) `complex128`
arrays, which is numpy's default layout for 2D arrays. This matters for:

- **Row FFT (axis 1)**: rows are contiguous in memory — no reordering needed, SIMD
  loads are aligned.
- **Column FFT (axis 0)**: ndrustfft uses internal lane buffers to handle the
  non-contiguous access; rayon parallelizes across independent column groups.
- **Matrix exp construction**: `par_chunks_mut(npup_y)` on the flat slice gives each
  rayon worker a contiguous chunk of one output row — cache-line aligned writes.

### Normalization convention

poppy's sign/normalization convention differs from numpy's default:

| Direction | numpy default | poppy convention | Rust correction |
|---|---|---|---|
| Forward  | unnormalized  | `× 1/N`          | `× 1/N` |
| Inverse  | `× 1/(N·M)`   | `× N`            | `× N · 1/(N·M) = 1/M` |

`rustfft` (via ndrustfft) applies no normalization in either direction, so
the Rust code applies the combined factor explicitly after the transform.

## Building from Source

```bash
# Install maturin
pip install maturin

# Build and install into the current virtualenv (development mode)
cd poppy-rs
maturin develop --release

# Build a distributable wheel
maturin build --release
```

Requires: Rust ≥ 1.85 (edition 2024), Python ≥ 3.9.

## Integration with poppy

poppy detects the extension at import time:

```python
# poppy/accel_math.py
try:
    import poppy_rs
    _RUST_AVAILABLE = True
except ImportError:
    poppy_rs = None
    _RUST_AVAILABLE = False
```

Enable the backend:

```python
import poppy
poppy.conf.use_rust = True
poppy.accel_math.update_math_settings()
```

Or set it once as a default in your `~/.poppy.cfg`:

```ini
[configuration]
use_rust = True
```

The poppy change required to activate the MFT dispatch path is a 7-line addition to
`poppy/matrixDFT.py` (see `patches/matrixDFT_rust_dispatch.patch`).
