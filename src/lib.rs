use ndarray::{Array1, Array2};
use num_complex::Complex64;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use ndrustfft::{ndfft_inplace_par, ndifft_inplace_par, FftHandler};
use rayon::prelude::*;
use std::f64::consts::PI;

/// 2D FFT with optional fftshift and normalization.
/// Drop-in replacement for poppy.accel_math.fft_2d.
#[pyfunction]
#[pyo3(signature = (wavefront, forward=true, normalization=None, fftshift=true))]
fn fft_2d<'py>(
    py: Python<'py>,
    wavefront: PyReadonlyArray2<'py, Complex64>,
    forward: bool,
    normalization: Option<f64>,
    fftshift: bool,
) -> PyResult<Bound<'py, PyArray2<Complex64>>> {
    // Pythonic implementation of 2D FFT with optional fftshift and normalization.
    //     do_fft = cp.fft.fft2 if forward else cp.fft.ifft2
    // if normalization is None:
    //     normalization = 1./wavefront.shape[0] if forward else wavefront.shape[0]
    // wavefront = do_fft(wavefront)
    let input = wavefront.as_array();
    let (rows, cols) = input.dim();

    // Copy into a mutable buffer (row-major)
    let mut buf: Array2<Complex64> = input.to_owned();

    // Pre-FFT shift: ifftshift before forward FFT, fftshift before inverse
    if fftshift && forward {
        ifftshift_2d(&mut buf);
    }

    // Normalization
    let norm = normalization.unwrap_or_else(|| {
        if forward {
            1.0 / rows as f64
        } else {
            rows as f64
        }
    });

    // ------ The actual FFT steps ------
    // ndrustfft handles each axis efficiently without manual column copies.
    // Two explicit calls — one per axis — is the idiomatic 2D approach.
    let mut handler_ax0 = FftHandler::<f64>::new(rows);
    let mut handler_ax1 = FftHandler::<f64>::new(cols);

    if forward {
        ndfft_inplace_par(&mut buf, &mut handler_ax1, 1);
        ndfft_inplace_par(&mut buf, &mut handler_ax0, 0);
    } else {
        ndifft_inplace_par(&mut buf, &mut handler_ax1, 1);
        ndifft_inplace_par(&mut buf, &mut handler_ax0, 0);
    }

    // Post-FFT shift
    if fftshift && !forward {
        fftshift_2d(&mut buf);
    }

    // rustfft is always unnormalized in both directions:
    //   rustfft forward  == numpy.fft2             (no correction needed)
    //   rustfft inverse  == numpy.ifft2 * (N*M)    (need to divide by N*M)
    //
    // Poppy's default norm is 1/N for forward and N for inverse (applied on top of numpy).
    // Combined scales:
    //   forward:  (1/rows) * 1              = 1/rows
    //   inverse:  rows     * 1/(rows*cols)  = 1/cols
    let rustfft_correction = if forward { 1.0 } else { 1.0 / (rows as f64 * cols as f64) };
    let scale = Complex64::new(norm * rustfft_correction, 0.0);
    buf.mapv_inplace(|v| v * scale);

    Ok(buf.into_pyarray(py))
}

/// Matrix DFT following Soummer et al. 2007.
/// Drop-in replacement for poppy.matrixDFT.matrix_dft.
#[pyfunction]
#[pyo3(signature = (plane, nlam_d, npix, offset=None, inverse=false, centering="FFTSTYLE"))]
fn matrix_dft<'py>(
    py: Python<'py>,
    plane: PyReadonlyArray2<'py, Complex64>,
    nlam_d: (f64, f64),
    npix: (usize, usize),
    offset: Option<(f64, f64)>,
    inverse: bool,
    centering: &str,
) -> PyResult<Bound<'py, PyArray2<Complex64>>> {
    let input = plane.as_array();
    let (npup_y, npup_x) = input.dim();
    let (nlam_dy, nlam_dx) = nlam_d;
    let (npix_y, npix_x) = npix;

    let npix_xf = npix_x as f64;
    let npix_yf = npix_y as f64;
    let npup_xf = npup_x as f64;
    let npup_yf = npup_y as f64;

    let (dx, dy, du, dv) = if inverse {
        (
            nlam_dx / npup_xf,
            nlam_dy / npup_yf,
            1.0 / npix_xf,
            1.0 / npix_yf,
        )
    } else {
        (
            1.0 / npup_xf,
            1.0 / npup_yf,
            nlam_dx / npix_xf,
            nlam_dy / npix_yf,
        )
    };

    let centering_upper = centering.to_uppercase();

    let (offset_x, offset_y) = offset.unwrap_or((0.0, 0.0));

    // Build coordinate arrays
    let xs: Array1<f64> = Array1::from_iter((0..npup_x).map(|i| {
        let fi = i as f64;
        match centering_upper.as_str() {
            "FFTSTYLE" => (fi - npup_xf / 2.0) * dx,
            "ADJUSTABLE" => (fi - npup_xf / 2.0 - offset_x + 0.5) * dx,
            "SYMMETRIC" => (fi - npup_xf / 2.0 + 0.5) * dx,
            _ => (fi - npup_xf / 2.0) * dx,
        }
    }));

    let ys: Array1<f64> = Array1::from_iter((0..npup_y).map(|i| {
        let fi = i as f64;
        match centering_upper.as_str() {
            "FFTSTYLE" => (fi - npup_yf / 2.0) * dy,
            "ADJUSTABLE" => (fi - npup_yf / 2.0 - offset_y + 0.5) * dy,
            "SYMMETRIC" => (fi - npup_yf / 2.0 + 0.5) * dy,
            _ => (fi - npup_yf / 2.0) * dy,
        }
    }));

    let us: Array1<f64> = Array1::from_iter((0..npix_x).map(|i| {
        let fi = i as f64;
        match centering_upper.as_str() {
            "FFTSTYLE" => (fi - npix_xf / 2.0) * du,
            "ADJUSTABLE" => (fi - npix_xf / 2.0 - offset_x + 0.5) * du,
            "SYMMETRIC" => (fi - npix_xf / 2.0 + 0.5) * du,
            _ => (fi - npix_xf / 2.0) * du,
        }
    }));

    let vs: Array1<f64> = Array1::from_iter((0..npix_y).map(|i| {
        let fi = i as f64;
        match centering_upper.as_str() {
            "FFTSTYLE" => (fi - npix_yf / 2.0) * dv,
            "ADJUSTABLE" => (fi - npix_yf / 2.0 - offset_y + 0.5) * dv,
            "SYMMETRIC" => (fi - npix_yf / 2.0 + 0.5) * dv,
            _ => (fi - npix_yf / 2.0) * dv,
        }
    }));

    // Build exponential matrices and compute DFT via matrix triple product
    // expYV: (npix_y, npup_y)  expXU: (npup_x, npix_x)
    let sign = if inverse { -1.0 } else { 1.0 };

    // expYV[v, y] = exp(sign * 2πi * Vs[v] * Ys[y])
    let mut exp_yv = Array2::<Complex64>::zeros((npix_y, npup_y));
    exp_yv
        .as_slice_mut()
        .unwrap()
        .par_chunks_mut(npup_y)
        .enumerate()
        .for_each(|(v, row)| {
            for (y, cell) in row.iter_mut().enumerate() {
                let phase = sign * 2.0 * PI * vs[v] * ys[y];
                *cell = Complex64::new(phase.cos(), phase.sin());
            }
        });

    // expXU[x, u] = exp(sign * 2πi * Xs[x] * Us[u])
    let mut exp_xu = Array2::<Complex64>::zeros((npup_x, npix_x));
    exp_xu
        .as_slice_mut()
        .unwrap()
        .par_chunks_mut(npix_x)
        .enumerate()
        .for_each(|(x, row)| {
            for (u, cell) in row.iter_mut().enumerate() {
                let phase = sign * 2.0 * PI * xs[x] * us[u];
                *cell = Complex64::new(phase.cos(), phase.sin());
            }
        });

    // t1 = expYV @ plane  -> (npix_y, npup_x)
    let t1 = exp_yv.dot(&input);
    // result = t1 @ expXU -> (npix_y, npix_x)
    let mut result = t1.dot(&exp_xu);

    let norm_coeff =
        ((nlam_dy * nlam_dx) / (npup_yf * npup_xf * npix_yf * npix_xf)).sqrt();
    let norm = Complex64::new(norm_coeff, 0.0);
    result.mapv_inplace(|v| v * norm);

    Ok(result.into_pyarray(py))
}

/// Python module definition
#[pymodule]
fn poppy_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fft_2d, m)?)?;
    m.add_function(wrap_pyfunction!(matrix_dft, m)?)?;
    Ok(())
}

// --- Helper functions ---

fn fftshift_2d(arr: &mut Array2<Complex64>) {
    let (rows, cols) = arr.dim();
    let half_r = rows / 2;
    let half_c = cols / 2;

    // Swap quadrants
    for r in 0..half_r {
        for c in 0..cols {
            let target_r = r + half_r;
            let target_c = (c + half_c) % cols;
            let tmp = arr[[r, c]];
            arr[[r, c]] = arr[[target_r, target_c]];
            arr[[target_r, target_c]] = tmp;
        }
    }
}

fn ifftshift_2d(arr: &mut Array2<Complex64>) {
    let (rows, cols) = arr.dim();
    let half_r = (rows + 1) / 2;
    let half_c = (cols + 1) / 2;

    // For even arrays, ifftshift == fftshift, but for odd arrays they differ
    let mut tmp = Array2::<Complex64>::zeros((rows, cols));
    for r in 0..rows {
        for c in 0..cols {
            let nr = (r + half_r) % rows;
            let nc = (c + half_c) % cols;
            tmp[[nr, nc]] = arr[[r, c]];
        }
    }
    arr.assign(&tmp);
}
