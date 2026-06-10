// SIMD/bitplane kernels are inherently index-driven (a loop variable indexes
// both the packed weight planes and the float operands), so range-based loops
// are the natural idiom here rather than a lint to fix.
#![allow(clippy::needless_range_loop)]

use ndarray::{Array1, Array2};
use numpy::{
    IntoPyArray, PyArray1, PyArray2,
    PyReadonlyArray1, PyReadonlyArray2,
    PyArrayMethods, PyUntypedArrayMethods,
};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyDict};

pub mod dispatch;
pub mod kernels;
pub mod pack;
pub mod quantize;
pub mod tensor;

use tensor::TernaryTensor;

/// Quantize an FP32 numpy matrix (M×K) to ternary and pack into bitplanes.
///
/// `mode` selects the per-tensor scale: "absmax" (gamma = max|W|, default) or
/// "absmean" (gamma = mean|W|, the BitNet b1.58 weight formula). Real BitNet
/// checkpoints have large outliers that collapse absmax to ~0, so use "absmean"
/// for them.
#[pyfunction]
#[pyo3(signature = (w, mode = "absmax"))]
fn pack_tensor(
    _py: Python<'_>, w: PyReadonlyArray2<'_, f32>, mode: &str,
) -> PyResult<TernaryTensor> {
    let shape = w.shape();
    let (rows, cols) = (shape[0], shape[1]);
    if rows == 0 || cols == 0 {
        return Err(PyValueError::new_err("weight matrix must be non-empty"));
    }
    let data: Vec<f32> = if w.is_c_contiguous() {
        w.as_slice()?.to_vec()
    } else {
        w.to_owned_array().into_raw_vec_and_offset().0
    };
    let (quantized, scale) = match mode {
        "absmax"  => quantize::absmax_quantize(&data),
        "absmean" => quantize::absmean_quantize(&data),
        _ => return Err(PyValueError::new_err(
            "mode must be 'absmax' or 'absmean'",
        )),
    };
    let (nonzero, sign) = pack::encode(&quantized);
    Ok(TernaryTensor::new(rows, cols, nonzero, sign, vec![scale]))
}

/// Reconstruct the dense f32 weight (M×K) = ternary_code · scale.
///
/// For the PyTorch/BLAS prefill fallback (`F.linear`). O(M·K); the Python layer
/// calls this once and caches it, so the 2-bit packing is preserved as long as
/// only the decode (gemv) path is exercised. Returns a freshly allocated array
/// (a dense materialization inherently cannot be zero-copy).
#[pyfunction]
fn to_dense<'py>(
    py: Python<'py>, w: &TernaryTensor,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let (rows, cols) = (w.rows, w.cols);
    let mut data = vec![0.0f32; rows * cols];
    for r in 0..rows {
        let s = w.row_scale(r);
        let base = r * cols;
        for c in 0..cols {
            data[base + c] = w.get(r, c) as f32 * s;
        }
    }
    let arr = Array2::from_shape_vec((rows, cols), data)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(arr.into_pyarray(py))
}

/// Compute ternary matrix w(M×K) · vector x(K) → vector y(M).
#[pyfunction]
fn gemv<'py>(
    py: Python<'py>,
    w: &TernaryTensor,
    x: PyReadonlyArray1<'py, f32>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let xlen = x.shape()[0];
    if xlen != w.cols {
        return Err(PyValueError::new_err(format!(
            "x length {} != tensor cols {}", xlen, w.cols
        )));
    }
    // Borrow x directly when contiguous; only copy on the rare strided path.
    let x_owned;
    let x_slice: &[f32] = match x.as_slice() {
        Ok(s) => s,
        Err(_) => { x_owned = x.as_array().to_vec(); &x_owned }
    };
    let mut y = vec![0.0f32; w.rows];
    dispatch::kernel().gemv(w, x_slice, &mut y);
    Ok(Array1::from(y).into_pyarray(py))
}

/// BitNet-style GEMV: quantize x to int8 (per-tensor absmax) then compute
/// w(M×K) · x(K) → y(M) with integer accumulation. Lossy vs gemv (int8 x), but
/// avoids FP32 multiplies in the inner loop.
#[pyfunction]
fn qgemv<'py>(
    py: Python<'py>,
    w: &TernaryTensor,
    x: PyReadonlyArray1<'py, f32>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let xlen = x.shape()[0];
    if xlen != w.cols {
        return Err(PyValueError::new_err(format!(
            "x length {} != tensor cols {}", xlen, w.cols
        )));
    }
    let x_owned;
    let x_slice: &[f32] = match x.as_slice() {
        Ok(s) => s,
        Err(_) => { x_owned = x.as_array().to_vec(); &x_owned }
    };
    let (x_q, x_scale) = quantize::quantize_act(x_slice);
    let mut y = vec![0.0f32; w.rows];
    dispatch::kernel().qgemv(w, &x_q, x_scale, &mut y);
    Ok(Array1::from(y).into_pyarray(py))
}

/// Compute ternary matrix w(M×K) · matrix X(K×N) → matrix Y(M×N). X shape: (K, N).
#[pyfunction]
fn gemm<'py>(
    py: Python<'py>,
    w: &TernaryTensor,
    x: PyReadonlyArray2<'py, f32>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let x_shape = x.shape();
    if x_shape[0] != w.cols {
        return Err(PyValueError::new_err(format!(
            "x rows {} != tensor cols {}", x_shape[0], w.cols
        )));
    }
    let n = x_shape[1];
    // Borrow X directly when contiguous; only copy on the rare strided path.
    // This avoids copying the full K×N activation matrix on every call.
    let x_owned;
    let x_slice: &[f32] = if x.is_c_contiguous() {
        x.as_slice()?
    } else {
        x_owned = x.to_owned_array().into_raw_vec_and_offset().0;
        &x_owned
    };
    let mut y = vec![0.0f32; w.rows * n];
    dispatch::kernel().gemm(w, x_slice, n, &mut y);
    let arr = Array2::from_shape_vec((w.rows, n), y)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(arr.into_pyarray(py))
}

/// BitNet-style GEMM: quantize each column (token) of X to int8 (per-column
/// absmax) then compute w(M×K) · X(K×N) → Y(M×N) with integer accumulation.
/// Lossy vs gemm (int8 X), but avoids FP32 multiplies in the inner loop.
#[pyfunction]
fn qgemm<'py>(
    py: Python<'py>,
    w: &TernaryTensor,
    x: PyReadonlyArray2<'py, f32>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let x_shape = x.shape();
    if x_shape[0] != w.cols {
        return Err(PyValueError::new_err(format!(
            "x rows {} != tensor cols {}", x_shape[0], w.cols
        )));
    }
    let n = x_shape[1];
    let x_owned;
    let x_slice: &[f32] = if x.is_c_contiguous() {
        x.as_slice()?
    } else {
        x_owned = x.to_owned_array().into_raw_vec_and_offset().0;
        &x_owned
    };
    let (x_q, x_scale) = quantize::quantize_act_2d(x_slice, w.cols, n);
    let mut y = vec![0.0f32; w.rows * n];
    dispatch::kernel().qgemm(w, &x_q, &x_scale, n, &mut y);
    let arr = Array2::from_shape_vec((w.rows, n), y)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(arr.into_pyarray(py))
}

/// Return runtime info dict: {backend, threads}.
#[pyfunction]
fn cpu_features<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
    let info = dispatch::dispatch_info();
    let d = PyDict::new(py);
    d.set_item("backend", info.backend)?;
    d.set_item("threads", info.threads)?;
    Ok(d)
}

#[pymodule]
fn _trimat(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<TernaryTensor>()?;
    m.add_function(wrap_pyfunction!(pack_tensor, m)?)?;
    m.add_function(wrap_pyfunction!(to_dense, m)?)?;
    m.add_function(wrap_pyfunction!(gemv, m)?)?;
    m.add_function(wrap_pyfunction!(qgemv, m)?)?;
    m.add_function(wrap_pyfunction!(gemm, m)?)?;
    m.add_function(wrap_pyfunction!(qgemm, m)?)?;
    m.add_function(wrap_pyfunction!(cpu_features, m)?)?;
    Ok(())
}
