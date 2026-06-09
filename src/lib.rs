use ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyDict};

mod dispatch;
mod kernels;
mod pack;
mod quantize;
mod tensor;

use tensor::TernaryTensor;

/// FP32 numpy 행렬(M×K)을 absmax 양자화 후 비트플레인으로 패킹한다.
#[pyfunction]
fn pack_tensor(py: Python<'_>, w: PyReadonlyArray2<'_, f32>) -> PyResult<TernaryTensor> {
    let shape = w.shape();
    let (rows, cols) = (shape[0], shape[1]);
    if rows == 0 || cols == 0 {
        return Err(PyValueError::new_err("weight matrix must be non-empty"));
    }
    let data: Vec<f32> = if w.is_c_contiguous() {
        w.as_slice()?.to_vec()
    } else {
        w.to_owned_array().as_slice().ok_or_else(|| PyValueError::new_err("non-contiguous array"))?.to_vec()
    };
    let (quantized, scale) = py.allow_threads(|| quantize::absmax_quantize(&data));
    let (nonzero, sign)    = pack::encode(&quantized);
    Ok(TernaryTensor::new(rows, cols, nonzero, sign, vec![scale]))
}

/// ternary 행렬 w(M×K) · 벡터 x(K) → 벡터 y(M).
#[pyfunction]
fn gemv<'py>(
    py: Python<'py>,
    w: &TernaryTensor,
    x: PyReadonlyArray1<'py, f32>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    if x.len() != w.cols {
        return Err(PyValueError::new_err(format!(
            "x length {} != tensor cols {}", x.len(), w.cols
        )));
    }
    let x_data: Vec<f32> = x.as_slice()?.to_vec();
    let mut y   = vec![0.0f32; w.rows];
    let kernel  = dispatch::best_kernel();
    py.allow_threads(|| kernel.gemv(w, &x_data, &mut y));
    Ok(Array1::from(y).into_pyarray(py))
}

/// ternary 행렬 w(M×K) · 행렬 X(K×N) → 행렬 Y(M×N). X shape: (K, N).
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
    let x_vec: Vec<f32> = if x.is_c_contiguous() {
        x.as_slice()?.to_vec()
    } else {
        x.to_owned_array().as_slice().ok_or_else(|| PyValueError::new_err("non-contiguous array"))?.to_vec()
    };
    let mut y  = vec![0.0f32; w.rows * n];
    let kernel = dispatch::best_kernel();
    py.allow_threads(|| kernel.gemm(w, &x_vec, n, &mut y));
    let arr = Array2::from_shape_vec((w.rows, n), y)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(arr.into_pyarray(py))
}

/// 현재 런타임 정보 반환 (backend, threads).
#[pyfunction]
fn cpu_features(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
    let info = dispatch::dispatch_info();
    let d    = PyDict::new(py);
    d.set_item("backend", info.backend)?;
    d.set_item("threads", info.threads)?;
    Ok(d.into_any())
}

#[pymodule]
fn _trimat(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<TernaryTensor>()?;
    m.add_function(wrap_pyfunction!(pack_tensor, m)?)?;
    m.add_function(wrap_pyfunction!(gemv, m)?)?;
    m.add_function(wrap_pyfunction!(gemm, m)?)?;
    m.add_function(wrap_pyfunction!(cpu_features, m)?)?;
    Ok(())
}
