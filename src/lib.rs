use pyo3::prelude::*;

mod dispatch;
mod kernels;
mod pack;
mod quantize;
mod tensor;

#[pymodule]
fn _trimat(m: &Bound<'_, PyModule>) -> PyResult<()> {
    Ok(())
}
