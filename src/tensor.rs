use pyo3::prelude::*;

#[pyclass]
pub struct TernaryTensor {
    pub rows:    usize,
    pub cols:    usize,
    pub nonzero: Vec<u8>,
    pub sign:    Vec<u8>,
    /// len=1: per-tensor scale / len=rows: per-channel scale
    pub scale:   Vec<f32>,
}

impl TernaryTensor {
    pub fn new(
        rows: usize, cols: usize,
        nonzero: Vec<u8>, sign: Vec<u8>,
        scale: Vec<f32>,
    ) -> Self {
        assert!(scale.len() == 1 || scale.len() == rows,
            "scale.len() must be 1 or rows");
        Self { rows, cols, nonzero, sign, scale }
    }

    /// 요소 (row, col)의 ternary 값 {-1,0,1} 반환.
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> i8 {
        let i  = row * self.cols + col;
        let nz = (self.nonzero[i / 8] >> (i % 8)) & 1;
        let sg = (self.sign[i / 8]    >> (i % 8)) & 1;
        if nz == 0 { 0 } else if sg == 0 { 1 } else { -1 }
    }

    #[inline]
    pub fn row_scale(&self, row: usize) -> f32 {
        if self.scale.len() == 1 { self.scale[0] } else { self.scale[row] }
    }
}

#[pymethods]
impl TernaryTensor {
    #[getter]
    pub fn rows(&self) -> usize { self.rows }

    #[getter]
    pub fn cols(&self) -> usize { self.cols }

    fn __repr__(&self) -> String {
        let s = if self.scale.len() == 1 {
            format!("{:.4}", self.scale[0])
        } else {
            format!("per-channel[{}]", self.scale.len())
        };
        format!("TernaryTensor({}×{}, scale={})", self.rows, self.cols, s)
    }
}
