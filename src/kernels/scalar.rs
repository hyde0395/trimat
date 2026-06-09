use rayon::prelude::*;

use crate::kernels::Kernel;
use crate::tensor::TernaryTensor;

pub struct Scalar;

impl Kernel for Scalar {
    fn gemv(&self, w: &TernaryTensor, x: &[f32], y: &mut [f32]) {
        y.par_iter_mut().enumerate().for_each(|(row, out)| {
            let mut acc = 0.0f32;
            for col in 0..w.cols {
                match w.get(row, col) {
                     1 => acc += x[col],
                    -1 => acc -= x[col],
                     _ => {}
                }
            }
            *out = acc * w.row_scale(row);
        });
    }

    fn gemm(&self, w: &TernaryTensor, x: &[f32], n: usize, y: &mut [f32]) {
        // x: (K×N) row-major → x[col*n + j]
        // y: (M×N) row-major → y[row*n + j]
        y.par_chunks_mut(n).enumerate().for_each(|(row, row_out)| {
            for j in 0..n {
                let mut acc = 0.0f32;
                for col in 0..w.cols {
                    match w.get(row, col) {
                         1 => acc += x[col * n + j],
                        -1 => acc -= x[col * n + j],
                         _ => {}
                    }
                }
                row_out[j] = acc * w.row_scale(row);
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{pack, quantize};

    fn make_tensor(data: &[f32], rows: usize, cols: usize) -> TernaryTensor {
        let (q, scale) = quantize::absmax_quantize(data);
        let (nz, sg)   = pack::encode(&q);
        TernaryTensor::new(rows, cols, nz, sg, vec![scale])
    }

    #[test]
    fn test_gemv_identity() {
        let w = make_tensor(&[1.0, 0.0, 0.0, 1.0], 2, 2);
        let x = [3.0f32, 5.0];
        let mut y = vec![0.0f32; 2];
        Scalar.gemv(&w, &x, &mut y);
        assert!((y[0] - 3.0).abs() < 1e-4, "y[0]={}", y[0]);
        assert!((y[1] - 5.0).abs() < 1e-4, "y[1]={}", y[1]);
    }

    #[test]
    fn test_gemv_basic() {
        let w = make_tensor(&[1.0, 0.0, -1.0, -1.0, 1.0, 0.0], 2, 3);
        let x = [1.0f32, 2.0, 3.0];
        let mut y = vec![0.0f32; 2];
        Scalar.gemv(&w, &x, &mut y);
        assert!((y[0] - (-2.0)).abs() < 1e-4, "y[0]={}", y[0]);
        assert!((y[1] - 1.0).abs() < 1e-4,   "y[1]={}", y[1]);
    }

    #[test]
    fn test_gemv_all_zero_weight() {
        let w = make_tensor(&[0.0, 0.0, 0.0], 1, 3);
        let x = [1.0f32, 2.0, 3.0];
        let mut y = vec![99.0f32; 1];
        Scalar.gemv(&w, &x, &mut y);
        assert!((y[0] - 0.0).abs() < 1e-4);
    }

    #[test]
    fn test_gemm_identity() {
        let w = make_tensor(&[1.0, 0.0, 0.0, 1.0], 2, 2);
        let x = [1.0f32, 2.0, 3.0, 4.0];
        let mut y = vec![0.0f32; 4];
        Scalar.gemm(&w, &x, 2, &mut y);
        assert!((y[0] - 1.0).abs() < 1e-4, "Y[0,0]={}", y[0]);
        assert!((y[1] - 2.0).abs() < 1e-4, "Y[0,1]={}", y[1]);
        assert!((y[2] - 3.0).abs() < 1e-4, "Y[1,0]={}", y[2]);
        assert!((y[3] - 4.0).abs() < 1e-4, "Y[1,1]={}", y[3]);
    }

    #[test]
    fn test_gemm_basic() {
        let w = make_tensor(&[1.0, -1.0, 0.0], 1, 3);
        let x = [1.0f32, 0.0, 0.0, 1.0, 1.0, 1.0];
        let mut y = vec![0.0f32; 2];
        Scalar.gemm(&w, &x, 2, &mut y);
        assert!((y[0] - 1.0).abs()    < 1e-4, "Y[0,0]={}", y[0]);
        assert!((y[1] - (-1.0)).abs() < 1e-4, "Y[0,1]={}", y[1]);
    }
}
