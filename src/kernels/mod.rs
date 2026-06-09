pub mod scalar;

#[cfg(target_arch = "aarch64")]
pub mod neon;

#[cfg(target_arch = "x86_64")]
pub mod avx2;

use rayon::prelude::*;

use crate::tensor::TernaryTensor;

pub trait Kernel: Send + Sync {
    /// Compute w(M×K) · x(K) → y(M), applying scale.
    fn gemv(&self, w: &TernaryTensor, x: &[f32], y: &mut [f32]);
    /// Compute w(M×K) · X(K×N row-major) → Y(M×N row-major), applying scale.
    fn gemm(&self, w: &TernaryTensor, x: &[f32], n: usize, y: &mut [f32]);

    /// BitNet-style ternary-weight × int8-activation GEMV:
    /// `y[m] = w_scale[m] · x_scale · Σ_k W[m,k]·x_q[k]`, accumulated in i32.
    /// The default is a parallel scalar reference; SIMD kernels override it.
    fn qgemv(&self, w: &TernaryTensor, x_q: &[i8], x_scale: f32, y: &mut [f32]) {
        y.par_iter_mut().enumerate().for_each(|(m, out)| {
            let mut acc: i32 = 0;
            for k in 0..w.cols {
                match w.get(m, k) {
                     1 => acc += x_q[k] as i32,
                    -1 => acc -= x_q[k] as i32,
                     _ => {}
                }
            }
            *out = w.row_scale(m) * x_scale * acc as f32;
        });
    }

    /// BitNet-style ternary-weight × int8-activation GEMM:
    /// `Y[m,j] = w_scale[m] · x_scale[j] · Σ_k W[m,k]·Xq[k,j]`, accumulated in
    /// i32. `x_q` is (K×N) row-major int8; `x_scale` has length N (per column).
    /// The default streams nonzero weight columns over the contiguous Xq rows
    /// into a per-row i32 accumulator; SIMD kernels override it.
    fn qgemm(
        &self, w: &TernaryTensor, x_q: &[i8], x_scale: &[f32], n: usize, y: &mut [f32],
    ) {
        y.par_chunks_mut(n).enumerate().for_each(|(row, row_out)| {
            let mut acc = vec![0i32; n];
            for k in 0..w.cols {
                let xrow = &x_q[k * n..k * n + n];
                match w.get(row, k) {
                    1  => for j in 0..n { acc[j] += xrow[j] as i32; },
                    -1 => for j in 0..n { acc[j] -= xrow[j] as i32; },
                    _  => {}
                }
            }
            let ws = w.row_scale(row);
            for j in 0..n {
                row_out[j] = ws * x_scale[j] * acc[j] as f32;
            }
        });
    }
}
