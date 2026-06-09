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
}
