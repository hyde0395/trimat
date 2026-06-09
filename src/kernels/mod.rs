pub mod scalar;

#[cfg(target_arch = "aarch64")]
pub mod neon;

use crate::tensor::TernaryTensor;

pub trait Kernel: Send + Sync {
    /// Compute w(M×K) · x(K) → y(M), applying scale.
    fn gemv(&self, w: &TernaryTensor, x: &[f32], y: &mut [f32]);
    /// Compute w(M×K) · X(K×N row-major) → Y(M×N row-major), applying scale.
    fn gemm(&self, w: &TernaryTensor, x: &[f32], n: usize, y: &mut [f32]);
}
