pub mod scalar;

use crate::tensor::TernaryTensor;

pub trait Kernel: Send + Sync {
    /// w(M×K) · x(K) → y(M), scale 적용 포함
    fn gemv(&self, w: &TernaryTensor, x: &[f32], y: &mut [f32]);
    /// w(M×K) · X(K×N row-major) → Y(M×N row-major), scale 적용 포함
    fn gemm(&self, w: &TernaryTensor, x: &[f32], n: usize, y: &mut [f32]);
}
