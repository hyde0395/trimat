use rayon::prelude::*;

use crate::kernels::Kernel;
use crate::tensor::TernaryTensor;

pub struct Neon;

impl Kernel for Neon {
    fn gemv(&self, w: &TernaryTensor, x: &[f32], y: &mut [f32]) {
        y.par_iter_mut().enumerate().for_each(|(row, out)| {
            *out = neon_dot_row(w, row, x) * w.row_scale(row);
        });
    }

    fn gemm(&self, w: &TernaryTensor, x: &[f32], n: usize, y: &mut [f32]) {
        y.par_chunks_mut(n).enumerate().for_each(|(row, row_out)| {
            let scale = w.row_scale(row);
            for j in 0..n {
                let col_x: Vec<f32> = (0..w.cols).map(|c| x[c * n + j]).collect();
                row_out[j] = neon_dot_row(w, row, &col_x) * scale;
            }
        });
    }
}

/// Compute dot product of ternary row `row` with float slice `x` using NEON on
/// aarch64, falling back to scalar on other targets.
#[cfg(target_arch = "aarch64")]
fn neon_dot_row(w: &TernaryTensor, row: usize, x: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    unsafe {
        let cols = w.cols;
        let mut acc = vdupq_n_f32(0.0);

        // Process 4 elements at a time with NEON.
        let chunks = cols / 4;
        for chunk in 0..chunks {
            let base = row * cols + chunk * 4;

            // Load x[col..col+4].
            let xv = vld1q_f32(x.as_ptr().add(chunk * 4));

            // Build ternary mask: +1 → keep, -1 → negate, 0 → zero.
            // nz bits select non-zero elements; sg bits select sign.
            let mut pos = [0u32; 4];
            let mut neg = [0u32; 4];
            for k in 0..4 {
                let i = base + k;
                let nz = (w.nonzero[i / 8] >> (i % 8)) & 1;
                let sg = (w.sign[i / 8]    >> (i % 8)) & 1;
                if nz == 1 {
                    if sg == 0 { pos[k] = 0xFFFF_FFFF; }
                    else        { neg[k] = 0xFFFF_FFFF; }
                }
            }

            let pos_mask = vld1q_u32(pos.as_ptr());
            let neg_mask = vld1q_u32(neg.as_ptr());

            // pos_contrib = x where pos, else 0
            let pos_vals = vreinterpretq_f32_u32(vandq_u32(
                vreinterpretq_u32_f32(xv), pos_mask));
            // neg_contrib = -x where neg, else 0
            let neg_vals = vreinterpretq_f32_u32(vandq_u32(
                vreinterpretq_u32_f32(xv), neg_mask));
            let neg_vals = vnegq_f32(neg_vals);

            acc = vaddq_f32(acc, vaddq_f32(pos_vals, neg_vals));
        }

        // Sum the 4 NEON lanes.
        let mut result = vaddvq_f32(acc);

        // Handle remaining columns (tail) with scalar.
        for col in (chunks * 4)..cols {
            let i = row * cols + col;
            let nz = (w.nonzero[i / 8] >> (i % 8)) & 1;
            let sg = (w.sign[i / 8]    >> (i % 8)) & 1;
            if nz == 1 {
                if sg == 0 { result += x[col]; }
                else        { result -= x[col]; }
            }
        }
        result
    }
}

/// Scalar fallback for non-aarch64 targets.
#[cfg(not(target_arch = "aarch64"))]
fn neon_dot_row(w: &TernaryTensor, row: usize, x: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    for col in 0..w.cols {
        match w.get(row, col) {
             1 => acc += x[col],
            -1 => acc -= x[col],
             _ => {}
        }
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{pack, quantize};
    use crate::kernels::scalar::Scalar;

    fn make_tensor(data: &[f32], rows: usize, cols: usize) -> TernaryTensor {
        let (q, scale) = quantize::absmax_quantize(data);
        let (nz, sg)   = pack::encode(&q);
        TernaryTensor::new(rows, cols, nz, sg, vec![scale])
    }

    #[test]
    fn test_neon_matches_scalar_gemv() {
        let data: Vec<f32> = (0..128).map(|i| (i as f32 % 3.0) - 1.0).collect();
        let w = make_tensor(&data, 8, 16);
        let x: Vec<f32> = (0..16).map(|i| i as f32).collect();

        let mut y_scalar = vec![0.0f32; 8];
        let mut y_neon   = vec![0.0f32; 8];
        Scalar.gemv(&w, &x, &mut y_scalar);
        Neon.gemv(&w, &x, &mut y_neon);

        for i in 0..8 {
            assert!((y_scalar[i] - y_neon[i]).abs() < 1e-3,
                "row {}: scalar={} neon={}", i, y_scalar[i], y_neon[i]);
        }
    }

    #[test]
    fn test_neon_matches_scalar_gemm() {
        let data: Vec<f32> = (0..64).map(|i| (i as f32 % 3.0) - 1.0).collect();
        let w = make_tensor(&data, 4, 16);
        let x: Vec<f32> = (0..48).map(|i| i as f32 * 0.1).collect();

        let mut y_scalar = vec![0.0f32; 12];
        let mut y_neon   = vec![0.0f32; 12];
        Scalar.gemm(&w, &x, 3, &mut y_scalar);
        Neon.gemm(&w, &x, 3, &mut y_neon);

        for i in 0..12 {
            assert!((y_scalar[i] - y_neon[i]).abs() < 1e-3,
                "element {}: scalar={} neon={}", i, y_scalar[i], y_neon[i]);
        }
    }
}
