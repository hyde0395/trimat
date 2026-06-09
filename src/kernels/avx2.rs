use rayon::prelude::*;

use crate::kernels::Kernel;
use crate::tensor::TernaryTensor;

pub struct Avx2;

impl Kernel for Avx2 {
    fn gemv(&self, w: &TernaryTensor, x: &[f32], y: &mut [f32]) {
        y.par_iter_mut().enumerate().for_each(|(row, out)| {
            // Safety: Avx2 is only constructed when dispatch detects AVX2 at runtime.
            *out = unsafe { avx2_dot_row(w, row, x) } * w.row_scale(row);
        });
    }

    fn gemm(&self, w: &TernaryTensor, x: &[f32], n: usize, y: &mut [f32]) {
        // Tiled GEMM: vectorize over the N (output column) dimension.
        // x is (K×N) row-major, so x[col*n .. col*n+n] is contiguous and a whole
        // weight column contributes ±x[col, :] to the output row. This streams x
        // contiguously instead of gathering one column at a time.
        y.par_chunks_mut(n).enumerate().for_each(|(row, row_out)| {
            // Safety: Avx2 is only constructed when dispatch detects AVX2 at runtime.
            unsafe { avx2_gemm_row(w, row, x, n, row_out) };
            let scale = w.row_scale(row);
            for v in row_out.iter_mut() {
                *v *= scale;
            }
        });
    }
}

/// Dot product of ternary row `row` with float slice `x`, 8-wide AVX2.
/// Builds per-lane sign masks, applies them to x, and horizontally sums.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2_dot_row(w: &TernaryTensor, row: usize, x: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let cols = w.cols;
    let mut acc = _mm256_setzero_ps();

    // Process 8 elements at a time with AVX2.
    let chunks = cols / 8;
    for chunk in 0..chunks {
        let base = row * cols + chunk * 8;

        // Load x[col..col+8].
        let xv = _mm256_loadu_ps(x.as_ptr().add(chunk * 8));

        // Build ternary masks: +1 → keep, -1 → negate, 0 → zero.
        let mut pos = [0u32; 8];
        let mut neg = [0u32; 8];
        for k in 0..8 {
            let i = base + k;
            let nz = (w.nonzero[i / 8] >> (i % 8)) & 1;
            let sg = (w.sign[i / 8]    >> (i % 8)) & 1;
            if nz == 1 {
                if sg == 0 { pos[k] = 0xFFFF_FFFF; }
                else        { neg[k] = 0xFFFF_FFFF; }
            }
        }

        let pos_mask = _mm256_loadu_si256(pos.as_ptr() as *const __m256i);
        let neg_mask = _mm256_loadu_si256(neg.as_ptr() as *const __m256i);

        // pos_vals = x where pos, else 0
        let pos_vals = _mm256_and_ps(xv, _mm256_castsi256_ps(pos_mask));
        // neg_vals = -x where neg, else 0
        let neg_vals = _mm256_and_ps(xv, _mm256_castsi256_ps(neg_mask));
        let neg_vals = _mm256_sub_ps(_mm256_setzero_ps(), neg_vals);

        acc = _mm256_add_ps(acc, _mm256_add_ps(pos_vals, neg_vals));
    }

    // Horizontal sum of the 8 lanes.
    let mut tmp = [0.0f32; 8];
    _mm256_storeu_ps(tmp.as_mut_ptr(), acc);
    let mut result = tmp.iter().sum::<f32>();

    // Handle remaining columns (tail) with scalar.
    for col in (chunks * 8)..cols {
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

/// Accumulate one GEMM output row (length `n`) by streaming nonzero weight
/// columns across the contiguous x rows, 8-wide AVX2. `row_out` is pre-zeroed
/// by the Vec allocation; scale is applied by the caller.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2_gemm_row(
    w: &TernaryTensor, row: usize, x: &[f32], n: usize, row_out: &mut [f32],
) {
    use std::arch::x86_64::*;

    let chunks = n / 8;
    for col in 0..w.cols {
        let i  = row * w.cols + col;
        let nz = (w.nonzero[i / 8] >> (i % 8)) & 1;
        if nz == 0 { continue; }
        let sg = (w.sign[i / 8] >> (i % 8)) & 1;
        let xrow = &x[col * n..col * n + n];

        if sg == 0 {
            // acc += x[col, :]
            for c in 0..chunks {
                let off = c * 8;
                let a = _mm256_loadu_ps(row_out.as_ptr().add(off));
                let b = _mm256_loadu_ps(xrow.as_ptr().add(off));
                _mm256_storeu_ps(row_out.as_mut_ptr().add(off), _mm256_add_ps(a, b));
            }
            for j in (chunks * 8)..n { row_out[j] += xrow[j]; }
        } else {
            // acc -= x[col, :]
            for c in 0..chunks {
                let off = c * 8;
                let a = _mm256_loadu_ps(row_out.as_ptr().add(off));
                let b = _mm256_loadu_ps(xrow.as_ptr().add(off));
                _mm256_storeu_ps(row_out.as_mut_ptr().add(off), _mm256_sub_ps(a, b));
            }
            for j in (chunks * 8)..n { row_out[j] -= xrow[j]; }
        }
    }
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
    fn test_avx2_matches_scalar_gemv() {
        let data: Vec<f32> = (0..128).map(|i| (i as f32 % 3.0) - 1.0).collect();
        let w = make_tensor(&data, 8, 16);
        let x: Vec<f32> = (0..16).map(|i| i as f32).collect();

        let mut y_scalar = vec![0.0f32; 8];
        let mut y_avx2   = vec![0.0f32; 8];
        Scalar.gemv(&w, &x, &mut y_scalar);
        Avx2.gemv(&w, &x, &mut y_avx2);

        for i in 0..8 {
            assert!((y_scalar[i] - y_avx2[i]).abs() < 1e-3,
                "row {}: scalar={} avx2={}", i, y_scalar[i], y_avx2[i]);
        }
    }

    #[test]
    fn test_avx2_matches_scalar_gemm() {
        let data: Vec<f32> = (0..64).map(|i| (i as f32 % 3.0) - 1.0).collect();
        let w = make_tensor(&data, 4, 16);
        let x: Vec<f32> = (0..48).map(|i| i as f32 * 0.1).collect();

        let mut y_scalar = vec![0.0f32; 12];
        let mut y_avx2   = vec![0.0f32; 12];
        Scalar.gemm(&w, &x, 3, &mut y_scalar);
        Avx2.gemm(&w, &x, 3, &mut y_avx2);

        for i in 0..12 {
            assert!((y_scalar[i] - y_avx2[i]).abs() < 1e-3,
                "element {}: scalar={} avx2={}", i, y_scalar[i], y_avx2[i]);
        }
    }

    #[test]
    fn test_avx2_gemm_wide_n() {
        // N=10 exercises both the 8-wide vector body and the scalar tail.
        let data: Vec<f32> = (0..96).map(|i| ((i * 7) % 3) as f32 - 1.0).collect();
        let w = make_tensor(&data, 6, 16);
        let x: Vec<f32> = (0..160).map(|i| (i as f32).sin()).collect();

        let mut y_scalar = vec![0.0f32; 60];
        let mut y_avx2   = vec![0.0f32; 60];
        Scalar.gemm(&w, &x, 10, &mut y_scalar);
        Avx2.gemm(&w, &x, 10, &mut y_avx2);

        for i in 0..60 {
            assert!((y_scalar[i] - y_avx2[i]).abs() < 1e-3,
                "element {}: scalar={} avx2={}", i, y_scalar[i], y_avx2[i]);
        }
    }
}
