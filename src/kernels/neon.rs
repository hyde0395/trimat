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
        // Tiled GEMM: vectorize over the N (output column) dimension instead of
        // gathering one x column at a time. x is (K×N) row-major, so a whole
        // weight column contributes ±x[col, :] (contiguous) to the output row.
        y.par_chunks_mut(n).enumerate().for_each(|(row, row_out)| {
            neon_gemm_row(w, row, x, n, row_out);
            let scale = w.row_scale(row);
            for v in row_out.iter_mut() {
                *v *= scale;
            }
        });
    }
}

/// Accumulate one GEMM output row (length `n`) by streaming nonzero weight
/// columns across the contiguous x rows, 4-wide NEON. `row_out` starts zeroed;
/// scale is applied by the caller.
#[cfg(target_arch = "aarch64")]
fn neon_gemm_row(w: &TernaryTensor, row: usize, x: &[f32], n: usize, row_out: &mut [f32]) {
    use std::arch::aarch64::*;
    unsafe {
        let chunks = n / 4;
        for col in 0..w.cols {
            let i  = row * w.cols + col;
            let nz = (w.nonzero[i / 8] >> (i % 8)) & 1;
            if nz == 0 { continue; }
            let sg = (w.sign[i / 8] >> (i % 8)) & 1;
            let xrow = &x[col * n..col * n + n];

            if sg == 0 {
                for c in 0..chunks {
                    let off = c * 4;
                    let a = vld1q_f32(row_out.as_ptr().add(off));
                    let b = vld1q_f32(xrow.as_ptr().add(off));
                    vst1q_f32(row_out.as_mut_ptr().add(off), vaddq_f32(a, b));
                }
                for j in (chunks * 4)..n { row_out[j] += xrow[j]; }
            } else {
                for c in 0..chunks {
                    let off = c * 4;
                    let a = vld1q_f32(row_out.as_ptr().add(off));
                    let b = vld1q_f32(xrow.as_ptr().add(off));
                    vst1q_f32(row_out.as_mut_ptr().add(off), vsubq_f32(a, b));
                }
                for j in (chunks * 4)..n { row_out[j] -= xrow[j]; }
            }
        }
    }
}

/// Scalar fallback for the tiled GEMM row on non-aarch64 targets.
#[cfg(not(target_arch = "aarch64"))]
fn neon_gemm_row(w: &TernaryTensor, row: usize, x: &[f32], n: usize, row_out: &mut [f32]) {
    for col in 0..w.cols {
        match w.get(row, col) {
            1  => for j in 0..n { row_out[j] += x[col * n + j]; },
            -1 => for j in 0..n { row_out[j] -= x[col * n + j]; },
            _  => {}
        }
    }
}

/// Compute dot product of ternary row `row` with float slice `x` using NEON on
/// aarch64, falling back to scalar on other targets.
#[cfg(target_arch = "aarch64")]
fn neon_dot_row(w: &TernaryTensor, row: usize, x: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    unsafe {
        let cols = w.cols;
        let nz = &w.nonzero;
        let sg = &w.sign;
        let nlen = nz.len();
        let mut acc = vdupq_n_f32(0.0);

        // Lane k tests bit k of a nibble; used to expand 4 packed bits into 4
        // all-ones / all-zero lane masks entirely in registers (no stack
        // round-trip, no per-element branch).
        let sel = vld1q_u32([1u32, 2, 4, 8].as_ptr());

        let chunks = cols / 4;
        for chunk in 0..chunks {
            let base = row * cols + chunk * 4; // global bit index of these 4 weights
            let byte = base >> 3;
            let off  = base & 7;

            // Read the (up to 2) bytes covering this nibble and shift it down.
            let hi = if byte + 1 < nlen { 1usize } else { 0usize };
            let nz_bits = ((nz[byte] as u16) | ((nz[byte + hi] as u16) << 8)) >> off;
            let sg_bits = ((sg[byte] as u16) | ((sg[byte + hi] as u16) << 8)) >> off;
            let nz4 = (nz_bits & 0xF) as u32;
            let sg4 = (sg_bits & 0xF) as u32;

            let xv = vld1q_f32(x.as_ptr().add(chunk * 4));
            let xu = vreinterpretq_u32_f32(xv);

            // vtstq: lane = all-ones where the corresponding bit is set.
            let nz_mask = vtstq_u32(vdupq_n_u32(nz4), sel); // weight != 0
            let sg_mask = vtstq_u32(vdupq_n_u32(sg4), sel); // weight  < 0
            let pos_mask = vbicq_u32(nz_mask, sg_mask);     // nz & !sg  -> +1
            let neg_mask = vandq_u32(nz_mask, sg_mask);     // nz &  sg  -> -1

            let pos_vals = vreinterpretq_f32_u32(vandq_u32(xu, pos_mask));
            let neg_vals = vreinterpretq_f32_u32(vandq_u32(xu, neg_mask));
            // +x on positive lanes, -x on negative lanes, 0 elsewhere.
            acc = vaddq_f32(acc, vsubq_f32(pos_vals, neg_vals));
        }

        let mut result = vaddvq_f32(acc);

        // Handle remaining columns (tail) with scalar.
        for col in (chunks * 4)..cols {
            let i = row * cols + col;
            let nzb = (nz[i / 8] >> (i % 8)) & 1;
            let sgb = (sg[i / 8] >> (i % 8)) & 1;
            if nzb == 1 {
                if sgb == 0 { result += x[col]; }
                else         { result -= x[col]; }
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
    fn test_neon_matches_scalar_gemv_unaligned_cols() {
        // cols not a multiple of 8 exercises the straddling two-byte nibble read
        // and the scalar tail in the vectorized decode.
        for &cols in &[13usize, 19, 23, 31] {
            let rows = 7;
            let data: Vec<f32> =
                (0..rows * cols).map(|i| ((i * 5) % 3) as f32 - 1.0).collect();
            let w = make_tensor(&data, rows, cols);
            let x: Vec<f32> = (0..cols).map(|i| (i as f32).cos()).collect();

            let mut y_scalar = vec![0.0f32; rows];
            let mut y_neon = vec![0.0f32; rows];
            Scalar.gemv(&w, &x, &mut y_scalar);
            Neon.gemv(&w, &x, &mut y_neon);

            for i in 0..rows {
                assert!((y_scalar[i] - y_neon[i]).abs() < 1e-3,
                    "cols={} row {}: scalar={} neon={}", cols, i, y_scalar[i], y_neon[i]);
            }
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

    #[test]
    fn test_neon_matches_scalar_gemm_large() {
        // Larger M with odd cols and an N tail (n % 4 != 0).
        let (rows, cols, n) = (20usize, 19usize, 10usize);
        let data: Vec<f32> =
            (0..rows * cols).map(|i| ((i * 7) % 3) as f32 - 1.0).collect();
        let w = make_tensor(&data, rows, cols);
        let x: Vec<f32> = (0..cols * n).map(|i| (i as f32 * 0.05).sin()).collect();

        let mut y_scalar = vec![0.0f32; rows * n];
        let mut y_neon = vec![0.0f32; rows * n];
        Scalar.gemm(&w, &x, n, &mut y_scalar);
        Neon.gemm(&w, &x, n, &mut y_neon);

        for i in 0..rows * n {
            assert!((y_scalar[i] - y_neon[i]).abs() < 1e-3,
                "elem {}: scalar={} neon={}", i, y_scalar[i], y_neon[i]);
        }
    }
}
