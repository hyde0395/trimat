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

    fn qgemv(&self, w: &TernaryTensor, x_q: &[i8], x_scale: f32, y: &mut [f32]) {
        y.par_iter_mut().enumerate().for_each(|(row, out)| {
            // Safety: Avx2 is only constructed when dispatch detects AVX2 at runtime.
            *out = unsafe { avx2_qdot_row(w, row, x_q) } as f32 * w.row_scale(row) * x_scale;
        });
    }

    fn qgemm(
        &self, w: &TernaryTensor, x_q: &[i8], x_scale: &[f32], n: usize, y: &mut [f32],
    ) {
        y.par_chunks_mut(n).enumerate().for_each(|(row, row_out)| {
            let mut acc = vec![0i32; n];
            // Safety: Avx2 is only constructed when dispatch detects AVX2 at runtime.
            unsafe { avx2_qgemm_accumulate(w, row, x_q, n, &mut acc) };
            let ws = w.row_scale(row);
            for j in 0..n {
                row_out[j] = ws * x_scale[j] * acc[j] as f32;
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
    let nz = &w.nonzero;
    let sg = &w.sign;
    let nlen = nz.len();
    let mut acc = _mm256_setzero_ps();

    // Lane k tests bit k of a byte; used to expand 8 packed bits into 8
    // all-ones / all-zero lane masks in registers (no stack round-trip,
    // no per-element branch).
    let sel = _mm256_setr_epi32(1, 2, 4, 8, 16, 32, 64, 128);

    let chunks = cols / 8;
    for chunk in 0..chunks {
        let base = row * cols + chunk * 8; // global bit index of these 8 weights
        let byte = base >> 3;
        let off  = base & 7;

        // Read the (up to 2) bytes covering these 8 bits and shift them down.
        let hi = if byte + 1 < nlen { 1usize } else { 0usize };
        let nz_bits = (((nz[byte] as u16) | ((nz[byte + hi] as u16) << 8)) >> off) & 0xFF;
        let sg_bits = (((sg[byte] as u16) | ((sg[byte + hi] as u16) << 8)) >> off) & 0xFF;

        let xv = _mm256_loadu_ps(x.as_ptr().add(chunk * 8));

        // (val & bit) == bit  ->  all-ones where that bit is set.
        let nz_mask = _mm256_cmpeq_epi32(
            _mm256_and_si256(_mm256_set1_epi32(nz_bits as i32), sel), sel);
        let sg_mask = _mm256_cmpeq_epi32(
            _mm256_and_si256(_mm256_set1_epi32(sg_bits as i32), sel), sel);
        let pos_mask = _mm256_andnot_si256(sg_mask, nz_mask); // nz & !sg -> +1
        let neg_mask = _mm256_and_si256(nz_mask, sg_mask);    // nz &  sg -> -1

        let pos_vals = _mm256_and_ps(xv, _mm256_castsi256_ps(pos_mask));
        let neg_vals = _mm256_and_ps(xv, _mm256_castsi256_ps(neg_mask));
        // +x on positive lanes, -x on negative lanes, 0 elsewhere.
        acc = _mm256_add_ps(acc, _mm256_sub_ps(pos_vals, neg_vals));
    }

    // Horizontal sum of the 8 lanes.
    let mut tmp = [0.0f32; 8];
    _mm256_storeu_ps(tmp.as_mut_ptr(), acc);
    let mut result = tmp.iter().sum::<f32>();

    // Handle remaining columns (tail) with scalar.
    for col in (chunks * 8)..cols {
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

/// Integer dot of ternary row `row` with an int8 activation slice, 32 lanes per
/// iteration on AVX2. Returns the i32 sum Σ_k W[m,k]·x_q[k].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2_qdot_row(w: &TernaryTensor, row: usize, x_q: &[i8]) -> i32 {
    use std::arch::x86_64::*;

    let cols = w.cols;
    let nz = &w.nonzero;
    let sg = &w.sign;
    let nlen = nz.len();
    let mut acc = _mm256_setzero_si256();

    // Per-lane bit selector (0x80 = -128 as i8) and a shuffle that broadcasts
    // byte (lane/8) of the packed 32-bit mask into each group of 8 lanes.
    let sel = _mm256_setr_epi8(
        1, 2, 4, 8, 16, 32, 64, -128, 1, 2, 4, 8, 16, 32, 64, -128,
        1, 2, 4, 8, 16, 32, 64, -128, 1, 2, 4, 8, 16, 32, 64, -128,
    );
    let bcast = _mm256_setr_epi8(
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
        2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
    );
    let ones_u8 = _mm256_set1_epi8(1);
    let ones_i16 = _mm256_set1_epi16(1);

    let chunks = cols / 32;
    for chunk in 0..chunks {
        let base = row * cols + chunk * 32;
        let byte = base >> 3;
        let off  = base & 7;

        // 32 bits at `off` span up to 5 bytes; read (guarded) and shift down.
        let rd = |b: &[u8]| -> u32 {
            let mut acc: u64 = 0;
            for t in 0..5 {
                if byte + t < nlen { acc |= (b[byte + t] as u64) << (8 * t); }
            }
            ((acc >> off) & 0xFFFF_FFFF) as u32
        };
        let nz32 = rd(nz);
        let sg32 = rd(sg);

        let nz_bytes = _mm256_shuffle_epi8(_mm256_set1_epi32(nz32 as i32), bcast);
        let sg_bytes = _mm256_shuffle_epi8(_mm256_set1_epi32(sg32 as i32), bcast);
        let nz_mask = _mm256_cmpeq_epi8(_mm256_and_si256(nz_bytes, sel), sel);
        let sg_mask = _mm256_cmpeq_epi8(_mm256_and_si256(sg_bytes, sel), sel);

        let xq  = _mm256_loadu_si256(x_q.as_ptr().add(chunk * 32) as *const __m256i);
        let neg = _mm256_sub_epi8(_mm256_setzero_si256(), xq);
        // -x_q where negative (mask MSB set), x_q elsewhere; then zero where W==0.
        let signed = _mm256_blendv_epi8(xq, neg, sg_mask);
        let val = _mm256_and_si256(signed, nz_mask);

        // Sum 32×i8 → 8×i32: pairwise add via maddubs(1,·) then madd(·,1).
        let t16 = _mm256_maddubs_epi16(ones_u8, val);
        let t32 = _mm256_madd_epi16(t16, ones_i16);
        acc = _mm256_add_epi32(acc, t32);
    }

    let mut tmp = [0i32; 8];
    _mm256_storeu_si256(tmp.as_mut_ptr() as *mut __m256i, acc);
    let mut result: i32 = tmp.iter().sum();

    // Tail columns (cols % 32) accumulated scalar-wise.
    for col in (chunks * 32)..cols {
        let i = row * cols + col;
        let nzb = (nz[i / 8] >> (i % 8)) & 1;
        if nzb == 1 {
            let sgb = (sg[i / 8] >> (i % 8)) & 1;
            if sgb == 0 { result += x_q[col] as i32; }
            else         { result -= x_q[col] as i32; }
        }
    }
    result
}

/// Accumulate one GEMM output row's i32 sums by streaming nonzero weight
/// columns over the contiguous int8 Xq rows, widening i8→i32, 8 cols/iter.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2_qgemm_accumulate(w: &TernaryTensor, row: usize, x_q: &[i8], n: usize, acc: &mut [i32]) {
    use std::arch::x86_64::*;

    let cols = w.cols;
    let chunks = n / 8;
    for k in 0..cols {
        let i  = row * cols + k;
        let nz = (w.nonzero[i / 8] >> (i % 8)) & 1;
        if nz == 0 { continue; }
        let sg = (w.sign[i / 8] >> (i % 8)) & 1;
        let xrow = x_q.as_ptr().add(k * n);

        for c in 0..chunks {
            let off = c * 8;
            let v8 = _mm_loadl_epi64(xrow.add(off) as *const __m128i);
            let wv = _mm256_cvtepi8_epi32(v8);
            let ap = acc.as_mut_ptr().add(off) as *mut __m256i;
            let cur = _mm256_loadu_si256(ap as *const __m256i);
            let nv = if sg == 0 { _mm256_add_epi32(cur, wv) } else { _mm256_sub_epi32(cur, wv) };
            _mm256_storeu_si256(ap, nv);
        }
        if sg == 0 {
            for j in (chunks * 8)..n { acc[j] += *xrow.add(j) as i32; }
        } else {
            for j in (chunks * 8)..n { acc[j] -= *xrow.add(j) as i32; }
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
    fn test_avx2_matches_scalar_gemm_large() {
        // Larger M with odd cols and an N tail (n % 8 != 0).
        let (rows, cols, n) = (20usize, 19usize, 10usize);
        let data: Vec<f32> =
            (0..rows * cols).map(|i| ((i * 7) % 3) as f32 - 1.0).collect();
        let w = make_tensor(&data, rows, cols);
        let x: Vec<f32> = (0..cols * n).map(|i| (i as f32 * 0.05).sin()).collect();

        let mut y_scalar = vec![0.0f32; rows * n];
        let mut y_avx2 = vec![0.0f32; rows * n];
        Scalar.gemm(&w, &x, n, &mut y_scalar);
        Avx2.gemm(&w, &x, n, &mut y_avx2);

        for i in 0..rows * n {
            assert!((y_scalar[i] - y_avx2[i]).abs() < 1e-3,
                "elem {}: scalar={} avx2={}", i, y_scalar[i], y_avx2[i]);
        }
    }

    #[test]
    fn test_avx2_matches_scalar_gemv_unaligned_cols() {
        // cols not a multiple of 8 exercises the straddling two-byte read and
        // the scalar tail in the vectorized decode.
        for &cols in &[13usize, 19, 23, 31] {
            let rows = 7;
            let data: Vec<f32> =
                (0..rows * cols).map(|i| ((i * 5) % 3) as f32 - 1.0).collect();
            let w = make_tensor(&data, rows, cols);
            let x: Vec<f32> = (0..cols).map(|i| (i as f32).cos()).collect();

            let mut y_scalar = vec![0.0f32; rows];
            let mut y_avx2 = vec![0.0f32; rows];
            Scalar.gemv(&w, &x, &mut y_scalar);
            Avx2.gemv(&w, &x, &mut y_avx2);

            for i in 0..rows {
                assert!((y_scalar[i] - y_avx2[i]).abs() < 1e-3,
                    "cols={} row {}: scalar={} avx2={}", cols, i, y_scalar[i], y_avx2[i]);
            }
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

    #[test]
    fn test_avx2_qgemv_matches_scalar() {
        use crate::quantize;
        // Cover aligned (32-multiple) and unaligned cols (straddle + tail).
        for &cols in &[32usize, 40, 64, 77, 128] {
            let rows = 9;
            let data: Vec<f32> =
                (0..rows * cols).map(|i| ((i * 5) % 3) as f32 - 1.0).collect();
            let w = make_tensor(&data, rows, cols);
            let xf: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.3).sin() * 4.0).collect();
            let (x_q, x_scale) = quantize::quantize_act(&xf);

            let mut y_scalar = vec![0.0f32; rows];
            let mut y_avx2 = vec![0.0f32; rows];
            Scalar.qgemv(&w, &x_q, x_scale, &mut y_scalar);
            Avx2.qgemv(&w, &x_q, x_scale, &mut y_avx2);

            for i in 0..rows {
                assert!((y_scalar[i] - y_avx2[i]).abs() < 1e-3,
                    "cols={} row {}: scalar={} avx2={}", cols, i, y_scalar[i], y_avx2[i]);
            }
        }
    }

    #[test]
    fn test_avx2_qgemm_matches_scalar() {
        use crate::quantize;
        for &(rows, cols, n) in &[(9usize, 64usize, 8usize), (12, 40, 10), (8, 77, 16)] {
            let data: Vec<f32> =
                (0..rows * cols).map(|i| ((i * 7) % 3) as f32 - 1.0).collect();
            let w = make_tensor(&data, rows, cols);
            let xf: Vec<f32> = (0..cols * n).map(|i| (i as f32 * 0.07).sin() * 3.0).collect();
            let (x_q, x_scale) = quantize::quantize_act_2d(&xf, cols, n);

            let mut y_scalar = vec![0.0f32; rows * n];
            let mut y_avx2 = vec![0.0f32; rows * n];
            Scalar.qgemm(&w, &x_q, &x_scale, n, &mut y_scalar);
            Avx2.qgemm(&w, &x_q, &x_scale, n, &mut y_avx2);

            for i in 0..rows * n {
                assert!((y_scalar[i] - y_avx2[i]).abs() < 1e-3,
                    "({},{},{}) elem {}: scalar={} avx2={}",
                    rows, cols, n, i, y_scalar[i], y_avx2[i]);
            }
        }
    }
}
