/// Quantize a FP32 slice to ternary {-1, 0, +1} using absmax scaling.
/// Returns (quantized, scale) where scale = max(|w|).
/// Reconstruction: w ≈ quantized * scale
pub fn absmax_quantize(w: &[f32]) -> (Vec<i8>, f32) {
    let max_abs = w.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    if max_abs == 0.0 {
        return (vec![0i8; w.len()], 1.0);
    }
    let scale = max_abs;
    let quantized = w
        .iter()
        .map(|v| (v / scale).round().clamp(-1.0, 1.0) as i8)
        .collect();
    (quantized, scale)
}

/// Quantize a FP32 slice to ternary {-1, 0, +1} using absmean scaling — the
/// BitNet b1.58 weight formula. Returns (quantized, scale) where
/// scale = mean(|w|). Unlike absmax, a single large outlier does not collapse
/// the rest of the matrix to zero, so real BitNet checkpoints are preserved.
/// Reconstruction: w ≈ quantized * scale. An all-zero input yields scale 1.0.
pub fn absmean_quantize(w: &[f32]) -> (Vec<i8>, f32) {
    if w.is_empty() {
        return (Vec::new(), 1.0);
    }
    let mean_abs = w.iter().map(|v| v.abs()).sum::<f32>() / w.len() as f32;
    if mean_abs == 0.0 {
        return (vec![0i8; w.len()], 1.0);
    }
    let scale = mean_abs;
    let quantized = w
        .iter()
        .map(|v| (v / scale).round().clamp(-1.0, 1.0) as i8)
        .collect();
    (quantized, scale)
}

/// Quantize FP32 activations to symmetric int8 using per-tensor absmax scaling
/// (BitNet-style). Returns (q, scale) with q[k] = round(x[k] / scale) clamped to
/// [-127, 127] and scale = max(|x|) / 127. Reconstruction: x ≈ q * scale.
/// An all-zero input yields all-zero codes and scale 0.0.
pub fn quantize_act(x: &[f32]) -> (Vec<i8>, f32) {
    let max_abs = x.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    if max_abs == 0.0 {
        return (vec![0i8; x.len()], 0.0);
    }
    let scale = max_abs / 127.0;
    let q = x
        .iter()
        .map(|v| (v / scale).round().clamp(-127.0, 127.0) as i8)
        .collect();
    (q, scale)
}

/// Quantize an FP32 activation matrix `x` of shape (k, n) row-major to int8
/// using a per-column (per-token) absmax/127 scale. Returns (q, scales) where
/// `scales` has length n. A column of all zeros gets scale 0.0 and codes 0.
pub fn quantize_act_2d(x: &[f32], k: usize, n: usize) -> (Vec<i8>, Vec<f32>) {
    let mut scales = vec![0.0f32; n];
    for j in 0..n {
        let mut max_abs = 0.0f32;
        for r in 0..k {
            max_abs = max_abs.max(x[r * n + j].abs());
        }
        scales[j] = max_abs / 127.0;
    }
    let mut q = vec![0i8; k * n];
    for r in 0..k {
        for j in 0..n {
            let s = scales[j];
            if s != 0.0 {
                q[r * n + j] = (x[r * n + j] / s).round().clamp(-127.0, 127.0) as i8;
            }
        }
    }
    (q, scales)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantize_act_2d_per_column() {
        // Column 0 ranges to 10, column 1 to 1 -> different scales.
        let x = [10.0f32, 1.0, -5.0, -0.5, 0.0, 0.25];
        let (q, s) = quantize_act_2d(&x, 3, 2);
        assert!((s[0] - 10.0 / 127.0).abs() < 1e-9);
        assert!((s[1] - 1.0 / 127.0).abs() < 1e-9);
        assert_eq!(q[0], 127); // x[0,0]=10.0 / (10/127)
        assert_eq!(q[2], -64); // x[1,0]=-5.0 -> round(-63.5)
        assert_eq!(q[1], 127); // x[0,1]=1.0 in column 1
    }

    #[test]
    fn test_quantize_act_2d_zero_column() {
        let x = [0.0f32, 2.0, 0.0, -2.0];
        let (q, s) = quantize_act_2d(&x, 2, 2);
        assert_eq!(s[0], 0.0);
        assert_eq!(q[0], 0);
        assert_eq!(q[2], 0);
        assert_eq!(q[1], 127); // column 1 max 2.0
    }

    #[test]
    fn test_quantize_act_basic() {
        let x = [1.0f32, -0.5, 0.0, 0.25];
        let (q, s) = quantize_act(&x);
        assert!((s - 1.0 / 127.0).abs() < 1e-9, "scale={}", s);
        assert_eq!(q[0], 127);
        assert_eq!(q[1], -64); // round(-0.5 / (1/127)) = round(-63.5) = -64
        assert_eq!(q[2], 0);
        // Dequantized values stay within int8 quantization error.
        for (qi, xi) in q.iter().zip(x.iter()) {
            assert!((*qi as f32 * s - xi).abs() <= s);
        }
    }

    #[test]
    fn test_quantize_act_all_zero() {
        let (q, s) = quantize_act(&[0.0, 0.0]);
        assert_eq!(q, vec![0, 0]);
        assert_eq!(s, 0.0);
    }

    #[test]
    fn test_quantize_act_range() {
        let x: Vec<f32> = (0..100).map(|i| (i as f32 - 50.0) * 0.3).collect();
        let (q, _s) = quantize_act(&x);
        assert!(q.iter().all(|&v| (-127..=127).contains(&(v as i32))));
    }

    #[test]
    fn test_all_zero() {
        let (q, s) = absmax_quantize(&[0.0, 0.0, 0.0]);
        assert_eq!(q, vec![0, 0, 0]);
        assert_eq!(s, 1.0);
    }

    #[test]
    fn test_absmean_keeps_nonzeros_under_outlier() {
        // One large outlier collapses absmax (everything else rounds to 0);
        // absmean (BitNet b1.58 formula, gamma = mean|W|) preserves the signal.
        let w = [10.0f32, 2.0, 2.0, 2.0];
        let (q, s) = absmean_quantize(&w);
        assert!((s - 4.0).abs() < 1e-6, "scale={}", s); // mean(|W|) = 16/4
        assert_eq!(q, vec![1, 1, 1, 1]);
        let (qmax, _) = absmax_quantize(&w);
        assert_eq!(qmax, vec![1, 0, 0, 0]); // absmax: 2/10 -> 0
    }

    #[test]
    fn test_absmean_all_zero() {
        let (q, s) = absmean_quantize(&[0.0, 0.0]);
        assert_eq!(q, vec![0, 0]);
        assert_eq!(s, 1.0);
    }

    #[test]
    fn test_basic() {
        let w = [2.0f32, -1.0, 0.0, 0.5];
        let (q, s) = absmax_quantize(&w);
        assert!((s - 2.0).abs() < 1e-6, "scale={}", s);
        assert_eq!(q[0], 1);
        assert_eq!(q[1], -1);
        assert_eq!(q[2], 0);
    }

    #[test]
    fn test_already_ternary() {
        let w = [1.0f32, 0.0, -1.0, 1.0];
        let (q, s) = absmax_quantize(&w);
        assert!((s - 1.0).abs() < 1e-6);
        assert_eq!(q, vec![1, 0, -1, 1]);
    }

    #[test]
    fn test_clamp() {
        let w = [3.0f32, -0.1];
        let (q, _s) = absmax_quantize(&w);
        assert!(q[0] <= 1 && q[0] >= -1);
    }
}
