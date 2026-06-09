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

#[cfg(test)]
mod tests {
    use super::*;

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
