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

#[cfg(test)]
mod tests {
    use super::*;

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
