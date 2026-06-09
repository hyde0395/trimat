use crate::kernels::Kernel;

pub struct DispatchInfo {
    pub backend: &'static str,
    pub threads: usize,
}

/// Returns the best available kernel for the current runtime.
/// On aarch64: NEON with rayon parallelism.
/// On x86_64: AVX2 when detected at runtime, else parallel scalar.
/// Elsewhere: parallel scalar.
pub fn best_kernel() -> Box<dyn Kernel> {
    #[cfg(target_arch = "aarch64")]
    { Box::new(crate::kernels::neon::Neon) }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            Box::new(crate::kernels::avx2::Avx2)
        } else {
            Box::new(crate::kernels::scalar::Scalar)
        }
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    { Box::new(crate::kernels::scalar::Scalar) }
}

/// Runtime info passed to the cpu_features() Python function.
pub fn dispatch_info() -> DispatchInfo {
    #[cfg(target_arch = "aarch64")]
    let backend = "neon";

    #[cfg(target_arch = "x86_64")]
    let backend = if is_x86_feature_detected!("avx2") { "avx2" } else { "scalar" };

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    let backend = "scalar";

    DispatchInfo {
        backend,
        threads: rayon::current_num_threads(),
    }
}
