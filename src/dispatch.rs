use crate::kernels::Kernel;

pub struct DispatchInfo {
    pub backend: &'static str,
    pub threads: usize,
}

/// Returns the best available kernel for the current runtime.
/// On aarch64: NEON with rayon parallelism.
/// Elsewhere: parallel scalar.
pub fn best_kernel() -> Box<dyn Kernel> {
    #[cfg(target_arch = "aarch64")]
    { Box::new(crate::kernels::neon::Neon) }
    #[cfg(not(target_arch = "aarch64"))]
    { Box::new(crate::kernels::scalar::Scalar) }
}

/// Runtime info passed to the cpu_features() Python function.
pub fn dispatch_info() -> DispatchInfo {
    #[cfg(target_arch = "aarch64")]
    let backend = "neon";
    #[cfg(not(target_arch = "aarch64"))]
    let backend = "scalar";

    DispatchInfo {
        backend,
        threads: rayon::current_num_threads(),
    }
}
