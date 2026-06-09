use crate::kernels::{scalar::Scalar, Kernel};

pub struct DispatchInfo {
    pub backend: &'static str,
    pub threads: usize,
}

/// 현재 런타임에서 최적 커널을 반환한다.
/// P0: scalar 고정. P1: aarch64 NEON 분기 추가.
pub fn best_kernel() -> Box<dyn Kernel> {
    Box::new(Scalar)
}

/// cpu_features() Python 함수에 전달할 런타임 정보.
pub fn dispatch_info() -> DispatchInfo {
    DispatchInfo {
        backend: "scalar",
        threads: 1,
    }
}
