"""Type stubs for the trimat Rust extension module (``trimat._trimat``)."""
from typing import Any

import numpy as np
from numpy.typing import NDArray

class TernaryTensor:
    """A packed ternary weight matrix ({-1, 0, +1}) with per-tensor or
    per-channel scale, produced by :func:`pack_tensor`."""

    @property
    def rows(self) -> int: ...
    @property
    def cols(self) -> int: ...
    def __repr__(self) -> str: ...

def pack_tensor(w: NDArray[np.float32]) -> TernaryTensor:
    """Quantize an FP32 ``(M, K)`` matrix with absmax and pack into bitplanes."""
    ...

def gemv(w: TernaryTensor, x: NDArray[np.float32]) -> NDArray[np.float32]:
    """Compute ``w (M×K) · x (K) -> y (M)``."""
    ...

def qgemv(w: TernaryTensor, x: NDArray[np.float32]) -> NDArray[np.float32]:
    """BitNet-style GEMV: quantize x to int8 (per-tensor absmax) then compute
    ``w (M×K) · x (K) -> y (M)`` with integer accumulation. Lossy vs gemv."""
    ...

def gemm(w: TernaryTensor, x: NDArray[np.float32]) -> NDArray[np.float32]:
    """Compute ``w (M×K) · X (K×N) -> Y (M×N)``."""
    ...

def qgemm(w: TernaryTensor, x: NDArray[np.float32]) -> NDArray[np.float32]:
    """BitNet-style GEMM: quantize each column of X to int8 (per-column absmax)
    then compute ``w (M×K) · X (K×N) -> Y (M×N)`` with integer accumulation.
    Lossy vs gemm."""
    ...

def cpu_features() -> dict[str, Any]:
    """Return runtime dispatch info, e.g. ``{"backend": "neon", "threads": 8}``."""
    ...
