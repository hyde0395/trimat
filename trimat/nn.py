"""BitLinear — a torch ``nn.Linear`` drop-in backed by trimat ternary kernels.

``torch`` is an optional dependency. The class is always importable; when torch
is absent it falls back to a plain object base and constructing it raises a
clear :class:`ImportError`. When torch is present, :class:`BitLinear` is a real
``torch.nn.Module`` (supporting ``.eval()``, ``.to()``, submodule registration).

BitLinear is an **inference-only** layer: the forward pass runs the ternary
GEMM/GEMV kernels on the CPU and does not participate in autograd. It computes
the standard linear map ``y = x @ Wᵀ (+ bias)`` where ``W`` is stored as a
packed ternary :class:`TernaryTensor`.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

import trimat
from trimat import TernaryTensor

try:
    import torch
    _Base = torch.nn.Module
    _HAS_TORCH = True
except ImportError:  # pragma: no cover - exercised only without torch installed
    _Base = object  # type: ignore[assignment, misc]
    _HAS_TORCH = False


def _to_numpy(x) -> np.ndarray:
    """Convert a torch tensor or array-like to a detached numpy array."""
    if hasattr(x, "detach"):  # torch.Tensor
        return x.detach().cpu().numpy()
    return np.asarray(x)


class BitLinear(_Base):  # type: ignore[misc, valid-type]
    """Ternary linear layer: ``y = x @ Wᵀ + bias`` using packed weights.

    Construct directly from a float weight matrix, from an existing
    ``torch.nn.Linear`` via :meth:`from_linear`, or from an already-packed
    :class:`TernaryTensor` via :meth:`from_packed`.

    Args:
        weight: ``(out_features, in_features)`` float weight (array or tensor),
            or a pre-packed :class:`TernaryTensor`.
        bias: optional ``(out_features,)`` bias (array or tensor).
    """

    def __init__(self, weight, bias=None) -> None:
        if not _HAS_TORCH:
            raise ImportError(
                "BitLinear requires PyTorch; install it with `pip install torch`"
            )
        super().__init__()

        if isinstance(weight, TernaryTensor):
            self._packed = weight
        else:
            w = np.ascontiguousarray(_to_numpy(weight), dtype=np.float32)
            if w.ndim != 2:
                raise ValueError(f"weight must be 2D (out, in); got shape {w.shape}")
            self._packed = trimat.pack(w)

        self.out_features = self._packed.rows
        self.in_features = self._packed.cols

        if bias is None:
            self._bias: Optional[np.ndarray] = None
        else:
            b = np.ascontiguousarray(_to_numpy(bias), dtype=np.float32).reshape(-1)
            if b.shape[0] != self.out_features:
                raise ValueError(
                    f"bias length {b.shape[0]} != out_features {self.out_features}"
                )
            self._bias = b

    @classmethod
    def from_linear(cls, linear) -> "BitLinear":
        """Quantize and pack an existing ``torch.nn.Linear`` layer."""
        weight = linear.weight.detach()
        bias = None if linear.bias is None else linear.bias.detach()
        return cls(weight, bias)

    @classmethod
    def from_packed(cls, tensor: TernaryTensor, bias=None) -> "BitLinear":
        """Wrap an already-packed :class:`TernaryTensor` (e.g. from the loader)."""
        return cls(tensor, bias)

    def forward(self, x):
        is_tensor = _HAS_TORCH and isinstance(x, torch.Tensor)
        x_np = _to_numpy(x)
        if x_np.shape[-1] != self.in_features:
            raise ValueError(
                f"input last dim {x_np.shape[-1]} != in_features {self.in_features}"
            )

        lead_shape = x_np.shape[:-1]
        x2d = np.ascontiguousarray(
            x_np.reshape(-1, self.in_features), dtype=np.float32
        )

        if x2d.shape[0] == 1:
            # Single sample -> GEMV. y = W · x.
            y2d = trimat.gemv(self._packed, x2d[0])[None, :]
        else:
            # Batch -> GEMM. trimat computes W(out×in) · X(in×batch) = (out×batch).
            xt = np.ascontiguousarray(x2d.T, dtype=np.float32)
            y2d = trimat.gemm(self._packed, xt).T

        if self._bias is not None:
            y2d = y2d + self._bias

        y = y2d.reshape(*lead_shape, self.out_features)
        if is_tensor:
            return torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32))
        return y

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self._bias is not None}"
        )
