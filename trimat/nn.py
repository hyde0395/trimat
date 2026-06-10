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
    """Convert a torch tensor or array-like to a detached numpy array.

    numpy has no bfloat16/float16, so low-precision tensors (as produced by real
    bf16 models) are upcast to float32 first.
    """
    if hasattr(x, "detach"):  # torch.Tensor
        t = x.detach().cpu()
        if t.dtype not in (torch.float32, torch.float64):
            t = t.float()
        return t.numpy()
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

    def __init__(
        self, weight, bias=None, *, quantized: bool = False, mode: str = "absmax"
    ) -> None:
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
            # mode "absmean" is the BitNet b1.58 weight formula (use it for real
            # BitNet checkpoints); "absmax" is the default.
            self._packed = trimat.pack(w, mode)

        self.out_features = self._packed.rows
        self.in_features = self._packed.cols
        # When True, the decode path quantizes activations to int8 and uses the
        # BitNet qgemv kernel (faster at scale, slightly lossy). When False,
        # decode is exact f32 via gemv.
        self.quantized = quantized

        # Lazily-materialized dense weight for the prefill (BLAS) fallback. Built
        # only on first prefill, so a decode-only deployment keeps the 2-bit
        # packing (no f32 copy of the weight). Not registered as a buffer (it is
        # a derived cache, not a parameter); kept off the module's state_dict.
        self._dense_t: Optional["torch.Tensor"] = None   # f32 (out, in) for torch
        self._dense_np: Optional[np.ndarray] = None       # f32 (out, in) for numpy
        self._bias_t: Optional["torch.Tensor"] = None     # bias as a torch tensor

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
    def from_linear(
        cls, linear, *, quantized: bool = False, mode: str = "absmax"
    ) -> "BitLinear":
        """Quantize and pack an existing ``torch.nn.Linear`` layer.

        Use ``mode="absmean"`` for BitNet b1.58 layers.
        """
        weight = linear.weight.detach()
        bias = None if linear.bias is None else linear.bias.detach()
        return cls(weight, bias, quantized=quantized, mode=mode)

    @classmethod
    def from_packed(
        cls, tensor: TernaryTensor, bias=None, *, quantized: bool = False
    ) -> "BitLinear":
        """Wrap an already-packed :class:`TernaryTensor` (e.g. from the loader)."""
        return cls(tensor, bias, quantized=quantized)

    # ---- dense-weight cache for the prefill fallback -------------------------
    def _dense_weight_torch(self, dtype, device):
        if self._dense_t is None:
            # to_dense returns a fresh f32 (out, in) array; share it zero-copy.
            self._dense_t = torch.from_numpy(trimat.to_dense(self._packed))
        return self._dense_t.to(dtype=dtype, device=device)

    def _dense_weight_numpy(self) -> np.ndarray:
        if self._dense_np is None:
            self._dense_np = trimat.to_dense(self._packed)
        return self._dense_np

    def _bias_torch(self, dtype, device):
        if self._bias is None:
            return None
        if self._bias_t is None:
            self._bias_t = torch.from_numpy(self._bias)
        return self._bias_t.to(dtype=dtype, device=device)

    def forward(self, x):
        """Hybrid routing: decode (1 token) -> Rust ternary kernel (zero-copy);
        prefill (>1 token) -> PyTorch BLAS/AMX on the dense weight."""
        is_tensor = _HAS_TORCH and isinstance(x, torch.Tensor)

        last = x.shape[-1]
        if last != self.in_features:
            raise ValueError(
                f"input last dim {last} != in_features {self.in_features}"
            )
        lead = tuple(x.shape[:-1])
        n_tokens = 1
        for d in lead:
            n_tokens *= int(d)

        # ---- PREFILL (seq_len > 1): hand off to PyTorch's accelerated matmul ----
        # The Rust ternary kernel loses to AMX/BLAS on batched GEMM, so for >1
        # token we run F.linear on the dense (dequantized) weight instead.
        if n_tokens > 1:
            if is_tensor:
                w = self._dense_weight_torch(x.dtype, x.device)
                b = self._bias_torch(x.dtype, x.device)
                return torch.nn.functional.linear(x, w, b)
            xn = np.ascontiguousarray(np.asarray(x), dtype=np.float32)
            y = xn.reshape(-1, self.in_features) @ self._dense_weight_numpy().T
            if self._bias is not None:
                y = y + self._bias
            return y.reshape(*lead, self.out_features)

        # ---- DECODE (seq_len == 1): zero-copy into the Rust ternary kernel ------
        if is_tensor:
            # numpy has no bf16; upcasting low-precision tensors copies once.
            # f32 + contiguous is the true zero-copy path: .numpy() is a view and
            # rust-numpy's PyReadonlyArray borrows it without copying.
            xt = x.reshape(-1)
            if xt.dtype != torch.float32:
                xt = xt.float()
            xt = xt.contiguous()
            x_np = xt.detach().numpy()
        else:
            x_np = np.ascontiguousarray(np.reshape(x, -1), dtype=np.float32)

        vec = trimat.qgemv if self.quantized else trimat.gemv
        y_np = vec(self._packed, x_np)
        if self._bias is not None:
            y_np = y_np + self._bias

        y = y_np.reshape(*lead, self.out_features)
        if is_tensor:
            out = torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32))
            return out.to(x.dtype)  # preserve input dtype for downstream modules
        return y

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self._bias is not None}"
        )
