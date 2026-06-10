"""HuggingFace safetensors loader for trimat.

Reads weights from a ``.safetensors`` file and packs eligible 2D floating-point
weight matrices into trimat :class:`TernaryTensor` objects, ready for
:func:`trimat.gemv` / :func:`trimat.gemm`. Tensors that are not 2D floats
(biases, norms, embeddings, ...) pass through unchanged as numpy arrays.

``safetensors`` and ``huggingface_hub`` are optional dependencies, imported
lazily so ``import trimat`` keeps working without them.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional, Union

import numpy as np

from trimat import pack
from trimat.errors import LoaderError

# A loaded entry is either a packed ternary weight or a pass-through array.
Entry = Union["object", np.ndarray]

ShouldPack = Callable[[str], bool]


def _load_file() -> Callable[[str], Dict[str, np.ndarray]]:
    """Return ``safetensors.numpy.load_file`` or raise a helpful LoaderError."""
    try:
        from safetensors.numpy import load_file
    except ImportError as exc:  # pragma: no cover - exercised without the dep
        raise LoaderError(
            "safetensors is required to load .safetensors files; "
            "install it with `pip install safetensors`"
        ) from exc
    return load_file


def _is_packable(arr: np.ndarray) -> bool:
    """A weight is packable when it is a 2D floating-point matrix."""
    return arr.ndim == 2 and np.issubdtype(arr.dtype, np.floating)


def pack_weights(
    state_dict: Dict[str, np.ndarray],
    *,
    should_pack: Optional[ShouldPack] = None,
    mode: str = "absmax",
) -> Dict[str, Entry]:
    """Pack eligible weights from a ``name -> ndarray`` mapping.

    A tensor is packed into a :class:`TernaryTensor` when it is a 2D float
    matrix, unless ``should_pack`` is supplied and returns ``False`` for its
    name. Every other tensor is returned unchanged as a numpy array.

    ``mode`` is the ternary quantization formula: ``"absmax"`` (default) or
    ``"absmean"``. **Use ``"absmean"`` for real BitNet b1.58 checkpoints** —
    their weights have large outliers that collapse absmax to ~0.
    """
    out: Dict[str, Entry] = {}
    for name, value in state_dict.items():
        arr = np.asarray(value)
        if _is_packable(arr) and (should_pack is None or should_pack(name)):
            out[name] = pack(np.ascontiguousarray(arr, dtype=np.float32), mode)
        else:
            out[name] = arr
    return out


def load_safetensors(
    path: str,
    *,
    should_pack: Optional[ShouldPack] = None,
    mode: str = "absmax",
) -> Dict[str, Entry]:
    """Load a ``.safetensors`` file and pack its 2D float weights.

    Returns a ``name -> TernaryTensor | ndarray`` mapping. See
    :func:`pack_weights` for ``mode`` (use ``"absmean"`` for BitNet b1.58).
    """
    load_file = _load_file()
    try:
        state = load_file(str(path))
    except LoaderError:
        raise
    except Exception as exc:
        raise LoaderError(f"failed to read safetensors file {path!r}: {exc}") from exc
    return pack_weights(state, should_pack=should_pack, mode=mode)


def from_pretrained(
    repo_id: str,
    *,
    filename: str = "model.safetensors",
    revision: Optional[str] = None,
    should_pack: Optional[ShouldPack] = None,
    mode: str = "absmax",
) -> Dict[str, Entry]:
    """Download a model's safetensors from the HF Hub and pack its weights.

    Requires the optional ``huggingface_hub`` dependency. See
    :func:`pack_weights` for ``mode`` (use ``"absmean"`` for BitNet b1.58).
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - exercised without the dep
        raise LoaderError(
            "huggingface_hub is required for from_pretrained; "
            "install it with `pip install huggingface_hub`"
        ) from exc
    try:
        local_path = hf_hub_download(
            repo_id=repo_id, filename=filename, revision=revision
        )
    except Exception as exc:
        raise LoaderError(
            f"failed to download {filename!r} from {repo_id!r}: {exc}"
        ) from exc
    return load_safetensors(local_path, should_pack=should_pack, mode=mode)
