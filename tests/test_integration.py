"""End-to-end integration test of the full trimat stack.

Exercises the real pipeline on a BitNet-style Transformer FFN block
(fc1: d -> 4d, ReLU, fc2: 4d -> d) with ternary weights:

    safetensors file -> trimat.loader -> TernaryTensor -> BitLinear(quantized)
    -> qgemv (batch=1) / qgemm (batch>1) -> multi-layer forward

A real HF BitNet checkpoint is GB-scale and stores latent/packed weights rather
than clean ternary float matrices, so this builds a faithful block instead and
validates the whole path two ways:
  1. the exact (quantized=False) path reproduces a plain NumPy reference, and
  2. the int8 (quantized=True) path stays within int8 quantization error of it.
"""
import os
import tempfile

import numpy as np
import pytest

import trimat
from trimat.loader import load_safetensors

torch = pytest.importorskip("torch")
safetensors_numpy = pytest.importorskip("safetensors.numpy")

from trimat.nn import BitLinear  # noqa: E402  (after torch importorskip)


D_MODEL = 128
D_FF = 512


def _build_ffn_safetensors(tmp_path, seed=0):
    rng = np.random.default_rng(seed)
    tensors = {
        "fc1.weight": rng.choice([-1.0, 0.0, 1.0], size=(D_FF, D_MODEL)).astype(np.float32),
        "fc1.bias": rng.standard_normal(D_FF).astype(np.float32),
        "fc2.weight": rng.choice([-1.0, 0.0, 1.0], size=(D_MODEL, D_FF)).astype(np.float32),
        "fc2.bias": rng.standard_normal(D_MODEL).astype(np.float32),
    }
    path = os.path.join(tmp_path, "ffn.safetensors")
    safetensors_numpy.save_file(tensors, path)
    return path, tensors


def _ffn(fc1, fc2, x):
    return fc2(torch.relu(fc1(x)))


def _numpy_reference(t, x):
    h = np.maximum(x @ t["fc1.weight"].T + t["fc1.bias"], 0.0)
    return h @ t["fc2.weight"].T + t["fc2.bias"]


def test_loader_packs_weights_keeps_bias(tmp_path):
    path, _ = _build_ffn_safetensors(tmp_path)
    loaded = load_safetensors(path)
    assert isinstance(loaded["fc1.weight"], trimat.TernaryTensor)
    assert isinstance(loaded["fc2.weight"], trimat.TernaryTensor)
    assert isinstance(loaded["fc1.bias"], np.ndarray)
    assert loaded["fc1.weight"].rows == D_FF and loaded["fc1.weight"].cols == D_MODEL


@pytest.mark.parametrize("batch", [1, 8])
def test_exact_path_matches_numpy(tmp_path, batch):
    path, tensors = _build_ffn_safetensors(tmp_path)
    loaded = load_safetensors(path)
    fc1 = BitLinear.from_packed(loaded["fc1.weight"], loaded["fc1.bias"])
    fc2 = BitLinear.from_packed(loaded["fc2.weight"], loaded["fc2.bias"])

    x = torch.randn(batch, D_MODEL)
    y = _ffn(fc1, fc2, x).numpy()
    ref = _numpy_reference(tensors, x.numpy())
    np.testing.assert_allclose(y, ref, atol=1e-3)


@pytest.mark.parametrize("batch", [1, 8])
def test_int8_path_close_to_exact(tmp_path, batch):
    """The quantized path (qgemv batch=1 / qgemm batch>1) stays within int8 error."""
    path, _ = _build_ffn_safetensors(tmp_path)
    loaded = load_safetensors(path)

    def layers(quantized):
        return (
            BitLinear.from_packed(loaded["fc1.weight"], loaded["fc1.bias"], quantized=quantized),
            BitLinear.from_packed(loaded["fc2.weight"], loaded["fc2.bias"], quantized=quantized),
        )

    x = torch.randn(batch, D_MODEL)
    y_exact = _ffn(*layers(False), x).numpy()
    y_int8 = _ffn(*layers(True), x).numpy()

    rel = np.max(np.abs(y_int8 - y_exact)) / (np.max(np.abs(y_exact)) + 1e-9)
    assert rel < 0.05, f"int8 vs exact relative error {rel:.4f} too large"
