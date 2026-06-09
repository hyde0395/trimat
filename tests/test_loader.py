import numpy as np
import pytest

import trimat
from trimat.loader import load_safetensors, pack_weights
from trimat.errors import LoaderError

# safetensors is an optional dependency; skip file-based tests when absent.
safetensors_numpy = pytest.importorskip("safetensors.numpy")


def _write(tmp_path, tensors):
    path = tmp_path / "model.safetensors"
    safetensors_numpy.save_file(tensors, str(path))
    return path


def test_pack_weights_packs_2d_floats():
    w = np.array([[1.0, 0.0, -1.0], [-1.0, 1.0, 0.0]], dtype=np.float32)
    bias = np.array([0.5, -0.5], dtype=np.float32)
    out = pack_weights({"layer.weight": w, "layer.bias": bias})

    assert isinstance(out["layer.weight"], trimat.TernaryTensor)
    assert out["layer.weight"].rows == 2 and out["layer.weight"].cols == 3
    # 1D bias passes through unchanged.
    assert isinstance(out["layer.bias"], np.ndarray)
    np.testing.assert_array_equal(out["layer.bias"], bias)


def test_load_safetensors_roundtrip(tmp_path):
    w = np.array([[1.0, 0.0, -1.0], [-1.0, 1.0, 0.0]], dtype=np.float32)
    bias = np.array([0.5, -0.5], dtype=np.float32)
    path = _write(tmp_path, {"layer.weight": w, "layer.bias": bias})

    loaded = load_safetensors(str(path))
    assert isinstance(loaded["layer.weight"], trimat.TernaryTensor)
    assert isinstance(loaded["layer.bias"], np.ndarray)

    # The packed weight reproduces the ternary GEMV (already ternary -> scale 1).
    x = np.array([2.0, 3.0, 4.0], dtype=np.float32)
    y = trimat.gemv(loaded["layer.weight"], x)
    np.testing.assert_allclose(y, w @ x, atol=1e-5)


def test_should_pack_filter(tmp_path):
    w = np.eye(4, dtype=np.float32)
    path = _write(tmp_path, {"keep.weight": w, "skip.weight": w})

    loaded = load_safetensors(str(path), should_pack=lambda name: "skip" not in name)
    assert isinstance(loaded["keep.weight"], trimat.TernaryTensor)
    assert isinstance(loaded["skip.weight"], np.ndarray)


def test_load_missing_file_raises_loader_error(tmp_path):
    with pytest.raises(LoaderError):
        load_safetensors(str(tmp_path / "does-not-exist.safetensors"))
