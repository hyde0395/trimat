import numpy as np
import pytest

import trimat
from trimat.nn import BitLinear

# torch is an optional dependency; skip the whole module when it is absent.
torch = pytest.importorskip("torch")


def _ternary_weight(out_features, in_features, seed=0):
    """A weight whose values are already in {-1, 0, 1} (so packing is exact)."""
    rng = np.random.default_rng(seed)
    return rng.choice([-1.0, 0.0, 1.0], size=(out_features, in_features)).astype(
        np.float32
    )


def test_forward_single_sample_matches_reference():
    w = _ternary_weight(5, 8)
    layer = BitLinear(w)
    x = torch.arange(8, dtype=torch.float32)
    y = layer(x)
    expected = w @ x.numpy()
    np.testing.assert_allclose(y.numpy(), expected, atol=1e-4)
    assert y.shape == (5,)


def test_forward_batch_matches_reference():
    w = _ternary_weight(6, 10)
    layer = BitLinear(w)
    x = torch.randn(4, 10)
    y = layer(x)
    expected = x.numpy() @ w.T
    np.testing.assert_allclose(y.numpy(), expected, atol=1e-4)
    assert y.shape == (4, 6)


def test_forward_preserves_leading_dims():
    w = _ternary_weight(3, 7)
    layer = BitLinear(w)
    x = torch.randn(2, 5, 7)
    y = layer(x)
    assert y.shape == (2, 5, 3)
    expected = x.numpy() @ w.T
    np.testing.assert_allclose(y.numpy(), expected, atol=1e-4)


def test_bias_is_added():
    w = _ternary_weight(4, 6)
    bias = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    layer = BitLinear(w, bias)
    x = torch.randn(3, 6)
    y = layer(x).numpy()
    expected = x.numpy() @ w.T + bias
    np.testing.assert_allclose(y, expected, atol=1e-4)


def test_from_linear_matches_quantized_reference():
    linear = torch.nn.Linear(8, 4)
    # Force already-ternary weights so quantization is a no-op (scale=1).
    with torch.no_grad():
        linear.weight.copy_(torch.tensor(_ternary_weight(4, 8)))
        linear.bias.copy_(torch.arange(4, dtype=torch.float32))

    layer = BitLinear.from_linear(linear)
    x = torch.randn(3, 8)
    y = layer(x).numpy()
    expected = (x @ linear.weight.T + linear.bias).detach().numpy()
    np.testing.assert_allclose(y, expected, atol=1e-4)


def test_from_packed():
    w = _ternary_weight(4, 5)
    packed = trimat.pack(w)
    layer = BitLinear.from_packed(packed)
    assert isinstance(layer, BitLinear)
    assert layer.in_features == 5 and layer.out_features == 4
    x = torch.randn(2, 5)
    np.testing.assert_allclose(layer(x).numpy(), x.numpy() @ w.T, atol=1e-4)


def test_is_torch_module():
    layer = BitLinear(_ternary_weight(3, 3))
    assert isinstance(layer, torch.nn.Module)
    # Module plumbing works (e.g. eval / repr).
    layer.eval()
    assert "in_features=3" in repr(layer)


def test_wrong_input_dim_raises():
    layer = BitLinear(_ternary_weight(3, 4))
    with pytest.raises(ValueError):
        layer(torch.randn(2, 5))


def test_quantized_single_sample_approximates_reference():
    w = _ternary_weight(8, 64)
    layer = BitLinear(w, quantized=True)
    assert layer.quantized
    x = torch.randn(64)
    y = layer(x).numpy()
    expected = w @ x.numpy()
    # int8 activation error: bounded by max|x|/127 per term.
    tol = (np.abs(x.numpy()).max() / 127.0) * 64
    assert np.max(np.abs(y - expected)) <= tol


def test_quantized_batch_approximates_reference():
    w = _ternary_weight(6, 48)
    layer = BitLinear(w, quantized=True)
    x = torch.randn(5, 48)
    y = layer(x).numpy()
    expected = x.numpy() @ w.T
    # Per-row (per-token) int8 error bound.
    tol = (np.abs(x.numpy()).max(axis=1, keepdims=True) / 127.0) * 48
    assert np.all(np.abs(y - expected) <= tol + 1e-4)


def test_quantized_from_linear_roundtrips_shape():
    linear = torch.nn.Linear(32, 16)
    with torch.no_grad():
        linear.weight.copy_(torch.tensor(_ternary_weight(16, 32)))
    layer = BitLinear.from_linear(linear, quantized=True)
    assert layer.quantized
    y = layer(torch.randn(4, 32))
    assert y.shape == (4, 16)
