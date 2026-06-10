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


def test_forward_accepts_bf16_input():
    # Real models run in bf16; hidden states reach BitLinear as bf16 tensors,
    # which numpy cannot consume directly. forward must upcast them.
    w = _ternary_weight(3, 4)
    layer = BitLinear(w)
    x = torch.randn(2, 4).to(torch.bfloat16)
    y = layer(x)
    assert y.shape == (2, 3)
    # drop-in must preserve input dtype so downstream bf16 modules (lm_head) work
    assert y.dtype == torch.bfloat16
    expected = x.float().numpy() @ w.T
    np.testing.assert_allclose(y.float().numpy(), expected, atol=1e-1)


def test_bitlinear_absmean_mode():
    # absmean keeps the small values that absmax would zero out under an outlier.
    w = np.array([[10.0, 3.0, 3.0, 3.0]], dtype=np.float32)
    x = torch.ones(4)
    layer_max = BitLinear(w, mode="absmax")
    layer_mean = BitLinear(w, mode="absmean")
    np.testing.assert_allclose(layer_max(x).numpy()[0], 10.0, atol=1e-4)
    np.testing.assert_allclose(layer_mean(x).numpy()[0], 19.0, atol=1e-4)


def test_prefill_routes_to_dense_blas():
    # >1 token must go through F.linear on the DEQUANTIZED weight (BLAS/AMX),
    # even when quantized=True — so it equals to_dense(W) @ xᵀ exactly, not int8.
    rng = np.random.default_rng(0)
    w = (rng.standard_normal((6, 10)) * 3).astype(np.float32)  # lossy quantization
    layer = BitLinear(w, mode="absmean", quantized=True)
    dense = trimat.to_dense(layer._packed)  # (6, 10) ternary*scale
    x = torch.randn(4, 10)  # 4 tokens -> prefill
    y = layer(x).numpy()
    np.testing.assert_allclose(y, x.numpy() @ dense.T, atol=1e-4)


def test_decode_routes_to_gemv():
    w = _ternary_weight(5, 8)
    layer = BitLinear(w)
    x = torch.arange(8, dtype=torch.float32)  # 1 token -> decode (Rust gemv)
    np.testing.assert_allclose(layer(x).numpy(), w @ x.numpy(), atol=1e-4)


def test_dense_weight_built_lazily_decode_keeps_2bit():
    # The dense f32 weight must NOT be materialized until the first prefill,
    # so a decode-only deployment preserves the 2-bit packing.
    layer = BitLinear(_ternary_weight(4, 6))
    assert layer._dense_t is None
    _ = layer(torch.randn(1, 6))   # decode -> no dense weight
    assert layer._dense_t is None
    _ = layer(torch.randn(3, 6))   # prefill -> dense weight built + cached
    assert layer._dense_t is not None


def test_quantized_from_linear_roundtrips_shape():
    linear = torch.nn.Linear(32, 16)
    with torch.no_grad():
        linear.weight.copy_(torch.tensor(_ternary_weight(16, 32)))
    layer = BitLinear.from_linear(linear, quantized=True)
    assert layer.quantized
    y = layer(torch.randn(4, 32))
    assert y.shape == (4, 16)
