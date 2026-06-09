import numpy as np
import pytest

import trimat


def test_qgemv_shape_dtype(small_ternary):
    x = np.ones(3, dtype=np.float32)
    y = trimat.qgemv(small_ternary, x)
    assert y.shape == (2,)
    assert y.dtype == np.float32


def test_qgemv_wrong_length(small_ternary):
    with pytest.raises(Exception):
        trimat.qgemv(small_ternary, np.ones(5, dtype=np.float32))


def test_qgemv_all_zero_activation():
    w = np.array([[1.0, -1.0, 1.0]], dtype=np.float32)
    t = trimat.pack(w)
    y = trimat.qgemv(t, np.zeros(3, dtype=np.float32))
    np.testing.assert_allclose(y, np.zeros(1), atol=1e-6)


def test_qgemv_approximates_gemv():
    # With ternary weights (scale 1) the only error is int8 activation
    # quantization, which is bounded by ~max|x|/127 per term.
    rng = np.random.default_rng(0)
    rows, cols = 64, 512
    w = rng.choice([-1.0, 0.0, 1.0], size=(rows, cols)).astype(np.float32)
    t = trimat.pack(w)
    x = rng.standard_normal(cols).astype(np.float32)

    y_q = trimat.qgemv(t, x)
    y_ref = w @ x

    # int8 per-element error <= max|x|/127; summed over cols nonzeros.
    tol = (np.abs(x).max() / 127.0) * cols
    assert np.max(np.abs(y_q - y_ref)) <= tol


def test_qgemv_matches_manual_int8_reference():
    rng = np.random.default_rng(7)
    rows, cols = 16, 40  # cols % 16 != 0 exercises the SIMD tail
    w = rng.choice([-1.0, 0.0, 1.0], size=(rows, cols)).astype(np.float32)
    t = trimat.pack(w)
    x = rng.standard_normal(cols).astype(np.float32) * 3.0

    # Reproduce the kernel's quantization and integer accumulation in numpy.
    scale = np.abs(x).max() / 127.0
    xq = np.clip(np.round(x / scale), -127, 127).astype(np.int32)
    expected = (w.astype(np.int32) @ xq).astype(np.float32) * scale

    y = trimat.qgemv(t, x)
    np.testing.assert_allclose(y, expected, atol=1e-3)
