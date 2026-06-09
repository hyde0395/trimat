import numpy as np
import pytest

import trimat


def test_qgemm_shape_dtype(small_ternary):
    X = np.ones((3, 4), dtype=np.float32)
    Y = trimat.qgemm(small_ternary, X)
    assert Y.shape == (2, 4)
    assert Y.dtype == np.float32


def test_qgemm_wrong_rows(small_ternary):
    with pytest.raises(Exception):
        trimat.qgemm(small_ternary, np.ones((5, 4), dtype=np.float32))


def test_qgemm_zero_column():
    # A zero activation column must yield a zero output column.
    w = np.array([[1.0, -1.0, 1.0]], dtype=np.float32)
    t = trimat.pack(w)
    X = np.array([[0.0, 1.0], [0.0, 2.0], [0.0, -1.0]], dtype=np.float32)
    Y = trimat.qgemm(t, X)
    np.testing.assert_allclose(Y[:, 0], np.zeros(1), atol=1e-6)


def test_qgemm_matches_manual_int8_reference():
    rng = np.random.default_rng(11)
    M, K, N = 16, 40, 10  # K % 16 != 0 exercises the SIMD tail
    w = rng.choice([-1.0, 0.0, 1.0], size=(M, K)).astype(np.float32)
    t = trimat.pack(w)
    X = rng.standard_normal((K, N)).astype(np.float32) * 2.0

    # Reproduce per-column int8 quantization + integer accumulation.
    scale = np.abs(X).max(axis=0) / 127.0  # (N,)
    safe = np.where(scale == 0, 1.0, scale)
    Xq = np.clip(np.round(X / safe), -127, 127).astype(np.int32)
    expected = (w.astype(np.int32) @ Xq).astype(np.float32) * scale[None, :]

    Y = trimat.qgemm(t, X)
    np.testing.assert_allclose(Y, expected, atol=1e-3)


def test_qgemm_approximates_gemm():
    rng = np.random.default_rng(3)
    M, K, N = 32, 256, 8
    w = rng.choice([-1.0, 0.0, 1.0], size=(M, K)).astype(np.float32)
    t = trimat.pack(w)
    X = rng.standard_normal((K, N)).astype(np.float32)

    Y_q = trimat.qgemm(t, X)
    Y_ref = w @ X
    # int8 per-column error <= max|col|/127 per term, summed over K.
    tol = (np.abs(X).max(axis=0) / 127.0)[None, :] * K
    assert np.all(np.abs(Y_q - Y_ref) <= tol + 1e-4)
