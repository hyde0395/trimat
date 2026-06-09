import numpy as np
import pytest
import trimat


def _ref_gemm(w_np: np.ndarray, x: np.ndarray) -> np.ndarray:
    """NumPy reference: ternary GEMM. w(M×K) @ x(K×N) → y(M×N)."""
    return (w_np @ x).astype(np.float32)


def test_gemm_shape(small_ternary):
    x = np.ones((3, 5), dtype=np.float32)
    y = trimat.gemm(small_ternary, x)
    assert y.shape == (2, 5)


def test_gemm_dtype(small_ternary):
    x = np.ones((3, 5), dtype=np.float32)
    y = trimat.gemm(small_ternary, x)
    assert y.dtype == np.float32


def test_gemm_known_values():
    w_np = np.array([[1.0, 0.0, -1.0],
                     [-1.0, 1.0, 0.0]], dtype=np.float32)
    t = trimat.pack(w_np)
    x = np.array([[1.0, 2.0],
                  [3.0, 4.0],
                  [5.0, 6.0]], dtype=np.float32)
    y = trimat.gemm(t, x)
    expected = _ref_gemm(w_np, x)
    np.testing.assert_allclose(y, expected, atol=1e-5)


def test_gemm_wrong_rows(small_ternary):
    with pytest.raises(Exception):
        trimat.gemm(small_ternary, np.ones((5, 4), dtype=np.float32))


def test_gemm_random(random_ternary):
    t, rows, cols, w_np = random_ternary
    n = 7
    x = np.random.default_rng(1).random((cols, n)).astype(np.float32)
    y = trimat.gemm(t, x)
    expected = _ref_gemm(w_np, x)
    np.testing.assert_allclose(y, expected, atol=1e-4)
