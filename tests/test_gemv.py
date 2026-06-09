import numpy as np
import pytest
import trimat


def _ref_gemv(w_np: np.ndarray, x: np.ndarray) -> np.ndarray:
    """NumPy reference: ternary GEMV using raw float weights (pre-quantized)."""
    return (w_np @ x).astype(np.float32)


def test_gemv_shape(small_ternary):
    x = np.ones(3, dtype=np.float32)
    y = trimat.gemv(small_ternary, x)
    assert y.shape == (2,)


def test_gemv_dtype(small_ternary):
    x = np.ones(3, dtype=np.float32)
    y = trimat.gemv(small_ternary, x)
    assert y.dtype == np.float32


def test_gemv_known_values():
    # w = [[1, 0, -1], [-1, 1, 0]] already ternary → scale=1
    w_np = np.array([[1.0, 0.0, -1.0],
                     [-1.0, 1.0, 0.0]], dtype=np.float32)
    t = trimat.pack(w_np)
    x = np.array([2.0, 3.0, 4.0], dtype=np.float32)
    y = trimat.gemv(t, x)
    expected = w_np @ x
    np.testing.assert_allclose(y, expected, atol=1e-5)


def test_gemv_wrong_length(small_ternary):
    with pytest.raises(Exception):
        trimat.gemv(small_ternary, np.ones(5, dtype=np.float32))


def test_gemv_random(random_ternary):
    t, rows, cols, w_np = random_ternary
    x = np.random.default_rng(0).random(cols).astype(np.float32)
    y = trimat.gemv(t, x)
    expected = (w_np @ x)
    np.testing.assert_allclose(y, expected, atol=1e-4)


def test_gemv_all_zeros():
    w = np.zeros((4, 8), dtype=np.float32)
    t = trimat.pack(w)
    x = np.ones(8, dtype=np.float32)
    y = trimat.gemv(t, x)
    np.testing.assert_allclose(y, np.zeros(4), atol=1e-6)


def test_gemv_returns_copy(small_ternary):
    x = np.ones(3, dtype=np.float32)
    y1 = trimat.gemv(small_ternary, x)
    y2 = trimat.gemv(small_ternary, x)
    assert y1 is not y2
