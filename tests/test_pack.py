import numpy as np
import pytest
import trimat
from trimat._trimat import TernaryTensor


def test_returns_tensor():
    w = np.array([[1.0, 0.0, -1.0]], dtype=np.float32)
    t = trimat.pack(w)
    assert isinstance(t, TernaryTensor)


def test_shape(small_ternary):
    assert small_ternary.rows == 2
    assert small_ternary.cols == 3


def test_repr(small_ternary):
    r = repr(small_ternary)
    assert "TernaryTensor" in r
    assert "2" in r
    assert "3" in r


def test_empty_raises():
    with pytest.raises(Exception):
        trimat.pack(np.zeros((0, 4), dtype=np.float32))


def test_fortran_order():
    w = np.asfortranarray(np.ones((4, 8), dtype=np.float32))
    t = trimat.pack(w)
    assert t.rows == 4
    assert t.cols == 8


def test_float64_raises():
    w = np.ones((4, 4), dtype=np.float64)
    with pytest.raises(Exception):
        trimat.pack(w)
