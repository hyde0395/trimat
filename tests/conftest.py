import numpy as np
import pytest
import trimat


@pytest.fixture
def small_ternary():
    """2×3 weight matrix with known values."""
    w = np.array([[1.0, 0.0, -1.0],
                  [-1.0, 1.0, 0.0]], dtype=np.float32)
    return trimat.pack(w)


@pytest.fixture(params=[
    (16, 32),
    (64, 128),
    (128, 256),
])
def random_ternary(request):
    """Parametrized random ternary matrix."""
    rows, cols = request.param
    rng = np.random.default_rng(42)
    w = rng.choice([-1.0, 0.0, 1.0], size=(rows, cols)).astype(np.float32)
    return trimat.pack(w), rows, cols, w
