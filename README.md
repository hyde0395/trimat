# trimat

**Ternary matrix multiplication for Python — Rust + SIMD, one `pip install`.**

trimat provides GEMV/GEMM operations for BitNet-family models whose weights are
ternary (`{-1, 0, +1}`). Kernels are written in Rust with hand-written SIMD
(NEON on Apple Silicon / ARM, AVX2 on x86-64) and exposed to Python through
PyO3 — no cmake, no system toolchain required at install time.

> **Status:** correctness-complete and tested (Rust + Python suites green on
> aarch64 and x86-64). Kernel performance optimization is in progress — see
> [Performance](#performance).

## Install

```bash
pip install trimat
```

Supported platforms: Apple Silicon (aarch64 macOS), ARM Linux, and x86-64
(Linux / macOS / Windows). The only required runtime dependency is NumPy.

## Quickstart

```python
import numpy as np
import trimat

# Quantize + pack an FP32 weight matrix (M×K) into a ternary tensor.
w = np.random.choice([-1.0, 0.0, 1.0], size=(256, 512)).astype(np.float32)
wt = trimat.pack(w)

# GEMV:  w (M×K) · x (K)   -> y (M)
x = np.random.randn(512).astype(np.float32)
y = trimat.gemv(wt, x)

# GEMM:  w (M×K) · X (K×N) -> Y (M×N)
X = np.random.randn(512, 32).astype(np.float32)
Y = trimat.gemm(wt, X)

# Which kernel is active on this machine?
print(trimat.cpu_features())   # {'backend': 'neon', 'threads': 8}
```

## BitLinear (PyTorch drop-in)

`trimat.nn.BitLinear` is an inference-only `nn.Linear` replacement backed by the
ternary kernels. Requires the optional `torch` dependency.

```python
import torch
from trimat.nn import BitLinear

# From an existing Linear layer (weights are quantized + packed):
linear = torch.nn.Linear(512, 256)
bit = BitLinear.from_linear(linear)
y = bit(torch.randn(8, 512))     # (8, 256)
```

## Loading HuggingFace weights

`trimat.loader` reads a `.safetensors` file and packs eligible 2D float weight
matrices into ternary tensors (other tensors pass through unchanged). Requires
the optional `safetensors` dependency; `from_pretrained` additionally needs
`huggingface_hub`.

```python
from trimat.loader import load_safetensors, from_pretrained

weights = load_safetensors("model.safetensors")
# weights["...weight"] -> TernaryTensor, weights["...bias"] -> np.ndarray

weights = from_pretrained("some-org/some-bitnet-model")
```

## Performance

The kernels are functionally correct and run row-parallel (rayon) with NEON /
AVX2 vectorization. Performance tuning against optimized BLAS baselines — in
particular Apple's Accelerate framework on Apple Silicon — is ongoing; the
current GEMV decode path and the Python call boundary are the active targets.
Benchmark numbers will be published once the kernels are competitive.

To run the microbenchmarks locally:

```bash
cargo bench            # criterion: benches/gemv_bench.rs, benches/gemm_bench.rs
```

## Build from source

```bash
# Rust toolchain + a Python venv are required.
pip install maturin
maturin develop --release --features extension-module
python -m pytest tests/ -v
```

## License

See repository for license details.
