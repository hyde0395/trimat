<div align="center">

# trimat

**Ternary matrix multiplication for Python — Rust + SIMD, one `pip install`.**

Fast GEMV/GEMM for BitNet-family models with ternary weights `{-1, 0, +1}`,
written in Rust with hand-tuned NEON / AVX2 kernels and bound to Python via PyO3.

[![CI](https://github.com/hyde0395/trimat/actions/workflows/CI.yml/badge.svg)](https://github.com/hyde0395/trimat/actions/workflows/CI.yml)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![Platforms](https://img.shields.io/badge/platforms-Apple%20Silicon%20%C2%B7%20ARM%20Linux%20%C2%B7%20x86--64-lightgrey)
![License](https://img.shields.io/badge/license-MIT-green)

</div>

---

## Why trimat

BitNet b1.58 models store weights as ternary `{-1, 0, +1}`, but mainstream
frameworks still run them through dense FP32 kernels. The specialized kernels
exist only in C++ (`bitnet.cpp`, ETH Zurich's ternaryLLM) — **no `pip install`,
no Python bindings.** trimat fills that gap: a cross-platform, pip-installable
ternary kernel library.

On the path that dominates LLM inference — **autoregressive decode (batch=1)** —
trimat's int8-activation kernel beats Apple's Accelerate (AMX) BLAS:

| Weight shape (M×K) | NumPy fp32 (Accelerate) | trimat `qgemv` (int8) | Speedup |
|--------------------|------------------------:|----------------------:|--------:|
| 1024 × 4096        | 126 µs                  | 87 µs                 | **1.45×** |
| 4096 × 14336 (FFN) | 3388 µs                 | 985 µs                | **3.44×** |

<sub>Apple M4 (NEON, release build), end-to-end through the Python extension. See [Benchmarks](#benchmarks) for the full picture, including where trimat is *slower*.</sub>

---

## Install

```bash
pip install trimat
```

No cmake, no system toolchain. Wheels target **Apple Silicon (aarch64 macOS),
ARM Linux, and x86-64 (Linux / macOS / Windows)**. NumPy is the only required
runtime dependency.

## Quickstart

```python
import numpy as np
import trimat

# Quantize + pack an FP32 weight matrix (M×K) into a ternary tensor.
w  = np.random.choice([-1.0, 0.0, 1.0], size=(256, 512)).astype(np.float32)
wt = trimat.pack(w)

x = np.random.randn(512).astype(np.float32)
y = trimat.gemv(wt, x)           # exact ternary:  w · x          -> (256,)
y = trimat.qgemv(wt, x)          # BitNet int8:    fast, ~int8 x  -> (256,)

X = np.random.randn(512, 32).astype(np.float32)
Y = trimat.gemm(wt, X)           # w · X (K×N)                    -> (256, 32)
Y = trimat.qgemm(wt, X)          # int8-activation GEMM

print(trimat.cpu_features())     # {'backend': 'neon', 'threads': 10}
```

## PyTorch drop-in: `BitLinear`

An inference-only `nn.Linear` replacement backed by the ternary kernels. Set
`quantized=True` for the fast BitNet int8 path (`qgemv` / `qgemm`).

```python
import torch
from trimat.nn import BitLinear

linear = torch.nn.Linear(512, 256)
bit = BitLinear.from_linear(linear, quantized=True)
y = bit(torch.randn(8, 512))     # (8, 256)
```

## Load HuggingFace weights

```python
from trimat.loader import load_safetensors, from_pretrained

weights = load_safetensors("model.safetensors")
# "...weight" -> TernaryTensor, "...bias" -> np.ndarray (passed through)
```

`safetensors` (and `huggingface_hub` for `from_pretrained`) are optional deps,
imported lazily.

## Benchmarks

Apple M4, NEON, release build, end-to-end. NumPy uses the Accelerate framework
(AMX matrix coprocessor) — an exceptionally strong baseline.

**GEMV (`qgemv`, int8 activations):**

| Weight shape | NumPy | `gemv` (fp32) | `qgemv` (int8) | `qgemv` vs NumPy |
|--------------|------:|--------------:|---------------:|-----------------:|
| 512 × 1024   | 7 µs  | 39 µs         | 23 µs          | 0.28× |
| 1024 × 4096  | 126 µs| 204 µs        | 87 µs          | **1.45×** |
| 4096 × 14336 | 3388 µs| 2729 µs      | 985 µs         | **3.44×** |

**Honest caveats:**
- **Small matrices lose** to NumPy's ~1 µs fixed overhead — irrelevant at LLM scale.
- **Dense batched GEMM stays behind AMX**: `qgemm` is ~0.15× NumPy on a
  1024×4096×128 matmul. trimat's edge is GEMV / batch=1 decode and the 2-bit
  weight footprint, not dense FP32 matmul against a hardware matrix coprocessor.
- **`qgemv`/`qgemm` are lossy** by design (int8 activations, BitNet b1.58):
  error is bounded by ~`max|x| / 127` per term.
- x86-64 (AVX2) numbers come from CI; the table above is Apple Silicon.

Reproduce: `cargo bench` (Rust microbenchmarks) or the scripts in the repo.

## How it works

- **Packing** — weights are quantized to ternary and stored as two bitplanes
  (`nonzero`, `sign`), 2 bits/weight: 16× smaller than FP32.
- **SIMD decode** — kernels expand the bitplanes to per-lane masks *in registers*
  (NEON `vtstq` / AVX2 `cmpeq`) — no per-element branches, no FP32 multiplies on
  the int8 path.
- **int8 activations** — `qgemv`/`qgemm` quantize activations to int8 and
  accumulate the ternary dot product in i32 (the real BitNet b1.58 scheme), then
  dequantize once. This is what removes FP32 multiplies and wins at scale.
- **Dispatch** — NEON on aarch64, AVX2 on x86-64 (runtime-detected), scalar
  elsewhere; all row-parallel via rayon.

## Roadmap

- [x] P0 — scalar core (pack, GEMV, GEMM)
- [x] P1 — NEON kernels + rayon
- [x] P2 — AVX2 kernels + CI + GEMM tiling
- [x] P3 — HuggingFace loader + `BitLinear`
- [x] int8 BitNet path — `qgemv` / `qgemm` (NEON + AVX2), `BitLinear(quantized=True)`
- [x] End-to-end integration test (loader → BitLinear → multi-layer forward)
- [ ] PyPI release
- [ ] int8 `qgemm` register microkernel (close the batched-GEMM gap)

## Build from source

```bash
pip install maturin
maturin develop --release --features extension-module
python -m pytest tests/ -v        # Python suite
cargo test                        # Rust suite
```

## Contributing

Issues and PRs welcome. Please keep the test suites green (`cargo test` +
`pytest`) and run `cargo clippy --all-targets -- -D warnings`. All code comments
are English-only.

## License

Released under the MIT License — see [`LICENSE`](LICENSE).
