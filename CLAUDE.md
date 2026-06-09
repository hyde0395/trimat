# trimat — Project Guide

## Overview

**trimat** = Ternary Matrix Multiplication Python library  
GEMV/GEMM operations for BitNet-family models with weights in {-1, 0, +1},  
implemented in Rust + SIMD and bound to Python via PyO3.

Core position: `pip install trimat` — no cmake — supports Apple Silicon · ARM Linux · x86.

---

## File Structure (target layout)

```
trimat/
├── __init__.py        # Public API: pack, gemv, gemm, cpu_features
├── nn.py              # BitLinear (torch drop-in, optional dep)
├── loader.py          # HuggingFace safetensors loader (Python layer)
├── _trimat.pyi        # Rust extension type stubs
└── py.typed           # PEP 561 marker

src/
├── lib.rs             # PyO3 entry point
├── tensor.rs          # TernaryTensor struct + pyclass
├── pack.rs            # ternary <-> bitplane encoding/decoding
├── quantize.rs        # FP32 -> ternary quantization (absmax)
├── dispatch.rs        # Runtime kernel selection
└── kernels/
    ├── mod.rs         # Kernel trait
    ├── scalar.rs      # Portable impl + rayon parallel
    ├── neon.rs        # aarch64 NEON intrinsics + rayon (P1 done)
    └── avx2.rs        # x86-64 AVX2 intrinsics (P2)

tests/
├── conftest.py        # Fixtures
├── test_pack.py
├── test_gemv.py
└── test_gemm.py

benches/
└── gemv_bench.rs      # criterion GEMV benchmarks
```

---

## Architecture Decisions

### 1. HF loader lives in Python (loader.py)

- safetensors Python API is mature and fast
- Per-model packing conventions handled easily in Python
- Network I/O via huggingface_hub stays in Python ecosystem
- Rust boundary crossed only once at trimat.pack()

### 2. quantize.rs — quantization and packing are separate modules

Quantization formula: w_q = sign(w) * round(|w| / max|w|)

### 3. Cargo.toml — extension-module is a feature, not a default dep

maturin adds --features extension-module via pyproject.toml.
cargo test and cargo bench run without it, linking Python from environment.
Required to support crate-type = ["cdylib", "rlib"] simultaneously.

### 4. Dispatch pattern

dispatch::best_kernel() returns Box<dyn Kernel>:
- aarch64: NEON kernel
- other: scalar kernel

Both use rayon for row-parallel GEMV/GEMM.

---

## Implementation Roadmap

P0: scalar core — DONE
  Results: 16 Rust tests + 22 pytest all green

P1: NEON + rayon — DONE (2026-06-09)
  - kernels/neon.rs: aarch64 NEON intrinsics, rayon parallel rows
  - kernels/scalar.rs: rayon par_iter_mut for rows
  - dispatch.rs: NEON on aarch64, rayon::current_num_threads()
  - kernels/mod.rs: cfg(target_arch) gated neon module
  - Cargo.toml: extension-module as feature, rlib added, criterion
  - benches/gemv_bench.rs
  Results: 16 Rust tests + 22 pytest all green

P2: AVX2 + CI (next)
  - kernels/avx2.rs
  - CI.yml (maturin-action multi-target)
  - GEMM tiling (gemm_bench.rs)

P3: HF loader + BitLinear
  - trimat/loader.py (safetensors)
  - trimat/nn.py (BitLinear)
  - tests/test_loader.py, test_nn.py

P4: Release
  - trimat/_trimat.pyi, py.typed
  - README benchmark table
  - PyPI deployment

---

## Code Style

**All comments must be written in English** — both `//`, `///` (Rust) and `#`, docstrings (Python).
Do not write comments in Korean. If you find existing Korean comments, convert them to English.

---

## Dev Environment

- Main dev machine: M4 Mac mini (aarch64, NEON native)
- pyo3 0.28, numpy 0.28, ndarray 0.17, rayon 1.10
- Python 3.14 (venv at .venv/)

## Dev Commands

  source "$HOME/.cargo/env" && source .venv/bin/activate
  cargo test
  maturin develop --features extension-module
  .venv/bin/python -m pytest tests/ -v
  cargo bench

## File Writing Note

Write tool fails with Korean directory path. Use Python or bash with
quoted paths for all file writes in this project.
