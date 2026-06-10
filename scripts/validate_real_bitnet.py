"""Validate trimat against real BitNet b1.58 checkpoint weights.

Downloads only a few representative linear-layer weights from
``microsoft/bitnet-b1.58-2B-4T-bf16`` via HTTP range requests (no 4.8 GB full
download), then measures:

  1. ACCURACY (the key question): how well trimat's ternary quantization
     preserves each layer's output, comparing trimat's **absmax** formula
     against BitNet's official **absmean** formula. If absmean is clearly
     better, quantize.rs should grow an absmean path.
  2. SPEED: qgemv vs NumPy at the model's real dimensions.

Run:  python scripts/validate_real_bitnet.py
Requires: torch, numpy, trimat (installed). Network access to huggingface.co.
"""
from __future__ import annotations

import json
import struct
import time
import urllib.request

import numpy as np
import torch

import trimat

REPO = "microsoft/bitnet-b1.58-2B-4T-bf16"
URL = f"https://huggingface.co/{REPO}/resolve/main/model.safetensors"

# Representative linear weights (varied shapes incl. GQA k/v projections).
LAYERS = [
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.self_attn.k_proj.weight",
    "model.layers.0.self_attn.o_proj.weight",
    "model.layers.0.mlp.gate_proj.weight",
    "model.layers.0.mlp.down_proj.weight",
]


def _get(url: str, start: int | None = None, end: int | None = None) -> bytes:
    headers = {}
    if start is not None:
        headers["Range"] = f"bytes={start}-{end}"
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60).read()


def read_header() -> tuple[dict, int]:
    """Return (tensor_metadata, data_section_start_offset)."""
    head = _get(URL, 0, 2_000_000)
    n = struct.unpack("<Q", head[:8])[0]
    meta = json.loads(head[8 : 8 + n])
    return meta, 8 + n


def fetch_weight(name: str, meta: dict, base: int) -> np.ndarray:
    """Range-download one bf16 tensor and return it as float32 (out, in)."""
    info = meta[name]
    s, e = info["data_offsets"]
    raw = _get(URL, base + s, base + e - 1)
    t = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).reshape(info["shape"])
    return t.to(torch.float32).numpy()


def quant_ternary(W: np.ndarray, mode: str) -> tuple[np.ndarray, float]:
    """Per-tensor ternary quantization. Returns (codes in {-1,0,1}, scale)."""
    if mode == "absmax":
        gamma = np.abs(W).max()
    elif mode == "absmean":
        gamma = np.abs(W).mean()
    else:
        raise ValueError(mode)
    if gamma == 0:
        return np.zeros_like(W), 1.0
    codes = np.clip(np.round(W / gamma), -1, 1)
    return codes.astype(np.float32), float(gamma)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a.ravel() @ b.ravel() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def rel_l2(ref: np.ndarray, y: np.ndarray) -> float:
    return float(np.linalg.norm(y - ref) / (np.linalg.norm(ref) + 1e-30))


def t_us(fn, *a, iters=50, trials=5, warmup=10):
    for _ in range(warmup):
        fn(*a)
    best = float("inf")
    for _ in range(trials):
        t0 = time.perf_counter()
        for _ in range(iters):
            fn(*a)
        best = min(best, (time.perf_counter() - t0) / iters * 1e6)
    return best


def main() -> None:
    print(f"backend: {trimat.cpu_features()}")
    print(f"model:   {REPO}\n")
    meta, base = read_header()
    rng = np.random.default_rng(0)

    print("=== WEIGHT DISTRIBUTION (is it already ~ternary?) ===")
    print(f"{'layer':<26} {'shape':>14} {'max|W|':>9} {'mean|W|':>9} {'near0%':>7}")
    weights = {}
    for name in LAYERS:
        W = fetch_weight(name, meta, base)
        weights[name] = W
        near0 = float((np.abs(W) < 0.5 * np.abs(W).mean()).mean() * 100)
        short = name.replace("model.layers.0.", "")
        print(f"{short:<26} {str(W.shape):>14} {np.abs(W).max():>9.4f} "
              f"{np.abs(W).mean():>9.4f} {near0:>6.1f}%")

    print("\n=== ACCURACY: output cosine sim vs full-precision (higher=better) ===")
    print(f"{'layer':<26} {'absmax':>9} {'absmean':>9} {'trimat':>9} {'qgemv':>9}")
    agg = {"absmax": [], "absmean": [], "trimat": [], "qgemv": []}
    for name, W in weights.items():
        out_f, in_f = W.shape
        x = rng.standard_normal(in_f).astype(np.float32)
        ref = (W.astype(np.float64) @ x.astype(np.float64))

        # numpy simulations of the two quantization formulas
        res = {}
        for mode in ("absmax", "absmean"):
            codes, g = quant_ternary(W, mode)
            res[mode] = cosine(ref, (codes * g) @ x)

        # trimat with the absmean path: exact f32 gemv and int8 qgemv. Should
        # now track the absmean simulation column instead of collapsing.
        t = trimat.pack(W, mode="absmean")
        res["trimat"] = cosine(ref, trimat.gemv(t, x))
        res["qgemv"] = cosine(ref, trimat.qgemv(t, x))

        for k in agg:
            agg[k].append(res[k])
        short = name.replace("model.layers.0.", "")
        print(f"{short:<26} {res['absmax']:>9.4f} {res['absmean']:>9.4f} "
              f"{res['trimat']:>9.4f} {res['qgemv']:>9.4f}")
    print(f"{'MEAN':<26} {np.mean(agg['absmax']):>9.4f} {np.mean(agg['absmean']):>9.4f} "
          f"{np.mean(agg['trimat']):>9.4f} {np.mean(agg['qgemv']):>9.4f}")

    print("\n=== SPEED: qgemv vs NumPy at real dims (ratio>1 = trimat faster) ===")
    for name in ["model.layers.0.mlp.gate_proj.weight",
                 "model.layers.0.mlp.down_proj.weight"]:
        W = weights[name]
        t = trimat.pack(W)
        x = rng.standard_normal(W.shape[1]).astype(np.float32)
        tri = t_us(trimat.qgemv, t, x)
        npt = t_us(lambda: W @ x)
        short = name.replace("model.layers.0.", "")
        print(f"{short:<26} {str(W.shape):>14}  trimat {tri:7.1f}us  "
              f"numpy {npt:7.1f}us  -> {npt/tri:.2f}x")


if __name__ == "__main__":
    main()
