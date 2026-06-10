"""Decompose the trimat-vs-BitNet mismatch step by step to localize the bug.

For one AutoBitLinear layer:
  1. Reproduce BitNet's online forward in numpy f64 (WeightQuant + ActQuant +
     F.linear). Check it matches the real torch layer (cosine ~1.0).
  2. Compare trimat's weight codes / activation codes / output against that
     reference to see exactly where they diverge.
"""
import numpy as np
import torch

from trimat.nn import BitLinear
import trimat

REPO = "microsoft/bitnet-b1.58-2B-4T-bf16"
NAME = "model.layers.0.self_attn.q_proj"


def cos(a, b):
    a, b = a.ravel(), b.ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def main():
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        REPO, dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).eval()
    orig = dict(model.named_modules())[NAME]
    print(f"online_quant={orig.online_quant}, has weight_scale="
          f"{hasattr(orig, 'weight_scale')}", flush=True)

    in_f = orig.in_features
    torch.manual_seed(0)
    x = torch.randn(1, 8, in_f, dtype=torch.bfloat16)

    # --- torch original ---
    with torch.no_grad():
        y_orig = orig(x).float().double().numpy()

    # --- numpy reproduction of BitNet online forward ---
    Wf = orig.weight.detach().float().double().numpy()  # (out, in)
    xf = x.float().double().numpy()                    # (1, 8, in)
    w_scale = 1.0 / max(np.abs(Wf).mean(), 1e-5)        # WeightQuant: 1/mean|W|
    w_codes_bn = np.clip(np.round(Wf * w_scale), -1, 1) # numpy round = half-even
    w_deq = w_codes_bn / w_scale
    a_scale = 127.0 / np.clip(np.abs(xf).max(-1, keepdims=True), 1e-5, None)
    x_q_bn = np.clip(np.round(xf * a_scale), -128, 127)
    x_deq = x_q_bn / a_scale
    y_ref = x_deq @ w_deq.T
    print(f"[1] numpy-repro vs torch-orig cosine: {cos(y_ref, y_orig):.6f} "
          f"(want ~1.0 -> my BitNet model is correct)", flush=True)

    # --- trimat ---
    bl = BitLinear(orig.weight.float(), mode="absmean", quantized=True)
    with torch.no_grad():
        y_tri = bl(x).float().double().numpy()
    print(f"[2] trimat vs torch-orig cosine:       {cos(y_tri, y_orig):.6f}", flush=True)
    print(f"[3] trimat vs numpy-repro cosine:      {cos(y_tri, y_ref):.6f}", flush=True)

    # --- weight codes: trimat absmean vs BitNet WeightQuant ---
    mean_abs = np.abs(Wf).mean()
    w_codes_tri = np.clip(np.round(Wf / mean_abs), -1, 1)  # numpy round (half-even)
    mism = float((w_codes_bn != w_codes_tri).mean())
    print(f"[4] weight code mismatch fraction:     {mism:.6f} "
          f"(w_scale={w_scale:.4f}, 1/mean={1.0/mean_abs:.4f})", flush=True)

    # --- exact f32 path (quantized=False): isolates weight error from activation ---
    bl_exact = BitLinear(orig.weight.float(), mode="absmean", quantized=False)
    with torch.no_grad():
        y_exact = bl_exact(x).float().double().numpy()
    # reference with exact (un-quantized) activations, same weight codes
    y_ref_exactact = xf @ w_deq.T
    print(f"[5] trimat-exact vs (ternaryW·realX):  {cos(y_exact, y_ref_exactact):.6f} "
          f"(weight path only)", flush=True)

    # --- extract trimat's ACTUAL packed codes via identity gemm: W·I = codes*scale ---
    Wf32 = orig.weight.detach().float().numpy()
    t = trimat.pack(np.ascontiguousarray(Wf32), "absmean")
    eye = np.eye(in_f, dtype=np.float32)
    codes_scaled_tri = trimat.gemm(t, eye)            # (out, in) = codes * scale
    scale_tri = np.median(np.abs(codes_scaled_tri[codes_scaled_tri != 0]))
    codes_tri_actual = np.round(codes_scaled_tri / scale_tri)
    mism2 = float((codes_tri_actual != w_codes_bn).mean())
    print(f"[6] trimat ACTUAL codes vs BitNet codes mismatch: {mism2:.6f} "
          f"(scale_tri≈{scale_tri:.4f} vs mean={1.0/w_scale:.4f})", flush=True)
    # where do they differ — sign flips vs zero/nonzero?
    nz_t = codes_tri_actual != 0
    nz_b = w_codes_bn != 0
    print(f"    nonzero count trimat={int(nz_t.sum())} bitnet={int(nz_b.sum())} "
          f"(of {w_codes_bn.size})", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
