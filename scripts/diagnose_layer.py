"""Diagnose the +28% perplexity: original AutoBitLinear vs trimat BitLinear,
same weight, same input. High cosine (~0.99) => the gap is accumulation across
layers, not a per-layer implementation difference.
"""
import numpy as np
import torch
from transformers import AutoModelForCausalLM

from trimat.nn import BitLinear

REPO = "microsoft/bitnet-b1.58-2B-4T-bf16"
NAMES = [
    "model.layers.0.self_attn.q_proj",
    "model.layers.0.mlp.gate_proj",
    "model.layers.0.mlp.down_proj",
]


def cosine(a, b):
    a, b = a.ravel(), b.ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def main():
    model = AutoModelForCausalLM.from_pretrained(
        REPO, dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).eval()
    mods = {n: m for n, m in model.named_modules()}
    sample = mods[NAMES[0]]
    print(f"AutoBitLinear.online_quant = {getattr(sample, 'online_quant', '?')}, "
          f"rms_norm = {getattr(sample, 'rms_norm', None) is not None}", flush=True)

    print(f"{'layer':<26} {'T=1 (decode)':>14} {'T=16 (prefill)':>16}", flush=True)
    for name in NAMES:
        orig = mods[name]
        bl = BitLinear(orig.weight.float(), mode="absmean", quantized=True)
        cols = []
        for T in (1, 16):
            x = torch.randn(1, T, orig.in_features, dtype=torch.bfloat16)
            with torch.no_grad():
                y_o = orig(x).float().numpy()
                y_t = bl(x).float().numpy()
            cols.append(cosine(y_o, y_t))
        short = name.replace("model.layers.0.", "")
        print(f"{short:<26} {cols[0]:>14.4f} {cols[1]:>16.4f}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
