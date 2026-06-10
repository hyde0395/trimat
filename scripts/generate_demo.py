"""Step 3-4: run real BitNet-2B end-to-end with trimat kernels.

Loads microsoft/bitnet-b1.58-2B-4T-bf16, swaps every AutoBitLinear for a trimat
BitLinear (absmean weights, int8 activations), then generates text on CPU and
reports tokens/sec. The lm_head (a plain Linear, tied to the embedding) is left
full-precision, as BitNet does.

Run after the weights are cached. CPU 2B is slow — keep max_new_tokens small.
"""
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from trimat.nn import BitLinear

REPO = "microsoft/bitnet-b1.58-2B-4T-bf16"
PROMPT = "The capital of France is"
MAX_NEW = 16

# Fixed passage for perplexity (original BitNet vs trimat-swapped, same text).
PPL_TEXT = (
    "The history of artificial intelligence began in antiquity, with myths and "
    "stories of artificial beings endowed with intelligence by master craftsmen. "
    "Modern machine learning research was founded at a workshop held on the campus "
    "of Dartmouth College in the summer of 1956. Since then, the field has gone "
    "through several cycles of optimism and disappointment, and is today one of the "
    "most active areas of computer science research."
)


def perplexity(model, ids) -> float:
    """exp(mean token NLL) of `ids` under `model` — lower is better."""
    with torch.no_grad():
        out = model(input_ids=ids, labels=ids)
    return float(torch.exp(out.loss))


def swap_bitlinear(module) -> int:
    """Recursively replace AutoBitLinear children with trimat BitLinear."""
    n = 0
    for name, child in list(module.named_children()):
        if type(child).__name__ == "AutoBitLinear":
            # latent weight -> float32 -> absmean ternary; int8 activation path
            bl = BitLinear(child.weight.float(), mode="absmean", quantized=True)
            setattr(module, name, bl)
            n += 1
        else:
            n += swap_bitlinear(child)
    return n


def main() -> None:
    print("loading tokenizer + model (cached)...", flush=True)
    tok = AutoTokenizer.from_pretrained(REPO)
    model = AutoModelForCausalLM.from_pretrained(
        REPO, dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    model.eval()

    # Perplexity: measure the ORIGINAL BitNet first, then swap and re-measure.
    ppl_ids = tok(PPL_TEXT, return_tensors="pt").input_ids
    print(f"measuring original perplexity ({ppl_ids.shape[1]} tokens)...", flush=True)
    ppl_orig = perplexity(model, ppl_ids)
    print(f"  original BitNet perplexity: {ppl_orig:.3f}", flush=True)

    n = swap_bitlinear(model)
    print(f"replaced {n} AutoBitLinear -> trimat BitLinear (absmean, int8)", flush=True)

    ppl_tri = perplexity(model, ppl_ids)
    delta = (ppl_tri / ppl_orig - 1) * 100
    print(f"  trimat perplexity:          {ppl_tri:.3f}  ({delta:+.1f}% vs original)",
          flush=True)

    ids = tok(PROMPT, return_tensors="pt").input_ids
    with torch.no_grad():
        # one warmup token (JIT caches, allocator) then time the rest
        _ = model.generate(ids, max_new_tokens=1, do_sample=False)
        t0 = time.perf_counter()
        out = model.generate(ids, max_new_tokens=MAX_NEW, do_sample=False)
        dt = time.perf_counter() - t0

    new = out.shape[1] - ids.shape[1]
    print("=" * 60, flush=True)
    print("PROMPT :", PROMPT, flush=True)
    print("OUTPUT :", tok.decode(out[0], skip_special_tokens=True), flush=True)
    print("=" * 60, flush=True)
    print(f"generated {new} tokens in {dt:.1f}s  ->  {new / dt:.2f} tok/s (trimat, CPU)",
          flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
