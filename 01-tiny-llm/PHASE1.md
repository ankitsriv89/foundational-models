# Phase 1 — Model Architecture

The transformer itself. One file, [`model.py`](./model.py), top-to-bottom readable.

See [`docs/glossary.md`](./docs/glossary.md) for term definitions.
See [`docs/tutorial.md`](./docs/tutorial.md) for the intuition-building walkthrough.

## What's in `model.py`

```
model.py
├── ModelConfig         dataclass — all hyperparameters in one place
├── RMSNorm             normalization layer
├── build_rope_cache    precompute cos/sin tables for RoPE
├── apply_rope          rotate Q/K vectors by their position
├── Attention           GQA with PyTorch's fused F.scaled_dot_product_attention
├── SwiGLU              gated FFN
├── Block               pre-norm: x + attn(norm(x)) → x + ffn(norm(x))
└── TinyLLM             the whole model + forward + generate
```

## Headline numbers (default config)

```
vocab_size = 8192    # matches tokenizer/
block_size = 1024    # context length
n_layer    = 6
n_head     = 6       # query heads
n_kv_head  = 2       # K/V heads (GQA group size = 3)
n_embd     = 384     # hidden dimension
                     # → ~10.3M total params, ~7M non-embedding
```

## Why each component, in one sentence

| Component | One-line justification |
|---|---|
| **RMSNorm** | Simpler & faster than LayerNorm. Llama/Mistral/Qwen all use it. |
| **Pre-norm** | Stable to train at depth without LR warmup tricks. |
| **RoPE** | Relative positional info via rotation; generalizes to unseen sequence lengths. |
| **GQA** | Smaller KV cache during inference. Modern default in Llama-2/3. |
| **SwiGLU** | Gated FFN, slightly better quality than GELU. |
| **Tied embeddings** | Save ~3M params and act as mild regularizer. |
| **Scaled init on residual projections** | Keep activation magnitudes bounded as depth grows. |
| **`F.scaled_dot_product_attention`** | Fused kernel that auto-uses Flash Attention on supported GPUs. |

## Running the tests (locally is fine — CPU only)

```bash
cd /home/ankit/ai/foundational-models/01-tiny-llm
uv sync --group test                              # one-time install
uv run --group test python -m pytest tests/ -v
```

The full suite runs in ~10-20 seconds on a CPU. It catches:

- ✅ Param count = ~10M (the headline number)
- ✅ Forward pass shapes
- ✅ Initial loss ≈ log(vocab_size) — sanity check that init is sane
- ✅ Causal masking — future tokens can't leak into past logits
- ✅ RoPE: identity at position 0, preserves vector norm (rotation is orthogonal)
- ✅ GQA shape contract (with `n_head != n_kv_head` and the degenerate MHA case)
- ✅ Backward pass: every parameter receives a gradient (no dead weights)
- ✅ Generation extends the prompt without altering it

If any of these fail, do not proceed to Phase 4 — every minute of GPU rental
is wasted on a broken model.

## Reading order (recommended for learning)

1. `docs/glossary.md` — refresh definitions of RMSNorm, RoPE, GQA, SwiGLU, pre-norm, tied embeddings.
2. `model.py` top to bottom — every component has a comment block explaining *why* it exists.
3. `tests/test_model.py` — the tests double as executable documentation of the contracts.
4. `docs/tutorial.md` sections 4-5 — for the math/intuition behind transformer blocks and RoPE.

## What this does NOT include (yet)

These come in later phases:

- KV cache for fast inference (Phase 7)
- Mixed-precision training scaffolding (Phase 4)
- Checkpoint save/load helpers (Phase 4)
- Optimizer setup with AdamW + weight decay grouping (Phase 4)

`model.py` is intentionally just the model — pure architecture, no training
plumbing. Separating these makes both halves easier to understand.
