# Phase 2 — Tokenizer

A byte-level BPE tokenizer trained on a ~200MB slice of FineWeb-Edu, vocab size 8192.

See [`../docs/glossary.md`](../docs/glossary.md) for definitions of BPE, byte-level, vocab size, special tokens.
See [`../docs/tutorial.md#1-tokens-and-tokenization`](../docs/tutorial.md) for intuition.

## Why these choices

- **Byte-level BPE** (GPT-style) — represents *any* UTF-8 input without out-of-vocab errors. Same family as GPT-2, Llama.
- **Vocab size 8192** — small. At 10M params, an embedding table of `8192 × 384 ≈ 3.1M` already uses ~31% of the param budget. A bigger vocab would crowd out the rest of the model.
- **Trained on FineWeb-Edu** — the same distribution the model will pretrain on. Train-test match matters for tokenizers too.
- **Special tokens reserved up front** — `<|endoftext|>`, `<|user|>`, `<|assistant|>`, `<|pad|>`. Adding them later is awkward because their IDs would shift.

## Steps to run

Phase 2 runs on a RunPod box (see [`../docs/compute.md`](../docs/compute.md) for why and how). The tokenizer training itself is CPU-bound — any RunPod GPU template works; you're paying for the box, not the GPU, in this phase.

```bash
# (One time per pod) — clone + install.
cd /workspace
git clone <your-repo> foundational-models
cd foundational-models/01-tiny-llm
curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.local/bin/env
uv sync

# 1. Download ~200MB of FineWeb-Edu. Needs HF login (free).
huggingface-cli login
uv run python tokenizer/download_corpus.py --target-mb 200      # ~5 min

# 2. Train the BPE tokenizer.
uv run python tokenizer/train_tokenizer.py --vocab-size 8192    # ~5-15 min

# 3. Verify.
uv run python tokenizer/inspect_tokenizer.py                    # ~30 sec
```

Phase 2 in total takes ~15-30 min on a cheap RunPod box (RTX 4090 spot, ~$0.34/hr → **<$0.20 total**).

## Verification checklist (from the plan)

After step 4, confirm:

- [ ] Tokenizer loads with vocab size = 8192.
- [ ] All four special tokens have IDs 0-3 (in declaration order).
- [ ] Sample tokenizations look reasonable (common words = 1 token, rare words split).
- [ ] **Round-trip test**: 100/100 random docs encode→decode without mismatch.
- [ ] **Compression ratio**: bytes-per-token between ~3.5 and ~4.5 for English web text. (Lower = worse; higher = better. GPT-2's tokenizer hits ~4.0 on web text, so a tiny custom tokenizer should land slightly worse but in the same ballpark.)

## Outputs

- `tokenizer/tiny-llm-bpe.json` — the trained tokenizer (committed to git? No — it's regenerable from the corpus; the `.gitignore` excludes it).
- `data/cache/corpus.txt` — the local training corpus slice (also gitignored).

## What this teaches

Working through this phase end-to-end gives you a concrete grasp of:
- How a tokenizer fits *into* a pipeline (text → IDs → model).
- Why vocab size is a *budget* trade-off, not a free hyperparameter.
- Why byte-level matters (the round-trip test will pass on emojis and accented characters).
- What the compression ratio number actually means and why it's worth tracking.
