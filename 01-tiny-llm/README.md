# 01 — Tiny LLM

A ~10M-parameter decoder-only transformer, pretrained from scratch on FineWeb-Edu and instruction-tuned via SFT.

**Why this exists:** to demonstrate end-to-end understanding of how foundation models are actually built — not just consumed via an API.

## What's in here

- **Modern architecture** (2025-2026 table stakes, not GPT-2 vanilla): RoPE, RMSNorm, SwiGLU, GQA, Flash Attention.
- **Custom BPE tokenizer** trained on the same corpus the model sees.
- **Two-stage training**: pretraining on ~1B tokens of FineWeb-Edu, then SFT on a public instruction dataset.
- **Standard evaluation** via EleutherAI's `lm-evaluation-harness`.
- **Inference demo** via FastAPI + Gradio on HuggingFace Spaces.
- **Learning docs** in [`docs/`](./docs/) — glossary and tutorial, beginner-friendly.

## Status

All phases scaffolded. Training runs on RunPod — see [`docs/runbook.md`](./docs/runbook.md).

- ✅ Phase 1 — model architecture ([`model.py`](./model.py), [`PHASE1.md`](./PHASE1.md))
- ✅ Phase 2 — tokenizer ([`tokenizer/`](./tokenizer/))
- ✅ Phase 3 — data sharding ([`data/prepare_pretrain.py`](./data/prepare_pretrain.py), [`data/prepare_sft.py`](./data/prepare_sft.py))
- ✅ Phase 4 — pretraining ([`train_pretrain.py`](./train_pretrain.py), [`configs/nano-10m-pretrain.yaml`](./configs/nano-10m-pretrain.yaml))
- ✅ Phase 5 — eval ([`eval/evaluate_ppl.py`](./eval/evaluate_ppl.py), [`eval/generate_samples.py`](./eval/generate_samples.py))
- ✅ Phase 6 — SFT ([`train_sft.py`](./train_sft.py), [`configs/nano-10m-sft.yaml`](./configs/nano-10m-sft.yaml))
- ✅ Phase 7 — serving ([`serve/app.py`](./serve/app.py))

## Layout

```
01-tiny-llm/
├── model.py                   # transformer (Phase 1)
├── PHASE1.md                  # Phase 1 doc
├── tests/test_model.py        # shape/correctness tests
├── tokenizer/
│   ├── download_corpus.py     # stream FineWeb-Edu slice
│   ├── train_tokenizer.py     # byte-level BPE
│   ├── inspect_tokenizer.py   # round-trip + compression check
│   └── README.md
├── data/
│   ├── prepare_pretrain.py    # tokenize → uint16 shards
│   └── prepare_sft.py         # format Dolly with chat template + loss mask
├── train_pretrain.py          # AdamW + cosine LR + grad accum + amp
├── train_sft.py               # SFT with loss-masked CE
├── eval/
│   ├── evaluate_ppl.py        # val loss, PPL, bits-per-byte
│   ├── generate_samples.py    # fixed prompts → sampled continuations
│   └── run_harness.py         # documented lm-eval-harness workflow
├── serve/app.py               # FastAPI + Gradio streaming chat
├── configs/                   # nano-10m-pretrain.yaml, nano-10m-sft.yaml
└── docs/
    ├── glossary.md            # every term, defined
    ├── tutorial.md            # step-by-step walkthrough
    ├── compute.md             # RunPod rental guide, per-phase cost plan
    └── runbook.md             # end-to-end pipeline recipe
```

## Where this runs

Everything except writing code happens on a **rented RunPod GPU**. See [`docs/compute.md`](./docs/compute.md) for the per-phase compute plan, cost breakdown, and pod setup recipe.

## Headline numbers (target)

| | |
|---|---|
| Params | ~10M |
| Layers | 6 |
| Hidden dim | 384 |
| Attention heads | 6 (with GQA → 2 KV heads) |
| Context length | 1024 |
| Vocab size | 8192 (custom BPE) |
| Pretraining tokens | ~1B (FineWeb-Edu) |
| SFT examples | ~20–50K (Ultrachat/Dolly/Alpaca) |
| Target GPU budget | ~15–30 H100-hours, $25–60 |

## How to follow along

If you're new to LLM internals, read in this order:
1. [`docs/glossary.md`](./docs/glossary.md) — definitions for every term used in this project.
2. [`docs/tutorial.md`](./docs/tutorial.md) — narrative walkthrough that builds intuition.
3. Then the code, top-down: `model.py` → `train_pretrain.py` → `train_sft.py`.
