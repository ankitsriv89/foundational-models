# Compute — Where to Run What

This project assumes **all training and data-prep happens on a rented RunPod GPU**. The local laptop is for writing code, reading docs, and reviewing results — not for execution.

## Why rent

Pretraining and SFT are GPU-bound. CPU training of even a 10M-param transformer on 1B tokens would take months. On an H100 it's ~10-20 hours. The whole project fits comfortably in a **~$25-60 GPU budget**.

## Provider: RunPod

Used as the default for every project in this repo. Reasons:

- Predictable pricing, decent UX, good docs.
- Spot ("Community Cloud") instances ~half the price of on-demand ("Secure Cloud").
- Pre-built PyTorch images so the container starts ready to go.
- Persistent volumes mean you don't re-download datasets across sessions.

Pricing as of 2026 (will drift — always check current rates):

| GPU | On-demand | Spot | Notes |
|---|---|---|---|
| RTX 4090 (24 GB) | $0.69/hr | $0.34/hr | Fine for 10M-param work. Slowest acceptable choice. |
| A100 40GB | $1.19/hr | $0.74/hr | Solid middle ground. |
| A100 80GB | $1.79/hr | $1.19/hr | Same speed as 40GB; only matters if memory-constrained. |
| H100 80GB | $2.99/hr | $1.99/hr | **Recommended.** 2-3× faster than A100. Often best $/token. |

## Workflow

1. **Write code locally**, push to GitHub. Test on tiny inputs (the smoke-test pattern this project uses).
2. **Spin up a RunPod box** when ready to train. Use the "PyTorch 2.4" template — it has CUDA, Python, and PyTorch pre-installed.
3. **Pick a persistent volume** (e.g. 100 GB) so dataset shards survive between sessions. Mount it at `/workspace`.
4. **Clone the repo** inside the pod: `git clone <your-repo> /workspace/foundational-models`.
5. **Install deps**: `cd 01-tiny-llm && uv sync --group train`.
6. **Run the phase scripts** in order (tokenizer → data → train → eval).
7. **Push checkpoints to HuggingFace Hub** so they survive after you destroy the pod.
8. **Destroy the pod** when done. Spot instances bill per minute — never leave one idle.

## What lives on the laptop vs. the pod

| Artifact | Laptop | Pod |
|---|---|---|
| Source code | ✅ source of truth, pushed to GitHub | ✅ cloned, may have small edits |
| `pyproject.toml` / `uv.lock` | ✅ | ✅ (the lock is what makes envs match) |
| `data/cache/corpus.txt` (~200MB raw text) | ❌ | ✅ |
| `data/shards/*.bin` (~2-4 GB tokenized) | ❌ | ✅ on persistent volume |
| `checkpoints/*.pt` (~40MB each × many) | ❌ | ✅ on persistent volume, mirrored to HF Hub |
| `tokenizer/tiny-llm-bpe.json` | optional copy | ✅ generated here |
| `wandb` logs | ❌ | streamed to W&B cloud |
| Final model card + small demo files | ✅ pulled down for review | originated here |

## Cost guardrails

- **Always use spot ("Community Cloud") pricing** unless the workload is short and you need reliability. Spot can be reclaimed but is half the cost. Checkpointing every ~1000 steps makes interruption cheap.
- **Save state aggressively.** When the pod dies, anything not on a persistent volume or pushed to HF Hub is gone.
- **Bundle phases into one rental session** when possible. Spinning up a box repeatedly costs you the setup time (~5-10 min each).
- **Test your scripts on tiny inputs first** — the smoke-test pattern in this repo (200MB corpus, 2048 vocab) catches most bugs before you've burned any GPU time.
- **Use `nvidia-smi` and `htop`** in the pod to confirm you're actually GPU-bound, not data-loading-bound. A 10M model that's data-bound is a script bug, not a hardware limit.

## Per-phase compute plan

| Phase | Where | Approx time | Approx cost |
|---|---|---|---|
| 1. Write `model.py` | laptop | hours of human time | $0 |
| 2. Tokenizer (download + train) | RunPod (any GPU) | 15-30 min | <$1 |
| 3. Pre-tokenize 1B tokens to shards | RunPod (any GPU; CPU-bound) | 1-2 hrs | $1-3 |
| 4. Pretraining 1B tokens | RunPod H100 spot | 10-20 hrs | $20-40 |
| 5. Evaluation | RunPod (same pod as Phase 4) | 1-2 hrs | $2-4 |
| 6. SFT | RunPod (same pod as Phase 4) | 2-4 hrs | $4-8 |
| 7. Inference demo on HuggingFace Spaces | HF Spaces free tier | n/a | $0 |
| | | **Total** | **~$30-55** |

## Minimal RunPod pod recipe (template for now, will be expanded)

```bash
# Inside the pod, first time only:
cd /workspace
git clone <your-repo> foundational-models
cd foundational-models/01-tiny-llm

# Install uv and sync deps.
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv sync --group train

# Log in to services you'll need.
huggingface-cli login   # for FineWeb-Edu, model card upload
wandb login             # for loss curves

# Now you can run any phase script.
uv run python tokenizer/download_corpus.py --target-mb 200
uv run python tokenizer/train_tokenizer.py
uv run python tokenizer/inspect_tokenizer.py
```

The phase-specific READMEs will repeat the exact commands when each phase is ready.
