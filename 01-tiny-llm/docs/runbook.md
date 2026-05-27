# RunPod Runbook — End-to-End Pipeline

Step-by-step recipe to run the entire project on a single RunPod box. Follow top to bottom.

For *why* RunPod and the per-phase cost plan, see [`compute.md`](./compute.md).

## 0. One-time prep on your laptop

```bash
# Push the repo to GitHub (one time).
cd /home/ankit/ai/foundational-models
git init && git branch -m main
git add -A && git commit -m "initial commit"
gh repo create ankitsriv89/foundational-models --public --source=. --push
```

You'll also need (free) accounts at:
- **HuggingFace** — for dataset access, model card upload. `huggingface-cli login` later.
- **Weights & Biases** — for loss curves. `wandb login` later.

## 1. Spin up the pod

In the RunPod web console:

1. **Pod template**: `RunPod PyTorch 2.4` (or latest PyTorch image).
2. **GPU**: `H100 PCIe 80GB` — Community Cloud (spot). ~$1.99/hr.
3. **Volume**: 100 GB at `/workspace`. Persistent across sessions.
4. **Disk**: 30 GB container disk.
5. Deploy and SSH in (`ssh root@<pod-ip> -p <port>` — RunPod shows the command).

## 2. Bootstrap the pod (5-10 min)

```bash
cd /workspace

# Install uv.
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# Clone your repo.
git clone https://github.com/ankitsriv89/foundational-models.git
cd foundational-models/01-tiny-llm

# Install all training-time deps. ~3-5 min.
uv sync --group train

# Log in to HF (for datasets) and W&B (for logging).
uv run huggingface-cli login   # paste token
uv run wandb login              # paste token
```

## 3. Sanity-check the model code (~30 sec)

Run the model tests *before* burning real GPU time on training:

```bash
uv sync --group test  # adds pytest
uv run --group test python -m pytest tests/ -v
```

All ~13 tests should pass. If `test_default_config_is_about_10M_params` fails, the model.py was edited; reconcile before continuing.

## 4. Phase 2 — Tokenizer (~15-30 min, <$1)

```bash
uv run python tokenizer/download_corpus.py --target-mb 200
uv run python tokenizer/train_tokenizer.py --vocab-size 8192
uv run python tokenizer/inspect_tokenizer.py
```

Verify:
- Vocab size = 8192
- All 4 special tokens at IDs 0-3
- 100/100 round-trip success
- Bytes-per-token ≈ 3.5-4.5

Output: `tokenizer/tiny-llm-bpe.json` (≈ 1 MB).

## 5. Phase 3 — Data sharding (~1-2 hours, $2-4)

```bash
uv run python data/prepare_pretrain.py \
    --target-tokens 1_000_000_000 \
    --val-tokens 5_000_000 \
    --shard-tokens 100_000_000

uv run python data/prepare_sft.py
```

Output:
- `data/shards/train_000000.bin` ... `train_000009.bin` (10 × 200 MB ≈ 2 GB)
- `data/shards/val_000000.bin` (~10 MB)
- `data/sft/train.bin`, `train_mask.bin` (~5 MB total)

## 6. Phase 4 — Pretraining (~10-20 hrs, $20-40)

```bash
# Quick smoke test first: 50 steps, no W&B, no checkpointing.
# Confirm loss decreases from ~9 (random init for 8192 vocab) toward something lower.
WANDB_DISABLED=true uv run --group train python train_pretrain.py \
    --config configs/nano-10m-pretrain.yaml 2>&1 | head -100

# If smoke test looks healthy (loss decreasing, no NaN, ~50-200k tok/s on H100):
# launch the full run inside a tmux session so the SSH disconnect doesn't kill it.
tmux new -s train
uv run --group train python train_pretrain.py --config configs/nano-10m-pretrain.yaml
# Ctrl-B then D to detach. Reattach later with: tmux attach -t train
```

Monitor:
- W&B dashboard for loss curves.
- `nvidia-smi` (in another tmux pane) — GPU util should be 80-95%. If lower, you're data-bound; investigate.

Output: `checkpoints/pretrain/best.pt` and `checkpoints/pretrain/latest.pt` (~40 MB each).

## 7. Phase 5 — Evaluation (~30 min, <$2)

```bash
uv run --group train python eval/evaluate_ppl.py \
    --ckpt checkpoints/pretrain/best.pt

uv run --group train python eval/generate_samples.py \
    --ckpt checkpoints/pretrain/best.pt \
    --mode base \
    --out eval/samples_base.md
```

Save `eval/samples_base.md` — it goes in the portfolio README.

Optional (advanced): `eval/run_harness.py` — see its docstring for the lm-eval-harness export workflow.

## 8. Phase 6 — SFT (~2-4 hrs, $4-8)

```bash
tmux new -s sft
uv run --group train python train_sft.py --config configs/nano-10m-sft.yaml
```

Then re-run generation in SFT mode for the before/after comparison:

```bash
uv run --group train python eval/generate_samples.py \
    --ckpt checkpoints/sft/latest.pt \
    --mode sft \
    --out eval/samples_sft.md
```

The diff between `samples_base.md` and `samples_sft.md` is the portfolio's hero artifact.

## 9. Phase 7 — Push artifacts off the pod

Before destroying the pod, save everything important:

```bash
# Push checkpoints to HuggingFace Hub.
uv run huggingface-cli upload ankitsriv89/tiny-llm-base \
    checkpoints/pretrain/best.pt model.safetensors-not-quite-but-fine

uv run huggingface-cli upload ankitsriv89/tiny-llm-sft \
    checkpoints/sft/latest.pt model.safetensors-not-quite-but-fine

# Push sample outputs and any tweaks to GitHub.
git add eval/samples_base.md eval/samples_sft.md
git commit -m "add eval samples from training run"
git push
```

## 10. Tear down

```bash
exit  # leave the pod SSH
```

Stop the pod in the RunPod console. **Don't just suspend — billing continues.** Destroy unless you're coming back within a few hours.

## Cost recap

| Phase | Time on H100 spot | Cost |
|---|---|---|
| Pod bootstrap | 10 min | $0.35 |
| 2. Tokenizer | 30 min | $1.00 |
| 3. Data sharding | 90 min | $3.00 |
| 4. Pretraining | 15 hrs | $30.00 |
| 5. Evaluation | 30 min | $1.00 |
| 6. SFT | 3 hrs | $6.00 |
| 7. Push artifacts | 15 min | $0.50 |
| | **~$42** | |

Comfortably inside the $50-80 budget.

## Troubleshooting

**OOM during pretraining**: drop `micro_batch_size` from 64 to 32 in the config, double `grad_accum_steps` to 16. Same tokens-per-step, half the memory.

**Loss is NaN**: usually a learning-rate spike. Restart from the last good checkpoint (use `--resume`) and lower `lr` 2x in the config.

**Slow tokens/sec (< 50k on H100)**: data loader is bottlenecking. Confirm shards are on the persistent volume (not network-mounted). Check `htop` — if CPU is pegged, the issue is data; if not, GPU is the bottleneck (usually fine).

**`huggingface-cli` 401**: regenerate token at huggingface.co/settings/tokens and re-login.

**Lost work after pod reclaim**: spot instances can be killed. The script checkpoints every `save_interval` steps. Resume with `--resume`.
