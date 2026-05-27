"""Supervised fine-tuning (SFT) of the pretrained tiny LLM.

Differences from pretraining:
  * Loads the base model checkpoint as init (not random weights).
  * Reads data/sft/{train.bin, train_mask.bin} produced by prepare_sft.py.
  * Computes loss only on tokens where mask==1 (the assistant response).
  * Trains for a fixed number of epochs over the SFT dataset (not a step budget).
  * Lower LR — we don't want to wipe out the pretraining knowledge.

Usage (RunPod):
    uv run --group train python train_sft.py --config configs/nano-10m-sft.yaml
"""

from __future__ import annotations

import argparse
import math
import os
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from model import ModelConfig, TinyLLM


def load_config(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def build_optimizer(model: torch.nn.Module, lr: float, weight_decay: float, betas: tuple[float, float]) -> torch.optim.Optimizer:
    decay, no_decay = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    fused = torch.cuda.is_available()
    return torch.optim.AdamW(groups, lr=lr, betas=betas, fused=fused)


def lr_at(step: int, max_lr: float, min_lr: float, warmup: int, max_steps: int) -> float:
    if step < warmup:
        return max_lr * (step + 1) / warmup
    if step >= max_steps:
        return min_lr
    progress = (step - warmup) / max(1, max_steps - warmup)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (max_lr - min_lr)


class SFTSampler:
    """Random fixed-length windows over the concatenated SFT corpus."""

    def __init__(self, ids: np.ndarray, mask: np.ndarray, block_size: int, batch_size: int, device: str, rng: np.random.Generator) -> None:
        assert ids.shape == mask.shape
        self.ids = ids
        self.mask = mask
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device
        self.rng = rng

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        max_start = len(self.ids) - self.block_size - 1
        starts = self.rng.integers(0, max_start, size=self.batch_size)
        x = np.stack([self.ids[s:s + self.block_size] for s in starts]).astype(np.int64)
        y = np.stack([self.ids[s + 1:s + self.block_size + 1] for s in starts]).astype(np.int64)
        m = np.stack([self.mask[s + 1:s + self.block_size + 1] for s in starts]).astype(np.bool_)
        xt = torch.from_numpy(x).to(self.device, non_blocking=True)
        yt = torch.from_numpy(y).to(self.device, non_blocking=True)
        mt = torch.from_numpy(m).to(self.device, non_blocking=True)
        return xt, yt, mt


def masked_loss(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Cross-entropy loss averaged over masked positions only.

    logits:  (B, T, V)
    targets: (B, T)   int64
    mask:    (B, T)   bool — True where loss should apply.
    """
    B, T, V = logits.shape
    flat_logits = logits.view(B * T, V)
    flat_targets = targets.view(B * T)
    flat_mask = mask.view(B * T)
    # F.cross_entropy with reduction='none' gives per-token loss; we then mask + average.
    per_tok = F.cross_entropy(flat_logits, flat_targets, reduction="none")
    masked = per_tok * flat_mask.float()
    denom = flat_mask.float().sum().clamp_min(1.0)
    return masked.sum() / denom


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    cfg = load_config(args.config)
    tcfg = cfg["train"]
    dcfg = cfg["data"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: SFT on CPU is impractical. Use a GPU.")
    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    dtype = dtype_map[tcfg.get("dtype", "bfloat16")]
    if dtype == torch.bfloat16 and device == "cuda" and not torch.cuda.is_bf16_supported():
        print("bf16 unsupported; falling back to fp16")
        dtype = torch.float16
    ctx = torch.amp.autocast(device_type="cuda", dtype=dtype) if device == "cuda" else torch.amp.autocast(device_type="cpu", dtype=torch.float32)

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    # ---------- init from base ckpt ----------
    init_from = Path(cfg["init_from"])
    if not init_from.exists():
        raise SystemExit(f"base checkpoint not found at {init_from} — finish pretraining first")
    base = torch.load(init_from, map_location=device, weights_only=True)
    mcfg = ModelConfig(**base["config"]["model"])
    model = TinyLLM(mcfg).to(device)
    model.load_state_dict(base["model"])
    print(f"loaded base from {init_from} (step={base.get('step', '?')}, val={base.get('best_val', float('nan')):.4f})")
    print(f"model: {model.num_params():,} params")

    optimizer = build_optimizer(
        model,
        lr=float(tcfg["lr"]),
        weight_decay=float(tcfg["weight_decay"]),
        betas=(float(tcfg["beta1"]), float(tcfg["beta2"])),
    )

    # ---------- data ----------
    sft_dir = Path(dcfg["sft_dir"])
    ids = np.memmap(sft_dir / dcfg["ids_file"], dtype=np.uint16, mode="r")
    mask = np.memmap(sft_dir / dcfg["mask_file"], dtype=np.uint8, mode="r")
    assert len(ids) == len(mask), f"ids/mask length mismatch ({len(ids)} vs {len(mask)})"
    print(f"sft corpus: {len(ids):,} tokens, loss on {int(np.asarray(mask).sum()):,} ({100*float(np.asarray(mask).mean()):.1f}%)")

    block_size = int(tcfg["block_size"])
    micro_batch = int(tcfg["micro_batch_size"])
    grad_accum = int(tcfg["grad_accum_steps"])
    tokens_per_step = micro_batch * block_size * grad_accum

    # Steps per epoch = corpus_tokens / tokens_per_step
    steps_per_epoch = max(1, len(ids) // tokens_per_step)
    max_steps = steps_per_epoch * int(tcfg["num_epochs"])
    print(f"steps/epoch ≈ {steps_per_epoch}, total steps = {max_steps}")

    sampler = SFTSampler(np.asarray(ids), np.asarray(mask), block_size, micro_batch, device, rng)

    # ---------- wandb ----------
    wb_cfg = tcfg.get("wandb", {})
    use_wandb = bool(wb_cfg.get("enabled")) and os.environ.get("WANDB_DISABLED") != "true"
    if use_wandb:
        import wandb
        wandb.init(
            project=wb_cfg.get("project", "tiny-llm"),
            name=wb_cfg.get("run_name", "sft"),
            config={"model": asdict(mcfg), "train": tcfg, "data": dcfg, "init_from": str(init_from)},
        )

    out_dir = Path(tcfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "latest.pt"

    max_lr = float(tcfg["lr"])
    min_lr = float(tcfg["min_lr"])
    warmup = int(tcfg["warmup_steps"])
    log_interval = int(tcfg["log_interval"])
    save_interval = int(tcfg["save_interval"])
    grad_clip = float(tcfg["grad_clip"])

    model.train()
    t0 = time.time()
    for step in range(max_steps):
        lr = lr_at(step, max_lr, min_lr, warmup, max_steps)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0
        for _ in range(grad_accum):
            x, y, m = sampler.next_batch()
            with ctx:
                logits, _ = model(x)
                loss = masked_loss(logits, y, m) / grad_accum
            loss.backward()
            loss_accum += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        if step % log_interval == 0:
            elapsed = time.time() - t0
            tok_per_sec = tokens_per_step * (step + 1) / max(elapsed, 1e-6)
            print(f"step {step:5d}/{max_steps} | loss {loss_accum:.4f} | lr {lr:.2e} | {tok_per_sec/1e3:.1f}k tok/s")
            if use_wandb:
                import wandb
                wandb.log({"sft/loss": loss_accum, "lr": lr, "tokens_per_sec": tok_per_sec, "step": step})

        if step > 0 and step % save_interval == 0:
            torch.save(
                {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "step": step, "best_val": float("nan"), "config": base["config"]},
                ckpt_path,
            )

    torch.save(
        {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
         "step": max_steps, "best_val": float("nan"), "config": base["config"]},
        ckpt_path,
    )
    print(f"SFT done. saved {ckpt_path}")


if __name__ == "__main__":
    main()
