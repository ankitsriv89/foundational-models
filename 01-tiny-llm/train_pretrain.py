"""Pretrain the tiny LLM on shard files written by data/prepare_pretrain.py.

Usage (on a RunPod H100):
    uv run --group train python train_pretrain.py --config configs/nano-10m-pretrain.yaml

Resume:
    uv run --group train python train_pretrain.py --config configs/nano-10m-pretrain.yaml --resume

Design notes:
  * Single-GPU. No DDP. At 10M params a single H100 is plenty.
  * Mixed precision via torch.amp.autocast — bf16 on H100/A100, fp16 fallback.
  * Gradient accumulation gets us to ~0.5M tokens/step on a 64-batch micro_batch.
  * Data is read via np.memmap so we never load shards into RAM.
  * AdamW with the standard "no weight decay on 1D params (biases, norms)" trick.
  * Cosine LR schedule with linear warmup.
  * Checkpoint = model + optimizer + step (resumable).
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import yaml

from model import ModelConfig, TinyLLM


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def load_config(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Data loading via memmap
# ---------------------------------------------------------------------------


class ShardLoader:
    """Round-robin reader over a set of .bin shards (uint16 token IDs).

    Each call to next_batch() returns (x, y) where:
        x: (B, T) int64 input tokens
        y: (B, T) int64 target tokens (x shifted by 1)

    Shards are memory-mapped, so we never load them fully into RAM.
    """

    def __init__(self, shard_paths: list[Path], block_size: int, batch_size: int, device: str, rng: np.random.Generator) -> None:
        assert shard_paths, "no shards found"
        self.shards = [np.memmap(p, dtype=np.uint16, mode="r") for p in shard_paths]
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device
        self.rng = rng

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        shard = self.shards[self.rng.integers(len(self.shards))]
        # Pick batch_size random starting positions in this shard.
        # We need block_size + 1 tokens (x is [0:T], y is [1:T+1]).
        max_start = len(shard) - self.block_size - 1
        starts = self.rng.integers(0, max_start, size=self.batch_size)
        x = np.stack([shard[s:s + self.block_size] for s in starts]).astype(np.int64)
        y = np.stack([shard[s + 1:s + self.block_size + 1] for s in starts]).astype(np.int64)
        x_t = torch.from_numpy(x).to(self.device, non_blocking=True)
        y_t = torch.from_numpy(y).to(self.device, non_blocking=True)
        return x_t, y_t


# ---------------------------------------------------------------------------
# Optimizer with weight-decay grouping
# ---------------------------------------------------------------------------


def build_optimizer(model: torch.nn.Module, lr: float, weight_decay: float, betas: tuple[float, float]) -> torch.optim.Optimizer:
    """AdamW with weight decay on 2D+ params only (skip biases, norms)."""
    decay, no_decay = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    # `fused=True` is faster on CUDA when available.
    fused = torch.cuda.is_available()
    return torch.optim.AdamW(groups, lr=lr, betas=betas, fused=fused)


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------


def lr_at(step: int, max_lr: float, min_lr: float, warmup: int, max_steps: int) -> float:
    if step < warmup:
        return max_lr * (step + 1) / warmup
    if step >= max_steps:
        return min_lr
    progress = (step - warmup) / max(1, max_steps - warmup)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (max_lr - min_lr)


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------


@torch.no_grad()
def estimate_loss(model: TinyLLM, loaders: dict[str, ShardLoader], iters: int, ctx) -> dict[str, float]:
    model.eval()
    out = {}
    for split, loader in loaders.items():
        losses = []
        for _ in range(iters):
            x, y = loader.next_batch()
            with ctx:
                _, loss = model(x, y)
            losses.append(loss.item())
        out[split] = float(np.mean(losses))
    model.train()
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", action="store_true", help="resume from latest checkpoint")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    cfg = load_config(args.config)
    mcfg = ModelConfig(**cfg["model"])
    tcfg = cfg["train"]
    dcfg = cfg["data"]

    # ---------- device / dtype ----------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: training on CPU — this will be unbearably slow. Use a GPU.")
    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    dtype = dtype_map[tcfg.get("dtype", "bfloat16")]
    # bf16 only on Ampere+; fall back to fp16 on older cards.
    if dtype == torch.bfloat16 and device == "cuda" and not torch.cuda.is_bf16_supported():
        print("bf16 unsupported on this GPU; falling back to fp16")
        dtype = torch.float16
    ctx = torch.amp.autocast(device_type="cuda", dtype=dtype) if device == "cuda" else torch.amp.autocast(device_type="cpu", dtype=torch.float32)

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    # ---------- model ----------
    model = TinyLLM(mcfg).to(device)
    n_params = model.num_params()
    print(f"model: {n_params:,} params ({n_params/1e6:.2f}M)")

    # ---------- optimizer ----------
    optimizer = build_optimizer(
        model,
        lr=float(tcfg["lr"]),
        weight_decay=float(tcfg["weight_decay"]),
        betas=(float(tcfg["beta1"]), float(tcfg["beta2"])),
    )

    # ---------- data ----------
    shard_dir = Path(dcfg["shard_dir"])
    train_shards = sorted(Path(p) for p in glob.glob(str(shard_dir / dcfg["train_glob"])))
    val_shards = sorted(Path(p) for p in glob.glob(str(shard_dir / dcfg["val_glob"])))
    if not train_shards:
        raise SystemExit(f"no train shards found at {shard_dir}/{dcfg['train_glob']}")
    if not val_shards:
        raise SystemExit(f"no val shards found at {shard_dir}/{dcfg['val_glob']}")
    print(f"train shards: {len(train_shards)}  val shards: {len(val_shards)}")

    loaders = {
        "train": ShardLoader(train_shards, mcfg.block_size, int(tcfg["micro_batch_size"]), device, rng),
        "val": ShardLoader(val_shards, mcfg.block_size, int(tcfg["micro_batch_size"]), device, rng),
    }

    # ---------- checkpoint ----------
    out_dir = Path(tcfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "latest.pt"
    start_step = 0
    best_val = float("inf")
    if args.resume and ckpt_path.exists():
        print(f"resuming from {ckpt_path}")
        # weights_only=True is safe here — our checkpoints contain only tensors
        # + plain dicts, no pickled Python objects.
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = int(ckpt["step"]) + 1
        best_val = float(ckpt.get("best_val", best_val))

    # ---------- wandb ----------
    wb_cfg = tcfg.get("wandb", {})
    use_wandb = bool(wb_cfg.get("enabled")) and os.environ.get("WANDB_DISABLED") != "true"
    if use_wandb:
        import wandb
        wandb.init(
            project=wb_cfg.get("project", "tiny-llm"),
            name=wb_cfg.get("run_name", "pretrain"),
            config={"model": asdict(mcfg), "train": tcfg, "data": dcfg},
        )

    # ---------- training loop ----------
    max_steps = int(tcfg["max_steps"])
    grad_accum = int(tcfg["grad_accum_steps"])
    grad_clip = float(tcfg["grad_clip"])
    max_lr = float(tcfg["lr"])
    min_lr = float(tcfg["min_lr"])
    warmup = int(tcfg["warmup_steps"])
    log_interval = int(tcfg["log_interval"])
    eval_interval = int(tcfg["eval_interval"])
    eval_iters = int(tcfg["eval_iters"])
    save_interval = int(tcfg["save_interval"])
    tokens_per_step = int(tcfg["micro_batch_size"]) * mcfg.block_size * grad_accum

    model.train()
    t0 = time.time()
    for step in range(start_step, max_steps):
        # set LR
        lr = lr_at(step, max_lr, min_lr, warmup, max_steps)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # eval / checkpoint
        if step % eval_interval == 0 and step > 0:
            losses = estimate_loss(model, loaders, eval_iters, ctx)
            print(f"step {step}: train {losses['train']:.4f}  val {losses['val']:.4f}  lr {lr:.2e}")
            if use_wandb:
                import wandb
                wandb.log({"val/loss": losses["val"], "train/loss_eval": losses["train"], "lr": lr, "step": step})
            if losses["val"] < best_val:
                best_val = losses["val"]
                torch.save(
                    {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                     "step": step, "best_val": best_val, "config": cfg},
                    out_dir / "best.pt",
                )

        # grad accumulation
        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0
        for _ in range(grad_accum):
            x, y = loaders["train"].next_batch()
            with ctx:
                _, loss = model(x, y)
                loss = loss / grad_accum
            loss.backward()
            loss_accum += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        if step % log_interval == 0:
            elapsed = time.time() - t0
            tok_per_sec = tokens_per_step * (step - start_step + 1) / max(elapsed, 1e-6)
            print(f"step {step:6d} | loss {loss_accum:.4f} | lr {lr:.2e} | {tok_per_sec/1e3:.1f}k tok/s")
            if use_wandb:
                import wandb
                wandb.log({"train/loss": loss_accum, "lr": lr, "tokens_per_sec": tok_per_sec, "step": step})

        if step > 0 and step % save_interval == 0:
            torch.save(
                {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "step": step, "best_val": best_val, "config": cfg},
                ckpt_path,
            )

    # final checkpoint
    torch.save(
        {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
         "step": max_steps, "best_val": best_val, "config": cfg},
        ckpt_path,
    )
    print(f"done. best val loss: {best_val:.4f}")


if __name__ == "__main__":
    main()
