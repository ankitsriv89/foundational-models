"""Compute validation loss, perplexity, and bits-per-byte on held-out shards.

This is the primary quantitative eval for a 10M model. Multiple-choice
benchmarks (HellaSwag etc.) will score near random at this scale — the
trajectory of val loss / BPB is what actually tells us if the model improved.

Bits-per-byte (BPB) is loss-per-token converted into bits-per-UTF8-byte of the
*original text*. It's comparable across tokenizers, unlike raw perplexity.
For a baseline: GPT-2 small (~124M) gets ~1.0 BPB on web text; a tiny 10M
model should land around 1.5-2.0 BPB.

Usage:
    uv run --group train python eval/evaluate_ppl.py \\
        --ckpt checkpoints/pretrain/best.pt \\
        --shards data/shards \\
        --val-glob "val_*.bin" \\
        --tokenizer tokenizer/tiny-llm-bpe.json
"""

from __future__ import annotations

import argparse
import glob
import math
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

from model import ModelConfig, TinyLLM


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--shards", type=Path, default=Path("data/shards"))
    p.add_argument("--val-glob", type=str, default="val_*.bin")
    p.add_argument("--tokenizer", type=Path, default=Path("tokenizer/tiny-llm-bpe.json"))
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-batches", type=int, default=200, help="-1 to evaluate everything")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=True)
    mcfg = ModelConfig(**ckpt["config"]["model"])
    model = TinyLLM(mcfg).to(device).eval()
    model.load_state_dict(ckpt["model"])
    print(f"loaded ckpt: step={ckpt.get('step', '?')}  best_val={ckpt.get('best_val', float('nan')):.4f}")

    val_paths = sorted(Path(p) for p in glob.glob(str(args.shards / args.val_glob)))
    if not val_paths:
        raise SystemExit(f"no val shards matching {args.shards}/{args.val_glob}")

    # Estimate average bytes-per-token from the tokenizer + a sample of val data.
    # Needed to convert loss (nats/token) into bits/byte.
    tok = Tokenizer.from_file(str(args.tokenizer))
    sample = np.memmap(val_paths[0], dtype=np.uint16, mode="r")[:200_000].tolist()
    sample_text = tok.decode(sample)
    bytes_per_tok = len(sample_text.encode("utf-8")) / max(len(sample), 1)
    print(f"avg bytes/token (val sample): {bytes_per_tok:.2f}")

    # Sequential, non-overlapping windows over val shards.
    B = args.batch_size
    T = mcfg.block_size
    total_loss = 0.0
    total_tokens = 0
    batches_done = 0
    for path in val_paths:
        data = np.memmap(path, dtype=np.uint16, mode="r")
        stride = B * T
        for start in range(0, len(data) - T - 1, stride):
            # Build B sequences of length T from contiguous chunks.
            chunks = []
            for b in range(B):
                s = start + b * T
                if s + T + 1 > len(data):
                    break
                chunks.append((s, s + T + 1))
            if not chunks:
                continue
            x = np.stack([np.asarray(data[s:e - 1], dtype=np.int64) for s, e in chunks])
            y = np.stack([np.asarray(data[s + 1:e], dtype=np.int64) for s, e in chunks])
            xt = torch.from_numpy(x).to(device)
            yt = torch.from_numpy(y).to(device)
            _, loss = model(xt, yt)
            n = xt.numel()
            total_loss += loss.item() * n
            total_tokens += n
            batches_done += 1
            if args.max_batches > 0 and batches_done >= args.max_batches:
                break
        if args.max_batches > 0 and batches_done >= args.max_batches:
            break

    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(avg_loss)
    bits_per_token = avg_loss / math.log(2)
    bpb = bits_per_token / bytes_per_tok

    print(f"\nevaluated {total_tokens:,} tokens across {batches_done} batches")
    print(f"  cross-entropy loss : {avg_loss:.4f}  (nats/token)")
    print(f"  perplexity         : {ppl:.2f}")
    print(f"  bits per token     : {bits_per_token:.3f}")
    print(f"  bits per byte (BPB): {bpb:.3f}")
    print(f"    baseline references:")
    print(f"      uniform 8192-vocab  : {math.log2(8192):.2f} bits/token  =  {math.log2(8192)/bytes_per_tok:.2f} BPB")
    print(f"      gpt-2 small on web  : ~1.0 BPB")


if __name__ == "__main__":
    main()
