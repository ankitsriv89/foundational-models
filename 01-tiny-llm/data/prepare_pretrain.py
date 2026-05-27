"""Pre-tokenize FineWeb-Edu and write fixed-size shards to disk.

Why pre-tokenize:
  Training reads ~1B tokens many times. Tokenizing inside the dataloader would
  bottleneck the GPU on CPU. So we tokenize ONCE up front and write the result
  as flat uint16 arrays. The training loop then just memory-maps these files
  and does fast random reads.

Output layout (matches nanoGPT/nanochat convention):
  data/shards/train_000000.bin
  data/shards/train_000001.bin
  ...
  data/shards/val_000000.bin

Each .bin file is a flat uint16 array. Documents are separated by the
<|endoftext|> token. Reading is just `np.memmap(path, dtype=np.uint16)`.

We use uint16 because our vocab is 8192 (fits in 14 bits). uint16 saves 50%
disk vs uint32. If you ever bump vocab > 65535, switch to uint32.

Usage (run on a RunPod box):
    uv run python data/prepare_pretrain.py \\
        --tokenizer tokenizer/tiny-llm-bpe.json \\
        --target-tokens 1_000_000_000 \\
        --val-tokens 5_000_000 \\
        --shard-tokens 100_000_000 \\
        --out-dir data/shards
"""

import argparse
from pathlib import Path

import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer
from tqdm import tqdm


EOT_TOKEN = "<|endoftext|>"


def write_shard(out_path: Path, buf: np.ndarray) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        f.write(buf.tobytes())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", type=Path, default=Path("tokenizer/tiny-llm-bpe.json"))
    p.add_argument("--out-dir", type=Path, default=Path("data/shards"))
    p.add_argument("--target-tokens", type=int, default=1_000_000_000,
                   help="approximate total training tokens to produce")
    p.add_argument("--val-tokens", type=int, default=5_000_000,
                   help="size of the held-out validation set")
    p.add_argument("--shard-tokens", type=int, default=100_000_000,
                   help="tokens per output shard file")
    p.add_argument("--dataset-name", type=str, default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--dataset-config", type=str, default="sample-10BT")
    args = p.parse_args()

    if not args.tokenizer.exists():
        raise SystemExit(f"tokenizer not found at {args.tokenizer} — run Phase 2 first")

    tok = Tokenizer.from_file(str(args.tokenizer))
    eot_id = tok.token_to_id(EOT_TOKEN)
    assert eot_id is not None, f"tokenizer has no {EOT_TOKEN}"
    assert tok.get_vocab_size() <= 65535, "vocab too large for uint16 shards"

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"streaming {args.dataset_name} (config={args.dataset_config})")
    ds = load_dataset(
        args.dataset_name,
        name=args.dataset_config,
        split="train",
        streaming=True,
    )

    # Split: first `val_tokens` go to val, the rest to train.
    total_target = args.target_tokens + args.val_tokens
    pbar = tqdm(total=total_target, unit="tok", unit_scale=True)

    # Shard-writing state.
    split = "val"
    shard_idx = 0
    buf = np.zeros(args.shard_tokens, dtype=np.uint16)
    pos = 0
    total_written = 0
    val_written = 0

    def flush_shard() -> None:
        nonlocal pos, shard_idx
        if pos == 0:
            return
        out = args.out_dir / f"{split}_{shard_idx:06d}.bin"
        write_shard(out, buf[:pos])
        pbar.write(f"  wrote {out.name} — {pos:,} tokens")
        pos = 0
        shard_idx += 1

    try:
        for example in ds:
            text = example["text"]
            if not text.strip():
                continue
            ids = tok.encode(text).ids
            ids.append(eot_id)  # document boundary
            arr = np.asarray(ids, dtype=np.uint16)

            # Place into shard buffer, flushing when full.
            i = 0
            while i < len(arr):
                space = args.shard_tokens - pos
                take = min(space, len(arr) - i)
                buf[pos:pos + take] = arr[i:i + take]
                pos += take
                i += take
                if pos == args.shard_tokens:
                    flush_shard()

            total_written += len(arr)
            pbar.update(len(arr))

            # Switch from val to train once we've hit val_tokens.
            if split == "val" and total_written >= args.val_tokens:
                flush_shard()
                val_written = total_written
                split = "train"
                shard_idx = 0
                pbar.write(f"--- val done ({val_written:,} tokens). switching to train. ---")

            if total_written >= total_target:
                break

        flush_shard()
    finally:
        pbar.close()

    print(f"\ndone. val tokens: {val_written:,}  train tokens: {total_written - val_written:,}")
    # Quick listing.
    for f in sorted(args.out_dir.iterdir()):
        size_mb = f.stat().st_size / 1e6
        print(f"  {f.name}  {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
