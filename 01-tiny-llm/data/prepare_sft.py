"""Format an instruction dataset for SFT.

Dataset: `databricks/databricks-dolly-15k`. ~15K human-written instruction/response
pairs. Small enough to download fast, large enough to teach instruction-following,
and human-written (no GPT contamination), which is rare and nice for a portfolio
artifact.

Chat template (used at both SFT training time and inference time):

    <|user|>{instruction}\\n{context}<|assistant|>{response}<|endoftext|>

Output format:
  data/sft/train.bin       — flat uint16 array of token ids
  data/sft/train_mask.bin  — flat uint8 array, 1 where loss should apply, else 0

The mask is what makes SFT "supervised on the response only" — we want the
model to learn to *produce* the assistant's reply, not to memorize the prompt.

Examples are concatenated end-to-end (separated by <|endoftext|>, which is
already the assistant turn terminator above). The training loop will read
random windows of length block_size and use the mask to skip loss on prompt
tokens.

Usage:
    uv run python data/prepare_sft.py \\
        --tokenizer tokenizer/tiny-llm-bpe.json \\
        --out-dir data/sft
"""

import argparse
from pathlib import Path

import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer
from tqdm import tqdm


def format_prompt(instruction: str, context: str) -> str:
    if context and context.strip():
        return f"<|user|>{instruction.strip()}\n{context.strip()}"
    return f"<|user|>{instruction.strip()}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", type=Path, default=Path("tokenizer/tiny-llm-bpe.json"))
    p.add_argument("--out-dir", type=Path, default=Path("data/sft"))
    p.add_argument("--dataset", type=str, default="databricks/databricks-dolly-15k")
    p.add_argument("--max-examples", type=int, default=None,
                   help="optional cap on examples (None = use all)")
    args = p.parse_args()

    if not args.tokenizer.exists():
        raise SystemExit(f"tokenizer not found at {args.tokenizer} — run Phase 2 first")

    tok = Tokenizer.from_file(str(args.tokenizer))

    def must_get(name: str) -> int:
        tid = tok.token_to_id(name)
        if tid is None:
            raise SystemExit(f"tokenizer is missing required special token {name}")
        return tid

    ASSISTANT = must_get("<|assistant|>")
    EOT = must_get("<|endoftext|>")
    _ = must_get("<|user|>")  # used as a literal inside format_prompt; just verify it exists

    args.out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(args.dataset, split="train")
    if args.max_examples is not None:
        ds = ds.select(range(min(args.max_examples, len(ds))))
    print(f"loaded {len(ds):,} examples from {args.dataset}")

    all_ids: list[int] = []
    all_mask: list[int] = []
    skipped = 0

    for ex in tqdm(ds, desc="formatting"):
        # HuggingFace Dataset rows are dict-like at runtime; cast for the type checker.
        row: dict = ex  # type: ignore[assignment]
        instruction = row.get("instruction", "") or ""
        context = row.get("context", "") or ""
        response = row.get("response", "") or ""
        if not instruction.strip() or not response.strip():
            skipped += 1
            continue

        prompt_text = format_prompt(instruction, context)
        prompt_ids = tok.encode(prompt_text).ids
        response_ids = tok.encode(response.strip()).ids

        # Sequence: <prompt tokens> <|assistant|> <response tokens> <|endoftext|>
        # Mask:     0...0           0             1...1              1
        # The model sees the prompt but never has loss computed on it.
        seq = prompt_ids + [ASSISTANT] + response_ids + [EOT]
        mask = [0] * len(prompt_ids) + [0] + [1] * len(response_ids) + [1]
        assert len(seq) == len(mask)

        all_ids.extend(seq)
        all_mask.extend(mask)

    ids_arr = np.asarray(all_ids, dtype=np.uint16)
    mask_arr = np.asarray(all_mask, dtype=np.uint8)
    assert ids_arr.shape == mask_arr.shape

    (args.out_dir / "train.bin").write_bytes(ids_arr.tobytes())
    (args.out_dir / "train_mask.bin").write_bytes(mask_arr.tobytes())

    print(f"\nwrote {len(ids_arr):,} tokens to {args.out_dir}")
    print(f"  skipped {skipped} empty examples")
    print(f"  loss-on tokens: {int(mask_arr.sum()):,}  ({100 * mask_arr.mean():.1f}% of total)")
    print(f"  files:")
    for f in sorted(args.out_dir.iterdir()):
        size_mb = f.stat().st_size / 1e6
        print(f"    {f.name}  {size_mb:.2f} MB")


if __name__ == "__main__":
    main()
