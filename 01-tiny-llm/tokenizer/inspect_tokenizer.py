"""Inspect a trained tokenizer: stats, sample tokenizations, and round-trip test.

This is the Phase 2 verification script. Confirms:
  1. The tokenizer loads.
  2. Special tokens have the expected IDs.
  3. Encoding/decoding round-trips losslessly on a random sample of docs.
  4. Compression ratio (bytes-per-token) is in a sane range (~3.5-4.5 for English web text).

Usage:
    python tokenizer/inspect_tokenizer.py \\
        --tokenizer tokenizer/tiny-llm-bpe.json \\
        --corpus data/cache/corpus.txt
"""

import argparse
import random
from pathlib import Path

from tokenizers import Tokenizer


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", type=Path, default=Path("tokenizer/tiny-llm-bpe.json"))
    p.add_argument("--corpus", type=Path, default=Path("data/cache/corpus.txt"))
    p.add_argument("--n-samples", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    tok = Tokenizer.from_file(str(args.tokenizer))
    print(f"loaded tokenizer: vocab_size={tok.get_vocab_size()}")

    for name in ["<|endoftext|>", "<|user|>", "<|assistant|>", "<|pad|>"]:
        print(f"  {name!r:20s} id={tok.token_to_id(name)}")

    print("\nsample tokenizations:")
    samples = [
        "The quick brown fox jumps over the lazy dog.",
        "Photosynthesis converts CO2 into glucose using sunlight.",
        "def hello():\n    print('hi')",
        "Café résumé naïve 北京 😀",
    ]
    for s in samples:
        enc = tok.encode(s)
        print(f"  {s!r}")
        print(f"    → {len(enc.ids)} tokens: {enc.tokens}")

    print(f"\nround-trip test on {args.n_samples} random docs from {args.corpus}")
    rng = random.Random(args.seed)
    with args.corpus.open("r", encoding="utf-8") as f:
        all_docs = [line.rstrip("\n") for line in f if line.strip()]
    docs = rng.sample(all_docs, min(args.n_samples, len(all_docs)))

    total_bytes = 0
    total_tokens = 0
    failures = 0
    for d in docs:
        enc = tok.encode(d)
        decoded = tok.decode(enc.ids)
        if decoded != d:
            failures += 1
            if failures <= 3:
                print(f"  MISMATCH:\n    original: {d[:120]!r}\n    decoded:  {decoded[:120]!r}")
        total_bytes += len(d.encode("utf-8"))
        total_tokens += len(enc.ids)

    bpt = total_bytes / max(total_tokens, 1)
    print(f"\n  round-trip failures: {failures}/{len(docs)}")
    print(f"  bytes / token:       {bpt:.2f}  (expect ~3.5-4.5 for English web text)")
    print(f"  total tokens:        {total_tokens:,}")
    print(f"  total bytes:         {total_bytes:,}")


if __name__ == "__main__":
    main()
