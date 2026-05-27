"""Train a byte-level BPE tokenizer on a local text corpus.

Design choices (see docs/glossary.md for what these terms mean):

  * Byte-level BPE (GPT-style), not WordPiece or SentencePiece-unigram.
    Byte-level means the tokenizer can represent ANY UTF-8 input, including
    emojis and non-English text, with no out-of-vocab errors.

  * Vocab size = 8192. Small, because at 10M params we don't want the
    embedding table (vocab_size * d_model = 8192 * 384 ≈ 3.1M params) to
    dominate the budget. Modern LLMs use 32K-256K but their models are 1000x
    bigger.

  * Pre-tokenizer = ByteLevel(add_prefix_space=False). Splits text into
    bytes before BPE merges run. Prevents tokens from spanning whitespace
    incorrectly. add_prefix_space=False matches GPT-2/Llama convention.

  * Decoder = ByteLevel(). The inverse of the pre-tokenizer — turns
    token IDs back into the original text losslessly.

  * Special tokens reserved up front:
        <|endoftext|>   document boundary, also used as BOS/EOS
        <|user|>        chat turn marker (used in Phase 6 SFT)
        <|assistant|>   chat turn marker (used in Phase 6 SFT)
        <|pad|>         padding (rarely used in causal LM but reserved)

Usage:
    python tokenizer/train_tokenizer.py \\
        --corpus data/cache/corpus.txt \\
        --vocab-size 8192 \\
        --out tokenizer/tiny-llm-bpe.json
"""

import argparse
from pathlib import Path

from tokenizers import Tokenizer, decoders, pre_tokenizers, trainers
from tokenizers.models import BPE


SPECIAL_TOKENS = [
    "<|endoftext|>",
    "<|user|>",
    "<|assistant|>",
    "<|pad|>",
]


def iter_lines(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line:
                yield line


def train(corpus_path: Path, out_path: Path, vocab_size: int) -> None:
    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )

    print(f"training on {corpus_path} → target vocab {vocab_size}")
    tokenizer.train_from_iterator(iter_lines(corpus_path), trainer=trainer)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(out_path))
    print(f"saved tokenizer → {out_path}")
    print(f"final vocab size: {tokenizer.get_vocab_size()}")

    for tok in SPECIAL_TOKENS:
        print(f"  {tok!r:20s} id={tokenizer.token_to_id(tok)}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", type=Path, default=Path("data/cache/corpus.txt"))
    p.add_argument("--vocab-size", type=int, default=8192)
    p.add_argument("--out", type=Path, default=Path("tokenizer/tiny-llm-bpe.json"))
    args = p.parse_args()

    if not args.corpus.exists():
        raise SystemExit(
            f"corpus not found at {args.corpus} — run tokenizer/download_corpus.py first"
        )

    train(args.corpus, args.out, args.vocab_size)


if __name__ == "__main__":
    main()
