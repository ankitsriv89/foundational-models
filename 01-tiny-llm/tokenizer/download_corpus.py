"""Stream a slice of FineWeb-Edu and save it as a local text file.

We stream rather than download the full dataset because FineWeb-Edu is enormous
(~1.3T tokens). For tokenizer training we only need ~200MB of text — that's
plenty for an 8192-vocab BPE.

The output file is one document per line. Documents are not shuffled; FineWeb-Edu
is already reasonably mixed at the source level, and we only want a representative
sample.

Usage:
    python tokenizer/download_corpus.py --out data/cache/corpus.txt --target-mb 200
"""

import argparse
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


def stream_corpus(out_path: Path, target_bytes: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # The 'sample-10BT' config is a curated 10B-token slice maintained by HF.
    # Streaming avoids downloading the whole thing.
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
    )

    bytes_written = 0
    docs_written = 0
    pbar = tqdm(total=target_bytes, unit="B", unit_scale=True, desc="streaming")

    with out_path.open("w", encoding="utf-8") as f:
        for example in ds:
            text = example["text"].replace("\n", " ").strip()
            if not text:
                continue
            line = text + "\n"
            n = len(line.encode("utf-8"))
            f.write(line)
            bytes_written += n
            docs_written += 1
            pbar.update(n)
            if bytes_written >= target_bytes:
                break

    pbar.close()
    print(f"\nwrote {docs_written:,} docs / {bytes_written / 1e6:.1f} MB to {out_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--out",
        type=Path,
        default=Path("data/cache/corpus.txt"),
        help="output text file (one doc per line)",
    )
    p.add_argument(
        "--target-mb",
        type=int,
        default=200,
        help="approximate target size in megabytes",
    )
    args = p.parse_args()

    if args.out.exists() and args.out.stat().st_size > 0:
        size_mb = args.out.stat().st_size / 1e6
        print(f"{args.out} already exists ({size_mb:.1f} MB). delete it to re-download.")
        return

    stream_corpus(args.out, target_bytes=args.target_mb * 1_000_000)


if __name__ == "__main__":
    main()
