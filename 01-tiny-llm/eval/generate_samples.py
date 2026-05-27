"""Generate sampled continuations from fixed prompts.

This is the qualitative eval. Run it after pretraining (base model: just text
completion) and again after SFT (instruction following). The before/after
contrast is one of the strongest portfolio artifacts in the project.

Output is plain text written to stdout and optionally a markdown file suitable
for pasting into the README.

Usage:
    uv run --group train python eval/generate_samples.py \\
        --ckpt checkpoints/pretrain/best.pt \\
        --tokenizer tokenizer/tiny-llm-bpe.json \\
        --mode base \\
        --out eval/samples_base.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tokenizers import Tokenizer

from model import ModelConfig, TinyLLM


# Fixed prompts. The SAME prompts are used for base and SFT so the comparison is fair.
BASE_PROMPTS = [
    "Photosynthesis is the process by which",
    "The capital of France is",
    "In machine learning, gradient descent is used to",
    "Once upon a time in a small village,",
    "Python is a programming language that",
]

# For the SFT model we wrap prompts in the chat template.
SFT_PROMPTS = [
    "What is photosynthesis?",
    "What is the capital of France?",
    "Explain gradient descent in simple terms.",
    "Write a short story about a brave mouse.",
    "What is Python used for?",
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, default=Path("tokenizer/tiny-llm-bpe.json"))
    p.add_argument("--mode", choices=["base", "sft"], default="base")
    p.add_argument("--max-new-tokens", type=int, default=120)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=None, help="optional markdown output file")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=True)
    mcfg = ModelConfig(**ckpt["config"]["model"])
    model = TinyLLM(mcfg).to(device).eval()
    model.load_state_dict(ckpt["model"])
    tok = Tokenizer.from_file(str(args.tokenizer))
    eot_id = tok.token_to_id("<|endoftext|>")

    prompts = BASE_PROMPTS if args.mode == "base" else SFT_PROMPTS

    md_lines: list[str] = [f"# Samples — {args.mode} model\n"]
    md_lines.append(f"checkpoint: `{args.ckpt}`  step={ckpt.get('step', '?')}  best_val={ckpt.get('best_val', float('nan')):.4f}\n")
    md_lines.append(f"sampling: temperature={args.temperature}, top_k={args.top_k}, max_new_tokens={args.max_new_tokens}\n")

    for raw_prompt in prompts:
        if args.mode == "sft":
            full = f"<|user|>{raw_prompt}<|assistant|>"
        else:
            full = raw_prompt

        ids = tok.encode(full).ids
        idx = torch.tensor([ids], dtype=torch.long, device=device)
        out = model.generate(idx, max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_k=args.top_k)
        gen_ids = out[0].tolist()[len(ids):]
        # For SFT, stop at the first EOT token if present.
        if args.mode == "sft" and eot_id in gen_ids:
            gen_ids = gen_ids[:gen_ids.index(eot_id)]
        continuation = tok.decode(gen_ids)

        print(f"\n=== prompt: {raw_prompt!r} ===")
        print(continuation)
        md_lines.append(f"### Prompt: `{raw_prompt}`\n")
        md_lines.append(f"```\n{continuation}\n```\n")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("\n".join(md_lines))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
