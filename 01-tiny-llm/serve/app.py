"""Gradio + FastAPI inference demo for the tiny LLM.

This is the Phase 7 deployment artifact. The same script runs in two modes:

  * `python serve/app.py` — launches the Gradio UI directly. Use this on
    HuggingFace Spaces (the standard portfolio venue).

  * `uvicorn serve.app:api --host 0.0.0.0 --port 8000` — runs only the FastAPI
    endpoint. Use this if you want to serve the model as a JSON API.

Both modes share the same generation function. The Gradio UI streams tokens
one at a time for a chatbot-style experience.

Run on HuggingFace Spaces:
  1. Push this repo + the SFT checkpoint to a Space.
  2. The Space's Python environment auto-installs from pyproject.toml.
  3. Set the entrypoint to `python serve/app.py`.

Usage locally:
    uv run --group serve python serve/app.py \\
        --ckpt checkpoints/sft/latest.pt \\
        --tokenizer tokenizer/tiny-llm-bpe.json
"""

from __future__ import annotations

import argparse
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Iterator

import torch
import torch.nn.functional as F
from fastapi import FastAPI
from pydantic import BaseModel
from tokenizers import Tokenizer

from model import ModelConfig, TinyLLM


# ---------------------------------------------------------------------------
# Globals — populated by load_model()
# ---------------------------------------------------------------------------

_MODEL: TinyLLM | None = None
_TOK: Tokenizer | None = None
_DEVICE: str = "cpu"
_EOT_ID: int = 0


def load_model(ckpt_path: Path, tokenizer_path: Path) -> None:
    global _MODEL, _TOK, _DEVICE, _EOT_ID
    _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(ckpt_path, map_location=_DEVICE, weights_only=True)
    mcfg = ModelConfig(**ckpt["config"]["model"])
    model = TinyLLM(mcfg).to(_DEVICE).eval()
    model.load_state_dict(ckpt["model"])
    tok = Tokenizer.from_file(str(tokenizer_path))
    eot = tok.token_to_id("<|endoftext|>")
    if eot is None:
        raise SystemExit("tokenizer missing <|endoftext|>")
    _MODEL = model
    _TOK = tok
    _EOT_ID = eot
    print(f"loaded model ({model.num_params():,} params) on {_DEVICE}")


# ---------------------------------------------------------------------------
# Generation (streaming, token-by-token)
# ---------------------------------------------------------------------------


@torch.no_grad()
def stream_generate(
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 50,
    use_chat_template: bool = True,
) -> Iterator[str]:
    """Yield decoded text chunks as tokens are generated."""
    assert _MODEL is not None and _TOK is not None, "call load_model() first"

    full_prompt = f"<|user|>{prompt}<|assistant|>" if use_chat_template else prompt
    ids = _TOK.encode(full_prompt).ids
    idx = torch.tensor([ids], dtype=torch.long, device=_DEVICE)

    generated_ids: list[int] = []
    # We re-decode the whole tail each step because BPE tokens can join into
    # multi-byte UTF-8 characters mid-stream; decoding incrementally would
    # split characters and produce garbage.
    last_text = ""

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -_MODEL.cfg.block_size:]
        logits, _ = _MODEL(idx_cond)
        logits = logits[:, -1, :] / max(temperature, 1e-8)
        if top_k:
            topk_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < topk_vals[:, [-1]]] = -float("inf")
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        next_id_int = int(next_id.item())
        if use_chat_template and next_id_int == _EOT_ID:
            break

        idx = torch.cat([idx, next_id], dim=1)
        generated_ids.append(next_id_int)
        text = _TOK.decode(generated_ids)
        if text != last_text:
            yield text[len(last_text):]
            last_text = text


def generate_full(prompt: str, max_new_tokens: int = 200, temperature: float = 0.8, top_k: int = 50, use_chat_template: bool = True) -> str:
    return "".join(stream_generate(prompt, max_new_tokens, temperature, top_k, use_chat_template))


# ---------------------------------------------------------------------------
# FastAPI
# ---------------------------------------------------------------------------

@asynccontextmanager
async def _lifespan(app: FastAPI):
    ckpt = Path(os.environ.get("TINY_LLM_CKPT", "checkpoints/sft/latest.pt"))
    tok = Path(os.environ.get("TINY_LLM_TOKENIZER", "tokenizer/tiny-llm-bpe.json"))
    if _MODEL is None and ckpt.exists():
        load_model(ckpt, tok)
    yield


api = FastAPI(title="Tiny LLM", lifespan=_lifespan)


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 200
    temperature: float = 0.8
    top_k: int = 50
    chat: bool = True


@api.post("/generate")
def http_generate(req: GenerateRequest) -> dict:
    text = generate_full(req.prompt, req.max_new_tokens, req.temperature, req.top_k, req.chat)
    return {"completion": text}


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------


def launch_gradio() -> None:
    import gradio as gr

    def chat_fn(message: str, _history, max_new_tokens: float, temperature: float, top_k: float):
        partial = ""
        for chunk in stream_generate(
            message,
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
            top_k=int(top_k),
            use_chat_template=True,
        ):
            partial += chunk
            yield partial

    with gr.Blocks(title="Tiny LLM") as ui:
        gr.Markdown(
            "# Tiny LLM — ~10M params, pretrained from scratch on FineWeb-Edu + SFT on Dolly\n"
            "Generation is intentionally short-and-imperfect — at this scale, fluency is the test, not knowledge."
        )
        with gr.Row():
            with gr.Column(scale=3):
                gr.ChatInterface(
                    chat_fn,
                    type="messages",  # type: ignore[arg-type]
                    additional_inputs=[
                        gr.Slider(20, 400, value=150, step=10, label="max new tokens"),
                        gr.Slider(0.1, 1.5, value=0.8, step=0.05, label="temperature"),
                        gr.Slider(1, 200, value=50, step=1, label="top-k"),
                    ],
                )
            with gr.Column(scale=1):
                gr.Markdown(
                    "## Suggested prompts\n"
                    "- What is photosynthesis?\n"
                    "- Explain gradient descent.\n"
                    "- Write a haiku about a cat.\n"
                    "- What is Python used for?\n"
                )

    ui.queue().launch(server_name="0.0.0.0")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, default=Path("checkpoints/sft/latest.pt"))
    p.add_argument("--tokenizer", type=Path, default=Path("tokenizer/tiny-llm-bpe.json"))
    args = p.parse_args()
    load_model(args.ckpt, args.tokenizer)
    launch_gradio()


if __name__ == "__main__":
    main()
