"""Tiny LLM — decoder-only transformer, ~10M params.

A single-file implementation, read top to bottom. Every architectural choice
here is defined in docs/glossary.md and walked through in docs/tutorial.md.

The architecture summary (modern 2025-2026 table stakes):

  * RMSNorm pre-norm placement
  * RoPE (rotary position embeddings)
  * Grouped-Query Attention (GQA)
  * SwiGLU FFN
  * Tied input/output embeddings
  * Causal mask via PyTorch's fused F.scaled_dot_product_attention
    (which dispatches to Flash Attention on modern GPUs)

Shape conventions used throughout:
    B = batch size
    T = sequence length (number of tokens in this forward pass)
    C = hidden dimension (d_model, n_embd)
    H = number of attention heads (query heads)
    Hkv = number of K/V heads (GQA — Hkv <= H, and H % Hkv == 0)
    D = per-head dimension (head_dim = C // H)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
    """Hyperparameters for the model.

    Defaults target ~10M parameters with our 8192-vocab tokenizer:
        embedding: 8192 * 384 = 3,145,728
        6 blocks ≈ ~7M (attention + FFN)
        Total ≈ ~10.3M (tied embeddings cut output proj cost to zero)
    """

    vocab_size: int = 8192       # matches the BPE tokenizer in tokenizer/
    block_size: int = 1024       # max context length the model is trained on
    n_layer: int = 6             # transformer blocks
    n_head: int = 6              # query heads — must divide n_embd
    n_kv_head: int = 2           # K/V heads (GQA) — must divide n_head
    n_embd: int = 384            # hidden dimension
    ffn_mult: int = 4            # FFN inner dim multiplier (will be scaled for SwiGLU)
    rope_base: float = 10000.0   # RoPE theta. 10000 is the original GPT-NeoX value.
    norm_eps: float = 1e-5       # RMSNorm epsilon
    dropout: float = 0.0         # we don't use dropout at this scale

    def __post_init__(self) -> None:
        assert self.n_embd % self.n_head == 0, "n_embd must divide n_head"
        assert self.n_head % self.n_kv_head == 0, "n_head must be a multiple of n_kv_head"

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------
# Simpler than LayerNorm: no mean-subtraction step. Just rescales each vector
# by its RMS magnitude, then applies a learned per-channel gain.
#
#   y = x / sqrt(mean(x^2) + eps) * gain
#
# Slightly faster than LayerNorm because we skip computing the mean.
# Standard in Llama, Mistral, Qwen.
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  # the learned gain

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute in fp32 for numerical stability even when activations are bf16.
        # This is the standard trick — bf16 has tiny mantissa precision.
        dtype = x.dtype
        x32 = x.float()
        rms = x32.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x32 * rms).to(dtype) * self.weight


# ---------------------------------------------------------------------------
# RoPE (Rotary Position Embeddings)
# ---------------------------------------------------------------------------
# Instead of *adding* a position embedding to the token embedding, RoPE
# *rotates* the Q and K vectors by an angle that depends on position.
#
# The clever property: after rotation, the dot product Q_m · K_n only depends
# on (m - n), not on m and n individually. So the model "sees" relative
# positions through attention, not absolute ones. This generalizes better to
# longer sequences than learned absolute embeddings.
#
# Implementation pattern (Llama-style, the now-canonical one):
#   1. Precompute cos/sin for every position and every head-dim pair.
#   2. For each Q/K vector, split into pairs and rotate each 2D pair.
#
# We precompute the cos/sin table once at init and slice it per-forward.
# ---------------------------------------------------------------------------


def build_rope_cache(seq_len: int, head_dim: int, base: float, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (cos, sin) tables of shape (seq_len, head_dim/2)."""
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"
    # Frequencies for each of the head_dim/2 rotation pairs.
    # Lower indices rotate faster; higher indices rotate slower (long-range).
    freqs = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    angles = torch.outer(t, freqs)  # (seq_len, head_dim/2)
    return angles.cos().to(dtype), angles.sin().to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary embedding to x.

    Args:
        x:   (B, H, T, D)        Q or K tensor.
        cos: (T, D/2)            cos table.
        sin: (T, D/2)            sin table.

    Returns:
        Rotated tensor, same shape as x.
    """
    # Split the last dim into (even, odd) halves which form 2D rotation pairs.
    # x[..., 0::2] gets the even indices; x[..., 1::2] gets the odd indices.
    x_even = x[..., 0::2]   # (B, H, T, D/2)
    x_odd = x[..., 1::2]    # (B, H, T, D/2)

    # The 2D rotation [[c, -s], [s, c]] applied to each pair (x_even, x_odd).
    rot_even = x_even * cos - x_odd * sin
    rot_odd = x_even * sin + x_odd * cos

    # Interleave them back to the original layout.
    out = torch.empty_like(x)
    out[..., 0::2] = rot_even
    out[..., 1::2] = rot_odd
    return out


# ---------------------------------------------------------------------------
# Grouped-Query Attention
# ---------------------------------------------------------------------------
# Standard multi-head attention has H Q, H K, H V projections.
# GQA shrinks K and V to Hkv < H heads. Multiple Q heads share each K/V head.
#
# Memory benefit: the KV cache during inference shrinks by H/Hkv.
# Quality cost: small to negligible at moderate ratios (e.g. 4x).
#
# At 10M params this is overkill, but we include it because:
#   1. It teaches the modern attention pattern.
#   2. Setting n_kv_head = n_head recovers standard MHA — same code, no branch.
# ---------------------------------------------------------------------------


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.n_head = cfg.n_head
        self.n_kv_head = cfg.n_kv_head
        self.head_dim = cfg.head_dim

        # Single fused QKV projection? We use three separate projections because
        # under GQA, Q has a different output size than K/V.
        self.wq = nn.Linear(cfg.n_embd, self.n_head * self.head_dim, bias=False)
        self.wk = nn.Linear(cfg.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.wv = nn.Linear(cfg.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_head * self.head_dim, cfg.n_embd, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape

        # Project to Q, K, V. Reshape to separate the head dimension.
        q = self.wq(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)        # (B, H,   T, D)
        k = self.wk(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)     # (B, Hkv, T, D)
        v = self.wv(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)     # (B, Hkv, T, D)

        # Apply RoPE to Q and K only (V doesn't carry positional info).
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # F.scaled_dot_product_attention does the heavy lifting:
        #   - softmax(Q · K^T / sqrt(D)) · V
        #   - causal masking (we set is_causal=True so future tokens are hidden)
        #   - GQA broadcasting (enable_gqa=True allows Q-heads > K/V-heads)
        #   - dispatches to Flash Attention on supported GPUs (H100, A100, etc.)
        # The fused kernel is *dramatically* faster than a hand-rolled version
        # and uses far less memory because it never materializes the full
        # B*H*T*T attention matrix.
        y = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
            enable_gqa=(self.n_head != self.n_kv_head),
        )  # (B, H, T, D)

        # Re-merge the heads.
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_head * self.head_dim)
        return self.wo(y)


# ---------------------------------------------------------------------------
# SwiGLU FFN
# ---------------------------------------------------------------------------
# The FFN inside each transformer block. Modern design uses a gated activation:
#
#     FFN(x) = W_o( swish(W_gate · x) ⊙ (W_up · x) )
#
# where swish(z) = z * sigmoid(z) (also called SiLU).
# Three matrices instead of two — slightly more params, but better quality.
#
# Inner dimension: traditionally 4 * d_model for GELU/ReLU FFNs.
# For SwiGLU we use 2/3 * 4 * d_model so the *total parameter count* matches
# a 4x GELU FFN (because SwiGLU has 3 matrices instead of 2).
# Rounded to the nearest multiple of 64 for hardware-friendly shapes.
# ---------------------------------------------------------------------------


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        # Match total FFN params to a vanilla 4x GELU FFN.
        hidden = int(cfg.ffn_mult * cfg.n_embd * 2 / 3)
        # Round up to nearest multiple of 64 for GPU tile alignment.
        hidden = ((hidden + 63) // 64) * 64

        self.w_gate = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.w_up = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.w_down = nn.Linear(hidden, cfg.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


# ---------------------------------------------------------------------------
# Transformer block — pre-norm wiring
# ---------------------------------------------------------------------------
# Each block is: x -> x + attn(norm(x)) -> x + ffn(norm(x))
#
# Pre-norm (norm BEFORE the sub-layer, residual around the whole thing) is
# stable to train at depth. Post-norm (the original Transformer's design) is
# now considered legacy because it needs careful learning-rate warmup.
# ---------------------------------------------------------------------------


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.norm_attn = RMSNorm(cfg.n_embd, eps=cfg.norm_eps)
        self.attn = Attention(cfg)
        self.norm_ffn = RMSNorm(cfg.n_embd, eps=cfg.norm_eps)
        self.ffn = SwiGLU(cfg)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm_attn(x), cos, sin)
        x = x + self.ffn(self.norm_ffn(x))
        return x


# ---------------------------------------------------------------------------
# The full model
# ---------------------------------------------------------------------------


class TinyLLM(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm_out = RMSNorm(cfg.n_embd, eps=cfg.norm_eps)
        # NOTE: the output projection is intentionally NOT a separate Linear.
        # We tie weights with tok_emb by computing logits = x @ tok_emb.weight.T
        # inside forward(). This saves vocab_size * n_embd params and is a mild
        # regularizer. Standard for small models.

        # RoPE tables are not parameters — we register them as buffers so they
        # move with .to(device) but aren't counted as learnable.
        cos, sin = build_rope_cache(
            seq_len=cfg.block_size,
            head_dim=cfg.head_dim,
            base=cfg.rope_base,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # GPT-2 trick: scale down residual-path projections by 1/sqrt(2*n_layer)
        # to keep activation magnitudes bounded as depth grows.
        scale = 1.0 / math.sqrt(2 * cfg.n_layer)
        for name, p in self.named_parameters():
            if name.endswith("wo.weight") or name.endswith("w_down.weight"):
                p.data.mul_(scale)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        # GPT-2 init: small Gaussian for Linears and Embeddings, ones for norms.
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Compute logits and (optionally) cross-entropy loss.

        Args:
            idx:     (B, T) int64 token IDs.
            targets: (B, T) int64 token IDs. If provided, returns loss.

        Returns:
            logits: (B, T, vocab_size)
            loss:   scalar tensor or None
        """
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"sequence length {T} exceeds block_size {self.cfg.block_size}"

        x = self.tok_emb(idx)                       # (B, T, C)
        cos = self.rope_cos[:T]                     # (T, D/2)
        sin = self.rope_sin[:T]                     # (T, D/2)

        for block in self.blocks:
            x = block(x, cos, sin)
        x = self.norm_out(x)

        # Tied embeddings: project back to vocab using the embedding matrix.
        logits = x @ self.tok_emb.weight.T          # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # Standard next-token cross-entropy. Flatten over (B*T) for the loss.
            loss = F.cross_entropy(
                logits.view(B * T, -1),
                targets.view(B * T),
                ignore_index=-100,  # for SFT loss-masking later
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Greedy / sampled generation. For Phase 7 demo use, not training."""
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]  # truncate to context window
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-8)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1) if temperature > 0 else logits.argmax(dim=-1, keepdim=True)
            idx = torch.cat([idx, next_token], dim=1)
        return idx

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_emb.weight.numel()
        return n
