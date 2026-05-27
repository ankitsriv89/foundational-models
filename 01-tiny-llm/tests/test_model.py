"""Shape and correctness tests for model.py.

These run on CPU and take a few seconds — designed to catch bugs *before* you
burn GPU time. Run with:

    uv run --group train python -m pytest tests/ -v

The tests are intentionally simple and explicit. They check:
  1. Default config builds a ~10M-param model (sanity for the headline number).
  2. Forward pass produces correct shapes.
  3. Loss is finite and roughly log(vocab_size) at init (= random guessing).
  4. Causal masking actually works (changing future tokens doesn't change past logits).
  5. RoPE is applied (rotated outputs differ from un-rotated).
  6. GQA shape contract holds (n_kv_head can be smaller than n_head).
  7. The MHA degenerate case (n_kv_head == n_head) also runs.
  8. Backward pass produces gradients on every parameter.
"""

import math

import pytest
import torch

from model import ModelConfig, RMSNorm, TinyLLM, apply_rope, build_rope_cache


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def small_cfg(**overrides) -> ModelConfig:
    """A faster config for unit tests — same architecture, smaller numbers."""
    defaults = dict(
        vocab_size=256,
        block_size=64,
        n_layer=2,
        n_head=4,
        n_kv_head=2,
        n_embd=64,
    )
    defaults.update(overrides)
    return ModelConfig(**defaults)


# ---------------------------------------------------------------------------
# Param count
# ---------------------------------------------------------------------------


def test_default_config_is_about_10M_params():
    """The headline number — must hold or we're shipping the wrong model."""
    model = TinyLLM(ModelConfig())
    n = model.num_params()
    # Allow ±10% wiggle room around 10M.
    assert 9_000_000 < n < 11_500_000, f"expected ~10M params, got {n:,}"


def test_param_count_excluding_embedding_is_reasonable():
    """Non-embedding params are the actual 'transformer' part."""
    model = TinyLLM(ModelConfig())
    total = model.num_params()
    non_emb = model.num_params(non_embedding=True)
    # The embedding alone is 8192 * 384 = 3,145,728.
    assert total - non_emb == 8192 * 384


# ---------------------------------------------------------------------------
# Forward shape
# ---------------------------------------------------------------------------


def test_forward_logits_shape():
    cfg = small_cfg()
    model = TinyLLM(cfg)
    B, T = 3, 16
    idx = torch.randint(0, cfg.vocab_size, (B, T))
    logits, loss = model(idx)
    assert logits.shape == (B, T, cfg.vocab_size)
    assert loss is None


def test_forward_with_targets_returns_loss():
    cfg = small_cfg()
    model = TinyLLM(cfg)
    B, T = 3, 16
    idx = torch.randint(0, cfg.vocab_size, (B, T))
    targets = torch.randint(0, cfg.vocab_size, (B, T))
    logits, loss = model(idx, targets)
    assert logits.shape == (B, T, cfg.vocab_size)
    assert loss is not None
    assert loss.ndim == 0  # scalar
    assert torch.isfinite(loss)


def test_initial_loss_is_near_log_vocab_size():
    """At init the model is random — expected loss ≈ ln(vocab_size).

    This is the single best sanity check that initialization is sane. If the
    loss is much larger, something is broken (probably weight scaling).
    """
    cfg = small_cfg()
    torch.manual_seed(0)
    model = TinyLLM(cfg)
    idx = torch.randint(0, cfg.vocab_size, (4, 32))
    targets = torch.randint(0, cfg.vocab_size, (4, 32))
    _, loss = model(idx, targets)
    expected = math.log(cfg.vocab_size)
    # Pretty loose bound — init noise can push it 30% either way.
    assert abs(loss.item() - expected) < 1.0, f"loss={loss.item():.3f} expected ~{expected:.3f}"


# ---------------------------------------------------------------------------
# Causal masking
# ---------------------------------------------------------------------------


def test_causal_mask_blocks_future_information():
    """Logits at position t must not depend on tokens at positions > t.

    Trick: run two forward passes with identical prefixes but different
    suffixes. Logits at every position in the prefix must be identical.
    If they differ, the model is leaking future info — usually a missing
    is_causal=True or a bug in attention shape.
    """
    cfg = small_cfg()
    torch.manual_seed(0)
    model = TinyLLM(cfg).eval()

    B, T = 2, 16
    shared = torch.randint(0, cfg.vocab_size, (B, T))
    idx_a = shared.clone()
    idx_b = shared.clone()
    # Differ only at the LAST token.
    idx_b[:, -1] = (idx_b[:, -1] + 1) % cfg.vocab_size

    with torch.no_grad():
        logits_a, _ = model(idx_a)
        logits_b, _ = model(idx_b)

    # All positions EXCEPT the last should be identical.
    assert torch.allclose(logits_a[:, :-1], logits_b[:, :-1], atol=1e-5), (
        "logits at earlier positions changed when only the last token changed — "
        "causal mask is broken"
    )


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------


def test_rope_cache_shape():
    cos, sin = build_rope_cache(seq_len=32, head_dim=16, base=10000.0, device=torch.device("cpu"), dtype=torch.float32)
    assert cos.shape == (32, 8)
    assert sin.shape == (32, 8)


def test_rope_position_zero_is_identity():
    """At position 0, rotation angle is 0, so the vector is unchanged."""
    head_dim = 8
    cos, sin = build_rope_cache(seq_len=4, head_dim=head_dim, base=10000.0, device=torch.device("cpu"), dtype=torch.float32)
    x = torch.randn(1, 1, 4, head_dim)
    rotated = apply_rope(x, cos, sin)
    assert torch.allclose(rotated[:, :, 0, :], x[:, :, 0, :], atol=1e-6)
    # Position 1 should NOT be the identity.
    assert not torch.allclose(rotated[:, :, 1, :], x[:, :, 1, :], atol=1e-3)


def test_rope_preserves_norm():
    """Rotation is orthogonal — it can't change vector magnitude."""
    head_dim = 16
    cos, sin = build_rope_cache(seq_len=8, head_dim=head_dim, base=10000.0, device=torch.device("cpu"), dtype=torch.float32)
    x = torch.randn(2, 3, 8, head_dim)
    rotated = apply_rope(x, cos, sin)
    norm_before = x.norm(dim=-1)
    norm_after = rotated.norm(dim=-1)
    assert torch.allclose(norm_before, norm_after, atol=1e-5)


# ---------------------------------------------------------------------------
# GQA shape contract
# ---------------------------------------------------------------------------


def test_gqa_with_unequal_q_kv_heads_runs():
    cfg = small_cfg(n_head=4, n_kv_head=2)
    model = TinyLLM(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, 8))
    logits, _ = model(idx)
    assert logits.shape == (2, 8, cfg.vocab_size)


def test_mha_degenerate_case_runs():
    """n_kv_head == n_head should reduce to standard multi-head attention."""
    cfg = small_cfg(n_head=4, n_kv_head=4)
    model = TinyLLM(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, 8))
    logits, _ = model(idx)
    assert logits.shape == (2, 8, cfg.vocab_size)


def test_invalid_gqa_config_raises():
    """n_head must be divisible by n_kv_head."""
    with pytest.raises(AssertionError):
        ModelConfig(n_head=6, n_kv_head=4)  # 6 % 4 != 0


# ---------------------------------------------------------------------------
# Backward pass
# ---------------------------------------------------------------------------


def test_backward_produces_gradients_on_all_params():
    """If any param has grad=None after .backward(), it's dead weight.

    This catches: parameters that are defined but never used in forward(),
    or shape bugs that disconnect part of the graph.
    """
    cfg = small_cfg()
    model = TinyLLM(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, 8))
    targets = torch.randint(0, cfg.vocab_size, (2, 8))
    _, loss = model(idx, targets)
    loss.backward()
    dead = [n for n, p in model.named_parameters() if p.grad is None]
    assert not dead, f"these parameters got no gradient: {dead}"


# ---------------------------------------------------------------------------
# RMSNorm sanity
# ---------------------------------------------------------------------------


def test_rmsnorm_normalizes_to_unit_rms():
    """After RMSNorm, the RMS of each vector should be ~1 (since gain init = 1)."""
    norm = RMSNorm(dim=64)
    x = torch.randn(4, 10, 64) * 5.0  # arbitrary scale
    y = norm(x)
    rms = y.pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-3)


# ---------------------------------------------------------------------------
# Generate
# ---------------------------------------------------------------------------


def test_generate_extends_sequence():
    cfg = small_cfg()
    model = TinyLLM(cfg)
    prompt = torch.randint(0, cfg.vocab_size, (1, 4))
    out = model.generate(prompt, max_new_tokens=10, temperature=1.0)
    assert out.shape == (1, 14)
    # First 4 tokens are unchanged (we just appended).
    assert torch.equal(out[:, :4], prompt)
