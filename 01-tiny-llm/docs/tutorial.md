# Tutorial — Build Intuition, Then Build the Model

A narrative walkthrough that builds intuition for *why* the model works, not just what each component does. Read alongside [`glossary.md`](./glossary.md): the glossary defines terms, this tutorial connects them.

**Status:** skeleton. Each section will be written in full as the corresponding part of the project is built. The headings below define the learning path.

**Before reading this:** if you want to understand the full pipeline (data → tokenizer → pretraining → SFT → alignment) before diving into the architecture details, read [`from-scratch.md`](./from-scratch.md) first.

---

## How to use this document

- Read it in order. Each section assumes you understand the previous one.
- The glossary is the dictionary; this is the textbook.
- Worked examples will use small numbers (e.g. `d_model=4`, `seq_len=3`) so you can do the arithmetic by hand.
- At the end of each section there's a "check yourself" prompt — a small question that confirms you got it.

---

## 1. Tokens and tokenization

> *Why does the model see `["The", " cat", " sat"]` and not `"The cat sat"`?*

**Goals of this section:**
- Understand that a model can only operate on numbers, not text.
- See how BPE builds a vocabulary by merging common pairs.
- Understand why tokenizer choice affects everything downstream (sequence length, embedding table size, perplexity comparability).
- Tokenize a sample sentence by hand with a tiny vocabulary.

**Worked example to come:** train a 256-token BPE on the first 1000 lines of Shakespeare and tokenize "to be or not to be."

**Check yourself:** if your vocab is 8192 and your `d_model` is 384, how big is the embedding table? Why does it matter?

---

## 2. A single attention head — the heart of it all

> *How does "the cat sat on the mat" let the model know "it" refers to "cat" three sentences later?*

**Goals:**
- See attention as a soft, differentiable dictionary lookup.
- Understand Q, K, V as three different "roles" each token plays.
- Walk through the `softmax(QKᵀ/√d)V` computation with a 3-token example.
- See why the `√d` scaling matters (without it, softmax saturates).
- Understand the causal mask as preventing information leakage from the future.

**Worked example to come:** 3 tokens, `d_model=4`, single head. Compute every Q, K, V, the attention scores, the mask, the softmax, the final output — by hand.

**Check yourself:** if every Q vector were identical, what would the attention output look like? Why is that bad?

---

## 3. Multi-head attention → GQA

> *Why have many small attention heads instead of one big one? And why do modern models make them share keys and values?*

**Goals:**
- Multi-head as "running attention in several different subspaces in parallel."
- See why splitting helps: different heads can specialize (e.g. syntactic vs semantic relations).
- Understand the K/V cache and why it's the dominant inference cost at long context.
- Walk from MHA → MQA → GQA and see exactly what's being shared.
- Understand why GQA is roughly free quality-wise but big memory-wise.

**Check yourself:** with `n_head=6, d_model=384`, what's `head_dim`? With `kv_heads=2` (GQA), how many `Q→K` head pairs share each K?

---

## 4. The full transformer block

> *Attention + FFN + norms + residuals. Why each one matters.*

**Goals:**
- See the block as `x + attention(norm(x))` then `x + ffn(norm(x))`.
- Understand the residual connection as the "highway" gradients flow on.
- See pre-norm vs post-norm and why pre-norm is the modern default.
- Understand why the FFN expands dimension by ~4× and then contracts.
- See SwiGLU vs GELU — the `gate ⊙ value` motif.
- Connect RMSNorm to LayerNorm (it's LayerNorm with the mean step deleted).

**Check yourself:** if you removed the residual connections, what happens to gradients in a 6-layer model? Why?

---

## 5. Positional information — from absolute to RoPE

> *The model has no inherent notion of "first token" vs "last token." How do we tell it?*

**Goals:**
- See why "permutation-invariant" attention is a problem.
- Walk through three approaches: learned absolute (GPT-2), sinusoidal (original Transformer), rotary (modern).
- Understand RoPE as rotating Q and K in 2D pairs by an angle proportional to position.
- See *why* this makes the dot product `Q·K` depend on the relative offset between positions.
- Understand why RoPE generalizes better to longer sequences than learned embeddings.

**Check yourself:** if you trained with `seq_len=1024` and tried to generate at `seq_len=2048`, which positional scheme would have the best shot at working?

---

## 6. Training — loss, optimizer, schedule

> *How does the model actually learn?*

**Goals:**
- Cross-entropy loss as "how surprised was the model by the truth?"
- See backpropagation conceptually (you don't need to derive it).
- AdamW as "SGD with per-parameter learning rates from gradient stats."
- Why warmup + cosine decay is the standard recipe.
- Why gradient clipping prevents loss spikes.
- bf16 mixed precision — why it works, why fp16 would be worse.
- Gradient accumulation as a "fake big batch."

**Check yourself:** if your loss is decreasing but your *validation* loss starts going up, what's happening and what would you do?

---

## 7. Pretraining → SFT → (RLHF/DPO)

> *Why pretraining alone produces a "completion engine" not a chatbot, and what SFT actually fixes.*

**Goals:**
- Pretraining = next-token prediction on raw text. Result: model continues whatever you start.
- SFT = next-token prediction on `(prompt, response)` pairs with loss masking on the prompt. Result: model learns to respond, not just continue.
- See concretely what the chat template `<|user|>...<|assistant|>...` does at training and inference time.
- Mention RLHF and DPO as "the third stage" — what they add (preference alignment) without going deep.

**Worked example to come:** show the same prompt fed to the base model and the SFT model in a side-by-side, with a single dramatic difference in output.

**Check yourself:** if you forgot to mask the loss on the prompt during SFT, what would the model learn to do?

---

## 8. Evaluation — how do we know it works?

> *A 10M model will fail every benchmark. So what does "evaluation" even mean here?*

**Goals:**
- See loss/perplexity/BPB as intrinsic measures.
- Why perplexity isn't comparable across tokenizers, but BPB is.
- Zero-shot multiple-choice eval via lm-evaluation-harness — how it works mechanically (rank candidate completions by log-likelihood).
- Why a 10M model scoring 25% on a 4-way multiple-choice benchmark = random = expected.
- Qualitative eval: fixed prompts, sampled generations, before/after SFT comparisons.
- The portfolio framing: show the *trajectory* (loss curves over time) and the *methodology*, not the headline benchmark number.

**Check yourself:** if your model scores 25% on HellaSwag, is it broken? How would you know?

---

## Going further

After finishing this project, the natural next directions:

- **DPO** — add preference alignment as a third training stage.
- **Speculative decoding** — generate faster by using the model to verify draft tokens from a smaller model.
- **MoE (Mixture of Experts)** — scale capacity without scaling compute per token.
- **Vision encoder** — make the model multimodal (project 02?).
- **State-space models (Mamba)** — an alternative to attention with linear-time inference.

These could each be future projects in this `foundational-models/` repo.
