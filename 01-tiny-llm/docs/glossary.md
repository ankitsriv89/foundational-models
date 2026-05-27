# Glossary — Every Term, Explained

Beginner-friendly definitions of every technical term used in this project. Organized by topic, in roughly the order you'll encounter them.

For deeper intuition with worked examples, see [`tutorial.md`](./tutorial.md).

---

## Core concepts

**Foundation model.** A large model trained on broad data (text, images, etc.) that can be adapted to many downstream tasks. GPT-4, Llama, Claude are foundation models. A "tiny" foundation model is the same idea, just small enough to train yourself on a single GPU.

**Pretraining.** The first and most expensive training stage. You show the model trillions of tokens of raw text and ask it to predict the next token at every position. It learns grammar, facts, and reasoning patterns implicitly.

**Fine-tuning.** A *second* training stage on a smaller, task-specific dataset. Adjusts a pretrained model for a specific use case (instruction-following, code, medical Q&A, etc.). Cheap relative to pretraining.

**From scratch.** Starting with randomly initialized weights and doing the pretraining yourself, instead of downloading someone else's pretrained model.

**Decoder-only transformer.** The architecture used by the GPT family. Processes text left-to-right and predicts the next token at each position. "Decoder-only" distinguishes it from encoder-decoder models like T5 or the original transformer paper's machine translation setup.

**Parameter (param).** A single learnable number inside the model — one weight. "10M params" means 10 million such numbers. More params → more capacity, more compute cost, more memory.

**Token.** The atomic unit of text the model sees. Not a word, not a character — typically a chunk like `" hello"`, `"ing"`, or `"."`. The **tokenizer** decides how text becomes tokens.

**Context length (`block_size`, `seq_len`).** The maximum number of tokens the model can see at once. Context length 1024 means the model reads up to 1024 tokens of input when predicting.

**Logits.** The raw, un-normalized scores the model outputs for each possible next token. Apply softmax to get probabilities.

**Cross-entropy loss.** The standard loss function for language modeling. Measures how surprised the model is by the true next token. Lower = better.

---

## Architecture

**Hidden dimension (`d_model`, `n_embd`).** The width of the model's internal representation vector. `d_model = 384` means every token is represented internally as a 384-number vector.

**Layers (`n_layer`).** How many stacked transformer blocks. 6 layers means the input passes through 6 rounds of (attention + feedforward).

**Attention head (`n_head`).** Attention is split into parallel "heads," each looking at different relationships. 6 heads at `d_model=384` → each head works in a 64-dim subspace (`head_dim = d_model / n_head`).

**Attention.** The mechanism that lets each token look at other tokens in the context and decide what's relevant. The defining innovation of transformers. Computed as `softmax(Q·Kᵀ / √d) · V`.

**Q, K, V (Query, Key, Value).** Three vectors derived from each token. Q asks "what am I looking for?", K answers "what do I offer?", V carries the actual content. Attention compares Q to all Ks to decide how much of each V to pull in.

**Causal mask.** A mask that prevents each token from attending to *future* tokens. Essential for decoder-only models — otherwise the model would cheat by looking at the answer.

**RoPE (Rotary Position Embeddings).** A way to encode *where* each token is in the sequence. Instead of adding a position vector to embeddings (the old way), RoPE *rotates* the Q and K vectors by an angle that depends on position. Modern default — better at generalizing to longer sequences than learned/absolute position embeddings.

**RMSNorm (Root Mean Square Normalization).** A normalization layer that stabilizes training. Simpler and faster than the older **LayerNorm** because it skips the mean-subtraction step. Standard in Llama, Mistral, Qwen.

**LayerNorm.** Older normalization scheme — subtracts the mean and divides by std. Slightly slower than RMSNorm, used in GPT-2 and earlier transformers.

**Pre-norm vs post-norm.** Whether the norm happens *before* the attention/FFN block (pre-norm, stable, modern default) or *after* (post-norm, original transformer, hard to train at depth without careful tricks).

**FFN (Feedforward Network).** Half of each transformer block. After attention mixes information *across* tokens, the FFN processes each token *individually* through a small MLP. Typically expands `d_model → 4·d_model → d_model`.

**GELU.** An older activation function used inside the FFN. Smooth approximation of ReLU.

**SwiGLU.** A modern FFN design using a *gated* activation: `swish(W₁x) ⊙ (W₂x)`, then projected back. Slightly better than GELU. Used in Llama, Mistral, most modern LLMs. Adds one extra projection, so the FFN has 3 matrices instead of 2.

**MHA (Multi-Head Attention).** Standard attention: every head has its own Q, K, and V projections.

**GQA (Grouped-Query Attention).** A memory optimization. Groups of Q-heads share the same K and V projections. Reduces the KV cache size (often 4–8×) with negligible quality cost. Standard in Llama-2/3, Mistral. Overkill at 10M params but worth including to show you understand the modern landscape.

**MQA (Multi-Query Attention).** Extreme version of GQA: *one* K and V shared across all Q-heads. Slightly worse quality than GQA, used in some inference-heavy setups.

**KV cache.** During autoregressive generation, the model caches the K and V tensors of all previous tokens so it doesn't recompute them at every step. The biggest memory cost during inference, especially at long context lengths — which is why GQA is so valuable in production.

**Flash Attention.** A fused, memory-efficient implementation of the attention computation. Same math, much faster, uses far less memory by avoiding materializing the full attention matrix. PyTorch's `F.scaled_dot_product_attention` automatically uses Flash Attention on modern GPUs.

**Tied embeddings (weight tying).** Sharing the same weights between the input embedding table (token id → vector) and the output projection (vector → logits over vocab). Saves `vocab_size × d_model` params and acts as a mild regularizer. Standard for small models where the embedding table dominates the param count.

**Weight initialization.** How the random starting values for parameters are chosen. Bad init → training diverges or stalls. GPT-2's recipe: Gaussian with std=0.02, scaled down by `1/√(2·n_layer)` on residual projections.

**Residual connection / skip connection.** `x + f(x)` instead of just `f(x)`. Lets gradients flow cleanly through deep networks. Every transformer block has two: one around attention, one around the FFN.

---

## Tokenizer

**Tokenization.** The process of breaking text into tokens. Done before the model ever sees the input.

**BPE (Byte Pair Encoding).** The standard subword tokenization algorithm. Starts with raw bytes, repeatedly merges the most frequent adjacent pair into a new token, until reaching a target vocab size. Result: common words become one token, rare words split into pieces.

**Byte-level BPE.** Variant used by GPT-2 onwards. Operates on UTF-8 bytes, so the tokenizer can represent *any* text (even emojis, code, non-English) without out-of-vocab errors.

**Vocab size.** Total number of distinct tokens the model knows. Bigger vocab = each token covers more text (shorter sequences) but the embedding table is larger. 8192 is small (modern LLMs use 32K–256K) but right for a 10M model — you don't want the embedding table to dominate the param budget.

**Special tokens.** Reserved tokens with structural meaning: `<|endoftext|>`, `<|user|>`, `<|assistant|>`, `<|pad|>`. The tokenizer must know not to split these.

**Custom tokenizer.** Trained on *your* corpus rather than reused from GPT-2/Llama. Shows you understand tokenization isn't a black box, and adapts the vocab to your domain.

---

## Data

**FineWeb / FineWeb-Edu.** HuggingFace datasets. FineWeb is filtered Common Crawl (~15T tokens); FineWeb-Edu is FineWeb further filtered by an educational-quality classifier (~1.3T tokens). The SmolLM result showed that ~1B tokens of FineWeb-Edu beats ~5B tokens of raw web text — quality beats quantity.

**Common Crawl.** The raw web-scrape dataset that most LLM pretraining corpora derive from.

**TinyStories.** A small synthetic dataset of children's stories. Useful for toy-scale experimentation. Less impressive on a portfolio than FineWeb-Edu.

**Chinchilla scaling.** A 2022 DeepMind paper showing that for compute-optimal training, you want roughly **20 tokens per parameter**. So 10M params → ~200M tokens optimal.

**Overtraining.** Training *past* Chinchilla-optimal — often 10–100× more tokens — because inference cost dominates over training cost in production. A smaller, more-trained model is cheaper to *serve*. Why this project budgets ~1B tokens, not ~200M.

**Streaming.** Reading a huge dataset in chunks from disk or HTTP without downloading the whole thing. HuggingFace `datasets` supports streaming for FineWeb-Edu.

**Sharding.** Splitting the tokenized corpus into many small files (e.g. 100MB `.bin` shards) for fast random access during training. nanoGPT-style.

**Pre-tokenization.** Running the tokenizer once up front and saving the token IDs to disk, so the training loop only does fast disk reads and never CPU-bound tokenization. Without this, the GPU starves waiting for data.

**Token budget.** The total number of tokens the model will see across training. Determines training cost more than any other single number.

---

## Training

**Forward pass.** Run input tokens through the model to produce logits and a loss.

**Backward pass / backpropagation.** Compute gradients of the loss with respect to every parameter.

**Optimizer.** The algorithm that updates parameters using the gradients.

**AdamW.** The standard optimizer for transformers. Adam with *decoupled* weight decay (regularization). Tracks running averages of gradient mean and variance per parameter.

**Learning rate (LR).** The step size for parameter updates. Too high → loss diverges. Too low → training crawls.

**Peak LR.** The maximum LR during training, typically ~3e-4 to ~1e-3 for small models.

**Cosine LR schedule.** Start at peak LR, decay smoothly to ~10% of peak following a cosine curve over the whole training run. Standard for LLMs.

**Warmup.** Linearly ramp the LR from 0 to peak over the first few hundred to few thousand steps. Prevents instability at the start when the model is mostly random.

**Gradient clipping.** Cap the global norm of gradients (e.g. at 1.0) before the optimizer step. Prevents rare exploding-gradient spikes from blowing up training.

**Weight decay.** A regularization term that gently shrinks weights toward zero each step. Typical value: 0.1 for transformers.

**Mixed precision.** Store master weights in fp32 but compute most ops in 16-bit. 2× faster, half the memory.

**fp32 / fp16 / bf16.** Float formats. fp32 = full precision. fp16 = half precision, narrow range, can underflow. **bf16** = "brain float", same exponent range as fp32 but less mantissa precision; the preferred 16-bit format on modern GPUs (A100, H100, RTX 30+/40+).

**Batch size.** How many sequences the model processes in parallel per forward pass. Larger = more stable gradients, more memory.

**Tokens per step.** A more useful number than batch size: `batch_size × seq_len`. Big LLMs aim for 0.5M–4M tokens per optimizer step.

**Gradient accumulation.** Trick to get a large effective batch on a small GPU: run many small forward/backward passes, sum the gradients, then take *one* optimizer step. E.g. `micro_batch=8, accum_steps=64 → effective_batch=512`.

**Checkpointing (saving).** Periodically save model + optimizer state to disk so you can resume after crashes or evaluate intermediate models.

**Gradient checkpointing (activation checkpointing).** A *different* trick: during the backward pass, recompute some activations instead of storing them. Trades compute for memory.

**W&B (Weights & Biases).** A free experiment-tracking service. Logs loss curves, hyperparams, GPU stats. Screenshots of W&B charts are gold in a portfolio README.

**Step / iteration.** One optimizer update.

**Epoch.** One full pass over the dataset. Most LLM pretraining doesn't think in epochs — it thinks in tokens, because the corpus is so large that one epoch is plenty.

**H100 / A100.** NVIDIA datacenter GPUs. H100 is newer and ~2-3× faster for training. Rentable hourly from Lambda Labs, RunPod, Vast.ai.

**Spot / preemptible pricing.** Discounted GPU rentals that the provider can reclaim with short notice. Roughly half the on-demand price. Fine for training if you checkpoint frequently.

**OOM (Out Of Memory).** When the GPU runs out of memory and crashes. Common during model bring-up. Fixes: smaller batch, gradient accumulation, gradient checkpointing, mixed precision.

**Loss curve.** The plot of training/validation loss over steps. Should generally decrease and roughly plateau. Spikes can mean instability.

---

## Evaluation

**Perplexity.** `exp(loss)`. The standard intrinsic LM metric. "If the model assigns true tokens probability `p`, perplexity is `1/p`." Lower = better. Sensitive to tokenizer choice — not comparable across different tokenizers.

**Bits-per-byte (BPB).** A normalized version of loss that *is* comparable across tokenizers, because it measures bits used per raw byte of text. Lower = better. Preferred for fair model comparisons.

**Held-out validation set.** Data the model never trained on, used to measure generalization. Different from the *test* set, which you use only at the very end.

**Zero-shot evaluation.** Asking the model a question without giving it any examples first. Hardest setting. A 10M model will score near random on most benchmarks — that's expected. The *methodology* of running the eval is what matters for the portfolio.

**Few-shot evaluation.** Providing 1–5 worked examples in the prompt before asking. Generally easier for the model.

**lm-evaluation-harness.** EleutherAI's open-source framework for running standardized LLM benchmarks. The industry standard.

**HellaSwag.** Multiple-choice commonsense reasoning benchmark. Pick the most natural completion of a scenario.

**ARC (Easy / Challenge).** Multiple-choice grade-school science questions.

**PIQA (Physical Interaction QA).** Multiple-choice physical commonsense.

**MMLU.** Massive multitask test across 57 academic subjects. The standard "general knowledge" benchmark. Tiny models score near random (25%); useful as a sanity check.

---

## SFT (Supervised Fine-Tuning)

**SFT.** Train the pretrained base model on `(instruction, response)` pairs so it learns to follow instructions. "Instruction tuning." Required for chat-like behavior — a base model just continues text, it doesn't *answer*.

**Chat template.** A fixed format for wrapping conversation, e.g.

```
<|user|>What is 2+2?<|assistant|>4<|endoftext|>
```

The model learns to expect this structure. Must be applied consistently between SFT and inference.

**Instruction dataset.** A collection of `(prompt, response)` pairs. Common public ones:
- **Alpaca** (2023) — 52K synthetic pairs from GPT-3.5. The original.
- **Dolly-15k** — 15K human-written pairs from Databricks employees.
- **Ultrachat-200k** — large, synthetic, multi-turn.

**Loss masking.** During SFT you typically compute the loss *only* on the assistant's response, not on the user's prompt. The model already knows what was said to it; we want it to learn what to *say*.

**Full-parameter SFT.** Update every weight in the model. Expensive at scale but cheap at 10M params.

**LoRA (Low-Rank Adaptation).** Freeze the base weights and only train tiny low-rank "adapter" matrices added to each layer. Massively cheaper memory-wise — used when you can't afford full fine-tuning. Not needed for a 10M model.

**RLHF (Reinforcement Learning from Human Feedback).** A third training stage after SFT: train a reward model on human preference pairs, then use RL (typically PPO) to optimize the LLM against the reward. What made ChatGPT chatty. Out of scope for this project.

**DPO (Direct Preference Optimization).** A simpler alternative to RLHF that skips the reward model and RL. Trains directly on preference pairs with a clever loss function. Increasingly the default. Out of scope for this project.

---

## Deployment

**FastAPI.** A Python web framework for HTTP APIs. Standard for serving ML models.

**Gradio.** A Python library for quick ML demo UIs. Trivially deployable.

**HuggingFace Spaces.** Free hosting for ML demos (Gradio/Streamlit apps). The standard portfolio venue for ML projects.

**HuggingFace Hub.** "GitHub for models." Hosts the weights + a **model card** (README documenting training, eval, intended use, limitations).

**Model card.** Standardized documentation for a model: what it is, what data it was trained on, what it's good/bad at, known biases. Required by most ML platforms.

**Streaming generation.** Producing output one token at a time and sending each to the UI as it's generated, instead of waiting for the full response. Standard chatbot UX.

**Greedy decoding.** Always pick the single most probable next token. Deterministic, often repetitive.

**Sampling.** Pick the next token randomly according to its predicted probability. Produces variety.

**Temperature.** A scalar that scales the logits before softmax. `T=1` is the model's natural distribution; `T<1` sharpens (more deterministic); `T>1` flattens (more random). `T=0` is equivalent to greedy.

**Top-k sampling.** Sample only from the top-k most likely tokens. Cuts off the long tail of bad choices.

**Top-p / nucleus sampling.** Sample from the smallest set of tokens whose cumulative probability exceeds `p`. Adapts to the shape of the distribution.

**Quantization.** Compress model weights to lower precision (int8, int4) for faster, cheaper inference. Out of scope for this project but worth knowing.

**vLLM / Ollama.** Production-grade LLM serving frameworks. Overkill for a 10M model but mentioned because they're the standard answer to "how do I deploy an LLM?"

---

## Useful mental model

The whole pipeline, in one breath:

> Raw text → tokenizer → token IDs → shards on disk → DataLoader feeds the GPU → model does a forward pass → cross-entropy loss → backward pass → AdamW step → repeat for ~50K steps → checkpoint → eval with lm-eval-harness → SFT for another ~5K steps with a chat template → checkpoint → FastAPI + Gradio → HuggingFace Spaces.

Every term in this glossary is a knob or component in that pipeline. When you read code in `model.py` or `train_pretrain.py`, every named variable should map back to something defined here.
