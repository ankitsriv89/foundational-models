# Building a Foundation Model From Scratch

A complete explanation of every stage between "raw text on disk" and "a model that can answer questions."
This is what Mistral's team did before you opened your fine-tuning notebook.

---

## The two things you get from a pretrained model

When you load `mistralai/Mistral-7B-Instruct-v0.2`, you get two things:

1. **Architecture** — a transformer decoder defined in code (~300 lines)
2. **Weights** — the result of training that architecture on ~1 trillion tokens of text

"Building from scratch" means you produce both of these yourself.

---

## Stage 1 — Data Collection and Cleaning

This is the hardest stage and the one that receives the least attention in tutorials.

### Sources

Common sources for a general-purpose foundation model:

| Source | Raw size | What's in it |
|--------|----------|-------------|
| Common Crawl | 70+ TB | Web pages scraped every month since 2008 |
| Wikipedia | ~20 GB | Encyclopedia articles, all languages |
| Books3 / Gutenberg | ~100 GB | Published books |
| GitHub | ~300 GB | Public source code |
| ArXiv | ~100 GB | Academic papers (STEM) |
| OpenWebText | ~40 GB | Reddit-upvoted web pages (cleaner than raw CC) |

For a domain-specific model (medical, legal, code), you'd weight the mix heavily toward that domain.

### The cleaning pipeline

Raw web text is mostly garbage. A typical pipeline:

```
raw HTML
    ↓ extract text (trafilatura or similar)
    ↓ language detection (fastText — keep English or target language)
    ↓ length filter (drop pages < 200 chars or > 100K chars)
    ↓ quality heuristics:
        - remove pages with too many special characters (>30%)
        - remove pages with too few unique words (likely spam)
        - remove pages that are >50% repeated n-grams (SEO junk)
        - remove pages without sentence-ending punctuation
    ↓ deduplication:
        - exact dedup: hash full document, remove identical copies
        - near-dedup: MinHash LSH, remove pages that are >80% similar
    ↓ PII scrubbing (email addresses, phone numbers)
    ↓ toxicity filtering (optional — depends on use case)
    ↓ domain mixing:
        sample 70% web, 15% code, 10% books, 5% papers
```

**Why deduplication matters:** without it, the model memorizes repeated content instead of generalizing. A page that appears 100 times in your corpus is trained on 100× more than a unique page — the model learns to recite it verbatim.

**Token budget target:**
- 100M param model: ~5–20B tokens
- 1B param model: ~20–100B tokens
- 7B param model (Mistral scale): ~1–2 trillion tokens

The Chinchilla scaling laws (Hoffmann et al. 2022) showed that the optimal ratio is roughly **20 tokens of training data per parameter**. Most modern models are trained at higher ratios (Llama-3 at ~15T tokens for 8B params) because inference costs dominate at deployment.

---

## Stage 2 — Tokenizer Training

The tokenizer converts text to integers. It is trained **before** the model, on your cleaned corpus.

```python
from tokenizers import ByteLevelBPETokenizer

tokenizer = ByteLevelBPETokenizer()
tokenizer.train(
    files=["corpus_sample.txt"],   # a representative sample, not the full corpus
    vocab_size=32000,              # Mistral uses 32000; GPT-4 uses 100277
    min_frequency=2,
    special_tokens=["<s>", "</s>", "<unk>", "<pad>"],
)
tokenizer.save_model("./tokenizer/")
```

**BPE (Byte Pair Encoding) explained:**

Start with individual bytes (256 symbols). Repeatedly:
1. Count all adjacent symbol pairs in the corpus
2. Merge the most frequent pair into a new symbol
3. Repeat until vocabulary reaches target size

Result: common English words become single tokens (`"the"`, `"and"`, `" is"`).
Rare or domain words get split into subword pieces (`"hypertension"` → `"hyper"` + `"tension"`).

**Why train your own tokenizer:**
A general tokenizer wastes vocabulary space on common English words when your domain is, say, medical text with long Latin terms. A domain-specific tokenizer packs more information per token → shorter sequences → cheaper training.

**Vocabulary size tradeoff:**

| Vocab size | Effect |
|-----------|--------|
| Too small (< 8K) | Long sequences, rare words split into many pieces |
| Too large (> 100K) | Large embedding table, sparse gradient updates for rare tokens |
| Sweet spot | 16K–64K for most use cases |

For this project (`01-tiny-llm`) we use vocab_size=8192 to keep the embedding table small relative to the model.

---

## Stage 3 — Model Architecture

A modern transformer decoder. Every component has a specific job.

```python
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        self.attn = GroupedQueryAttention(cfg)   # multi-head attention with GQA
        self.ff = SwiGLU(cfg)                    # gated feed-forward network
        self.norm1 = RMSNorm(cfg.n_embd)         # pre-norm: normalize before attention
        self.norm2 = RMSNorm(cfg.n_embd)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))         # residual: x passes through unchanged
        x = x + self.ff(self.norm2(x))           # + the block's contribution
        return x

class LanguageModel(nn.Module):
    def __init__(self, cfg):
        self.embed = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.layers = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layer)])
        self.norm = RMSNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        # tie weights: lm_head shares the embedding table (saves params, regularizes)
        self.lm_head.weight = self.embed.weight

    def forward(self, input_ids):
        x = self.embed(input_ids)          # (batch, seq_len) → (batch, seq_len, n_embd)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.lm_head(x)             # (batch, seq_len, vocab_size)
```

### Why each component

**RMSNorm (not LayerNorm):**
LayerNorm subtracts the mean and divides by std. RMSNorm skips the mean subtraction (just divides by RMS). Same stabilization effect, ~10% faster. Llama, Mistral, Qwen all use it.

**Pre-norm (`norm` before attention, not after):**
Original "Attention is All You Need" paper used post-norm (normalize after adding residual). Pre-norm (`x + attn(norm(x))`) is easier to train at depth — gradients flow back through the residual highway without going through a norm layer.

**Residual connections:**
`x = x + attn(...)` means the input `x` always passes through unchanged, with the attention output added on top. Without this, a 6-layer model has vanishing gradients — they shrink exponentially as they flow backward through each layer.

**GQA (Grouped Query Attention):**
Standard multi-head attention has N query heads and N key/value heads. GQA uses N query heads but only G key/value heads (G < N), where each group of N/G query heads shares one K/V head.

Why: the KV cache (what makes inference fast at long context) stores one K vector and one V vector per head per token. With N=6 heads and G=2 KV heads, the KV cache is 3× smaller — crucial for serving at long context lengths.

Quality loss: minimal. Llama-2/3 and Mistral both use GQA.

**RoPE (Rotary Position Embedding):**
The model needs to know which token is first, second, third, etc. — pure attention is permutation-invariant.

RoPE encodes position by *rotating* the Q and K vectors. For each 2D pair of dimensions, rotate by angle `θ × position`. The dot product `Q · K` then naturally encodes the *relative* distance between positions.

Why better than learned absolute embeddings (GPT-2 style): generalizes to longer sequences than seen in training, because it's based on relative offset not absolute index.

**SwiGLU (feed-forward network):**
After attention, each token is processed independently through a 2-layer MLP that expands dimension by ~2.67× and contracts back.

SwiGLU: `output = (W1·x) ⊙ σ(W3·x) · W2`
The gate `σ(W3·x)` selectively suppresses or amplifies different dimensions. Slightly better quality than vanilla GELU FFN.

**Scale determines capability — not architecture:**

The architecture above is essentially the same for all sizes:

| Model | n_embd | n_layer | n_head | kv_head | Params |
|-------|--------|---------|--------|---------|--------|
| This project | 384 | 6 | 6 | 2 | ~10M |
| GPT-2 small | 768 | 12 | 12 | 12 | 117M |
| LLaMA-7B | 4096 | 32 | 32 | 32 | 7B |
| Mistral-7B | 4096 | 32 | 32 | 8 | 7.24B |
| LLaMA-70B | 8192 | 80 | 64 | 8 | 70B |

Same code, different numbers.

---

## Stage 4 — Pre-training

The training objective is **next token prediction**: given tokens 1..N, predict token N+1. Cross-entropy loss over the vocabulary.

```python
# Forward pass
logits = model(input_ids[:, :-1])      # predict from all tokens except last
targets = input_ids[:, 1:]             # targets are all tokens except first

# Loss: how surprised was the model by the actual next token?
loss = F.cross_entropy(
    logits.reshape(-1, vocab_size),
    targets.reshape(-1),
)

# Backward pass
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # prevent gradient explosions
optimizer.step()
scheduler.step()
optimizer.zero_grad()
```

**What the model is actually learning:**
It is learning to compress text. To predict the next token well, the model must internalize grammar, facts, reasoning patterns, code syntax, argument structure — because all of that reduces surprise about what comes next.

This is why scale produces "emergent" capabilities: at some scale, the optimal compression strategy involves actually learning the underlying structure of language and knowledge.

**Key training hyperparameters:**

| Hyperparameter | Typical value | Why |
|---------------|---------------|-----|
| Learning rate | 3e-4 | Higher = faster but unstable; lower = safe but slow |
| LR warmup | 1–5% of steps | Prevents large early gradient updates from corrupting init |
| LR schedule | Cosine decay | Smooth decay to near-zero by end of training |
| Batch size | 256–2048 sequences | Larger = more stable gradients, but needs more memory |
| Gradient clipping | 1.0 | Prevents any single step from taking too large a stride |
| Weight decay | 0.1 | Mild regularization; applied to weight matrices, not biases/norms |
| Precision | bf16 | bfloat16: same exponent range as float32, less precision but numerically stable |

**AdamW optimizer:**
Standard Adam maintains two per-parameter statistics: first moment (gradient mean) and second moment (gradient variance). AdamW adds decoupled weight decay — applies weight decay directly to parameters, not through the gradient. This matters: standard Adam's L2 regularization is distorted by the adaptive scaling.

**bf16 vs fp16:**
Both are 16-bit formats but with different bit allocation.
- fp16: 5 exponent bits, 10 mantissa bits — can represent very precise small numbers but overflows above ~65000
- bf16: 8 exponent bits, 7 mantissa bits — same range as float32, less precision

For training: bf16 is safer. Gradient norms can occasionally spike above 65000, which causes fp16 to produce `inf` or `nan`. bf16 handles this gracefully.

**Expected loss curve:**
- Initial loss ≈ `log(vocab_size)` — the model starts predicting uniformly at random
- Loss drops quickly in early training as the model learns common patterns
- Final loss for a well-trained small model: ~3.5–4.5 (on BPB = bits-per-byte)

---

## Stage 5 — What Pretraining Produces (and Why It's Not Enough)

A pretrained model is a **text completion engine**. It has learned to continue whatever you give it.

If you prompt it with `"The capital of France is"`, it will likely output `"Paris"` — not because it was trained to answer questions, but because that completion appeared in its training data many times.

If you prompt it with `"Q: What is 2+2?\nA:"`, it may or may not output `"4"`. It depends on whether Q&A-format text appeared in the training corpus.

The model has no concept of "I am an assistant, my job is to help." It's a distribution over token sequences, conditioned on input.

---

## Stage 6 — SFT: Teaching the Model to Respond

SFT (Supervised Fine-Tuning) is exactly what your Phase 1 Kaggle notebook does — but applied to a base model you trained yourself instead of a pretrained one.

The key difference from pretraining:

```python
# Pretraining: compute loss on all tokens
loss = cross_entropy(logits, targets)

# SFT: compute loss ONLY on response tokens
# instruction tokens are masked (loss = 0 for those positions)
loss = cross_entropy(logits[response_mask], targets[response_mask])
```

**Why mask the instruction:**
You want the model to learn to produce good responses. If you compute loss on the instruction too, the model learns to predict instructions — which is useless. Response masking is what `SFTTrainer` handles for you automatically.

**Format:**
```
<s>[INST] What is the capital of France? [/INST] The capital of France is Paris. </s>
```

The chat template (`[INST]...[/INST]`) tells the model where instructions end and responses begin. During training, loss is zero for everything up to and including `[/INST]`, nonzero for everything after.

**After SFT:** the model has learned to respond to the chat template format. Prompt it with `[INST] ... [/INST]` and it generates a response rather than continuing the prompt arbitrarily.

---

## Stage 7 — RLHF / DPO (The Alignment Stage)

This is what separates a fine-tuned assistant from a *safe, helpful* assistant — and what Mistral-7B-Instruct does on top of Mistral-7B-base.

**RLHF (Reinforcement Learning from Human Feedback):**
1. Humans compare two model outputs and pick the better one
2. A reward model is trained to predict human preference scores
3. The policy (your LLM) is trained with PPO to maximize the reward model's score
4. KL penalty keeps the model from drifting too far from the SFT checkpoint

**DPO (Direct Preference Optimization):**
Newer, simpler alternative to RLHF. Uses the same preference data (pairs of preferred vs rejected responses) but trains directly with a contrastive loss — no separate reward model, no RL.

```
preference data: (prompt, chosen_response, rejected_response)
↓
DPO loss: increase probability of chosen, decrease probability of rejected
          relative to the SFT baseline
```

For this project, you don't need RLHF/DPO. It matters when you're deploying to users and care about the model not being harmful or evasive.

---

## The Full Pipeline in One View

```
Raw text corpus (TB scale)
    ↓ clean, deduplicate, mix
Training corpus (~B–T tokens)
    ↓ BPE tokenizer training
vocab: 8K–100K tokens
    ↓ tokenize corpus
token ID arrays on disk
    ↓ pre-training (next token prediction, billions of steps)
Base model weights
    ↓ SFT on instruction-response pairs (loss masked on instruction)
Instruction-following model
    ↓ (optional) RLHF or DPO on preference data
Aligned assistant
```

---

## Compute Reality Check

| Scale | Tokens | GPU-hours (A100) | Approximate cost |
|-------|--------|-------------------|------------------|
| 10M params (this project) | 1–2B | ~20–50 hrs | ~$40–100 |
| 117M params (GPT-2 small) | 5–10B | ~200–500 hrs | ~$400–1000 |
| 1B params | 20B | ~5,000 hrs | ~$10,000 |
| 7B params (Mistral scale) | 1T | ~100,000 hrs | ~$200,000 |
| 70B params | 2T | ~1,000,000 hrs | ~$2,000,000 |

One A100-hour on AWS spot ≈ $1–2.

**What's realistic for a learning project:**
Train a 10M model on 1–2B tokens. With Kaggle's free T4 (30hr/week) or Colab free tier, you can get through a meaningful training run in a few sessions. It won't be GPT-4, but it will be a real language model whose every component you understand.

---

## What "from scratch" actually teaches you

Fine-tuning teaches you how to adapt a model.
Pre-training teaches you why the model can be adapted at all.

After pre-training a small model yourself, you understand:
- Why loss = `log(vocab_size)` at init and what it means when it falls
- Why more data and more parameters consistently improve things (scaling laws)
- Why the tokenizer choice affects every downstream metric
- Why SFT works: the base model already has the knowledge, SFT just teaches it the response format
- Why RLHF is a separate stage: SFT teaches format, RLHF teaches preference

The fine-tuning project (Phase 1–5 in `03-llm-finetuning/`) and this project are two halves of the same picture.

---

## Further Reading

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — original transformer paper
- [Training Compute-Optimal LLMs (Chinchilla)](https://arxiv.org/abs/2203.15556) — scaling laws for data/params tradeoff
- [QLoRA](https://arxiv.org/abs/2305.14314) — 4-bit fine-tuning paper
- [RoFormer (RoPE)](https://arxiv.org/abs/2104.09864) — rotary position embeddings
- [GQA](https://arxiv.org/abs/2305.13245) — grouped query attention paper
- [DPO](https://arxiv.org/abs/2305.18290) — direct preference optimization
- [nanoGPT](https://github.com/karpathy/nanoGPT) — Karpathy's ~300-line GPT implementation, best starting point for hands-on pre-training
