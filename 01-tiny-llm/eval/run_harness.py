"""Standard zero-shot benchmarks via EleutherAI's lm-evaluation-harness.

NOTE: a 10M-parameter model will score at or near *chance* on every benchmark
in lm-eval-harness. We run it anyway because:
  1. The methodology is the portfolio signal — "I ran lm-eval-harness".
  2. It produces a comparable artifact that any reader can interpret.
  3. The relative improvement between base and SFT is still informative.

lm-eval-harness expects a model that exposes a standard interface. The easiest
adapter is to export our checkpoint as a HuggingFace-compatible directory and
use the `hf-causal` model type. That requires writing a small HF-style model
wrapper, which we skip here for simplicity — instead this file documents the
two-step workflow.

Workflow (run on a RunPod box with the `eval` group installed):

    # 1. Install the eval dep group.
    uv sync --group eval

    # 2. Export the checkpoint as an HF-compatible model directory.
    #    (Not yet implemented — see "Implementation note" below.)
    python eval/export_to_hf.py --ckpt checkpoints/pretrain/best.pt --out hf_export/

    # 3. Run the harness.
    uv run --group eval lm_eval \\
        --model hf \\
        --model_args pretrained=hf_export \\
        --tasks hellaswag,arc_easy,piqa \\
        --device cuda \\
        --batch_size 16 \\
        --output_path eval/harness_results.json

Implementation note:
  Writing the HF wrapper is straightforward but non-trivial — you need
  config.json, a model class that subclasses PreTrainedModel, and a
  tokenizer that lm-eval can load. For a 10M model the ROI is low because
  the benchmark scores will be near random anyway. Suggested punt: skip
  the harness and lean on the bits-per-byte number from evaluate_ppl.py
  plus the qualitative samples from generate_samples.py.

If you decide to do it later, the cleanest path is:
  - Subclass `transformers.PreTrainedModel` and `PretrainedConfig`.
  - Re-use the existing TinyLLM forward by adapting input/output naming.
  - Save the tokenizer JSON next to the model and register a tokenizer
    config so AutoTokenizer.from_pretrained works.
"""


def main() -> None:
    raise SystemExit(
        "run_harness.py is a documented workflow, not a runnable script.\n"
        "See the docstring at the top of this file for the steps.\n"
        "For meaningful eval at this model scale, use eval/evaluate_ppl.py and\n"
        "eval/generate_samples.py instead."
    )


if __name__ == "__main__":
    main()
