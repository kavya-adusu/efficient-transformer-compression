
import json
from pathlib import Path

import torch
from torch.ao.quantization import quantize_dynamic
from datasets import load_dataset
from transformers import AutoTokenizer

from evaluate import evaluate_perplexity, load_optimized_model
from quantization import convert_conv1d

MODEL_NAME = "distilgpt2"
CHECKPOINT = Path("checkpoints/quantized_int8.pt")
RESULTS = Path("results/quantized_validation.json")
MAX_TOKENS = 4096


def main():
    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            "Run python src/quantization.py first."
        )

    # Quantized operators run on CPU.
    torch.backends.quantized.engine = "qnnpack"
    device = "cpu"

    print("Loading WikiText-2 validation dataset...")

    dataset = load_dataset(
        "Salesforce/wikitext",
        "wikitext-2-raw-v1",
        split="validation",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    text = "\n\n".join(
        row["text"]
        for row in dataset
        if row["text"].strip()
    )

    token_ids = tokenizer(
        text,
        add_special_tokens=False,
        verbose=False,
    )["input_ids"][:MAX_TOKENS]

    tokens = torch.tensor(
        [token_ids],
        dtype=torch.long,
    )

    print(f"Evaluation tokens: {tokens.shape[1]}")

    # Evaluate the pruned FP32 model.
    print("\nEvaluating pruned FP32 model...")

    model = load_optimized_model().cpu().eval()

    fp32_metrics = evaluate_perplexity(
        model, tokens, device
    )

    # Rebuild the quantized architecture.
    print("\nLoading saved INT8 checkpoint...")

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=True,
    )

    convert_conv1d(model.transformer)

    target_modules = set(
        checkpoint["quantized_modules"]
    )

    quantized = quantize_dynamic(
        model,
        qconfig_spec=target_modules,
        dtype=torch.qint8,
        inplace=True,
    )

    quantized.load_state_dict(
        checkpoint["state_dict"],
        strict=True,
    )
    quantized.eval()

    # Evaluate the saved quantized weights.
    print("\nEvaluating INT8 model...")

    int8_metrics = evaluate_perplexity(
        quantized, tokens, device
    )

    # Compare language-model quality.
    loss_increase = (
        int8_metrics["loss"]
        - fp32_metrics["loss"]
    )

    results = {
        "dataset": "WikiText-2 validation",
        "tokens": len(token_ids),
        "model": MODEL_NAME,
        "fp32_pruned": fp32_metrics,
        "int8_pruned": int8_metrics,
        "loss_increase": loss_increase,
    }

    RESULTS.parent.mkdir(
        parents=True, exist_ok=True
    )

    RESULTS.write_text(
        json.dumps(results, indent=2)
    )

    print("\n--- Quantized Model Evaluation ---")
    print(
        f"Pruned FP32 perplexity: "
        f"{fp32_metrics['perplexity']:.2f}"
    )
    print(
        f"Pruned INT8 perplexity: "
        f"{int8_metrics['perplexity']:.2f}"
    )
    print(f"Validation loss increase: {loss_increase:.4f}")

    print("\nEvaluation completed successfully!")


if __name__ == "__main__":
    main()
