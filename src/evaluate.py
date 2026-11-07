
import gc
import json
import math
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)
from transformers.pytorch_utils import Conv1D

MODEL_NAME = "distilgpt2"
CHECKPOINT = Path("checkpoints/optimized_model.pt")

MAX_TOKENS = 4096
CONTEXT_SIZE = 256
STRIDE = 128


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def memory_mib(model):
    return sum(
        p.numel() * p.element_size()
        for p in model.parameters()
    ) / (1024 ** 2)


def load_optimized_model():
    """Reconstruct the smaller transformer."""

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=True,
    )

    config = AutoConfig.from_pretrained(
        checkpoint["base_model"]
    )

    model = AutoModelForCausalLM.from_config(config)
    widths = checkpoint["layer_widths"]

    if len(widths) != len(model.transformer.h):
        raise ValueError("Incorrect number of layer widths")

    for block, width in zip(model.transformer.h, widths):
        width = int(width)

        block.mlp.c_fc = Conv1D(
            width, config.n_embd
        )
        block.mlp.c_proj = Conv1D(
            config.n_embd, width
        )

    model.load_state_dict(
        checkpoint["state_dict"],
        strict=True,
    )

    model.eval()
    return model


@torch.inference_mode()
def evaluate_perplexity(model, token_ids, device):
    """Calculate sliding-window language model loss."""

    model = model.to(device)
    model.eval()

    total_nll = 0.0
    total_predicted = 0

    previous_end = 0
    sequence_length = token_ids.size(1)

    for end in range(
        STRIDE, sequence_length + STRIDE, STRIDE
    ):
        end = min(end, sequence_length)
        begin = max(end - CONTEXT_SIZE, 0)

        input_chunk = token_ids[
            :, begin:end
        ].to(device)

        new_tokens = end - previous_end

        labels = input_chunk.clone()

        # Ignore context tokens already evaluated.
        labels[:, :-new_tokens] = -100

        outputs = model(
            input_ids=input_chunk,
            labels=labels,
        )

        # Causal LM loss predicts the next token.
        predicted = int(
            (labels[:, 1:] != -100).sum().item()
        )

        if predicted > 0:
            total_nll += outputs.loss.item() * predicted
            total_predicted += predicted

        previous_end = end

        if end == sequence_length:
            break

    if total_predicted == 0:
        raise ValueError("Not enough tokens to evaluate")

    average_loss = total_nll / total_predicted
    perplexity = math.exp(average_loss)

    return {
        "loss": average_loss,
        "perplexity": perplexity,
        "predicted_tokens": total_predicted,
    }


def main():
    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            "Run python src/global_optimize.py first."
        )

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

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

    ids = tokenizer(
        text,
        add_special_tokens=False,
        verbose=False,
    )["input_ids"][:MAX_TOKENS]

    token_ids = torch.tensor(
        [ids], dtype=torch.long
    )

    print(f"Device: {device}")
    print(f"Input tokens: {len(ids)}")

    print("\nEvaluating original model...")

    baseline = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME
    )

    baseline_params = count_parameters(baseline)
    baseline_memory = memory_mib(baseline)

    baseline_metrics = evaluate_perplexity(
        baseline, token_ids, device
    )

    del baseline
    gc.collect()

    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()

    print("Evaluating compressed model...")

    compressed = load_optimized_model()

    compressed_params = count_parameters(compressed)
    compressed_memory = memory_mib(compressed)

    compressed_metrics = evaluate_perplexity(
        compressed, token_ids, device
    )

    reduction = (
        1 - compressed_memory / baseline_memory
    ) * 100

    results = {
        "dataset": "WikiText-2 validation",
        "input_tokens": len(ids),
        "context_size": CONTEXT_SIZE,
        "stride": STRIDE,
        "baseline": {
            **baseline_metrics,
            "parameters": baseline_params,
            "parameter_memory_mib": baseline_memory,
        },
        "compressed": {
            **compressed_metrics,
            "parameters": compressed_params,
            "parameter_memory_mib": compressed_memory,
        },
        "memory_reduction_percent": reduction,
        "loss_increase": (
            compressed_metrics["loss"]
            - baseline_metrics["loss"]
        ),
    }

    print("\n--- Evaluation Results ---")

    print(
        f"Baseline perplexity: "
        f"{baseline_metrics['perplexity']:.2f}"
    )
    print(
        f"Compressed perplexity: "
        f"{compressed_metrics['perplexity']:.2f}"
    )
    print(
        f"Validation loss increase: "
        f"{results['loss_increase']:.4f}"
    )
    print(f"Memory reduction: {reduction:.2f}%")

    Path("results").mkdir(exist_ok=True)

    with open(
        "results/validation_metrics.json", "w"
    ) as file:
        json.dump(results, file, indent=2)

    print("\nEvaluation completed successfully!")


if __name__ == "__main__":
    main()
