
import csv
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.pytorch_utils import Conv1D

from sensitivity import evaluate_loss

MODEL_NAME = "distilgpt2"
MAX_LOSS_INCREASE = 0.10

RESULTS_PATH = Path("outputs/sensitivity.csv")
CHECKPOINT_DIR = Path("checkpoints")


def parameter_memory(model):
    """Return parameter storage in MiB."""
    return sum(
        p.numel() * p.element_size()
        for p in model.parameters()
    ) / (1024 ** 2)


def select_pruning_rates(num_layers):
    """Select the highest tested safe rate per layer."""
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            "Run python src/sensitivity.py first."
        )

    rates = {i: 0.0 for i in range(num_layers)}

    with RESULTS_PATH.open(newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            layer = int(row["layer"])
            rate = float(row["pruning_rate"])
            loss_increase = float(row["loss_increase"])

            if layer not in rates:
                raise ValueError("Unexpected layer in CSV")

            if loss_increase <= MAX_LOSS_INCREASE:
                rates[layer] = max(rates[layer], rate)

    return rates


def prune_mlp(block, pruning_rate):
    """Physically remove selected FFN neurons."""
    old_fc = block.mlp.c_fc
    old_proj = block.mlp.c_proj

    original_width = old_fc.weight.shape[1]

    keep_count = max(
        1, round(original_width * (1 - pruning_rate))
    )

    if keep_count == original_width:
        return original_width

    # Rank neurons using the same metric as
    # our earlier sensitivity experiment.
    importance = torch.linalg.vector_norm(
        old_fc.weight.detach().float(),
        dim=0
    )

    # Retain the highest-importance neurons.
    indices = torch.topk(
        importance, k=keep_count
    ).indices.sort().values

    device = old_fc.weight.device
    dtype = old_fc.weight.dtype

    # Smaller input-to-hidden projection.
    new_fc = Conv1D(
        keep_count, old_fc.weight.shape[0]
    ).to(device=device, dtype=dtype)

    # Smaller hidden-to-output projection.
    new_proj = Conv1D(
        old_proj.weight.shape[1], keep_count
    ).to(device=device, dtype=dtype)

    with torch.no_grad():
        new_fc.weight.copy_(
            old_fc.weight[:, indices]
        )
        new_fc.bias.copy_(
            old_fc.bias[indices]
        )

        new_proj.weight.copy_(
            old_proj.weight[indices, :]
        )
        new_proj.bias.copy_(
            old_proj.bias
        )

    # Replace original modules with smaller ones.
    block.mlp.c_fc = new_fc
    block.mlp.c_proj = new_proj

    return keep_count


def main():
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    print("Loading original transformer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME
    ).to(device)
    model.eval()

    original_params = sum(
        p.numel() for p in model.parameters()
    )
    original_memory = parameter_memory(model)

    baseline_loss = evaluate_loss(
        model, tokenizer, device
    )

    # Choose a different pruning rate for each layer.
    layers = model.transformer.h
    rates = select_pruning_rates(len(layers))

    print("\nAdaptive pruning plan:")

    widths = []

    for layer_idx, block in enumerate(layers):
        rate = rates[layer_idx]

        width = prune_mlp(block, rate)
        widths.append(width)

        print(
            f"Layer {layer_idx}: "
            f"{rate:.0%} pruning, "
            f"{width} neurons retained"
        )

    # Evaluate the entire compressed model.
    model.eval()
    compressed_loss = evaluate_loss(
        model, tokenizer, device
    )

    compressed_params = sum(
        p.numel() for p in model.parameters()
    )
    compressed_memory = parameter_memory(model)

    reduction = (
        (original_memory - compressed_memory)
        / original_memory * 100
    )

    print("\n--- Compression Results ---")
    print(f"Original params: {original_params:,}")
    print(f"Compressed params: {compressed_params:,}")
    print(f"Original memory: {original_memory:.2f} MiB")
    print(f"Compressed memory: {compressed_memory:.2f} MiB")
    print(f"Memory reduction: {reduction:.2f}%")
    print(f"Baseline loss: {baseline_loss:.4f}")
    print(f"Compressed loss: {compressed_loss:.4f}")
    print(
        f"Total loss increase: "
        f"{compressed_loss - baseline_loss:.4f}"
    )

    CHECKPOINT_DIR.mkdir(exist_ok=True)

    # Save model weights and layer-width metadata.
    model.cpu()

    torch.save(
        {
            "base_model": MODEL_NAME,
            "layer_widths": widths,
            "state_dict": model.state_dict(),
        },
        CHECKPOINT_DIR / "compressed_model.pt"
    )

    summary = {
        "model": MODEL_NAME,
        "pruning_rates": rates,
        "layer_widths": widths,
        "original_parameters": original_params,
        "compressed_parameters": compressed_params,
        "original_memory_mib": original_memory,
        "compressed_memory_mib": compressed_memory,
        "memory_reduction_percent": reduction,
        "baseline_loss": baseline_loss,
        "compressed_loss": compressed_loss,
    }

    Path("outputs").mkdir(exist_ok=True)

    with open("outputs/compression.json", "w") as file:
        json.dump(summary, file, indent=2)

    print("\nCompressed checkpoint saved successfully!")


if __name__ == "__main__":
    main()
