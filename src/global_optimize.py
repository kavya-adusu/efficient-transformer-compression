
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from pruning import prune_mlp, parameter_memory
from sensitivity import evaluate_loss

MODEL_NAME = "distilgpt2"

# Maximum acceptable total loss increase
MAX_LOSS_INCREASE = 0.10

# Candidate pruning rates: 0%, 10%, 20%, 30%, 40%
STEP_SIZE = 0.10
MAX_STEPS = 4


def get_device():
    if torch.cuda.is_available():
        return "cuda"

    if torch.backends.mps.is_available():
        return "mps"

    return "cpu"


def build_masks(model):
    """Precompute neuron masks for every layer and rate."""
    masks = {}

    for layer_idx, block in enumerate(model.transformer.h):
        weights = block.mlp.c_fc.weight.detach()

        importance = torch.linalg.vector_norm(
            weights.float(), dim=0
        )

        width = importance.numel()

        for step in range(1, MAX_STEPS + 1):
            rate = step * STEP_SIZE

            keep_count = max(
                1, round(width * (1 - rate))
            )

            keep_indices = torch.topk(
                importance, k=keep_count
            ).indices

            mask = torch.zeros(
                width,
                device=weights.device,
                dtype=weights.dtype,
            )

            mask[keep_indices] = 1
            masks[(layer_idx, step)] = mask

    return masks


def evaluate_plan(model, tokenizer, device, levels, masks):
    """Evaluate simultaneous pruning across all layers."""
    handles = []

    try:
        for layer_idx, step in enumerate(levels):
            if step == 0:
                continue

            mask = masks[(layer_idx, step)]
            block = model.transformer.h[layer_idx]

            def hook(module, inputs, output, mask=mask):
                return output * mask

            handle = block.mlp.act.register_forward_hook(
                hook
            )
            handles.append(handle)

        return evaluate_loss(model, tokenizer, device)

    finally:
        for handle in handles:
            handle.remove()


def main():
    device = get_device()

    print("Loading transformer...")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME
    ).to(device)

    model.eval()

    original_memory = parameter_memory(model)
    original_params = sum(
        p.numel() for p in model.parameters()
    )

    baseline_loss = evaluate_loss(
        model, tokenizer, device
    )

    print(f"Baseline loss: {baseline_loss:.4f}")

    masks = build_masks(model)
    num_layers = len(model.transformer.h)

    # Each integer represents a 10% pruning step.
    levels = [0] * num_layers
    current_loss = baseline_loss

    print("\nStarting global pruning optimization...")

    for iteration in range(num_layers * MAX_STEPS):
        best = None

        # Try increasing the pruning rate for each layer.
        for layer_idx in range(num_layers):
            if levels[layer_idx] >= MAX_STEPS:
                continue

            candidate = levels.copy()
            candidate[layer_idx] += 1

            candidate_loss = evaluate_plan(
                model,
                tokenizer,
                device,
                candidate,
                masks,
            )

            total_increase = (
                candidate_loss - baseline_loss
            )

            # Reject candidates that exceed the budget.
            if total_increase > MAX_LOSS_INCREASE:
                continue

            # Each step removes approximately the same
            # number of FFN neurons, so choose the candidate
            # with the lowest resulting total model loss.
            if best is None or candidate_loss < best["loss"]:
                best = {
                    "layer": layer_idx,
                    "loss": candidate_loss,
                }

        if best is None:
            print("No further pruning fits the loss budget.")
            break

        chosen_layer = best["layer"]
        levels[chosen_layer] += 1
        current_loss = best["loss"]

        print(
            f"Iteration {iteration + 1}: "
            f"Layer {chosen_layer} -> "
            f"{levels[chosen_layer] * STEP_SIZE:.0%}, "
            f"total loss increase = "
            f"{current_loss - baseline_loss:.4f}"
        )

    # Convert the selected levels into pruning rates.
    rates = [step * STEP_SIZE for step in levels]

    print("\nFinal pruning plan:")
    for layer_idx, rate in enumerate(rates):
        print(f"Layer {layer_idx}: {rate:.0%}")

    # Physically shrink the model's FFN matrices.
    widths = []

    for block, rate in zip(model.transformer.h, rates):
        new_width = prune_mlp(block, rate)
        widths.append(new_width)

    model.eval()

    compressed_loss = evaluate_loss(
        model, tokenizer, device
    )

    # The physical model should closely reproduce
    # the masked model's loss.
    if abs(compressed_loss - current_loss) > 0.02:
        raise RuntimeError(
            "Physical pruning does not match masked evaluation."
        )

    compressed_memory = parameter_memory(model)
    compressed_params = sum(
        p.numel() for p in model.parameters()
    )

    reduction = (
        1 - compressed_memory / original_memory
    ) * 100

    print("\n--- Global Optimization Results ---")
    print(f"Original parameters: {original_params:,}")
    print(f"Compressed parameters: {compressed_params:,}")
    print(f"Original memory: {original_memory:.2f} MiB")
    print(f"Compressed memory: {compressed_memory:.2f} MiB")
    print(f"Memory reduction: {reduction:.2f}%")
    print(f"Baseline loss: {baseline_loss:.4f}")
    print(f"Compressed loss: {compressed_loss:.4f}")
    print(
        f"Total loss increase: "
        f"{compressed_loss - baseline_loss:.4f}"
    )

    # Save separate outputs so previous experiments
    # remain available for comparison.
    Path("checkpoints").mkdir(exist_ok=True)
    Path("outputs").mkdir(exist_ok=True)

    model.cpu()

    torch.save(
        {
            "base_model": MODEL_NAME,
            "layer_widths": widths,
            "state_dict": model.state_dict(),
        },
        "checkpoints/optimized_model.pt",
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
        "loss_increase": compressed_loss - baseline_loss,
        "loss_budget": MAX_LOSS_INCREASE,
    }

    with open("outputs/global_optimization.json", "w") as file:
        json.dump(summary, file, indent=2)

    print("\nGlobal optimization completed!")


if __name__ == "__main__":
    main()
