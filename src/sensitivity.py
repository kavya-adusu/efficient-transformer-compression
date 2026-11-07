
import csv
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "distilgpt2"
PRUNING_RATES = [0.10, 0.20, 0.30, 0.40]

# Small evaluation sample to verify our pipeline.
# Later we'll use a larger, held-out dataset.
TEXTS = [
    "Machine learning models learn patterns from data.",
    "Transformers use attention mechanisms to process text.",
    "Efficient neural networks require fewer resources.",
]


@torch.inference_mode()
def evaluate_loss(model, tokenizer, device):
    """Measure average language modeling loss."""
    losses = []

    for text in TEXTS:
        inputs = tokenizer(
            text, return_tensors="pt"
        ).to(device)

        outputs = model(
            **inputs,
            labels=inputs["input_ids"]
        )

        losses.append(outputs.loss.item())

    return sum(losses) / len(losses)


def main():
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    print("Loading model...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME
    ).to(device)
    model.eval()

    baseline_loss = evaluate_loss(
        model, tokenizer, device
    )
    print(f"Baseline loss: {baseline_loss:.4f}")

    results = []

    # DistilGPT2 contains 6 transformer blocks.
    for layer_idx, block in enumerate(model.transformer.h):

        # GPT2 Conv1D uses shape [input, output].
        weights = block.mlp.c_fc.weight.detach()

        # Rank FFN neurons by incoming weight magnitude.
        importance = torch.linalg.vector_norm(
            weights.float(), dim=0
        )

        for rate in PRUNING_RATES:
            num_pruned = int(len(importance) * rate)

            indices = torch.argsort(importance)[
                :num_pruned
            ]

            mask = torch.ones_like(importance)
            mask[indices] = 0

            # Temporarily mask FFN neuron activations.
            def prune_hook(module, inputs, output):
                return output * mask

            handle = block.mlp.act.register_forward_hook(
                prune_hook
            )

            try:
                pruned_loss = evaluate_loss(
                    model, tokenizer, device
                )
            finally:
                handle.remove()

            loss_increase = pruned_loss - baseline_loss

            results.append({
                "layer": layer_idx,
                "pruning_rate": rate,
                "baseline_loss": baseline_loss,
                "pruned_loss": pruned_loss,
                "loss_increase": loss_increase,
            })

            print(
                f"Layer {layer_idx} | "
                f"Pruning {rate:.0%} | "
                f"Loss increase: {loss_increase:.4f}"
            )

    # Save experiment results.
    Path("outputs").mkdir(exist_ok=True)

    with open(
        "outputs/sensitivity.csv",
        "w",
        newline=""
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=results[0].keys()
        )
        writer.writeheader()
        writer.writerows(results)

    print("Results saved to outputs/sensitivity.csv")


if __name__ == "__main__":
    main()
