
import json
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.pytorch_utils import Conv1D

CHECKPOINT = Path("checkpoints/compressed_model.pt")
SUMMARY = Path("outputs/compression.json")


def load_compressed_model():
    """Reconstruct and load the compressed transformer."""

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=True,
    )

    model_name = checkpoint["base_model"]
    widths = checkpoint["layer_widths"]

    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_config(config)

    # Restore the compressed FFN layer dimensions.
    for block, width in zip(model.transformer.h, widths):
        block.mlp.c_fc = Conv1D(width, config.n_embd)
        block.mlp.c_proj = Conv1D(config.n_embd, width)

    # Load the saved compressed parameters.
    model.load_state_dict(
        checkpoint["state_dict"],
        strict=True
    )

    model.eval()

    return model, model_name


def main():
    print("Loading compressed transformer...")

    model, model_name = load_compressed_model()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    text = "Machine learning models can be compressed"

    inputs = tokenizer(text, return_tensors="pt")

    with torch.inference_mode():
        outputs = model(**inputs)

    assert torch.isfinite(outputs.logits).all()

    parameters = sum(
        p.numel() for p in model.parameters()
    )

    memory_mib = sum(
        p.numel() * p.element_size()
        for p in model.parameters()
    ) / (1024 ** 2)

    summary = json.loads(SUMMARY.read_text())

    assert parameters == summary["compressed_parameters"]

    print("\n--- Verification Results ---")
    print("Checkpoint loaded successfully!")
    print(f"Parameters: {parameters:,}")
    print(f"Parameter memory: {memory_mib:.2f} MiB")
    print(f"Output shape: {tuple(outputs.logits.shape)}")
    print(
        f"Memory reduction: "
        f"{summary['memory_reduction_percent']:.2f}%"
    )
    print("Inference verification passed!")


if __name__ == "__main__":
    main()
