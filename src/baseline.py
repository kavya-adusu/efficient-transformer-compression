
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "distilgpt2"


def calculate_memory(model):
    """Calculate model parameter memory in MiB."""
    total_bytes = sum(
        p.numel() * p.element_size()
        for p in model.parameters()
    )
    return total_bytes / (1024 ** 2)


def main():
    print("Loading baseline transformer...")

    # Select available hardware
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # Load pretrained model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    model = model.to(device)
    model.eval()

    # Count parameters
    total_params = sum(
        p.numel() for p in model.parameters()
    )

    # Measure parameter memory
    memory_mb = calculate_memory(model)

    print(f"Model: {MODEL_NAME}")
    print(f"Device: {device}")
    print(f"Total parameters: {total_params:,}")
    print(f"Parameter memory: {memory_mb:.2f} MiB")

    # Test inference
    text = "Machine learning models can be compressed"

    inputs = tokenizer(
        text, return_tensors="pt"
    ).to(device)

    with torch.inference_mode():
        outputs = model(**inputs)

    print(f"Output shape: {outputs.logits.shape}")
    print("Baseline inference successful!")


if __name__ == "__main__":
    main()
