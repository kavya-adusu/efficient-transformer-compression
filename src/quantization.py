
import json
import tempfile
from pathlib import Path

import torch
from torch import nn
from torch.ao.quantization import quantize_dynamic
from transformers import AutoTokenizer
from transformers.pytorch_utils import Conv1D

from evaluate import load_optimized_model


MODEL_NAME = "distilgpt2"

SOURCE = Path("checkpoints/optimized_model.pt")
OUTPUT = Path("checkpoints/quantized_int8.pt")
RESULTS = Path("results/quantization.json")


def convert_conv1d(module):
    """
    Convert GPT-2 Conv1D projections into nn.Linear layers.

    GPT-2 Conv1D weights have shape [input, output],
    while nn.Linear weights have shape [output, input].
    """

    for name, child in list(module.named_children()):

        if isinstance(child, Conv1D):

            in_features, out_features = child.weight.shape

            linear = nn.Linear(
                in_features,
                out_features,
                device=child.weight.device,
                dtype=child.weight.dtype,
            )

            with torch.no_grad():
                linear.weight.copy_(child.weight.T)
                linear.bias.copy_(child.bias)

            setattr(module, name, linear)

        else:
            convert_conv1d(child)


def main():

    # -----------------------------------------
    # 1. Configure INT8 quantization
    # -----------------------------------------

    available = torch.backends.quantized.supported_engines

    if "qnnpack" not in available:
        raise RuntimeError(
            f"QNNPACK unavailable. Supported: {available}"
        )

    torch.backends.quantized.engine = "qnnpack"

    print(
        "Quantization engine:",
        torch.backends.quantized.engine
    )

    if not SOURCE.exists():
        raise FileNotFoundError(
            "Run python src/global_optimize.py first."
        )

    # -----------------------------------------
    # 2. Load compressed FP32 transformer
    # -----------------------------------------

    print("\nLoading optimized transformer...")

    model = load_optimized_model()
    model = model.cpu().eval()

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    inputs = tokenizer(
        "Transformer compression improves model efficiency.",
        return_tensors="pt",
    )

    # -----------------------------------------
    # 3. Record original FP32 model outputs
    # -----------------------------------------

    with torch.inference_mode():
        original_logits = model(
            **inputs,
            use_cache=False
        ).logits

    checkpoint = torch.load(
        SOURCE,
        map_location="cpu",
        weights_only=True,
    )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    RESULTS.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # -----------------------------------------
    # 4. Measure original checkpoint size
    # -----------------------------------------

    with tempfile.TemporaryDirectory() as tmp:

        fp32_path = Path(tmp) / "fp32.pt"

        torch.save(
            {
                "base_model": MODEL_NAME,
                "layer_widths": checkpoint["layer_widths"],
                "state_dict": model.state_dict(),
            },
            fp32_path,
        )

        original_bytes = fp32_path.stat().st_size

    # -----------------------------------------
    # 5. Convert Conv1D to Linear
    # -----------------------------------------

    print("\nConverting Conv1D layers...")

    convert_conv1d(model.transformer)

    model.eval()

    # Test the converted model BEFORE quantization.
    with torch.inference_mode():
        converted_logits = model(
            **inputs,
            use_cache=False
        ).logits

    conversion_difference = (
        original_logits - converted_logits
    ).abs().mean().item()

    print(
        f"FP32 conversion difference: "
        f"{conversion_difference:.6f}"
    )

    # -----------------------------------------
    # 6. Select layers for INT8 quantization
    # -----------------------------------------

    target_modules = {
        name
        for name, module in model.named_modules()
        if name.startswith("transformer.")
        and isinstance(module, nn.Linear)
    }

    print(
        f"\nQuantizing {len(target_modules)} projections..."
    )

    # -----------------------------------------
    # 7. Perform dynamic INT8 quantization
    # -----------------------------------------

    quantized = quantize_dynamic(
        model,
        qconfig_spec=target_modules,
        dtype=torch.qint8,
        inplace=True,
    )

    quantized.eval()

    # -----------------------------------------
    # 8. Evaluate INT8 model outputs
    # -----------------------------------------

    with torch.inference_mode():
        quantized_logits = quantized(
            **inputs,
            use_cache=False
        ).logits

    # -----------------------------------------
    # 9. Accuracy diagnostics
    # -----------------------------------------

    quantization_difference = (
        converted_logits - quantized_logits
    ).abs().mean().item()

    mean_difference = (
        original_logits - quantized_logits
    ).abs().mean().item()

    print("\n--- Accuracy Diagnostics ---")

    print(
        f"FP32 conversion difference: "
        f"{conversion_difference:.6f}"
    )

    print(
        f"INT8-only difference: "
        f"{quantization_difference:.6f}"
    )

    print(
        f"Total logit difference: "
        f"{mean_difference:.6f}"
    )

    if conversion_difference > 1e-3:
        print(
            "WARNING: Conv1D conversion changed model outputs."
        )

    if quantization_difference > 1.0:
        print(
            "WARNING: Large INT8 output difference detected."
        )

    # -----------------------------------------
    # 10. Save quantized checkpoint
    # -----------------------------------------

    torch.save(
        {
            "base_model": MODEL_NAME,
            "layer_widths": checkpoint["layer_widths"],
            "quantized_modules": sorted(target_modules),
            "state_dict": quantized.state_dict(),
        },
        OUTPUT,
    )

    # -----------------------------------------
    # 11. Calculate checkpoint size reduction
    # -----------------------------------------

    quantized_bytes = OUTPUT.stat().st_size

    reduction = (
        1 - quantized_bytes / original_bytes
    ) * 100

    # -----------------------------------------
    # 12. Save experimental results
    # -----------------------------------------

    results = {
        "method": "dynamic_int8_cpu",
        "quantization_engine": torch.backends.quantized.engine,
        "original_checkpoint_bytes": original_bytes,
        "quantized_checkpoint_bytes": quantized_bytes,
        "checkpoint_size_reduction_percent": reduction,
        "quantized_projection_count": len(target_modules),
        "fp32_conversion_difference": conversion_difference,
        "int8_only_difference": quantization_difference,
        "mean_absolute_logit_difference": mean_difference,
    }

    RESULTS.write_text(
        json.dumps(results, indent=2)
    )

    # -----------------------------------------
    # 13. Display final results
    # -----------------------------------------

    print("\n--- INT8 Quantization Results ---")

    print(
        f"FP32 checkpoint: "
        f"{original_bytes / 2**20:.2f} MiB"
    )

    print(
        f"INT8 checkpoint: "
        f"{quantized_bytes / 2**20:.2f} MiB"
    )

    print(
        f"Size reduction: {reduction:.2f}%"
    )

    print(
        f"Mean logit difference: "
        f"{mean_difference:.6f}"
    )

    print("\nQuantization experiment completed!")


if __name__ == "__main__":
    main()
