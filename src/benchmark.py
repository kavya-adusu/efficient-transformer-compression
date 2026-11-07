
import gc
import json
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluate import load_optimized_model, memory_mib

MODEL_NAME = "distilgpt2"
WARMUP_RUNS = 5
BENCHMARK_RUNS = 20


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def synchronize(device):
    """Wait until device operations finish."""
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


@torch.inference_mode()
def benchmark_model(model, inputs, device):
    """Measure forward-pass inference latency."""
    model = model.to(device)
    model.eval()

    # Warm up the model before measuring.
    for _ in range(WARMUP_RUNS):
        model(input_ids=inputs, use_cache=False)

    synchronize(device)

    times = []

    for _ in range(BENCHMARK_RUNS):
        synchronize(device)

        start = time.perf_counter()

        model(input_ids=inputs, use_cache=False)

        synchronize(device)

        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)

    return {
        "mean_latency_ms": statistics.mean(times),
        "median_latency_ms": statistics.median(times),
        "parameter_memory_mib": memory_mib(model),
    }


def main():
    device = get_device()
    print(f"Benchmarking on: {device}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    text = (
        "Transformer compression reduces the resources "
        "needed for language model inference. "
    ) * 100

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
    )["input_ids"].to(device)

    print(f"Sequence length: {inputs.shape[1]}")

    print("\nBenchmarking original model...")

    original = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME
    )

    original_results = benchmark_model(
        original, inputs, device
    )

    del original
    gc.collect()

    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()

    print("Benchmarking compressed model...")

    compressed = load_optimized_model()

    compressed_results = benchmark_model(
        compressed, inputs, device
    )

    memory_reduction = (
        1 -
        compressed_results["parameter_memory_mib"]
        / original_results["parameter_memory_mib"]
    ) * 100

    latency_change = (
        compressed_results["mean_latency_ms"]
        / original_results["mean_latency_ms"] - 1
    ) * 100

    results = {
        "device": device,
        "sequence_length": int(inputs.shape[1]),
        "warmup_runs": WARMUP_RUNS,
        "benchmark_runs": BENCHMARK_RUNS,
        "original": original_results,
        "compressed": compressed_results,
        "memory_reduction_percent": memory_reduction,
        "latency_change_percent": latency_change,
    }

    print("\n--- Inference Benchmark ---")

    print(
        f"Original latency: "
        f"{original_results['mean_latency_ms']:.2f} ms"
    )
    print(
        f"Compressed latency: "
        f"{compressed_results['mean_latency_ms']:.2f} ms"
    )
    print(f"Latency change: {latency_change:+.2f}%")
    print(f"Weight memory reduction: {memory_reduction:.2f}%")

    Path("results").mkdir(exist_ok=True)

    with open("results/benchmark.json", "w") as file:
        json.dump(results, file, indent=2)

    print("\nBenchmark completed successfully!")


if __name__ == "__main__":
    main()
