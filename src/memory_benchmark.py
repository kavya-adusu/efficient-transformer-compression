
import argparse
import json
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil
import torch
from transformers import AutoModelForCausalLM

from evaluate import load_optimized_model

MODEL_NAME = "distilgpt2"
ROOT = Path(__file__).resolve().parents[1]

SEQUENCE_LENGTH = 128
WARMUP_RUNS = 5
BENCHMARK_RUNS = 20
MIB = 1024 ** 2


def measure_model(variant):
    # Keep CPU thread count consistent.
    torch.set_num_threads(2)
    torch.manual_seed(42)

    print(f"Loading {variant} model...", file=sys.stderr)

    if variant == "original":
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME
        )
    else:
        model = load_optimized_model()

    model = model.cpu().eval()

    parameter_count = sum(
        p.numel() for p in model.parameters()
    )

    weight_memory = sum(
        p.numel() * p.element_size()
        for p in model.parameters()
    ) / MIB

    inputs = torch.randint(
        0,
        model.config.vocab_size,
        (1, SEQUENCE_LENGTH),
        dtype=torch.long,
    )

    process = psutil.Process()
    rss_after_load = process.memory_info().rss

    # Sample process memory during forward passes.
    peak_rss = [rss_after_load]
    stop_event = threading.Event()

    def sample_memory():
        while not stop_event.wait(0.005):
            current = process.memory_info().rss
            peak_rss[0] = max(peak_rss[0], current)

    monitor = threading.Thread(
        target=sample_memory,
        daemon=True,
    )
    monitor.start()

    latencies = []

    try:
        with torch.inference_mode():
            for _ in range(WARMUP_RUNS):
                model(
                    input_ids=inputs,
                    use_cache=False,
                )

            for _ in range(BENCHMARK_RUNS):
                start = time.perf_counter()

                model(
                    input_ids=inputs,
                    use_cache=False,
                )

                latencies.append(
                    (time.perf_counter() - start) * 1000
                )

    finally:
        stop_event.set()
        monitor.join()

    peak_rss[0] = max(
        peak_rss[0],
        process.memory_info().rss,
    )

    return {
        "variant": variant,
        "device": "cpu",
        "parameters": parameter_count,
        "parameter_memory_mib": weight_memory,
        "rss_after_load_mib": rss_after_load / MIB,
        "sampled_peak_rss_mib": peak_rss[0] / MIB,
        "sampled_inference_rss_increase_mib": (
            peak_rss[0] - rss_after_load
        ) / MIB,
        "median_latency_ms": statistics.median(
            latencies
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--worker",
        choices=["original", "pruned"],
    )
    args = parser.parse_args()

    if args.worker:
        result = measure_model(args.worker)
        print("RESULT_JSON:" + json.dumps(result))
        return

    results = {}

    # Each model runs in a fresh Python process.
    for variant in ("original", "pruned"):
        print(f"\nBenchmarking {variant}...")

        completed = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                variant,
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        if completed.returncode != 0:
            print(completed.stderr)
            raise RuntimeError(
                f"{variant} benchmark failed."
            )

        result_line = next(
            line for line in completed.stdout.splitlines()
            if line.startswith("RESULT_JSON:")
        )

        results[variant] = json.loads(
            result_line.removeprefix("RESULT_JSON:")
        )

        print(
            f"Sampled peak RSS: "
            f"{results[variant]['sampled_peak_rss_mib']:.2f} MiB"
        )

        print(
            f"Median latency: "
            f"{results[variant]['median_latency_ms']:.2f} ms"
        )

    original = results["original"]
    pruned = results["pruned"]

    peak_reduction = (
        1
        - pruned["sampled_peak_rss_mib"]
        / original["sampled_peak_rss_mib"]
    ) * 100

    results["comparison"] = {
        "sampled_peak_rss_reduction_percent": peak_reduction,
        "sequence_length": SEQUENCE_LENGTH,
        "warmup_runs": WARMUP_RUNS,
        "benchmark_runs": BENCHMARK_RUNS,
    }

    output = ROOT / "results" / "memory_benchmark.json"
    output.parent.mkdir(exist_ok=True)

    output.write_text(
        json.dumps(results, indent=2)
    )

    print("\n--- Memory Benchmark Results ---")
    print(
        f"Original peak RSS: "
        f"{original['sampled_peak_rss_mib']:.2f} MiB"
    )
    print(
        f"Pruned peak RSS: "
        f"{pruned['sampled_peak_rss_mib']:.2f} MiB"
    )
    print(f"Peak RSS change: {-peak_reduction:+.2f}%")
    print(f"Results saved to {output}")


if __name__ == "__main__":
    main()
