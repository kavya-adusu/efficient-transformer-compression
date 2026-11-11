
import argparse
import gc
import json
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil
import torch
from torch.ao.quantization import quantize_dynamic

from evaluate import load_optimized_model
from quantization import convert_conv1d


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "checkpoints/quantized_int8.pt"
OUTPUT = ROOT / "results/int8_runtime_benchmark.json"

SEQUENCE_LENGTH = 128
WARMUP_RUNS = 5
BENCHMARK_RUNS = 20
MIB = 1024 ** 2


def load_model(variant):
    """Load the pruned FP32 or saved INT8 model."""

    model = load_optimized_model().cpu().eval()

    if variant == "int8":
        if "qnnpack" not in torch.backends.quantized.supported_engines:
            raise RuntimeError("QNNPACK is unavailable.")

        torch.backends.quantized.engine = "qnnpack"

        checkpoint = torch.load(
            CHECKPOINT,
            map_location="cpu",
            weights_only=True,
        )

        # Reconstruct the quantized architecture.
        convert_conv1d(model.transformer)

        target_modules = set(
            checkpoint["quantized_modules"]
        )

        model = quantize_dynamic(
            model,
            qconfig_spec=target_modules,
            dtype=torch.qint8,
            inplace=True,
        )

        model.load_state_dict(
            checkpoint["state_dict"],
            strict=True,
        )

        print(
            f"Loaded {len(target_modules)} INT8 projections.",
            file=sys.stderr,
        )

        del checkpoint
        gc.collect()

    return model.eval()


def run_worker(variant):
    """Measure one model in an isolated process."""

    torch.set_num_threads(2)
    torch.manual_seed(42)

    model = load_model(variant)

    inputs = torch.randint(
        0,
        model.config.vocab_size,
        (1, SEQUENCE_LENGTH),
        dtype=torch.long,
    )

    process = psutil.Process()
    rss_after_load = process.memory_info().rss

    peak_rss = [rss_after_load]
    stop_event = threading.Event()

    def monitor_memory():
        while not stop_event.is_set():
            memory = process.memory_info().rss
            peak_rss[0] = max(peak_rss[0], memory)
            stop_event.wait(0.005)

    monitor = threading.Thread(
        target=monitor_memory,
        daemon=True,
    )

    monitor.start()
    latencies = []

    try:
        with torch.inference_mode():

            # Warm up inference.
            for _ in range(WARMUP_RUNS):
                model(
                    input_ids=inputs,
                    use_cache=False,
                )

            # Measure inference latency.
            for _ in range(BENCHMARK_RUNS):
                start = time.perf_counter()

                outputs = model(
                    input_ids=inputs,
                    use_cache=False,
                )

                elapsed_ms = (
                    time.perf_counter() - start
                ) * 1000

                latencies.append(elapsed_ms)

                assert torch.isfinite(
                    outputs.logits
                ).all().item()

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
        "sequence_length": SEQUENCE_LENGTH,
        "rss_after_load_mib": rss_after_load / MIB,
        "sampled_peak_rss_mib": peak_rss[0] / MIB,
        "median_latency_ms": statistics.median(
            latencies
        ),
        "mean_latency_ms": statistics.mean(
            latencies
        ),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--worker",
        choices=["fp32", "int8"],
    )

    args = parser.parse_args()

    if args.worker:
        result = run_worker(args.worker)
        print("RESULT_JSON:" + json.dumps(result))
        return

    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            "Missing INT8 checkpoint. "
            "Run python src/quantization.py first."
        )

    results = {}

    # Fresh process for each model.
    for variant in ["fp32", "int8"]:
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
        )

        if completed.returncode != 0:
            print(completed.stderr)
            raise RuntimeError(
                f"{variant} benchmark failed."
            )

        result_lines = [
            line
            for line in completed.stdout.splitlines()
            if line.startswith("RESULT_JSON:")
        ]

        if not result_lines:
            raise RuntimeError(
                f"No results returned for {variant}."
            )

        results[variant] = json.loads(
            result_lines[-1].removeprefix("RESULT_JSON:")
        )

    fp32 = results["fp32"]
    int8 = results["int8"]

    memory_reduction = (
        1 - int8["sampled_peak_rss_mib"]
        / fp32["sampled_peak_rss_mib"]
    ) * 100

    latency_change = (
        int8["median_latency_ms"]
        / fp32["median_latency_ms"]
        - 1
    ) * 100

    results["comparison"] = {
        "sampled_peak_rss_reduction_percent": memory_reduction,
        "median_latency_change_percent": latency_change,
        "warmup_runs": WARMUP_RUNS,
        "benchmark_runs": BENCHMARK_RUNS,
    }

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT.write_text(
        json.dumps(results, indent=2)
    )

    print("\n--- INT8 Runtime Benchmark ---")

    print(
        f"Pruned FP32 peak RSS: "
        f"{fp32['sampled_peak_rss_mib']:.2f} MiB"
    )

    print(
        f"Pruned INT8 peak RSS: "
        f"{int8['sampled_peak_rss_mib']:.2f} MiB"
    )

    print(
        f"Peak RSS reduction: {memory_reduction:.2f}%"
    )

    print(
        f"FP32 median latency: "
        f"{fp32['median_latency_ms']:.2f} ms"
    )

    print(
        f"INT8 median latency: "
        f"{int8['median_latency_ms']:.2f} ms"
    )

    print(
        f"Latency change: {latency_change:+.2f}%"
    )

    print("\nBenchmark completed successfully!")


if __name__ == "__main__":
    main()
