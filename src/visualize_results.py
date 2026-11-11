
import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

FIGURES.mkdir(exist_ok=True)


def load_result(filename):
    path = RESULTS / filename

    if not path.exists():
        print(f"Missing result: {filename}")
        return None

    return json.loads(path.read_text())


def save_chart(filename):
    plt.tight_layout()
    plt.savefig(
        FIGURES / filename,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close()
    print(f"Generated figures/{filename}")


def plot_perplexity():
    validation = load_result("validation_metrics.json")

    full = load_result(
        "quantized_validation_all_layers.json"
    )
    ffn = load_result("quantized_validation.json")

    if not all([validation, full, ffn]):
        print("Skipping perplexity comparison.")
        return

    labels = [
        "Original FP32",
        "Pruned FP32",
        "Full INT8",
        "FFN-only INT8",
    ]

    values = [
        validation["baseline"]["perplexity"],
        validation["compressed"]["perplexity"],
        full["int8_pruned"]["perplexity"],
        ffn["int8_pruned"]["perplexity"],
    ]

    plt.figure(figsize=(9, 5))
    bars = plt.bar(
        labels,
        values,
        color=["#7895B2", "#7895B2", "#B8A5D6", "#B8A5D6"],
    )

    plt.title("Transformer Compression vs. Model Quality")
    plt.ylabel("WikiText-2 Perplexity (Lower is Better)")
    plt.ylim(0, max(values) * 1.2)

    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.8,
            f"{value:.2f}",
            ha="center",
        )

    save_chart("perplexity_comparison.png")


def plot_checkpoint_size():
    full = load_result("quantization_all_layers.json")
    ffn = load_result("quantization.json")

    if not all([full, ffn]):
        print("Skipping checkpoint comparison.")
        return

    mib = 1024 ** 2

    labels = [
        "Pruned FP32",
        "Full INT8",
        "FFN-only INT8",
    ]

    values = [
        full["original_checkpoint_bytes"] / mib,
        full["quantized_checkpoint_bytes"] / mib,
        ffn["quantized_checkpoint_bytes"] / mib,
    ]

    plt.figure(figsize=(8, 5))

    bars = plt.bar(
        labels,
        values,
        color=["#7895B2", "#B8A5D6", "#B8A5D6"],
    )

    plt.title("Quantization Checkpoint Size Comparison")
    plt.ylabel("Serialized Checkpoint Size (MiB)")
    plt.ylim(0, max(values) * 1.2)

    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 3,
            f"{value:.1f} MiB",
            ha="center",
        )

    save_chart("checkpoint_comparison.png")


def plot_runtime():
    data = load_result("int8_runtime_benchmark.json")

    if not data:
        print("Skipping runtime comparison.")
        return

    labels = ["Pruned FP32", "Pruned INT8"]

    memory = [
        data["fp32"]["sampled_peak_rss_mib"],
        data["int8"]["sampled_peak_rss_mib"],
    ]

    latency = [
        data["fp32"]["median_latency_ms"],
        data["int8"]["median_latency_ms"],
    ]

    for filename, values, ylabel, title in [
        (
            "runtime_memory.png",
            memory,
            "Sampled Peak Process RSS (MiB)",
            "INT8 vs. FP32 CPU Memory",
        ),
        (
            "runtime_latency.png",
            latency,
            "Median Latency (ms)",
            "INT8 vs. FP32 CPU Latency",
        ),
    ]:
        plt.figure(figsize=(7, 5))

        bars = plt.bar(
            labels,
            values,
            color=["#7895B2", "#B8A5D6"],
        )

        plt.title(title)
        plt.ylabel(ylabel)
        plt.ylim(0, max(values) * 1.2)

        for bar, value in zip(bars, values):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) * 0.02,
                f"{value:.2f}",
                ha="center",
            )

        save_chart(filename)


def main():
    print("Generating transformer compression charts...")

    plot_perplexity()
    plot_checkpoint_size()
    plot_runtime()

    print("\nVisualization completed!")


if __name__ == "__main__":
    main()
