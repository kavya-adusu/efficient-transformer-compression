
import json
from pathlib import Path

import mlflow


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/memory_benchmark.json"
DATABASE = ROOT / "mlflow.db"

EXPERIMENT = "Efficient Transformer Compression"


def main():
    if not RESULTS.exists():
        raise FileNotFoundError(
            "Run python src/memory_benchmark.py first."
        )

    data = json.loads(RESULTS.read_text())

    # Connect to the existing SQLite database.
    mlflow.set_tracking_uri(
        f"sqlite:///{DATABASE}"
    )

    mlflow.set_experiment(EXPERIMENT)

    print("Logging CPU memory benchmarks...")

    for variant in ["original", "pruned"]:
        metrics = data[variant]

        with mlflow.start_run(
            run_name=f"Memory Benchmark - {variant}"
        ):
            # Record experimental configuration.
            mlflow.log_param("model", "distilgpt2")
            mlflow.log_param("variant", variant)
            mlflow.log_param("device", "cpu")
            mlflow.log_param(
                "sequence_length",
                data["comparison"]["sequence_length"]
            )
            mlflow.log_param(
                "benchmark_runs",
                data["comparison"]["benchmark_runs"]
            )

            # Record actual measured performance.
            for name in [
                "parameters",
                "parameter_memory_mib",
                "rss_after_load_mib",
                "sampled_peak_rss_mib",
                "sampled_inference_rss_increase_mib",
                "median_latency_ms",
            ]:
                mlflow.log_metric(
                    name,
                    float(metrics[name])
                )

            if variant == "pruned":
                mlflow.log_metric(
                    "sampled_peak_rss_reduction_pct",
                    data["comparison"][
                        "sampled_peak_rss_reduction_percent"
                    ]
                )

            mlflow.log_artifact(str(RESULTS))

        print(f"Logged {variant} memory benchmark.")

    print("\n--- Memory Comparison ---")

    original = data["original"]
    pruned = data["pruned"]

    print(
        f"Original peak RSS: "
        f"{original['sampled_peak_rss_mib']:.2f} MiB"
    )
    print(
        f"Pruned peak RSS: "
        f"{pruned['sampled_peak_rss_mib']:.2f} MiB"
    )
    print(
        f"Memory reduction: "
        f"{data['comparison']['sampled_peak_rss_reduction_percent']:.2f}%"
    )

    print("\nMemory metrics logged to MLflow!")


if __name__ == "__main__":
    main()
