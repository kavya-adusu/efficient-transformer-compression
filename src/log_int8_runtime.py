
import json
from pathlib import Path

import mlflow


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/int8_runtime_benchmark.json"
DATABASE = ROOT / "mlflow.db"

EXPERIMENT = "Efficient Transformer Compression"


def main():
    if not RESULTS.exists():
        raise FileNotFoundError(
            "Run python src/benchmark_int8_runtime.py first."
        )

    data = json.loads(RESULTS.read_text())

    mlflow.set_tracking_uri(f"sqlite:///{DATABASE}")
    mlflow.set_experiment(EXPERIMENT)

    print("Logging INT8 runtime benchmarks...")

    variants = {
        "fp32": "Pruned FP32 Runtime",
        "int8": "Pruned INT8 Runtime",
    }

    for variant, run_name in variants.items():
        metrics = data[variant]

        with mlflow.start_run(run_name=run_name):
            mlflow.log_params({
                "model": "distilgpt2",
                "variant": variant,
                "device": "cpu",
                "sequence_length": metrics["sequence_length"],
                "warmup_runs": data["comparison"]["warmup_runs"],
                "benchmark_runs": data["comparison"]["benchmark_runs"],
            })

            mlflow.log_metrics({
                "sampled_peak_rss_mib":
                    metrics["sampled_peak_rss_mib"],
                "rss_after_load_mib":
                    metrics["rss_after_load_mib"],
                "median_latency_ms":
                    metrics["median_latency_ms"],
                "mean_latency_ms":
                    metrics["mean_latency_ms"],
            })

            if variant == "int8":
                mlflow.log_metrics({
                    "peak_rss_reduction_percent":
                        data["comparison"][
                            "sampled_peak_rss_reduction_percent"
                        ],
                    "latency_change_percent":
                        data["comparison"][
                            "median_latency_change_percent"
                        ],
                })

            mlflow.log_artifact(str(RESULTS))

        print(f"Logged: {run_name}")

    print("\n--- Runtime Comparison ---")
    print(
        f"FP32 peak RSS: "
        f"{data['fp32']['sampled_peak_rss_mib']:.2f} MiB"
    )
    print(
        f"INT8 peak RSS: "
        f"{data['int8']['sampled_peak_rss_mib']:.2f} MiB"
    )
    print(
        f"Memory reduction: "
        f"{data['comparison']['sampled_peak_rss_reduction_percent']:.2f}%"
    )
    print(
        f"Latency change: "
        f"{data['comparison']['median_latency_change_percent']:+.2f}%"
    )

    print("\nINT8 runtime metrics logged successfully!")


if __name__ == "__main__":
    main()
