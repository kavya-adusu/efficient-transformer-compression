
import json
from pathlib import Path

import mlflow

RESULTS = Path("results")
TRACKING_DIR = Path("mlruns")


def load_json(filename):
    path = RESULTS / filename

    if not path.exists():
        print(f"Skipping missing file: {path}")
        return None

    return json.loads(path.read_text())


def log_experiment(name, method, metrics, source_files):
    """Save an experiment and its metrics to MLflow."""

    with mlflow.start_run(run_name=name):
        mlflow.log_param("model", "distilgpt2")
        mlflow.log_param("compression_method", method)
        mlflow.log_param("evaluation_dataset", "WikiText-2")
        mlflow.log_param("validation_tokens", 4096)

        for key, value in metrics.items():
            if value is not None:
                mlflow.log_metric(key, float(value))

        for filename in source_files:
            path = RESULTS / filename
            if path.exists():
                mlflow.log_artifact(str(path))

        print(f"Logged experiment: {name}")


def main():
    
    # Use SQLite instead of the legacy file backend.
    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    print("MLflow tracking backend: SQLite")

    mlflow.set_experiment(
        "Efficient Transformer Compression"
    )

    validation = load_json("validation_metrics.json")
    benchmark = load_json("benchmark.json")

    full_quant = load_json(
        "quantization_all_layers.json"
    )
    full_validation = load_json(
        "quantized_validation_all_layers.json"
    )

    ffn_quant = load_json("quantization.json")
    ffn_validation = load_json(
        "quantized_validation.json"
    )

    # Experiment 1: Original FP32 model
    if validation:
        metrics = {
            "perplexity": validation["baseline"]["perplexity"],
            "validation_loss": validation["baseline"]["loss"],
            "parameter_memory_mib": validation[
                "baseline"
            ]["parameter_memory_mib"],
        }

        if benchmark:
            metrics["mean_latency_ms"] = benchmark[
                "original"
            ]["mean_latency_ms"]

        log_experiment(
            "Original FP32",
            "none",
            metrics,
            ["validation_metrics.json"],
        )

        # Experiment 2: Structured pruning
        metrics = {
            "perplexity": validation[
                "compressed"
            ]["perplexity"],
            "validation_loss": validation[
                "compressed"
            ]["loss"],
            "parameter_memory_mib": validation[
                "compressed"
            ]["parameter_memory_mib"],
            "parameter_memory_reduction_pct": validation[
                "memory_reduction_percent"
            ],
        }

        if benchmark:
            metrics["mean_latency_ms"] = benchmark[
                "compressed"
            ]["mean_latency_ms"]

        log_experiment(
            "Adaptive Structured Pruning",
            "structured_pruning",
            metrics,
            ["validation_metrics.json", "benchmark.json"],
        )

    # Experiment 3: All-projection INT8
    if full_quant and full_validation:
        log_experiment(
            "All Projection INT8",
            "structured_pruning_plus_full_int8",
            {
                "perplexity": full_validation[
                    "int8_pruned"
                ]["perplexity"],
                "validation_loss": full_validation[
                    "int8_pruned"
                ]["loss"],
                "checkpoint_reduction_vs_pruned_fp32_pct":
                    full_quant[
                        "checkpoint_size_reduction_percent"
                    ],
                "checkpoint_size_mib":
                    full_quant[
                        "quantized_checkpoint_bytes"
                    ] / (1024 ** 2),
            },
            [
                "quantization_all_layers.json",
                "quantized_validation_all_layers.json",
            ],
        )

    # Experiment 4: FFN-only INT8
    if ffn_quant and ffn_validation:
        log_experiment(
            "FFN Only INT8",
            "structured_pruning_plus_ffn_int8",
            {
                "perplexity": ffn_validation[
                    "int8_pruned"
                ]["perplexity"],
                "validation_loss": ffn_validation[
                    "int8_pruned"
                ]["loss"],
                "checkpoint_reduction_vs_pruned_fp32_pct":
                    ffn_quant[
                        "checkpoint_size_reduction_percent"
                    ],
                "checkpoint_size_mib":
                    ffn_quant[
                        "quantized_checkpoint_bytes"
                    ] / (1024 ** 2),
            },
            [
                "quantization.json",
                "quantized_validation.json",
            ],
        )

    print("\nMLflow experiments recorded successfully!")


if __name__ == "__main__":
    main()
