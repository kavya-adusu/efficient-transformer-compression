
import json
import os
import time
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import mlflow
import ray
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

MODEL_NAME = "distilgpt2"
MAX_TOKENS = 2048
WORKERS = 2

# Three compression configurations to compare.
EXPERIMENTS = {
    "uniform_10": [0.10] * 6,
    "uniform_20": [0.20] * 6,
    "layer_adaptive": [0.10, 0.30, 0.10, 0.30, 0.10, 0.20],
}


@ray.remote(num_cpus=1, max_retries=0)
def run_experiment(name, rates, token_ids):
    """Run one pruning experiment in a Ray worker."""

    import math
    import torch
    from transformers import AutoModelForCausalLM
    from transformers.pytorch_utils import Conv1D

    torch.set_num_threads(1)
    started = time.perf_counter()

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME
    ).cpu().eval()

    tokens = torch.tensor(
        [token_ids], dtype=torch.long
    )

    def count_params():
        return sum(p.numel() for p in model.parameters())

    @torch.inference_mode()
    def evaluate():
        total_nll = 0.0
        total_count = 0
        previous_end = 0

        context = 256
        stride = 128
        length = tokens.shape[1]

        for end in range(stride, length + stride, stride):
            end = min(end, length)
            begin = max(0, end - context)

            ids = tokens[:, begin:end]
            labels = ids.clone()

            new_tokens = end - previous_end
            labels[:, :-new_tokens] = -100

            outputs = model(
                input_ids=ids,
                labels=labels,
                use_cache=False,
            )

            count = int(
                (labels[:, 1:] != -100).sum().item()
            )

            if count:
                total_nll += outputs.loss.item() * count
                total_count += count

            previous_end = end

            if end == length:
                break

        loss = total_nll / total_count
        return loss, math.exp(loss)

    original_params = count_params()
    baseline_loss, baseline_ppl = evaluate()

    # Physically shrink the FFN in each transformer block.
    for block, rate in zip(model.transformer.h, rates):
        if rate <= 0:
            continue

        fc = block.mlp.c_fc
        proj = block.mlp.c_proj

        weights = fc.weight.detach()
        width = weights.shape[1]

        keep = max(1, round(width * (1 - rate)))

        importance = torch.linalg.vector_norm(
            weights.float(), dim=0
        )

        indices = torch.topk(
            importance, k=keep
        ).indices.sort().values

        new_fc = Conv1D(keep, weights.shape[0])

        new_proj = Conv1D(
            proj.weight.shape[1], keep
        )

        with torch.no_grad():
            new_fc.weight.copy_(
                fc.weight[:, indices]
            )
            new_fc.bias.copy_(
                fc.bias[indices]
            )
            new_proj.weight.copy_(
                proj.weight[indices, :]
            )
            new_proj.bias.copy_(
                proj.bias
            )

        block.mlp.c_fc = new_fc
        block.mlp.c_proj = new_proj

    compressed_params = count_params()
    compressed_loss, compressed_ppl = evaluate()

    return {
        "name": name,
        "rates": rates,
        "baseline_perplexity": baseline_ppl,
        "compressed_perplexity": compressed_ppl,
        "baseline_loss": baseline_loss,
        "compressed_loss": compressed_loss,
        "loss_increase": compressed_loss - baseline_loss,
        "original_parameters": original_params,
        "compressed_parameters": compressed_params,
        "parameter_reduction_pct": (
            1 - compressed_params / original_params
        ) * 100,
        "duration_seconds": time.perf_counter() - started,
    }


def main():
    print("Loading WikiText-2 validation text...")

    dataset = load_dataset(
        "Salesforce/wikitext",
        "wikitext-2-raw-v1",
        split="validation",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    # Collect a reproducible, limited evaluation sample.
    tokens = []

    for row in dataset:
        if row["text"].strip():
            tokens.extend(
                tokenizer.encode(
                    row["text"] + "\n",
                    add_special_tokens=False,
                )
            )

        if len(tokens) >= MAX_TOKENS:
            break

    tokens = tokens[:MAX_TOKENS]

    print(f"Evaluation tokens: {len(tokens)}")
    print(f"Starting Ray with {WORKERS} CPU workers...")

    ray.init(
        num_cpus=WORKERS,
        include_dashboard=False,
        log_to_driver=False,
    )

    try:
        # Submit all experiments to Ray.
        futures = [
            run_experiment.remote(name, rates, tokens)
            for name, rates in EXPERIMENTS.items()
        ]

        # Collect results from the workers.
        results = ray.get(futures)

    finally:
        ray.shutdown()

    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    # Record all results in one summary.
    summary_path = output_dir / "ray_summary.json"
    summary_path.write_text(
        json.dumps(results, indent=2)
    )

    # Log experiment results to MLflow.
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(
        "Efficient Transformer Compression"
    )

    for result in results:
        name = result["name"]

        trial_path = output_dir / f"ray_{name}.json"
        trial_path.write_text(
            json.dumps(result, indent=2)
        )

        with mlflow.start_run(run_name=f"Ray {name}"):
            mlflow.log_param("model", MODEL_NAME)
            mlflow.log_param("method", "structured_pruning")
            mlflow.log_param("pruning_rates", str(result["rates"]))
            mlflow.log_param("validation_tokens", len(tokens))

            mlflow.log_metric(
                "perplexity",
                result["compressed_perplexity"]
            )
            mlflow.log_metric(
                "validation_loss",
                result["compressed_loss"]
            )
            mlflow.log_metric(
                "loss_increase",
                result["loss_increase"]
            )
            mlflow.log_metric(
                "parameter_reduction_pct",
                result["parameter_reduction_pct"]
            )
            mlflow.log_metric(
                "duration_seconds",
                result["duration_seconds"]
            )

            mlflow.log_artifact(str(trial_path))

        print(
            f"{name}: "
            f"PPL={result['compressed_perplexity']:.2f}, "
            f"Reduction={result['parameter_reduction_pct']:.2f}%, "
            f"Loss increase={result['loss_increase']:.4f}"
        )

    print("\nRay experiments completed!")
    print(f"Results saved to {summary_path}")
    print("Experiment metrics logged to MLflow.")


if __name__ == "__main__":
    main()
