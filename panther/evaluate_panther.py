"""Evaluate a trained PANTHER checkpoint and write CLAM-style reports."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .config_loader import load_config, resolve_evaluation_run
from .metrics import classification_metrics
from .panther_dataset import build_datasets, class_counts, load_split_manifest
from .panther_model import LinearClassifier
from .train_panther import MODEL_SCHEMA, resolve_device, seed_everything


@torch.no_grad()
def evaluate_arrays(
    model: LinearClassifier,
    representations: torch.Tensor,
    labels: torch.Tensor,
    class_names: Sequence[str],
    device: torch.device,
    batch_size: int,
) -> tuple[Dict[str, Any], np.ndarray]:
    model.eval()
    loader = DataLoader(
        TensorDataset(representations, labels), batch_size=batch_size, shuffle=False
    )
    probabilities = []
    labels_all = []
    loss_total = 0.0
    for features, target in loader:
        features = features.to(device)
        target = target.to(device)
        logits = model(features)
        loss_total += float(nn.functional.cross_entropy(logits, target)) * len(target)
        probabilities.append(torch.softmax(logits, dim=1).cpu())
        labels_all.extend(target.cpu().tolist())
    probability_array = torch.cat(probabilities).numpy()
    metrics = classification_metrics(labels_all, probability_array, class_names)
    metrics["loss"] = loss_total / len(labels_all)
    metrics["num_slides"] = len(labels_all)
    return metrics, probability_array


def save_split_results(
    split: str,
    metrics: Mapping[str, Any],
    probabilities: np.ndarray,
    cache: Mapping[str, Any],
    class_names: Sequence[str],
    output_dir: Path,
    dpi: int,
) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"slide_{split}_evaluation.json"
    predictions_csv = output_dir / f"slide_{split}_predictions.csv"
    predictions_json = output_dir / f"slide_{split}_predictions.json"
    matrix_path = output_dir / f"slide_{split}_confusion_matrix.png"

    summary = dict(metrics)
    predictions = list(summary.pop("predictions"))
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    rows = []
    labels = cache["labels"].tolist()
    for index, label in enumerate(labels):
        prediction = int(predictions[index])
        row: Dict[str, Any] = {
            "slide_name": cache["slide_names"][index],
            "slide_key": cache["slide_keys"][index],
            "num_tiles": int(cache["tile_counts"][index]),
            "true_label": int(label),
            "predicted_label": prediction,
            "true_class": class_names[int(label)],
            "predicted_class": class_names[prediction],
        }
        for class_index, class_name in enumerate(class_names):
            row[f"probability_{class_name}"] = float(
                probabilities[index, class_index]
            )
        rows.append(row)
    with predictions_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with predictions_json.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)

    matrix = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    figure, axis = plt.subplots(figsize=(10, 9))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=list(class_names),
        yticklabels=list(class_names),
        annot_kws={"size": 14},
        ax=axis,
    )
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    axis.set_title(f"PANTHER slide-level {split} confusion matrix")
    figure.tight_layout()
    figure.savefig(matrix_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return {
        "summary": str(summary_path),
        "predictions_csv": str(predictions_csv),
        "predictions_json": str(predictions_json),
        "confusion_matrix": str(matrix_path),
    }


def run_evaluation(config_path: str | None = None) -> Dict[str, Any]:
    runtime_config = load_config(config_path)
    run_dir = resolve_evaluation_run(runtime_config)
    checkpoint_path = Path(runtime_config["paths"]["checkpoint"])
    output_dir = Path(runtime_config["paths"]["evaluation_output"])
    print(f"Using checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("model_schema") != MODEL_SCHEMA:
        raise ValueError(
            f"Checkpoint schema must be '{MODEL_SCHEMA}', received "
            f"'{checkpoint.get('model_schema')}'."
        )
    checkpoint_config = checkpoint.get("config")
    class_names = checkpoint.get("class_folders")
    if not isinstance(checkpoint_config, Mapping) or not isinstance(class_names, list):
        raise ValueError("Checkpoint is missing its exact config or ordered classes.")
    manifest_path = run_dir / "split_manifest.json"
    saved_classes, assignments = load_split_manifest(manifest_path)
    if saved_classes != class_names:
        raise ValueError("Checkpoint classes disagree with the split manifest.")
    datasets, discovered_classes, _ = build_datasets(
        checkpoint_config,
        class_folders=class_names,
        split_assignments=assignments,
    )
    if discovered_classes != class_names:
        raise ValueError("Current dataset class order disagrees with the checkpoint.")

    embedding_path = run_dir / "slide_embeddings.pt"
    if not embedding_path.is_file():
        raise FileNotFoundError(
            f"Slide embedding cache is missing: {embedding_path}. Re-run training."
        )
    embeddings = torch.load(embedding_path, map_location="cpu", weights_only=False)
    for split, dataset in datasets.items():
        expected_keys = [record["slide_key"] for record in dataset.records]
        if embeddings[split]["slide_keys"] != expected_keys:
            raise ValueError(f"Embedding cache does not match the {split} manifest.")

    seed_everything(int(checkpoint_config["random_seed"]))
    device = resolve_device(runtime_config)
    input_dim = int(checkpoint["training_details"]["input_dim"])
    model = LinearClassifier(
        input_dim,
        len(class_names),
        bias=bool(checkpoint_config["training"]["classifier_bias"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    requested_splits = list(runtime_config["evaluation"]["splits"])
    if bool(runtime_config["evaluation"]["include_train"]) and "train" not in requested_splits:
        requested_splits.append("train")

    results: Dict[str, Any] = {}
    for split in requested_splits:
        split_cache = embeddings[split]
        metrics, probabilities = evaluate_arrays(
            model,
            split_cache["representations"],
            split_cache["labels"],
            class_names,
            device,
            int(checkpoint_config["training"]["batch_size"]),
        )
        artifacts = save_split_results(
            split,
            metrics,
            probabilities,
            split_cache,
            class_names,
            output_dir,
            int(runtime_config["evaluation"]["confusion_matrix_dpi"]),
        )
        printable_auc = metrics["multiclass_roc_auc"]
        auc_text = "not valid" if printable_auc is None else f"{printable_auc:.4f}"
        print(
            f"slide/{split}: n={len(datasets[split])} {class_counts(datasets[split])} "
            f"accuracy={metrics['accuracy']:.4f} "
            f"balanced_accuracy={metrics['balanced_accuracy']:.4f} "
            f"macro_f1={metrics['macro_f1']:.4f} auc={auc_text}"
        )
        compact = dict(metrics)
        compact.pop("predictions")
        results[f"slide/{split}"] = {**compact, "artifacts": artifacts}

    manifest = {
        "checkpoint": str(checkpoint_path),
        "model_schema": MODEL_SCHEMA,
        "class_folders": class_names,
        "bag_level": "slide",
        "native_panther_representation": "allcat",
        "results": results,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "evaluation_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    arguments = parser.parse_args()
    run_evaluation(arguments.config)


if __name__ == "__main__":
    main()

