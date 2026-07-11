"""Evaluate canonical CLAM checkpoints using their exact saved data contract."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader

try:
    from .clam_dataset import WSIBagDataset, collate_fn, create_bag_dataset
    from .config_loader import load_config
    from .train_clam import MODEL_SCHEMA, create_model, seed_everything, seed_worker
except ImportError:
    from clam_dataset import WSIBagDataset, collate_fn, create_bag_dataset
    from config_loader import load_config
    from train_clam import MODEL_SCHEMA, create_model, seed_everything, seed_worker


def get_class_sample_counts(dataset: WSIBagDataset) -> Dict[str, int]:
    """Count evaluated bag units per class.

    Args:
        dataset (WSIBagDataset): Tissue- or slide-level split dataset.

    Returns:
        Dict[str, int]: Counts keyed by ordered class name.
    """
    counts = {class_name: 0 for class_name in dataset.class_folders}
    for bag_index in dataset.indices:
        counts[dataset._bags[bag_index]["class_name"]] += 1
    return counts


def _multiclass_auc(
    labels: Sequence[int],
    probabilities: Sequence[Sequence[float]],
    num_classes: int,
) -> Optional[float]:
    """Compute macro one-vs-rest ROC AUC when every class is represented.

    Args:
        labels (Sequence[int]): Ground-truth class indices.
        probabilities (Sequence[Sequence[float]]): Per-class probabilities.
        num_classes (int): Fixed checkpoint class count.

    Returns:
        Optional[float]: Macro ROC AUC, or ``None`` when mathematically invalid.
    """
    if len(set(labels)) != num_classes:
        return None
    try:
        probability_array = np.asarray(probabilities, dtype=np.float64)
        if num_classes == 2:
            return float(roc_auc_score(labels, probability_array[:, 1]))
        return float(
            roc_auc_score(
                labels,
                probability_array,
                labels=list(range(num_classes)),
                multi_class="ovr",
                average="macro",
            )
        )
    except ValueError:
        return None


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    class_names: Sequence[str],
) -> Dict[str, Any]:
    """Evaluate one split with fixed-class metrics and predictions.

    Args:
        model (nn.Module): Loaded canonical CLAM model.
        dataloader (DataLoader): Evaluation bag DataLoader.
        device (torch.device): Compute device.
        class_names (Sequence[str]): Checkpoint class names in label order.

    Returns:
        Dict[str, Any]: Aggregate metrics, per-class report, fixed confusion
            matrix, probabilities, predictions, and bag identifiers.
    """
    if len(dataloader.dataset) == 0:
        raise ValueError("Cannot evaluate an empty dataset split.")
    model.eval()
    labels_all: List[int] = []
    predictions_all: List[int] = []
    probabilities_all: List[List[float]] = []
    slide_names: List[str] = []
    tissue_names: List[str] = []

    with torch.no_grad():
        for batch in dataloader:
            features = batch["features"].to(device, non_blocking=True)
            masks = batch["masks"].to(device, non_blocking=True)
            outputs = model(features, mask=masks, labels=None, instance_eval=False)
            labels_all.extend(batch["labels"].tolist())
            predictions_all.extend(outputs["predictions"].cpu().tolist())
            probabilities_all.extend(outputs["probabilities"].cpu().tolist())
            slide_names.extend(str(name) for name in batch["slide_names"])
            tissue_names.extend(str(name) for name in batch["tissue_names"])

    fixed_labels = list(range(len(class_names)))
    matrix = confusion_matrix(
        labels_all, predictions_all, labels=fixed_labels
    )
    report = classification_report(
        labels_all,
        predictions_all,
        labels=fixed_labels,
        target_names=list(class_names),
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(labels_all, predictions_all)),
        "balanced_accuracy": float(
            balanced_accuracy_score(labels_all, predictions_all)
        ),
        "macro_f1": float(
            f1_score(
                labels_all,
                predictions_all,
                labels=fixed_labels,
                average="macro",
                zero_division=0,
            )
        ),
        "multiclass_roc_auc": _multiclass_auc(
            labels_all, probabilities_all, len(class_names)
        ),
        "classification_report": report,
        "confusion_matrix": matrix,
        "labels": labels_all,
        "predictions": predictions_all,
        "probabilities": probabilities_all,
        "slide_names": slide_names,
        "tissue_names": tissue_names,
    }


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Sequence[str],
    save_path: Optional[str] = None,
) -> None:
    """Plot a fixed-label confusion matrix.

    Args:
        cm (np.ndarray): Integer matrix shaped ``[classes, classes]``.
        class_names (Sequence[str]): Ordered axis labels.
        save_path (Optional[str]): PNG path, or ``None`` to display.

    Returns:
        None: The plot is saved or displayed.
    """
    figure, axis = plt.subplots(figsize=(10, 9))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=list(class_names),
        yticklabels=list(class_names),
        cbar=False,
        annot_kws={"size": 16},
        ax=axis,
    )
    axis.tick_params(axis="both", labelsize=14)
    axis.set_xlabel("Predicted", fontsize=16)
    axis.set_ylabel("Actual", fontsize=16)
    figure.tight_layout()
    if save_path is None:
        plt.show()
    else:
        figure.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def print_metrics(metrics: Mapping[str, Any], class_names: Sequence[str]) -> None:
    """Print aggregate and per-class evaluation metrics.

    Args:
        metrics (Mapping[str, Any]): Result returned by ``evaluate``.
        class_names (Sequence[str]): Ordered checkpoint class names.

    Returns:
        None: Metrics are printed to standard output.
    """
    auc = metrics["multiclass_roc_auc"]
    auc_text = "not valid" if auc is None else f"{float(auc):.4f}"
    print(
        f"accuracy={metrics['accuracy']:.4f} "
        f"balanced_accuracy={metrics['balanced_accuracy']:.4f} "
        f"macro_f1={metrics['macro_f1']:.4f} "
        f"multiclass_roc_auc={auc_text}"
    )
    report = metrics["classification_report"]
    for class_name in class_names:
        values = report[class_name]
        print(
            f"  {class_name}: precision={values['precision']:.4f} "
            f"recall={values['recall']:.4f} f1={values['f1-score']:.4f} "
            f"support={int(values['support'])}"
        )


def create_dataset_for_level(
    level: str,
    config: Mapping[str, Any],
    class_folders: Sequence[str],
    split: str,
) -> WSIBagDataset:
    """Create an evaluation dataset from checkpoint configuration.

    Args:
        level (str): ``tissue`` or ``slide``.
        config (Mapping[str, Any]): Exact configuration stored in the checkpoint.
        class_folders (Sequence[str]): Exact checkpoint class order.
        split (str): ``train``, ``val``, or ``test``.

    Returns:
        WSIBagDataset: Configured evaluation bag dataset.
    """
    if level not in ("tissue", "slide"):
        raise ValueError("level must be 'tissue' or 'slide'.")
    return create_bag_dataset(
        config,
        split,
        class_folders=class_folders,
        bag_level=level,
    )


def _json_summary(metrics: Mapping[str, Any]) -> Dict[str, Any]:
    """Convert evaluation metrics to a JSON-safe summary.

    Args:
        metrics (Mapping[str, Any]): Complete result returned by ``evaluate``.

    Returns:
        Dict[str, Any]: Aggregate metrics, report, and fixed confusion matrix.
    """
    return {
        "accuracy": float(metrics["accuracy"]),
        "balanced_accuracy": float(metrics["balanced_accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
        "multiclass_roc_auc": (
            None
            if metrics["multiclass_roc_auc"] is None
            else float(metrics["multiclass_roc_auc"])
        ),
        "classification_report": metrics["classification_report"],
        "confusion_matrix": np.asarray(metrics["confusion_matrix"]).tolist(),
        "num_bags": len(metrics["labels"]),
    }


def save_level_results(
    metrics: Mapping[str, Any],
    class_folders: Sequence[str],
    output_dir: str,
    level: str,
    split: str,
) -> Dict[str, str]:
    """Save summary JSON, prediction CSV/JSON, and confusion plot.

    Args:
        metrics (Mapping[str, Any]): Complete result returned by ``evaluate``.
        class_folders (Sequence[str]): Ordered checkpoint class names.
        output_dir (str): Artifact directory.
        level (str): Evaluated bag level.
        split (str): Evaluated dataset split.

    Returns:
        Dict[str, str]: Paths to all saved artifacts.
    """
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    stem = f"{level}_{split}"
    summary_path = destination / f"{stem}_evaluation.json"
    predictions_json_path = destination / f"{stem}_predictions.json"
    predictions_csv_path = destination / f"{stem}_predictions.csv"
    matrix_path = destination / f"{stem}_confusion_matrix.png"

    with summary_path.open("w", encoding="utf-8") as summary_file:
        json.dump(_json_summary(metrics), summary_file, indent=2)

    rows: List[Dict[str, Any]] = []
    for index, label in enumerate(metrics["labels"]):
        prediction = int(metrics["predictions"][index])
        row: Dict[str, Any] = {
            "slide_name": metrics["slide_names"][index],
            "tissue_name": metrics["tissue_names"][index],
            "true_label": int(label),
            "predicted_label": prediction,
            "true_class": class_folders[int(label)],
            "predicted_class": class_folders[prediction],
        }
        for class_index, class_name in enumerate(class_folders):
            row[f"probability_{class_name}"] = float(
                metrics["probabilities"][index][class_index]
            )
        rows.append(row)

    with predictions_json_path.open("w", encoding="utf-8") as predictions_file:
        json.dump(rows, predictions_file, indent=2)
    with predictions_csv_path.open(
        "w", encoding="utf-8", newline=""
    ) as predictions_csv:
        writer = csv.DictWriter(predictions_csv, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot_confusion_matrix(
        np.asarray(metrics["confusion_matrix"]),
        class_folders,
        str(matrix_path),
    )
    return {
        "summary": str(summary_path),
        "predictions_json": str(predictions_json_path),
        "predictions_csv": str(predictions_csv_path),
        "confusion_matrix": str(matrix_path),
    }


def run_level_split_evaluation(
    model: nn.Module,
    config: Mapping[str, Any],
    class_folders: Sequence[str],
    device: torch.device,
    output_dir: str,
    level: str,
    split: str,
) -> Dict[str, Any]:
    """Evaluate and persist one bag-level/split pair.

    Args:
        model (nn.Module): Loaded canonical CLAM model.
        config (Mapping[str, Any]): Exact checkpoint configuration.
        class_folders (Sequence[str]): Exact checkpoint class order.
        device (torch.device): Compute device.
        output_dir (str): Artifact directory.
        level (str): Evaluated bag level.
        split (str): Evaluated split.

    Returns:
        Dict[str, Any]: JSON-safe summary plus artifact paths.
    """
    dataset = create_dataset_for_level(level, config, class_folders, split)
    if dataset.num_classes != int(config["num_classes"]):
        raise ValueError(
            f"Checkpoint num_classes={config['num_classes']} does not equal "
            f"dataset classes={dataset.num_classes}."
        )
    generator = torch.Generator().manual_seed(int(config["random_seed"]))
    dataloader = DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=int(config.get("num_workers", 0)),
        worker_init_fn=seed_worker,
        generator=generator,
        pin_memory=torch.cuda.is_available(),
    )
    metrics = evaluate(model, dataloader, device, class_folders)
    print(f"{level}/{split} bags={len(dataset)} counts={get_class_sample_counts(dataset)}")
    print_metrics(metrics, class_folders)
    artifacts = save_level_results(
        metrics, class_folders, output_dir, level, split
    )
    return {**_json_summary(metrics), "artifacts": artifacts}


def _evaluation_controls(
    runtime_config: Mapping[str, Any],
    primary_level: str,
    supplementary_level: Optional[str],
    include_train: Optional[bool],
) -> tuple[List[str], bool]:
    """Resolve optional evaluation controls without changing checkpoint data choices.

    Args:
        runtime_config (Mapping[str, Any]): Runtime config used only for controls.
        primary_level (str): Checkpoint training bag level.
        supplementary_level (Optional[str]): Explicit extra evaluation level.
        include_train (Optional[bool]): Explicit train-diagnostic switch.

    Returns:
        tuple[List[str], bool]: Ordered levels and train-diagnostic setting.
    """
    section = runtime_config.get("evaluation", {})
    if not isinstance(section, Mapping):
        raise ValueError("evaluation config must be a mapping when present.")
    configured_level = section.get(
        "supplementary_bag_level",
        runtime_config.get("supplementary_bag_level"),
    )
    extra = supplementary_level if supplementary_level is not None else configured_level
    levels = [primary_level]
    if extra is not None:
        extra = str(extra)
        if extra not in ("tissue", "slide"):
            raise ValueError("supplementary bag level must be tissue or slide.")
        if extra not in levels:
            levels.append(extra)
    configured_train = bool(
        section.get("include_train", runtime_config.get("evaluate_train", False))
    )
    return levels, configured_train if include_train is None else bool(include_train)


def evaluate_checkpoint(
    config_path: Optional[str] = None,
    supplementary_level: Optional[str] = None,
    include_train: Optional[bool] = None,
) -> Dict[str, Any]:
    """Load and evaluate a canonical checkpoint on validation and test splits.

    Args:
        config_path (Optional[str]): Runtime YAML path used to locate checkpoint
            and outputs, or ``None`` for the module default.
        supplementary_level (Optional[str]): Explicit extra ``tissue`` or
            ``slide`` evaluation, in addition to the checkpoint bag level.
        include_train (Optional[bool]): Whether to add a train diagnostic;
            ``None`` reads the optional evaluation config.

    Returns:
        Dict[str, Any]: Results keyed by ``bag_level/split``.
    """
    runtime_config = load_config(config_path)
    checkpoint_path = Path(runtime_config["paths"]["checkpoint"])
    output_dir = str(runtime_config["paths"]["evaluation_output"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("model_schema") != MODEL_SCHEMA:
        raise ValueError(
            f"Checkpoint model_schema must be '{MODEL_SCHEMA}'; old formats "
            "are not supported."
        )
    checkpoint_config = checkpoint.get("config")
    class_folders = checkpoint.get("class_folders")
    if not isinstance(checkpoint_config, Mapping):
        raise ValueError("Checkpoint must contain a configuration mapping.")
    if not isinstance(class_folders, list) or not all(
        isinstance(name, str) for name in class_folders
    ):
        raise ValueError("Checkpoint must contain an ordered class_folders list.")
    if len(class_folders) != int(checkpoint_config["num_classes"]):
        raise ValueError("Checkpoint class order length does not equal num_classes.")
    primary_level = str(checkpoint.get("bag_level"))
    if primary_level != str(checkpoint_config["bag_level"]):
        raise ValueError("Checkpoint bag_level disagrees with checkpoint config.")

    seed_everything(int(checkpoint_config["random_seed"]))
    model = create_model(checkpoint_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    levels, train_diagnostic = _evaluation_controls(
        runtime_config, primary_level, supplementary_level, include_train
    )
    splits = ["val", "test"]
    if train_diagnostic:
        splits.append("train")

    results: Dict[str, Any] = {}
    for level in levels:
        for split in splits:
            dataset = create_dataset_for_level(
                level, checkpoint_config, class_folders, split
            )
            if len(dataset) == 0:
                results[f"{level}/{split}"] = {
                    "skipped": True,
                    "reason": "empty split",
                }
                continue
            results[f"{level}/{split}"] = run_level_split_evaluation(
                model,
                checkpoint_config,
                class_folders,
                device,
                output_dir,
                level,
                split,
            )

    manifest_path = Path(output_dir) / "evaluation_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as manifest_file:
        json.dump(
            {
                "checkpoint": str(checkpoint_path),
                "model_schema": MODEL_SCHEMA,
                "class_folders": class_folders,
                "primary_bag_level": primary_level,
                "results": results,
            },
            manifest_file,
            indent=2,
        )
    return results


def main() -> None:
    """Evaluate the configured checkpoint.

    Args:
        None: This entry point takes no arguments.

    Returns:
        None: Evaluation artifacts are written to configured paths.
    """
    evaluate_checkpoint()


if __name__ == "__main__":
    main()
