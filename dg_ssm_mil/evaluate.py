"""
Tissue-level evaluation script for trained DG-SSM-MIL models.
"""
import argparse
import json
import os
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from .config_loader import load_config, resolve_device, resolve_feature_file_suffix
    from .dataset import collate_fn, get_class_sample_counts
    from .model import DGSSMMILModel
    from .train import _safe_multiclass_auc, create_datasets
except ImportError:
    from config_loader import (  # type: ignore
        load_config,
        resolve_device,
        resolve_feature_file_suffix,
    )
    from dataset import collate_fn, get_class_sample_counts  # type: ignore
    from model import DGSSMMILModel  # type: ignore
    from train import _safe_multiclass_auc, create_datasets  # type: ignore

sns.set_theme(style="white")


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for tissue-level evaluation.

    Args:
        None: Arguments are read from the command line.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Evaluate tissue-level DG-SSM-MIL.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.yml. Defaults to dg_ssm_mil/config.yml.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional checkpoint path. Defaults to paths.checkpoint from config.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional output directory. Defaults to paths.evaluation_output.",
    )
    return parser.parse_args()


def load_trained_model(
    checkpoint_path: str,
    device: torch.device,
) -> Tuple[DGSSMMILModel, Dict[str, Any]]:
    """
    Load a trained DG-SSM-MIL model from checkpoint.

    Args:
        checkpoint_path (str): Path to a checkpoint created by `train.py`.
        device (torch.device): Device where model weights should be loaded.

    Returns:
        Tuple[DGSSMMILModel, Dict[str, Any]]: Loaded model and checkpoint dictionary.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "config" not in checkpoint:
        raise KeyError("Checkpoint is missing required key 'config'.")
    if "model_state_dict" not in checkpoint:
        raise KeyError("Checkpoint is missing required key 'model_state_dict'.")

    model = DGSSMMILModel.from_config(checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def evaluate_split(
    model: DGSSMMILModel,
    dataloader: DataLoader,
    device: torch.device,
    class_names: List[str],
) -> Dict[str, Any]:
    """
    Evaluate a model on one tissue-level split.

    Args:
        model (DGSSMMILModel): Trained model in evaluation mode.
        dataloader (DataLoader): Dataloader for one tissue split.
        device (torch.device): Device used for inference.
        class_names (List[str]): Ordered class names for metric labels.

    Returns:
        Dict[str, Any]: Evaluation metrics, predictions, labels, probabilities,
        and sample identifiers.
    """
    all_preds: List[int] = []
    all_labels: List[int] = []
    all_probs: List[List[float]] = []
    all_slide_names: List[str] = []
    all_tissue_names: List[str] = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            features = batch["features"].to(device)
            coords = batch["coords"].to(device)
            masks = batch["masks"].to(device)
            labels = batch["labels"].to(device)

            tissue_indices = batch.get("tissue_indices")
            if tissue_indices is not None:
                tissue_indices = tissue_indices.to(device)
            outputs = model(features, coords, masks, tissue_indices=tissue_indices)
            probs = torch.softmax(outputs["logits"], dim=1)
            preds = torch.argmax(outputs["logits"], dim=1)

            all_preds.extend(int(pred) for pred in preds.cpu().numpy())
            all_labels.extend(int(label) for label in labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy().tolist())
            all_slide_names.extend(batch["slide_names"])
            all_tissue_names.extend(batch["tissue_names"])

    labels_order = list(range(len(class_names)))
    cm = confusion_matrix(all_labels, all_preds, labels=labels_order)
    report = classification_report(
        all_labels,
        all_preds,
        labels=labels_order,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(all_labels, all_preds)),
        "balanced_accuracy": float(balanced_accuracy_score(all_labels, all_preds)),
        "auc": _safe_multiclass_auc(all_labels, all_probs),
        "confusion_matrix": cm,
        "classification_report": report,
        "predictions": all_preds,
        "labels": all_labels,
        "probabilities": all_probs,
        "slide_names": all_slide_names,
        "tissue_names": all_tissue_names,
    }


def plot_confusion_matrix(
    confusion: np.ndarray,
    class_names: List[str],
    split: str,
    save_path: str,
) -> None:
    """
    Save a confusion matrix heatmap.

    Args:
        confusion (np.ndarray): Confusion matrix with shape `[num_classes, num_classes]`.
        class_names (List[str]): Ordered class names for axis tick labels.
        split (str): Split name used in the plot title.
        save_path (str): Destination path for the PNG figure.

    Returns:
        None: The confusion matrix plot is saved to disk.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(10, 10))
    ax = sns.heatmap(
        confusion,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=False,
        annot_kws={"size": 16},
    )
    ax.tick_params(axis="both", labelsize=12)
    plt.xlabel("Predicted", fontsize=14)
    plt.ylabel("Actual", fontsize=14)
    plt.title(f"Tissue-Level Confusion Matrix ({split})", fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_split_results(
    metrics: Dict[str, Any],
    class_names: List[str],
    output_dir: str,
    split: str,
) -> None:
    """
    Save tissue-level metrics, predictions, and confusion matrix plot for one split.

    Args:
        metrics (Dict[str, Any]): Metrics returned by `evaluate_split`.
        class_names (List[str]): Ordered class names.
        output_dir (str): Directory where evaluation artifacts are written.
        split (str): Split name, typically `train` or `val`.

    Returns:
        None: JSON and PNG artifacts are written to disk.
    """
    os.makedirs(output_dir, exist_ok=True)
    summary = {
        "level": "tissue",
        "split": split,
        "accuracy": float(metrics["accuracy"]),
        "balanced_accuracy": float(metrics["balanced_accuracy"]),
        "auc": float(metrics["auc"]),
        "confusion_matrix": metrics["confusion_matrix"].tolist(),
        "classification_report": metrics["classification_report"],
        "labels": [int(label) for label in metrics["labels"]],
        "predictions": [int(pred) for pred in metrics["predictions"]],
        "slide_names": metrics["slide_names"],
        "tissue_names": metrics["tissue_names"],
    }
    summary_path = os.path.join(output_dir, f"tissue_evaluation_{split}.json")
    with open(summary_path, "w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2)

    predictions: List[Dict[str, Any]] = []
    for sample_idx, slide_name in enumerate(metrics["slide_names"]):
        true_label = int(metrics["labels"][sample_idx])
        predicted_label = int(metrics["predictions"][sample_idx])
        predictions.append(
            {
                "slide_name": slide_name,
                "tissue_name": metrics["tissue_names"][sample_idx],
                "true_label": true_label,
                "predicted_label": predicted_label,
                "true_class": class_names[true_label],
                "predicted_class": class_names[predicted_label],
                "probabilities": {
                    class_names[class_idx]: float(metrics["probabilities"][sample_idx][class_idx])
                    for class_idx in range(len(class_names))
                },
            }
        )
    predictions_path = os.path.join(output_dir, f"tissue_predictions_{split}.json")
    with open(predictions_path, "w", encoding="utf-8") as predictions_file:
        json.dump(predictions, predictions_file, indent=2)

    confusion_path = os.path.join(output_dir, f"tissue_confusion_matrix_{split}.png")
    plot_confusion_matrix(
        metrics["confusion_matrix"],
        class_names,
        split,
        confusion_path,
    )
    print(f"{split} results saved to {summary_path}")
    print(f"{split} predictions saved to {predictions_path}")
    print(f"{split} confusion matrix saved to {confusion_path}")


def print_split_summary(metrics: Dict[str, Any], split: str) -> None:
    """
    Print a compact evaluation summary for one split.

    Args:
        metrics (Dict[str, Any]): Metrics returned by `evaluate_split`.
        split (str): Split name, typically `train` or `val`.

    Returns:
        None: Summary is printed to stdout.
    """
    print(
        f"{split}: accuracy={metrics['accuracy']:.4f}, "
        f"balanced_accuracy={metrics['balanced_accuracy']:.4f}, "
        f"auc={metrics['auc']:.4f}"
    )


def run_split_evaluation(
    model: DGSSMMILModel,
    dataset: Any,
    config: Dict[str, Any],
    device: torch.device,
    split: str,
    output_dir: str,
) -> None:
    """
    Evaluate and save artifacts for one tissue-level split.

    Args:
        model (DGSSMMILModel): Loaded DG-SSM-MIL model.
        dataset (Any): Tissue-level dataset instance.
        config (Dict[str, Any]): Parsed evaluation configuration.
        device (torch.device): Device used for inference.
        split (str): Split name, typically `train` or `val`.
        output_dir (str): Directory where artifacts are written.

    Returns:
        None: Evaluation artifacts are saved to disk.
    """
    class_counts = get_class_sample_counts(dataset)
    print("\n" + "-" * 60)
    print(f"Running tissue-level evaluation on {split} split...")
    print(f"Number of tissue samples ({split}): {len(dataset)}")
    for class_name in dataset.class_folders:
        print(f"  {class_name}: {class_counts[class_name]}")

    dataloader = DataLoader(
        dataset,
        batch_size=int(config.get("evaluation_batch_size", 1)),
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=int(config.get("num_workers", 0)),
    )
    metrics = evaluate_split(model, dataloader, device, dataset.class_folders)
    print_split_summary(metrics, split)
    save_split_results(metrics, dataset.class_folders, output_dir, split)


def main() -> None:
    """
    Run tissue-level evaluation for train and validation splits.

    Args:
        None: Configuration and optional overrides are read from CLI arguments.

    Returns:
        None: Evaluation artifacts are written to the configured output directory.
    """
    args = parse_args()
    config = load_config(args.config)
    checkpoint_path = args.checkpoint or config["paths"]["checkpoint"]
    if args.checkpoint is None and not os.path.exists(checkpoint_path):
        parent, filename = os.path.split(checkpoint_path)
        checkpoint_path = os.path.join(parent, "repeat_00", filename)
    output_dir = args.output_dir or config["paths"]["evaluation_output"]
    device = torch.device(resolve_device(config))

    print(f"Using device: {device}")
    print(f"Loading checkpoint from {checkpoint_path}")
    print(
        f"Using feature model '{config['feature_model']}' "
        f"with suffix '{resolve_feature_file_suffix(config)}'"
    )
    model, checkpoint = load_trained_model(checkpoint_path, device)
    checkpoint_config = checkpoint["config"]
    class_folders = checkpoint.get("class_folders")
    if not class_folders:
        raise KeyError("Checkpoint is missing frozen 'class_folders'.")
    train_dataset, val_dataset, test_dataset = create_datasets(
        checkpoint_config,
        class_folders=class_folders,
        apply_training_filter=False,
    )
    run_split_evaluation(model, train_dataset, config, device, "train", output_dir)
    run_split_evaluation(model, val_dataset, config, device, "val", output_dir)
    run_split_evaluation(model, test_dataset, config, device, "test", output_dir)


if __name__ == "__main__":
    main()
