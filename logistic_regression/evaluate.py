"""Evaluate independent logistic-regression checkpoints at one or two bag levels."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

try:
    from . import model as _model_module
    from .config_loader import (
        load_config,
        resolve_inference_run_paths,
    )
    from .dataset import WSIBagDataset, create_bag_dataset
    from .model import RawFeatureStatisticsPooler, TorchLogisticRegression
    from .train import (
        MODEL_SCHEMA,
        POOLING_CONTRACT,
        calculate_metrics,
        collect_pooled_vectors,
        create_dataloader,
        full_class_probabilities,
        json_safe,
        seed_everything,
    )
except ImportError:
    import model as _model_module
    from config_loader import load_config, resolve_inference_run_paths
    from dataset import WSIBagDataset, create_bag_dataset
    from model import RawFeatureStatisticsPooler, TorchLogisticRegression
    from train import (
        MODEL_SCHEMA,
        POOLING_CONTRACT,
        calculate_metrics,
        collect_pooled_vectors,
        create_dataloader,
        full_class_probabilities,
        json_safe,
        seed_everything,
    )


def load_checkpoint_bundle(path: str | Path) -> Dict[str, Any]:
    """Load and validate a schema-versioned logistic-regression dictionary.

    Args:
        path (str | Path): Existing joblib checkpoint path.

    Returns:
        Dict[str, Any]: Validated checkpoint bundle.
    """
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {source}")
    sys.modules.setdefault("model", _model_module)
    loaded = joblib.load(source)
    if not isinstance(loaded, dict):
        raise ValueError("Checkpoint must contain a dictionary bundle.")
    if loaded.get("model_schema") != MODEL_SCHEMA:
        raise ValueError(
            f"Checkpoint model_schema must be '{MODEL_SCHEMA}'; "
            "unsupported formats cannot be evaluated."
        )
    model = loaded.get("model")
    if not isinstance(model, TorchLogisticRegression):
        raise TypeError("Checkpoint model is not TorchLogisticRegression.")
    if loaded.get("backend") != "pytorch":
        raise ValueError("Checkpoint backend is not the required PyTorch GPU backend.")
    if str(loaded.get("fit_device")) != "cuda" or model.fit_device_ != "cuda":
        raise ValueError("Checkpoint was not fitted on the required CUDA device.")
    config = loaded.get("config")
    class_folders = loaded.get("class_folders")
    if not isinstance(config, Mapping):
        raise ValueError("Checkpoint must contain an embedded config mapping.")
    if not isinstance(class_folders, list) or not class_folders or not all(
        isinstance(class_name, str) for class_name in class_folders
    ):
        raise ValueError("Checkpoint must contain a nonempty ordered class_folders list.")
    if len(set(class_folders)) != len(class_folders):
        raise ValueError("Checkpoint class_folders must be unique.")
    num_classes = int(config.get("num_classes", -1))
    if len(class_folders) != num_classes:
        raise ValueError("Checkpoint class order length does not equal num_classes.")
    if str(loaded.get("bag_level")) != str(config.get("bag_level")):
        raise ValueError("Checkpoint bag_level disagrees with embedded config.")
    input_dim = int(loaded.get("input_dim", -1))
    pooling_dim = int(loaded.get("pooling_dim", -1))
    if input_dim <= 0 or pooling_dim != 2 * input_dim:
        raise ValueError("Checkpoint input and pooling dimensions are inconsistent.")
    if loaded.get("pooling_contract") != POOLING_CONTRACT:
        raise ValueError("Checkpoint pooling contract is unsupported.")
    classifier_classes = np.asarray(model.classes_, dtype=np.int64)
    if (
        classifier_classes.ndim != 1
        or classifier_classes.size < 2
        or len(np.unique(classifier_classes)) != classifier_classes.size
        or np.any(classifier_classes < 0)
        or np.any(classifier_classes >= num_classes)
    ):
        raise ValueError("Checkpoint classifier classes violate the fixed class space.")
    return loaded


def evaluation_controls(
    runtime_config: Mapping[str, Any],
    primary_level: str,
    supplementary_level: Optional[str],
    include_train: Optional[bool],
) -> Tuple[List[str], bool]:
    """Resolve primary/supplementary levels and optional train evaluation.

    Args:
        runtime_config (Mapping[str, Any]): Runtime YAML configuration.
        primary_level (str): Checkpoint training bag level.
        supplementary_level (Optional[str]): CLI supplementary level override.
        include_train (Optional[bool]): CLI train-evaluation override.

    Returns:
        Tuple[List[str], bool]: Ordered levels and whether to include train.
    """
    evaluation = runtime_config.get("evaluation", {})
    if not isinstance(evaluation, Mapping):
        raise ValueError("Runtime evaluation section must be a mapping.")
    configured_level = evaluation.get("supplementary_bag_level")
    extra_level = (
        supplementary_level
        if supplementary_level is not None
        else configured_level
    )
    levels = [primary_level]
    if extra_level is not None:
        normalized_level = str(extra_level)
        if normalized_level not in ("tissue", "slide"):
            raise ValueError("supplementary bag level must be tissue or slide.")
        if normalized_level not in levels:
            levels.append(normalized_level)
    configured_include_train = bool(evaluation.get("include_train", False))
    return (
        levels,
        configured_include_train if include_train is None else bool(include_train),
    )


def evaluation_data_config(
    checkpoint_config: Mapping[str, Any],
    runtime_config: Mapping[str, Any],
) -> Dict[str, Any]:
    """Apply runtime location/loading overrides to the embedded data contract.

    Args:
        checkpoint_config (Mapping[str, Any]): Embedded training configuration.
        runtime_config (Mapping[str, Any]): Runtime configuration with local paths.

    Returns:
        Dict[str, Any]: Evaluation configuration preserving checkpoint data choices.
    """
    resolved = dict(checkpoint_config)
    resolved["data_root"] = str(runtime_config["data_root"])
    for key in ("batch_size", "num_workers", "prefetch_factor"):
        resolved[key] = runtime_config[key]
    return resolved


def create_dataset_for_level(
    level: str,
    config: Mapping[str, Any],
    class_folders: Sequence[str],
    split: str,
) -> WSIBagDataset:
    """Create one split using the embedded contract and fixed class order.

    Args:
        level (str): ``tissue`` or ``slide``.
        config (Mapping[str, Any]): Checkpoint-derived data configuration.
        class_folders (Sequence[str]): Fixed checkpoint class order.
        split (str): ``train``, ``val``, or ``test``.

    Returns:
        WSIBagDataset: Independent evaluation bag dataset.
    """
    if level not in ("tissue", "slide"):
        raise ValueError("level must be tissue or slide.")
    return create_bag_dataset(
        config,
        split,
        class_folders=class_folders,
        bag_level=level,
    )


def evaluate_dataset(
    model: TorchLogisticRegression,
    dataset: WSIBagDataset,
    config: Mapping[str, Any],
    class_folders: Sequence[str],
    pooler: RawFeatureStatisticsPooler,
) -> Dict[str, Any]:
    """Collect vectors and evaluate one nonempty dataset split.

    Args:
        model (TorchLogisticRegression): Loaded fitted classifier.
        dataset (WSIBagDataset): Nonempty split dataset.
        config (Mapping[str, Any]): Evaluation loading configuration.
        class_folders (Sequence[str]): Fixed checkpoint class order.
        pooler (RawFeatureStatisticsPooler): Checkpoint-compatible pooler.

    Returns:
        Dict[str, Any]: Aggregate metrics, report, predictions, and identifiers.
    """
    if len(dataset) == 0:
        raise ValueError("Cannot evaluate an empty dataset.")
    dataset.set_epoch(0)
    bag_features, labels, identifiers = collect_pooled_vectors(
        create_dataloader(dataset, config), pooler
    )
    probabilities = full_class_probabilities(
        model, bag_features, len(class_folders)
    )
    predictions = np.asarray(model.predict(bag_features), dtype=np.int64)
    metrics = calculate_metrics(
        labels, predictions, probabilities, len(class_folders)
    )
    fixed_labels = list(range(len(class_folders)))
    metrics.update(
        {
            "classification_report": classification_report(
                labels,
                predictions,
                labels=fixed_labels,
                target_names=list(class_folders),
                output_dict=True,
                zero_division=0,
            ),
            "confusion_matrix": confusion_matrix(
                labels, predictions, labels=fixed_labels
            ),
            "labels": labels,
            "predictions": predictions,
            "probabilities": probabilities,
            "identifiers": identifiers,
            "pooled_vectors": bag_features,
        }
    )
    return metrics


def evaluation_summary(metrics: Mapping[str, Any]) -> Dict[str, Any]:
    """Create a JSON-safe aggregate evaluation summary.

    Args:
        metrics (Mapping[str, Any]): Complete dataset evaluation result.

    Returns:
        Dict[str, Any]: Aggregate metrics, report, matrix, and bag count.
    """
    return {
        "accuracy": float(metrics["accuracy"]),
        "balanced_accuracy": float(metrics["balanced_accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
        "macro_ovr_roc_auc": (
            None
            if metrics["macro_ovr_roc_auc"] is None
            else float(metrics["macro_ovr_roc_auc"])
        ),
        "log_loss": float(metrics["log_loss"]),
        "classification_report": json_safe(metrics["classification_report"]),
        "confusion_matrix": json_safe(metrics["confusion_matrix"]),
        "num_bags": int(len(metrics["labels"])),
    }


def plot_confusion_matrix(
    matrix: np.ndarray,
    class_folders: Sequence[str],
    destination: Path,
) -> Path:
    """Save an annotated fixed-class confusion matrix PNG.

    Args:
        matrix (np.ndarray): Integer confusion matrix shaped ``[K, K]``.
        class_folders (Sequence[str]): Ordered class axis labels.
        destination (Path): Output PNG path.

    Returns:
        Path: Absolute saved plot path.
    """
    figure, axis = plt.subplots(figsize=(10, 9))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        xticks=np.arange(len(class_folders)),
        yticks=np.arange(len(class_folders)),
        xticklabels=list(class_folders),
        yticklabels=list(class_folders),
        xlabel="Predicted",
        ylabel="Actual",
    )
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right")
    threshold = float(matrix.max()) / 2.0 if matrix.size else 0.0
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            axis.text(
                column_index,
                row_index,
                str(int(matrix[row_index, column_index])),
                ha="center",
                va="center",
                color=(
                    "white"
                    if float(matrix[row_index, column_index]) > threshold
                    else "black"
                ),
            )
    figure.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return destination.resolve()


def prediction_rows(
    metrics: Mapping[str, Any],
    class_folders: Sequence[str],
) -> List[Dict[str, Any]]:
    """Build CLAM-style per-bag prediction records.

    Args:
        metrics (Mapping[str, Any]): Complete evaluation result.
        class_folders (Sequence[str]): Ordered fixed class names.

    Returns:
        List[Dict[str, Any]]: Aligned identifiers, labels, and probabilities.
    """
    rows: List[Dict[str, Any]] = []
    for index, label_value in enumerate(metrics["labels"]):
        label = int(label_value)
        prediction = int(metrics["predictions"][index])
        identifier = metrics["identifiers"][index]
        row: Dict[str, Any] = {
            "slide_name": str(identifier["slide_name"]),
            "tissue_name": str(identifier["tissue_name"]),
            "true_label": label,
            "predicted_label": prediction,
            "true_class": class_folders[label],
            "predicted_class": class_folders[prediction],
        }
        for class_index, class_name in enumerate(class_folders):
            row[f"probability_{class_name}"] = float(
                metrics["probabilities"][index, class_index]
            )
        rows.append(row)
    return rows


def save_level_results(
    metrics: Mapping[str, Any],
    class_folders: Sequence[str],
    output_dir: Path,
    level: str,
    split: str,
) -> Dict[str, str]:
    """Save summary, predictions, and confusion plot for one evaluation.

    Args:
        metrics (Mapping[str, Any]): Complete evaluation result.
        class_folders (Sequence[str]): Ordered fixed class names.
        output_dir (Path): Evaluation artifact directory.
        level (str): Evaluated tissue or slide bag level.
        split (str): Evaluated train, validation, or test split.

    Returns:
        Dict[str, str]: Paths to four CLAM-style artifacts.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{level}_{split}"
    summary_path = output_dir / f"{stem}_evaluation.json"
    predictions_json_path = output_dir / f"{stem}_predictions.json"
    predictions_csv_path = output_dir / f"{stem}_predictions.csv"
    confusion_path = output_dir / f"{stem}_confusion_matrix.png"
    with summary_path.open("w", encoding="utf-8") as summary_file:
        json.dump(evaluation_summary(metrics), summary_file, indent=2)
    rows = prediction_rows(metrics, class_folders)
    with predictions_json_path.open("w", encoding="utf-8") as predictions_file:
        json.dump(json_safe(rows), predictions_file, indent=2)
    with predictions_csv_path.open(
        "w", encoding="utf-8", newline=""
    ) as predictions_file:
        writer = csv.DictWriter(predictions_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot_confusion_matrix(
        np.asarray(metrics["confusion_matrix"]),
        class_folders,
        confusion_path,
    )
    return {
        "summary": str(summary_path.resolve()),
        "predictions_json": str(predictions_json_path.resolve()),
        "predictions_csv": str(predictions_csv_path.resolve()),
        "confusion_matrix": str(confusion_path.resolve()),
    }


def print_metrics(level: str, split: str, metrics: Mapping[str, Any]) -> None:
    """Print concise aggregate metrics for one evaluated split.

    Args:
        level (str): Evaluated bag level.
        split (str): Evaluated split.
        metrics (Mapping[str, Any]): Complete evaluation result.

    Returns:
        None: One concise metrics line is written to standard output.
    """
    auc = metrics["macro_ovr_roc_auc"]
    auc_text = "not-valid" if auc is None else f"{float(auc):.4f}"
    print(
        f"{level}/{split} bags={len(metrics['labels'])} "
        f"accuracy={float(metrics['accuracy']):.4f} "
        f"balanced_accuracy={float(metrics['balanced_accuracy']):.4f} "
        f"macro_f1={float(metrics['macro_f1']):.4f} "
        f"macro_ovr_roc_auc={auc_text} "
        f"log_loss={float(metrics['log_loss']):.4f}"
    )


def evaluate_checkpoint(
    config_path: Optional[str] = None,
    checkpoint: Optional[str] = None,
    output_dir: Optional[str] = None,
    supplementary_bag_level: Optional[str] = None,
    include_train: Optional[bool] = None,
) -> Dict[str, Any]:
    """Evaluate a checkpoint over validation/test and optional train/level.

    Args:
        config_path (Optional[str]): Runtime YAML path or package default.
        checkpoint (Optional[str]): Explicit checkpoint path override.
        output_dir (Optional[str]): Explicit evaluation directory override.
        supplementary_bag_level (Optional[str]): Optional tissue/slide level.
        include_train (Optional[bool]): Whether to include the training split.

    Returns:
        Dict[str, Any]: JSON-safe manifest including result summaries and paths.
    """
    runtime_config = load_config(config_path)
    resolve_inference_run_paths(runtime_config)
    checkpoint_path = Path(
        checkpoint
        if checkpoint is not None
        else str(runtime_config["paths"]["checkpoint"])
    ).expanduser().resolve()
    destination = Path(
        output_dir
        if output_dir is not None
        else str(runtime_config["paths"]["evaluation_output"])
    ).expanduser().resolve()
    bundle = load_checkpoint_bundle(checkpoint_path)
    checkpoint_config = bundle["config"]
    class_folders = list(bundle["class_folders"])
    data_config = evaluation_data_config(checkpoint_config, runtime_config)
    if int(data_config["num_classes"]) != len(class_folders):
        raise ValueError("Evaluation num_classes disagrees with checkpoint classes.")
    seed_everything(int(data_config["random_seed"]))
    levels, train_diagnostic = evaluation_controls(
        runtime_config,
        str(bundle["bag_level"]),
        supplementary_bag_level,
        include_train,
    )
    splits = ["val", "test"] + (["train"] if train_diagnostic else [])
    pooler = RawFeatureStatisticsPooler(
        epsilon=float(bundle["pooling_population_std_epsilon"])
    )
    results: Dict[str, Any] = {}
    for level in levels:
        for split in splits:
            result_key = f"{level}/{split}"
            dataset = create_dataset_for_level(
                level, data_config, class_folders, split
            )
            if dataset.num_classes != len(class_folders):
                raise ValueError(
                    f"{result_key} dataset class count violates checkpoint contract."
                )
            if dataset.class_folders != class_folders:
                raise ValueError(
                    f"{result_key} dataset class order violates checkpoint contract."
                )
            if len(dataset) == 0:
                results[result_key] = {
                    "skipped": True,
                    "reason": "empty split",
                    "artifacts": {},
                }
                print(f"{result_key} skipped: empty split")
                continue
            metrics = evaluate_dataset(
                bundle["model"], dataset, data_config, class_folders, pooler
            )
            artifacts = save_level_results(
                metrics, class_folders, destination, level, split
            )
            results[result_key] = {
                **evaluation_summary(metrics),
                "artifacts": artifacts,
            }
            print_metrics(level, split, metrics)

    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / "evaluation_manifest.json"
    manifest: Dict[str, Any] = {
        "checkpoint": str(checkpoint_path),
        "model_schema": MODEL_SCHEMA,
        "class_folders": class_folders,
        "primary_bag_level": str(bundle["bag_level"]),
        "evaluated_levels": levels,
        "evaluated_splits": splits,
        "data_root": str(data_config["data_root"]),
        "output_dir": str(destination),
        "results": results,
        "manifest": str(manifest_path),
    }
    with manifest_path.open("w", encoding="utf-8") as manifest_file:
        json.dump(json_safe(manifest), manifest_file, indent=2)
    print(f"Saved evaluation manifest: {manifest_path}")
    return json_safe(manifest)


def parse_args() -> argparse.Namespace:
    """Parse standalone evaluation command-line arguments.

    Args:
        None: This function reads process command-line arguments.

    Returns:
        argparse.Namespace: Parsed configuration, paths, and controls.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate raw-feature statistics logistic regression."
    )
    parser.add_argument("--config", type=str, default=None, help="Runtime YAML.")
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Checkpoint joblib override."
    )
    parser.add_argument(
        "--output-dir", type=str, default=None, help="Evaluation output override."
    )
    parser.add_argument(
        "--supplementary-bag-level",
        choices=("tissue", "slide"),
        default=None,
        help="Optional second evaluation bag level.",
    )
    parser.add_argument(
        "--include-train",
        action="store_true",
        default=None,
        help="Also evaluate the training split.",
    )
    return parser.parse_args()


def main() -> None:
    """Run standalone checkpoint evaluation.

    Args:
        None: This entry point reads command-line arguments.

    Returns:
        None: Evaluation artifacts are written to the selected output directory.
    """
    arguments = parse_args()
    evaluate_checkpoint(
        config_path=arguments.config,
        checkpoint=arguments.checkpoint,
        output_dir=arguments.output_dir,
        supplementary_bag_level=arguments.supplementary_bag_level,
        include_train=arguments.include_train,
    )


if __name__ == "__main__":
    main()
