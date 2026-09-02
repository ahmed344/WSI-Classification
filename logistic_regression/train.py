"""Train the independent raw-feature logistic-regression baseline."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss as sklearn_log_loss,
    roc_auc_score,
)
from torch.utils.data import DataLoader

try:
    from .config_loader import allocate_training_run, load_config
    from .dataset import WSIBagDataset, collate_fn, create_bag_dataset
    from .model import (
        RawFeatureStatisticsPooler,
        TorchLogisticRegression,
        pooling_contract,
        pooling_output_dim,
    )
except ImportError:
    from config_loader import allocate_training_run, load_config
    from dataset import WSIBagDataset, collate_fn, create_bag_dataset
    from model import (
        RawFeatureStatisticsPooler,
        TorchLogisticRegression,
        pooling_contract,
        pooling_output_dim,
    )


MODEL_SCHEMA = "raw_feature_statistics_cuda_logistic_regression_v3"
LEGACY_MODEL_SCHEMA = "raw_feature_statistics_cuda_logistic_regression_v2"


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch deterministically.

    Args:
        seed (int): Nonnegative global random seed.

    Returns:
        None: Process random generators are seeded in place.
    """
    if seed < 0:
        raise ValueError("seed must be nonnegative.")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """Seed one DataLoader worker from its PyTorch worker seed.

    Args:
        worker_id (int): DataLoader worker identifier.

    Returns:
        None: Python and NumPy worker generators are seeded.
    """
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def create_dataloader(
    dataset: WSIBagDataset,
    config: Mapping[str, Any],
) -> DataLoader:
    """Create a deterministic, non-shuffling bag DataLoader.

    Args:
        dataset (WSIBagDataset): Dataset whose bags should be loaded.
        config (Mapping[str, Any]): Runtime loading configuration.

    Returns:
        DataLoader: Deterministic padded-bag loader using ``collate_fn``.
    """
    num_workers = int(config.get("num_workers", 0))
    generator = torch.Generator().manual_seed(int(config["random_seed"]))
    arguments: Dict[str, Any] = {
        "dataset": dataset,
        "batch_size": int(config["batch_size"]),
        "shuffle": False,
        "num_workers": num_workers,
        "collate_fn": collate_fn,
        "worker_init_fn": seed_worker,
        "generator": generator,
        "persistent_workers": False,
    }
    if num_workers > 0:
        arguments["prefetch_factor"] = int(config.get("prefetch_factor", 2))
    return DataLoader(**arguments)


def collect_pooled_vectors(
    dataloader: DataLoader,
    pooler: RawFeatureStatisticsPooler,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, str]]]:
    """Collect mask-aware pooled vectors, labels, and bag identifiers.

    Args:
        dataloader (DataLoader): Padded raw-feature bag loader.
        pooler (RawFeatureStatisticsPooler): Mask-aware population-moment pooler.

    Returns:
        Tuple[np.ndarray, np.ndarray, List[Dict[str, str]]]: Finite pooled
            matrix, integer labels, and aligned slide/tissue identifiers.
    """
    pooled_batches: List[np.ndarray] = []
    label_batches: List[np.ndarray] = []
    identifiers: List[Dict[str, str]] = []
    for batch in dataloader:
        pooled = pooler(batch["features"], batch["masks"])
        pooled_batches.append(pooled.detach().cpu().numpy())
        label_batches.append(batch["labels"].detach().cpu().numpy())
        identifiers.extend(
            {
                "slide_name": str(slide_name),
                "tissue_name": str(tissue_name),
            }
            for slide_name, tissue_name in zip(
                batch["slide_names"], batch["tissue_names"]
            )
        )
    if not pooled_batches:
        raise ValueError("Cannot collect pooled vectors from an empty DataLoader.")
    features = np.ascontiguousarray(np.concatenate(pooled_batches), dtype=np.float64)
    labels = np.concatenate(label_batches).astype(np.int64, copy=False)
    if not np.isfinite(features).all():
        raise ValueError("Pooled bag vectors contain non-finite values.")
    return features, labels, identifiers


def full_class_probabilities(
    model: TorchLogisticRegression,
    bag_features: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """Expand estimator probabilities into the fixed checkpoint class space.

    Args:
        model (TorchLogisticRegression): Fitted classifier wrapper.
        bag_features (np.ndarray): Pooled bag matrix shaped ``[N, features]``.
        num_classes (int): Fixed total number of target classes.

    Returns:
        np.ndarray: Probability matrix shaped ``[N, num_classes]``.
    """
    classifier_classes = np.asarray(model.classes_, dtype=np.int64)
    if (
        classifier_classes.ndim != 1
        or classifier_classes.size < 2
        or np.any(classifier_classes < 0)
        or np.any(classifier_classes >= num_classes)
        or len(np.unique(classifier_classes)) != classifier_classes.size
    ):
        raise ValueError("Fitted classifier classes are invalid for the fixed class space.")
    partial = np.asarray(model.predict_proba(bag_features), dtype=np.float64)
    if partial.shape != (bag_features.shape[0], classifier_classes.size):
        raise ValueError("Classifier probability output has an unexpected shape.")
    probabilities = np.zeros((bag_features.shape[0], num_classes), dtype=np.float64)
    probabilities[:, classifier_classes] = partial
    if not np.isfinite(probabilities).all():
        raise ValueError("Classifier probabilities contain non-finite values.")
    row_sums = probabilities.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0.0):
        raise ValueError("Classifier probabilities have a nonpositive row sum.")
    probabilities /= row_sums
    return probabilities


def multiclass_roc_auc(
    labels: Sequence[int],
    probabilities: np.ndarray,
    num_classes: int,
) -> Optional[float]:
    """Compute macro one-vs-rest ROC AUC when all fixed classes occur.

    Args:
        labels (Sequence[int]): Ground-truth fixed-class indices.
        probabilities (np.ndarray): Full fixed-class probability matrix.
        num_classes (int): Number of fixed target classes.

    Returns:
        Optional[float]: Macro OvR ROC AUC, or ``None`` when invalid.
    """
    label_array = np.asarray(labels, dtype=np.int64)
    if set(label_array.tolist()) != set(range(num_classes)):
        return None
    try:
        if num_classes == 2:
            return float(roc_auc_score(label_array, probabilities[:, 1]))
        return float(
            roc_auc_score(
                label_array,
                probabilities,
                labels=list(range(num_classes)),
                multi_class="ovr",
                average="macro",
            )
        )
    except ValueError:
        return None


def calculate_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    num_classes: int,
) -> Dict[str, Any]:
    """Calculate fixed-label classification and probability metrics.

    Args:
        labels (np.ndarray): Ground-truth labels shaped ``[N]``.
        predictions (np.ndarray): Predicted labels shaped ``[N]``.
        probabilities (np.ndarray): Full probabilities shaped ``[N, classes]``.
        num_classes (int): Fixed target class count.

    Returns:
        Dict[str, Any]: Accuracy, balanced accuracy, macro F1, optional macro
            OvR ROC AUC, and multiclass log loss.
    """
    fixed_labels = list(range(num_classes))
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(
            f1_score(
                labels,
                predictions,
                labels=fixed_labels,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_ovr_roc_auc": multiclass_roc_auc(
            labels, probabilities, num_classes
        ),
        "log_loss": float(
            sklearn_log_loss(labels, probabilities, labels=fixed_labels)
        ),
    }


def evaluate_candidate(
    model: TorchLogisticRegression,
    bag_features: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
) -> Dict[str, Any]:
    """Evaluate one fitted regularization candidate.

    Args:
        model (TorchLogisticRegression): Fitted candidate.
        bag_features (np.ndarray): Pooled validation matrix.
        labels (np.ndarray): Validation labels.
        num_classes (int): Fixed target class count.

    Returns:
        Dict[str, Any]: Fixed-label validation metrics.
    """
    probabilities = full_class_probabilities(model, bag_features, num_classes)
    predictions = np.asarray(model.predict(bag_features), dtype=np.int64)
    return calculate_metrics(labels, predictions, probabilities, num_classes)


def config_without_private_keys(value: Any) -> Any:
    """Recursively remove private mapping keys from configuration data.

    Args:
        value (Any): Configuration value to sanitize.

    Returns:
        Any: Equivalent JSON/joblib-safe structure without underscore keys.
    """
    if isinstance(value, Mapping):
        return {
            str(key): config_without_private_keys(item)
            for key, item in value.items()
            if not str(key).startswith("_")
        }
    if isinstance(value, (list, tuple)):
        return [config_without_private_keys(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def json_safe(value: Any) -> Any:
    """Convert nested scientific Python values to JSON-safe values.

    Args:
        value (Any): Arbitrarily nested value.

    Returns:
        Any: JSON-serializable equivalent.
    """
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save_regularization_plot(
    history: Sequence[Mapping[str, Any]],
    selection_metric: str,
    destination: Path,
) -> Path:
    """Plot validation regularization-search metrics against configured C.

    Args:
        history (Sequence[Mapping[str, Any]]): Candidate validation records.
        selection_metric (str): Metric used to select the best candidate.
        destination (Path): Output PNG path.

    Returns:
        Path: Absolute saved plot path.
    """
    ordered = sorted(history, key=lambda record: float(record["C"]))
    c_values = [float(record["C"]) for record in ordered]
    figure, primary_axis = plt.subplots(figsize=(9, 6))
    for metric_name in ("accuracy", "balanced_accuracy", "macro_f1"):
        primary_axis.plot(
            c_values,
            [float(record["validation_metrics"][metric_name]) for record in ordered],
            marker="o",
            label=metric_name,
        )
    primary_axis.set_xscale("log")
    primary_axis.set_xlabel("Inverse regularization strength (C)")
    primary_axis.set_ylabel("Validation score")
    primary_axis.set_title(
        f"Regularization search (selected by {selection_metric})"
    )
    secondary_axis = primary_axis.twinx()
    secondary_axis.plot(
        c_values,
        [float(record["validation_metrics"]["log_loss"]) for record in ordered],
        color="black",
        linestyle="--",
        marker="x",
        label="log_loss",
    )
    secondary_axis.set_ylabel("Validation log loss")
    lines = primary_axis.lines + secondary_axis.lines
    primary_axis.legend(lines, [line.get_label() for line in lines], loc="best")
    primary_axis.grid(True, alpha=0.25)
    figure.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return destination.resolve()


def _validate_training_datasets(
    train_dataset: WSIBagDataset,
    val_dataset: WSIBagDataset,
    configured_num_classes: int,
) -> None:
    """Validate nonempty splits and their frozen class contract.

    Args:
        train_dataset (WSIBagDataset): Training split dataset.
        val_dataset (WSIBagDataset): Validation split dataset.
        configured_num_classes (int): Required fixed class count.

    Returns:
        None: Validation succeeds by returning normally.
    """
    if len(train_dataset) == 0:
        raise ValueError("Training split contains no bags.")
    if len(val_dataset) == 0:
        raise ValueError("Validation split contains no bags.")
    if train_dataset.class_folders != val_dataset.class_folders:
        raise ValueError("Training and validation class order is not frozen.")
    if train_dataset.num_classes != configured_num_classes:
        raise ValueError(
            f"Configured num_classes={configured_num_classes} does not equal "
            f"discovered classes={train_dataset.num_classes}."
        )


def train_logistic_regression(config_path: Optional[str] = None) -> Dict[str, str]:
    """Train, select, and save the validation-selected baseline.

    Args:
        config_path (Optional[str]): YAML path, or ``None`` for package default.

    Returns:
        Dict[str, str]: Paths to the run directory and saved training artifacts.
    """
    config = load_config(config_path)
    seed_everything(int(config["random_seed"]))
    run_dir = allocate_training_run(config)
    train_dataset = create_bag_dataset(config, "train")
    class_folders = list(train_dataset.class_folders)
    val_dataset = create_bag_dataset(config, "val", class_folders=class_folders)
    _validate_training_datasets(
        train_dataset, val_dataset, int(config["num_classes"])
    )
    train_dataset.set_epoch(0)
    val_dataset.set_epoch(0)
    pooling_statistics = list(config["pooling_statistics"])
    pooler = RawFeatureStatisticsPooler(
        epsilon=float(config["pooling_population_std_epsilon"]),
        statistics=pooling_statistics,
    )
    train_vectors, train_labels, _ = collect_pooled_vectors(
        create_dataloader(train_dataset, config), pooler
    )
    val_vectors, val_labels, _ = collect_pooled_vectors(
        create_dataloader(val_dataset, config), pooler
    )
    if len(np.unique(train_labels)) < 2:
        raise ValueError("Training split must contain at least two represented classes.")

    selection_metric = str(config["selection_metric"])
    history: List[Dict[str, Any]] = []
    candidates: List[TorchLogisticRegression] = []
    for c_value in config["c_values"]:
        candidate = TorchLogisticRegression(
            C=float(c_value),
            device=str(config["device"]),
            max_iter=int(config["max_iter"]),
            tolerance=float(config["tolerance"]),
            learning_rate=float(config["learning_rate"]),
            class_weight=config.get("class_weight"),
            random_seed=int(config["random_seed"]),
        )
        candidate.fit(train_vectors, train_labels)
        validation_metrics = evaluate_candidate(
            candidate, val_vectors, val_labels, int(config["num_classes"])
        )
        candidates.append(candidate)
        history.append(
            {
                "C": float(c_value),
                "validation_metrics": validation_metrics,
                "classifier_classes": candidate.classes_.tolist(),
                "n_iter": candidate.n_iter_,
                "fit_device": candidate.fit_device_,
            }
        )
        print(
            f"C={float(c_value):g} "
            f"{selection_metric}={validation_metrics[selection_metric]:.4f} "
            f"log_loss={validation_metrics['log_loss']:.4f}"
        )

    best_index = min(
        range(len(history)),
        key=lambda index: (
            -float(history[index]["validation_metrics"][selection_metric]),
            float(history[index]["validation_metrics"]["log_loss"]),
            float(history[index]["C"]),
        ),
    )
    best_model = candidates[best_index]
    best_record = history[best_index]
    sanitized_config = config_without_private_keys(config)
    selection_metadata = {
        "metric": selection_metric,
        "direction": "descending",
        "tie_breakers": ["lower_log_loss", "smaller_C"],
        "selected_candidate_index": best_index,
        "validation_only": True,
    }
    bundle: Dict[str, Any] = {
        "model": best_model,
        "model_schema": MODEL_SCHEMA,
        "backend": "pytorch",
        "fit_device": best_model.fit_device_,
        "config": sanitized_config,
        "class_folders": class_folders,
        "bag_level": str(config["bag_level"]),
        "input_dim": int(config["input_dim"]),
        "pooling_statistics": pooling_statistics,
        "pooling_dim": pooling_output_dim(
            int(config["input_dim"]), pooling_statistics
        ),
        "pooling_contract": pooling_contract(pooling_statistics),
        "pooling_population_std_epsilon": float(
            config["pooling_population_std_epsilon"]
        ),
        "selected_C": float(best_record["C"]),
        "selection": selection_metadata,
        "validation_metrics": best_record["validation_metrics"],
    }
    checkpoint_path = Path(str(config["paths"]["checkpoint"])).resolve()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, checkpoint_path)

    history_document = {
        "model_schema": MODEL_SCHEMA,
        "class_folders": class_folders,
        "bag_level": str(config["bag_level"]),
        "train_bags": len(train_dataset),
        "validation_bags": len(val_dataset),
        "train_sampling_epoch": 0,
        "selection": selection_metadata,
        "candidates": history,
    }
    history_path = run_dir / "training_history.json"
    with history_path.open("w", encoding="utf-8") as history_file:
        json.dump(json_safe(history_document), history_file, indent=2)
    plot_path = save_regularization_plot(
        history, selection_metric, run_dir / "training_history.png"
    )
    report = {
        "model_schema": MODEL_SCHEMA,
        "backend": "pytorch",
        "fit_device": best_model.fit_device_,
        "checkpoint": str(checkpoint_path),
        "selected_C": float(best_record["C"]),
        "selection": selection_metadata,
        "validation_metrics": best_record["validation_metrics"],
        "class_folders": class_folders,
        "bag_level": str(config["bag_level"]),
        "input_dim": int(config["input_dim"]),
        "pooling_statistics": pooling_statistics,
        "pooling_dim": pooling_output_dim(
            int(config["input_dim"]), pooling_statistics
        ),
        "pooling_contract": pooling_contract(pooling_statistics),
        "train_bags": len(train_dataset),
        "validation_bags": len(val_dataset),
        "test_evaluated": False,
    }
    report_path = run_dir / "best_model_report.json"
    with report_path.open("w", encoding="utf-8") as report_file:
        json.dump(json_safe(report), report_file, indent=2)
    print(
        f"Selected C={float(best_record['C']):g}; "
        f"{selection_metric}="
        f"{float(best_record['validation_metrics'][selection_metric]):.4f}"
    )
    print(f"Saved checkpoint: {checkpoint_path}")
    return {
        "run_dir": str(run_dir.resolve()),
        "checkpoint": str(checkpoint_path),
        "training_history": str(history_path.resolve()),
        "training_plot": str(plot_path),
        "best_model_report": str(report_path.resolve()),
    }


def parse_args() -> argparse.Namespace:
    """Parse training command-line arguments.

    Args:
        None: This function reads process command-line arguments.

    Returns:
        argparse.Namespace: Parsed optional configuration path.
    """
    parser = argparse.ArgumentParser(
        description="Train raw-feature statistics logistic regression."
    )
    parser.add_argument("--config", type=str, default=None, help="Configuration YAML.")
    return parser.parse_args()


def main() -> None:
    """Run logistic-regression training from the command line.

    Args:
        None: This entry point reads command-line arguments.

    Returns:
        None: Training artifacts are written to the configured run directory.
    """
    arguments = parse_args()
    train_logistic_regression(arguments.config)


if __name__ == "__main__":
    main()
