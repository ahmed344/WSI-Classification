"""
Training entrypoint for the independent DG-SSM-MIL tissue workflow.
"""
import argparse
import json
import os
import subprocess
import sys
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

sns.set_theme(style="darkgrid")

try:
    from .config_loader import (
        get_coordinate_columns,
        load_config,
        resolve_device,
        resolve_feature_file_suffix,
    )
    from .dataset import (
        DGSSMMILTissueDataset,
        collate_fn,
        compute_class_weights,
        compute_sample_weights,
        get_class_sample_counts,
    )
    from .model import DGSSMMILModel
except ImportError:
    from config_loader import (  # type: ignore
        get_coordinate_columns,
        load_config,
        resolve_device,
        resolve_feature_file_suffix,
    )
    from dataset import (  # type: ignore
        DGSSMMILTissueDataset,
        collate_fn,
        compute_class_weights,
        compute_sample_weights,
        get_class_sample_counts,
    )
    from model import DGSSMMILModel  # type: ignore


def set_random_seeds(random_seed: int) -> None:
    """
    Set random seeds for reproducible training.

    Args:
        random_seed (int): Seed applied to NumPy and PyTorch.

    Returns:
        None: Random generator state is updated in-place.
    """
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)


def resolve_best_checkpoint_metric(metric_name: str) -> Tuple[str, bool, str]:
    """
    Resolve a configured checkpoint metric.

    Args:
        metric_name (str): Configured metric name.

    Returns:
        Tuple[str, bool, str]: Metric key, maximize flag, and normalized name.
    """
    normalized = metric_name.strip().lower()
    mapping = {
        "auc": ("auc", True, "auc"),
        "balanced_accuracy": ("balanced_accuracy", True, "balanced_accuracy"),
        "accuracy": ("accuracy", True, "accuracy"),
        "loss": ("loss", False, "loss"),
    }
    if normalized not in mapping:
        raise ValueError(
            "Invalid best_checkpoint_metric. "
            f"Valid options are: {', '.join(mapping.keys())}."
        )
    return mapping[normalized]


def create_datasets(
    config: Dict[str, Any],
    class_folders: List[str] | None = None,
    apply_training_filter: bool = True,
) -> Tuple[
    DGSSMMILTissueDataset,
    DGSSMMILTissueDataset,
    DGSSMMILTissueDataset,
]:
    """
    Create train and validation datasets from config.

    Args:
        config (Dict[str, Any]): Parsed DG-SSM-MIL configuration.
        class_folders (List[str] | None): Optional frozen class order.
        apply_training_filter (bool): Whether the training split should exclude
            tissues above `skip_tissues_above_tiles`.

    Returns:
        Tuple[DGSSMMILTissueDataset, DGSSMMILTissueDataset,
        DGSSMMILTissueDataset]: Train, validation, and test datasets.
    """
    feature_file_suffix = resolve_feature_file_suffix(config)
    coordinate_columns = get_coordinate_columns(config)
    dataset_kwargs = {
        "data_root": str(config["data_root"]),
        "class_folders": class_folders,
        "train_ratio": float(config["train_ratio"]),
        "val_ratio": float(config.get("val_ratio", 0.1)),
        "test_ratio": float(config.get("test_ratio", 0.1)),
        "random_seed": int(config["random_seed"]),
        "feature_file_suffix": feature_file_suffix,
        "coordinate_columns": coordinate_columns,
        "coordinate_mismatch": str(config.get("coordinate_mismatch", "error")),
        "sort_tiles_spatially": bool(config.get("sort_tiles_spatially", False)),
        "normalize_coordinates": bool(config.get("normalize_coordinates", False)),
        "expected_feature_dim": int(config["input_dim"]),
        "bag_level": str(config.get("bag_level", "tissue")),
        "tile_sampling": str(config.get("tile_sampling", "random")),
        "feature_normalization": str(config.get("feature_normalization", "none")),
    }
    train_dataset = DGSSMMILTissueDataset(
        split="train",
        max_tiles_per_tissue=config.get("max_tiles_per_tissue_train"),
        skip_tissues_above_tiles=(
            config.get("skip_tissues_above_tiles")
            if apply_training_filter
            else None
        ),
        **dataset_kwargs,
    )
    val_dataset = DGSSMMILTissueDataset(
        split="val",
        max_tiles_per_tissue=config.get("max_tiles_per_tissue_val"),
        **dataset_kwargs,
    )
    test_dataset = DGSSMMILTissueDataset(
        split="test",
        max_tiles_per_tissue=config.get("max_tiles_per_tissue_test"),
        **dataset_kwargs,
    )
    return train_dataset, val_dataset, test_dataset


def create_dataloaders(
    train_dataset: DGSSMMILTissueDataset,
    val_dataset: DGSSMMILTissueDataset,
    config: Dict[str, Any],
    class_weights: torch.Tensor,
    test_dataset: DGSSMMILTissueDataset | None = None,
) -> Tuple[DataLoader, ...]:
    """
    Create train and validation dataloaders.

    Args:
        train_dataset (DGSSMMILTissueDataset): Training split dataset.
        val_dataset (DGSSMMILTissueDataset): Validation split dataset.
        config (Dict[str, Any]): Parsed training configuration.
        class_weights (torch.Tensor): Class weights used for optional sampling.

    Returns:
        Tuple[DataLoader, ...]: Train and validation loaders, plus a test loader
        when `test_dataset` is supplied.
    """
    batch_size = int(config["batch_size"])
    num_workers = int(config.get("num_workers", 0))
    if bool(config.get("use_weighted_sampler", True)):
        sample_weights = compute_sample_weights(train_dataset, class_weights)
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=collate_fn,
            num_workers=num_workers,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=num_workers,
        )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(config.get("evaluation_batch_size", 1)),
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )
    if test_dataset is None:
        return train_loader, val_loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=int(config.get("evaluation_batch_size", 1)),
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )
    return train_loader, val_loader, test_loader


def train_epoch(
    model: DGSSMMILModel,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    gradient_clip_norm: float,
    gradient_accumulation_steps: int = 1,
) -> Dict[str, float]:
    """
    Train DG-SSM-MIL for one epoch.

    Args:
        model (DGSSMMILModel): Model to train.
        dataloader (DataLoader): Training dataloader.
        criterion (nn.Module): Classification loss function.
        optimizer (optim.Optimizer): Optimizer for model parameters.
        device (torch.device): Device used for tensor computation.
        gradient_clip_norm (float): Max norm for gradient clipping; <=0 disables it.
        gradient_accumulation_steps (int): Number of micro-batches to accumulate
            before each optimizer step. Defaults to 1.

    Returns:
        Dict[str, float]: Mean loss, accuracy, and balanced accuracy.
    """
    model.train()
    running_loss = 0.0
    all_preds: List[int] = []
    all_labels: List[int] = []
    all_probs: List[List[float]] = []
    accumulation_steps = max(1, int(gradient_accumulation_steps))
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(dataloader):
        features = batch["features"].to(device)
        coords = batch["coords"].to(device)
        masks = batch["masks"].to(device)
        labels = batch["labels"].to(device)

        tissue_indices = batch.get("tissue_indices")
        if tissue_indices is not None:
            tissue_indices = tissue_indices.to(device)
        outputs = model(features, coords, masks, tissue_indices=tissue_indices)
        loss = criterion(outputs["logits"], labels)

        scaled_loss = loss / accumulation_steps
        scaled_loss.backward()
        is_accumulation_boundary = (
            (batch_idx + 1) % accumulation_steps == 0
            or (batch_idx + 1) == len(dataloader)
        )
        if is_accumulation_boundary:
            if gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            optimizer.step()
            optimizer.zero_grad()

        running_loss += float(loss.item())
        preds = torch.argmax(outputs["logits"], dim=1).detach().cpu().numpy()
        all_preds.extend(int(pred) for pred in preds)
        all_labels.extend(int(label) for label in labels.detach().cpu().numpy())
        all_probs.extend(torch.softmax(outputs["logits"], dim=1).detach().cpu().tolist())

    return _compute_epoch_metrics(
        running_loss, len(dataloader), all_labels, all_preds, all_probs
    )


def validate(
    model: DGSSMMILModel,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Validate DG-SSM-MIL for one epoch.

    Args:
        model (DGSSMMILModel): Model to evaluate.
        dataloader (DataLoader): Validation dataloader.
        criterion (nn.Module): Classification loss function.
        device (torch.device): Device used for tensor computation.

    Returns:
        Dict[str, Any]: Mean loss, metrics, predictions, labels, and confusion matrix.
    """
    model.eval()
    running_loss = 0.0
    all_preds: List[int] = []
    all_labels: List[int] = []
    all_probs: List[List[float]] = []

    with torch.no_grad():
        for batch in dataloader:
            features = batch["features"].to(device)
            coords = batch["coords"].to(device)
            masks = batch["masks"].to(device)
            labels = batch["labels"].to(device)

            tissue_indices = batch.get("tissue_indices")
            if tissue_indices is not None:
                tissue_indices = tissue_indices.to(device)
            outputs = model(features, coords, masks, tissue_indices=tissue_indices)
            loss = criterion(outputs["logits"], labels)
            running_loss += float(loss.item())
            preds = torch.argmax(outputs["logits"], dim=1).detach().cpu().numpy()
            all_preds.extend(int(pred) for pred in preds)
            all_labels.extend(int(label) for label in labels.detach().cpu().numpy())
            all_probs.extend(torch.softmax(outputs["logits"], dim=1).cpu().tolist())

    metrics = _compute_epoch_metrics(
        running_loss, len(dataloader), all_labels, all_preds, all_probs
    )
    metrics["confusion_matrix"] = confusion_matrix(all_labels, all_preds).tolist()
    metrics["predictions"] = all_preds
    metrics["labels"] = all_labels
    return metrics


def _compute_epoch_metrics(
    running_loss: float,
    num_batches: int,
    labels: List[int],
    preds: List[int],
    probabilities: List[List[float]] | None = None,
) -> Dict[str, float]:
    """
    Compute common epoch-level classification metrics.

    Args:
        running_loss (float): Sum of batch losses.
        num_batches (int): Number of processed batches.
        labels (List[int]): Ground-truth class labels.
        preds (List[int]): Predicted class labels.
        probabilities (List[List[float]] | None): Class probabilities.

    Returns:
        Dict[str, float]: Loss, accuracy, and balanced accuracy values.
    """
    if num_batches == 0 or len(labels) == 0:
        return {
            "loss": float("nan"),
            "accuracy": 0.0,
            "balanced_accuracy": 0.0,
            "auc": float("nan"),
        }
    metrics = {
        "loss": running_loss / num_batches,
        "accuracy": float(accuracy_score(labels, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
    }
    metrics["auc"] = _safe_multiclass_auc(labels, probabilities)
    return metrics


def _safe_multiclass_auc(
    labels: List[int],
    probabilities: List[List[float]] | None,
) -> float:
    """
    Compute macro one-vs-rest AUC when every class is represented.

    Args:
        labels (List[int]): Ground-truth class indices.
        probabilities (List[List[float]] | None): Per-class probabilities.

    Returns:
        float: Macro AUC, or NaN when the split cannot define it.
    """
    if not probabilities:
        return float("nan")
    probability_array = np.asarray(probabilities)
    class_aucs = []
    label_array = np.asarray(labels)
    for class_idx in range(probability_array.shape[1]):
        binary_labels = (label_array == class_idx).astype(np.int64)
        if np.unique(binary_labels).size < 2:
            continue
        class_aucs.append(
            roc_auc_score(binary_labels, probability_array[:, class_idx])
        )
    return float(np.mean(class_aucs)) if class_aucs else float("nan")


def is_improved(
    current_value: float,
    best_value: float,
    maximize: bool,
    current_loss: float,
    best_loss: float,
) -> bool:
    """
    Decide whether a checkpoint metric improved.

    Args:
        current_value (float): Current checkpoint metric value.
        best_value (float): Best metric value seen so far.
        maximize (bool): Whether larger metric values are better.
        current_loss (float): Current validation loss used as tie-breaker.
        best_loss (float): Best validation loss among tied checkpoint metrics.

    Returns:
        bool: True when the current epoch should replace the best checkpoint.
    """
    if maximize and current_value > best_value:
        return True
    if not maximize and current_value < best_value:
        return True
    if np.isclose(current_value, best_value) and current_loss < best_loss:
        return True
    return False


def save_checkpoint(
    path: str,
    model: DGSSMMILModel,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.ReduceLROnPlateau,
    epoch: int,
    config: Dict[str, Any],
    metrics: Dict[str, Any],
    class_folders: List[str],
) -> None:
    """
    Save a DG-SSM-MIL checkpoint.

    Args:
        path (str): Destination checkpoint path.
        model (DGSSMMILModel): Model whose state should be saved.
        optimizer (optim.Optimizer): Optimizer state to save.
        scheduler (optim.lr_scheduler.ReduceLROnPlateau): Scheduler state to save.
        epoch (int): One-based epoch number.
        config (Dict[str, Any]): Training configuration.
        metrics (Dict[str, Any]): Validation metrics associated with the checkpoint.
        class_folders (List[str]): Frozen class order used by the model.

    Returns:
        None: Checkpoint is written to disk.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": config,
            "metrics": metrics,
            "class_folders": class_folders,
        },
        path,
    )


def save_json(path: str, payload: Dict[str, Any]) -> None:
    """
    Save a dictionary as formatted JSON.

    Args:
        path (str): Destination JSON path.
        payload (Dict[str, Any]): JSON-serializable data to write.

    Returns:
        None: JSON file is written to disk.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2)


def plot_training_history(
    history: Dict[str, Dict[str, List[float]]],
    best_epoch: int,
    save_path: str,
) -> None:
    """
    Save CLAM-style training curves for loss and classification metrics.

    Args:
        history (Dict[str, Dict[str, List[float]]]): Training history with
            `train` and `val` metric lists.
        best_epoch (int): One-based best checkpoint epoch. A value <= 0 skips
            the best-epoch marker.
        save_path (str): Destination path for the PNG figure.

    Returns:
        None: The plot is saved to disk.
    """
    plot_keys = ["loss", "accuracy", "balanced_accuracy", "auc"]
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(nrows=len(plot_keys), figsize=(10, 12))
    train_handle = None
    val_handle = None
    best_epoch_handle = None

    for axis_idx, key in enumerate(plot_keys):
        train_line, = axes[axis_idx].plot(history["train"][key], label="Train")
        val_line, = axes[axis_idx].plot(history["val"][key], label="Val")
        if best_epoch > 0:
            best_line = axes[axis_idx].axvline(
                x=best_epoch - 1,
                color="red",
                linestyle="--",
                linewidth=1.5,
                label="Best model epoch",
            )
            if best_epoch_handle is None:
                best_epoch_handle = best_line
        if axis_idx == 0:
            train_handle = train_line
            val_handle = val_line
        axes[axis_idx].set_ylabel(key)
        if key == "loss":
            axes[axis_idx].set_yscale("log")

    axes[-1].set_xlabel("Epoch")
    handles = [handle for handle in [train_handle, val_handle, best_epoch_handle] if handle]
    labels = ["Train", "Val"]
    if best_epoch_handle is not None:
        labels.append("Best model epoch")
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=True)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def print_class_counts(
    train_dataset: DGSSMMILTissueDataset,
    val_dataset: DGSSMMILTissueDataset,
) -> None:
    """
    Print class counts for train and validation splits.

    Args:
        train_dataset (DGSSMMILTissueDataset): Training dataset.
        val_dataset (DGSSMMILTissueDataset): Validation dataset.

    Returns:
        None: Counts are printed to stdout.
    """
    train_counts = get_class_sample_counts(train_dataset)
    val_counts = get_class_sample_counts(val_dataset)
    print("Training samples per class:")
    for class_name in train_dataset.class_folders:
        print(f"  {class_name}: {train_counts[class_name]}")
    print("Validation samples per class:")
    for class_name in train_dataset.class_folders:
        print(f"  {class_name}: {val_counts[class_name]}")


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for DG-SSM-MIL training.

    Args:
        None: Arguments are read from the command line.

    Returns:
        argparse.Namespace: Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(description="Train tissue-level DG-SSM-MIL.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.yml. Defaults to dg_ssm_mil/config.yml.",
    )
    parser.add_argument(
        "--repeat-index",
        type=int,
        default=None,
        help="Run one zero-based Monte Carlo repeat instead of orchestrating all repeats.",
    )
    return parser.parse_args()


def _scope_repeat_paths(config: Dict[str, Any], repeat_index: int) -> None:
    """
    Place one Monte Carlo repeat's artifacts in an isolated directory.

    Args:
        config (Dict[str, Any]): Resolved mutable configuration.
        repeat_index (int): Zero-based repeat index.

    Returns:
        None: Output paths are updated in-place.
    """
    repeat_name = f"repeat_{repeat_index:02d}"
    for key, path in config["paths"].items():
        if path is None:
            continue
        parent, filename = os.path.split(str(path))
        config["paths"][key] = os.path.join(parent, repeat_name, filename)


def _orchestrate_repeats(config_path: str | None, repeats: int) -> None:
    """
    Launch all configured Monte Carlo repeats and aggregate their reports.

    Args:
        config_path (str | None): Optional explicit configuration path.
        repeats (int): Number of independent repeat seeds.

    Returns:
        None: Child runs and aggregate JSON output are written to disk.
    """
    command_base = [sys.executable, "-m", "dg_ssm_mil.train"]
    if config_path is not None:
        command_base.extend(["--config", config_path])
    for repeat_index in range(repeats):
        subprocess.run(
            [*command_base, "--repeat-index", str(repeat_index)],
            check=True,
        )
    config = load_config(config_path)
    reports = []
    for repeat_index in range(repeats):
        repeat_config = dict(config)
        repeat_config["paths"] = dict(config["paths"])
        _scope_repeat_paths(repeat_config, repeat_index)
        with open(
            repeat_config["paths"]["training_report"], "r", encoding="utf-8"
        ) as report_file:
            reports.append(json.load(report_file))
    aggregate: Dict[str, Any] = {"repeats": repeats, "reports": reports}
    for split_name in ("val_metrics", "test_metrics"):
        aggregate[split_name] = {}
        for metric_name in ("auc", "accuracy", "balanced_accuracy"):
            values = [
                float(report[split_name][metric_name])
                for report in reports
                if not np.isnan(float(report[split_name][metric_name]))
            ]
            aggregate[split_name][metric_name] = {
                "mean": float(np.mean(values)) if values else float("nan"),
                "std": float(np.std(values)) if values else float("nan"),
            }
    aggregate_path = os.path.join(
        os.path.dirname(config["paths"]["training_report"]),
        "monte_carlo_summary.json",
    )
    save_json(aggregate_path, aggregate)


def main() -> None:
    """
    Run DG-SSM-MIL training from a YAML configuration.

    Args:
        None: Configuration is loaded from CLI arguments.

    Returns:
        None: Training artifacts are written to configured output paths.
    """
    args = parse_args()
    config = load_config(args.config)
    repeats = int(config.get("monte_carlo_repeats", 10))
    if args.repeat_index is None and repeats > 1:
        _orchestrate_repeats(args.config, repeats)
        return
    repeat_index = int(args.repeat_index or 0)
    if repeat_index < 0 or repeat_index >= repeats:
        raise ValueError(f"repeat-index must be in [0, {repeats - 1}].")
    config["repeat_index"] = repeat_index
    config["random_seed"] = int(config["random_seed"]) + repeat_index
    _scope_repeat_paths(config, repeat_index)
    set_random_seeds(int(config["random_seed"]))
    device = torch.device(resolve_device(config))

    print(f"Using device: {device}")
    print(
        f"Using feature model '{config['feature_model']}' "
        f"with suffix '{resolve_feature_file_suffix(config)}'"
    )

    train_dataset, val_dataset, test_dataset = create_datasets(config)
    if len(train_dataset) == 0:
        raise RuntimeError("Training dataset is empty. Check data_root and feature suffix.")
    if len(val_dataset) == 0:
        raise RuntimeError("Validation dataset is empty. Check train_ratio and data layout.")
    if len(test_dataset) == 0:
        raise RuntimeError("Test dataset is empty. Check test_ratio and data layout.")
    if train_dataset.num_classes != int(config["num_classes"]):
        raise ValueError(
            f"Configured num_classes={config['num_classes']} but discovered "
            f"{train_dataset.num_classes} classes: {train_dataset.class_folders}."
        )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Classes: {train_dataset.class_folders}")
    batch_size = int(config["batch_size"])
    gradient_accumulation_steps = int(config.get("gradient_accumulation_steps", 1))
    effective_batch_size = batch_size * gradient_accumulation_steps
    print(f"Physical batch size: {batch_size}")
    print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    print(f"Effective batch size: {effective_batch_size}")
    print(f"Max train tiles per tissue: {config.get('max_tiles_per_tissue_train')}")
    print(f"Max val tiles per tissue: {config.get('max_tiles_per_tissue_val')}")
    print(
        "Skipped oversized tissues: "
        f"{len(train_dataset.skipped_tissues)} "
        f"(threshold={config.get('skip_tissues_above_tiles')})"
    )
    print_class_counts(train_dataset, val_dataset)

    class_weights = compute_class_weights(train_dataset)
    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset,
        val_dataset,
        config,
        class_weights,
        test_dataset,
    )

    model = DGSSMMILModel.from_config(config).to(device)
    print(f"Model parameters: {sum(param.numel() for param in model.parameters()):,}")

    if bool(config.get("use_class_weighted_loss", False)):
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(config.get("lr_scheduler_factor", 0.5)),
        patience=int(config.get("lr_scheduler_patience", 15)),
    )

    metric_key, maximize_metric, metric_name = resolve_best_checkpoint_metric(
        str(config.get("best_checkpoint_metric", "balanced_accuracy"))
    )
    best_metric = -float("inf") if maximize_metric else float("inf")
    best_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    history: Dict[str, Dict[str, List[float]]] = {
        "train": {"loss": [], "accuracy": [], "balanced_accuracy": [], "auc": []},
        "val": {"loss": [], "accuracy": [], "balanced_accuracy": [], "auc": []},
    }

    print(f"Best checkpoint metric: {metric_name}")
    for epoch_idx in tqdm(range(int(config["epochs"]))):
        epoch = epoch_idx + 1
        train_dataset.set_epoch(epoch_idx)
        train_metrics = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            (
                float(config.get("gradient_clip_norm", 1.0))
                if bool(config.get("use_gradient_clipping", False))
                else 0.0
            ),
            gradient_accumulation_steps,
        )
        val_metrics = validate(model, val_loader, criterion, device)
        if bool(config.get("use_lr_scheduler", False)):
            scheduler.step(float(val_metrics["loss"]))

        for split_name, metrics in (("train", train_metrics), ("val", val_metrics)):
            for key in ["loss", "accuracy", "balanced_accuracy", "auc"]:
                history[split_name][key].append(float(metrics[key]))

        tqdm.write(
            f"Epoch {epoch}: "
            f"loss {train_metrics['loss']:.4f}/{val_metrics['loss']:.4f}, "
            f"acc {train_metrics['accuracy']:.4f}/{val_metrics['accuracy']:.4f}, "
            "bal_acc "
            f"{train_metrics['balanced_accuracy']:.4f}/"
            f"{val_metrics['balanced_accuracy']:.4f}"
            f", auc {train_metrics['auc']:.4f}/{val_metrics['auc']:.4f}"
        )

        current_metric = float(val_metrics[metric_key])
        if is_improved(
            current_metric,
            best_metric,
            maximize_metric,
            float(val_metrics["loss"]),
            best_loss,
        ):
            best_metric = current_metric
            best_loss = float(val_metrics["loss"])
            best_epoch = epoch
            patience_counter = 0
            save_checkpoint(
                config["paths"]["checkpoint"],
                model,
                optimizer,
                scheduler,
                epoch,
                config,
                val_metrics,
                train_dataset.class_folders,
            )
            report = classification_report(
                val_metrics["labels"],
                val_metrics["predictions"],
                target_names=train_dataset.class_folders,
                labels=list(range(len(train_dataset.class_folders))),
                output_dict=True,
                zero_division=0,
            )
            save_json(
                os.path.join(
                    os.path.dirname(config["paths"]["checkpoint"]),
                    "best_model_report.json",
                ),
                report,
            )
            save_json(
                config["paths"]["training_report"],
                {
                    "best_epoch": best_epoch,
                    "best_checkpoint_metric": metric_name,
                    "best_metric_value": best_metric,
                    "best_val_loss": best_loss,
                    "val_metrics": val_metrics,
                    "class_folders": train_dataset.class_folders,
                },
            )
        else:
            patience_counter += 1

        min_epochs = int(config.get("min_epochs_before_early_stopping", 0))
        if epoch >= min_epochs and patience_counter >= int(config.get("patience", 50)):
            print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}.")
            break

    save_checkpoint(
        config["paths"]["final_checkpoint"],
        model,
        optimizer,
        scheduler,
        epoch,
        config,
        val_metrics,
        train_dataset.class_folders,
    )
    best_checkpoint = torch.load(
        config["paths"]["checkpoint"], map_location=device, weights_only=False
    )
    model.load_state_dict(best_checkpoint["model_state_dict"])
    test_metrics = validate(model, test_loader, criterion, device)
    save_json(
        config["paths"]["training_report"],
        {
            "repeat_index": repeat_index,
            "random_seed": int(config["random_seed"]),
            "best_epoch": best_epoch,
            "best_checkpoint_metric": metric_name,
            "best_metric_value": best_metric,
            "best_val_loss": best_loss,
            "val_metrics": best_checkpoint["metrics"],
            "test_metrics": test_metrics,
            "class_folders": train_dataset.class_folders,
        },
    )
    save_json(
        config["paths"]["training_history"],
        {
            "history": history,
            "best_epoch": best_epoch,
            "best_checkpoint_metric": metric_name,
            "best_metric_value": best_metric,
            "class_folders": train_dataset.class_folders,
        },
    )
    plot_training_history(
        history=history,
        best_epoch=best_epoch,
        save_path=config["paths"]["training_plot"],
    )
    print(f"Training complete. Best epoch: {best_epoch}.")


if __name__ == "__main__":
    main()
