"""Train canonical CLAM-SB or CLAM-MB models on tissue or slide bags."""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)
from torch import nn, optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

try:
    from .clam_dataset import WSIBagDataset, collate_fn, create_bag_dataset
    from .clam_model import CLAM_MB, CLAM_SB
    from .config_loader import load_config
except ImportError:
    from clam_dataset import WSIBagDataset, collate_fn, create_bag_dataset
    from clam_model import CLAM_MB, CLAM_SB
    from config_loader import load_config


MODEL_SCHEMA = "canonical_clam_v1"
METRIC_KEYS = (
    "loss",
    "classification_loss",
    "instance_loss",
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
)


def create_model(config: Mapping[str, Any]) -> nn.Module:
    """Create the configured canonical CLAM architecture.

    Args:
        config (Mapping[str, Any]): Resolved configuration containing model
            architecture values.

    Returns:
        nn.Module: A ``CLAM_SB`` or ``CLAM_MB`` instance.
    """
    model_type = str(config["model_type"]).lower()
    model_classes = {"clam_sb": CLAM_SB, "clam_mb": CLAM_MB}
    if model_type not in model_classes:
        raise ValueError("model_type must be 'clam_sb' or 'clam_mb'.")
    return model_classes[model_type](
        input_dim=int(config["input_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        attention_dim=int(config["attention_dim"]),
        num_classes=int(config["num_classes"]),
        gated=bool(config["gated_attention"]),
        dropout=float(config["dropout"]),
        k_sample=int(config["k_sample"]),
        subtyping=bool(config["subtyping"]),
    )


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch reproducibly.

    Args:
        seed (int): Global nonnegative random seed.

    Returns:
        None: Random generators and deterministic backend settings are updated.
    """
    if seed < 0:
        raise ValueError("random_seed must be nonnegative.")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """Seed one data-loading worker from PyTorch's worker seed.

    Args:
        worker_id (int): DataLoader worker identifier.

    Returns:
        None: Python and NumPy worker generators are seeded.
    """
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def get_class_sample_counts(dataset: WSIBagDataset) -> Dict[str, int]:
    """Count configured bag units per class in one split.

    Args:
        dataset (WSIBagDataset): Tissue- or slide-level split dataset.

    Returns:
        Dict[str, int]: Counts ordered by ``dataset.class_folders``.
    """
    counts = {class_name: 0 for class_name in dataset.class_folders}
    for bag_index in dataset.indices:
        counts[dataset._bags[bag_index]["class_name"]] += 1
    return counts


def compute_class_weights(dataset: WSIBagDataset) -> torch.Tensor:
    """Compute inverse-frequency CE weights from configured bag units.

    Args:
        dataset (WSIBagDataset): Training dataset at its configured bag level.

    Returns:
        torch.Tensor: Float weights shaped ``[num_classes]``.
    """
    counts = get_class_sample_counts(dataset)
    missing = [name for name, count in counts.items() if count == 0]
    if missing:
        raise ValueError(
            "Training split has no configured bag units for class(es): "
            + ", ".join(missing)
        )
    total = sum(counts.values())
    return torch.tensor(
        [
            total / (dataset.num_classes * counts[class_name])
            for class_name in dataset.class_folders
        ],
        dtype=torch.float32,
    )


def compute_sample_weights(
    dataset: WSIBagDataset,
    class_weights: torch.Tensor,
) -> List[float]:
    """Map class weights to split-local configured bag units.

    Args:
        dataset (WSIBagDataset): Training dataset.
        class_weights (torch.Tensor): Class weights shaped ``[num_classes]``.

    Returns:
        List[float]: Sample weights aligned with split-local dataset indices.
    """
    if class_weights.numel() != dataset.num_classes:
        raise ValueError("class_weights length must equal dataset.num_classes.")
    return [
        float(
            class_weights[
                dataset.class_to_idx[dataset._bags[bag_index]["class_name"]]
            ].item()
        )
        for bag_index in dataset.indices
    ]


def resolve_best_checkpoint_metric(metric_name: str) -> Tuple[str, bool, str]:
    """Resolve a configured checkpoint metric and optimization direction.

    Args:
        metric_name (str): One of the checkpoint options documented in config.

    Returns:
        Tuple[str, bool, str]: Metric key, maximize flag, and normalized name.
    """
    normalized = str(metric_name).strip().lower()
    directions = {
        "balanced_accuracy": True,
        "accuracy": True,
        "loss": False,
        "classification_loss": False,
        "instance_loss": False,
    }
    if normalized not in directions:
        raise ValueError(
            f"Invalid best_checkpoint_metric '{metric_name}'. Valid options: "
            + ", ".join(directions)
        )
    return normalized, directions[normalized], normalized


def _validate_training_config(config: Mapping[str, Any]) -> None:
    """Validate training-only values not covered by the config loader.

    Args:
        config (Mapping[str, Any]): Resolved training configuration.

    Returns:
        None: Validation succeeds by returning normally.
    """
    for key in ("batch_size", "epochs", "gradient_accumulation_steps"):
        value = config.get(key, 1 if key == "gradient_accumulation_steps" else None)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{key} must be a positive integer.")
    for key in ("lr_cls", "weight_decay_cls"):
        value = float(config[key])
        if not np.isfinite(value) or value < 0.0 or (key == "lr_cls" and value == 0.0):
            raise ValueError(f"{key} has an invalid value: {value}.")
    if int(config["num_classes"]) < 2:
        raise ValueError("num_classes must be at least 2.")


def _make_loader(
    dataset: WSIBagDataset,
    config: Mapping[str, Any],
    training: bool,
    class_weights: Optional[torch.Tensor] = None,
) -> DataLoader:
    """Build a deterministic loader, optionally with weighted sampling.

    Args:
        dataset (WSIBagDataset): Dataset to load.
        config (Mapping[str, Any]): Training configuration.
        training (bool): Whether to enable training order and sampling.
        class_weights (Optional[torch.Tensor]): Training class weights.

    Returns:
        DataLoader: Configured bag DataLoader.
    """
    generator = torch.Generator().manual_seed(int(config["random_seed"]))
    sampler = None
    shuffle = training
    if training and bool(config.get("use_weighted_sampler", False)):
        if class_weights is None:
            raise ValueError("class_weights are required for weighted sampling.")
        sample_weights = compute_sample_weights(dataset, class_weights)
        sampler = WeightedRandomSampler(
            sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
            generator=generator,
        )
        shuffle = False
    return DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=shuffle,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=int(config.get("num_workers", 0)),
        worker_init_fn=seed_worker,
        generator=generator,
        pin_memory=torch.cuda.is_available(),
    )


def _classification_metrics(
    labels: Sequence[int],
    predictions: Sequence[int],
    num_classes: int,
) -> Dict[str, Any]:
    """Compute fixed-label classification metrics.

    Args:
        labels (Sequence[int]): Ground-truth class indices.
        predictions (Sequence[int]): Predicted class indices.
        num_classes (int): Total configured class count.

    Returns:
        Dict[str, Any]: Accuracy, balanced accuracy, macro F1, and fixed matrix.
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
        "confusion_matrix": confusion_matrix(
            labels, predictions, labels=fixed_labels
        ).tolist(),
    }


def _run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    bag_weight: float,
    optimizer: Optional[optim.Optimizer] = None,
    gradient_accumulation_steps: int = 1,
) -> Dict[str, Any]:
    """Run one canonical CLAM training or validation epoch.

    Args:
        model (nn.Module): Canonical CLAM model.
        dataloader (DataLoader): Bag DataLoader.
        criterion (nn.Module): Bag-level cross-entropy criterion.
        device (torch.device): Compute device.
        bag_weight (float): Classification-loss mixture weight.
        optimizer (Optional[optim.Optimizer]): Optimizer, or ``None`` for validation.
        gradient_accumulation_steps (int): Number of micro-batches per update.

    Returns:
        Dict[str, Any]: Losses, classification metrics, matrix, labels, and predictions.
    """
    if len(dataloader.dataset) == 0:
        raise ValueError("Cannot run an epoch on an empty dataset split.")
    training = optimizer is not None
    model.train(training)
    accumulation_steps = max(1, int(gradient_accumulation_steps))
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)

    totals = {"loss": 0.0, "classification_loss": 0.0, "instance_loss": 0.0}
    labels_all: List[int] = []
    predictions_all: List[int] = []
    sample_count = 0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch_index, batch in enumerate(dataloader):
            features = batch["features"].to(device, non_blocking=True)
            masks = batch["masks"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            outputs = model(features, mask=masks, labels=labels, instance_eval=True)
            classification_loss = criterion(outputs["logits"], labels)
            instance_loss = outputs["instance_loss"]
            loss = bag_weight * classification_loss + (1.0 - bag_weight) * instance_loss

            if optimizer is not None:
                group_start = (batch_index // accumulation_steps) * accumulation_steps
                group_size = min(accumulation_steps, len(dataloader) - group_start)
                (loss / group_size).backward()
                boundary = (
                    (batch_index + 1) % accumulation_steps == 0
                    or batch_index + 1 == len(dataloader)
                )
                if boundary:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            batch_size = int(labels.shape[0])
            sample_count += batch_size
            totals["loss"] += float(loss.detach().item()) * batch_size
            totals["classification_loss"] += (
                float(classification_loss.detach().item()) * batch_size
            )
            totals["instance_loss"] += float(instance_loss.detach().item()) * batch_size
            labels_all.extend(labels.detach().cpu().tolist())
            predictions_all.extend(outputs["predictions"].detach().cpu().tolist())

    metrics: Dict[str, Any] = {
        key: value / sample_count for key, value in totals.items()
    }
    metrics.update(
        _classification_metrics(labels_all, predictions_all, model.num_classes)
    )
    metrics["labels"] = labels_all
    metrics["predictions"] = predictions_all
    return metrics


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    bag_weight: float,
    gradient_accumulation_steps: int = 1,
) -> Dict[str, Any]:
    """Train a canonical CLAM model for one epoch.

    Args:
        model (nn.Module): Canonical CLAM model.
        dataloader (DataLoader): Training bag DataLoader.
        criterion (nn.Module): Bag-level cross-entropy criterion.
        optimizer (optim.Optimizer): Single Adam optimizer.
        device (torch.device): Compute device.
        bag_weight (float): Classification-loss mixture weight.
        gradient_accumulation_steps (int): Micro-batches per optimizer update.

    Returns:
        Dict[str, Any]: Canonical epoch metrics and predictions.
    """
    return _run_epoch(
        model,
        dataloader,
        criterion,
        device,
        bag_weight,
        optimizer,
        gradient_accumulation_steps,
    )


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    bag_weight: float,
) -> Dict[str, Any]:
    """Validate a canonical CLAM model for one epoch.

    Args:
        model (nn.Module): Canonical CLAM model.
        dataloader (DataLoader): Validation bag DataLoader.
        criterion (nn.Module): Bag-level cross-entropy criterion.
        device (torch.device): Compute device.
        bag_weight (float): Classification-loss mixture weight.

    Returns:
        Dict[str, Any]: Canonical epoch metrics and predictions.
    """
    return _run_epoch(model, dataloader, criterion, device, bag_weight)


def _checkpoint_payload(
    checkpoint_type: str,
    epoch: int,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.ReduceLROnPlateau,
    config: Mapping[str, Any],
    class_folders: Sequence[str],
    history: Mapping[str, Any],
    best_metric: Mapping[str, Any],
) -> Dict[str, Any]:
    """Create a self-contained canonical checkpoint.

    Args:
        checkpoint_type (str): ``best`` or ``final``.
        epoch (int): One-based completed epoch.
        model (nn.Module): Trained model.
        optimizer (optim.Optimizer): Single Adam optimizer.
        scheduler (optim.lr_scheduler.ReduceLROnPlateau): Plateau scheduler.
        config (Mapping[str, Any]): Exact resolved training configuration.
        class_folders (Sequence[str]): Ordered class names.
        history (Mapping[str, Any]): Complete metric history.
        best_metric (Mapping[str, Any]): Best-selection metadata.

    Returns:
        Dict[str, Any]: Serializable checkpoint payload.
    """
    return {
        "model_schema": MODEL_SCHEMA,
        "checkpoint_type": checkpoint_type,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "config": copy.deepcopy(dict(config)),
        "class_folders": list(class_folders),
        "bag_level": str(config["bag_level"]),
        "history": copy.deepcopy(dict(history)),
        "best_metric": copy.deepcopy(dict(best_metric)),
    }


def plot_history(
    history: Mapping[str, Mapping[str, Sequence[float]]],
    save_path: str,
    best_epoch: int,
) -> None:
    """Plot canonical train and validation history.

    Args:
        history (Mapping[str, Mapping[str, Sequence[float]]]): Metric history.
        save_path (str): Destination PNG path.
        best_epoch (int): One-based best epoch.

    Returns:
        None: The plot is written to disk.
    """
    figure, axes = plt.subplots(len(METRIC_KEYS), 1, figsize=(10, 20))
    for axis, key in zip(axes, METRIC_KEYS):
        axis.plot(history["train"][key], label="train")
        axis.plot(history["val"][key], label="val")
        axis.axvline(best_epoch - 1, color="red", linestyle="--", label="best")
        axis.set_ylabel(key)
        axis.grid(True, alpha=0.3)
        axis.legend()
    axes[-1].set_xlabel("epoch")
    figure.tight_layout()
    figure.savefig(save_path, dpi=200)
    plt.close(figure)


def train(config_path: Optional[str] = None) -> Dict[str, str]:
    """Run canonical CLAM training and save best and final checkpoints.

    Args:
        config_path (Optional[str]): YAML path, or ``None`` for the module default.

    Returns:
        Dict[str, str]: Paths to best checkpoint, final checkpoint, and history.
    """
    config = load_config(config_path)
    _validate_training_config(config)
    seed_everything(int(config["random_seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = create_bag_dataset(config, "train")
    class_folders = list(train_dataset.class_folders)
    if int(config["num_classes"]) != train_dataset.num_classes:
        raise ValueError(
            f"Configured num_classes={config['num_classes']} does not equal "
            f"dataset classes={train_dataset.num_classes}: {class_folders}."
        )
    val_dataset = create_bag_dataset(config, "val", class_folders=class_folders)
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError("Training and validation splits must both contain bags.")

    class_weights = compute_class_weights(train_dataset)
    train_loader = _make_loader(train_dataset, config, True, class_weights)
    val_loader = _make_loader(val_dataset, config, False)
    model = create_model(config).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device)
        if bool(config.get("use_class_weighted_loss", False))
        else None
    )
    optimizer = optim.Adam(
        model.parameters(),
        lr=float(config["lr_cls"]),
        weight_decay=float(config["weight_decay_cls"]),
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(config.get("lr_scheduler_factor_cls", 0.5)),
        patience=int(config.get("lr_scheduler_patience_cls", 10)),
    )

    metric_key, maximize, metric_name = resolve_best_checkpoint_metric(
        str(config.get("best_checkpoint_metric", "balanced_accuracy"))
    )
    best_value = -float("inf") if maximize else float("inf")
    best_epoch = 0
    patience_counter = 0
    history: Dict[str, Dict[str, List[float]]] = {
        split: {key: [] for key in METRIC_KEYS} for split in ("train", "val")
    }
    checkpoint_dir = Path(config["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / "best_model.pth"
    final_path = checkpoint_dir / "final_model.pth"
    bag_weight = float(config["bag_weight"])
    last_epoch = 0

    for epoch_index in tqdm(range(int(config["epochs"])), desc="CLAM training"):
        epoch = epoch_index + 1
        last_epoch = epoch
        train_dataset.set_epoch(epoch_index)
        train_metrics = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            bag_weight,
            int(config.get("gradient_accumulation_steps", 1)),
        )
        val_metrics = validate(model, val_loader, criterion, device, bag_weight)
        scheduler.step(float(val_metrics["loss"]))
        for split, metrics in (("train", train_metrics), ("val", val_metrics)):
            for key in METRIC_KEYS:
                history[split][key].append(float(metrics[key]))

        current = float(val_metrics[metric_key])
        improved = current > best_value if maximize else current < best_value
        if improved:
            best_value = current
            best_epoch = epoch
            patience_counter = 0
            best_metadata = {
                "name": metric_name,
                "mode": "max" if maximize else "min",
                "value": best_value,
                "epoch": best_epoch,
            }
            torch.save(
                _checkpoint_payload(
                    "best",
                    epoch,
                    model,
                    optimizer,
                    scheduler,
                    config,
                    class_folders,
                    history,
                    best_metadata,
                ),
                best_path,
            )
        else:
            patience_counter += 1

        tqdm.write(
            f"epoch={epoch} "
            f"train_loss={train_metrics['loss']:.5f} "
            f"val_loss={val_metrics['loss']:.5f} "
            f"train_acc={train_metrics['accuracy']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"train_bal_acc={train_metrics['balanced_accuracy']:.4f} "
            f"val_bal_acc={val_metrics['balanced_accuracy']:.4f} "
            f"train_f1={train_metrics['macro_f1']:.4f} "
            f"val_f1={val_metrics['macro_f1']:.4f}"
        )
        if (
            epoch >= int(config.get("min_epochs_before_early_stopping", 0))
            and patience_counter >= int(config["patience"])
        ):
            break

    best_metadata = {
        "name": metric_name,
        "mode": "max" if maximize else "min",
        "value": best_value,
        "epoch": best_epoch,
    }
    torch.save(
        _checkpoint_payload(
            "final",
            last_epoch,
            model,
            optimizer,
            scheduler,
            config,
            class_folders,
            history,
            best_metadata,
        ),
        final_path,
    )
    history_path = checkpoint_dir / "training_history.json"
    with history_path.open("w", encoding="utf-8") as history_file:
        json.dump(history, history_file, indent=2)
    plot_history(history, str(checkpoint_dir / "training_history.png"), best_epoch)
    return {
        "best_checkpoint": str(best_path),
        "final_checkpoint": str(final_path),
        "history": str(history_path),
    }


def main() -> None:
    """Run training with the default configuration.

    Args:
        None: This entry point takes no arguments.

    Returns:
        None: Training artifacts are written to configured paths.
    """
    artifacts = train()
    print(json.dumps(artifacts, indent=2))


if __name__ == "__main__":
    main()
