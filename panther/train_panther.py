"""Train the independent, original-style PANTHER classification pipeline."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .config_loader import allocate_training_run, load_config
from .metrics import classification_metrics
from .panther_dataset import (
    BAG_LEVELS,
    PantherBagDataset,
    build_datasets,
    class_counts,
    save_split_manifest,
)
from .panther_model import LinearClassifier, PANTHER
from .prototype import fit_prototypes


MODEL_SCHEMA = "panther-original-v3"
EMBEDDING_COMPONENT_KEYS = {
    "pi": "mixture_weights",
    "mean": "means",
    "variance": "variances",
}
HISTORY_METRIC_KEYS = (
    "loss",
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "multiclass_roc_auc",
)
ROOT_SELECTION_METRIC = "val_balanced_accuracy"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(config: Mapping[str, Any]) -> torch.device:
    requested = str(config["runtime"]["device"])
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("runtime.device is cuda, but CUDA is unavailable.")
    return torch.device(
        "cuda" if requested == "auto" and torch.cuda.is_available() else
        "cpu" if requested == "auto" else requested
    )


def create_panther(
    prototypes: torch.Tensor,
    config: Mapping[str, Any],
    output_type: str = "allcat",
) -> PANTHER:
    settings = config["model"]
    return PANTHER(
        prototypes=prototypes,
        em_iterations=int(settings["em_iterations"]),
        tau=float(settings["tau"]),
        covariance_regularizer=float(settings["covariance_regularizer"]),
        variance_floor=float(settings["variance_floor"]),
        output_type=output_type,
        fix_prototypes=bool(settings["fix_prototypes"]),
        em_chunk_size=settings.get("em_chunk_size"),
    )


def component_embedding_paths(
    directory: Path, bag_level: str = "slide"
) -> Dict[str, Path]:
    """Return the three persistent component-cache paths for one bag level."""
    if bag_level not in BAG_LEVELS:
        raise ValueError(f"bag_level must be one of {BAG_LEVELS}, received {bag_level}.")
    return {
        component: directory / f"{bag_level}_embeddings_{component}.pt"
        for component in EMBEDDING_COMPONENT_KEYS
    }


def compose_slide_embeddings(
    component_embeddings: Mapping[str, Mapping[str, Mapping[str, Any]]],
    output_type: str,
) -> Dict[str, Dict[str, Any]]:
    """Select one component or concatenate all components in canonical order."""
    if output_type != "allcat" and output_type not in EMBEDDING_COMPONENT_KEYS:
        raise ValueError(f"Unsupported PANTHER output type: {output_type}.")

    reference = component_embeddings["pi"]
    selected_components = (
        tuple(EMBEDDING_COMPONENT_KEYS)
        if output_type == "allcat"
        else (output_type,)
    )
    composed: Dict[str, Dict[str, Any]] = {}
    for split, reference_cache in reference.items():
        representations = []
        for component in selected_components:
            cache = component_embeddings[component][split]
            if cache["slide_keys"] != reference_cache["slide_keys"]:
                raise ValueError(
                    f"{component} embedding cache does not match the {split} slides."
                )
            if not torch.equal(cache["labels"], reference_cache["labels"]):
                raise ValueError(
                    f"{component} embedding cache does not match the {split} labels."
                )
            representations.append(cache["representations"])
        composed[split] = dict(reference_cache)
        composed[split]["representations"] = (
            representations[0]
            if len(representations) == 1
            else torch.cat(representations, dim=1)
        )
    return composed


@torch.no_grad()
def aggregate_slide_embeddings(
    datasets: Mapping[str, PantherBagDataset],
    panther: PANTHER,
    device: torch.device,
    destination: Path,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Encode every bag at one level and persist its PANTHER components."""
    levels = {dataset.bag_level for dataset in datasets.values()}
    if len(levels) != 1:
        raise ValueError(f"All datasets must use one bag level, received {levels}.")
    bag_level = levels.pop()
    panther.eval().to(device)
    payloads: Dict[str, Dict[str, Dict[str, Any]]] = {
        component: {} for component in EMBEDDING_COMPONENT_KEYS
    }
    for split, dataset in datasets.items():
        started = time.time()
        representations = {
            component: [] for component in EMBEDDING_COMPONENT_KEYS
        }
        labels = []
        slide_names = []
        slide_keys = []
        bag_names = []
        bag_keys = []
        tissue_names = []
        tile_counts = []
        for index in range(len(dataset)):
            item = dataset[index]
            features = item["features"].to(device, non_blocking=True)
            encoded = panther(features)
            for component, result_key in EMBEDDING_COMPONENT_KEYS.items():
                result = encoded[result_key].squeeze(0).flatten().cpu()
                if not torch.isfinite(result).all():
                    raise FloatingPointError(
                        f"Non-finite PANTHER {component} for {item['slide_key']}."
                    )
                representations[component].append(result)
            labels.append(int(item["label"]))
            slide_names.append(str(item["slide_name"]))
            slide_keys.append(str(item["slide_key"]))
            bag_names.append(str(item["bag_name"]))
            bag_keys.append(str(item["bag_key"]))
            tissue_names.append(list(item["tissue_names"]))
            tile_counts.append(int(item["num_tiles"]))
            if (index + 1) % 25 == 0 or index + 1 == len(dataset):
                print(
                    f"PANTHER aggregation {bag_level}/{split}: "
                    f"{index + 1}/{len(dataset)} bags"
                )
        metadata = {
            "bag_level": bag_level,
            "bag_names": bag_names,
            "bag_keys": bag_keys,
            "labels": torch.tensor(labels, dtype=torch.long),
            "slide_names": slide_names,
            "slide_keys": slide_keys,
            "tissue_names": tissue_names,
            "tile_counts": tile_counts,
            "elapsed_seconds": float(time.time() - started),
        }
        for component in EMBEDDING_COMPONENT_KEYS:
            payloads[component][split] = {
                "representations": torch.stack(representations[component]),
                **metadata,
            }

    destination.mkdir(parents=True, exist_ok=True)
    for component, path in component_embedding_paths(destination, bag_level).items():
        torch.save(payloads[component], path)
    return payloads


def _class_weights(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    counts = torch.bincount(labels, minlength=num_classes).to(torch.float32)
    return labels.numel() / (num_classes * counts.clamp_min(1.0))


def _make_optimizer(
    model: nn.Module, settings: Mapping[str, Any]
) -> torch.optim.Optimizer:
    arguments = {
        "lr": float(settings["learning_rate"]),
        "weight_decay": float(settings["weight_decay"]),
    }
    if settings["optimizer"] == "adamw":
        return torch.optim.AdamW(model.parameters(), **arguments)
    return torch.optim.SGD(model.parameters(), momentum=0.9, **arguments)


def _make_scheduler(
    optimizer: torch.optim.Optimizer,
    settings: Mapping[str, Any],
    steps_per_epoch: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    total_steps = max(1, int(settings["epochs"]) * steps_per_epoch)
    warmup_steps = int(settings["warmup_epochs"]) * steps_per_epoch
    mode = str(settings["scheduler"])

    def multiplier(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        if mode == "constant":
            return 1.0
        if mode == "linear":
            return 1.0 - progress
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def _classifier_pass(
    model: LinearClassifier,
    representations: torch.Tensor,
    labels: torch.Tensor,
    class_names: Sequence[str],
    device: torch.device,
    batch_size: int,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    accumulation_steps: int = 1,
    input_dropout: float = 0.0,
) -> Dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    loader = DataLoader(
        TensorDataset(representations, labels),
        batch_size=batch_size,
        shuffle=False,
    )
    if training:
        optimizer.zero_grad(set_to_none=True)
    probabilities = []
    labels_all = []
    loss_total = 0.0
    seen = 0
    for batch_index, (features, target) in enumerate(loader):
        features = features.to(device)
        target = target.to(device)
        if training and input_dropout:
            features = F.dropout(features, p=input_dropout, training=True)
        with torch.set_grad_enabled(training):
            logits = model(features)
            loss = loss_fn(logits, target)
        if training:
            (loss / accumulation_steps).backward()
            if (batch_index + 1) % accumulation_steps == 0 or batch_index + 1 == len(loader):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()
        probabilities.append(torch.softmax(logits.detach(), dim=1).cpu())
        labels_all.extend(target.cpu().tolist())
        loss_total += float(loss.detach()) * len(target)
        seen += len(target)
    metrics = classification_metrics(
        labels_all, torch.cat(probabilities).numpy(), class_names
    )
    # Keep history compact; full reports are generated by standalone evaluation.
    metrics.pop("classification_report")
    metrics.pop("confusion_matrix")
    metrics.pop("predictions")
    metrics["loss"] = loss_total / seen
    return metrics


def train_classifier(
    embeddings: Mapping[str, Mapping[str, Any]],
    class_names: Sequence[str],
    config: Mapping[str, Any],
    device: torch.device,
    model_dir: Path,
    validation_embeddings: Mapping[
        str, Mapping[str, Mapping[str, Any]]
    ] | None = None,
) -> tuple[LinearClassifier, list[Dict[str, Any]], Dict[str, Any]]:
    settings = config["training"]
    training_level = str(settings.get("bag_level", "slide"))
    checkpoint_level = str(settings.get("checkpoint_level", training_level))
    if training_level not in BAG_LEVELS or checkpoint_level not in BAG_LEVELS:
        raise ValueError(
            f"Training/checkpoint levels must be in {BAG_LEVELS}; received "
            f"{training_level}/{checkpoint_level}."
        )
    validation_by_level = (
        dict(validation_embeddings)
        if validation_embeddings is not None
        else {level: embeddings for level in BAG_LEVELS}
    )
    missing_levels = [
        level for level in BAG_LEVELS if level not in validation_by_level
    ]
    if missing_levels:
        raise ValueError(f"Missing validation embeddings for levels: {missing_levels}.")
    model_dir.mkdir(parents=True, exist_ok=True)
    train_data = embeddings["train"]
    input_dim = int(train_data["representations"].shape[1])
    model = LinearClassifier(
        input_dim, len(class_names), bias=bool(settings["classifier_bias"])
    ).to(device)
    optimizer = _make_optimizer(model, settings)
    batches = math.ceil(len(train_data["labels"]) / int(settings["batch_size"]))
    optimizer_steps = math.ceil(
        batches / int(settings["gradient_accumulation_steps"])
    )
    scheduler = _make_scheduler(optimizer, settings, optimizer_steps)
    weights = None
    if bool(settings["class_weighted_loss"]):
        weights = _class_weights(train_data["labels"], len(class_names)).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)

    history: list[Dict[str, Any]] = []
    selection = str(settings["checkpoint_selection"])
    metric_key, maximize = _checkpoint_metric(selection)
    best_value = -float("inf") if maximize else float("inf")
    best_loss = float("inf")
    best_loss_epoch = -1
    best_state: Dict[str, torch.Tensor] | None = None
    best_epoch = -1
    for epoch in range(int(settings["epochs"])):
        train_metrics = _classifier_pass(
            model,
            train_data["representations"],
            train_data["labels"],
            class_names,
            device,
            int(settings["batch_size"]),
            loss_fn,
            optimizer,
            scheduler,
            int(settings["gradient_accumulation_steps"]),
            float(settings["input_dropout"]),
        )
        validation_metrics: Dict[str, Dict[str, Any]] = {}
        for level in BAG_LEVELS:
            val_data = validation_by_level[level]["val"]
            validation_metrics[level] = _classifier_pass(
                model,
                val_data["representations"],
                val_data["labels"],
                class_names,
                device,
                int(settings["batch_size"]),
                loss_fn,
            )
        selected_validation = validation_metrics[checkpoint_level]
        entry = {
            "epoch": epoch + 1,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train": train_metrics,
            "val_tissue": validation_metrics["tissue"],
            "val_slide": validation_metrics["slide"],
            # Compatibility alias: validation at the configured checkpoint level.
            "val": selected_validation,
        }
        history.append(entry)
        current = float(selected_validation[metric_key])
        current_loss = float(selected_validation["loss"])
        if current_loss < best_loss:
            best_loss = current_loss
            best_loss_epoch = epoch + 1
        improved = current > best_value if maximize else current < best_value
        if improved:
            best_value = current
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
        print(
            f"epoch={epoch + 1} | "
            f"loss={train_metrics['loss']:.5f}, "
            f"{validation_metrics['tissue']['loss']:.5f}, "
            f"{validation_metrics['slide']['loss']:.5f} | "
            f"acc={train_metrics['accuracy']:.4f}, "
            f"{validation_metrics['tissue']['accuracy']:.4f}, "
            f"{validation_metrics['slide']['accuracy']:.4f} | "
            f"bal_acc={train_metrics['balanced_accuracy']:.4f}, "
            f"{validation_metrics['tissue']['balanced_accuracy']:.4f}, "
            f"{validation_metrics['slide']['balanced_accuracy']:.4f} | "
            f"f1={train_metrics['macro_f1']:.4f}, "
            f"{validation_metrics['tissue']['macro_f1']:.4f}, "
            f"{validation_metrics['slide']['macro_f1']:.4f}"
        )

    final_state = copy.deepcopy(model.state_dict())
    torch.save(
        {"model_state_dict": final_state, "epoch": int(settings["epochs"])},
        model_dir / "final_model.pth",
    )
    assert best_state is not None
    torch.save(
        {"model_state_dict": best_state, "epoch": best_epoch},
        model_dir / "best_validation_model.pth",
    )
    selected_state = final_state if selection == "last" else best_state
    selected_epoch = int(settings["epochs"]) if selection == "last" else best_epoch
    model.load_state_dict(selected_state)
    details = {
        "input_dim": input_dim,
        "best_validation_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "best_validation_loss_epoch": best_loss_epoch,
        "best_checkpoint_value": best_value,
        "best_checkpoint_metric": metric_key,
        "training_level": training_level,
        "checkpoint_level": checkpoint_level,
        "selected_epoch": selected_epoch,
        "checkpoint_selection": selection,
    }
    return model, history, details


def _checkpoint_metric(selection: str) -> tuple[str, bool]:
    """Resolve a checkpoint-selection mode to its metric key and direction."""
    if selection in ("last", "best_val_loss"):
        return "loss", False
    prefix = "best_val_"
    if selection.startswith(prefix):
        return selection[len(prefix):], True
    raise ValueError(f"Unsupported checkpoint selection: {selection}.")


def select_primary_output_type(
    trained_models: Mapping[str, Mapping[str, Any]],
    training_histories: Mapping[str, Sequence[Mapping[str, Any]]],
    output_types: Sequence[str],
) -> tuple[str, Dict[str, Any]]:
    """Pick the representation whose selected epoch has the highest val balanced accuracy.

    Ties prefer the lower validation loss at that epoch, then the earlier entry in
    ``output_types``. Test metrics are never consulted.

    Args:
        trained_models (Mapping[str, Mapping[str, Any]]): Per-output payloads that
            include ``training_details`` with ``selected_epoch``.
        training_histories (Mapping[str, Sequence[Mapping[str, Any]]]): Per-output
            epoch records containing ``val`` metric mappings.
        output_types (Sequence[str]): Configured representation order used as the
            final tie-breaker.

    Returns:
        tuple[str, Dict[str, Any]]: Winning output type and an auditable selection
            record with per-type selected-epoch scores.
    """
    if not output_types:
        raise ValueError("At least one output_type is required to select a primary model.")
    candidates: list[Dict[str, Any]] = []
    for output_type in output_types:
        if output_type not in trained_models:
            raise ValueError(f"Missing trained model for output_type={output_type}.")
        if output_type not in training_histories:
            raise ValueError(f"Missing training history for output_type={output_type}.")
        details = trained_models[output_type].get("training_details")
        if not isinstance(details, Mapping):
            raise ValueError(f"training_details missing for output_type={output_type}.")
        selected_epoch = int(details["selected_epoch"])
        history = list(training_histories[output_type])
        if selected_epoch < 1 or selected_epoch > len(history):
            raise ValueError(
                f"selected_epoch {selected_epoch} is out of range for "
                f"output_type={output_type} with {len(history)} epochs."
            )
        checkpoint_level = details.get("checkpoint_level")
        history_key = (
            f"val_{checkpoint_level}"
            if checkpoint_level in BAG_LEVELS
            else "val"
        )
        val_metrics = history[selected_epoch - 1].get(history_key)
        if not isinstance(val_metrics, Mapping):
            raise ValueError(
                f"{history_key} metrics missing for output_type={output_type}."
            )
        candidates.append(
            {
                "output_type": output_type,
                "selected_epoch": selected_epoch,
                "checkpoint_level": checkpoint_level,
                "val_balanced_accuracy": float(val_metrics["balanced_accuracy"]),
                "val_loss": float(val_metrics["loss"]),
            }
        )
    winner = min(
        enumerate(candidates),
        key=lambda item: (
            -item[1]["val_balanced_accuracy"],
            item[1]["val_loss"],
            item[0],
        ),
    )[1]
    levels = {candidate["checkpoint_level"] for candidate in candidates}
    configured_level = levels.pop() if len(levels) == 1 else None
    return winner["output_type"], {
        "metric": (
            f"val_{configured_level}_balanced_accuracy"
            if configured_level in BAG_LEVELS
            else ROOT_SELECTION_METRIC
        ),
        "checkpoint_level": configured_level,
        "winner": winner["output_type"],
        "candidates": candidates,
    }


def resolve_checkpoint_output_type(checkpoint: Mapping[str, Any]) -> str:
    """Return the deployed representation type stored on a run-root checkpoint.

    Args:
        checkpoint (Mapping[str, Any]): Loaded run-root checkpoint payload.

    Returns:
        str: ``default_output_type`` when present, else the training-details
            output type, else ``allcat``.
    """
    default = checkpoint.get("default_output_type")
    if isinstance(default, str) and default:
        return default
    details = checkpoint.get("training_details")
    if isinstance(details, Mapping):
        output_type = details.get("output_type")
        if isinstance(output_type, str) and output_type:
            return output_type
    return "allcat"


def plot_training_history(
    history: Sequence[Mapping[str, Any]],
    save_path: Path | str,
    best_epoch: int,
) -> None:
    """Plot train and validation metric curves in the CLAM stacked-panel style.

    Args:
        history (Sequence[Mapping[str, Any]]): Per-epoch records containing
            ``train`` and ``val`` metric mappings.
        save_path (Path | str): Destination PNG path.
        best_epoch (int): One-based best validation epoch. Values ``<= 0`` skip
            the vertical marker.

    Returns:
        None: The figure is written to disk.
    """
    if not history:
        raise ValueError("Cannot plot an empty training history.")
    plotted_splits = (
        ("train", "val_tissue", "val_slide")
        if all(
            "val_tissue" in entry and "val_slide" in entry
            for entry in history
        )
        else ("train", "val")
    )
    metric_keys = [
        key
        for key in HISTORY_METRIC_KEYS
        if all(
            entry.get(split, {}).get(key) is not None
            for entry in history
            for split in plotted_splits
        )
    ]
    if not metric_keys:
        raise ValueError("Training history contains no plottable metrics.")

    figure, axes = plt.subplots(
        len(metric_keys), 1, figsize=(10, 4 * len(metric_keys)), squeeze=False
    )
    for axis, key in zip(axes.ravel(), metric_keys):
        for split in plotted_splits:
            axis.plot([entry[split][key] for entry in history], label=split)
        if best_epoch > 0:
            axis.axvline(best_epoch - 1, color="red", linestyle="--", label="best")
        axis.set_ylabel(key)
        axis.grid(True, alpha=0.3)
        axis.legend()
    axes.ravel()[-1].set_xlabel("epoch")
    figure.tight_layout()
    destination = Path(save_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=200)
    plt.close(figure)


def _run_configured_post_training_stages(
    config: Mapping[str, Any], config_path: str | None
) -> None:
    """Run evaluation and visualization after training when those flags are set.

    Imports are deferred so ``evaluate_panther`` and ``visualize_assignments``
    can keep importing this module without a circular load.

    Args:
        config (Mapping[str, Any]): Resolved training configuration.
        config_path (str | None): YAML path passed to the standalone stages,
            or ``None`` for the package default.

    Returns:
        None: Requested stages write their artifacts under the current run.
    """
    if bool(config["evaluation"]["run_after_training"]):
        from .evaluate_panther import run_evaluation

        run_evaluation(config_path)
    if bool(config["visualization"]["run_after_training"]):
        from .visualize_assignments import run_visualization

        run_visualization(config_path)


def run_training(config_path: str | None = None) -> Path:
    """Fit prototypes, encode slides, train the linear head, and save a run.

    When ``evaluation.run_after_training`` or ``visualization.run_after_training``
    is true, the matching standalone stage runs after the checkpoint is saved.

    Args:
        config_path (str | None): YAML path, or ``None`` for the module default.

    Returns:
        Path: Created or reused training-run directory.
    """
    config = load_config(config_path)
    run_dir = allocate_training_run(config)
    seed_everything(int(config["random_seed"]))
    device = resolve_device(config)
    print(f"Run directory: {run_dir}")
    print(f"Device: {device}")

    slide_datasets, class_names, records = build_datasets(config, bag_level="slide")
    assignments = {record["slide_key"]: record["split"] for record in records}
    tissue_datasets, tissue_classes, _ = build_datasets(
        config,
        class_folders=class_names,
        split_assignments=assignments,
        bag_level="tissue",
    )
    if tissue_classes != class_names:
        raise ValueError("Tissue and slide datasets disagree on class order.")
    datasets_by_level = {
        "tissue": tissue_datasets,
        "slide": slide_datasets,
    }
    if len(class_names) < 2:
        raise ValueError("PANTHER classification requires at least two classes.")
    print(f"Classes: {class_names}")
    for level in BAG_LEVELS:
        for split, dataset in datasets_by_level[level].items():
            print(
                f"{level}/{split}: {len(dataset)} bags {class_counts(dataset)}"
            )
    prototype_level = str(config["prototype"]["bag_level"])
    training_level = str(config["training"]["bag_level"])
    checkpoint_level = str(config["training"]["checkpoint_level"])
    print(
        f"Levels: prototype={prototype_level} | training={training_level} | "
        f"checkpoint={checkpoint_level}"
    )
    print(
        f"Epoch metric order: train_{training_level}, val_tissue, val_slide"
    )
    save_split_manifest(records, class_names, run_dir)

    prototype_path = run_dir / "prototypes.pkl"
    prototypes, prototype_metadata = fit_prototypes(
        datasets_by_level[prototype_level]["train"], config, prototype_path
    )
    panther = create_panther(prototypes, config)
    component_embeddings = {
        level: aggregate_slide_embeddings(
            datasets_by_level[level], panther, device, run_dir
        )
        for level in BAG_LEVELS
    }
    output_types = list(config["model"]["output_type"])
    trained_models: Dict[str, Dict[str, Any]] = {}
    training_histories: Dict[str, list[Dict[str, Any]]] = {}
    for output_type in output_types:
        seed_everything(int(config["random_seed"]))
        embeddings_by_level = {
            level: compose_slide_embeddings(
                component_embeddings[level], output_type
            )
            for level in BAG_LEVELS
        }
        embeddings = embeddings_by_level[training_level]
        model_dir = run_dir / "models" / output_type
        print(
            f"Training output_type={output_type} with "
            f"{embeddings['train']['representations'].shape[1]} features"
        )
        classifier, history, training_details = train_classifier(
            embeddings,
            class_names,
            config,
            device,
            model_dir,
            validation_embeddings=embeddings_by_level,
        )
        training_details["output_type"] = output_type
        history_path = model_dir / "training_history.json"
        with history_path.open("w", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2)
        history_plot_path = model_dir / "training_history.png"
        plot_training_history(
            history,
            history_plot_path,
            int(training_details["best_validation_epoch"]),
        )
        selected_checkpoint_path = model_dir / "best_model.pth"
        model_payload = {
            "model_state_dict": classifier.state_dict(),
            "training_details": training_details,
            "output_type": output_type,
        }
        torch.save(model_payload, selected_checkpoint_path)
        trained_models[output_type] = {
            **model_payload,
            "model_dir": str(model_dir),
            "checkpoint": str(selected_checkpoint_path),
            "training_history": str(history_path),
            "training_history_plot": str(history_plot_path),
        }
        training_histories[output_type] = history

    primary_output_type, root_model_selection = select_primary_output_type(
        trained_models, training_histories, output_types
    )
    winner_scores = next(
        candidate
        for candidate in root_model_selection["candidates"]
        if candidate["output_type"] == primary_output_type
    )
    print(
        f"Root best_model.pth: {primary_output_type} "
        f"(val_bacc={winner_scores['val_balanced_accuracy']:.4f})"
    )
    primary_model = trained_models[primary_output_type]
    primary_history = training_histories[primary_output_type]
    with (run_dir / "training_history.json").open("w", encoding="utf-8") as handle:
        json.dump(primary_history, handle, indent=2)
    plot_training_history(
        primary_history,
        run_dir / "training_history.png",
        int(primary_model["training_details"]["best_validation_epoch"]),
    )
    checkpoint = {
        "model_schema": MODEL_SCHEMA,
        "output_types": output_types,
        "models": trained_models,
        "prototypes": prototypes,
        "class_folders": list(class_names),
        "config": dict(config),
        "prototype_metadata": prototype_metadata,
        "split_manifest": str(run_dir / "split_manifest.json"),
        "embedding_caches": {
            level: {
                component: str(path)
                for component, path in component_embedding_paths(
                    run_dir, level
                ).items()
            }
            for level in BAG_LEVELS
        },
        "default_output_type": primary_output_type,
        "model_state_dict": primary_model["model_state_dict"],
        "training_details": primary_model["training_details"],
        "root_model_selection": root_model_selection,
    }
    torch.save(checkpoint, Path(config["paths"]["checkpoint"]))
    with (run_dir / "resolved_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
    print(f"Saved selected checkpoint: {config['paths']['checkpoint']}")
    _run_configured_post_training_stages(config, config_path)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    arguments = parser.parse_args()
    run_training(arguments.config)


if __name__ == "__main__":
    main()
