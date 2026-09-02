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

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .config_loader import allocate_training_run, load_config
from .metrics import classification_metrics
from .panther_dataset import (
    PantherSlideDataset,
    build_datasets,
    class_counts,
    save_split_manifest,
)
from .panther_model import LinearClassifier, PANTHER
from .prototype import fit_prototypes


MODEL_SCHEMA = "panther-original-v1"


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
    prototypes: torch.Tensor, config: Mapping[str, Any]
) -> PANTHER:
    settings = config["model"]
    return PANTHER(
        prototypes=prototypes,
        em_iterations=int(settings["em_iterations"]),
        tau=float(settings["tau"]),
        covariance_regularizer=float(settings["covariance_regularizer"]),
        variance_floor=float(settings["variance_floor"]),
        output_type=str(settings["output_type"]),
        fix_prototypes=bool(settings["fix_prototypes"]),
        em_chunk_size=settings.get("em_chunk_size"),
    )


@torch.no_grad()
def aggregate_slide_embeddings(
    datasets: Mapping[str, PantherSlideDataset],
    panther: PANTHER,
    device: torch.device,
    destination: Path,
) -> Dict[str, Dict[str, Any]]:
    """Run the frozen MAP-EM slide encoder once and cache all split embeddings."""
    panther.eval().to(device)
    payload: Dict[str, Dict[str, Any]] = {}
    for split, dataset in datasets.items():
        started = time.time()
        representations = []
        labels = []
        slide_names = []
        slide_keys = []
        tile_counts = []
        for index in range(len(dataset)):
            item = dataset[index]
            features = item["features"].to(device, non_blocking=True)
            result = panther(features)["representation"].squeeze(0).cpu()
            if not torch.isfinite(result).all():
                raise FloatingPointError(
                    f"Non-finite PANTHER representation for {item['slide_key']}."
                )
            representations.append(result)
            labels.append(int(item["label"]))
            slide_names.append(str(item["slide_name"]))
            slide_keys.append(str(item["slide_key"]))
            tile_counts.append(int(item["num_tiles"]))
            if (index + 1) % 25 == 0 or index + 1 == len(dataset):
                print(f"PANTHER aggregation {split}: {index + 1}/{len(dataset)} slides")
        payload[split] = {
            "representations": torch.stack(representations),
            "labels": torch.tensor(labels, dtype=torch.long),
            "slide_names": slide_names,
            "slide_keys": slide_keys,
            "tile_counts": tile_counts,
            "elapsed_seconds": float(time.time() - started),
        }
    torch.save(payload, destination)
    return payload


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
    run_dir: Path,
) -> tuple[LinearClassifier, list[Dict[str, Any]], Dict[str, Any]]:
    settings = config["training"]
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
    best_loss = float("inf")
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
        val_data = embeddings["val"]
        val_metrics = _classifier_pass(
            model,
            val_data["representations"],
            val_data["labels"],
            class_names,
            device,
            int(settings["batch_size"]),
            loss_fn,
        )
        entry = {
            "epoch": epoch + 1,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(entry)
        if float(val_metrics["loss"]) < best_loss:
            best_loss = float(val_metrics["loss"])
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
        print(
            f"Epoch {epoch + 1:03d}/{settings['epochs']} "
            f"train_loss={train_metrics['loss']:.5f} "
            f"train_bacc={train_metrics['balanced_accuracy']:.4f} "
            f"val_loss={val_metrics['loss']:.5f} "
            f"val_bacc={val_metrics['balanced_accuracy']:.4f}"
        )

    final_state = copy.deepcopy(model.state_dict())
    torch.save(
        {"model_state_dict": final_state, "epoch": int(settings["epochs"])},
        run_dir / "final_model.pth",
    )
    assert best_state is not None
    torch.save(
        {"model_state_dict": best_state, "epoch": best_epoch},
        run_dir / "best_validation_model.pth",
    )
    selection = str(settings["checkpoint_selection"])
    selected_state = final_state if selection == "last" else best_state
    selected_epoch = int(settings["epochs"]) if selection == "last" else best_epoch
    model.load_state_dict(selected_state)
    details = {
        "input_dim": input_dim,
        "best_validation_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "selected_epoch": selected_epoch,
        "checkpoint_selection": selection,
    }
    return model, history, details


def run_training(config_path: str | None = None) -> Path:
    config = load_config(config_path)
    run_dir = allocate_training_run(config)
    seed_everything(int(config["random_seed"]))
    device = resolve_device(config)
    print(f"Run directory: {run_dir}")
    print(f"Device: {device}")

    datasets, class_names, records = build_datasets(config)
    if len(class_names) < 2:
        raise ValueError("PANTHER classification requires at least two classes.")
    print(f"Classes: {class_names}")
    for split, dataset in datasets.items():
        print(f"{split}: {len(dataset)} slides {class_counts(dataset)}")
    save_split_manifest(records, class_names, run_dir)

    prototype_path = run_dir / "prototypes.pkl"
    prototypes, prototype_metadata = fit_prototypes(
        datasets["train"], config, prototype_path
    )
    panther = create_panther(prototypes, config)
    embedding_path = run_dir / "slide_embeddings.pt"
    embeddings = aggregate_slide_embeddings(
        datasets, panther, device, embedding_path
    )
    classifier, history, training_details = train_classifier(
        embeddings, class_names, config, device, run_dir
    )

    with (run_dir / "training_history.json").open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    checkpoint = {
        "model_schema": MODEL_SCHEMA,
        "model_state_dict": classifier.state_dict(),
        "prototypes": prototypes,
        "class_folders": list(class_names),
        "config": dict(config),
        "prototype_metadata": prototype_metadata,
        "training_details": training_details,
        "split_manifest": str(run_dir / "split_manifest.json"),
        "embedding_cache": str(embedding_path),
    }
    torch.save(checkpoint, Path(config["paths"]["checkpoint"]))
    with (run_dir / "resolved_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
    print(f"Saved selected checkpoint: {config['paths']['checkpoint']}")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    arguments = parser.parse_args()
    run_training(arguments.config)


if __name__ == "__main__":
    main()
