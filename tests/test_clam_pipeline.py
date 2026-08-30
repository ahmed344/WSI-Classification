"""Synthetic integration tests for the canonical CLAM pipeline."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Type

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
import yaml
from matplotlib.collections import PolyCollection
from matplotlib.figure import Figure
from torch import nn
from torch.utils.data import DataLoader

CLAM_DIRECTORY = Path(__file__).resolve().parents[1] / "clam"
if str(CLAM_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(CLAM_DIRECTORY))

import evaluate_clam
import train_clam
import visualize_attention
from clam_dataset import collate_fn, create_bag_dataset
from clam_model import CLAM_MB, CLAM_SB
from config_loader import (
    allocate_training_run,
    apply_run_artifact_paths,
    clam_results_root,
    load_config,
    resolve_latest_training_run,
)
from losses import GeneralizedCrossEntropyLoss


def _write_coordinates(path: Path, tile_count: int) -> None:
    """Write a deterministic coordinate CSV.

    Args:
        path (Path): Destination CSV path.
        tile_count (int): Number of coordinate rows to write.

    Returns:
        None: The coordinate file is written to disk.
    """
    with path.open("w", encoding="utf-8", newline="") as coordinate_file:
        writer = csv.DictWriter(coordinate_file, fieldnames=["x", "y"])
        writer.writeheader()
        for index in range(tile_count):
            writer.writerow({"x": index * 10, "y": index * 10 + 5})


def _write_fixture_dataset(
    root: Path,
    class_count: int = 2,
    slides_per_class: int = 4,
    tissues_per_slide: int = 2,
    tile_count: int = 5,
    feature_dim: int = 4,
) -> None:
    """Create a deterministic on-disk feature and coordinate dataset.

    Args:
        root (Path): Directory in which fixture classes are created.
        class_count (int): Number of class directories.
        slides_per_class (int): Number of slides in each class.
        tissues_per_slide (int): Number of tissues in each slide.
        tile_count (int): Number of tiles in each tissue.
        feature_dim (int): Width of each synthetic feature vector.

    Returns:
        None: Feature tensors and coordinate CSV files are written.
    """
    for class_index in range(class_count):
        for slide_index in range(slides_per_class):
            slide_directory = root / f"Class{class_index}" / f"slide_{slide_index:02d}"
            slide_directory.mkdir(parents=True)
            for tissue_index in range(tissues_per_slide):
                tissue_name = f"tissue_{tissue_index}"
                offset = class_index * 100 + slide_index * 10 + tissue_index
                features = (
                    torch.arange(tile_count * feature_dim, dtype=torch.float32)
                    .reshape(tile_count, feature_dim)
                    .add(offset)
                    .div(100.0)
                )
                torch.save(
                    features,
                    slide_directory / f"{tissue_name}_features.pt",
                )
                _write_coordinates(
                    slide_directory / f"{tissue_name}_tiles.csv",
                    tile_count,
                )


def _pipeline_config(
    root: Path,
    bag_level: str,
    model_type: str = "clam_sb",
    num_classes: int = 2,
) -> Dict[str, Any]:
    """Build a minimal resolved canonical pipeline configuration.

    Args:
        root (Path): Synthetic dataset root.
        bag_level (str): ``tissue`` or ``slide``.
        model_type (str): ``clam_sb`` or ``clam_mb``.
        num_classes (int): Configured class count.

    Returns:
        Dict[str, Any]: Configuration accepted by dataset and model factories.
    """
    return {
        "data_root": str(root),
        "feature_file_suffix": "_features.pt",
        "input_dim": 4,
        "bag_level": bag_level,
        "train_ratio": 0.5,
        "val_ratio": 0.25,
        "test_ratio": 0.25,
        "random_seed": 11,
        "max_tiles_per_bag": {"train": None, "val": None, "test": None},
        "tile_sampling": "first",
        "feature_normalization": "none",
        "model_type": model_type,
        "hidden_dim": 8,
        "attention_dim": 4,
        "num_classes": num_classes,
        "gated_attention": True,
        "attention_normalization": "sigmoid_mean",
        "pooling_layernorm": True,
        "pooling_mode": "distributional",
        "pooling_use_variance": True,
        "pooling_num_prototypes": 0,
        "dropout": 0.0,
        "feature_projection_dropout": 0.0,
        "k_sample": 8,
        "subtyping": True,
        "bag_weight": 0.7,
        "batch_size": 64,
        "epochs": 1,
        "gradient_accumulation_steps": 1,
        "lr_cls": 1e-3,
        "weight_decay_cls": 0.0,
    }


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("bag_level", "patient", "Invalid bag_level"),
        ("bag_weight", 1.5, "bag_weight"),
        ("k_sample", 0, "k_sample"),
        ("q", 1.1, "q"),
        ("epsilon", 1.0, "epsilon"),
        ("attention_normalization", "softmax", "attention_normalization"),
        ("pooling_layernorm", False, "pooling_layernorm"),
        ("pooling_mode", "histogram", "pooling_mode"),
        ("pooling_use_variance", "yes", "pooling_use_variance"),
        ("pooling_num_prototypes", 16, "pooling_num_prototypes"),
    ],
)
def test_config_validation_rejects_invalid_values(
    tmp_path: Path,
    key: str,
    value: object,
    message: str,
) -> None:
    """Verify invalid canonical configuration values fail during loading.

    Args:
        tmp_path (Path): Pytest temporary directory.
        key (str): Configuration key to invalidate.
        value (object): Invalid replacement value.
        message (str): Expected error-message fragment.

    Returns:
        None: Assertions verify eager configuration validation.
    """
    canonical_path = CLAM_DIRECTORY / "config.yml"
    with canonical_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    config[key] = value
    invalid_path = tmp_path / "invalid.yml"
    with invalid_path.open("w", encoding="utf-8") as config_file:
        yaml.safe_dump(config, config_file)
    with pytest.raises(ValueError, match=message):
        load_config(str(invalid_path))


@pytest.mark.parametrize("dpi", [0, -1, 1.5, True])
def test_config_validation_rejects_invalid_visualization_dpi(
    tmp_path: Path,
    dpi: object,
) -> None:
    """Verify visualization DPI must be a positive integer.

    Args:
        tmp_path (Path): Pytest temporary directory.
        dpi (object): Invalid visualization DPI replacement.

    Returns:
        None: Assertions verify eager visualization-DPI validation.
    """
    canonical_path = CLAM_DIRECTORY / "config.yml"
    with canonical_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    config["visualization"]["dpi"] = dpi
    invalid_path = tmp_path / "invalid_visualization.yml"
    with invalid_path.open("w", encoding="utf-8") as config_file:
        yaml.safe_dump(config, config_file)
    with pytest.raises(ValueError, match="visualization.dpi"):
        load_config(str(invalid_path))


@pytest.mark.parametrize("render_workers", [0, -1, 1.5, True])
def test_config_validation_rejects_invalid_render_workers(
    tmp_path: Path,
    render_workers: object,
) -> None:
    """Verify render worker count must be a positive integer.

    Args:
        tmp_path (Path): Pytest temporary directory.
        render_workers (object): Invalid render worker replacement.

    Returns:
        None: Assertions verify eager render-worker validation.
    """
    canonical_path = CLAM_DIRECTORY / "config.yml"
    with canonical_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    config["visualization"]["render_workers"] = render_workers
    invalid_path = tmp_path / "invalid_render_workers.yml"
    with invalid_path.open("w", encoding="utf-8") as config_file:
        yaml.safe_dump(config, config_file)
    with pytest.raises(ValueError, match="visualization.render_workers"):
        load_config(str(invalid_path))


@pytest.mark.parametrize(
    ("evaluation", "message"),
    [
        ("slide", "evaluation.*mapping"),
        ({"supplementary_bag_level": "patient"}, "supplementary_bag_level"),
        ({"include_train": "yes"}, "include_train"),
    ],
)
def test_config_validation_rejects_invalid_evaluation_controls(
    tmp_path: Path,
    evaluation: object,
    message: str,
) -> None:
    """Verify malformed dual-level evaluation controls fail during loading.

    Args:
        tmp_path (Path): Pytest temporary directory.
        evaluation (object): Invalid evaluation section replacement.
        message (str): Expected error-message expression.

    Returns:
        None: Assertions verify eager evaluation-control validation.
    """
    canonical_path = CLAM_DIRECTORY / "config.yml"
    with canonical_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    config["evaluation"] = evaluation
    invalid_path = tmp_path / "invalid_evaluation.yml"
    with invalid_path.open("w", encoding="utf-8") as config_file:
        yaml.safe_dump(config, config_file)
    with pytest.raises(ValueError, match=message):
        load_config(str(invalid_path))


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("num_workers", -1, "num_workers"),
        ("num_workers", True, "num_workers"),
        ("prefetch_factor", 0, "prefetch_factor"),
        ("prefetch_factor", 1.5, "prefetch_factor"),
    ],
)
def test_training_config_rejects_invalid_loader_values(
    tmp_path: Path,
    key: str,
    value: object,
    message: str,
) -> None:
    """Verify invalid DataLoader controls fail before training starts.

    Args:
        tmp_path (Path): Pytest temporary directory.
        key (str): Loader configuration key to invalidate.
        value (object): Invalid replacement value.
        message (str): Expected error-message fragment.

    Returns:
        None: Assertions verify eager training configuration validation.
    """
    config = _pipeline_config(tmp_path, "tissue")
    config[key] = value
    with pytest.raises(ValueError, match=message):
        train_clam._validate_training_config(config)


def test_make_loader_uses_workers_and_prefetch_factor(tmp_path: Path) -> None:
    """Verify the production loader applies parallel loading controls.

    Args:
        tmp_path (Path): Pytest temporary directory.

    Returns:
        None: Assertions verify worker, prefetch, and lifecycle settings.
    """
    _write_fixture_dataset(tmp_path)
    config = _pipeline_config(tmp_path, "tissue")
    config.update({"batch_size": 2, "num_workers": 2, "prefetch_factor": 3})
    dataset = create_bag_dataset(config, "train")

    loader = train_clam._make_loader(dataset, config, training=False)
    batch = next(iter(loader))

    assert loader.num_workers == 2
    assert loader.prefetch_factor == 3
    assert loader.persistent_workers is False
    assert batch["features"].shape[0] == 2


def test_generalized_cross_entropy_q_zero_matches_cross_entropy() -> None:
    """Verify that ``q=0`` recovers ordinary cross entropy.

    Args:
        None.

    Returns:
        None: Assertions compare loss values and gradients.
    """
    logits = torch.tensor(
        [[1.2, -0.4, 0.7], [-0.2, 1.1, 0.3]],
        dtype=torch.float64,
        requires_grad=True,
    )
    targets = torch.tensor([0, 2])
    expected = nn.functional.cross_entropy(logits, targets)
    actual = GeneralizedCrossEntropyLoss()(logits, targets)

    assert actual.item() == pytest.approx(expected.item())
    actual.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_generalized_cross_entropy_supports_label_smoothing_and_weights() -> None:
    """Verify smoothed generalized cross entropy remains finite and trainable.

    Args:
        None.

    Returns:
        None: Assertions verify the configured loss and criterion factory.
    """
    config = {
        "q": 0.7,
        "epsilon": 0.1,
        "use_class_weighted_loss": True,
    }
    class_weights = torch.tensor([1.0, 2.0, 3.0])
    criterion = train_clam.create_classification_criterion(config, class_weights)
    logits = torch.randn(4, 3, requires_grad=True)
    targets = torch.tensor([0, 1, 2, 1])
    loss = criterion(logits, targets)

    assert criterion.q == pytest.approx(0.7)
    assert criterion.epsilon == pytest.approx(0.1)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_training_rejects_configured_class_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify training rejects a class count inconsistent with the dataset.

    Args:
        tmp_path (Path): Pytest temporary directory.
        monkeypatch (pytest.MonkeyPatch): Fixture used to supply resolved config.

    Returns:
        None: The expected class-count ``ValueError`` is asserted.
    """
    _write_fixture_dataset(tmp_path)
    config = _pipeline_config(tmp_path, "tissue", num_classes=3)

    def _load_test_config(config_path: Optional[str] = None) -> Dict[str, Any]:
        """Return the synthetic mismatch configuration.

        Args:
            config_path (Optional[str]): Ignored configuration path.

        Returns:
            Dict[str, Any]: Resolved synthetic configuration.
        """
        del config_path
        return config

    monkeypatch.setattr(train_clam, "load_config", _load_test_config)
    with pytest.raises(ValueError, match="Configured num_classes=3"):
        train_clam.train("synthetic.yml")


@pytest.mark.parametrize("model_type", ["clam_sb", "clam_mb"])
@pytest.mark.parametrize("bag_level", ["tissue", "slide"])
def test_one_batch_train_validate_and_evaluate(
    tmp_path: Path,
    model_type: str,
    bag_level: str,
) -> None:
    """Run one synthetic batch through training, validation, and evaluation.

    Args:
        tmp_path (Path): Pytest temporary directory.
        model_type (str): Canonical CLAM architecture under test.
        bag_level (str): Tissue- or slide-level dataset contract under test.

    Returns:
        None: Assertions verify losses, fixed metrics, and bag metadata.
    """
    _write_fixture_dataset(tmp_path)
    config = _pipeline_config(tmp_path, bag_level, model_type)
    dataset = create_bag_dataset(config, "train")
    loader = DataLoader(
        dataset,
        batch_size=len(dataset),
        shuffle=False,
        collate_fn=collate_fn,
    )
    model = train_clam.create_model(config)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    device = torch.device("cpu")

    training = train_clam.train_epoch(
        model,
        loader,
        criterion,
        optimizer,
        device,
        bag_weight=0.7,
    )
    validation = train_clam.validate(
        model,
        loader,
        criterion,
        device,
        bag_weight=0.7,
    )
    evaluation = evaluate_clam.evaluate(
        model,
        loader,
        device,
        class_names=["Class0", "Class1"],
    )

    assert len(training["labels"]) == len(dataset)
    assert len(validation["predictions"]) == len(dataset)
    assert len(training["confusion_matrix"]) == 2
    assert len(validation["confusion_matrix"]) == 2
    assert evaluation["confusion_matrix"].shape == (2, 2)
    assert len(evaluation["slide_names"]) == len(dataset)
    assert all(
        torch.isfinite(torch.tensor(training[key]))
        for key in (
            "loss",
            "classification_loss",
            "instance_loss",
        )
    )
    sample = dataset[0]
    assert sample["provenance"]["bag_level"] == bag_level
    assert sample["num_tissues"] == (1 if bag_level == "tissue" else 2)


def test_validate_defers_metric_reduction_without_changing_values(
    tmp_path: Path,
) -> None:
    """Verify deferred device reductions preserve unequal-batch epoch metrics.

    Args:
        tmp_path (Path): Pytest temporary directory.

    Returns:
        None: Assertions compare validation metrics with manual batch reductions.
    """
    _write_fixture_dataset(tmp_path)
    config = _pipeline_config(tmp_path, "tissue")
    dataset = create_bag_dataset(config, "train")
    loader = DataLoader(
        dataset,
        batch_size=3,
        shuffle=False,
        collate_fn=collate_fn,
    )
    model = train_clam.create_model(config)
    criterion = nn.CrossEntropyLoss()
    bag_weight = 0.7
    expected_totals = {
        "loss": 0.0,
        "classification_loss": 0.0,
        "instance_loss": 0.0,
    }

    model.eval()
    with torch.inference_mode():
        for batch in loader:
            outputs = model(
                batch["features"],
                mask=batch["masks"],
                labels=batch["labels"],
                instance_eval=True,
            )
            classification_loss = criterion(outputs["logits"], batch["labels"])
            instance_loss = outputs["instance_loss"]
            loss = bag_weight * classification_loss + (1.0 - bag_weight) * instance_loss
            batch_size = int(batch["labels"].shape[0])
            expected_totals["loss"] += float(loss.item()) * batch_size
            expected_totals["classification_loss"] += (
                float(classification_loss.item()) * batch_size
            )
            expected_totals["instance_loss"] += float(instance_loss.item()) * batch_size

    metrics = train_clam.validate(
        model,
        loader,
        criterion,
        torch.device("cpu"),
        bag_weight,
    )

    for key, expected_total in expected_totals.items():
        assert metrics[key] == pytest.approx(expected_total / len(dataset))


def test_fixed_metrics_retain_absent_classes() -> None:
    """Verify metric tensors retain every configured class.

    Args:
        None: This test uses fixed synthetic labels and predictions.

    Returns:
        None: Assertions verify three-class matrix and macro averaging behavior.
    """
    metrics = train_clam._classification_metrics(
        labels=[0, 0, 1],
        predictions=[0, 1, 1],
        num_classes=3,
    )
    assert metrics["confusion_matrix"] == [[1, 1, 0], [0, 1, 0], [0, 0, 0]]
    assert metrics["macro_f1"] == pytest.approx((2 / 3 + 2 / 3 + 0) / 3)
    assert (
        evaluate_clam._multiclass_auc(
            labels=[0, 0, 1],
            probabilities=[[0.8, 0.1, 0.1], [0.4, 0.5, 0.1], [0.1, 0.8, 0.1]],
            num_classes=3,
        )
        is None
    )


def test_slide_checkpoint_metric_and_patience_warmup() -> None:
    """Verify slide checkpoint resolution and post-warm-up patience counting.

    Args:
        None: This test exercises deterministic metric-policy helpers.

    Returns:
        None: Assertions verify metric routing and early-stopping semantics.
    """
    metric_key, maximize, metric_name = train_clam.resolve_best_checkpoint_metric(
        "slide_balanced_accuracy"
    )
    assert metric_key == "balanced_accuracy"
    assert maximize is True
    assert metric_name == "slide_balanced_accuracy"

    counter = 0
    for epoch in range(1, 11):
        counter = train_clam.update_patience_counter(
            counter,
            improved=False,
            epoch=epoch,
            minimum_epochs=10,
        )
    assert counter == 0
    counter = train_clam.update_patience_counter(
        counter,
        improved=False,
        epoch=11,
        minimum_epochs=10,
    )
    assert counter == 1
    assert (
        train_clam.update_patience_counter(
            counter,
            improved=True,
            epoch=12,
            minimum_epochs=10,
        )
        == 0
    )


def test_default_evaluation_controls_include_tissue_and_slide() -> None:
    """Verify configured standalone evaluation runs both CLAM bag levels.

    Args:
        None: This test uses an in-memory runtime configuration.

    Returns:
        None: Assertions verify ordered levels and the train-split switch.
    """
    levels, include_train = evaluate_clam._evaluation_controls(
        {
            "evaluation": {
                "supplementary_bag_level": "slide",
                "include_train": False,
            }
        },
        primary_level="tissue",
        supplementary_level=None,
        include_train=None,
    )
    assert levels == ["tissue", "slide"]
    assert include_train is False


def test_slide_level_evaluation_writes_all_artifacts(tmp_path: Path) -> None:
    """Verify concatenated-slide evaluation emits summary and matrix files.

    Args:
        tmp_path (Path): Temporary dataset and artifact root.

    Returns:
        None: Assertions verify slide bag composition and persisted artifacts.
    """
    data_root = tmp_path / "data"
    _write_fixture_dataset(data_root)
    config = _pipeline_config(data_root, "tissue")
    config["slide_evaluation_batch_size"] = 1
    model = train_clam.create_model(config)
    output_dir = tmp_path / "evaluation"

    result = evaluate_clam.run_level_split_evaluation(
        model=model,
        config=config,
        class_folders=["Class0", "Class1"],
        device=torch.device("cpu"),
        output_dir=str(output_dir),
        level="slide",
        split="val",
    )

    assert result["num_bags"] == 2
    assert Path(result["artifacts"]["summary"]).name == "slide_val_evaluation.json"
    assert (
        Path(result["artifacts"]["confusion_matrix"]).name
        == "slide_val_confusion_matrix.png"
    )
    assert all(Path(path).is_file() for path in result["artifacts"].values())


def test_training_records_slide_history_and_checkpoint_metric(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify one training epoch records and selects slide validation metrics.

    Args:
        tmp_path (Path): Temporary dataset and checkpoint root.
        monkeypatch (pytest.MonkeyPatch): Fixture used to supply synthetic config.
        capsys (pytest.CaptureFixture[str]): Fixture capturing grouped epoch output.

    Returns:
        None: Assertions verify history, metadata, and printed metric grouping.
    """
    data_root = tmp_path / "data"
    _write_fixture_dataset(data_root)
    checkpoint_dir = tmp_path / "checkpoints"
    config = _pipeline_config(data_root, "tissue")
    config.update(
        {
            "output_dir": str(tmp_path / "results"),
            "checkpoint_dir": str(checkpoint_dir),
            "paths": {
                "checkpoint": str(checkpoint_dir / "best_model.pth"),
                "evaluation_output": str(tmp_path / "evaluation"),
                "attention_output": str(tmp_path / "attention"),
            },
            "_explicit_paths": {
                "checkpoint": True,
                "evaluation_output": True,
                "attention_output": True,
            },
            "best_checkpoint_metric": "slide_balanced_accuracy",
            "min_epochs_before_early_stopping": 0,
            "patience": 2,
            "slide_evaluation_batch_size": 1,
        }
    )

    def _load_test_config(config_path: Optional[str] = None) -> Dict[str, Any]:
        """Return the synthetic dual-level training configuration.

        Args:
            config_path (Optional[str]): Ignored configuration path.

        Returns:
            Dict[str, Any]: Mutable configuration used by the training entry point.
        """
        del config_path
        return config

    monkeypatch.setattr(train_clam, "load_config", _load_test_config)
    artifacts = train_clam.train("synthetic.yml")

    with Path(artifacts["history"]).open("r", encoding="utf-8") as history_file:
        history = json.load(history_file)
    checkpoint = torch.load(
        artifacts["best_checkpoint"],
        map_location="cpu",
        weights_only=False,
    )
    output = capsys.readouterr().out

    assert len(history["train"]["loss"]) == 1
    assert len(history["val"]["loss"]) == 1
    assert len(history["val_slide"]["loss"]) == 1
    assert checkpoint["best_metric"]["name"] == "slide_balanced_accuracy"
    assert checkpoint["best_metric"]["level"] == "slide"
    assert "Epoch metric order: train_tissue, val_tissue, val_slide" in output
    assert "epoch=1 | loss=" in output


@pytest.mark.parametrize("model_class", [CLAM_SB, CLAM_MB])
def test_short_bags_bound_top_bottom_instance_sampling(
    model_class: Type[nn.Module],
) -> None:
    """Verify oversized ``k_sample`` is safe for one- and three-tile bags.

    Args:
        model_class (Type[nn.Module]): Canonical CLAM variant under test.

    Returns:
        None: Assertions verify empty and bounded top/bottom supervision.
    """
    model = model_class(
        input_dim=4,
        hidden_dim=8,
        attention_dim=4,
        num_classes=2,
        dropout=0.0,
        k_sample=8,
        subtyping=False,
    )
    one_tile = model(
        torch.randn(1, 1, 4),
        labels=torch.tensor([0]),
        instance_eval=True,
    )
    three_tiles = model(
        torch.randn(1, 3, 4),
        labels=torch.tensor([1]),
        instance_eval=True,
    )
    assert one_tile["instance_targets"].numel() == 0
    assert one_tile["instance_loss"].item() == pytest.approx(0.0)
    assert three_tiles["instance_targets"].tolist() == [1, 0]
    assert three_tiles["instance_predictions"].shape == (2,)


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        ("feature_dim", "Feature dimension mismatch"),
        ("coordinate_rows", "Feature/coordinate row mismatch"),
    ],
)
def test_dataset_rejects_feature_coordinate_errors(
    tmp_path: Path,
    failure: str,
    message: str,
) -> None:
    """Verify feature width and coordinate alignment errors are explicit.

    Args:
        tmp_path (Path): Pytest temporary directory.
        failure (str): Synthetic corruption to introduce.
        message (str): Expected error-message fragment.

    Returns:
        None: The expected dataset ``ValueError`` is asserted.
    """
    _write_fixture_dataset(
        tmp_path,
        class_count=1,
        slides_per_class=1,
        tissues_per_slide=1,
    )
    config = _pipeline_config(tmp_path, "tissue", num_classes=2)
    config.update({"train_ratio": 1.0, "val_ratio": 0.0, "test_ratio": 0.0})
    if failure == "feature_dim":
        config["input_dim"] = 5
    else:
        coordinate_path = tmp_path / "Class0" / "slide_00" / "tissue_0_tiles.csv"
        _write_coordinates(coordinate_path, tile_count=4)
    dataset = create_bag_dataset(config, "train")
    with pytest.raises(ValueError, match=message):
        _ = dataset[0]


@pytest.mark.parametrize("model_type", ["clam_sb", "clam_mb"])
def test_checkpoint_payload_load_round_trip(
    tmp_path: Path,
    model_type: str,
) -> None:
    """Verify canonical checkpoint payloads strictly restore both variants.

    Args:
        tmp_path (Path): Pytest temporary directory.
        model_type (str): Canonical architecture to serialize and restore.

    Returns:
        None: Assertions verify schema metadata, state, and identical logits.
    """
    config = _pipeline_config(tmp_path, "slide", model_type)
    model = train_clam.create_model(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    payload = train_clam._checkpoint_payload(
        checkpoint_type="best",
        epoch=1,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        class_folders=["Class0", "Class1"],
        history={"train": {}, "val": {}},
        best_metric={"name": "loss", "mode": "min", "value": 1.0, "epoch": 1},
    )
    checkpoint_path = tmp_path / f"{model_type}.pth"
    torch.save(payload, checkpoint_path)
    loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    restored = train_clam.create_model(loaded["config"])
    restored.load_state_dict(loaded["model_state_dict"], strict=True)
    model.eval()
    restored.eval()
    features = torch.randn(2, 4, 4)
    masks = torch.ones(2, 4, dtype=torch.bool)

    with torch.no_grad():
        expected = model(features, mask=masks, instance_eval=False)["logits"]
        actual = restored(features, mask=masks, instance_eval=False)["logits"]
    assert loaded["model_schema"] == train_clam.MODEL_SCHEMA
    assert loaded["bag_level"] == "slide"
    assert loaded["class_folders"] == ["Class0", "Class1"]
    assert torch.equal(actual, expected)


def test_visualization_rejects_softmax_checkpoint_schema(tmp_path: Path) -> None:
    """Verify visualization refuses pre-sigmoid CLAM checkpoints.

    Args:
        tmp_path (Path): Temporary checkpoint destination.

    Returns:
        None: The old schema rejection is asserted.
    """
    config = _pipeline_config(tmp_path, "tissue")
    model = train_clam.create_model(config)
    checkpoint_path = tmp_path / "old_softmax_checkpoint.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "class_folders": ["Class0", "Class1"],
            "model_schema": "canonical_clam_v1",
            "bag_level": "tissue",
        },
        checkpoint_path,
    )

    with pytest.raises(ValueError, match="canonical_clam_v1"):
        visualize_attention.load_checkpoint_model(
            str(checkpoint_path),
            torch.device("cpu"),
        )


def test_allocate_training_run_colocates_artifact_paths(tmp_path: Path) -> None:
    """Verify dated run allocation co-locates checkpoint and artifact paths.

    Args:
        tmp_path (Path): Pytest temporary directory used as ``output_dir``.

    Returns:
        None: Assertions verify the timestamped run layout.
    """
    config: Dict[str, Any] = {
        "output_dir": str(tmp_path),
        "checkpoint_dir": str(tmp_path / "legacy_checkpoints"),
        "paths": {},
        "_explicit_paths": {
            "checkpoint": False,
            "evaluation_output": False,
            "attention_output": False,
        },
    }
    run_dir = allocate_training_run(config)
    assert run_dir.parent == clam_results_root(tmp_path)
    assert run_dir.is_dir()
    assert Path(config["checkpoint_dir"]) == run_dir
    assert Path(config["paths"]["checkpoint"]) == run_dir / "best_model.pth"
    assert Path(config["paths"]["evaluation_output"]) == run_dir / "evaluation_results"
    assert Path(config["paths"]["attention_output"]) == run_dir / "attention_heatmaps"


def test_resolve_latest_training_run_picks_newest(tmp_path: Path) -> None:
    """Verify newest timestamped run directories are preferred.

    Args:
        tmp_path (Path): Pytest temporary directory hosting synthetic runs.

    Returns:
        None: Assertions verify newest-by-name selection.
    """
    clam_root = clam_results_root(tmp_path)
    older = clam_root / "2026-01-01_120000"
    newer = clam_root / "2026-07-19_153045"
    older.mkdir(parents=True)
    newer.mkdir(parents=True)
    (older / "best_model.pth").write_bytes(b"old")
    (newer / "best_model.pth").write_bytes(b"new")
    assert resolve_latest_training_run(clam_root) == newer


def test_allocate_training_run_respects_explicit_checkpoint(tmp_path: Path) -> None:
    """Verify explicit checkpoint paths skip dated run allocation.

    Args:
        tmp_path (Path): Pytest temporary directory for an explicit checkpoint dir.

    Returns:
        None: Assertions verify the legacy checkpoint directory is reused.
    """
    checkpoint_dir = tmp_path / "explicit_checkpoints"
    config: Dict[str, Any] = {
        "output_dir": str(tmp_path / "results"),
        "checkpoint_dir": str(checkpoint_dir),
        "paths": {
            "checkpoint": str(checkpoint_dir / "best_model.pth"),
            "evaluation_output": str(tmp_path / "eval"),
            "attention_output": str(tmp_path / "attn"),
        },
        "_explicit_paths": {
            "checkpoint": True,
            "evaluation_output": True,
            "attention_output": True,
        },
    }
    run_dir = allocate_training_run(config)
    assert run_dir == checkpoint_dir.resolve()
    assert not clam_results_root(tmp_path / "results").exists()
    assert config["paths"]["checkpoint"] == str(checkpoint_dir / "best_model.pth")


def test_load_config_binds_null_paths_to_latest_run(tmp_path: Path) -> None:
    """Verify null path defaults resolve to the newest dated training run.

    Args:
        tmp_path (Path): Pytest temporary directory for YAML and run folders.

    Returns:
        None: Assertions verify loader path defaults follow the newest run.
    """
    canonical_path = CLAM_DIRECTORY / "config.yml"
    with canonical_path.open("r", encoding="utf-8") as config_file:
        raw = yaml.safe_load(config_file)
    output_dir = tmp_path / "Results"
    older = clam_results_root(output_dir) / "2026-01-01_010101"
    newer = clam_results_root(output_dir) / "2026-07-19_020202"
    older.mkdir(parents=True)
    newer.mkdir(parents=True)
    (newer / "best_model.pth").write_bytes(b"ckpt")
    raw["data_root"] = str(tmp_path / "data")
    (tmp_path / "data").mkdir()
    raw["output_dir"] = str(output_dir)
    raw["paths"] = {
        "checkpoint": None,
        "evaluation_output": None,
        "attention_output": None,
    }
    config_path = tmp_path / "config.yml"
    with config_path.open("w", encoding="utf-8") as config_file:
        yaml.safe_dump(raw, config_file)

    loaded = load_config(str(config_path))
    assert Path(loaded["paths"]["checkpoint"]) == newer / "best_model.pth"
    assert Path(loaded["paths"]["evaluation_output"]) == newer / "evaluation_results"
    assert Path(loaded["paths"]["attention_output"]) == newer / "attention_heatmaps"
    assert Path(loaded["checkpoint_dir"]) == newer


def test_apply_run_artifact_paths_updates_all_keys(tmp_path: Path) -> None:
    """Verify run path application updates checkpoint and artifact keys together.

    Args:
        tmp_path (Path): Pytest temporary directory used as the run root.

    Returns:
        None: Assertions verify co-located path assignment.
    """
    run_dir = tmp_path / "2026-07-19_111111"
    config: Dict[str, Any] = {"paths": {}}
    apply_run_artifact_paths(config, run_dir)
    assert config["checkpoint_dir"] == str(run_dir.resolve())
    assert config["paths"]["checkpoint"].endswith("best_model.pth")
    assert config["paths"]["evaluation_output"].endswith("evaluation_results")
    assert config["paths"]["attention_output"].endswith("attention_heatmaps")


@pytest.mark.parametrize("model_class", [CLAM_SB, CLAM_MB])
def test_tile_evidence_exactly_reconstructs_logits(
    model_class: Type[nn.Module],
) -> None:
    """Verify SB and MB tile evidence reconstruct logits and ignores padding.

    Args:
        model_class (Type[nn.Module]): Canonical CLAM variant under test.

    Returns:
        None: Evidence shape, masking, and reconstruction are asserted.
    """
    model = model_class(
        input_dim=4,
        hidden_dim=8,
        attention_dim=4,
        num_classes=3,
        dropout=0.0,
        pooling_mode="distributional",
        pooling_use_variance=True,
    )
    model.eval()
    with torch.no_grad():
        model.pooling_layernorm.weight.copy_(
            torch.linspace(0.5, 1.5, model.bag_feature_dim)
        )
        model.pooling_layernorm.bias.copy_(
            torch.linspace(-0.3, 0.3, model.bag_feature_dim)
        )
    features = torch.randn(2, 6, 4)
    masks = torch.tensor(
        [
            [True, True, True, True, False, False],
            [True, True, True, True, True, True],
        ]
    )
    with torch.no_grad():
        outputs = model(features, mask=masks, instance_eval=False)
        evidence, errors, class_baselines = visualize_attention.compute_tile_evidence(
            model=model,
            features=features,
            masks=masks,
            attention_weights=outputs["attention_weights"],
            logits=outputs["logits"],
        )

    assert evidence.shape == (2, 3, 6)
    assert torch.count_nonzero(evidence[0, :, 4:]) == 0
    assert torch.allclose(
        evidence.sum(dim=-1) + class_baselines.unsqueeze(0),
        outputs["logits"],
        atol=1e-4,
        rtol=0.0,
    )
    assert float(errors.max()) <= 1e-4


def test_distributional_evidence_avoids_class_tile_feature_expansion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify evidence pooling never concatenates a four-dimensional tensor.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture used to observe concatenations.

    Returns:
        None: Concatenated tensors remain bag-sized rather than tile-expanded.
    """
    model = CLAM_MB(
        input_dim=4,
        hidden_dim=8,
        attention_dim=4,
        num_classes=3,
        dropout=0.0,
        pooling_mode="distributional",
        pooling_use_variance=True,
    )
    model.eval()
    features = torch.randn(2, 16, 4)
    masks = torch.ones(2, 16, dtype=torch.bool)
    with torch.no_grad():
        outputs = model(features, mask=masks, instance_eval=False)

    original_cat = torch.cat
    observed_input_ranks: list[int] = []

    def recording_cat(
        tensors: Any,
        dim: int = 0,
        *,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Record tensor ranks before delegating to ``torch.cat``.

        Args:
            tensors (Any): Iterable of tensors to concatenate.
            dim (int): Concatenation dimension.
            out (Optional[torch.Tensor]): Optional output tensor.

        Returns:
            torch.Tensor: Concatenated tensor from the original operation.
        """
        materialized_tensors = tuple(tensors)
        observed_input_ranks.extend(tensor.ndim for tensor in materialized_tensors)
        return original_cat(materialized_tensors, dim=dim, out=out)

    monkeypatch.setattr(torch, "cat", recording_cat)
    with torch.no_grad():
        evidence, errors, class_baselines = visualize_attention.compute_tile_evidence(
            model=model,
            features=features,
            masks=masks,
            attention_weights=outputs["attention_weights"],
            logits=outputs["logits"],
        )

    assert observed_input_ranks
    assert max(observed_input_ranks) <= 3
    assert torch.allclose(
        evidence.sum(dim=-1) + class_baselines.unsqueeze(0),
        outputs["logits"],
        atol=1e-4,
        rtol=0.0,
    )
    assert float(errors.max()) <= 1e-4


def test_sb_evidence_is_class_specific_with_shared_attention() -> None:
    """Verify SB applies one attention branch to distinct class projections.

    Args:
        None.

    Returns:
        None: Shared attention and class-specific evidence are asserted.
    """
    model = CLAM_SB(
        input_dim=2,
        hidden_dim=2,
        attention_dim=2,
        num_classes=2,
        dropout=0.0,
    )
    model.eval()
    with torch.no_grad():
        model.classifiers[0].weight.fill_(1.0)
        model.classifiers[1].weight.fill_(-1.0)
        model.classifiers[0].bias.zero_()
        model.classifiers[1].bias.zero_()
    features = torch.ones(1, 3, 2)
    masks = torch.ones(1, 3, dtype=torch.bool)
    with torch.no_grad():
        outputs = model(features, mask=masks, instance_eval=False)
        evidence, _, _ = visualize_attention.compute_tile_evidence(
            model=model,
            features=features,
            masks=masks,
            attention_weights=outputs["attention_weights"],
            logits=outputs["logits"],
        )

    assert outputs["attention_weights"].shape == (1, 1, 3)
    assert torch.allclose(evidence[:, 0], -evidence[:, 1])


def test_evidence_color_limit_uses_absolute_quantile() -> None:
    """Verify evidence colors use a robust positive symmetric limit.

    Args:
        None.

    Returns:
        None: The absolute 99th-percentile color limit is asserted.
    """
    evidence = np.asarray([-4.0, -1.0, 0.0, 2.0], dtype=np.float64)
    expected = float(np.quantile(np.abs(evidence), 0.99))
    assert visualize_attention.evidence_color_limit(evidence) == pytest.approx(expected)
    assert visualize_attention.evidence_color_limit(np.zeros(3, dtype=np.float64)) > 0.0


def test_attention_and_evidence_are_saved_in_one_figure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify aligned attention and evidence are rendered into one PNG.

    Args:
        tmp_path (Path): Temporary destination for rendered PNG files.
        monkeypatch (pytest.MonkeyPatch): Fixture used to observe output DPI.

    Returns:
        None: The combined heatmap artifact is asserted.
    """
    coordinates = np.asarray([[0.0, 0.0], [448.0, 0.0], [0.0, 448.0]], dtype=np.float64)
    attention = np.asarray([[0.2, 0.3, 0.5]], dtype=np.float64)
    evidence = np.asarray([[0.2, -0.1, 0.4], [-0.3, 0.2, 0.1]], dtype=np.float64)
    combined_path = tmp_path / "slide_tissue_attention.png"
    saved_dpi: Dict[str, Any] = {}
    original_savefig = Figure.savefig

    def _capture_savefig(
        figure: Figure,
        output_path: Path,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Capture render DPI while delegating to Matplotlib.

        Args:
            figure (Figure): Figure being saved.
            output_path (Path): Destination image path.
            *args (Any): Additional positional save arguments.
            **kwargs (Any): Additional keyword save arguments.

        Returns:
            None: The original save operation is completed.
        """
        saved_dpi["value"] = kwargs.get("dpi")
        original_savefig(figure, output_path, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", _capture_savefig)

    visualize_attention.save_attention_evidence_figure(
        branch_attention=attention,
        class_evidence=evidence,
        coordinates=coordinates,
        branch_names=["Shared attention"],
        class_names=["Class0", "Class1"],
        bag_evidence_sums=evidence.sum(axis=1),
        predicted_index=1,
        slide_name="slide",
        tissue_name="tissue",
        true_class="Class0",
        predicted_class="Class1",
        predicted_probability=0.7,
        image_path=None,
        output_path=combined_path,
        tile_size=448,
        thumbnail_size=128,
        render_dpi=72,
    )

    assert combined_path.is_file()
    assert saved_dpi["value"] == 72


def _axis_by_title(figure: Figure, title: str) -> plt.Axes:
    """Return the unique figure axis whose title matches ``title``.

    Args:
        figure (Figure): Combined attention/evidence figure.
        title (str): Exact axis title to locate.

    Returns:
        plt.Axes: The matching axis.
    """
    matches = [axis for axis in figure.axes if axis.get_title() == title]
    if len(matches) != 1:
        raise AssertionError(
            f"Expected one axis titled {title!r}, found {len(matches)}."
        )
    return matches[0]


def test_mean_attention_sits_above_original_in_five_class_figure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify a 5-class figure uses a 2x6 grid with mean attention over original.

    Args:
        tmp_path (Path): Temporary destination for the rendered PNG.
        monkeypatch (pytest.MonkeyPatch): Fixture used to inspect the figure.

    Returns:
        None: First-column layout and 2x6 GridSpec are asserted.
    """
    coordinates = np.asarray([[0.0, 0.0], [448.0, 0.0], [0.0, 448.0]], dtype=np.float64)
    attention = np.asarray(
        [
            [0.10, 0.20, 0.30],
            [0.40, 0.10, 0.20],
            [0.15, 0.35, 0.25],
            [0.05, 0.50, 0.10],
            [0.30, 0.05, 0.40],
        ],
        dtype=np.float64,
    )
    evidence = np.asarray(
        [
            [0.2, -0.1, 0.4],
            [-0.3, 0.2, 0.1],
            [0.1, 0.0, -0.2],
            [0.4, -0.2, 0.1],
            [-0.1, 0.3, 0.2],
        ],
        dtype=np.float64,
    )
    class_names = [f"Class{index}" for index in range(5)]
    captured_figure: Dict[str, Figure] = {}
    original_savefig = Figure.savefig

    def _capture_figure(
        figure: Figure,
        output_path: Path,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Record the figure being saved, then write it.

        Args:
            figure (Figure): Figure being saved.
            output_path (Path): Destination image path.
            *args (Any): Additional positional save arguments.
            **kwargs (Any): Additional keyword save arguments.

        Returns:
            None: The original save operation is completed.
        """
        captured_figure["figure"] = figure
        original_savefig(figure, output_path, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", _capture_figure)
    output_path = tmp_path / "slide_tissue_attention.png"
    visualize_attention.save_attention_evidence_figure(
        branch_attention=attention,
        class_evidence=evidence,
        coordinates=coordinates,
        branch_names=class_names,
        class_names=class_names,
        bag_evidence_sums=evidence.sum(axis=1),
        predicted_index=2,
        slide_name="slide",
        tissue_name="tissue",
        true_class="Class0",
        predicted_class="Class2",
        predicted_probability=0.6,
        image_path=None,
        output_path=output_path,
        tile_size=448,
        thumbnail_size=128,
        render_dpi=72,
    )

    figure = captured_figure["figure"]
    mean_axis = _axis_by_title(figure, "Attention — Mean")
    original_axis = _axis_by_title(figure, "Original")
    grid = original_axis.get_gridspec()
    assert grid is not None
    assert grid.nrows == 2
    assert grid.ncols == 6
    assert mean_axis.get_position().y0 > original_axis.get_position().y0
    assert len(mean_axis.collections) == 1
    assert isinstance(mean_axis.collections[0], PolyCollection)
    assert len(mean_axis.collections[0].get_paths()) == coordinates.shape[0]
    assert len(original_axis.collections) == 0
    assert output_path.is_file()


def test_heatmap_tiles_use_one_vectorized_collection_per_axis() -> None:
    """Verify attention and evidence avoid one artist per tile.

    Args:
        None.

    Returns:
        None: Each heatmap axis contains one collection and no rectangle patches.
    """
    coordinates = np.asarray([[0.0, 0.0], [448.0, 0.0], [0.0, 448.0]], dtype=np.float64)
    values = np.asarray([0.2, 0.3, 0.5], dtype=np.float64)
    figure, axes = plt.subplots(1, 2)
    try:
        visualize_attention.draw_attention(axes[0], coordinates, values, tile_size=448)
        visualize_attention.draw_evidence(
            axes[1], coordinates, values - 0.3, tile_size=448, color_limit=0.3
        )

        for axis in axes:
            assert len(axis.collections) == 1
            assert len(axis.patches) == 0
            assert len(axis.collections[0].get_paths()) == coordinates.shape[0]
    finally:
        plt.close(figure)


@pytest.mark.parametrize("render_workers", [1, 2])
def test_visualization_summary_includes_evidence_diagnostics(
    tmp_path: Path,
    render_workers: int,
) -> None:
    """Verify end-to-end visualization preserves attention and adds evidence.

    Args:
        tmp_path (Path): Temporary synthetic dataset and output root.
        render_workers (int): Number of figure-rendering processes.

    Returns:
        None: Paired artifacts and per-class evidence diagnostics are asserted.
    """
    data_root = tmp_path / "data"
    _write_fixture_dataset(data_root)
    config = _pipeline_config(data_root, "tissue", model_type="clam_sb")
    dataset = create_bag_dataset(config, "train")
    dataset.indices = dataset.indices[:1]
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
    )
    model = CLAM_SB(
        input_dim=4,
        hidden_dim=8,
        attention_dim=4,
        num_classes=2,
        dropout=0.0,
    )
    output_dir = tmp_path / "heatmaps"

    results = visualize_attention.evaluate_with_attention(
        model=model,
        dataloader=dataloader,
        device=torch.device("cpu"),
        class_names=["Class0", "Class1"],
        data_root=str(data_root),
        output_dir=str(output_dir),
        bag_level="tissue",
        tile_size=448,
        thumbnail_size=128,
        render_workers=render_workers,
    )

    assert len(results) == 1
    result = results[0]
    assert Path(result["heatmap_path"]).is_file()
    assert Path(result["evidence_heatmap_path"]).is_file()
    assert result["evidence_heatmap_path"] == result["heatmap_path"]
    assert set(result["evidence_by_class"]) == {"Class0", "Class1"}
    for class_index, class_name in enumerate(("Class0", "Class1")):
        diagnostics = result["evidence_by_class"][class_name]
        assert diagnostics["logit_reconstruction_error"] <= 1e-4
        assert diagnostics["reconstructed_logit"] == pytest.approx(
            diagnostics["bag_tile_evidence_sum"]
            + float(model.classifiers[class_index].bias.item()),
            abs=1e-6,
        )
