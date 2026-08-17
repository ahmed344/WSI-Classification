"""Synthetic integration tests for the canonical CLAM pipeline."""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Type

import pytest
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

CLAM_DIRECTORY = Path(__file__).resolve().parents[1] / "clam"
if str(CLAM_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(CLAM_DIRECTORY))

import evaluate_clam
import train_clam
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
            slide_directory = (
                root / f"Class{class_index}" / f"slide_{slide_index:02d}"
            )
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
    assert all(torch.isfinite(torch.tensor(training[key])) for key in (
        "loss",
        "classification_loss",
        "instance_loss",
    ))
    sample = dataset[0]
    assert sample["provenance"]["bag_level"] == bag_level
    assert sample["num_tissues"] == (1 if bag_level == "tissue" else 2)


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
    assert evaluate_clam._multiclass_auc(
        labels=[0, 0, 1],
        probabilities=[[0.8, 0.1, 0.1], [0.4, 0.5, 0.1], [0.1, 0.8, 0.1]],
        num_classes=3,
    ) is None


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
