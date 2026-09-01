"""Focused tests for the independent logistic-regression baseline."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import joblib
import numpy as np
import pytest
import torch
import yaml
from torch.utils.data import Dataset

from clam.clam_dataset import WSIBagDataset as ClamBagDataset
from logistic_regression.config_loader import (
    allocate_training_run,
    load_config,
)
from logistic_regression.dataset import (
    WSIBagDataset,
    collate_fn,
)
from logistic_regression.evaluate import (
    evaluate_checkpoint,
    evaluate_dataset,
    load_checkpoint_bundle,
)
from logistic_regression.model import (
    RawFeatureStatisticsPooler,
    TorchLogisticRegression,
    pool_raw_features,
)
from logistic_regression.train import (
    MODEL_SCHEMA,
    create_dataloader,
    train_logistic_regression,
)


CLASS_NAMES = ("Alpha", "Beta", "Gamma")
FEATURE_SUFFIX = "_features_synthetic.pt"
FEATURE_DIM = 3


def write_tissue(
    slide_directory: Path,
    tissue_name: str,
    features: torch.Tensor,
    coordinate_count: Optional[int] = None,
) -> None:
    """Write one aligned synthetic feature tensor and coordinate CSV.

    Args:
        slide_directory (Path): Parent class/slide directory.
        tissue_name (str): Shared tissue file stem.
        features (torch.Tensor): Tile feature matrix shaped ``[T, D]``.
        coordinate_count (Optional[int]): CSV row count, or ``None`` for ``T``.

    Returns:
        None: Synthetic files are written beneath ``slide_directory``.
    """
    slide_directory.mkdir(parents=True, exist_ok=True)
    torch.save(features, slide_directory / f"{tissue_name}{FEATURE_SUFFIX}")
    count = features.shape[0] if coordinate_count is None else coordinate_count
    with (slide_directory / f"{tissue_name}_tiles.csv").open(
        "w", encoding="utf-8", newline=""
    ) as coordinate_file:
        writer = csv.DictWriter(coordinate_file, fieldnames=["x", "y"])
        writer.writeheader()
        for tile_index in range(count):
            writer.writerow({"x": tile_index * 448, "y": (tile_index % 2) * 448})


@pytest.fixture
def synthetic_data_root(tmp_path: Path) -> Path:
    """Create compact class/slide/tissue feature fixtures.

    Args:
        tmp_path (Path): Pytest-managed temporary directory.

    Returns:
        Path: Synthetic on-disk dataset root.
    """
    data_root = tmp_path / "data"
    for class_index, class_name in enumerate(CLASS_NAMES):
        center = torch.zeros(FEATURE_DIM)
        center[class_index] = 6.0
        for slide_index in range(10):
            slide_directory = (
                data_root / class_name / f"{class_name}_slide_{slide_index:02d}"
            )
            for tissue_index, tile_count in enumerate((2, 4)):
                offsets = torch.arange(tile_count, dtype=torch.float32).unsqueeze(1)
                direction = torch.tensor([[0.02, -0.01, 0.03]])
                features = (
                    center
                    + 0.01 * slide_index
                    + 0.05 * tissue_index
                    + offsets * direction
                )
                write_tissue(
                    slide_directory,
                    f"tissue_{tissue_index}",
                    features,
                )
    return data_root


def write_config(
    tmp_path: Path,
    data_root: Path,
    overrides: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Write a resolved-test YAML derived from the package defaults.

    Args:
        tmp_path (Path): Temporary directory receiving config and artifacts.
        data_root (Path): Synthetic dataset root.
        overrides (Optional[Mapping[str, Any]]): Top-level values to replace.

    Returns:
        Path: Written YAML configuration path.
    """
    default_path = Path(__file__).parents[1] / "logistic_regression" / "config.yml"
    with default_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    config.update(
        {
            "data_root": str(data_root),
            "checkpoint_dir": str(tmp_path / "checkpoints"),
            "output_dir": str(tmp_path / "results"),
            "feature_model": "synthetic",
            "feature_model_suffixes": {"synthetic": FEATURE_SUFFIX},
            "feature_model_input_dims": {"synthetic": FEATURE_DIM},
            "num_classes": len(CLASS_NAMES),
            "batch_size": 16,
            "num_workers": 0,
            "train_ratio": 0.6,
            "val_ratio": 0.2,
            "test_ratio": 0.2,
            "c_values": [0.1],
            "max_iter": 300,
            "class_weight": None,
            "max_tiles_per_bag": {"train": None, "val": None, "test": None},
            "paths": {"checkpoint": None, "evaluation_output": None},
            "evaluation": {
                "supplementary_bag_level": "slide",
                "include_train": False,
            },
        }
    )
    if overrides is not None:
        config.update(dict(overrides))
    config_path = tmp_path / "logistic_config.yml"
    with config_path.open("w", encoding="utf-8") as config_file:
        yaml.safe_dump(config, config_file, sort_keys=False)
    return config_path


def selected_slide_keys(dataset: WSIBagDataset) -> set[str]:
    """Return class-qualified slide keys selected by one dataset split.

    Args:
        dataset (WSIBagDataset): Dataset whose selected bags are inspected.

    Returns:
        set[str]: Class-qualified selected slide identifiers.
    """
    return {
        str(dataset._bags[index]["slide_key"])  # noqa: SLF001 - split contract test
        for index in dataset.indices
    }


def test_default_config_resolution_and_validation_errors(
    tmp_path: Path, synthetic_data_root: Path
) -> None:
    """Check package defaults resolve and representative invalid YAML is rejected.

    Args:
        tmp_path (Path): Pytest-managed temporary directory.
        synthetic_data_root (Path): Synthetic dataset root.

    Returns:
        None: Assertions verify resolution and validation failures.
    """
    default = load_config()
    assert Path(default["data_root"]).is_absolute()
    assert default["input_dim"] == 1536
    assert default["feature_file_suffix"] == "_features_uni2h.pt"
    assert Path(default["paths"]["checkpoint"]).is_absolute()

    invalid_cases = (
        ({"bag_level": "patient"}, "Invalid bag_level"),
        ({"device": "cpu"}, "Invalid device"),
        ({"use_standard_scaler": False}, "use_standard_scaler"),
        ({"c_values": [0.0]}, "c_values"),
        (
            {"train_ratio": 0.7, "val_ratio": 0.2, "test_ratio": 0.2},
            "sum to one",
        ),
        ({"feature_model": "missing"}, "Unknown feature_model"),
    )
    for case_index, (overrides, message) in enumerate(invalid_cases):
        case_directory = tmp_path / f"invalid_{case_index}"
        case_directory.mkdir()
        config_path = write_config(case_directory, synthetic_data_root, overrides)
        with pytest.raises(ValueError, match=message):
            load_config(str(config_path))


def test_dated_run_colocates_training_artifacts(
    tmp_path: Path, synthetic_data_root: Path
) -> None:
    """Verify automatic model and evaluation paths share one dated run.

    Args:
        tmp_path (Path): Pytest-managed temporary directory.
        synthetic_data_root (Path): Synthetic dataset root.

    Returns:
        None: Assertions verify dated-run path colocation.
    """
    config = load_config(str(write_config(tmp_path, synthetic_data_root)))
    run_directory = allocate_training_run(config)
    assert run_directory.parent == Path(config["output_dir"]) / "logistic_regression"
    assert Path(config["paths"]["checkpoint"]).parent == run_directory
    assert Path(config["paths"]["evaluation_output"]).parent == run_directory


def test_split_parity_with_clam_and_slide_disjointness(
    synthetic_data_root: Path,
) -> None:
    """Compare independent split membership exactly with CLAM in tests only.

    Args:
        synthetic_data_root (Path): Synthetic dataset root.

    Returns:
        None: Assertions verify exact parity and no slide leakage.
    """
    independent_keys: Dict[str, set[str]] = {}
    common_arguments = {
        "data_root": str(synthetic_data_root),
        "class_folders": list(CLASS_NAMES),
        "train_ratio": 0.6,
        "val_ratio": 0.2,
        "test_ratio": 0.2,
        "random_seed": 42,
        "feature_file_suffix": FEATURE_SUFFIX,
        "expected_feature_dim": FEATURE_DIM,
        "bag_level": "tissue",
    }
    for split in ("train", "val", "test"):
        independent = WSIBagDataset(split=split, **common_arguments)
        clam = ClamBagDataset(split=split, **common_arguments)
        independent_keys[split] = selected_slide_keys(independent)
        clam_keys = {
            str(clam._bags[index]["slide_key"])  # noqa: SLF001 - parity contract
            for index in clam.indices
        }
        assert independent_keys[split] == clam_keys
    assert independent_keys["train"].isdisjoint(independent_keys["val"])
    assert independent_keys["train"].isdisjoint(independent_keys["test"])
    assert independent_keys["val"].isdisjoint(independent_keys["test"])


def test_feature_dimension_and_coordinate_mismatch_rejection(
    tmp_path: Path,
) -> None:
    """Reject incorrect feature width and feature/coordinate row misalignment.

    Args:
        tmp_path (Path): Pytest-managed temporary directory.

    Returns:
        None: Assertions verify both on-disk contract errors.
    """
    root = tmp_path / "mismatch"
    slide_directory = root / "Only" / "slide"
    write_tissue(slide_directory, "bad_dim", torch.ones((2, FEATURE_DIM + 1)))
    dimension_dataset = WSIBagDataset(
        str(root),
        split="train",
        train_ratio=1.0,
        val_ratio=0.0,
        test_ratio=0.0,
        feature_file_suffix=FEATURE_SUFFIX,
        expected_feature_dim=FEATURE_DIM,
    )
    with pytest.raises(ValueError, match="Feature dimension mismatch"):
        _ = dimension_dataset[0]

    root = tmp_path / "coordinate_mismatch"
    slide_directory = root / "Only" / "slide"
    write_tissue(
        slide_directory,
        "bad_rows",
        torch.ones((3, FEATURE_DIM)),
        coordinate_count=2,
    )
    coordinate_dataset = WSIBagDataset(
        str(root),
        split="train",
        train_ratio=1.0,
        val_ratio=0.0,
        test_ratio=0.0,
        feature_file_suffix=FEATURE_SUFFIX,
        expected_feature_dim=FEATURE_DIM,
    )
    with pytest.raises(ValueError, match="row mismatch"):
        _ = coordinate_dataset[0]


def test_pooling_is_deterministic_mask_aware_population_statistics() -> None:
    """Verify exact means/stds and exclusion of arbitrary padded values.

    Args:
        None: Tensor fixtures are constructed internally.

    Returns:
        None: Assertions verify the pooling contract.
    """
    features = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 6.0], [1000.0, -1000.0]],
            [[2.0, 4.0], [2.0, 8.0], [2.0, 12.0]],
        ]
    )
    masks = torch.tensor([[True, True, False], [True, True, True]])
    expected = torch.tensor([[2.0, 4.0, 1.0, 2.0], [2.0, 8.0, 0.0, 3.2659864]])
    pooled = pool_raw_features(features, masks)
    assert torch.allclose(pooled, expected, atol=1e-6)
    changed = features.clone()
    changed[~masks] = torch.tensor([[-77_777.0, 88_888.0]])
    assert torch.equal(pooled, pool_raw_features(changed, masks))
    assert torch.equal(pooled, RawFeatureStatisticsPooler()(features, masks))


def test_tissue_and_slide_bags(
    synthetic_data_root: Path,
) -> None:
    """Verify tissue bags remain separate and slide bags concatenate tissues.

    Args:
        synthetic_data_root (Path): Synthetic dataset root.

    Returns:
        None: Assertions verify both supported bag granularities.
    """
    arguments = {
        "data_root": str(synthetic_data_root),
        "class_folders": list(CLASS_NAMES),
        "split": "train",
        "train_ratio": 1.0,
        "val_ratio": 0.0,
        "test_ratio": 0.0,
        "feature_file_suffix": FEATURE_SUFFIX,
        "expected_feature_dim": FEATURE_DIM,
    }
    tissue_dataset = WSIBagDataset(bag_level="tissue", **arguments)
    slide_dataset = WSIBagDataset(bag_level="slide", **arguments)
    assert len(tissue_dataset) == 2 * len(slide_dataset)
    tissue_sample = tissue_dataset[0]
    slide_sample = slide_dataset[0]
    assert tissue_sample["num_tissues"] == 1
    assert slide_sample["num_tissues"] == 2
    assert slide_sample["features"].shape == (6, FEATURE_DIM)
    assert slide_sample["tissue_slices"] == [(0, 2), (2, 6)]
    assert torch.equal(slide_sample["tissue_indices"], torch.tensor([0, 0, 1, 1, 1, 1]))


def test_training_selects_c_from_validation_and_never_builds_test(
    tmp_path: Path,
    synthetic_data_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Check validation tie-breaking and absence of training-time test access.

    Args:
        tmp_path (Path): Pytest-managed temporary directory.
        synthetic_data_root (Path): Synthetic dataset root.
        monkeypatch (pytest.MonkeyPatch): Pytest patching utility.

    Returns:
        None: Assertions verify smaller-C tie selection and split access.
    """
    import logistic_regression.train as training_module

    config_path = write_config(
        tmp_path,
        synthetic_data_root,
        {"c_values": [1.0, 0.01, 0.1]},
    )
    requested_splits: list[str] = []
    original_factory = training_module.create_bag_dataset

    def recording_factory(
        config: Mapping[str, Any],
        split: str,
        class_folders: Optional[Sequence[str]] = None,
        **overrides: Any,
    ) -> WSIBagDataset:
        """Record and delegate one dataset construction.

        Args:
            config (Mapping[str, Any]): Resolved dataset configuration.
            split (str): Requested split.
            class_folders (Optional[Sequence[str]]): Fixed class order.
            **overrides (Any): Dataset overrides.

        Returns:
            WSIBagDataset: Real synthetic dataset.
        """
        requested_splits.append(split)
        return original_factory(config, split, class_folders, **overrides)

    def tied_validation_metrics(
        model: TorchLogisticRegression,
        bag_features: np.ndarray,
        labels: np.ndarray,
        num_classes: int,
    ) -> Dict[str, Any]:
        """Return deliberately tied validation metrics for C tie-breaking.

        Args:
            model (TorchLogisticRegression): Fitted candidate.
            bag_features (np.ndarray): Validation bag vectors.
            labels (np.ndarray): Validation labels.
            num_classes (int): Fixed class count.

        Returns:
            Dict[str, Any]: Equal candidate metrics including equal log loss.
        """
        del model, bag_features, labels, num_classes
        return {
            "accuracy": 0.75,
            "balanced_accuracy": 0.75,
            "macro_f1": 0.75,
            "macro_ovr_roc_auc": 0.75,
            "log_loss": 0.5,
        }

    monkeypatch.setattr(training_module, "create_bag_dataset", recording_factory)
    monkeypatch.setattr(training_module, "evaluate_candidate", tied_validation_metrics)
    artifacts = train_logistic_regression(str(config_path))
    bundle = joblib.load(artifacts["checkpoint"])
    assert requested_splits == ["train", "val"]
    assert bundle["selected_C"] == 0.01
    assert bundle["fit_device"] == "cuda"
    assert bundle["selection"]["validation_only"] is True
    with Path(artifacts["best_model_report"]).open("r", encoding="utf-8") as report_file:
        assert json.load(report_file)["test_evaluated"] is False


def test_joblib_schema_roundtrip_and_evaluation_artifacts(
    tmp_path: Path,
    synthetic_data_root: Path,
) -> None:
    """Train/load a schema bundle and emit primary plus supplementary results.

    Args:
        tmp_path (Path): Pytest-managed temporary directory.
        synthetic_data_root (Path): Synthetic dataset root.

    Returns:
        None: Assertions verify roundtrip schema, artifacts, and manifest.
    """
    config_path = write_config(tmp_path, synthetic_data_root)
    artifacts = train_logistic_regression(str(config_path))
    bundle = load_checkpoint_bundle(artifacts["checkpoint"])
    assert bundle["model_schema"] == MODEL_SCHEMA
    assert bundle["pooling_dim"] == 2 * FEATURE_DIM
    assert np.array_equal(
        bundle["model"].predict(np.eye(2 * FEATURE_DIM)),
        load_checkpoint_bundle(artifacts["checkpoint"])["model"].predict(
            np.eye(2 * FEATURE_DIM)
        ),
    )

    manifest = evaluate_checkpoint(
        config_path=str(config_path),
        checkpoint=artifacts["checkpoint"],
        supplementary_bag_level="slide",
    )
    assert manifest["evaluated_levels"] == ["tissue", "slide"]
    assert set(manifest["results"]) == {
        "tissue/val",
        "tissue/test",
        "slide/val",
        "slide/test",
    }
    assert Path(manifest["manifest"]).is_file()
    for result in manifest["results"].values():
        assert not result.get("skipped", False)
        assert set(result["artifacts"]) == {
            "summary",
            "predictions_json",
            "predictions_csv",
            "confusion_matrix",
        }
        assert all(Path(path).is_file() for path in result["artifacts"].values())


class SyntheticEvaluationDataset(Dataset):
    """Provide in-memory bags whose evaluation labels omit one fixed class."""

    class_folders = list(CLASS_NAMES)
    num_classes = len(CLASS_NAMES)

    def __init__(self) -> None:
        """Create two-class samples in a three-class contract.

        Args:
            None: Samples are generated internally.

        Returns:
            None: In-memory samples are initialized.
        """
        self.epoch = 0
        self.samples = [
            self._sample(torch.tensor([[6.0, 0.0, 0.0]]), 0, "zero"),
            self._sample(torch.tensor([[0.0, 6.0, 0.0]]), 1, "one"),
        ]

    @staticmethod
    def _sample(features: torch.Tensor, label: int, name: str) -> Dict[str, Any]:
        """Build one collate-compatible in-memory bag.

        Args:
            features (torch.Tensor): Tile feature matrix.
            label (int): Fixed-space class index.
            name (str): Slide and tissue identifier.

        Returns:
            Dict[str, Any]: Collate-compatible single-tissue sample.
        """
        tile_count = features.shape[0]
        return {
            "features": features,
            "label": label,
            "slide_name": name,
            "tissue_name": name,
            "tissue_names": [name],
            "tissue_slices": [(0, tile_count)],
            "coordinates": torch.zeros((tile_count, 2)),
            "tile_indices": torch.arange(tile_count),
            "tissue_indices": torch.zeros(tile_count, dtype=torch.long),
            "num_tissues": 1,
            "provenance": {},
        }

    def __len__(self) -> int:
        """Return the number of in-memory bags.

        Args:
            None: This method takes no arguments.

        Returns:
            int: Number of synthetic bags.
        """
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Return one in-memory bag.

        Args:
            index (int): Sample index.

        Returns:
            Dict[str, Any]: Collate-compatible bag.
        """
        return self.samples[index]

    def set_epoch(self, epoch: int) -> None:
        """Store the evaluation epoch expected by the evaluator.

        Args:
            epoch (int): Evaluation epoch.

        Returns:
            None: Epoch is stored in place.
        """
        self.epoch = epoch


def test_absent_class_keeps_fixed_confusion_and_report_metrics() -> None:
    """Keep all configured classes in reports when one class is absent.

    Args:
        None: In-memory training and evaluation data are generated internally.

    Returns:
        None: Assertions verify fixed confusion/report dimensions and zeros.
    """
    training_vectors = np.array(
        [
            [6.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [5.5, 0.2, 0.0, 0.0, 0.0, 0.0],
            [0.0, 6.0, 0.0, 0.0, 0.0, 0.0],
            [0.2, 5.5, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 6.0, 0.0, 0.0, 0.0],
            [0.0, 0.2, 5.5, 0.0, 0.0, 0.0],
        ]
    )
    model = TorchLogisticRegression(
        class_weight=None, device="cuda", max_iter=300
    )
    model.fit(training_vectors, np.array([0, 0, 1, 1, 2, 2]))
    config = {
        "batch_size": 2,
        "num_workers": 0,
        "prefetch_factor": 2,
        "random_seed": 42,
    }
    dataset = SyntheticEvaluationDataset()
    metrics = evaluate_dataset(
        model,
        dataset,  # type: ignore[arg-type] - intentional in-memory protocol fixture
        config,
        CLASS_NAMES,
        RawFeatureStatisticsPooler(),
    )
    assert metrics["confusion_matrix"].shape == (3, 3)
    assert metrics["classification_report"]["Gamma"]["support"] == 0.0
    assert metrics["classification_report"]["macro avg"]["recall"] == pytest.approx(
        2.0 / 3.0
    )
