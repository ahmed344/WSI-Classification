"""Focused tests for the independent logistic-regression baseline."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
import yaml
from matplotlib.figure import Figure
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
    pooling_contract,
)
from logistic_regression.train import (
    LEGACY_MODEL_SCHEMA,
    MODEL_SCHEMA,
    create_dataloader,
    train_logistic_regression,
)
from logistic_regression.visualize_attribution import (
    aggregate_split_diagnostics,
    attribution_grid_maps,
    center_multinomial_parameters,
    center_tile_evidence,
    compute_split_color_limits,
    compute_tile_attribution,
    evidence_color_limit,
    evidence_contrast_ratio,
    require_mean_std_pooling,
    save_attribution_figure,
    split_attribution_output_dir,
    visualization_loader_config,
    visualize_attribution,
    within_tissue_effective_dimension,
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
            "paths": {"checkpoint": None, "evaluation_output": None, "attribution_output": None},
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
    assert default["pooling_statistics"] == ["mean", "standard_deviation"]
    assert default["visualization"]["samples"] == ["test"]
    assert Path(default["paths"]["checkpoint"]).is_absolute()
    assert Path(default["paths"]["attribution_output"]).is_absolute()

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
        ({"pooling_statistics": []}, "Invalid pooling_statistics"),
        ({"pooling_statistics": "mean"}, "Invalid pooling_statistics"),
        ({"pooling_statistics": ["median"]}, "Invalid pooling_statistics"),
        ({"pooling_statistics": ["mean", "mean"]}, "Invalid pooling_statistics"),
        ({"visualization": {"samples": []}}, "visualization.samples"),
        (
            {
                "visualization": {
                    "samples": ["holdout"],
                    "tile_size": 448,
                    "dpi": 150,
                    "render_workers": 1,
                    "thumbnail_size": 64,
                }
            },
            "visualization.samples",
        ),
        (
            {
                "visualization": {
                    "samples": ["test", "test"],
                    "tile_size": 448,
                    "dpi": 150,
                    "render_workers": 1,
                    "thumbnail_size": 64,
                }
            },
            "visualization.samples",
        ),
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
    assert Path(config["paths"]["attribution_output"]).parent == run_directory


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


def test_pooling_statistics_select_and_order_blocks() -> None:
    """Verify mean-only, std-only, and reversed concatenation widths.

    Args:
        None: Tensor fixtures are constructed internally.

    Returns:
        None: Assertions verify selected blocks and padding independence.
    """
    features = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 6.0], [1000.0, -1000.0]],
            [[2.0, 4.0], [2.0, 8.0], [2.0, 12.0]],
        ]
    )
    masks = torch.tensor([[True, True, False], [True, True, True]])
    expected_mean = torch.tensor([[2.0, 4.0], [2.0, 8.0]])
    expected_std = torch.tensor([[1.0, 2.0], [0.0, 3.2659864]])
    mean_only = pool_raw_features(features, masks, statistics=["mean"])
    std_only = pool_raw_features(
        features, masks, statistics=["standard_deviation"]
    )
    reversed_blocks = pool_raw_features(
        features, masks, statistics=["standard_deviation", "mean"]
    )
    assert torch.allclose(mean_only, expected_mean, atol=1e-6)
    assert torch.allclose(std_only, expected_std, atol=1e-6)
    assert torch.allclose(
        reversed_blocks, torch.cat((expected_std, expected_mean), dim=1), atol=1e-6
    )
    changed = features.clone()
    changed[~masks] = torch.tensor([[-77_777.0, 88_888.0]])
    assert torch.equal(
        mean_only, pool_raw_features(changed, masks, statistics=["mean"])
    )
    assert torch.equal(
        std_only,
        pool_raw_features(changed, masks, statistics=["standard_deviation"]),
    )
    assert pooling_contract(["mean"]) == (
        "concat(population_mean); mask excludes padding"
    )


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
    assert bundle["pooling_statistics"] == ["mean", "standard_deviation"]
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


def test_mean_only_checkpoint_pooler_ignores_runtime_yaml(
    tmp_path: Path,
    synthetic_data_root: Path,
) -> None:
    """Train on mean-only vectors and evaluate with a mean-plus-std runtime YAML.

    Args:
        tmp_path (Path): Pytest-managed temporary directory.
        synthetic_data_root (Path): Synthetic dataset root.

    Returns:
        None: Assertions verify checkpoint pooling width and successful evaluation.
    """
    train_directory = tmp_path / "train"
    train_directory.mkdir()
    train_config = write_config(
        train_directory,
        synthetic_data_root,
        {"pooling_statistics": ["mean"]},
    )
    artifacts = train_logistic_regression(str(train_config))
    bundle = load_checkpoint_bundle(artifacts["checkpoint"])
    assert bundle["pooling_statistics"] == ["mean"]
    assert bundle["pooling_dim"] == FEATURE_DIM
    assert bundle["pooling_contract"] == pooling_contract(["mean"])
    assert bundle["model"].coef_.shape[1] == FEATURE_DIM

    eval_directory = tmp_path / "eval"
    eval_directory.mkdir()
    eval_config = write_config(eval_directory, synthetic_data_root)
    manifest = evaluate_checkpoint(
        config_path=str(eval_config),
        checkpoint=artifacts["checkpoint"],
        supplementary_bag_level="slide",
    )
    assert set(manifest["results"]) == {
        "tissue/val",
        "tissue/test",
        "slide/val",
        "slide/test",
    }
    for result in manifest["results"].values():
        assert not result.get("skipped", False)


def test_legacy_v2_checkpoint_defaults_to_mean_and_std(
    tmp_path: Path,
    synthetic_data_root: Path,
) -> None:
    """Load a v2 bundle that omits pooling_statistics as mean-plus-std.

    Args:
        tmp_path (Path): Pytest-managed temporary directory.
        synthetic_data_root (Path): Synthetic dataset root.

    Returns:
        None: Assertions verify implied statistics and successful evaluation.
    """
    config_path = write_config(tmp_path, synthetic_data_root)
    artifacts = train_logistic_regression(str(config_path))
    bundle = joblib.load(artifacts["checkpoint"])
    bundle["model_schema"] = LEGACY_MODEL_SCHEMA
    del bundle["pooling_statistics"]
    legacy_path = tmp_path / "legacy_v2.joblib"
    joblib.dump(bundle, legacy_path)
    loaded = load_checkpoint_bundle(legacy_path)
    assert loaded["pooling_statistics"] == ["mean", "standard_deviation"]
    assert loaded["pooling_dim"] == 2 * FEATURE_DIM
    manifest = evaluate_checkpoint(
        config_path=str(config_path),
        checkpoint=str(legacy_path),
        supplementary_bag_level="slide",
    )
    assert not manifest["results"]["tissue/val"].get("skipped", False)


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


def make_fitted_attribution_model(
    input_dim: int = 2,
    num_classes: int = 2,
    class_center: bool = True,
) -> TorchLogisticRegression:
    """Build a CPU estimator with known linear parameters for attribution tests.

    Args:
        input_dim (int): Raw tile feature width.
        num_classes (int): Number of represented classes.
        class_center (bool): If True, subtract the across-class mean from
            coefficients so the per-tile class-sum gauge holds.

    Returns:
        TorchLogisticRegression: Fitted-looking estimator on CPU.
    """
    pooled_dim = 2 * input_dim
    model = TorchLogisticRegression(device="cpu", class_weight=None, max_iter=1)
    model.classes_ = np.arange(num_classes, dtype=np.int64)
    model.mean_ = np.zeros(pooled_dim, dtype=np.float32)
    model.scale_ = np.ones(pooled_dim, dtype=np.float32)
    coef = np.zeros((num_classes, pooled_dim), dtype=np.float32)
    for class_index in range(num_classes):
        coef[class_index, class_index % input_dim] = 1.0
        coef[class_index, input_dim + (class_index % input_dim)] = (
            0.5 - 0.25 * class_index
        )
    if class_center:
        coef = coef - coef.mean(axis=0, keepdims=True)
    model.coef_ = coef
    model.intercept_ = 0.1 * np.arange(num_classes, dtype=np.float32)
    model.fit_device_ = "cpu"
    return model


def test_require_mean_std_pooling_rejects_other_contracts() -> None:
    """Reject attribution pooling contracts that are not mean then std.

    Args:
        None: This test uses in-memory statistic names.

    Returns:
        None: Assertions verify acceptance of the required contract only.
    """
    assert require_mean_std_pooling(["mean", "standard_deviation"]) == (
        "mean",
        "standard_deviation",
    )
    with pytest.raises(ValueError, match="Attribution heatmaps require"):
        require_mean_std_pooling(["mean"])
    with pytest.raises(ValueError, match="Attribution heatmaps require"):
        require_mean_std_pooling(["standard_deviation", "mean"])


def test_tile_attribution_reconstructs_logits_and_ignores_padding() -> None:
    """Verify exact logit reconstruction and padding independence.

    Args:
        None: Synthetic bags are constructed internally.

    Returns:
        None: Assertions verify reconstruction error and masked padding.
    """
    features = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 6.0], [5.0, 2.0], [99.0, -99.0]],
            [[2.0, 4.0], [2.0, 8.0], [2.0, 12.0], [0.0, 0.0]],
        ]
    )
    masks = torch.tensor(
        [[True, True, True, False], [True, True, True, False]]
    )
    model = make_fitted_attribution_model(input_dim=2, num_classes=2)
    mean_evidence, std_evidence, errors, baselines, logits, pooled = compute_tile_attribution(
        features,
        masks,
        model,
        ["mean", "standard_deviation"],
        epsilon=0.0,
        num_classes=2,
    )
    reconstructed = mean_evidence.sum(dim=-1) + std_evidence.sum(dim=-1) + baselines
    assert torch.allclose(reconstructed, logits, atol=1e-8)
    assert float(errors.max().item()) <= 1e-8
    assert pooled.shape == (2, 4)
    assert torch.equal(mean_evidence[..., 3], torch.zeros_like(mean_evidence[..., 3]))
    assert torch.equal(std_evidence[..., 3], torch.zeros_like(std_evidence[..., 3]))
    changed = features.clone()
    changed[~masks] = torch.tensor([[-77_777.0, 88_888.0]])
    mean_changed, std_changed, _, _, logits_changed, pooled_changed = compute_tile_attribution(
        changed,
        masks,
        model,
        ["mean", "standard_deviation"],
        epsilon=0.0,
        num_classes=2,
    )
    assert torch.allclose(mean_evidence, mean_changed, atol=1e-10)
    assert torch.allclose(std_evidence, std_changed, atol=1e-10)
    assert torch.allclose(logits, logits_changed, atol=1e-10)
    assert torch.allclose(pooled, pooled_changed, atol=1e-10)
    mean_eps, std_eps, errors_eps, baselines_eps, logits_eps, _ = compute_tile_attribution(
        features,
        masks,
        model,
        ["mean", "standard_deviation"],
        epsilon=1e-4,
        num_classes=2,
    )
    reconstructed_eps = mean_eps.sum(dim=-1) + std_eps.sum(dim=-1) + baselines_eps
    assert torch.allclose(reconstructed_eps, logits_eps, atol=1e-8)
    assert float(errors_eps.max().item()) <= 1e-8


def test_attribution_grid_l1_identities() -> None:
    """Verify all-class L1 panels are the class-wise absolute sums.

    Args:
        None: Signed evidence arrays are constructed internally.

    Returns:
        None: Assertions verify L1 identities used by the figure grid.
    """
    mean_evidence = np.array(
        [[1.0, -0.5, 0.25], [0.0, 0.5, -1.0], [-0.25, 0.0, 0.75]],
        dtype=np.float64,
    )
    std_evidence = np.array(
        [[0.2, 0.1, 0.0], [-0.1, 0.0, 0.3], [0.05, -0.2, 0.1]],
        dtype=np.float64,
    )
    maps = attribution_grid_maps(mean_evidence, std_evidence)
    expected_class_l1 = np.abs(mean_evidence) + np.abs(std_evidence)
    assert np.allclose(maps["class_l1"], expected_class_l1)
    assert np.allclose(maps["all_l1"], expected_class_l1.sum(axis=0))
    assert np.allclose(maps["all_mean_abs"], np.abs(mean_evidence).sum(axis=0))
    assert np.allclose(maps["all_std_abs"], np.abs(std_evidence).sum(axis=0))


def test_center_tile_evidence_is_zero_sum() -> None:
    """Centred evidence subtracts the across-tile mean per class.

    Args:
        None: Synthetic evidence is constructed internally.

    Returns:
        None: Assertions verify zero-sum rows and the explicit formula.
    """
    evidence = np.array(
        [[1.0, -0.5, 0.25], [0.0, 0.5, -1.0]],
        dtype=np.float64,
    )
    centred = center_tile_evidence(evidence)
    assert centred.shape == evidence.shape
    assert np.allclose(centred.mean(axis=1), 0.0)
    expected = evidence - evidence.mean(axis=1, keepdims=True)
    assert np.allclose(centred, expected)


def test_evidence_contrast_ratio_constant_and_two_tile() -> None:
    """Contrast ratio is zero for a constant map and matches a two-tile case.

    Args:
        None: Synthetic evidence is constructed internally.

    Returns:
        None: Assertions verify the ``sd / |mean|`` formula.
    """
    constant = np.array([[2.0, 2.0, 2.0], [-1.0, -1.0, -1.0]], dtype=np.float64)
    assert np.allclose(evidence_contrast_ratio(constant), np.zeros(2))
    two_tile = np.array([[0.0, 2.0]], dtype=np.float64)
    ratio = evidence_contrast_ratio(two_tile)
    assert ratio.shape == (1,)
    assert ratio[0] == pytest.approx(1.0)


def test_within_tissue_effective_dimension_rank_cases() -> None:
    """Effective dimension is 0 for identical tiles and about 1 for rank-1 variation.

    Args:
        None: Synthetic tile matrices are constructed internally.

    Returns:
        None: Assertions verify the participation-ratio extremes.
    """
    identical = np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]], dtype=np.float64)
    assert within_tissue_effective_dimension(identical) == pytest.approx(0.0)
    rank_one = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=np.float64)
    assert within_tissue_effective_dimension(rank_one) == pytest.approx(1.0, abs=1e-6)
    single = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
    assert within_tissue_effective_dimension(single) == pytest.approx(0.0)


def test_center_multinomial_parameters_zero_class_mean() -> None:
    """Class-centring subtracts the across-class mean from weights and intercepts.

    Args:
        None: Synthetic parameter tensors are constructed internally.

    Returns:
        None: Assertions verify zero class-means and softmax-gauge invariance.
    """
    weights = torch.tensor(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        dtype=torch.float64,
    )
    intercept = torch.tensor([0.5, -0.1], dtype=torch.float64)
    centred_weights, centred_intercept = center_multinomial_parameters(
        weights, intercept
    )
    assert torch.allclose(centred_weights.mean(dim=0), torch.zeros(3, dtype=torch.float64))
    assert float(centred_intercept.sum().item()) == pytest.approx(0.0)
    features = torch.tensor([[0.2, -0.3, 0.4]], dtype=torch.float64)
    original = torch.nn.functional.linear(features, weights, intercept)
    centred = torch.nn.functional.linear(features, centred_weights, centred_intercept)
    assert torch.allclose(original - original.mean(dim=1, keepdim=True), centred)


def test_class_sum_gauge_requires_centred_coefficients() -> None:
    """The per-tile class-sum assertion fires unless coefficients are class-centred.

    Args:
        None: Synthetic bags and estimators are constructed internally.

    Returns:
        None: Assertions verify failure on an uncentred model, success after
            class-centring (explicit or automatic), and a missing-class skip.
    """
    features = torch.tensor([[[1.0, 2.0], [3.0, 6.0], [5.0, 2.0]]])
    masks = torch.tensor([[True, True, True]])
    uncentred = make_fitted_attribution_model(
        input_dim=2, num_classes=2, class_center=False
    )
    with pytest.raises(RuntimeError, match="class-sum gauge"):
        compute_tile_attribution(
            features,
            masks,
            uncentred,
            ["mean", "standard_deviation"],
            epsilon=0.0,
            num_classes=2,
            center_class_coefficients=False,
        )
    mean_auto, std_auto, _, _, _, _ = compute_tile_attribution(
        features,
        masks,
        uncentred,
        ["mean", "standard_deviation"],
        epsilon=0.0,
        num_classes=2,
    )
    auto_class_sum = (mean_auto + std_auto).sum(dim=1)
    assert float(auto_class_sum.abs().max().item()) <= 1e-8
    centred = make_fitted_attribution_model(
        input_dim=2, num_classes=2, class_center=True
    )
    mean_evidence, std_evidence, _, _, _, _ = compute_tile_attribution(
        features,
        masks,
        centred,
        ["mean", "standard_deviation"],
        epsilon=0.0,
        num_classes=2,
    )
    class_sum = (mean_evidence + std_evidence).sum(dim=1)
    assert float(class_sum.abs().max().item()) <= 1e-8
    missing = make_fitted_attribution_model(
        input_dim=2, num_classes=2, class_center=False
    )
    mean_missing, std_missing, _, _, _, _ = compute_tile_attribution(
        features,
        masks,
        missing,
        ["mean", "standard_deviation"],
        epsilon=0.0,
        num_classes=3,
    )
    assert mean_missing.shape[1] == 3
    assert torch.equal(mean_missing[:, 2], torch.zeros_like(mean_missing[:, 2]))
    assert torch.equal(std_missing[:, 2], torch.zeros_like(std_missing[:, 2]))


def test_visualization_loader_config_forces_unit_batch() -> None:
    """Attribution loading copies the eval config and sets ``batch_size`` to 1.

    Args:
        None: A small in-memory config mapping is used.

    Returns:
        None: Assertions verify the override and that the input is not mutated.
    """
    config = {"batch_size": 128, "num_workers": 8, "random_seed": 42}
    payload = visualization_loader_config(config)
    assert payload["batch_size"] == 1
    assert payload["num_workers"] == 8
    assert config["batch_size"] == 128


def test_compute_split_color_limits_are_shared() -> None:
    """Split-wide limits come from concatenated centred maps, not one tissue.

    Args:
        None: Two synthetic tissues are constructed internally.

    Returns:
        None: Assertions verify limits match the concatenated quantile.
    """
    mean_a = np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=np.float64)
    std_a = np.array([[0.0, 0.2], [0.1, -0.1]], dtype=np.float64)
    mean_b = np.array([[0.0, 10.0], [-10.0, 0.0]], dtype=np.float64)
    std_b = np.array([[0.0, 2.0], [1.0, -1.0]], dtype=np.float64)
    centred_means = [center_tile_evidence(mean_a), center_tile_evidence(mean_b)]
    centred_stds = [center_tile_evidence(std_a), center_tile_evidence(std_b)]
    limits = compute_split_color_limits(centred_means, centred_stds)
    combined_mean = np.concatenate(centred_means, axis=1)
    assert limits["mean"] == pytest.approx(evidence_color_limit(combined_mean))
    assert limits["mean"] > evidence_color_limit(centred_means[0])


def test_aggregate_split_diagnostics_means_and_medians() -> None:
    """Split diagnostics aggregate per-tissue contrast ratios and D_eff.

    Args:
        None: Two synthetic tissue records are constructed internally.

    Returns:
        None: Assertions verify mean and median summaries.
    """
    tissues = [
        {
            "mean_contrast_ratio": [0.0, 2.0],
            "std_contrast_ratio": [1.0, 3.0],
            "effective_dimension": 1.0,
        },
        {
            "mean_contrast_ratio": [4.0, 6.0],
            "std_contrast_ratio": [5.0, 7.0],
            "effective_dimension": 3.0,
        },
    ]
    diagnostics = aggregate_split_diagnostics(tissues)
    assert diagnostics["num_tissues"] == 2
    assert diagnostics["mean_contrast_ratio_mean"] == pytest.approx(3.0)
    assert diagnostics["mean_contrast_ratio_median"] == pytest.approx(3.0)
    assert diagnostics["std_contrast_ratio_mean"] == pytest.approx(4.0)
    assert diagnostics["effective_dimension_mean"] == pytest.approx(2.0)
    assert diagnostics["effective_dimension_median"] == pytest.approx(2.0)


def _axis_by_title(figure: Figure, title: str) -> plt.Axes:
    """Return the unique figure axis whose title matches ``title``.

    Args:
        figure (Figure): Attribution figure whose axes are searched.
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


def _axis_by_ylabel_prefix(figure: Figure, prefix: str) -> plt.Axes:
    """Return the unique figure axis whose y-axis label starts with ``prefix``.

    Args:
        figure (Figure): Attribution figure whose axes are searched.
        prefix (str): Required ylabel prefix.

    Returns:
        plt.Axes: The matching axis.
    """
    matches = [
        axis for axis in figure.axes if str(axis.get_ylabel()).startswith(prefix)
    ]
    if len(matches) != 1:
        raise AssertionError(
            f"Expected one axis labeled with prefix {prefix!r}, found {len(matches)}."
        )
    return matches[0]


def test_save_attribution_figure_writes_png(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Write one 3-by-(1+C) attribution figure without a source thumbnail.

    Args:
        tmp_path (Path): Pytest-managed temporary directory.
        monkeypatch (pytest.MonkeyPatch): Fixture used to inspect the figure.

    Returns:
        None: Assertions verify PNG output, left thumbnail, first-row titles,
            equal heatmap sizes, labeled shared colorbars, and an all-classes
            colorbar closer to its heatmap than to the first class map.
    """
    mean_evidence = np.array(
        [[1.0, -0.5, 0.25], [0.0, 0.5, -1.0]],
        dtype=np.float64,
    )
    std_evidence = np.array(
        [[0.2, 0.1, 0.0], [-0.1, 0.0, 0.3]],
        dtype=np.float64,
    )
    coordinates = np.array(
        [[0.0, 0.0], [448.0, 0.0], [0.0, 448.0]],
        dtype=np.float64,
    )
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
    output_path = tmp_path / "attribution.png"
    save_attribution_figure(
        mean_evidence=mean_evidence,
        std_evidence=std_evidence,
        coordinates=coordinates,
        class_names=("Alpha", "Beta"),
        predicted_index=1,
        slide_name="slide",
        tissue_name="tissue_0",
        true_class="Alpha",
        predicted_class="Beta",
        predicted_probability=0.8,
        class_probabilities=(0.2, 0.8),
        image_path=None,
        output_path=output_path,
        tile_size=448,
        thumbnail_size=64,
        render_dpi=50,
    )
    assert output_path.is_file()
    assert output_path.stat().st_size > 0
    figure = captured_figure["figure"]
    original_axis = _axis_by_title(figure, "Original")
    all_classes_axis = _axis_by_title(figure, "All classes")
    alpha_axis = _axis_by_title(figure, "Alpha (0.200)")
    beta_axis = _axis_by_title(figure, "★ Beta (0.800)")
    grid = original_axis.get_gridspec()
    assert grid is not None
    assert grid.nrows == 3
    assert grid.ncols == 6
    assert original_axis.get_position().x0 < all_classes_axis.get_position().x0
    assert {axis.get_title() for axis in figure.axes if axis.get_title()} == {
        "Original",
        "All classes",
        "Alpha (0.200)",
        "★ Beta (0.800)",
    }
    ylabel_prefixes = (
        "L1 magnitude (",
        "Mean evidence (",
        "Std evidence (",
        "L1 attribution magnitude (",
        "Signed mean logit evidence",
        "Signed std logit evidence",
    )
    for prefix in ylabel_prefixes:
        _axis_by_ylabel_prefix(figure, prefix)
    mean_label = _axis_by_ylabel_prefix(
        figure, "Signed mean logit evidence"
    ).get_ylabel()
    std_label = _axis_by_ylabel_prefix(
        figure, "Signed std logit evidence"
    ).get_ylabel()
    l1_label = _axis_by_ylabel_prefix(
        figure, "L1 attribution magnitude ("
    ).get_ylabel()
    assert mean_label == "Signed mean logit evidence"
    assert std_label == "Signed std logit evidence"
    assert "red supports" not in mean_label.lower()
    assert "blue opposes" not in mean_label.lower()
    assert "red supports" not in std_label.lower()
    assert "blue opposes" not in std_label.lower()
    assert "red supports" not in l1_label.lower()
    suptitle = figure._suptitle.get_text() if figure._suptitle is not None else ""
    assert "Red supports" not in suptitle
    assert "blue opposes" not in suptitle
    heatmap_widths = [
        all_classes_axis.get_position().width,
        alpha_axis.get_position().width,
        beta_axis.get_position().width,
    ]
    assert heatmap_widths[0] == pytest.approx(heatmap_widths[1], rel=0.02)
    assert heatmap_widths[1] == pytest.approx(heatmap_widths[2], rel=0.02)
    class_colorbar = _axis_by_ylabel_prefix(figure, "L1 attribution magnitude (")
    assert alpha_axis.get_position().x1 < class_colorbar.get_position().x0
    assert beta_axis.get_position().x1 < class_colorbar.get_position().x0
    all_colorbar = _axis_by_ylabel_prefix(figure, "L1 magnitude (")
    assert all_classes_axis.get_position().x1 < all_colorbar.get_position().x0
    assert all_colorbar.get_position().x1 < alpha_axis.get_position().x0
    left_gap = all_colorbar.get_position().x0 - all_classes_axis.get_position().x1
    right_gap = alpha_axis.get_position().x0 - all_colorbar.get_position().x1
    assert left_gap < right_gap


def test_save_attribution_figure_uses_shared_color_limits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Shared split colour limits appear in magnitude colorbar labels.

    Args:
        tmp_path (Path): Pytest-managed temporary directory.
        monkeypatch (pytest.MonkeyPatch): Fixture used to inspect the figure.

    Returns:
        None: Assertions verify magnitude numeric limits and signed name-only
            labels.
    """
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
    save_attribution_figure(
        mean_evidence=np.array([[1.0, -0.5, 0.25], [0.0, 0.5, -1.0]], dtype=np.float64),
        std_evidence=np.array([[0.2, 0.1, 0.0], [-0.1, 0.0, 0.3]], dtype=np.float64),
        coordinates=np.array(
            [[0.0, 0.0], [448.0, 0.0], [0.0, 448.0]], dtype=np.float64
        ),
        class_names=("Alpha", "Beta"),
        predicted_index=1,
        slide_name="slide",
        tissue_name="tissue_0",
        true_class="Alpha",
        predicted_class="Beta",
        predicted_probability=0.8,
        class_probabilities=(0.2, 0.8),
        image_path=None,
        output_path=tmp_path / "shared.png",
        tile_size=448,
        thumbnail_size=64,
        render_dpi=50,
        color_limits={
            "l1": 1.5,
            "all_l1": 2.5,
            "mean": 1.23,
            "std": 0.77,
            "all_mean": 3.1,
            "all_std": 4.2,
        },
    )
    figure = captured_figure["figure"]
    ylabels = [str(axis.get_ylabel()) for axis in figure.axes if axis.get_ylabel()]
    joined = "\n".join(ylabels)
    assert "Signed mean logit evidence" in joined
    assert "Signed std logit evidence" in joined
    assert "centred" not in joined
    assert "±" not in joined
    assert "0-1.5" in joined
    assert "0-2.5" in joined


def test_split_attribution_output_dir_nests_each_sample(tmp_path: Path) -> None:
    """Keep each visualization sample in its own subdirectory.

    Args:
        tmp_path (Path): Pytest-managed temporary directory.

    Returns:
        None: Assertions check per-split nesting under the output root.
    """
    assert split_attribution_output_dir(tmp_path, "train") == tmp_path / "train"
    assert split_attribution_output_dir(tmp_path, "val") == tmp_path / "val"
    assert split_attribution_output_dir(tmp_path, "test") == tmp_path / "test"
    with pytest.raises(ValueError, match="nonempty"):
        split_attribution_output_dir(tmp_path, "")


def test_visualize_attribution_renders_only_requested_splits(
    tmp_path: Path,
    synthetic_data_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Train a checkpoint and render only the configured test split.

    Args:
        tmp_path (Path): Pytest-managed temporary directory.
        synthetic_data_root (Path): Synthetic dataset root.
        monkeypatch (pytest.MonkeyPatch): Fixture used to capture loader config.

    Returns:
        None: Assertions verify test-only outputs under ``{output_dir}/test/``,
            PNG summaries, diagnostics, and a unit-batch attribution DataLoader.
    """
    config_path = write_config(
        tmp_path,
        synthetic_data_root,
        {
            "visualization": {
                "samples": ["test"],
                "tile_size": 448,
                "dpi": 50,
                "render_workers": 1,
                "thumbnail_size": 64,
            }
        },
    )
    artifacts = train_logistic_regression(str(config_path))
    heatmap_dir = tmp_path / "attribution"
    captured_batch_sizes: list[int] = []
    import logistic_regression.visualize_attribution as visualization_module

    original_create_dataloader = visualization_module.create_dataloader

    def _capture_batch_size(
        dataset: Dataset,
        config: Mapping[str, Any],
    ) -> Any:
        """Record the attribution DataLoader batch size, then build the loader.

        Args:
            dataset (Dataset): Split dataset passed to the loader factory.
            config (Mapping[str, Any]): Loading configuration.

        Returns:
            Any: The original DataLoader.
        """
        captured_batch_sizes.append(int(config["batch_size"]))
        return original_create_dataloader(dataset, config)

    monkeypatch.setattr(
        visualization_module, "create_dataloader", _capture_batch_size
    )
    manifest = visualize_attribution(
        config_path=str(config_path),
        checkpoint=artifacts["checkpoint"],
        output_dir=str(heatmap_dir),
        samples=["test"],
        dpi=50,
        render_workers=1,
    )
    assert captured_batch_sizes
    assert captured_batch_sizes == [1] * len(captured_batch_sizes)
    assert manifest["samples"] == ["test"]
    assert set(manifest["results"]) == {"test"}
    assert manifest["results"]["test"]["skipped"] is False
    split_dir = split_attribution_output_dir(heatmap_dir, "test")
    summary_path = Path(manifest["results"]["test"]["summary"])
    assert summary_path.is_file()
    assert summary_path.parent == split_dir
    assert Path(manifest["results"]["test"]["output_dir"]) == split_dir
    assert (heatmap_dir / "attribution_manifest.json").is_file()
    assert not (heatmap_dir / "train").exists()
    assert not (heatmap_dir / "val").exists()
    assert not (heatmap_dir / "attribution_summary_test.json").exists()
    assert not (heatmap_dir / "attribution_summary_train.json").exists()
    assert not (heatmap_dir / "attribution_summary_val.json").exists()
    diagnostics = manifest["results"]["test"]["diagnostics"]
    assert diagnostics is not None
    assert diagnostics["num_tissues"] > 0
    assert "effective_dimension_median" in diagnostics
    assert "mean_contrast_ratio_median" in diagnostics
    assert "std_contrast_ratio_median" in diagnostics
    assert "color_limits" in diagnostics
    assert diagnostics["color_limits"]["mean"] > 0.0
    tissues = manifest["results"]["test"]["tissues"]
    assert tissues
    for tissue in tissues:
        assert Path(tissue["heatmap_path"]).is_file()
        assert Path(tissue["heatmap_path"]).parent == split_dir
        assert tissue["split"] == "test"
        assert len(tissue["mean_contrast_ratio"]) == len(CLASS_NAMES)
        assert len(tissue["std_contrast_ratio"]) == len(CLASS_NAMES)
        assert tissue["effective_dimension"] >= 0.0
        assert "bag_mean_evidence_sum" in tissue
        assert "bag_std_evidence_sum" in tissue

