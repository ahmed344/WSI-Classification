"""Tests for the paper-faithful DG-SSM-MIL implementation."""

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from dg_ssm_mil.config_loader import resolve_feature_file_suffix, resolve_input_dim
from dg_ssm_mil.dataset import DGSSMMILTissueDataset, collate_fn
from dg_ssm_mil.model import (
    DGSSMMILModel,
    DynamicGraphFusion,
    NonCausalMamba,
)
from dg_ssm_mil.visualize_attention import normalize_attention
from dg_ssm_mil.train import save_checkpoint, train_epoch, validate


def _write_dataset(root: Path, mismatch: bool = False) -> None:
    """Create a deterministic aligned feature/coordinate fixture.

    Args:
        root (Path): Temporary dataset root.
        mismatch (bool): Add one extra coordinate row to the first tissue.

    Returns:
        None: Fixture artifacts are written under `root`.
    """
    for class_idx, class_name in enumerate(("ClassA", "ClassB")):
        for slide_idx in range(8):
            slide_dir = root / class_name / f"slide_{slide_idx:02d}"
            slide_dir.mkdir(parents=True)
            features = torch.arange(24, dtype=torch.float32).reshape(6, 4)
            features = features + class_idx * 100 + slide_idx
            torch.save(features, slide_dir / "tissue_features.pt")
            row_count = 7 if mismatch and class_idx == 0 and slide_idx == 0 else 6
            pd.DataFrame(
                {
                    "x": np.arange(row_count, dtype=np.float32),
                    "y": np.arange(row_count, dtype=np.float32) * 2,
                }
            ).to_csv(slide_dir / "tissue_tiles.csv", index=False)


def _dataset_kwargs(root: Path) -> Dict[str, object]:
    """Build strict dataset keyword arguments.

    Args:
        root (Path): Fixture root.

    Returns:
        Dict[str, object]: Arguments accepted by `DGSSMMILTissueDataset`.
    """
    return {
        "data_root": str(root),
        "train_ratio": 0.5,
        "val_ratio": 0.25,
        "test_ratio": 0.25,
        "random_seed": 9,
        "feature_file_suffix": "_features.pt",
        "expected_feature_dim": 4,
        "sort_tiles_spatially": False,
        "coordinate_mismatch": "error",
    }


def test_dynamic_graph_matches_equations_6_to_12() -> None:
    """Compare dynamic graph output with a direct tiny-tensor calculation.

    Args:
        None.

    Returns:
        None: Assertions verify the article equations.
    """
    module = DynamicGraphFusion(
        hidden_dim=2,
        top_k=1,
        chunk_size=2,
        lambda_weight=0.0,
        activation="relu",
        dropout=0.0,
    )
    with torch.no_grad():
        module.additive_transform.weight.copy_(torch.eye(2))
        module.additive_transform.bias.zero_()
        module.multiplicative_transform.weight.copy_(torch.eye(2))
        module.multiplicative_transform.bias.zero_()
    h = torch.tensor([[[1.0, 0.5], [0.25, 1.0], [0.8, 0.2]]])
    t = torch.tensor([[[0.9, 0.1], [0.1, 1.1], [0.7, 0.3]]])
    mask = torch.ones(1, 3, dtype=torch.bool)
    actual = module(h, t, mask)

    expected_neighbors = []
    omega = torch.softmax(h[0] @ t[0].T, dim=-1)
    for index in range(3):
        candidates = omega[index].clone()
        candidates[index] = -1
        neighbor_idx = int(torch.argmax(candidates))
        expected_neighbors.append(t[0, neighbor_idx])
    neighbors = torch.stack(expected_neighbors)
    expected = torch.relu(h[0] + neighbors) + torch.relu(h[0] * neighbors)
    assert torch.allclose(actual[0], expected, atol=1e-6)


def test_noncausal_mamba_uses_future_context() -> None:
    """Verify symmetric convolution lets a future token affect the first output.

    Args:
        None.

    Returns:
        None: Output sensitivity is asserted.
    """
    torch.manual_seed(3)
    branch = NonCausalMamba(d_model=4, d_state=2, d_conv=3, expand=1)
    branch.eval()
    baseline = torch.zeros(1, 5, 4)
    baseline[:, 0] = torch.tensor([0.2, -0.3, 0.4, 0.1])
    changed = baseline.clone()
    changed[:, 1] = 1.0
    assert not torch.allclose(branch(baseline)[:, 0], branch(changed)[:, 0])


def test_model_is_padding_invariant_and_differentiable() -> None:
    """Check masks, padding invariance, attention normalization, and gradients.

    Args:
        None.

    Returns:
        None: Model contracts are asserted.
    """
    torch.manual_seed(4)
    model = DGSSMMILModel(
        input_dim=4,
        hidden_dim=8,
        num_classes=3,
        gat_heads=2,
        spatial_knn_k=2,
        dynamic_graph_top_k=2,
        mamba_d_state=2,
        mamba_d_conv=3,
        mamba_expand=1,
        projection_dropout=0.0,
        gat_dropout=0.0,
        dynamic_graph_dropout=0.0,
        block_dropout=0.0,
        classifier_dropout=0.0,
    )
    model.eval()
    features = torch.randn(1, 5, 4)
    coords = torch.tensor([[[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]]]).float()
    short = model(features, coords)
    padded_features = torch.cat([features, torch.zeros(1, 3, 4)], dim=1)
    padded_coords = torch.cat([coords, torch.full((1, 3, 2), float("nan"))], dim=1)
    padded_mask = torch.tensor([[True] * 5 + [False] * 3])
    padded = model(padded_features, padded_coords, padded_mask)
    assert torch.allclose(short["logits"], padded["logits"], atol=1e-5)
    assert torch.allclose(padded["attention_weights"].sum(1), torch.ones(1))
    padded["logits"].sum().backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_model_can_bypass_mamba_block() -> None:
    """Verify the optional Mamba block is omitted and bypassed when disabled.

    Args:
        None.

    Returns:
        None: Architecture, forward, and backward contracts are asserted.
    """
    base_config = {
        "input_dim": 4,
        "hidden_dim": 8,
        "num_classes": 3,
        "gat_heads": 2,
        "spatial_knn_k": 2,
        "dynamic_graph_top_k": 2,
        "mamba_d_state": 2,
        "mamba_d_conv": 3,
        "mamba_expand": 1,
        "projection_dropout": 0.0,
        "gat_dropout": 0.0,
        "dynamic_graph_dropout": 0.0,
        "block_dropout": 0.0,
        "classifier_dropout": 0.0,
    }
    enabled_model = DGSSMMILModel.from_config(base_config)
    assert enabled_model.sequence_block is not None

    disabled_model = DGSSMMILModel.from_config(
        {**base_config, "use_mamba_block": False}
    )
    assert disabled_model.sequence_block is None
    assert not any(
        key.startswith("sequence_block.") for key in disabled_model.state_dict()
    )

    features = torch.randn(2, 5, 4)
    coords = torch.randn(2, 5, 2)
    mask = torch.tensor(
        [[True, True, True, False, False], [True, True, True, True, True]]
    )
    outputs = disabled_model(features, coords, mask)
    assert torch.equal(outputs["sequence_features"], outputs["fused_features"])
    outputs["logits"].sum().backward()
    assert any(
        parameter.grad is not None for parameter in disabled_model.parameters()
    )


def test_dataset_splits_are_disjoint_and_preserve_indices(tmp_path: Path) -> None:
    """Verify slide-safe splitting and exact tile provenance.

    Args:
        tmp_path (Path): Pytest temporary directory.

    Returns:
        None: Split and collation contracts are asserted.
    """
    _write_dataset(tmp_path)
    datasets = {
        split: DGSSMMILTissueDataset(split=split, **_dataset_kwargs(tmp_path))
        for split in ("train", "val", "test")
    }
    slide_sets = {
        split: {
            datasets[split]._bags[index]["slide_key"]
            for index in datasets[split].indices
        }
        for split in datasets
    }
    assert slide_sets["train"].isdisjoint(slide_sets["val"])
    assert slide_sets["train"].isdisjoint(slide_sets["test"])
    assert slide_sets["val"].isdisjoint(slide_sets["test"])
    sample = datasets["train"][0]
    assert torch.equal(sample["tile_indices"], torch.arange(6))
    batch = collate_fn([sample])
    assert torch.equal(batch["tile_indices"][0], torch.arange(6))
    assert batch["tissue_indices"].shape == batch["masks"].shape


def test_dataset_rejects_coordinate_mismatch(tmp_path: Path) -> None:
    """Verify mismatched spatial artifacts fail instead of being truncated.

    Args:
        tmp_path (Path): Pytest temporary directory.

    Returns:
        None: The strict alignment exception is asserted.
    """
    _write_dataset(tmp_path, mismatch=True)
    datasets = [
        DGSSMMILTissueDataset(split=split, **_dataset_kwargs(tmp_path))
        for split in ("train", "val", "test")
    ]
    dataset, mismatched_index = next(
        (candidate, index)
        for candidate in datasets
        for index in range(len(candidate))
        if candidate._bags[candidate.indices[index]]["slide_name"] == "slide_00"
        and candidate._bags[candidate.indices[index]]["class_name"] == "ClassA"
    )
    with pytest.raises(ValueError, match="Feature/coordinate row mismatch"):
        dataset[mismatched_index]


def test_dataset_skips_tissues_above_configured_tile_limit(tmp_path: Path) -> None:
    """Verify oversized tissues are excluded before split construction.

    Args:
        tmp_path (Path): Pytest temporary directory.

    Returns:
        None: Filtered and skipped record counts are asserted.
    """
    _write_dataset(tmp_path)
    unfiltered = DGSSMMILTissueDataset(
        split="train",
        **_dataset_kwargs(tmp_path),
    )
    dataset = DGSSMMILTissueDataset(
        split="train",
        skip_tissues_above_tiles=5,
        **_dataset_kwargs(tmp_path),
    )
    assert len(dataset) == 0
    assert len(dataset.skipped_tissues) == len(unfiltered)
    assert {record["num_tiles"] for record in dataset.skipped_tissues} == {6}
    validation = DGSSMMILTissueDataset(
        split="val",
        skip_tissues_above_tiles=5,
        **_dataset_kwargs(tmp_path),
    )
    assert len(validation) > 0
    assert validation.skipped_tissues == []


@pytest.mark.parametrize(
    ("model_name", "suffix", "dimension"),
    [
        ("hoptimus", "_features_hoptimus.pt", 1536),
        ("uni2h", "_features_uni2h.pt", 1536),
        ("genbio", "_features_genbio.pt", 4608),
    ],
)
def test_all_feature_extractors_resolve(
    model_name: str, suffix: str, dimension: int
) -> None:
    """Verify all supported extractor artifacts resolve consistently.

    Args:
        model_name (str): Configured feature model name.
        suffix (str): Expected tensor filename suffix.
        dimension (int): Expected feature width.

    Returns:
        None: Resolver outputs are asserted.
    """
    config = {
        "feature_model": model_name,
        "feature_model_suffixes": {
            "hoptimus": "_features_hoptimus.pt",
            "uni2h": "_features_uni2h.pt",
            "genbio": "_features_genbio.pt",
        },
        "feature_model_input_dims": {
            "hoptimus": 1536,
            "uni2h": 1536,
            "genbio": 4608,
        },
    }
    assert resolve_feature_file_suffix(config) == suffix
    assert resolve_input_dim(config) == dimension


def test_attention_normalization_uses_percentile_ranks() -> None:
    """Verify article-style percentile attention normalization.

    Args:
        None.

    Returns:
        None: Percentile ranks are asserted.
    """
    normalized = normalize_attention(np.asarray([0.3, 0.1, 0.2]))
    assert np.allclose(normalized, np.asarray([1.0, 0.0, 0.5]))


def test_training_validation_and_checkpoint_round_trip(tmp_path: Path) -> None:
    """Exercise one optimizer step, validation metrics, and checkpoint reload.

    Args:
        tmp_path (Path): Pytest temporary directory.

    Returns:
        None: End-to-end training contracts are asserted.
    """
    samples = []
    for label in range(3):
        samples.append(
            {
                "features": torch.randn(5, 4),
                "coords": torch.randn(5, 2),
                "coordinates": torch.randn(5, 2),
                "label": label,
                "slide_name": f"slide_{label}",
                "tissue_name": f"tissue_{label}",
                "tissue_names": [f"tissue_{label}"],
                "tissue_slices": [(0, 5)],
                "num_tissues": 1,
                "tile_indices": torch.arange(5),
                "tissue_indices": torch.zeros(5, dtype=torch.long),
                "provenance": {},
                "feature_path": "fixture.pt",
                "tiles_path": "fixture.csv",
                "feature_paths": ["fixture.pt"],
                "tiles_paths": ["fixture.csv"],
            }
        )
    loader = DataLoader(samples, batch_size=3, collate_fn=collate_fn)
    model = DGSSMMILModel(
        input_dim=4,
        hidden_dim=8,
        num_classes=3,
        gat_heads=2,
        spatial_knn_k=2,
        dynamic_graph_top_k=2,
        mamba_d_state=2,
        mamba_d_conv=3,
        mamba_expand=1,
        projection_dropout=0.0,
        gat_dropout=0.0,
        dynamic_graph_dropout=0.0,
        block_dropout=0.0,
        classifier_dropout=0.0,
    )
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    train_metrics = train_epoch(
        model, loader, criterion, optimizer, torch.device("cpu"), 0.0
    )
    val_metrics = validate(model, loader, criterion, torch.device("cpu"))
    assert np.isfinite(train_metrics["loss"])
    assert np.isfinite(val_metrics["auc"])
    checkpoint_path = tmp_path / "checkpoint.pth"
    save_checkpoint(
        str(checkpoint_path),
        model,
        optimizer,
        scheduler,
        1,
        {"input_dim": 4},
        val_metrics,
        ["ClassA", "ClassB", "ClassC"],
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert checkpoint["class_folders"] == ["ClassA", "ClassB", "ClassC"]
    assert checkpoint["epoch"] == 1
