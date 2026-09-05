"""Unit tests for the canonical CLAM model and unified bag dataset."""

from pathlib import Path
from typing import Dict

import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from clam.dataset import collate_fn, create_bag_dataset
from clam.model import CLAM_MB, CLAM_SB


def _write_fixture_dataset(root: Path) -> None:
    """Create a small deterministic WSI feature dataset.

    Args:
        root (Path): Directory in which class and slide folders are created.

    Returns:
        None: Fixture tensors and coordinate CSV files are written to disk.
    """
    for class_index, class_name in enumerate(("ClassA", "ClassB")):
        for slide_index in range(6):
            slide_dir = root / class_name / f"slide_{slide_index:02d}"
            slide_dir.mkdir(parents=True)
            for tissue_index in range(2):
                tissue_name = f"tissue_{tissue_index}"
                base = class_index * 1000 + slide_index * 100 + tissue_index * 10
                features = torch.arange(base, base + 24, dtype=torch.float32).reshape(6, 4)
                torch.save(features, slide_dir / f"{tissue_name}_features.pt")
                pd.DataFrame(
                    {
                        "x": torch.arange(6).numpy() + tissue_index * 100,
                        "y": torch.arange(6).numpy() + slide_index * 100,
                    }
                ).to_csv(slide_dir / f"{tissue_name}_tiles.csv", index=False)


def _dataset_config(root: Path, bag_level: str) -> Dict[str, object]:
    """Build a minimal resolved dataset configuration.

    Args:
        root (Path): Fixture data root.
        bag_level (str): Requested ``tissue`` or ``slide`` bag level.

    Returns:
        Dict[str, object]: Configuration accepted by ``create_bag_dataset``.
    """
    return {
        "data_root": str(root),
        "feature_file_suffix": "_features.pt",
        "input_dim": 4,
        "bag_level": bag_level,
        "train_ratio": 0.5,
        "val_ratio": 0.25,
        "test_ratio": 0.25,
        "random_seed": 7,
        "max_tiles_per_bag": {"train": 5, "val": 5, "test": 5},
        "tile_sampling": "random",
        "feature_normalization": "none",
    }


@pytest.mark.parametrize("model_class", [CLAM_SB, CLAM_MB])
def test_canonical_clam_forward_and_backward(model_class: type[torch.nn.Module]) -> None:
    """Verify canonical outputs, masks, instance targets, and gradients.

    Args:
        model_class (type[torch.nn.Module]): CLAM variant under test.

    Returns:
        None: Assertions verify the model contract.
    """
    model = model_class(
        input_dim=4,
        hidden_dim=8,
        attention_dim=4,
        num_classes=3,
        gated=True,
        dropout=0.0,
        k_sample=2,
        subtyping=True,
    )
    features = torch.randn(2, 7, 4)
    masks = torch.tensor(
        [[True, True, True, True, True, False, False], [True] * 7]
    )
    labels = torch.tensor([0, 2])
    outputs = model(features, masks, labels, instance_eval=True)

    assert outputs["logits"].shape == (2, 3)
    assert outputs["attention_weights"].shape[0] == 2
    assert torch.all(outputs["attention_weights"].masked_select(~masks.unsqueeze(1)) == 0)
    assert torch.allclose(
        outputs["attention_weights"].sum(dim=-1),
        torch.ones_like(outputs["attention_weights"].sum(dim=-1)),
    )
    assert set(outputs["instance_targets"].tolist()).issubset({0, 1})

    loss = torch.nn.functional.cross_entropy(outputs["logits"], labels)
    loss = 0.7 * loss + 0.3 * outputs["instance_loss"]
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_clam_rejects_empty_bags() -> None:
    """Verify that all-padding bags fail before attention softmax.

    Args:
        None.

    Returns:
        None: The expected ``ValueError`` is asserted.
    """
    model = CLAM_MB(input_dim=4, hidden_dim=8, attention_dim=4, num_classes=2)
    with pytest.raises(ValueError, match="All-empty bags"):
        model(torch.randn(1, 3, 4), torch.zeros(1, 3, dtype=torch.bool))


def test_tissue_and_slide_bags_preserve_alignment(tmp_path: Path) -> None:
    """Verify both bag levels keep sampled features and coordinates aligned.

    Args:
        tmp_path (Path): Pytest temporary directory.

    Returns:
        None: Assertions verify bag construction and collation.
    """
    _write_fixture_dataset(tmp_path)
    tissue_dataset = create_bag_dataset(
        _dataset_config(tmp_path, "tissue"), "train"
    )
    slide_dataset = create_bag_dataset(
        _dataset_config(tmp_path, "slide"), "train"
    )

    tissue_sample = tissue_dataset[0]
    repeated_sample = tissue_dataset[0]
    slide_sample = slide_dataset[0]

    assert tissue_sample["features"].shape == (5, 4)
    assert torch.equal(tissue_sample["features"], repeated_sample["features"])
    assert torch.equal(tissue_sample["tile_indices"], repeated_sample["tile_indices"])
    assert slide_sample["features"].shape == (5, 4)
    assert slide_sample["num_tissues"] == 2
    assert slide_sample["coordinates"].shape[0] == slide_sample["features"].shape[0]

    batch = collate_fn([tissue_sample, repeated_sample])
    assert batch["features"].shape == (2, 5, 4)
    assert batch["masks"].all()
    assert torch.equal(batch["tile_indices"][0], tissue_sample["tile_indices"])


def test_training_sampling_changes_only_after_epoch_update(tmp_path: Path) -> None:
    """Verify deterministic within-epoch sampling and seeded epoch refresh.

    Args:
        tmp_path (Path): Pytest temporary directory.

    Returns:
        None: Assertions verify the sampling lifecycle.
    """
    _write_fixture_dataset(tmp_path)
    dataset = create_bag_dataset(_dataset_config(tmp_path, "slide"), "train")
    first = dataset[0]["tile_indices"].clone()
    assert torch.equal(first, dataset[0]["tile_indices"])
    dataset.set_epoch(1)
    second = dataset[0]["tile_indices"]
    assert first.shape == second.shape
    assert not torch.equal(first, second)


def test_multiprocess_loader_observes_epoch_updates(tmp_path: Path) -> None:
    """Verify nonpersistent workers receive deterministic epoch changes.

    Args:
        tmp_path (Path): Pytest temporary directory.

    Returns:
        None: Assertions compare worker-produced sampling seeds and tile indices.
    """
    _write_fixture_dataset(tmp_path)
    dataset = create_bag_dataset(_dataset_config(tmp_path, "slide"), "train")
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        prefetch_factor=2,
        persistent_workers=False,
    )

    first_epoch = list(loader)
    repeated_epoch = list(loader)
    dataset.set_epoch(1)
    second_epoch = list(loader)

    first_seed = first_epoch[0]["provenance"][0]["sampling_seed"]
    repeated_seed = repeated_epoch[0]["provenance"][0]["sampling_seed"]
    second_seed = second_epoch[0]["provenance"][0]["sampling_seed"]
    assert first_seed == repeated_seed
    assert first_seed != second_seed
    assert torch.equal(
        first_epoch[0]["tile_indices"],
        repeated_epoch[0]["tile_indices"],
    )
    assert not torch.equal(
        first_epoch[0]["tile_indices"],
        second_epoch[0]["tile_indices"],
    )

