"""
Integration smoke test for the DG-SSM-MIL tissue workflow.
"""
import sys
from typing import Any, Dict

import torch

try:
    from .config_loader import (
        load_config,
        resolve_device,
        resolve_feature_file_suffix,
        resolve_input_dim,
    )
    from .dataset import DGSSMMILTissueDataset, collate_fn
    from .train import create_datasets
    from .model import DGSSMMILModel
except ImportError:
    from config_loader import (  # type: ignore
        load_config,
        resolve_device,
        resolve_feature_file_suffix,
        resolve_input_dim,
    )
    from dataset import DGSSMMILTissueDataset, collate_fn  # type: ignore
    from train import create_datasets  # type: ignore
    from model import DGSSMMILModel  # type: ignore


def _trim_sample(sample: Dict[str, Any], max_tiles: int) -> Dict[str, Any]:
    """
    Trim a sample to keep the integration test lightweight.

    Args:
        sample (Dict[str, Any]): Dataset sample containing features and coordinates.
        max_tiles (int): Maximum number of tiles to keep.

    Returns:
        Dict[str, Any]: Copy of the sample with features and coordinates trimmed.
    """
    trimmed = dict(sample)
    n_tiles = min(max_tiles, int(sample["features"].shape[0]))
    trimmed["features"] = sample["features"][:n_tiles]
    trimmed["coords"] = sample["coords"][:n_tiles]
    return trimmed


def _assert_batch_shapes(batch: Dict[str, Any]) -> None:
    """
    Assert that a collated DG-SSM-MIL batch has expected tensor shapes.

    Args:
        batch (Dict[str, Any]): Batch returned by `collate_fn`.

    Returns:
        None: Raises AssertionError if a shape is invalid.
    """
    assert batch["features"].ndim == 3
    assert batch["coords"].ndim == 3
    assert batch["coords"].shape[-1] == 2
    assert batch["masks"].shape == batch["features"].shape[:2]
    assert batch["labels"].ndim == 1


def _assert_attention_weights(attention_weights: torch.Tensor, mask: torch.Tensor) -> None:
    """
    Assert that MIL attention weights are finite and normalized on valid tiles.

    Args:
        attention_weights (torch.Tensor): Attention weights with shape `[B, N]`.
        mask (torch.Tensor): Boolean valid-tile mask with shape `[B, N]`.

    Returns:
        None: Raises AssertionError if the attention contract is invalid.
    """
    assert attention_weights.shape == mask.shape
    assert torch.isfinite(attention_weights).all()
    padded_weights = attention_weights.masked_select(~mask)
    assert torch.allclose(padded_weights, torch.zeros_like(padded_weights))
    valid_attention_sums = attention_weights.sum(dim=1)
    expected_sums = torch.ones_like(valid_attention_sums)
    assert torch.allclose(valid_attention_sums, expected_sums, atol=1e-5)


def test_integration() -> bool:
    """
    Run the DG-SSM-MIL integration smoke test.

    Args:
        None: The default `config.yml` is used.

    Returns:
        bool: True when data loading, forward pass, loss, and backward pass succeed.
    """
    print("Testing DG-SSM-MIL tissue workflow...")
    print("=" * 60)

    try:
        config = load_config()
        genbio_config = dict(config)
        genbio_config["feature_model"] = "genbio"
        assert resolve_feature_file_suffix(genbio_config) == "_features_genbio.pt"
        assert resolve_input_dim(genbio_config) == 4608
        train_dataset, _ = create_datasets(config)
        print(f"Loaded training samples: {len(train_dataset)}")
        print(f"Classes: {train_dataset.class_folders}")
    except Exception as exc:
        print(f"Data setup failed: {exc}")
        return False

    if len(train_dataset) == 0:
        print("No training samples found.")
        return False

    try:
        max_tiles = int(config.get("integration_max_tiles", 128))
        sample = _trim_sample(train_dataset[0], max_tiles)
        assert sample["features"].shape[-1] == int(config["input_dim"]), (
            "Feature dimension mismatch: "
            f"sample has {sample['features'].shape[-1]}, "
            f"config expects {config['input_dim']}"
        )
        batch = collate_fn([sample])
        _assert_batch_shapes(batch)
        print(f"Batch features: {tuple(batch['features'].shape)}")
        print(f"Batch coords: {tuple(batch['coords'].shape)}")
        print(f"Valid tiles: {int(batch['masks'].sum().item())}")
    except Exception as exc:
        print(f"Dataset/collate test failed: {exc}")
        return False

    try:
        device = torch.device(resolve_device(config))
        model = DGSSMMILModel.from_config(config).to(device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        features = batch["features"].to(device)
        coords = batch["coords"].to(device)
        masks = batch["masks"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(features, coords, masks)
        assert outputs["logits"].shape == (1, int(config["num_classes"]))
        _assert_attention_weights(outputs["attention_weights"], masks)
        loss = criterion(outputs["logits"], labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Logits: {tuple(outputs['logits'].shape)}")
        print(f"Attention: {tuple(outputs['attention_weights'].shape)}")
        print(f"Loss: {float(loss.item()):.4f}")
    except Exception as exc:
        print(f"Model/loss/backward test failed: {exc}")
        import traceback

        traceback.print_exc()
        return False

    print("=" * 60)
    print("DG-SSM-MIL integration smoke test passed.")
    return True


if __name__ == "__main__":
    success = test_integration()
    sys.exit(0 if success else 1)
