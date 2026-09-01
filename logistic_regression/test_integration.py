"""Data-free smoke test for the independent logistic-regression baseline."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch

try:
    from .model import (
        RawFeatureStatisticsPooler,
        TorchLogisticRegression,
    )
except ImportError:
    from model import RawFeatureStatisticsPooler, TorchLogisticRegression


def build_synthetic_bags() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create padded, linearly separable multiclass tile bags.

    Args:
        None: Synthetic data are generated internally.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Padded tile features,
            validity masks, and integer bag labels.
    """
    generator = torch.Generator().manual_seed(17)
    labels = torch.arange(3, dtype=torch.long).repeat_interleave(8)
    features = torch.full((labels.numel(), 6, 2), 10_000.0)
    masks = torch.zeros((labels.numel(), 6), dtype=torch.bool)
    centers = torch.tensor([[-4.0, -4.0], [0.0, 4.0], [4.0, -2.0]])
    for bag_index, label in enumerate(labels.tolist()):
        tile_count = 3 + bag_index % 4
        noise = 0.15 * torch.randn((tile_count, 2), generator=generator)
        features[bag_index, :tile_count] = centers[label] + noise
        masks[bag_index, :tile_count] = True
    return features, masks, labels


def run_smoke_test() -> None:
    """Exercise pooling, fitting, prediction, and joblib persistence.

    Args:
        None: The smoke test creates all data and temporary artifacts.

    Returns:
        None: Raises on contract failure and prints success otherwise.
    """
    features, masks, labels = build_synthetic_bags()
    pooler = RawFeatureStatisticsPooler()
    pooled = pooler(features, masks)

    changed_padding = features.clone()
    changed_padding[~masks] = -99_999.0
    if not torch.equal(pooled, pooler(changed_padding, masks)):
        raise AssertionError("Masked pooling changed when padding values changed.")

    model = TorchLogisticRegression(
        C=1.0, device="cuda", max_iter=500, random_seed=17
    )
    model.fit(pooled, labels)
    if model.fit_device_ != "cuda":
        raise AssertionError("Logistic regression was not fitted on CUDA.")
    predictions = model.predict(pooled)
    if not np.array_equal(predictions, labels.numpy()):
        raise AssertionError("Synthetic separable bags were not fitted exactly.")

    with tempfile.TemporaryDirectory() as temporary_directory:
        model_path = Path(temporary_directory) / "model.joblib"
        model.save(model_path)
        restored = TorchLogisticRegression.load(model_path)
        if not np.array_equal(restored.predict(pooled), predictions):
            raise AssertionError("Predictions changed after save/load.")
    print("Logistic-regression integration smoke test passed.")


def main() -> None:
    """Run the data-free integration smoke test.

    Args:
        None: This entry point takes no arguments.

    Returns:
        None: The smoke-test result is printed to standard output.
    """
    run_smoke_test()


if __name__ == "__main__":
    main()
