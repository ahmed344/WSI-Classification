"""Data-independent smoke verification for the canonical CLAM rewrite."""

from __future__ import annotations

import sys
from typing import Type

import torch
from torch import nn

try:
    from .clam_model import CLAM_MB, CLAM_SB
except ImportError:
    from clam_model import CLAM_MB, CLAM_SB


def _run_model_smoke(model_class: Type[nn.Module], bag_level: str) -> None:
    """Run a synthetic forward, loss, and optimizer step.

    Args:
        model_class (Type[nn.Module]): Canonical ``CLAM_SB`` or ``CLAM_MB`` class.
        bag_level (str): Synthetic provenance level, ``tissue`` or ``slide``.

    Returns:
        None: Assertions verify the canonical model and supervision contract.
    """
    if bag_level not in ("tissue", "slide"):
        raise ValueError("bag_level must be tissue or slide.")
    torch.manual_seed(17)
    model = model_class(
        input_dim=4,
        hidden_dim=8,
        attention_dim=4,
        num_classes=3,
        gated=True,
        attention_dropout=0.2,
        classifier_dropout=0.2,
        k_sample=8,
        subtyping=True,
        pooling_mode="distributional",
        pooling_use_variance=True,
        pooling_num_prototypes=4,
    )
    features = torch.randn(2, 5 if bag_level == "tissue" else 7, 4)
    masks = torch.ones(features.shape[:2], dtype=torch.bool)
    masks[0, -2:] = False
    labels = torch.tensor([0, 2], dtype=torch.long)
    outputs = model(features, mask=masks, labels=labels, instance_eval=True)

    expected_branches = 1 if model_class is CLAM_SB else 3
    assert outputs["logits"].shape == (2, 3)
    assert outputs["attention_weights"].shape == (
        2,
        expected_branches,
        features.shape[1],
    )
    padded = ~masks.unsqueeze(1).expand_as(outputs["attention_weights"])
    assert torch.all(outputs["attention_weights"].masked_select(padded) == 0)
    valid_counts = masks.sum(dim=1, keepdim=True).unsqueeze(1)
    expected_attention = torch.sigmoid(outputs["attention_scores"]) / valid_counts
    expected_attention = expected_attention.masked_fill(~masks.unsqueeze(1), 0.0)
    assert torch.allclose(outputs["attention_weights"], expected_attention)
    assert outputs["distribution_mean"].shape == (2, 1, model.hidden_dim)
    assert outputs["distribution_std"].shape == (2, 1, model.hidden_dim)
    assert outputs["prototype_assignments"].shape == (
        2,
        features.shape[1],
        model.pooling_num_prototypes,
    )
    assert outputs["prototype_histogram"].shape == (
        2,
        1,
        model.pooling_num_prototypes,
    )
    assert outputs["mean_gate"].shape == (2, expected_branches, 1)
    assert torch.allclose(
        outputs["mean_gate"],
        outputs["attention_weights"].sum(dim=-1, keepdim=True),
    )
    assert outputs["raw_pooled_features"].shape == (
        2,
        expected_branches,
        3 * model.hidden_dim + model.pooling_num_prototypes + 1,
    )
    assert torch.allclose(
        outputs["pooled_features"],
        model.pooling_layernorm(outputs["raw_pooled_features"]),
    )
    assert torch.isfinite(outputs["instance_loss"])
    assert set(outputs["instance_targets"].tolist()).issubset({0, 1})

    classification_loss = nn.functional.cross_entropy(outputs["logits"], labels)
    loss = 0.7 * classification_loss + 0.3 * outputs["instance_loss"]
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_integration() -> bool:
    """Run canonical CLAM smoke checks without external WSI data.

    Args:
        None: The compatibility entry point takes no arguments.

    Returns:
        bool: ``True`` when both architectures and bag levels pass.
    """
    try:
        for model_class in (CLAM_SB, CLAM_MB):
            for bag_level in ("tissue", "slide"):
                _run_model_smoke(model_class, bag_level)
    except Exception as error:
        print(f"Canonical CLAM integration failed: {error}")
        return False
    print("Canonical CLAM integration passed: SB/MB × tissue/slide")
    return True


test_integration.__test__ = False


def test_synthetic_canonical_integration() -> None:
    """Expose the boolean compatibility smoke check to pytest.

    Args:
        None: This pytest entry point takes no arguments.

    Returns:
        None: The compatibility result is asserted.
    """
    assert test_integration()


if __name__ == "__main__":
    sys.exit(0 if test_integration() else 1)
