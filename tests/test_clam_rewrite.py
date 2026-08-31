"""Unit tests for the canonical CLAM model and unified bag dataset."""

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from clam.clam_dataset import collate_fn, create_bag_dataset
from clam.clam_model import CLAM_MB, CLAM_SB
from clam.prototype_init import (
    collect_training_tile_features,
    compute_median_normalized_assignment_entropy,
    estimate_prototype_temperature,
    fit_prototype_centroids,
    initialize_prototype_assignment,
)


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
                features = torch.arange(base, base + 24, dtype=torch.float32).reshape(
                    6, 4
                )
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
def test_canonical_clam_forward_and_backward(
    model_class: type[torch.nn.Module],
) -> None:
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
        attention_dropout=0.0,
        classifier_dropout=0.0,
        k_sample=2,
        subtyping=True,
        pooling_mode="distributional",
        pooling_use_variance=True,
        pooling_num_prototypes=4,
    )
    features = torch.randn(2, 7, 4)
    masks = torch.tensor([[True, True, True, True, True, False, False], [True] * 7])
    labels = torch.tensor([0, 2])
    outputs = model(features, masks, labels, instance_eval=True)

    assert outputs["logits"].shape == (2, 3)
    assert outputs["attention_weights"].shape[0] == 2
    assert torch.all(
        outputs["attention_weights"].masked_select(~masks.unsqueeze(1)) == 0
    )
    valid_counts = masks.sum(dim=1, keepdim=True).unsqueeze(1)
    expected_attention = torch.sigmoid(outputs["attention_scores"]) / valid_counts
    expected_attention = expected_attention.masked_fill(~masks.unsqueeze(1), 0.0)
    assert torch.allclose(
        outputs["attention_weights"],
        expected_attention,
    )
    assert torch.all(outputs["attention_weights"].sum(dim=-1) < 1.0)
    expected_branches = 1 if model_class is CLAM_SB else 3
    assert outputs["raw_pooled_features"].shape == (2, expected_branches, 29)
    assert outputs["mean_gate"].shape == (2, expected_branches, 1)
    assert torch.allclose(
        outputs["mean_gate"],
        outputs["attention_weights"].sum(dim=-1, keepdim=True),
    )
    assert outputs["distribution_mean"].shape == (2, 1, 8)
    assert outputs["distribution_std"].shape == (2, 1, 8)
    assert outputs["prototype_assignments"].shape == (2, 7, 4)
    assert outputs["prototype_histogram"].shape == (2, 1, 4)
    assert torch.allclose(
        outputs["prototype_histogram"].sum(dim=-1),
        torch.ones(2, 1),
    )
    assert torch.allclose(
        outputs["logits"],
        outputs["base_logits"]
        + outputs["prevalence_contributions"]
        + outputs["detection_contributions"],
    )
    assert set(outputs["instance_targets"].tolist()).issubset({0, 1})

    loss = torch.nn.functional.cross_entropy(outputs["logits"], labels)
    loss = 0.7 * loss + 0.3 * outputs["instance_loss"]
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
    assert model.prototype_assignment is not None
    assert model.prototype_assignment.weight.grad is not None
    assert model.log_prototype_temperature is not None
    assert model.log_prototype_temperature.grad is not None


def test_detection_pooling_uses_masked_mean_and_top_k_gradients() -> None:
    """Verify prevalence averaging and stronger selected-tile detection gradients.

    Args:
        None: The test constructs a deterministic embedded bag.

    Returns:
        None: Values, selections, padding behavior, and gradients are asserted.
    """
    model = CLAM_MB(
        input_dim=2,
        hidden_dim=2,
        attention_dim=2,
        num_classes=2,
        attention_dropout=0.0,
        classifier_dropout=0.0,
        detection_top_k=2,
    )
    with torch.no_grad():
        model.tile_evidence.weight.copy_(torch.eye(2))
    embedded = torch.tensor(
        [[[1.0, 4.0], [3.0, 2.0], [2.0, 3.0], [99.0, 99.0]]],
        requires_grad=True,
    )
    mask = torch.tensor([[True, True, True, False]])

    scores, selected, prevalence, detection = model._pool_detection_evidence(
        embedded, mask
    )

    assert torch.equal(scores[0, :, 3], torch.zeros(2))
    assert torch.equal(selected.sum(dim=-1), torch.tensor([[2, 2]]))
    assert not bool(selected[:, :, 3].any())
    assert torch.allclose(prevalence, torch.tensor([[2.0, 3.0]]))
    assert torch.allclose(detection, torch.tensor([[2.5, 3.5]]))

    (prevalence[0, 0] + detection[0, 0]).backward()
    selected_gradient = 1.0 / 3.0 + 1.0 / 2.0
    assert torch.allclose(
        embedded.grad[0, 1:3, 0],
        torch.full((2,), selected_gradient),
    )
    assert torch.allclose(embedded.grad[0, 0, 0], torch.tensor(1.0 / 3.0))
    assert torch.count_nonzero(embedded.grad[0, 3]) == 0


def test_selective_pooling_normalization_preserves_nonstatistical_blocks() -> None:
    """Verify only mean and standard-deviation pooling blocks are normalized.

    Args:
        None: The test runs one deterministic distributional bag.

    Returns:
        None: Attention, histogram, and gate blocks retain their original scale.
    """
    model = CLAM_SB(
        input_dim=4,
        hidden_dim=8,
        attention_dim=4,
        num_classes=3,
        attention_dropout=0.0,
        classifier_dropout=0.0,
        pooling_mode="distributional",
        pooling_use_variance=True,
        pooling_num_prototypes=4,
    )
    model.eval()
    with torch.no_grad():
        outputs = model(torch.randn(2, 5, 4), instance_eval=False)

    hidden_dim = model.hidden_dim
    raw = outputs["raw_pooled_features"]
    normalized = outputs["pooled_features"]
    assert torch.equal(normalized[..., :hidden_dim], raw[..., :hidden_dim])
    assert torch.allclose(
        normalized[..., hidden_dim : 2 * hidden_dim],
        model.distribution_mean_layernorm(outputs["distribution_mean"]),
    )
    assert torch.allclose(
        normalized[..., 2 * hidden_dim : 3 * hidden_dim],
        model.distribution_std_layernorm(outputs["distribution_std"]),
    )
    assert torch.equal(normalized[..., 3 * hidden_dim :], raw[..., 3 * hidden_dim :])


@pytest.mark.parametrize("model_class", [CLAM_SB, CLAM_MB])
def test_sigmoid_attention_is_invariant_to_right_padding(
    model_class: type[torch.nn.Module],
) -> None:
    """Verify padding does not alter sigmoid-over-T pooling or predictions.

    Args:
        model_class (type[torch.nn.Module]): CLAM variant under test.

    Returns:
        None: Unpadded and padded outputs are compared.
    """
    model = model_class(
        input_dim=4,
        hidden_dim=8,
        attention_dim=4,
        num_classes=3,
        attention_dropout=0.0,
        classifier_dropout=0.0,
        pooling_mode="distributional",
        pooling_use_variance=True,
        pooling_num_prototypes=4,
    )
    model.eval()
    features = torch.randn(1, 5, 4)
    padded_features = torch.cat((features, torch.randn(1, 3, 4)), dim=1)
    padded_mask = torch.tensor([[True] * 5 + [False] * 3])

    with torch.no_grad():
        unpadded = model(features, instance_eval=False)
        padded = model(
            padded_features,
            mask=padded_mask,
            instance_eval=False,
        )

    assert torch.allclose(
        unpadded["attention_weights"],
        padded["attention_weights"][:, :, :5],
    )
    assert torch.allclose(
        unpadded["raw_pooled_features"], padded["raw_pooled_features"]
    )
    assert torch.allclose(unpadded["distribution_mean"], padded["distribution_mean"])
    assert torch.allclose(unpadded["distribution_std"], padded["distribution_std"])
    assert torch.allclose(
        unpadded["prototype_assignments"],
        padded["prototype_assignments"][:, :5],
    )
    assert torch.allclose(
        unpadded["prototype_histogram"], padded["prototype_histogram"]
    )
    assert torch.allclose(unpadded["mean_gate"], padded["mean_gate"])
    assert torch.count_nonzero(padded["prototype_assignments"][:, 5:]) == 0
    assert torch.allclose(unpadded["logits"], padded["logits"])


@pytest.mark.parametrize("model_class", [CLAM_SB, CLAM_MB])
def test_distributional_statistics_match_masked_population_moments(
    model_class: type[torch.nn.Module],
) -> None:
    """Verify statistics pooling uses uniform valid-tile population moments.

    Args:
        model_class (type[torch.nn.Module]): CLAM variant under test.

    Returns:
        None: Masked means and standard deviations are checked analytically.
    """
    model = model_class(
        input_dim=3,
        hidden_dim=4,
        attention_dim=2,
        num_classes=2,
        attention_dropout=0.0,
        classifier_dropout=0.0,
        pooling_mode="distributional",
        pooling_use_variance=True,
    )
    model.eval()
    features = torch.tensor(
        [
            [
                [1.0, 2.0, 3.0],
                [3.0, 4.0, 5.0],
                [5.0, 6.0, 7.0],
                [100.0, 100.0, 100.0],
            ]
        ]
    )
    masks = torch.tensor([[True, True, True, False]])

    with torch.no_grad():
        outputs = model(features, mask=masks, instance_eval=False)
        embedded = model.embedding(features)[:, :3]
        expected_mean = embedded.mean(dim=1, keepdim=True)
        expected_std = torch.sqrt(
            (embedded - expected_mean).square().mean(dim=1, keepdim=True)
            + model.statistics_pooling_epsilon
        )

    assert torch.allclose(outputs["distribution_mean"], expected_mean)
    assert torch.allclose(outputs["distribution_std"], expected_std)


def test_distributional_standard_deviation_has_finite_constant_bag_gradients() -> None:
    """Verify epsilon stabilizes standard deviation for constant embeddings.

    Args:
        None.

    Returns:
        None: Standard deviations and input gradients remain finite.
    """
    model = CLAM_SB(
        input_dim=3,
        hidden_dim=4,
        attention_dim=2,
        num_classes=2,
        attention_dropout=0.0,
        classifier_dropout=0.0,
        pooling_mode="distributional",
        pooling_use_variance=True,
    )
    features = torch.ones(1, 4, 3, requires_grad=True)
    outputs = model(features, instance_eval=False)
    outputs["logits"].sum().backward()

    assert torch.isfinite(outputs["distribution_std"]).all()
    assert torch.allclose(
        outputs["distribution_std"],
        torch.full_like(
            outputs["distribution_std"],
            model.statistics_pooling_epsilon**0.5,
        ),
    )
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()


def test_prototype_histogram_matches_masked_soft_assignments() -> None:
    """Verify histogram pooling averages valid-tile assignments uniformly.

    Args:
        None: This test constructs a deterministic synthetic model and bag.

    Returns:
        None: Analytic assignments and histogram values are asserted.
    """
    model = CLAM_SB(
        input_dim=3,
        hidden_dim=4,
        attention_dim=2,
        num_classes=2,
        attention_dropout=0.0,
        classifier_dropout=0.0,
        pooling_mode="distributional",
        pooling_use_variance=False,
        pooling_num_prototypes=3,
        pooling_prototype_temperature=0.75,
    )
    assert model.prototype_assignment is not None
    assert model.log_prototype_temperature is not None
    with torch.no_grad():
        model.prototype_assignment.weight.copy_(
            torch.tensor(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            )
        )
        model.prototype_assignment.bias.copy_(torch.tensor([0.1, -0.2, 0.3]))
    features = torch.tensor(
        [[[1.0, 2.0, 3.0], [2.0, 1.0, 0.0], [9.0, 9.0, 9.0]]]
    )
    masks = torch.tensor([[True, True, False]])

    with torch.no_grad():
        outputs = model(features, mask=masks, instance_eval=False)
        expected_assignments = torch.softmax(
            model.prototype_assignment(features) / 0.75, dim=-1
        )
        expected_assignments[:, 2:] = 0.0
        expected_histogram = expected_assignments[:, :2].mean(
            dim=1, keepdim=True
        )

    assert torch.allclose(
        outputs["prototype_assignments"], expected_assignments, atol=1e-6
    )
    assert torch.allclose(
        outputs["prototype_histogram"], expected_histogram, atol=1e-6
    )
    assert model.bag_feature_dim == 2 * model.hidden_dim + 3 + 1


def test_kmeans_prototype_initialization_is_deterministic_and_freezable(
    tmp_path: Path,
) -> None:
    """Verify training-only k-means initialization is repeatable and freezable.

    Args:
        tmp_path (Path): Temporary synthetic feature-data root.

    Returns:
        None: Samples, centroids, temperature, and frozen state are asserted.
    """
    _write_fixture_dataset(tmp_path)
    train_dataset = create_bag_dataset(
        _dataset_config(tmp_path, "tissue"), "train"
    )
    val_dataset = create_bag_dataset(_dataset_config(tmp_path, "tissue"), "val")
    torch.manual_seed(23)
    first_model = CLAM_SB(
        input_dim=4,
        hidden_dim=6,
        attention_dim=3,
        num_classes=2,
        pooling_mode="distributional",
        pooling_num_prototypes=3,
    )
    torch.manual_seed(23)
    second_model = CLAM_SB(
        input_dim=4,
        hidden_dim=6,
        attention_dim=3,
        num_classes=2,
        pooling_mode="distributional",
        pooling_num_prototypes=3,
    )
    first_embeddings = collect_training_tile_features(
        first_model, train_dataset, torch.device("cpu"), 20, 4, 31
    )
    second_embeddings = collect_training_tile_features(
        second_model, train_dataset, torch.device("cpu"), 20, 4, 31
    )
    first_centroids = fit_prototype_centroids(first_embeddings, 3, 4, 31)
    second_centroids = fit_prototype_centroids(second_embeddings, 3, 4, 31)
    temperature = estimate_prototype_temperature(
        first_embeddings, first_centroids, 4, 0.3
    )
    initialize_prototype_assignment(
        first_model, first_centroids, temperature, freeze_prototypes=True
    )

    assert np.array_equal(first_embeddings, second_embeddings)
    assert first_embeddings.shape == (20, first_model.input_dim)
    assert np.allclose(first_centroids, second_centroids)
    assert temperature > 0.0
    normalized_entropy = compute_median_normalized_assignment_entropy(
        first_embeddings,
        first_centroids,
        4,
        temperature,
    )
    assert normalized_entropy == pytest.approx(0.3, abs=1e-6)
    assert first_model.prototype_assignment is not None
    assert torch.allclose(
        first_model.prototype_assignment.weight,
        2.0 * torch.from_numpy(first_centroids),
    )
    assert torch.allclose(
        first_model.prototype_assignment.bias,
        -torch.from_numpy(first_centroids).square().sum(dim=1),
    )
    assert first_model.prototype_assignment.weight.requires_grad is False
    assert first_model.prototype_assignment.bias.requires_grad is False
    assert first_model.log_prototype_temperature is not None
    assert torch.allclose(
        first_model.log_prototype_temperature.exp(),
        torch.tensor(temperature),
    )
    assert first_model.log_prototype_temperature.requires_grad is True
    with pytest.raises(ValueError, match="training-split"):
        collect_training_tile_features(
            second_model, val_dataset, torch.device("cpu"), 20, 4, 31
        )


def test_attention_pooling_mode_preserves_original_bag_width() -> None:
    """Verify attention-only compatibility omits distributional statistics.

    Args:
        None.

    Returns:
        None: Original hidden width and attention pool are preserved.
    """
    model = CLAM_MB(
        input_dim=4,
        hidden_dim=8,
        attention_dim=4,
        num_classes=3,
        attention_dropout=0.0,
        classifier_dropout=0.0,
        pooling_mode="attention",
        pooling_use_variance=False,
    )
    outputs = model(torch.randn(2, 5, 4), instance_eval=False)

    assert model.bag_feature_dim == model.hidden_dim
    assert outputs["distribution_mean"].shape == (2, 1, 0)
    assert outputs["distribution_std"].shape == (2, 1, 0)
    assert torch.equal(
        outputs["raw_pooled_features"],
        outputs["attention_pooled_features"],
    )


def test_attention_dropout_does_not_contaminate_distribution_statistics() -> None:
    """Verify training-time attention dropout leaves population moments stable.

    Args:
        None.

    Returns:
        None: Attention scores vary while deterministic mean and standard
        deviation tensors remain identical.
    """
    model = CLAM_MB(
        input_dim=4,
        hidden_dim=8,
        attention_dim=4,
        num_classes=3,
        attention_dropout=0.5,
        classifier_dropout=0.0,
        pooling_mode="distributional",
        pooling_use_variance=True,
    )
    model.train()
    features = torch.randn(2, 32, 4)

    first = model(features, instance_eval=False)
    second = model(features, instance_eval=False)

    assert not any(
        isinstance(module, torch.nn.Dropout) for module in model.embedding.modules()
    )
    assert not torch.equal(first["attention_scores"], second["attention_scores"])
    assert torch.equal(first["distribution_mean"], second["distribution_mean"])
    assert torch.equal(first["distribution_std"], second["distribution_std"])


def test_classifier_dropout_only_changes_training_logits() -> None:
    """Verify classifier dropout follows stable pooled bag representations.

    Args:
        None.

    Returns:
        None: Training logits vary after identical pooling, while evaluation
        outputs are deterministic.
    """
    model = CLAM_SB(
        input_dim=4,
        hidden_dim=8,
        attention_dim=4,
        num_classes=3,
        attention_dropout=0.0,
        classifier_dropout=0.5,
        pooling_mode="distributional",
        pooling_use_variance=True,
    )
    features = torch.randn(2, 32, 4)
    model.train()
    first_train = model(features, instance_eval=False)
    second_train = model(features, instance_eval=False)

    assert torch.equal(first_train["pooled_features"], second_train["pooled_features"])
    assert not torch.equal(first_train["logits"], second_train["logits"])

    model.eval()
    with torch.no_grad():
        first_eval = model(features, instance_eval=False)
        second_eval = model(features, instance_eval=False)
    assert torch.equal(first_eval["pooled_features"], second_eval["pooled_features"])
    assert torch.equal(first_eval["logits"], second_eval["logits"])


def test_instance_supervision_keeps_raw_score_ranking() -> None:
    """Verify monotonic sigmoid scores select the same supervised instances.

    Args:
        None.

    Returns:
        None: Losses, predictions, and targets remain identical.
    """
    model = CLAM_MB(
        input_dim=4,
        hidden_dim=4,
        attention_dim=3,
        num_classes=2,
        attention_dropout=0.0,
        classifier_dropout=0.0,
        k_sample=2,
        subtyping=True,
    )
    embedded = torch.randn(1, 6, 4)
    scores = torch.tensor(
        [[[-4.0, 2.0, 0.5, 7.0, -1.0, 3.0], [1.0, -3.0, 5.0, 0.0, 4.0, -2.0]]]
    )
    mask = torch.ones(1, 6, dtype=torch.bool)
    labels = torch.tensor([1])

    raw_result = model._instance_supervision(embedded, scores, mask, labels)
    sigmoid_result = model._instance_supervision(
        embedded,
        torch.sigmoid(scores),
        mask,
        labels,
    )

    for raw_value, sigmoid_value in zip(raw_result, sigmoid_result):
        assert torch.equal(raw_value, sigmoid_value)


def test_clam_rejects_empty_bags() -> None:
    """Verify that all-padding bags fail before attention normalization.

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
    tissue_dataset = create_bag_dataset(_dataset_config(tmp_path, "tissue"), "train")
    slide_dataset = create_bag_dataset(_dataset_config(tmp_path, "slide"), "train")

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
