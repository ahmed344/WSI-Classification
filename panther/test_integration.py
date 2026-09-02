"""Focused unit and smoke tests for the independent PANTHER pipeline."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from panther.panther_dataset import build_datasets
from panther.panther_model import LinearClassifier, PANTHER
from panther.prototype import fit_prototypes
from panther.train_panther import (
    aggregate_slide_embeddings,
    plot_training_history,
    train_classifier,
)
from panther.visualize_assignments import (
    _safe_slug,
    assignment_output_paths,
    blend_categorical_overlay,
    get_default_color_map,
    save_mixture_plot,
    save_slide_overview,
)


def _config(root: Path) -> dict:
    """Build a tiny independent PANTHER config for synthetic data.

    Args:
        root (Path): Temporary data-root directory.

    Returns:
        dict: Resolved-style configuration mapping.
    """
    return {
        "data_root": str(root),
        "input_dim": 4,
        "feature_file_suffix": "_features.pt",
        "class_folders": None,
        "train_ratio": 0.5,
        "val_ratio": 0.25,
        "test_ratio": 0.25,
        "random_seed": 7,
        "feature_normalization": "none",
        "max_tiles_per_slide": None,
        "tile_sampling": "random",
        "prototype": {
            "method": "kmeans",
            "num_prototypes": 2,
            "patches_per_prototype": 8,
            "max_iterations": 10,
            "num_initializations": 1,
            "algorithm": "lloyd",
        },
        "model": {
            "em_iterations": 1,
            "tau": 1.0,
            "covariance_regularizer": 1.0,
            "variance_floor": 1e-6,
            "output_type": "allcat",
            "fix_prototypes": True,
            "em_chunk_size": 3,
        },
        "training": {
            "epochs": 2,
            "batch_size": 4,
            "learning_rate": 1e-3,
            "weight_decay": 1e-5,
            "optimizer": "adamw",
            "scheduler": "constant",
            "warmup_epochs": 0,
            "gradient_accumulation_steps": 1,
            "input_dropout": 0.0,
            "classifier_bias": False,
            "class_weighted_loss": False,
            "checkpoint_selection": "last",
        },
    }


def _write_data(root: Path) -> None:
    """Write two-class synthetic feature tensors and aligned tile CSVs.

    Args:
        root (Path): Temporary data-root directory.

    Returns:
        None: Files are written under ``root``.
    """
    generator = torch.Generator().manual_seed(12)
    for class_index, class_name in enumerate(("A", "B")):
        for slide_index in range(8):
            slide = root / class_name / f"slide_{slide_index:02d}"
            slide.mkdir(parents=True)
            features = torch.randn(6, 4, generator=generator) + 4 * class_index
            torch.save(features, slide / "tissue_features.pt")
            pd.DataFrame({"x": range(6), "y": range(6)}).to_csv(
                slide / "tissue_tiles.csv", index=False
            )


def test_panther_map_em_shapes_probabilities_and_chunk_invariance() -> None:
    """Verify MAP-EM outputs, posterior rows, and chunked equivalence.

    Args:
        None: The test uses a 3-tile synthetic bag.

    Returns:
        None: Assertions check shapes, probabilities, and numerical agreement.
    """
    prototypes = torch.tensor([[0.0, 0.0], [2.0, 2.0]])
    features = torch.tensor([[0.0, 0.1], [0.2, 0.0], [2.0, 2.1]])
    chunked = PANTHER(prototypes, em_chunk_size=2)
    unchunked = PANTHER(prototypes, em_chunk_size=None)
    first = chunked(features, return_assignments=True)
    second = unchunked(features, return_assignments=True)

    assert first["representation"].shape == (1, 10)
    assert first["assignments"].shape == (1, 3, 2)
    assert torch.allclose(first["assignments"].sum(dim=-1), torch.ones(1, 3))
    assert torch.all(first["variances"] > 0)
    assert torch.allclose(first["representation"], second["representation"], atol=1e-6)


def test_small_pipeline_is_independent_and_trainable(tmp_path: Path) -> None:
    """Verify a tiny dataset trains an independent linear classifier.

    Args:
        tmp_path (Path): Pytest temporary directory.

    Returns:
        None: Assertions check split counts, history length, and saved weights.
    """
    data_root = tmp_path / "data"
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_data(data_root)
    config = _config(data_root)
    datasets, classes, records = build_datasets(config)

    assert classes == ["A", "B"]
    assert [len(datasets[name]) for name in ("train", "val", "test")] == [8, 4, 4]
    assert len({record["slide_key"] for record in records}) == 16
    prototypes, _ = fit_prototypes(
        datasets["train"], config, run_dir / "prototypes.pkl"
    )
    panther = PANTHER(prototypes, em_chunk_size=3)
    embeddings = aggregate_slide_embeddings(
        datasets, panther, torch.device("cpu"), run_dir / "embeddings.pt"
    )
    classifier, history, details = train_classifier(
        embeddings, classes, config, torch.device("cpu"), run_dir
    )

    assert isinstance(classifier, LinearClassifier)
    assert len(history) == 2
    assert details["input_dim"] == 18
    assert (run_dir / "final_model.pth").is_file()
    assert (run_dir / "best_validation_model.pth").is_file()


def test_official_visualization_palette_and_center_coordinate_overlay() -> None:
    """Verify the purple-free palette, center-coordinate overlay, and no tile grid.

    Args:
        None: The test uses a synthetic white preview.

    Returns:
        None: Assertions check palette, tile interiors, and the absent dark border.
    """
    color_map = get_default_color_map(16)
    assert color_map[0] == (74, 74, 74)
    assert color_map[15] == (253, 216, 53)
    for red, green, blue in color_map.values():
        is_pink_or_purple = blue > 40 and red > 40 and green < 0.65 * min(red, blue)
        assert not is_pink_or_purple

    preview = np.full((12, 12, 3), 255, dtype=np.uint8)
    coordinates = np.asarray([[3.0, 3.0], [9.0, 9.0]])
    labels = np.asarray([0, 15])
    overlay = np.asarray(
        blend_categorical_overlay(
            preview,
            coordinates,
            labels,
            original_dimensions=(12, 12),
            tile_size=4,
            alpha=1.0,
            color_map=color_map,
        )
    )

    assert overlay.shape == preview.shape
    first_tile = overlay[1:5, 1:5]
    second_tile = overlay[7:11, 7:11]
    assert np.array_equal(first_tile, np.broadcast_to(color_map[0], first_tile.shape))
    assert np.array_equal(second_tile, np.broadcast_to(color_map[15], second_tile.shape))
    assert np.array_equal(overlay[5:7, 5:7], preview[5:7, 5:7])
    assert not np.any(np.all(overlay == np.array([50, 50, 50], dtype=np.uint8), axis=-1))


def test_plot_training_history_writes_png(tmp_path: Path) -> None:
    """Verify stacked train/val curves are written even when AUC is missing.

    Args:
        tmp_path (Path): Pytest temporary directory.

    Returns:
        None: Assertions check that a nonempty PNG is created.
    """
    history = [
        {
            "epoch": 1,
            "train": {
                "loss": 1.0,
                "accuracy": 0.4,
                "balanced_accuracy": 0.4,
                "macro_f1": 0.3,
                "multiclass_roc_auc": None,
            },
            "val": {
                "loss": 1.1,
                "accuracy": 0.35,
                "balanced_accuracy": 0.35,
                "macro_f1": 0.25,
                "multiclass_roc_auc": None,
            },
        },
        {
            "epoch": 2,
            "train": {
                "loss": 0.8,
                "accuracy": 0.5,
                "balanced_accuracy": 0.5,
                "macro_f1": 0.4,
                "multiclass_roc_auc": None,
            },
            "val": {
                "loss": 0.9,
                "accuracy": 0.45,
                "balanced_accuracy": 0.45,
                "macro_f1": 0.35,
                "multiclass_roc_auc": None,
            },
        },
    ]
    output_path = tmp_path / "training_history.png"
    plot_training_history(history, output_path, best_epoch=2)
    assert output_path.is_file()
    assert output_path.stat().st_size > 0


def test_save_slide_overview_pairs_originals_without_mixture(tmp_path: Path) -> None:
    """Verify the overview is a horizontal original-over-assignment figure.

    Args:
        tmp_path (Path): Pytest temporary directory.

    Returns:
        None: Assertions check the PNG exists and is a two-row layout.
    """
    original = Image.new("RGB", (16, 12), (255, 255, 255))
    overlay = Image.new("RGB", (16, 12), (255, 0, 0))
    output_path = tmp_path / "assignment_overview.png"
    save_slide_overview(
        [("tissue_a", original, overlay), ("tissue_b", original, overlay)],
        slide_name="slide_00",
        true_class="A",
        predicted_class="B",
        class_probabilities={"A": 0.2, "B": 0.8},
        split="test",
        output_path=output_path,
        dpi=50,
    )
    assert output_path.is_file()
    with Image.open(output_path) as image:
        width, height = image.size
    # Two tissues side by side, originals above assignments: wider than tall.
    assert width > 16
    assert width >= height


def test_assignment_outputs_are_flat_under_split_and_mixture_is_optional(
    tmp_path: Path,
) -> None:
    """Verify PNGs sit in the split directory and mixture output can be omitted.

    Args:
        tmp_path (Path): Pytest temporary directory.

    Returns:
        None: Assertions check flattened names and the optional mixture path.
    """
    enabled = assignment_output_paths(
        tmp_path,
        "test",
        "Dystrophic",
        "slide A",
        ["tissue_01"],
        save_mixture=True,
        save_individual_tissues=True,
    )
    disabled = assignment_output_paths(
        tmp_path,
        "test",
        "Dystrophic",
        "slide A",
        ["tissue_01"],
        save_mixture=False,
        save_individual_tissues=False,
    )
    split_dir = tmp_path / "test"
    overview = enabled["assignment_overview"]
    mixture = enabled["mixture_plot"]
    tissue_map = enabled["tissues"]["tissue_01"]
    assert overview.parent == split_dir
    assert mixture.parent == split_dir
    assert tissue_map.parent == split_dir
    assert overview.name.endswith("_assignment_overview.png")
    assert mixture.name.endswith("_mixture_proportions.png")
    assert tissue_map.name.endswith("_assignment.png")
    assert "tissue_assignment_maps" not in str(tissue_map)
    assert overview.parent == tissue_map.parent == mixture.parent
    assert disabled["mixture_plot"] is None
    assert disabled["tissues"] == {}
    assert _safe_slug("Dystrophic") in overview.name

    color_map = get_default_color_map(2)
    save_mixture_plot(np.asarray([0.7, 0.3]), color_map, mixture, dpi=50)
    assert mixture.is_file()

