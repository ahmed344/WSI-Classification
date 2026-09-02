"""Focused unit and smoke tests for the independent PANTHER pipeline."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from panther.panther_dataset import build_datasets
from panther.panther_model import LinearClassifier, PANTHER
from panther.prototype import fit_prototypes
from panther.train_panther import aggregate_slide_embeddings, train_classifier
from panther.visualize_assignments import (
    blend_categorical_overlay,
    get_default_color_map,
)


def _config(root: Path) -> dict:
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
    color_map = get_default_color_map(16)
    assert color_map[0] == (105, 105, 105)
    assert color_map[15] == (255, 255, 84)

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
    assert np.any(overlay[1:5, 1:5] != preview[1:5, 1:5])
    assert np.any(overlay[7:11, 7:11] != preview[7:11, 7:11])
    assert np.array_equal(overlay[5:7, 5:7], preview[5:7, 5:7])

