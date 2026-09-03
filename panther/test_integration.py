"""Focused unit and smoke tests for the independent PANTHER pipeline."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image

from panther.panther_dataset import build_datasets
from panther.panther_model import LinearClassifier, PANTHER
from panther.prototype import fit_prototypes
from panther.train_panther import (
    aggregate_slide_embeddings,
    component_embedding_paths,
    compose_slide_embeddings,
    plot_training_history,
    run_training,
    select_primary_output_type,
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
            "bag_level": "slide",
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
            "output_type": ["allcat", "pi", "mean", "variance"],
            "fix_prototypes": True,
            "em_chunk_size": 3,
        },
        "training": {
            "bag_level": "slide",
            "checkpoint_level": "slide",
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
            for tissue_index in range(2):
                features = (
                    torch.randn(6, 4, generator=generator)
                    + 4 * class_index
                    + 0.1 * tissue_index
                )
                name = f"tissue_{tissue_index}"
                torch.save(features, slide / f"{name}_features.pt")
                pd.DataFrame({"x": range(6), "y": range(6)}).to_csv(
                    slide / f"{name}_tiles.csv", index=False
                )


def test_tissue_and_slide_bag_views_share_slide_splits(tmp_path: Path) -> None:
    """Verify bag levels change units without introducing split leakage."""
    data_root = tmp_path / "data"
    _write_data(data_root)
    config = _config(data_root)
    slide_datasets, classes, records = build_datasets(config, bag_level="slide")
    assignments = {record["slide_key"]: record["split"] for record in records}
    tissue_datasets, tissue_classes, _ = build_datasets(
        config,
        class_folders=classes,
        split_assignments=assignments,
        bag_level="tissue",
    )

    assert tissue_classes == classes
    assert len(tissue_datasets["train"]) == 2 * len(slide_datasets["train"])
    for split in ("train", "val", "test"):
        assert {
            record["slide_key"] for record in tissue_datasets[split].records
        } == {
            record["slide_key"] for record in slide_datasets[split].records
        }
        assert all(
            len(record["tissues"]) == 1
            for record in tissue_datasets[split].records
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


def _history_entry(balanced_accuracy: float, loss: float) -> dict:
    """Build one epoch record with the validation scores used for root selection.

    Args:
        balanced_accuracy (float): Validation balanced accuracy.
        loss (float): Validation loss.

    Returns:
        dict: History entry containing a ``val`` metric mapping.
    """
    return {
        "val": {
            "balanced_accuracy": balanced_accuracy,
            "loss": loss,
        }
    }


def test_select_primary_output_type_prefers_higher_val_balanced_accuracy() -> None:
    """Verify the root checkpoint winner is the highest validation balanced accuracy.

    Args:
        None: The test uses two fake one-epoch histories.

    Returns:
        None: Assertions check the selected output type and recorded scores.
    """
    trained_models = {
        "allcat": {"training_details": {"selected_epoch": 1, "input_dim": 10}},
        "pi": {"training_details": {"selected_epoch": 1, "input_dim": 2}},
    }
    histories = {
        "allcat": [_history_entry(0.40, 1.10)],
        "pi": [_history_entry(0.55, 1.20)],
    }
    winner, record = select_primary_output_type(
        trained_models, histories, ("allcat", "pi")
    )
    assert winner == "pi"
    assert record["winner"] == "pi"
    assert record["metric"] == "val_balanced_accuracy"
    assert record["candidates"][1]["val_balanced_accuracy"] == 0.55


def test_select_primary_output_type_breaks_ties_with_val_loss() -> None:
    """Verify equal balanced accuracy prefers the lower validation loss.

    Args:
        None: The test uses two fake one-epoch histories with equal balanced accuracy.

    Returns:
        None: Assertions check that the lower-loss representation wins.
    """
    trained_models = {
        "allcat": {"training_details": {"selected_epoch": 1, "input_dim": 10}},
        "mean": {"training_details": {"selected_epoch": 1, "input_dim": 8}},
    }
    histories = {
        "allcat": [_history_entry(0.50, 1.30)],
        "mean": [_history_entry(0.50, 1.05)],
    }
    winner, record = select_primary_output_type(
        trained_models, histories, ("allcat", "mean")
    )
    assert winner == "mean"
    assert record["candidates"][0]["val_loss"] == 1.30
    assert record["candidates"][1]["val_loss"] == 1.05


def test_panther_selects_each_output_component() -> None:
    """Verify every supported output type uses the intended flattened block.

    Args:
        None: The test uses a 3-tile synthetic bag.

    Returns:
        None: Assertions check representation shapes and component identity.
    """
    prototypes = torch.tensor([[0.0, 0.0], [2.0, 2.0]])
    features = torch.tensor([[0.0, 0.1], [0.2, 0.0], [2.0, 2.1]])
    expected_keys = {
        "pi": "mixture_weights",
        "mean": "means",
        "variance": "variances",
    }
    outputs = {
        output_type: PANTHER(prototypes, output_type=output_type)(features)
        for output_type in ("allcat", "pi", "mean", "variance")
    }

    assert outputs["allcat"]["representation"].shape == (1, 10)
    assert outputs["pi"]["representation"].shape == (1, 2)
    assert outputs["mean"]["representation"].shape == (1, 4)
    assert outputs["variance"]["representation"].shape == (1, 4)
    for output_type, result_key in expected_keys.items():
        assert torch.equal(
            outputs[output_type]["representation"],
            outputs[output_type][result_key].flatten(start_dim=1),
        )


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
    component_embeddings = aggregate_slide_embeddings(
        datasets, panther, torch.device("cpu"), run_dir
    )
    assert all(
        path.is_file() for path in component_embedding_paths(run_dir).values()
    )
    embeddings = compose_slide_embeddings(component_embeddings, "allcat")
    model_dir = run_dir / "models" / "allcat"
    classifier, history, details = train_classifier(
        embeddings, classes, config, torch.device("cpu"), model_dir
    )

    assert isinstance(classifier, LinearClassifier)
    assert len(history) == 2
    assert details["input_dim"] == 18
    assert (model_dir / "final_model.pth").is_file()
    assert (model_dir / "best_validation_model.pth").is_file()


def test_training_and_evaluation_run_every_requested_output(
    tmp_path: Path, capsys
) -> None:
    """Verify one run trains every representation and copies the val-bacc winner.

    Args:
        tmp_path (Path): Pytest temporary directory.

    Returns:
        None: Assertions check per-type artifacts, evaluation outputs, and the
            run-root primary head.
    """
    data_root = tmp_path / "data"
    _write_data(data_root)
    run_dir = tmp_path / "run"
    config = _config(data_root)
    config["prototype"]["bag_level"] = "tissue"
    config["training"].update(
        {
            "bag_level": "tissue",
            "checkpoint_level": "slide",
            "checkpoint_selection": "best_val_balanced_accuracy",
        }
    )
    config.update(
        {
            "output_dir": str(tmp_path / "results"),
            "feature_model": "default",
            "feature_model_suffixes": {"default": "_features.pt"},
            "feature_model_input_dims": {"default": 4},
            "evaluation": {
                "run_after_training": True,
                "bag_levels": ["tissue", "slide"],
                "splits": ["test"],
                "include_train": False,
                "confusion_matrix_dpi": 50,
            },
            "visualization": {
                "run_after_training": False,
                "split": "test",
                "max_slides": 1,
                "tile_size": 448,
                "downsample": 128,
                "alpha": 0.4,
                "dpi": 50,
                "save_mixture_proportions": False,
                "save_individual_tissues": False,
            },
            "runtime": {
                "device": "cpu",
                "num_workers": 0,
                "pin_memory": False,
            },
            "paths": {
                "run_dir": str(run_dir),
                "checkpoint": None,
                "evaluation_output": None,
                "visualization_output": None,
            },
        }
    )
    config_path = tmp_path / "config.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle)

    assert run_training(str(config_path)) == run_dir.resolve()
    output = capsys.readouterr().out
    assert "Epoch metric order: train_tissue, val_tissue, val_slide" in output
    assert "| loss=" in output
    assert "| acc=" in output
    assert "| bal_acc=" in output
    output_types = ("allcat", "pi", "mean", "variance")
    assert set(component_embedding_paths(run_dir)) == {"pi", "mean", "variance"}
    assert all(
        path.is_file()
        for path in component_embedding_paths(run_dir, "tissue").values()
    )
    for output_type in output_types:
        model_dir = run_dir / "models" / output_type
        assert (model_dir / "best_model.pth").is_file()
        assert (model_dir / "training_history.json").is_file()
        assert (model_dir / "training_history.png").is_file()

    with (run_dir / "evaluation_results" / "evaluation_manifest.json").open(
        encoding="utf-8"
    ) as handle:
        manifest = json.load(handle)
    assert manifest["output_types"] == list(output_types)
    assert set(manifest["results"]) == set(output_types)
    for output_type in output_types:
        evaluation_dir = run_dir / "evaluation_results" / output_type
        assert (evaluation_dir / "evaluation_manifest.json").is_file()
        assert (evaluation_dir / "slide_test_predictions.csv").is_file()
        assert (evaluation_dir / "tissue_test_predictions.csv").is_file()

    histories = {}
    trained_models = {}
    for output_type in output_types:
        history_path = run_dir / "models" / output_type / "training_history.json"
        with history_path.open(encoding="utf-8") as handle:
            histories[output_type] = json.load(handle)
        trained_models[output_type] = torch.load(
            run_dir / "models" / output_type / "best_model.pth",
            map_location="cpu",
            weights_only=False,
        )
    winner, selection = select_primary_output_type(
        trained_models, histories, output_types
    )
    checkpoint = torch.load(
        run_dir / "best_model.pth", map_location="cpu", weights_only=False
    )
    assert checkpoint["default_output_type"] == winner
    assert checkpoint["root_model_selection"]["winner"] == winner
    assert checkpoint["training_details"]["output_type"] == winner
    assert (
        checkpoint["training_details"]["input_dim"]
        == trained_models[winner]["training_details"]["input_dim"]
    )
    assert selection["metric"] == "val_slide_balanced_accuracy"
    assert checkpoint["training_details"]["training_level"] == "tissue"
    assert checkpoint["training_details"]["checkpoint_level"] == "slide"
    assert checkpoint["prototype_metadata"]["bag_level"] == "tissue"
    assert all(
        {"train", "val_tissue", "val_slide"} <= set(entry)
        for entry in histories[winner]
    )


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
