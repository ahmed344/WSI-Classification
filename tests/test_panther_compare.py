"""Tests for PANTHER feature-model comparison tables and run reuse."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import torch
import yaml

from panther.compare_feature_models import (
    CSV_FIELDNAMES,
    comparison_row,
    experiment_grid,
    find_matching_run,
    has_complete_evaluation,
    representation_dimension,
    reuse_signature,
    rows_from_evaluation_manifest,
    run_matches_grid_cell,
    write_comparison_csv,
    write_comparison_markdown,
    write_experiment_config,
)
from panther.panther_model import PANTHER
from panther.train_panther import MODEL_SCHEMA


def _signature() -> Dict[str, Any]:
    """Return a compact reusable-run hyperparameter signature.

    Args:
        None: This helper uses fixed values.

    Returns:
        Dict[str, Any]: Signature matching ``reuse_signature`` output.
    """
    return {
        "random_seed": 42,
        "train_ratio": 0.8,
        "val_ratio": 0.1,
        "test_ratio": 0.1,
        "feature_normalization": "none",
        "max_tiles_per_slide": None,
        "tile_sampling": "random",
        "prototype": {
            "method": "kmeans",
            "patches_per_prototype": 10000,
            "max_iterations": 50,
            "num_initializations": 5,
            "algorithm": "lloyd",
        },
        "model": {
            "em_iterations": 1,
            "tau": 1.0,
            "covariance_regularizer": 1.0,
            "variance_floor": 1.0e-6,
            "fix_prototypes": True,
            "em_chunk_size": 65536,
        },
        "training": {
            "epochs": 200,
            "batch_size": 256,
            "learning_rate": 0.0001,
            "weight_decay": 0.00001,
            "optimizer": "adamw",
            "scheduler": "cosine",
            "warmup_epochs": 1,
            "gradient_accumulation_steps": 1,
            "input_dropout": 0.0,
            "classifier_bias": False,
            "class_weighted_loss": True,
            "checkpoint_selection": "last",
        },
    }


def _resolved_config(
    feature_model: str,
    num_prototypes: int,
    input_dim: int,
) -> Dict[str, Any]:
    """Build a resolved-config mapping for one synthetic run.

    Args:
        feature_model (str): Extractor name.
        num_prototypes (int): Prototype count.
        input_dim (int): Tile feature width.

    Returns:
        Dict[str, Any]: JSON-serializable resolved configuration.
    """
    signature = _signature()
    prototype = dict(signature["prototype"])
    prototype["num_prototypes"] = int(num_prototypes)
    return {
        "feature_model": feature_model,
        "input_dim": int(input_dim),
        "random_seed": signature["random_seed"],
        "train_ratio": signature["train_ratio"],
        "val_ratio": signature["val_ratio"],
        "test_ratio": signature["test_ratio"],
        "feature_normalization": signature["feature_normalization"],
        "max_tiles_per_slide": signature["max_tiles_per_slide"],
        "tile_sampling": signature["tile_sampling"],
        "prototype": prototype,
        "model": {
            **signature["model"],
            "output_type": ["allcat", "pi", "mean", "variance"],
        },
        "training": signature["training"],
    }


def _split_metrics(
    accuracy: float,
    balanced_accuracy: float,
    num_slides: int = 70,
) -> Dict[str, Any]:
    """Build one evaluation-metric mapping.

    Args:
        accuracy (float): Overall accuracy.
        balanced_accuracy (float): Balanced accuracy.
        num_slides (int): Number of evaluated slides.

    Returns:
        Dict[str, Any]: Metric dictionary with the keys collected by comparison.
    """
    return {
        "num_slides": num_slides,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": balanced_accuracy - 0.01,
        "multiclass_roc_auc": min(0.99, balanced_accuracy + 0.2),
        "loss": 1.2 - balanced_accuracy,
    }


def _write_completed_run(
    root: Path,
    name: str,
    feature_model: str,
    num_prototypes: int,
    input_dim: int,
    include_evaluation: bool = True,
) -> Path:
    """Create a synthetic completed PANTHER run directory.

    Args:
        root (Path): Parent results directory.
        name (str): Dated run folder name.
        feature_model (str): Extractor name.
        num_prototypes (int): Prototype count.
        input_dim (int): Tile feature width.
        include_evaluation (bool): When ``False``, omit the evaluation manifest.

    Returns:
        Path: Created run directory.
    """
    run_dir = root / name
    run_dir.mkdir(parents=True)
    resolved = _resolved_config(feature_model, num_prototypes, input_dim)
    (run_dir / "resolved_config.json").write_text(
        json.dumps(resolved), encoding="utf-8"
    )
    for component in ("pi", "mean", "variance"):
        torch.save({"train": {}}, run_dir / f"slide_embeddings_{component}.pt")
    models: Dict[str, Dict[str, str]] = {}
    for output_type in ("allcat", "pi", "mean", "variance"):
        model_dir = run_dir / "models" / output_type
        model_dir.mkdir(parents=True)
        torch.save({"output_type": output_type}, model_dir / "best_model.pth")
        models[output_type] = {"output_type": output_type}
    torch.save(
        {"model_schema": MODEL_SCHEMA, "models": models},
        run_dir / "best_model.pth",
    )
    if include_evaluation:
        results = {}
        for output_type in ("allcat", "pi", "mean", "variance"):
            results[output_type] = {
                "slide/val": _split_metrics(0.4, 0.41, 69),
                "slide/test": _split_metrics(0.5, 0.52, 70),
            }
        evaluation_dir = run_dir / "evaluation_results"
        evaluation_dir.mkdir()
        (evaluation_dir / "evaluation_manifest.json").write_text(
            json.dumps({"results": results}),
            encoding="utf-8",
        )
    return run_dir


def test_experiment_grid_is_extractor_major_then_prototype_count() -> None:
    """Verify the nine extractor × prototype cells keep report order.

    Args:
        None: The test uses the default three-by-three grid.

    Returns:
        None: Assertions check pair order and length.
    """
    grid = experiment_grid(("hoptimus", "uni2h", "genbio"), (8, 16, 32))
    assert len(grid) == 9
    assert grid[0] == ("hoptimus", 8)
    assert grid[2] == ("hoptimus", 32)
    assert grid[3] == ("uni2h", 8)
    assert grid[-1] == ("genbio", 32)


def test_representation_dimension_matches_panther_module() -> None:
    """Verify table widths match the live PANTHER output_dim property.

    Args:
        None: The test constructs a two-prototype, four-dimensional model.

    Returns:
        None: Assertions check each output type against ``PANTHER.output_dim``.
    """
    prototypes = torch.zeros(2, 4)
    for output_type in ("allcat", "pi", "mean", "variance"):
        model = PANTHER(prototypes, output_type=output_type)
        assert representation_dimension(output_type, 2, 4) == model.output_dim
    assert representation_dimension("allcat", 16, 1536) == 49168
    assert representation_dimension("pi", 16, 1536) == 16
    assert representation_dimension("mean", 8, 4608) == 36864


def test_comparison_row_and_manifest_collection_emit_four_heads(
    tmp_path: Path,
) -> None:
    """Verify one run expands to four table rows in configured head order.

    Args:
        tmp_path (Path): Pytest-managed temporary directory.

    Returns:
        None: Assertions check row count, field names, and derived dimensions.
    """
    run_dir = _write_completed_run(tmp_path, "2026-09-02_161803", "uni2h", 16, 1536)
    rows = rows_from_evaluation_manifest(
        run_dir, ("allcat", "pi", "mean", "variance")
    )
    assert [row["output_type"] for row in rows] == [
        "allcat",
        "pi",
        "mean",
        "variance",
    ]
    assert all(set(row) == set(CSV_FIELDNAMES) for row in rows)
    assert rows[0]["representation_dim"] == 49168
    assert rows[1]["representation_dim"] == 16
    assert rows[0]["test_bacc"] == 0.52
    assert rows[0]["val_slides"] == 69


def test_write_comparison_tables_contain_thirty_six_rows(tmp_path: Path) -> None:
    """Verify CSV and Markdown reports list all 36 extractor/head combinations.

    Args:
        tmp_path (Path): Pytest-managed temporary directory.

    Returns:
        None: Assertions check file contents and table dimensions.
    """
    rows = []
    for extractor, dimension in (
        ("hoptimus", 1536),
        ("uni2h", 1536),
        ("genbio", 4608),
    ):
        for count in (8, 16, 32):
            for output_type in ("allcat", "pi", "mean", "variance"):
                rows.append(
                    comparison_row(
                        extractor,
                        count,
                        output_type,
                        f"{extractor}_p{count}",
                        dimension,
                        {
                            "val": _split_metrics(0.4, 0.41, 69),
                            "test": _split_metrics(0.5, 0.52, 70),
                        },
                    )
                )
    csv_path = write_comparison_csv(rows, tmp_path / "feature_model_comparison.csv")
    markdown_path = write_comparison_markdown(
        rows,
        tmp_path / "feature_model_comparison.md",
        "2026-09-02",
        n_grid_cells=9,
    )
    csv_text = csv_path.read_text(encoding="utf-8")
    markdown = markdown_path.read_text(encoding="utf-8")
    assert len(rows) == 36
    assert csv_text.count("\n") == 37
    data_rows = [
        line
        for line in markdown.splitlines()
        if line.startswith("| ") and "Extractor" not in line and "Prototypes" not in line
    ]
    assert len(data_rows) == 72  # summary table plus full-metrics table
    assert "Each of 9 extractor × prototype-count combinations trains 4 linear heads" in markdown
    assert "| Extractor | Prototypes | Output | Run | Test bAcc | Val bAcc | Test AUC |" in markdown
    assert csv_text.splitlines()[1].startswith("hoptimus,8,allcat,")
    assert csv_text.splitlines()[-1].startswith("genbio,32,variance,")


def test_reuse_signature_ignores_extractor_and_prototype_count() -> None:
    """Verify reuse matching is independent of the two swept hyperparameters.

    Args:
        None: The test uses two resolved configs that differ only on the grid axes.

    Returns:
        None: Assertions check signature equality.
    """
    first = _resolved_config("uni2h", 16, 1536)
    second = _resolved_config("genbio", 32, 4608)
    assert reuse_signature(first) == reuse_signature(second)
    second["training"] = dict(second["training"])
    second["training"]["epochs"] = 50
    assert reuse_signature(first) != reuse_signature(second)


def test_find_matching_run_returns_newest_compatible_directory(
    tmp_path: Path,
) -> None:
    """Verify reuse selects the newest matching run and rejects old schemas.

    Args:
        tmp_path (Path): Pytest-managed temporary directory.

    Returns:
        None: Assertions check the selected directory name.
    """
    _write_completed_run(tmp_path, "2026-09-02_101033", "uni2h", 16, 1536)
    older = tmp_path / "2026-09-02_101033"
    checkpoint = torch.load(older / "best_model.pth", map_location="cpu", weights_only=False)
    checkpoint["model_schema"] = "legacy"
    torch.save(checkpoint, older / "best_model.pth")
    _write_completed_run(tmp_path, "2026-09-02_161803", "uni2h", 16, 1536)
    _write_completed_run(tmp_path, "2026-09-02_170000", "uni2h", 8, 1536)
    matched = find_matching_run(
        tmp_path,
        "uni2h",
        16,
        _signature(),
        ("allcat", "pi", "mean", "variance"),
    )
    assert matched is not None
    assert matched.name == "2026-09-02_161803"
    assert run_matches_grid_cell(
        matched,
        "uni2h",
        16,
        _signature(),
        ("allcat", "pi", "mean", "variance"),
    )
    assert not run_matches_grid_cell(
        matched,
        "uni2h",
        32,
        _signature(),
        ("allcat", "pi", "mean", "variance"),
    )


def test_has_complete_evaluation_requires_every_head_and_split(
    tmp_path: Path,
) -> None:
    """Verify incomplete evaluation manifests are not treated as finished.

    Args:
        tmp_path (Path): Pytest-managed temporary directory.

    Returns:
        None: Assertions check the completeness predicate.
    """
    complete = _write_completed_run(tmp_path, "complete", "hoptimus", 8, 1536)
    incomplete = _write_completed_run(
        tmp_path, "incomplete", "hoptimus", 8, 1536, include_evaluation=False
    )
    output_types = ("allcat", "pi", "mean", "variance")
    assert has_complete_evaluation(complete, output_types)
    assert not has_complete_evaluation(incomplete, output_types)


def test_write_experiment_config_overrides_extractor_and_run_dir(
    tmp_path: Path,
) -> None:
    """Verify sweep YAML binds the grid cell and optional existing run.

    Args:
        tmp_path (Path): Pytest-managed temporary directory.

    Returns:
        None: Assertions check written YAML keys.
    """
    base = {
        "feature_model": "uni2h",
        "prototype": {"num_prototypes": 16, "method": "kmeans"},
        "paths": {
            "run_dir": None,
            "checkpoint": None,
            "evaluation_output": None,
            "visualization_output": None,
        },
    }
    destination = write_experiment_config(
        base,
        tmp_path / "genbio_p32.yaml",
        "genbio",
        32,
        run_dir=tmp_path / "existing_run",
    )
    loaded = yaml.safe_load(destination.read_text(encoding="utf-8"))
    assert loaded["feature_model"] == "genbio"
    assert loaded["prototype"]["num_prototypes"] == 32
    assert loaded["paths"]["run_dir"].endswith("existing_run")
