"""Tests for slide-level logistic-regression feature-model comparison tables."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import yaml

from logistic_regression.compare_feature_models import (
    COMPARISON_STEM,
    CSV_FIELDNAMES,
    comparison_row,
    experiment_grid,
    find_matching_run,
    has_complete_evaluation,
    pooling_label,
    reuse_signature,
    row_from_run,
    run_matches_grid_cell,
    write_comparison_csv,
    write_comparison_markdown,
    write_experiment_config,
    write_resolved_config,
)
from logistic_regression.train import MODEL_SCHEMA


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
        "max_tiles_per_bag": {"train": 2048, "val": None, "test": None},
        "tile_sampling": "random",
        "num_classes": 5,
        "pooling_population_std_epsilon": 0.0,
        "use_standard_scaler": True,
        "c_values": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        "optimizer": "lbfgs",
        "learning_rate": 1.0,
        "max_iter": 1000,
        "tolerance": 0.00001,
        "class_weight": "balanced",
        "selection_metric": "balanced_accuracy",
    }


def _resolved_config(
    feature_model: str,
    pooling_statistics: Sequence[str],
    input_dim: int,
) -> Dict[str, Any]:
    """Build a resolved-config mapping for one synthetic slide-level run.

    Args:
        feature_model (str): Extractor name.
        pooling_statistics (Sequence[str]): Ordered pooling statistic names.
        input_dim (int): Tile feature width.

    Returns:
        Dict[str, Any]: JSON-serializable resolved configuration.
    """
    return {
        "feature_model": feature_model,
        "bag_level": "slide",
        "pooling_statistics": list(pooling_statistics),
        "input_dim": int(input_dim),
        **_signature(),
    }


def _split_metrics(
    accuracy: float,
    balanced_accuracy: float,
    num_bags: int,
    auc: float = 0.85,
) -> Dict[str, Any]:
    """Build a compact evaluation-summary mapping.

    Args:
        accuracy (float): Accuracy value.
        balanced_accuracy (float): Balanced-accuracy value.
        num_bags (int): Number of evaluated bags.
        auc (float): Macro one-vs-rest ROC AUC.

    Returns:
        Dict[str, Any]: Metric dictionary matching evaluation summaries.
    """
    return {
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "macro_f1": float(balanced_accuracy) - 0.01,
        "macro_ovr_roc_auc": float(auc),
        "log_loss": 1.0 - float(accuracy) / 2.0,
        "num_bags": int(num_bags),
    }


def _write_completed_run(
    run_dir: Path,
    feature_model: str,
    pooling_statistics: Sequence[str],
    input_dim: int,
    selected_c: float = 0.1,
) -> Path:
    """Write fake checkpoint, report, resolved config, and evaluation files.

    Args:
        run_dir (Path): Destination run directory.
        feature_model (str): Extractor name.
        pooling_statistics (Sequence[str]): Ordered pooling statistic names.
        input_dim (int): Tile feature width.
        selected_c (float): Validation-selected C recorded in the report.

    Returns:
        Path: The created run directory.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "best_model.joblib").write_bytes(b"placeholder")
    pooling_dim = int(input_dim) * len(pooling_statistics)
    report = {
        "model_schema": MODEL_SCHEMA,
        "selected_C": float(selected_c),
        "bag_level": "slide",
        "input_dim": int(input_dim),
        "pooling_dim": pooling_dim,
        "pooling_statistics": list(pooling_statistics),
    }
    with (run_dir / "best_model_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle)
    resolved = _resolved_config(feature_model, pooling_statistics, input_dim)
    with (run_dir / "resolved_config.json").open("w", encoding="utf-8") as handle:
        json.dump(resolved, handle)
    evaluation_dir = run_dir / "evaluation_results"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "slide/val": _split_metrics(0.55, 0.50, 69),
        "slide/test": _split_metrics(0.60, 0.58, 70, auc=0.88),
        "tissue/val": _split_metrics(0.52, 0.49, 489),
        "tissue/test": _split_metrics(0.57, 0.54, 559, auc=0.86),
    }
    manifest = {
        "primary_bag_level": "slide",
        "results": results,
    }
    with (evaluation_dir / "evaluation_manifest.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(manifest, handle)
    return run_dir


def test_experiment_grid_and_pooling_label_order() -> None:
    """Verify the nine-cell extractor × pooling report order.

    Args:
        None: The test uses the package defaults.

    Returns:
        None: Assertions check labels and grid pairing.
    """
    grid = experiment_grid(
        ("hoptimus", "uni2h", "genbio"),
        (("mean", "standard_deviation"), ("mean",), ("standard_deviation",)),
    )
    assert len(grid) == 9
    assert grid[0] == ("hoptimus", ("mean", "standard_deviation"))
    assert grid[1] == ("hoptimus", ("mean",))
    assert grid[4] == ("uni2h", ("mean",))
    assert grid[-1] == ("genbio", ("standard_deviation",))
    assert pooling_label(("mean", "standard_deviation")) == "mean+standard_deviation"
    assert pooling_label(["standard_deviation"]) == "standard_deviation"


def test_write_experiment_config_forces_slide_bags(tmp_path: Path) -> None:
    """Verify sweep YAML trains slides and evaluates tissues as supplementary.

    Args:
        tmp_path (Path): Pytest-managed temporary directory.

    Returns:
        None: Assertions check bag level, pooling, and evaluation keys.
    """
    base_config = {
        "feature_model": "uni2h",
        "bag_level": "tissue",
        "pooling_statistics": ["mean", "standard_deviation"],
        "batch_size": 128,
        "num_workers": 8,
        "evaluation": {
            "supplementary_bag_level": "slide",
            "include_train": True,
        },
        "paths": {"checkpoint": None, "evaluation_output": None},
    }
    destination = tmp_path / "uni2h_mean.yaml"
    write_experiment_config(
        base_config,
        destination,
        "hoptimus",
        ["mean"],
        run_dir=tmp_path / "2026-09-03_120000",
        batch_size=1,
        num_workers=4,
    )
    with destination.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    assert payload["feature_model"] == "hoptimus"
    assert payload["bag_level"] == "slide"
    assert payload["pooling_statistics"] == ["mean"]
    assert payload["batch_size"] == 1
    assert payload["num_workers"] == 4
    assert payload["evaluation"]["supplementary_bag_level"] == "tissue"
    assert payload["evaluation"]["include_train"] is False
    assert payload["paths"]["checkpoint"].endswith("best_model.joblib")
    assert payload["paths"]["evaluation_output"].endswith("evaluation_results")
    assert payload["paths"]["attribution_output"].endswith("attribution_heatmaps")


def test_reuse_signature_ignores_extractor_pooling_and_bag_level() -> None:
    """Verify reuse matching is independent of the swept grid axes.

    Args:
        None: The test uses two resolved configs that differ only on grid axes.

    Returns:
        None: Assertions check signature equality and sensitivity.
    """
    first = _resolved_config("uni2h", ("mean",), 1536)
    second = _resolved_config("genbio", ("standard_deviation",), 4608)
    assert reuse_signature(first) == reuse_signature(second)
    second["max_iter"] = 50
    assert reuse_signature(first) != reuse_signature(second)


def test_find_matching_run_ignores_tissue_level_directories(tmp_path: Path) -> None:
    """Reuse only slide-level runs that match extractor and pooling.

    Args:
        tmp_path (Path): Pytest-managed temporary directory.

    Returns:
        None: Assertions check that the newest matching slide run is returned.
    """
    tissue = _write_completed_run(
        tmp_path / "2026-09-01_081303", "uni2h", ("mean",), 1536
    )
    resolved = json.loads((tissue / "resolved_config.json").read_text(encoding="utf-8"))
    resolved["bag_level"] = "tissue"
    (tissue / "resolved_config.json").write_text(json.dumps(resolved), encoding="utf-8")
    report = json.loads((tissue / "best_model_report.json").read_text(encoding="utf-8"))
    report["bag_level"] = "tissue"
    (tissue / "best_model_report.json").write_text(json.dumps(report), encoding="utf-8")

    older = _write_completed_run(
        tmp_path / "2026-09-03_100000", "uni2h", ("mean",), 1536, selected_c=1.0
    )
    newest = _write_completed_run(
        tmp_path / "2026-09-03_110000", "uni2h", ("mean",), 1536, selected_c=0.01
    )
    _write_completed_run(
        tmp_path / "2026-09-03_120000", "hoptimus", ("mean",), 1536
    )
    matching = find_matching_run(
        tmp_path, "uni2h", ("mean",), _signature()
    )
    assert matching == newest.resolve()
    assert run_matches_grid_cell(older, "uni2h", ("mean",), _signature())
    assert not run_matches_grid_cell(tissue, "uni2h", ("mean",), _signature())
    assert has_complete_evaluation(newest)


def test_comparison_row_and_tables_use_slide_filenames(tmp_path: Path) -> None:
    """Verify CSV/Markdown use the slide stem and keep original tissue files unused.

    Args:
        tmp_path (Path): Pytest-managed temporary directory.

    Returns:
        None: Assertions check field names, nine rows, and file names.
    """
    rows = []
    for extractor, dimension in (
        ("hoptimus", 1536),
        ("uni2h", 1536),
        ("genbio", 4608),
    ):
        for statistics in (("mean", "standard_deviation"), ("mean",), ("standard_deviation",)):
            pooling_dim = dimension * len(statistics)
            rows.append(
                comparison_row(
                    extractor,
                    statistics,
                    f"{extractor}_{pooling_label(statistics)}",
                    0.1,
                    dimension,
                    pooling_dim,
                    _split_metrics(0.63, 0.60, 559, auc=0.88),
                    _split_metrics(0.67, 0.65, 70, auc=0.90),
                )
            )
    csv_path = write_comparison_csv(rows, tmp_path / f"{COMPARISON_STEM}.csv")
    markdown_path = write_comparison_markdown(
        rows,
        tmp_path / f"{COMPARISON_STEM}.md",
        "2026-09-03",
        n_grid_cells=9,
    )
    assert csv_path.name == "feature_model_comparison_slide.csv"
    assert markdown_path.name == "feature_model_comparison_slide.md"
    assert not (tmp_path / "feature_model_comparison.csv").exists()
    assert not (tmp_path / "feature_model_comparison.md").exists()
    csv_text = csv_path.read_text(encoding="utf-8")
    markdown = markdown_path.read_text(encoding="utf-8")
    assert csv_text.splitlines()[0] == ",".join(CSV_FIELDNAMES)
    assert "slide_test_auc" in csv_text.splitlines()[0]
    assert csv_text.count("\n") == 10
    assert csv_text.splitlines()[1].startswith("hoptimus,mean+standard_deviation,")
    assert csv_text.splitlines()[-1].startswith("genbio,standard_deviation,")
    assert "slide-level bags" in markdown
    assert "do not replace those tissue-trained files" in markdown
    assert "| Extractor | Pooling | Run | Selected C | Slide test bAcc |" in markdown


def test_row_from_run_reads_slide_and_tissue_test_metrics(tmp_path: Path) -> None:
    """Verify a completed run directory becomes one comparison row.

    Args:
        tmp_path (Path): Pytest-managed temporary directory.

    Returns:
        None: Assertions check selected metrics and pooling token.
    """
    run_dir = _write_completed_run(
        tmp_path / "2026-09-03_130000",
        "genbio",
        ("mean", "standard_deviation"),
        4608,
        selected_c=100.0,
    )
    write_resolved_config(
        run_dir, _resolved_config("genbio", ("mean", "standard_deviation"), 4608)
    )
    row = row_from_run(run_dir)
    assert row["extractor"] == "genbio"
    assert row["pooling_statistics"] == "mean+standard_deviation"
    assert row["run"] == "2026-09-03_130000"
    assert row["selected_C"] == 100.0
    assert row["input_dim"] == 4608
    assert row["pooling_dim"] == 9216
    assert row["tissue_test_bags"] == 559
    assert row["slide_test_bags"] == 70
    assert row["slide_test_bacc"] == 0.58
    assert row["slide_test_auc"] == 0.88
    assert set(row) == set(CSV_FIELDNAMES)


def test_incomplete_evaluation_is_rejected(tmp_path: Path) -> None:
    """Reject a slide run whose evaluation manifest omits tissue test metrics.

    Args:
        tmp_path (Path): Pytest-managed temporary directory.

    Returns:
        None: Assertions check that incomplete manifests are not reused.
    """
    run_dir = _write_completed_run(
        tmp_path / "2026-09-03_140000", "uni2h", ("mean",), 1536
    )
    manifest_path = run_dir / "evaluation_results" / "evaluation_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["results"]["tissue/test"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert not has_complete_evaluation(run_dir)
