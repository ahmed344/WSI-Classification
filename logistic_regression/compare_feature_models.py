"""Benchmark slide-level logistic regression across extractors and pooling.

Each extractor × pooling-statistic combination is one training run. Models are
trained on slide bags (all tissues of a slide concatenated) and evaluated at
both slide (primary) and tissue (supplementary) levels. Results are written to
``feature_model_comparison_slide.csv`` and ``feature_model_comparison_slide.md``
so the existing tissue-level comparison tables are left unchanged.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import yaml

try:
    from .config_loader import load_config
    from .evaluate import evaluate_checkpoint
    from .model import normalize_pooling_statistics, pooling_output_dim
    from .train import MODEL_SCHEMA, train_logistic_regression
except ImportError:
    from config_loader import load_config
    from evaluate import evaluate_checkpoint
    from model import normalize_pooling_statistics, pooling_output_dim
    from train import MODEL_SCHEMA, train_logistic_regression


DEFAULT_FEATURE_MODELS: Tuple[str, ...] = ("hoptimus", "uni2h", "genbio")
DEFAULT_POOLING_VARIANTS: Tuple[Tuple[str, ...], ...] = (
    ("mean", "standard_deviation"),
    ("mean",),
    ("standard_deviation",),
)
REQUIRED_TEST_KEYS: Tuple[str, ...] = ("tissue/test", "slide/test")
REQUIRED_EVALUATION_KEYS: Tuple[str, ...] = (
    "slide/val",
    "slide/test",
    "tissue/val",
    "tissue/test",
)
COMPARISON_STEM = "feature_model_comparison_slide"
SLIDE_DATALOADER_BATCH_SIZE = 1
SLIDE_DATALOADER_NUM_WORKERS = 4
CSV_FIELDNAMES: Tuple[str, ...] = (
    "extractor",
    "pooling_statistics",
    "run",
    "selected_C",
    "input_dim",
    "pooling_dim",
    "tissue_test_bags",
    "tissue_test_acc",
    "tissue_test_bacc",
    "tissue_test_macro_f1",
    "tissue_test_auc",
    "tissue_test_log_loss",
    "slide_test_bags",
    "slide_test_acc",
    "slide_test_bacc",
    "slide_test_macro_f1",
    "slide_test_auc",
    "slide_test_log_loss",
)


def pooling_label(statistics: Sequence[str]) -> str:
    """Format pooling statistics as the comparison-table token.

    Args:
        statistics (Sequence[str]): Ordered pooling statistic names.

    Returns:
        str: ``+``-joined statistic names, for example
            ``mean+standard_deviation``.
    """
    names = normalize_pooling_statistics(statistics)
    return "+".join(names)


def experiment_grid(
    feature_models: Sequence[str],
    pooling_variants: Sequence[Sequence[str]],
) -> List[Tuple[str, Tuple[str, ...]]]:
    """Expand extractor × pooling combinations in report order.

    Args:
        feature_models (Sequence[str]): Feature extractor names in table order.
        pooling_variants (Sequence[Sequence[str]]): Pooling lists in table order.

    Returns:
        List[Tuple[str, Tuple[str, ...]]]: Ordered
            ``(feature_model, pooling_statistics)`` pairs.
    """
    if not feature_models:
        raise ValueError("At least one feature_model is required.")
    if not pooling_variants:
        raise ValueError("At least one pooling variant is required.")
    if any(not str(name).strip() for name in feature_models):
        raise ValueError("feature_models must be nonempty strings.")
    grid: List[Tuple[str, Tuple[str, ...]]] = []
    for feature_model in feature_models:
        for statistics in pooling_variants:
            grid.append((str(feature_model), tuple(normalize_pooling_statistics(statistics))))
    return grid


def load_raw_config(config_path: Path) -> Dict[str, Any]:
    """Load a YAML mapping without resolving paths or derived fields.

    Args:
        config_path (Path): Path to the logistic-regression YAML configuration.

    Returns:
        Dict[str, Any]: Shallow copy of the YAML mapping.
    """
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Configuration '{config_path}' must contain a YAML mapping.")
    return dict(loaded)


def write_experiment_config(
    base_config: Mapping[str, Any],
    destination: Path,
    feature_model: str,
    pooling_statistics: Sequence[str],
    run_dir: Optional[Path] = None,
    batch_size: int = SLIDE_DATALOADER_BATCH_SIZE,
    num_workers: int = SLIDE_DATALOADER_NUM_WORKERS,
) -> Path:
    """Write one sweep YAML forced to slide-level bags.

    Args:
        base_config (Mapping[str, Any]): Unresolved package YAML mapping.
        destination (Path): Output YAML path.
        feature_model (str): Extractor name used to select feature files.
        pooling_statistics (Sequence[str]): Ordered pooling statistic names.
        run_dir (Optional[Path]): Explicit run directory, or ``None`` for a new
            dated training run.
        batch_size (int): Positive DataLoader batch size for slide bags.
        num_workers (int): Nonnegative DataLoader worker count.

    Returns:
        Path: Absolute path of the written YAML file.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if num_workers < 0:
        raise ValueError("num_workers must be nonnegative.")
    payload = deepcopy(dict(base_config))
    payload["feature_model"] = str(feature_model)
    payload["bag_level"] = "slide"
    payload["pooling_statistics"] = list(normalize_pooling_statistics(pooling_statistics))
    payload["batch_size"] = int(batch_size)
    payload["num_workers"] = int(num_workers)
    evaluation = dict(payload.get("evaluation") or {})
    evaluation["supplementary_bag_level"] = "tissue"
    evaluation["include_train"] = False
    payload["evaluation"] = evaluation
    paths = dict(payload.get("paths") or {})
    if run_dir is None:
        paths["checkpoint"] = None
        paths["evaluation_output"] = None
    else:
        resolved_run = Path(run_dir).expanduser().resolve()
        paths["checkpoint"] = str(resolved_run / "best_model.joblib")
        paths["evaluation_output"] = str(resolved_run / "evaluation_results")
    payload["paths"] = paths
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
    return destination.resolve()


def reuse_signature(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the hyperparameter subset that must match before a run is reused.

    Extractor name, pooling statistics, and bag level are excluded; those
    identify the grid cell. Paths, loader sizes, and derived fields are also
    excluded.

    Args:
        config (Mapping[str, Any]): Loaded or resolved logistic-regression
            configuration.

    Returns:
        Dict[str, Any]: JSON-comparable training/method signature.
    """
    caps = config.get("max_tiles_per_bag")
    if not isinstance(caps, Mapping):
        raise ValueError("max_tiles_per_bag must be a mapping.")
    return {
        "random_seed": int(config["random_seed"]),
        "train_ratio": float(config["train_ratio"]),
        "val_ratio": float(config["val_ratio"]),
        "test_ratio": float(config["test_ratio"]),
        "feature_normalization": str(config["feature_normalization"]),
        "max_tiles_per_bag": {
            "train": caps.get("train"),
            "val": caps.get("val"),
            "test": caps.get("test"),
        },
        "tile_sampling": str(config["tile_sampling"]),
        "num_classes": int(config["num_classes"]),
        "pooling_population_std_epsilon": float(
            config["pooling_population_std_epsilon"]
        ),
        "use_standard_scaler": bool(config["use_standard_scaler"]),
        "c_values": [float(value) for value in config["c_values"]],
        "optimizer": str(config["optimizer"]),
        "learning_rate": float(config["learning_rate"]),
        "max_iter": int(config["max_iter"]),
        "tolerance": float(config["tolerance"]),
        "class_weight": config.get("class_weight"),
        "selection_metric": str(config["selection_metric"]),
    }


def write_resolved_config(run_dir: Path, config: Mapping[str, Any]) -> Path:
    """Write the JSON used later to match and reuse a completed sweep cell.

    Args:
        run_dir (Path): Dated training-run directory.
        config (Mapping[str, Any]): Resolved configuration used for the run.

    Returns:
        Path: Absolute written ``resolved_config.json`` path.
    """
    payload = {
        "model_schema": MODEL_SCHEMA,
        "feature_model": str(config["feature_model"]),
        "bag_level": str(config["bag_level"]),
        "pooling_statistics": list(normalize_pooling_statistics(config["pooling_statistics"])),
        "input_dim": int(config["input_dim"]),
        "pooling_dim": pooling_output_dim(
            int(config["input_dim"]), config["pooling_statistics"]
        ),
        **reuse_signature(config),
    }
    destination = Path(run_dir) / "resolved_config.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return destination.resolve()


def find_matching_run(
    results_root: Path,
    feature_model: str,
    pooling_statistics: Sequence[str],
    expected_signature: Mapping[str, Any],
) -> Optional[Path]:
    """Return the newest completed slide-level run that matches one grid cell.

    Args:
        results_root (Path): Directory containing dated logistic-regression runs.
        feature_model (str): Required extractor name.
        pooling_statistics (Sequence[str]): Required pooling statistic names.
        expected_signature (Mapping[str, Any]): Remaining hyperparameters from
            ``reuse_signature``.

    Returns:
        Optional[Path]: Newest matching run directory, or ``None``.
    """
    if not results_root.is_dir():
        return None
    candidates = sorted(
        (item for item in results_root.iterdir() if item.is_dir()),
        key=lambda item: item.name,
        reverse=True,
    )
    for run_dir in candidates:
        if run_matches_grid_cell(
            run_dir, feature_model, pooling_statistics, expected_signature
        ):
            return run_dir.resolve()
    return None


def run_matches_grid_cell(
    run_dir: Path,
    feature_model: str,
    pooling_statistics: Sequence[str],
    expected_signature: Mapping[str, Any],
) -> bool:
    """Return whether one dated run can stand in for a slide-level sweep cell.

    Args:
        run_dir (Path): Candidate logistic-regression run directory.
        feature_model (str): Required extractor name.
        pooling_statistics (Sequence[str]): Required pooling statistic names.
        expected_signature (Mapping[str, Any]): Remaining hyperparameters.

    Returns:
        bool: ``True`` when the run's resolved config and checkpoint match.
    """
    config_path = run_dir / "resolved_config.json"
    checkpoint_path = run_dir / "best_model.joblib"
    report_path = run_dir / "best_model_report.json"
    if (
        not config_path.is_file()
        or not checkpoint_path.is_file()
        or not report_path.is_file()
    ):
        return False
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            resolved = json.load(handle)
        with report_path.open("r", encoding="utf-8") as handle:
            report = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(resolved, Mapping) or not isinstance(report, Mapping):
        return False
    if str(resolved.get("feature_model")) != str(feature_model):
        return False
    if str(resolved.get("bag_level")) != "slide":
        return False
    try:
        run_statistics = list(
            normalize_pooling_statistics(resolved.get("pooling_statistics"))
        )
        expected_statistics = list(normalize_pooling_statistics(pooling_statistics))
    except (TypeError, ValueError):
        return False
    if run_statistics != expected_statistics:
        return False
    if str(report.get("bag_level")) != "slide":
        return False
    if str(report.get("model_schema")) not in (MODEL_SCHEMA,):
        return False
    try:
        run_signature = reuse_signature(resolved)
    except (KeyError, TypeError, ValueError):
        return False
    return run_signature == dict(expected_signature)


def has_complete_evaluation(run_dir: Path) -> bool:
    """Return whether slide and tissue val/test metrics exist on disk.

    Args:
        run_dir (Path): Logistic-regression run directory.

    Returns:
        bool: ``True`` when the evaluation manifest has the required splits.
    """
    manifest_path = run_dir / "evaluation_results" / "evaluation_manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False
    results = manifest.get("results")
    if not isinstance(results, Mapping):
        return False
    if str(manifest.get("primary_bag_level")) != "slide":
        return False
    for key in REQUIRED_EVALUATION_KEYS:
        metrics = results.get(key)
        if not isinstance(metrics, Mapping) or metrics.get("skipped"):
            return False
        for field in (
            "num_bags",
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "log_loss",
        ):
            if field not in metrics:
                return False
    return True


def _metric_or_none(metrics: Mapping[str, Any], key: str) -> Optional[float]:
    """Read one optional float metric.

    Args:
        metrics (Mapping[str, Any]): Evaluation-summary mapping.
        key (str): Metric name.

    Returns:
        Optional[float]: Finite float, or ``None`` when missing.
    """
    value = metrics.get(key)
    if value is None:
        return None
    return float(value)


def comparison_row(
    extractor: str,
    pooling_statistics: Sequence[str],
    run_name: str,
    selected_c: float,
    input_dim: int,
    pooling_dim: int,
    tissue_test: Mapping[str, Any],
    slide_test: Mapping[str, Any],
) -> Dict[str, Any]:
    """Build one CSV/Markdown row for a slide-trained checkpoint.

    Args:
        extractor (str): Feature extractor name.
        pooling_statistics (Sequence[str]): Ordered pooling statistic names.
        run_name (str): Dated run directory name.
        selected_c (float): Validation-selected inverse regularization.
        input_dim (int): Tile feature width.
        pooling_dim (int): Concatenated bag-vector width.
        tissue_test (Mapping[str, Any]): Tissue-level test metrics.
        slide_test (Mapping[str, Any]): Slide-level test metrics.

    Returns:
        Dict[str, Any]: Row keyed by ``CSV_FIELDNAMES``.
    """
    return {
        "extractor": str(extractor),
        "pooling_statistics": pooling_label(pooling_statistics),
        "run": str(run_name),
        "selected_C": float(selected_c),
        "input_dim": int(input_dim),
        "pooling_dim": int(pooling_dim),
        "tissue_test_bags": int(tissue_test["num_bags"]),
        "tissue_test_acc": float(tissue_test["accuracy"]),
        "tissue_test_bacc": float(tissue_test["balanced_accuracy"]),
        "tissue_test_macro_f1": float(tissue_test["macro_f1"]),
        "tissue_test_auc": _metric_or_none(tissue_test, "macro_ovr_roc_auc"),
        "tissue_test_log_loss": float(tissue_test["log_loss"]),
        "slide_test_bags": int(slide_test["num_bags"]),
        "slide_test_acc": float(slide_test["accuracy"]),
        "slide_test_bacc": float(slide_test["balanced_accuracy"]),
        "slide_test_macro_f1": float(slide_test["macro_f1"]),
        "slide_test_auc": _metric_or_none(slide_test, "macro_ovr_roc_auc"),
        "slide_test_log_loss": float(slide_test["log_loss"]),
    }


def row_from_run(run_dir: Path) -> Dict[str, Any]:
    """Collect one comparison row from a completed slide-level run.

    Args:
        run_dir (Path): Run directory with report, resolved config, and
            evaluation artifacts.

    Returns:
        Dict[str, Any]: Comparison row for the run.
    """
    resolved_path = run_dir / "resolved_config.json"
    report_path = run_dir / "best_model_report.json"
    manifest_path = run_dir / "evaluation_results" / "evaluation_manifest.json"
    with resolved_path.open("r", encoding="utf-8") as handle:
        resolved = json.load(handle)
    with report_path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    results = manifest["results"]
    for key in REQUIRED_TEST_KEYS:
        if key not in results:
            raise KeyError(f"Evaluation manifest is missing '{key}'.")
    return comparison_row(
        str(resolved["feature_model"]),
        list(resolved["pooling_statistics"]),
        run_dir.name,
        float(report["selected_C"]),
        int(report["input_dim"]),
        int(report["pooling_dim"]),
        results["tissue/test"],
        results["slide/test"],
    )


def write_comparison_csv(rows: Sequence[Mapping[str, Any]], destination: Path) -> Path:
    """Write the machine-readable slide-level comparison table.

    Args:
        rows (Sequence[Mapping[str, Any]]): Comparison rows in report order.
        destination (Path): Output CSV path.

    Returns:
        Path: Absolute written CSV path.
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_FIELDNAMES))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row[key]) for key in CSV_FIELDNAMES})
    return destination.resolve()


def write_comparison_markdown(
    rows: Sequence[Mapping[str, Any]],
    destination: Path,
    generated_on: str,
    n_grid_cells: int,
) -> Path:
    """Write the human-readable slide-level comparison report.

    Args:
        rows (Sequence[Mapping[str, Any]]): Comparison rows in report order.
        destination (Path): Output Markdown path.
        generated_on (str): Calendar date shown in the report header.
        n_grid_cells (int): Number of extractor × pooling runs.

    Returns:
        Path: Absolute written Markdown path.
    """
    summary_headers = (
        "Extractor",
        "Pooling",
        "Run",
        "Selected C",
        "Slide test bAcc",
        "Tissue test bAcc",
        "Slide test AUC",
    )
    summary_lines = [
        [
            str(row["extractor"]),
            f"`{row['pooling_statistics']}`",
            f"`{row['run']}`",
            _format_c(row["selected_C"]),
            _format_float(row["slide_test_bacc"], 3),
            _format_float(row["tissue_test_bacc"], 3),
            _format_float(row["slide_test_auc"], 3),
        ]
        for row in rows
    ]
    full_headers = (
        "Extractor",
        "Pooling",
        "Run",
        "C",
        "Dim",
        "Pool dim",
        "Tissue bags",
        "Tissue acc",
        "Tissue bAcc",
        "Tissue macro F1",
        "Tissue AUC",
        "Tissue log loss",
        "Slide bags",
        "Slide acc",
        "Slide bAcc",
        "Slide macro F1",
        "Slide AUC",
        "Slide log loss",
    )
    full_lines = [
        [
            str(row["extractor"]),
            f"`{row['pooling_statistics']}`",
            f"`{row['run']}`",
            _format_c(row["selected_C"]),
            str(int(row["input_dim"])),
            str(int(row["pooling_dim"])),
            str(int(row["tissue_test_bags"])),
            _format_float(row["tissue_test_acc"], 4),
            _format_float(row["tissue_test_bacc"], 4),
            _format_float(row["tissue_test_macro_f1"], 4),
            _format_float(row["tissue_test_auc"], 4),
            _format_float(row["tissue_test_log_loss"], 4),
            str(int(row["slide_test_bags"])),
            _format_float(row["slide_test_acc"], 4),
            _format_float(row["slide_test_bacc"], 4),
            _format_float(row["slide_test_macro_f1"], 4),
            _format_float(row["slide_test_auc"], 4),
            _format_float(row["slide_test_log_loss"], 4),
        ]
        for row in rows
    ]
    body = [
        "# Logistic regression feature-model comparison (slide-level training)",
        "",
        f"Independent raw-feature statistics logistic-regression models trained and evaluated on {generated_on}.",
        f"Each of {n_grid_cells} extractor × pooling combinations is trained on "
        "**slide-level bags** (all tissues of a slide concatenated, then pooled). "
        "Evaluation reports the native slide test split and a supplementary tissue "
        "test split. Pooling variants are mean only, standard deviation only, and "
        "mean concatenated with standard deviation.",
        "",
        "These tables are the slide-trained counterpart of `feature_model_comparison.csv` "
        "and `feature_model_comparison.md`; they do not replace those tissue-trained files.",
        "",
        _markdown_table(summary_headers, summary_lines),
        "",
        "## Full test metrics",
        "",
        _markdown_table(full_headers, full_lines),
        "",
    ]
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(body), encoding="utf-8")
    return destination.resolve()


def run_feature_model_comparison(
    config_path: Optional[str] = None,
    feature_models: Sequence[str] = DEFAULT_FEATURE_MODELS,
    pooling_variants: Sequence[Sequence[str]] = DEFAULT_POOLING_VARIANTS,
    reuse_existing: bool = True,
) -> Dict[str, Any]:
    """Train or reuse the slide-level sweep, evaluate, and write new tables.

    Args:
        config_path (Optional[str]): Base YAML path, or ``None`` for the package
            default.
        feature_models (Sequence[str]): Extractors to include, in table order.
        pooling_variants (Sequence[Sequence[str]]): Pooling lists to include, in
            table order.
        reuse_existing (bool): When ``True``, skip training for grid cells that
            already have a matching completed slide-level run.

    Returns:
        Dict[str, Any]: Run records, comparison rows, and written table paths.
    """
    base_path = (
        Path(config_path).expanduser().resolve()
        if config_path is not None
        else Path(__file__).with_name("config.yml").resolve()
    )
    raw_config = load_raw_config(base_path)
    reference_config = load_config(str(base_path))
    expected_signature = reuse_signature(reference_config)
    results_root = Path(str(reference_config["logistic_regression_results_root"]))
    grid = experiment_grid(feature_models, pooling_variants)
    sweep_dir = results_root / "_sweep_configs_slide"
    records: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []
    errors: List[str] = []

    for index, (feature_model, statistics) in enumerate(grid, start=1):
        label = f"{feature_model} × {pooling_label(statistics)}"
        print(f"=== [{index}/{len(grid)}] {label} (slide bags) ===")
        try:
            matching = (
                find_matching_run(
                    results_root,
                    feature_model,
                    statistics,
                    expected_signature,
                )
                if reuse_existing
                else None
            )
            yaml_name = f"{feature_model}_{pooling_label(statistics)}.yaml"
            yaml_path = sweep_dir / yaml_name
            if matching is not None:
                print(f"Reusing run {matching.name}")
                write_experiment_config(
                    raw_config,
                    yaml_path,
                    feature_model,
                    statistics,
                    run_dir=matching,
                )
                resolved = load_config(str(yaml_path))
                write_resolved_config(matching, resolved)
                if not has_complete_evaluation(matching):
                    print(f"Evaluating reused run {matching.name}")
                    run_evaluation_for_run(str(yaml_path), matching)
                run_dir = matching
                reused = True
            else:
                write_experiment_config(
                    raw_config,
                    yaml_path,
                    feature_model,
                    statistics,
                )
                artifacts = train_logistic_regression(str(yaml_path))
                run_dir = Path(artifacts["run_dir"])
                write_experiment_config(
                    raw_config,
                    yaml_path,
                    feature_model,
                    statistics,
                    run_dir=run_dir,
                )
                resolved = load_config(str(yaml_path))
                write_resolved_config(run_dir, resolved)
                run_evaluation_for_run(str(yaml_path), run_dir)
                reused = False
            if not has_complete_evaluation(run_dir):
                raise FileNotFoundError(
                    f"Incomplete evaluation artifacts in {run_dir}."
                )
            rows.append(row_from_run(run_dir))
            records.append(
                {
                    "feature_model": feature_model,
                    "pooling_statistics": list(statistics),
                    "run_dir": str(run_dir),
                    "reused": reused,
                }
            )
        except Exception as error:
            message = f"{label}: {type(error).__name__}: {error}"
            print(message)
            errors.append(message)
            records.append(
                {
                    "feature_model": feature_model,
                    "pooling_statistics": list(statistics),
                    "run_dir": None,
                    "reused": False,
                    "error": message,
                }
            )
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    generated_on = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d")
    csv_path = results_root / f"{COMPARISON_STEM}.csv"
    markdown_path = results_root / f"{COMPARISON_STEM}.md"
    registry_path = results_root / f"{COMPARISON_STEM}_runs.json"
    if rows:
        write_comparison_csv(rows, csv_path)
        write_comparison_markdown(rows, markdown_path, generated_on, len(grid))
    registry = {
        "generated_on": generated_on,
        "model_schema": MODEL_SCHEMA,
        "bag_level": "slide",
        "feature_models": list(feature_models),
        "pooling_variants": [list(item) for item in pooling_variants],
        "n_grid_cells": len(grid),
        "n_rows": len(rows),
        "records": records,
        "errors": errors,
        "csv": str(csv_path) if rows else None,
        "markdown": str(markdown_path) if rows else None,
    }
    results_root.mkdir(parents=True, exist_ok=True)
    with registry_path.open("w", encoding="utf-8") as handle:
        json.dump(registry, handle, indent=2)
    if errors:
        raise RuntimeError(
            "Logistic-regression slide-level comparison finished with "
            f"{len(errors)} failed grid cell(s): " + "; ".join(errors)
        )
    if len(rows) != len(grid):
        raise RuntimeError(
            f"Expected {len(grid)} comparison rows, received {len(rows)}."
        )
    print(f"Wrote {csv_path}")
    print(f"Wrote {markdown_path}")
    return registry


def run_evaluation_for_run(config_path: str, run_dir: Path) -> Dict[str, Any]:
    """Evaluate one slide-trained checkpoint at slide and tissue levels.

    Args:
        config_path (str): Sweep YAML whose paths point at ``run_dir``.
        run_dir (Path): Dated run directory containing ``best_model.joblib``.

    Returns:
        Dict[str, Any]: Evaluation manifest.
    """
    checkpoint = Path(run_dir) / "best_model.joblib"
    output_dir = Path(run_dir) / "evaluation_results"
    return evaluate_checkpoint(
        config_path=config_path,
        checkpoint=str(checkpoint),
        output_dir=str(output_dir),
        supplementary_bag_level="tissue",
        include_train=False,
    )


def _csv_value(value: Any) -> Any:
    """Convert a table cell to a CSV-safe scalar.

    Args:
        value (Any): Row value.

    Returns:
        Any: Empty string for ``None``, otherwise the original value.
    """
    return "" if value is None else value


def _format_float(value: Any, digits: int) -> str:
    """Format an optional float for Markdown tables.

    Args:
        value (Any): Numeric cell, or ``None`` when a metric is undefined.
        digits (int): Number of digits after the decimal point.

    Returns:
        str: Fixed-precision number, or an em dash when missing.
    """
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def _format_c(value: Any) -> str:
    """Format a selected C value without trailing zeros.

    Args:
        value (Any): Inverse regularization strength.

    Returns:
        str: Compact decimal representation.
    """
    return f"{float(value):g}"


def _markdown_table(
    headers: Sequence[str], rows: Sequence[Sequence[str]]
) -> str:
    """Render a GitHub-flavored Markdown table.

    Args:
        headers (Sequence[str]): Column titles.
        rows (Sequence[Sequence[str]]): Table body cells.

    Returns:
        str: Markdown table text without a trailing newline.
    """
    header_line = "| " + " | ".join(headers) + " |"
    separator = "|" + "|".join("---" for _ in headers) + "|"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header_line, separator, *body])


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse slide-level feature-model comparison command-line arguments.

    Args:
        argv (Optional[Sequence[str]]): Argument vector, or ``None`` for
            ``sys.argv``.

    Returns:
        argparse.Namespace: Parsed comparison options.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Path to config.yml")
    parser.add_argument(
        "--feature-models",
        nargs="+",
        default=list(DEFAULT_FEATURE_MODELS),
        help="Feature extractors to compare, in table order.",
    )
    parser.add_argument(
        "--pooling-variants",
        nargs="+",
        default=None,
        help=(
            "Pooling variants as '+'-joined names, for example "
            "mean+standard_deviation mean standard_deviation."
        ),
    )
    parser.add_argument(
        "--reuse-existing",
        dest="reuse_existing",
        action="store_true",
        default=True,
        help="Reuse matching completed slide-level runs (default).",
    )
    parser.add_argument(
        "--no-reuse-existing",
        dest="reuse_existing",
        action="store_false",
        help="Train every grid cell even when a matching run exists.",
    )
    return parser.parse_args(argv)


def _parse_pooling_variants(
    raw_variants: Optional[Sequence[str]],
) -> Sequence[Sequence[str]]:
    """Parse CLI pooling tokens into statistic-name lists.

    Args:
        raw_variants (Optional[Sequence[str]]): ``+``-joined pooling tokens, or
            ``None`` for the default three-variant grid.

    Returns:
        Sequence[Sequence[str]]: Pooling statistic lists.
    """
    if raw_variants is None:
        return DEFAULT_POOLING_VARIANTS
    parsed: List[Tuple[str, ...]] = []
    for token in raw_variants:
        names = tuple(part for part in str(token).split("+") if part)
        parsed.append(tuple(normalize_pooling_statistics(names)))
    return parsed


def main() -> None:
    """Run the slide-level feature-model comparison from the command line.

    Args:
        None: This entry point reads command-line arguments.

    Returns:
        None: Comparison tables are written under the configured results root.
    """
    arguments = parse_args()
    run_feature_model_comparison(
        arguments.config,
        feature_models=arguments.feature_models,
        pooling_variants=_parse_pooling_variants(arguments.pooling_variants),
        reuse_existing=arguments.reuse_existing,
    )


if __name__ == "__main__":
    main()
