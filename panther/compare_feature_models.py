"""Benchmark PANTHER across feature extractors, prototype counts, and heads.

Each extractor × ``num_prototypes`` combination is one training run. That run
fits one linear classifier per configured ``output_type`` (``allcat``, ``pi``,
``mean``, ``variance``) and writes a comparison CSV/Markdown pair in the same
style as the logistic-regression feature-model tables.
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

from .config_loader import load_config
from .evaluate_panther import run_evaluation
from .panther_model import OUTPUT_TYPES
from .train_panther import (
    MODEL_SCHEMA,
    component_embedding_paths,
    run_training,
)


DEFAULT_FEATURE_MODELS: Tuple[str, ...] = ("hoptimus", "uni2h", "genbio")
DEFAULT_NUM_PROTOTYPES: Tuple[int, ...] = (8, 16, 32)
REQUIRED_SPLITS: Tuple[str, ...] = ("val", "test")
COMPARISON_STEM = "feature_model_comparison"
CSV_FIELDNAMES: Tuple[str, ...] = (
    "extractor",
    "num_prototypes",
    "output_type",
    "run",
    "input_dim",
    "representation_dim",
    "val_slides",
    "val_acc",
    "val_bacc",
    "val_macro_f1",
    "val_auc",
    "val_loss",
    "test_slides",
    "test_acc",
    "test_bacc",
    "test_macro_f1",
    "test_auc",
    "test_loss",
)
METRIC_KEYS: Tuple[str, ...] = (
    "num_slides",
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "multiclass_roc_auc",
    "loss",
)


def representation_dimension(
    output_type: str, num_prototypes: int, input_dim: int
) -> int:
    """Return the flattened GMM-vector width for one PANTHER head.

    Args:
        output_type (str): Representation block; one of ``OUTPUT_TYPES``.
        num_prototypes (int): Number of morphological prototypes ``C``.
        input_dim (int): Tile feature width ``D``.

    Returns:
        int: ``C`` for ``pi``, ``C * D`` for ``mean`` or ``variance``, and
            ``C + 2 * C * D`` for ``allcat``.
    """
    if num_prototypes <= 0 or input_dim <= 0:
        raise ValueError("num_prototypes and input_dim must be positive.")
    if output_type == "pi":
        return int(num_prototypes)
    if output_type in ("mean", "variance"):
        return int(num_prototypes) * int(input_dim)
    if output_type == "allcat":
        return int(num_prototypes) + 2 * int(num_prototypes) * int(input_dim)
    raise ValueError(
        f"Unsupported output_type '{output_type}'; expected one of {OUTPUT_TYPES}."
    )


def experiment_grid(
    feature_models: Sequence[str],
    num_prototypes: Sequence[int],
) -> List[Tuple[str, int]]:
    """Expand the extractor × prototype-count combinations in report order.

    Args:
        feature_models (Sequence[str]): Feature extractor names in table order.
        num_prototypes (Sequence[int]): Prototype counts in table order.

    Returns:
        List[Tuple[str, int]]: Ordered ``(feature_model, num_prototypes)`` pairs.
    """
    if not feature_models:
        raise ValueError("At least one feature_model is required.")
    if not num_prototypes:
        raise ValueError("At least one num_prototypes value is required.")
    if any(not str(name).strip() for name in feature_models):
        raise ValueError("feature_models must be nonempty strings.")
    if any(int(count) <= 0 for count in num_prototypes):
        raise ValueError("num_prototypes values must be positive integers.")
    return [
        (str(feature_model), int(count))
        for feature_model in feature_models
        for count in num_prototypes
    ]


def load_raw_config(config_path: Path) -> Dict[str, Any]:
    """Load a YAML mapping without resolving paths or derived fields.

    Args:
        config_path (Path): Path to the PANTHER YAML configuration.

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
    num_prototypes: int,
    run_dir: Optional[Path] = None,
) -> Path:
    """Write one sweep YAML with extractor, prototype count, and optional run dir.

    Args:
        base_config (Mapping[str, Any]): Unresolved package YAML mapping.
        destination (Path): Output YAML path.
        feature_model (str): Extractor name used to select feature files.
        num_prototypes (int): Number of K-means prototypes to fit.
        run_dir (Optional[Path]): Explicit run directory, or ``None`` for a new
            dated training run.

    Returns:
        Path: Absolute path of the written YAML file.
    """
    payload = deepcopy(dict(base_config))
    payload["feature_model"] = str(feature_model)
    prototype = dict(payload.get("prototype") or {})
    prototype["num_prototypes"] = int(num_prototypes)
    payload["prototype"] = prototype
    paths = dict(payload.get("paths") or {})
    if run_dir is None:
        paths["run_dir"] = None
        paths["checkpoint"] = None
        paths["evaluation_output"] = None
        paths["visualization_output"] = None
    else:
        paths["run_dir"] = str(Path(run_dir).resolve())
    payload["paths"] = paths
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
    return destination.resolve()


def reuse_signature(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the hyperparameter subset that must match before a run is reused.

    Extractor name and prototype count are excluded; those identify the grid
    cell. Paths and resolved derived fields are also excluded.

    Args:
        config (Mapping[str, Any]): Loaded or resolved PANTHER configuration.

    Returns:
        Dict[str, Any]: JSON-comparable training/method signature.
    """
    prototype = dict(_section(config, "prototype"))
    prototype.pop("num_prototypes", None)
    model = dict(_section(config, "model"))
    model.pop("output_type", None)
    return {
        "random_seed": int(config["random_seed"]),
        "train_ratio": float(config["train_ratio"]),
        "val_ratio": float(config["val_ratio"]),
        "test_ratio": float(config["test_ratio"]),
        "feature_normalization": str(config["feature_normalization"]),
        "max_tiles_per_slide": config.get("max_tiles_per_slide"),
        "tile_sampling": str(config["tile_sampling"]),
        "prototype": prototype,
        "model": model,
        "training": dict(_section(config, "training")),
    }


def configured_output_types(config: Mapping[str, Any]) -> List[str]:
    """Return the ordered representation heads requested by a configuration.

    Args:
        config (Mapping[str, Any]): Loaded or unresolved PANTHER configuration.

    Returns:
        List[str]: Unique ``output_type`` names in config order.
    """
    output_types = _section(config, "model").get("output_type")
    if isinstance(output_types, str):
        values = [output_types]
    elif isinstance(output_types, list):
        values = [str(item) for item in output_types]
    else:
        raise ValueError("model.output_type must be a string or a list of strings.")
    if not values or len(set(values)) != len(values):
        raise ValueError("model.output_type must be a nonempty unique list.")
    if any(item not in OUTPUT_TYPES for item in values):
        raise ValueError(
            f"model.output_type values must be in {OUTPUT_TYPES}; received {values}."
        )
    return values


def find_matching_run(
    results_root: Path,
    feature_model: str,
    num_prototypes: int,
    expected_signature: Mapping[str, Any],
    output_types: Sequence[str],
) -> Optional[Path]:
    """Return the newest completed run that matches one grid cell.

    Args:
        results_root (Path): Directory containing dated PANTHER run folders.
        feature_model (str): Required extractor name.
        num_prototypes (int): Required prototype count.
        expected_signature (Mapping[str, Any]): Remaining hyperparameters from
            ``reuse_signature``.
        output_types (Sequence[str]): Representation heads that must be trained.

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
            run_dir,
            feature_model,
            num_prototypes,
            expected_signature,
            output_types,
        ):
            return run_dir.resolve()
    return None


def run_matches_grid_cell(
    run_dir: Path,
    feature_model: str,
    num_prototypes: int,
    expected_signature: Mapping[str, Any],
    output_types: Sequence[str],
) -> bool:
    """Return whether one dated run can stand in for a sweep cell.

    Args:
        run_dir (Path): Candidate PANTHER run directory.
        feature_model (str): Required extractor name.
        num_prototypes (int): Required prototype count.
        expected_signature (Mapping[str, Any]): Remaining hyperparameters.
        output_types (Sequence[str]): Representation heads that must exist.

    Returns:
        bool: ``True`` when the run's resolved config, caches, and heads match.
    """
    config_path = run_dir / "resolved_config.json"
    checkpoint_path = run_dir / "best_model.pth"
    if not config_path.is_file() or not checkpoint_path.is_file():
        return False
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            resolved = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(resolved, Mapping):
        return False
    if str(resolved.get("feature_model")) != str(feature_model):
        return False
    try:
        run_prototypes = int(_section(resolved, "prototype")["num_prototypes"])
    except (KeyError, TypeError, ValueError):
        return False
    if run_prototypes != int(num_prototypes):
        return False
    try:
        run_output_types = configured_output_types(resolved)
    except ValueError:
        return False
    if any(output_type not in run_output_types for output_type in output_types):
        return False
    if reuse_signature(resolved) != dict(expected_signature):
        return False
    missing_caches = [
        path for path in component_embedding_paths(run_dir).values() if not path.is_file()
    ]
    if missing_caches:
        return False
    for output_type in output_types:
        if not (run_dir / "models" / output_type / "best_model.pth").is_file():
            return False
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, ValueError):
        return False
    if not isinstance(checkpoint, Mapping):
        return False
    if checkpoint.get("model_schema") != MODEL_SCHEMA:
        return False
    models = checkpoint.get("models")
    if not isinstance(models, Mapping):
        return False
    return all(output_type in models for output_type in output_types)


def has_complete_evaluation(
    run_dir: Path,
    output_types: Sequence[str],
    splits: Sequence[str] = REQUIRED_SPLITS,
) -> bool:
    """Return whether every requested head has val/test slide metrics on disk.

    Args:
        run_dir (Path): PANTHER run directory.
        output_types (Sequence[str]): Representation heads that must be scored.
        splits (Sequence[str]): Split names required under ``slide/<split>``.

    Returns:
        bool: ``True`` when the combined evaluation manifest is complete.
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
    for output_type in output_types:
        type_results = results.get(output_type)
        if not isinstance(type_results, Mapping):
            return False
        for split in splits:
            metrics = type_results.get(f"slide/{split}")
            if not isinstance(metrics, Mapping):
                return False
            if any(key not in metrics for key in METRIC_KEYS):
                return False
    return True


def comparison_row(
    extractor: str,
    num_prototypes: int,
    output_type: str,
    run_name: str,
    input_dim: int,
    split_metrics: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    """Build one CSV/Markdown row for a single trained representation head.

    Args:
        extractor (str): Feature extractor name.
        num_prototypes (int): Prototype count used to fit the GMM.
        output_type (str): Representation head name.
        run_name (str): Dated run directory name.
        input_dim (int): Tile feature width.
        split_metrics (Mapping[str, Mapping[str, Any]]): Mapping from split
            name to evaluation metric dictionaries.

    Returns:
        Dict[str, Any]: Row keyed by ``CSV_FIELDNAMES``.
    """
    row: Dict[str, Any] = {
        "extractor": str(extractor),
        "num_prototypes": int(num_prototypes),
        "output_type": str(output_type),
        "run": str(run_name),
        "input_dim": int(input_dim),
        "representation_dim": representation_dimension(
            output_type, num_prototypes, input_dim
        ),
    }
    for split in REQUIRED_SPLITS:
        metrics = split_metrics[split]
        prefix = split
        row[f"{prefix}_slides"] = int(metrics["num_slides"])
        row[f"{prefix}_acc"] = float(metrics["accuracy"])
        row[f"{prefix}_bacc"] = float(metrics["balanced_accuracy"])
        row[f"{prefix}_macro_f1"] = float(metrics["macro_f1"])
        auc = metrics["multiclass_roc_auc"]
        row[f"{prefix}_auc"] = None if auc is None else float(auc)
        row[f"{prefix}_loss"] = float(metrics["loss"])
    return row


def rows_from_evaluation_manifest(
    run_dir: Path,
    output_types: Sequence[str],
) -> List[Dict[str, Any]]:
    """Collect one comparison row per representation head in a completed run.

    Args:
        run_dir (Path): PANTHER run directory with evaluation artifacts.
        output_types (Sequence[str]): Representation heads to emit, in order.

    Returns:
        List[Dict[str, Any]]: Comparison rows for ``output_types``.
    """
    resolved_path = run_dir / "resolved_config.json"
    manifest_path = run_dir / "evaluation_results" / "evaluation_manifest.json"
    with resolved_path.open("r", encoding="utf-8") as handle:
        resolved = json.load(handle)
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    extractor = str(resolved["feature_model"])
    num_prototypes = int(resolved["prototype"]["num_prototypes"])
    input_dim = int(resolved["input_dim"])
    results = manifest["results"]
    rows: List[Dict[str, Any]] = []
    for output_type in output_types:
        type_results = results[output_type]
        split_metrics = {
            split: type_results[f"slide/{split}"] for split in REQUIRED_SPLITS
        }
        rows.append(
            comparison_row(
                extractor,
                num_prototypes,
                output_type,
                run_dir.name,
                input_dim,
                split_metrics,
            )
        )
    return rows


def write_comparison_csv(rows: Sequence[Mapping[str, Any]], destination: Path) -> Path:
    """Write the machine-readable comparison table.

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
    """Write the human-readable comparison report with summary and full tables.

    Args:
        rows (Sequence[Mapping[str, Any]]): Comparison rows in report order.
        destination (Path): Output Markdown path.
        generated_on (str): Calendar date shown in the report header.
        n_grid_cells (int): Number of extractor × prototype-count runs.

    Returns:
        Path: Absolute written Markdown path.
    """
    n_heads = 0 if not rows else len({str(row["output_type"]) for row in rows})
    summary_headers = (
        "Extractor",
        "Prototypes",
        "Output",
        "Run",
        "Test bAcc",
        "Val bAcc",
        "Test AUC",
    )
    summary_lines = [
        [
            str(row["extractor"]),
            str(int(row["num_prototypes"])),
            f"`{row['output_type']}`",
            f"`{row['run']}`",
            _format_float(row["test_bacc"], 3),
            _format_float(row["val_bacc"], 3),
            _format_float(row["test_auc"], 3),
        ]
        for row in rows
    ]
    full_headers = (
        "Extractor",
        "Prototypes",
        "Output",
        "Run",
        "Dim",
        "Rep dim",
        "Val slides",
        "Val acc",
        "Val bAcc",
        "Val macro F1",
        "Val AUC",
        "Val loss",
        "Test slides",
        "Test acc",
        "Test bAcc",
        "Test macro F1",
        "Test AUC",
        "Test loss",
    )
    full_lines = [
        [
            str(row["extractor"]),
            str(int(row["num_prototypes"])),
            f"`{row['output_type']}`",
            f"`{row['run']}`",
            str(int(row["input_dim"])),
            str(int(row["representation_dim"])),
            str(int(row["val_slides"])),
            _format_float(row["val_acc"], 4),
            _format_float(row["val_bacc"], 4),
            _format_float(row["val_macro_f1"], 4),
            _format_float(row["val_auc"], 4),
            _format_float(row["val_loss"], 4),
            str(int(row["test_slides"])),
            _format_float(row["test_acc"], 4),
            _format_float(row["test_bacc"], 4),
            _format_float(row["test_macro_f1"], 4),
            _format_float(row["test_auc"], 4),
            _format_float(row["test_loss"], 4),
        ]
        for row in rows
    ]
    body = [
        "# PANTHER feature-model comparison",
        "",
        f"Independent PANTHER slide-level models trained and evaluated on {generated_on}.",
        f"Each of {n_grid_cells} extractor × prototype-count combinations trains "
        f"{n_heads} linear heads on cached GMM parameters: `allcat`, `pi`, "
        "`mean`, and `variance`. `allcat` concatenates mixture weights, component "
        "means, and diagonal variances. Evaluation is native slide-level "
        "(no tissue bags).",
        "",
        _markdown_table(summary_headers, summary_lines),
        "",
        "## Full val/test metrics",
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
    num_prototypes: Sequence[int] = DEFAULT_NUM_PROTOTYPES,
    reuse_existing: bool = True,
) -> Dict[str, Any]:
    """Train or reuse the sweep, evaluate every head, and write comparison tables.

    Args:
        config_path (Optional[str]): Base YAML path, or ``None`` for the package
            default.
        feature_models (Sequence[str]): Extractors to include, in table order.
        num_prototypes (Sequence[int]): Prototype counts to include, in table
            order.
        reuse_existing (bool): When ``True``, skip training for grid cells that
            already have a matching completed run.

    Returns:
        Dict[str, Any]: Run records, comparison rows, and written table paths.
    """
    base_path = (
        Path(config_path).expanduser().resolve()
        if config_path is not None
        else Path(__file__).with_name("config.yaml").resolve()
    )
    raw_config = load_raw_config(base_path)
    reference_config = load_config(str(base_path))
    output_types = configured_output_types(reference_config)
    expected_signature = reuse_signature(reference_config)
    results_root = Path(str(reference_config["panther_results_root"]))
    grid = experiment_grid(feature_models, num_prototypes)
    sweep_dir = results_root / "_sweep_configs"
    records: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []
    errors: List[str] = []

    for index, (feature_model, prototype_count) in enumerate(grid, start=1):
        label = f"{feature_model} × {prototype_count} prototypes"
        print(f"=== [{index}/{len(grid)}] {label} ===")
        try:
            matching = (
                find_matching_run(
                    results_root,
                    feature_model,
                    prototype_count,
                    expected_signature,
                    output_types,
                )
                if reuse_existing
                else None
            )
            yaml_path = sweep_dir / f"{feature_model}_p{prototype_count}.yaml"
            if matching is not None:
                print(f"Reusing run {matching.name}")
                write_experiment_config(
                    raw_config,
                    yaml_path,
                    feature_model,
                    prototype_count,
                    run_dir=matching,
                )
                if not has_complete_evaluation(matching, output_types):
                    print(f"Evaluating reused run {matching.name}")
                    run_evaluation(str(yaml_path))
                run_dir = matching
                reused = True
            else:
                write_experiment_config(
                    raw_config,
                    yaml_path,
                    feature_model,
                    prototype_count,
                )
                run_dir = run_training(str(yaml_path))
                write_experiment_config(
                    raw_config,
                    yaml_path,
                    feature_model,
                    prototype_count,
                    run_dir=run_dir,
                )
                run_evaluation(str(yaml_path))
                reused = False
            if not has_complete_evaluation(run_dir, output_types):
                raise FileNotFoundError(
                    f"Incomplete evaluation artifacts in {run_dir}."
                )
            cell_rows = rows_from_evaluation_manifest(run_dir, output_types)
            rows.extend(cell_rows)
            records.append(
                {
                    "feature_model": feature_model,
                    "num_prototypes": prototype_count,
                    "run_dir": str(run_dir),
                    "reused": reused,
                    "n_rows": len(cell_rows),
                }
            )
        except Exception as error:
            message = f"{label}: {type(error).__name__}: {error}"
            print(message)
            errors.append(message)
            records.append(
                {
                    "feature_model": feature_model,
                    "num_prototypes": prototype_count,
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
        "feature_models": list(feature_models),
        "num_prototypes": [int(value) for value in num_prototypes],
        "output_types": list(output_types),
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
            "PANTHER feature-model comparison finished with "
            f"{len(errors)} failed grid cell(s): " + "; ".join(errors)
        )
    if len(rows) != len(grid) * len(output_types):
        raise RuntimeError(
            f"Expected {len(grid) * len(output_types)} comparison rows, "
            f"received {len(rows)}."
        )
    print(f"Wrote {csv_path}")
    print(f"Wrote {markdown_path}")
    return registry


def _section(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    """Return a nested mapping configuration section.

    Args:
        config (Mapping[str, Any]): Parent configuration mapping.
        key (str): Section name.

    Returns:
        Mapping[str, Any]: Nested mapping stored at ``key``.
    """
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"Config key '{key}' must be a mapping.")
    return value


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
    """Parse feature-model comparison command-line arguments.

    Args:
        argv (Optional[Sequence[str]]): Argument vector, or ``None`` for
            ``sys.argv``.

    Returns:
        argparse.Namespace: Parsed comparison options.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument(
        "--feature-models",
        nargs="+",
        default=list(DEFAULT_FEATURE_MODELS),
        help="Feature extractors to compare, in table order.",
    )
    parser.add_argument(
        "--num-prototypes",
        nargs="+",
        type=int,
        default=list(DEFAULT_NUM_PROTOTYPES),
        help="Prototype counts to compare, in table order.",
    )
    parser.add_argument(
        "--reuse-existing",
        dest="reuse_existing",
        action="store_true",
        default=True,
        help="Reuse matching completed runs (default).",
    )
    parser.add_argument(
        "--no-reuse-existing",
        dest="reuse_existing",
        action="store_false",
        help="Train every grid cell even when a matching run exists.",
    )
    return parser.parse_args(argv)


def main() -> None:
    """Run the feature-model comparison from the command line.

    Args:
        None: This entry point reads process command-line arguments.

    Returns:
        None: Comparison tables are written under the configured results root.
    """
    arguments = parse_args()
    run_feature_model_comparison(
        arguments.config,
        feature_models=arguments.feature_models,
        num_prototypes=arguments.num_prototypes,
        reuse_existing=arguments.reuse_existing,
    )


if __name__ == "__main__":
    main()
