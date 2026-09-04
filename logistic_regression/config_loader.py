"""Load and resolve the independent logistic-regression configuration."""

from __future__ import annotations

import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence

import yaml

try:
    from .model import normalize_pooling_statistics
except ImportError:
    from model import normalize_pooling_statistics


_OPTIONS: Dict[str, Sequence[Any]] = {
    "bag_level": ("tissue", "slide"),
    "feature_normalization": ("none", "l2", "layer_norm"),
    "tile_sampling": ("random", "uniform", "first"),
    "device": ("cuda",),
    "optimizer": ("lbfgs",),
    "class_weight": ("balanced", None),
    "selection_metric": ("balanced_accuracy", "accuracy"),
}
_SPLITS = ("train", "val", "test")
_RUN_ID_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{6}(?:_\d{2})?$")
_EXPLICIT_PATH_KEYS = ("checkpoint", "evaluation_output", "attribution_output")
_VISUALIZATION_KEYS = (
    "samples",
    "tile_size",
    "dpi",
    "render_workers",
    "thumbnail_size",
)


def logistic_regression_results_root(output_dir: str | Path) -> Path:
    """Return the logistic-regression results root.

    Args:
        output_dir (str | Path): Base results directory.

    Returns:
        Path: ``{output_dir}/logistic_regression`` path.
    """
    return Path(output_dir) / "logistic_regression"


def make_run_id(moment: Optional[datetime] = None) -> str:
    """Build a sortable timestamp run identifier.

    Args:
        moment (Optional[datetime]): Instant to format, or ``None`` for now.

    Returns:
        str: Run identifier shaped ``YYYY-MM-DD_HHMMSS``.
    """
    value = moment if moment is not None else datetime.now()
    return value.strftime("%Y-%m-%d_%H%M%S")


def apply_run_artifact_paths(config: Dict[str, Any], run_dir: str | Path) -> None:
    """Point model and evaluation paths at one dated run.

    Args:
        config (Dict[str, Any]): Configuration dictionary to mutate.
        run_dir (str | Path): Run directory receiving artifacts.

    Returns:
        None: Configuration paths are updated in place.
    """
    destination = Path(run_dir).expanduser().resolve()
    config["checkpoint_dir"] = str(destination)
    paths = dict(config.get("paths", {}) or {})
    paths["checkpoint"] = str(destination / "best_model.joblib")
    paths["evaluation_output"] = str(destination / "evaluation_results")
    paths["attribution_output"] = str(destination / "attribution_heatmaps")
    config["paths"] = paths


def list_training_run_dirs(results_root: str | Path) -> List[Path]:
    """List dated or model-bearing logistic-regression run directories.

    Args:
        results_root (str | Path): ``{output_dir}/logistic_regression`` directory.

    Returns:
        List[Path]: Candidate directories ordered newest first.
    """
    root = Path(results_root)
    if not root.is_dir():
        return []
    timestamped: List[Path] = []
    model_dirs: List[Path] = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        if _RUN_ID_PATTERN.match(child.name):
            timestamped.append(child)
        elif (child / "best_model.joblib").is_file():
            model_dirs.append(child)
    timestamped.sort(key=lambda path: path.name, reverse=True)
    model_dirs.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    seen = {path.resolve() for path in timestamped}
    ordered = list(timestamped)
    for path in model_dirs:
        if path.resolve() not in seen:
            ordered.append(path)
            seen.add(path.resolve())
    return ordered


def resolve_latest_training_run(results_root: str | Path) -> Optional[Path]:
    """Select the newest logistic-regression run.

    Args:
        results_root (str | Path): Results root containing run directories.

    Returns:
        Optional[Path]: Newest run directory, or ``None`` when unavailable.
    """
    runs = list_training_run_dirs(results_root)
    return runs[0] if runs else None


def should_allocate_dated_training_run(config: Mapping[str, Any]) -> bool:
    """Return whether training should allocate a dated directory.

    Args:
        config (Mapping[str, Any]): Configuration carrying ``_explicit_paths``.

    Returns:
        bool: ``True`` when the YAML checkpoint path was null.
    """
    explicit = config.get("_explicit_paths")
    return isinstance(explicit, Mapping) and not bool(explicit.get("checkpoint", False))


def allocate_training_run(config: Dict[str, Any]) -> Path:
    """Create and bind a dated training run when automatic paths are enabled.

    Args:
        config (Dict[str, Any]): Resolved configuration to mutate.

    Returns:
        Path: Directory that should receive model and evaluation artifacts.
    """
    if not should_allocate_dated_training_run(config):
        destination = Path(str(config.get("checkpoint_dir", ".")))
        destination.mkdir(parents=True, exist_ok=True)
        return destination.resolve()
    root = logistic_regression_results_root(config["output_dir"])
    root.mkdir(parents=True, exist_ok=True)
    base_id = make_run_id()
    run_dir = root / base_id
    suffix = 1
    while run_dir.exists():
        run_dir = root / f"{base_id}_{suffix:02d}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    apply_run_artifact_paths(config, run_dir)
    return run_dir


def uses_explicit_checkpoint_path(config: Mapping[str, Any]) -> bool:
    """Return whether the checkpoint path was explicitly supplied.

    Args:
        config (Mapping[str, Any]): Configuration carrying path flags.

    Returns:
        bool: ``True`` when ``paths.checkpoint`` was non-null in YAML.
    """
    explicit = config.get("_explicit_paths")
    if isinstance(explicit, Mapping):
        return bool(explicit.get("checkpoint", False))
    return True


def resolve_inference_run_paths(config: Dict[str, Any]) -> Optional[Path]:
    """Bind automatic artifact paths to the newest dated run.

    Args:
        config (Dict[str, Any]): Resolved configuration to update.

    Returns:
        Optional[Path]: Applied run directory, or ``None`` if no binding occurs.
    """
    explicit = config.get("_explicit_paths", {})
    if not isinstance(explicit, Mapping):
        explicit = {}
    if any(bool(explicit.get(key, False)) for key in _EXPLICIT_PATH_KEYS):
        return None
    latest = resolve_latest_training_run(
        logistic_regression_results_root(config["output_dir"])
    )
    if latest is None:
        return None
    apply_run_artifact_paths(config, latest)
    return latest


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load, validate, and resolve a logistic-regression YAML configuration.

    Args:
        config_path (Optional[str]): YAML path, or ``None`` for package default.

    Returns:
        Dict[str, Any]: Validated configuration with resolved paths and features.
    """
    path = (
        Path(config_path).expanduser()
        if config_path is not None
        else Path(__file__).with_name("config.yml")
    ).resolve()
    with path.open("r", encoding="utf-8") as config_file:
        loaded = yaml.safe_load(config_file)
    if not isinstance(loaded, MutableMapping):
        raise ValueError(f"Configuration '{path}' must contain a YAML mapping.")
    config: Dict[str, Any] = dict(loaded)
    _validate_config(config)
    config["input_dim"] = resolve_input_dim(config)
    config["feature_file_suffix"] = resolve_feature_file_suffix(config)
    _resolve_paths(config, path.parent)
    return config


def resolve_input_dim(config: Mapping[str, Any]) -> int:
    """Resolve the selected raw feature width.

    Args:
        config (Mapping[str, Any]): Feature model and width mapping.

    Returns:
        int: Positive raw tile feature dimension.
    """
    selected = str(config.get("feature_model", "default"))
    dimensions = config.get("feature_model_input_dims", {})
    if not isinstance(dimensions, Mapping):
        raise ValueError("feature_model_input_dims must be a mapping.")
    if selected not in dimensions:
        raise ValueError(
            f"Unknown feature_model '{selected}'. Available: "
            + ", ".join(sorted(str(key) for key in dimensions))
        )
    dimension = int(dimensions[selected])
    if dimension <= 0:
        raise ValueError("Resolved feature input dimension must be positive.")
    return dimension


def resolve_feature_file_suffix(config: Mapping[str, Any]) -> str:
    """Resolve the selected on-disk feature suffix.

    Args:
        config (Mapping[str, Any]): Feature model and suffix mapping.

    Returns:
        str: Nonempty feature filename suffix ending in ``.pt``.
    """
    selected = str(config.get("feature_model", "default"))
    suffixes = config.get("feature_model_suffixes", {})
    if not isinstance(suffixes, Mapping):
        raise ValueError("feature_model_suffixes must be a mapping.")
    if selected not in suffixes:
        raise ValueError(
            f"Unknown feature_model '{selected}'. Available: "
            + ", ".join(sorted(str(key) for key in suffixes))
        )
    suffix = str(suffixes[selected])
    if not suffix.endswith(".pt"):
        raise ValueError("Resolved feature suffix must end in '.pt'.")
    return suffix


def _validate_config(config: MutableMapping[str, Any]) -> None:
    """Validate dataset, evaluation, and logistic-regression settings.

    Args:
        config (MutableMapping[str, Any]): Parsed configuration mapping.

    Returns:
        None: Validation succeeds by returning normally.
    """
    for key, choices in _OPTIONS.items():
        if config.get(key) not in choices:
            raise ValueError(f"Invalid {key} '{config.get(key)}'; expected {choices}.")
    for key in ("batch_size", "max_iter", "num_classes"):
        _require_positive_int(config, key)
    if int(config["num_classes"]) < 2:
        raise ValueError("num_classes must be at least two.")
    pooling_epsilon = _require_number(config, "pooling_population_std_epsilon")
    if pooling_epsilon < 0.0:
        raise ValueError("pooling_population_std_epsilon must be nonnegative.")
    if "pooling_statistics" not in config or config.get("pooling_statistics") is None:
        raise ValueError("pooling_statistics must be a nonempty list.")
    try:
        config["pooling_statistics"] = list(
            normalize_pooling_statistics(config.get("pooling_statistics"))
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid pooling_statistics: {error}") from error
    if config.get("use_standard_scaler") is not True:
        raise ValueError("use_standard_scaler must be true for this baseline.")
    workers = config.get("num_workers")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 0:
        raise ValueError("num_workers must be a nonnegative integer.")
    _require_positive_int(config, "prefetch_factor")
    seed = config.get("random_seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("random_seed must be a nonnegative integer.")
    for key in ("learning_rate", "tolerance"):
        value = _require_number(config, key)
        if value <= 0.0:
            raise ValueError(f"{key} must be positive.")
    c_values = config.get("c_values")
    if not isinstance(c_values, list) or not c_values:
        raise ValueError("c_values must be a nonempty list.")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
        for value in c_values
    ):
        raise ValueError("Every c_values entry must be positive and finite.")
    ratios = tuple(_require_number(config, f"{split}_ratio") for split in _SPLITS)
    if any(ratio < 0.0 or ratio > 1.0 for ratio in ratios):
        raise ValueError("Split ratios must each be between zero and one.")
    if abs(sum(ratios) - 1.0) > 1e-8 or ratios[0] <= 0.0:
        raise ValueError("Split ratios must sum to one and train_ratio must be positive.")
    caps = config.get("max_tiles_per_bag")
    if not isinstance(caps, Mapping) or set(caps) - set(_SPLITS):
        raise ValueError("max_tiles_per_bag must contain only train/val/test keys.")
    for split in _SPLITS:
        cap = caps.get(split)
        if cap is not None and (
            isinstance(cap, bool) or not isinstance(cap, int) or cap <= 0
        ):
            raise ValueError(f"max_tiles_per_bag.{split} must be positive or null.")
    evaluation = config.get("evaluation")
    if not isinstance(evaluation, Mapping):
        raise ValueError("evaluation must be a mapping.")
    if set(evaluation) - {"supplementary_bag_level", "include_train"}:
        raise ValueError("evaluation contains unknown keys.")
    if evaluation.get("supplementary_bag_level") not in (None, "tissue", "slide"):
        raise ValueError("evaluation.supplementary_bag_level is invalid.")
    if not isinstance(evaluation.get("include_train"), bool):
        raise ValueError("evaluation.include_train must be a boolean.")
    _validate_visualization(config.get("visualization"))
    if not isinstance(config.get("paths"), Mapping):
        raise ValueError("paths must be a mapping.")


def _validate_visualization(visualization: Any) -> None:
    """Validate attribution-heatmap visualization settings.

    Args:
        visualization (Any): Parsed ``visualization`` mapping.

    Returns:
        None: Validation succeeds by returning normally.
    """
    if not isinstance(visualization, Mapping):
        raise ValueError("visualization must be a mapping.")
    unknown = set(visualization) - set(_VISUALIZATION_KEYS)
    if unknown:
        raise ValueError(
            "visualization contains unknown keys: " + ", ".join(sorted(unknown))
        )
    samples = visualization.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("visualization.samples must be a nonempty list.")
    if any(not isinstance(sample, str) or not sample for sample in samples):
        raise ValueError("Every visualization.samples entry must be a nonempty string.")
    unknown_splits = [sample for sample in samples if sample not in _SPLITS]
    if unknown_splits:
        raise ValueError(
            "visualization.samples entries must be train, val, or test; "
            f"unknown: {unknown_splits}."
        )
    if len(set(samples)) != len(samples):
        raise ValueError("visualization.samples must not contain duplicate names.")
    for key in ("tile_size", "dpi", "render_workers", "thumbnail_size"):
        value = visualization.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"visualization.{key} must be a positive integer.")


def normalize_visualization_samples(samples: Optional[Sequence[str]]) -> List[str]:
    """Validate and freeze an ordered visualization split list.

    Args:
        samples (Optional[Sequence[str]]): Split names, or ``None`` for test only.

    Returns:
        List[str]: Unique split names in the requested order.
    """
    if samples is None:
        return ["test"]
    if isinstance(samples, (str, bytes)) or not isinstance(samples, Sequence):
        raise ValueError("visualization.samples must be a nonempty list of split names.")
    names = list(samples)
    if not names:
        raise ValueError("visualization.samples must be a nonempty list.")
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("Every visualization.samples entry must be a nonempty string.")
    unknown = [name for name in names if name not in _SPLITS]
    if unknown:
        raise ValueError(
            "visualization.samples entries must be train, val, or test; "
            f"unknown: {unknown}."
        )
    if len(set(names)) != len(names):
        raise ValueError("visualization.samples must not contain duplicate names.")
    return names


def _require_number(config: Mapping[str, Any], key: str) -> float:
    """Read one required finite numeric value.

    Args:
        config (Mapping[str, Any]): Configuration containing the value.
        key (str): Required key.

    Returns:
        float: Validated finite value.
    """
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{key} must be finite.")
    return result


def _require_positive_int(config: Mapping[str, Any], key: str) -> int:
    """Read one required positive integer.

    Args:
        config (Mapping[str, Any]): Configuration containing the value.
        key (str): Required key.

    Returns:
        int: Validated positive integer.
    """
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{key} must be a positive integer.")
    return value


def _resolve_paths(config: Dict[str, Any], config_dir: Path) -> None:
    """Resolve required and optional paths in place.

    Args:
        config (Dict[str, Any]): Configuration to mutate.
        config_dir (Path): Base directory for relative YAML paths.

    Returns:
        None: Absolute paths and automatic defaults are written in place.
    """
    data_root = _absolute_path(config.get("data_root"), config_dir, "data_root")
    checkpoint_dir = _absolute_path(
        config.get("checkpoint_dir"), config_dir, "checkpoint_dir"
    )
    output_dir = _absolute_path(config.get("output_dir"), config_dir, "output_dir")
    config["data_root"] = str(data_root)
    config["checkpoint_dir"] = str(checkpoint_dir)
    config["output_dir"] = str(output_dir)
    root = logistic_regression_results_root(output_dir)
    config["logistic_regression_results_root"] = str(root)
    raw_paths = config.get("paths", {})
    if not isinstance(raw_paths, Mapping):
        raise ValueError("paths must be a mapping.")
    paths = dict(raw_paths)
    explicit_checkpoint = _optional_path(paths.get("checkpoint"), config_dir)
    explicit_evaluation = _optional_path(paths.get("evaluation_output"), config_dir)
    explicit_attribution = _optional_path(paths.get("attribution_output"), config_dir)
    config["_explicit_paths"] = {
        "checkpoint": explicit_checkpoint is not None,
        "evaluation_output": explicit_evaluation is not None,
        "attribution_output": explicit_attribution is not None,
    }
    if explicit_checkpoint is not None:
        paths["checkpoint"] = str(explicit_checkpoint)
    if explicit_evaluation is not None:
        paths["evaluation_output"] = str(explicit_evaluation)
    if explicit_attribution is not None:
        paths["attribution_output"] = str(explicit_attribution)
    config["paths"] = paths
    latest = resolve_latest_training_run(root)
    if latest is not None:
        if explicit_checkpoint is None:
            config["paths"]["checkpoint"] = str(latest / "best_model.joblib")
            config["checkpoint_dir"] = str(latest)
        if explicit_evaluation is None:
            config["paths"]["evaluation_output"] = str(latest / "evaluation_results")
        if explicit_attribution is None:
            config["paths"]["attribution_output"] = str(
                latest / "attribution_heatmaps"
            )
        return
    if explicit_checkpoint is None:
        config["paths"]["checkpoint"] = str(checkpoint_dir / "best_model.joblib")
    if explicit_evaluation is None:
        config["paths"]["evaluation_output"] = str(
            root / "evaluation_results"
        )
    if explicit_attribution is None:
        config["paths"]["attribution_output"] = str(root / "attribution_heatmaps")


def _absolute_path(value: Any, base_dir: Path, key: str) -> Path:
    """Resolve one required path.

    Args:
        value (Any): Path-like value.
        base_dir (Path): Base for relative values.
        key (str): Configuration key used in errors.

    Returns:
        Path: Expanded absolute path.
    """
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError(f"{key} must be a nonempty path.")
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base_dir / path).resolve()


def _optional_path(value: Any, base_dir: Path) -> Optional[Path]:
    """Resolve one optional path.

    Args:
        value (Any): Optional path-like value.
        base_dir (Path): Base for relative values.

    Returns:
        Optional[Path]: Absolute path, or ``None`` when omitted.
    """
    if value is None:
        return None
    return _absolute_path(value, base_dir, "paths entry")
