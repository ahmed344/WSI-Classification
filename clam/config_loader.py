"""Load, validate, and resolve CLAM configuration values."""

from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence

import yaml


_OPTIONS: Dict[str, Sequence[str]] = {
    "model_type": ("clam_mb", "clam_sb"),
    "bag_level": ("tissue", "slide"),
    "feature_normalization": ("none", "l2", "layer_norm"),
    "tile_sampling": ("random", "uniform", "first"),
}
_SPLITS = ("train", "val", "test")


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load and validate a YAML configuration.

    Args:
        config_path (Optional[str]): YAML file path. When ``None``, use the
            ``config.yml`` beside this module.

    Returns:
        Dict[str, Any]: Validated configuration with absolute paths plus the
            resolved ``input_dim`` and ``feature_file_suffix`` values.
    """
    path = (
        Path(config_path).expanduser()
        if config_path is not None
        else Path(__file__).with_name("config.yml")
    )
    path = path.resolve()
    with path.open("r", encoding="utf-8") as config_file:
        loaded = yaml.safe_load(config_file)
    if not isinstance(loaded, MutableMapping):
        raise ValueError(f"Configuration '{path}' must contain a YAML mapping.")

    config: Dict[str, Any] = dict(loaded)
    config.setdefault("q", 0.0)
    config.setdefault("epsilon", 0.0)
    _validate_config(config)
    config["input_dim"] = resolve_input_dim(config)
    config["feature_file_suffix"] = resolve_feature_file_suffix(config)
    _resolve_paths(config, path.parent)
    return config


def resolve_input_dim(config: Mapping[str, Any]) -> int:
    """Resolve the input dimensionality for the selected feature model.

    Args:
        config (Mapping[str, Any]): Configuration containing ``feature_model``
            and ``feature_model_input_dims``.

    Returns:
        int: Positive feature dimensionality consumed by CLAM.
    """
    selected_model = str(config.get('feature_model', 'default'))
    input_dim_map = config.get('feature_model_input_dims', {})

    if not isinstance(input_dim_map, Mapping):
        raise ValueError(
            "Invalid config key 'feature_model_input_dims': expected a mapping."
        )

    if selected_model not in input_dim_map:
        available_models = ', '.join(sorted(str(k) for k in input_dim_map.keys()))
        raise ValueError(
            f"Unknown feature_model '{selected_model}'. "
            f"Available input dimensions: {available_models}."
        )

    input_dim = int(input_dim_map[selected_model])
    if input_dim <= 0:
        raise ValueError(
            f"Invalid input_dim '{input_dim}' for feature_model '{selected_model}'. "
            "Input dimension must be positive."
        )
    return input_dim


def resolve_feature_file_suffix(config: Mapping[str, Any]) -> str:
    """Resolve the filename suffix for the selected feature model.

    Args:
        config (Mapping[str, Any]): Configuration containing ``feature_model``
            and ``feature_model_suffixes``.

    Returns:
        str: Nonempty ``.pt`` suffix used for feature discovery.
    """
    selected_model = str(config.get('feature_model', 'default'))
    suffix_map = config.get('feature_model_suffixes', {})

    if not isinstance(suffix_map, Mapping):
        raise ValueError(
            "Invalid config key 'feature_model_suffixes': expected a mapping."
        )

    if selected_model not in suffix_map:
        available_models = ', '.join(sorted(str(k) for k in suffix_map.keys()))
        raise ValueError(
            f"Unknown feature_model '{selected_model}'. "
            f"Available models: {available_models}."
        )

    suffix = str(suffix_map[selected_model])
    if not suffix.endswith('.pt'):
        raise ValueError(
            f"Invalid suffix '{suffix}' for feature_model '{selected_model}'. "
            "Suffix must end with '.pt'."
        )
    return suffix


def _validate_config(config: Mapping[str, Any]) -> None:
    """Validate canonical CLAM and dataset settings.

    Args:
        config (Mapping[str, Any]): Parsed configuration to validate.

    Returns:
        None: Validation succeeds by returning normally.
    """
    for key, choices in _OPTIONS.items():
        value = config.get(key)
        if value not in choices:
            raise ValueError(
                f"Invalid {key} '{value}'. Expected one of: {', '.join(choices)}."
            )

    for key in ("gated_attention", "subtyping"):
        if not isinstance(config.get(key), bool):
            raise ValueError(f"Config key '{key}' must be a boolean.")

    for key in (
        "hidden_dim",
        "attention_dim",
        "num_classes",
        "batch_size",
        "gradient_accumulation_steps",
        "epochs",
        "lr_scheduler_patience_cls",
        "patience",
    ):
        _require_positive_int(config, key)
    if int(config["num_classes"]) < 2:
        raise ValueError("Config key 'num_classes' must be at least 2.")
    _require_positive_int(config, "k_sample")
    dropout = _require_number(config, "dropout")
    if not 0.0 <= dropout < 1.0:
        raise ValueError("Config key 'dropout' must be in [0, 1).")
    bag_weight = _require_number(config, "bag_weight")
    if not 0.0 <= bag_weight <= 1.0:
        raise ValueError("Config key 'bag_weight' must be between 0 and 1.")
    q = _require_number(config, "q")
    if not 0.0 <= q <= 1.0:
        raise ValueError("Config key 'q' must be between 0 and 1.")
    epsilon = _require_number(config, "epsilon")
    if not 0.0 <= epsilon < 1.0:
        raise ValueError("Config key 'epsilon' must be in [0, 1).")
    for key in (
        "lr_cls",
        "lr_scheduler_factor_cls",
        "weight_decay_cls",
    ):
        value = _require_number(config, key)
        if value < 0.0 or (key != "weight_decay_cls" and value == 0.0):
            raise ValueError(f"Config key '{key}' must be positive.")
    scheduler_factor = float(config["lr_scheduler_factor_cls"])
    if scheduler_factor >= 1.0:
        raise ValueError("Config key 'lr_scheduler_factor_cls' must be less than 1.")
    for key in ("use_weighted_sampler", "use_class_weighted_loss"):
        if not isinstance(config.get(key), bool):
            raise ValueError(f"Config key '{key}' must be a boolean.")
    random_seed = config.get("random_seed")
    if isinstance(random_seed, bool) or not isinstance(random_seed, int) or random_seed < 0:
        raise ValueError("Config key 'random_seed' must be a nonnegative integer.")
    minimum_epochs = config.get("min_epochs_before_early_stopping")
    if (
        isinstance(minimum_epochs, bool)
        or not isinstance(minimum_epochs, int)
        or minimum_epochs < 0
    ):
        raise ValueError(
            "Config key 'min_epochs_before_early_stopping' must be nonnegative."
        )
    valid_checkpoint_metrics = {
        "balanced_accuracy",
        "accuracy",
        "loss",
        "classification_loss",
        "instance_loss",
    }
    if config.get("best_checkpoint_metric") not in valid_checkpoint_metrics:
        raise ValueError(
            "Config key 'best_checkpoint_metric' must be one of: "
            + ", ".join(sorted(valid_checkpoint_metrics))
            + "."
        )

    ratios = [_require_number(config, f"{split}_ratio") for split in _SPLITS]
    if any(ratio < 0.0 or ratio > 1.0 for ratio in ratios):
        raise ValueError("train/val/test ratios must each be between 0 and 1.")
    if abs(sum(ratios) - 1.0) > 1e-8:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.")
    if ratios[0] <= 0.0:
        raise ValueError("train_ratio must be greater than 0.")

    caps = config.get("max_tiles_per_bag")
    if not isinstance(caps, Mapping):
        raise ValueError("Config key 'max_tiles_per_bag' must be a mapping.")
    unknown_caps = set(caps) - set(_SPLITS)
    if unknown_caps:
        raise ValueError(
            "Unknown max_tiles_per_bag split(s): "
            + ", ".join(sorted(str(key) for key in unknown_caps))
        )
    for split in _SPLITS:
        cap = caps.get(split)
        if cap is not None and (isinstance(cap, bool) or not isinstance(cap, int) or cap <= 0):
            raise ValueError(
                f"max_tiles_per_bag.{split} must be a positive integer or null."
            )

    visualization = config.get("visualization", {})
    if not isinstance(visualization, Mapping):
        raise ValueError("Config key 'visualization' must be a mapping.")
    visualization_split = visualization.get("split", "val")
    if visualization_split not in _SPLITS:
        raise ValueError("visualization.split must be train, val, or test.")


def _require_number(config: Mapping[str, Any], key: str) -> float:
    """Read a required finite numeric configuration value.

    Args:
        config (Mapping[str, Any]): Configuration containing the value.
        key (str): Required key to read.

    Returns:
        float: Validated finite numeric value.
    """
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Config key '{key}' must be numeric.")
    result = float(value)
    if result != result or result in (float("inf"), float("-inf")):
        raise ValueError(f"Config key '{key}' must be finite.")
    return result


def _require_positive_int(config: Mapping[str, Any], key: str) -> int:
    """Read a required positive integer configuration value.

    Args:
        config (Mapping[str, Any]): Configuration containing the value.
        key (str): Required key to read.

    Returns:
        int: Validated positive integer.
    """
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"Config key '{key}' must be a positive integer.")
    return value


def _resolve_paths(config: Dict[str, Any], config_dir: Path) -> None:
    """Resolve data, output, checkpoint, and artifact paths in place.

    Args:
        config (Dict[str, Any]): Configuration dictionary to mutate.
        config_dir (Path): Base directory for relative paths.

    Returns:
        None: Paths are updated in ``config`` in place.
    """
    data_root = _absolute_path(config.get("data_root"), config_dir, "data_root")
    checkpoint_dir = _absolute_path(
        config.get("checkpoint_dir", "checkpoints"), config_dir, "checkpoint_dir"
    )
    output_dir = _absolute_path(
        config.get("output_dir", "evaluation_results"), config_dir, "output_dir"
    )
    config["data_root"] = str(data_root)
    config["checkpoint_dir"] = str(checkpoint_dir)
    config["output_dir"] = str(output_dir)

    raw_paths = config.get("paths", {})
    if not isinstance(raw_paths, Mapping):
        raise ValueError("Config key 'paths' must be a mapping.")
    paths = dict(raw_paths)
    paths["checkpoint"] = str(
        _optional_path(paths.get("checkpoint"), config_dir)
        or checkpoint_dir / "best_model.pth"
    )
    paths["evaluation_output"] = str(
        _optional_path(paths.get("evaluation_output"), config_dir)
        or output_dir / "clam" / "evaluation_results"
    )
    paths["attention_output"] = str(
        _optional_path(paths.get("attention_output"), config_dir)
        or output_dir / "clam" / "attention_heatmaps"
    )
    config["paths"] = paths


def _absolute_path(value: Any, base_dir: Path, key: str) -> Path:
    """Resolve one required path value.

    Args:
        value (Any): Path-like value from configuration.
        base_dir (Path): Base directory for relative values.
        key (str): Configuration key used in errors.

    Returns:
        Path: Expanded absolute path.
    """
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError(f"Config key '{key}' must be a nonempty path.")
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base_dir / path).resolve()


def _optional_path(value: Any, base_dir: Path) -> Optional[Path]:
    """Resolve an optional path value.

    Args:
        value (Any): Optional path-like value.
        base_dir (Path): Base directory for relative values.

    Returns:
        Optional[Path]: Absolute path, or ``None`` when no value was supplied.
    """
    if value is None:
        return None
    return _absolute_path(value, base_dir, "paths entry")
