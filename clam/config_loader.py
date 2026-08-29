"""Load, validate, and resolve CLAM configuration values."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence

import yaml


_OPTIONS: Dict[str, Sequence[str]] = {
    "model_type": ("clam_mb", "clam_sb"),
    "bag_level": ("tissue", "slide"),
    "feature_normalization": ("none", "l2", "layer_norm"),
    "tile_sampling": ("random", "uniform", "first"),
}
_SPLITS = ("train", "val", "test")
_RUN_ID_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{6}(?:_\d{2})?$")
_EXPLICIT_PATH_KEYS = ("checkpoint", "evaluation_output", "attention_output")


def clam_results_root(output_dir: str | Path) -> Path:
    """Return the CLAM results root under an output directory.

    Args:
        output_dir (str | Path): Base results directory from configuration.

    Returns:
        Path: ``{output_dir}/clam`` path.
    """
    return Path(output_dir) / "clam"


def make_run_id(moment: Optional[datetime] = None) -> str:
    """Build a sortable timestamp run identifier.

    Args:
        moment (Optional[datetime]): Instant to format, or ``None`` for now.

    Returns:
        str: Run id shaped ``YYYY-MM-DD_HHMMSS``.
    """
    value = moment if moment is not None else datetime.now()
    return value.strftime("%Y-%m-%d_%H%M%S")


def apply_run_artifact_paths(config: Dict[str, Any], run_dir: str | Path) -> None:
    """Point checkpoint and artifact paths at one dated training run.

    Args:
        config (Dict[str, Any]): Configuration dictionary to mutate.
        run_dir (str | Path): Absolute or relative run directory.

    Returns:
        None: ``checkpoint_dir`` and ``paths`` entries are updated in place.
    """
    destination = Path(run_dir).expanduser().resolve()
    config["checkpoint_dir"] = str(destination)
    paths = dict(config.get("paths", {}) or {})
    paths["checkpoint"] = str(destination / "best_model.pth")
    paths["evaluation_output"] = str(destination / "evaluation_results")
    paths["attention_output"] = str(destination / "attention_heatmaps")
    config["paths"] = paths


def list_training_run_dirs(clam_root: str | Path) -> List[Path]:
    """List dated or checkpoint-bearing CLAM training run directories.

    Args:
        clam_root (str | Path): ``{output_dir}/clam`` directory.

    Returns:
        List[Path]: Candidate run directories, newest first.
    """
    root = Path(clam_root)
    if not root.is_dir():
        return []

    timestamped: List[Path] = []
    checkpoint_dirs: List[Path] = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        if _RUN_ID_PATTERN.match(child.name):
            timestamped.append(child)
        elif (child / "best_model.pth").is_file():
            checkpoint_dirs.append(child)

    timestamped.sort(key=lambda path: path.name, reverse=True)
    checkpoint_dirs.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    seen = {path.resolve() for path in timestamped}
    ordered = list(timestamped)
    for path in checkpoint_dirs:
        resolved = path.resolve()
        if resolved not in seen:
            ordered.append(path)
            seen.add(resolved)
    return ordered


def resolve_latest_training_run(clam_root: str | Path) -> Optional[Path]:
    """Select the newest CLAM training run directory.

    Args:
        clam_root (str | Path): ``{output_dir}/clam`` directory.

    Returns:
        Optional[Path]: Newest run directory, or ``None`` when none exist.
    """
    runs = list_training_run_dirs(clam_root)
    return runs[0] if runs else None


def should_allocate_dated_training_run(config: Mapping[str, Any]) -> bool:
    """Return whether training should create a new dated results directory.

    Args:
        config (Mapping[str, Any]): Configuration that may carry path flags from
            ``load_config``. Programmatic configs without ``_explicit_paths``
            skip dated allocation.

    Returns:
        bool: ``True`` when null YAML path defaults should receive a new run.
    """
    explicit = config.get("_explicit_paths")
    if not isinstance(explicit, Mapping):
        return False
    return not bool(explicit.get("checkpoint", False))


def allocate_training_run(config: Dict[str, Any]) -> Path:
    """Create a new dated run directory when path defaults are in use.

    When ``paths.checkpoint`` was explicitly set in YAML, or when the config
    was not produced by ``load_config``, the existing checkpoint directory is
    reused and no new dated folder is created.

    Args:
        config (Dict[str, Any]): Resolved configuration to mutate for a run.

    Returns:
        Path: Directory that should receive checkpoints and related artifacts.
    """
    if not should_allocate_dated_training_run(config):
        destination = Path(str(config.get("checkpoint_dir", ".")))
        destination.mkdir(parents=True, exist_ok=True)
        return destination.resolve()

    root = clam_results_root(config["output_dir"])
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
    """Return whether the checkpoint path was supplied explicitly in YAML.

    Args:
        config (Mapping[str, Any]): Configuration that may carry path flags.

    Returns:
        bool: ``True`` when ``paths.checkpoint`` was non-null in the YAML.
    """
    explicit = config.get("_explicit_paths")
    if isinstance(explicit, Mapping):
        return bool(explicit.get("checkpoint", False))
    return True


def resolve_inference_run_paths(config: Dict[str, Any]) -> Optional[Path]:
    """Bind null-derived artifact paths to the newest dated training run.

    Args:
        config (Dict[str, Any]): Configuration whose default paths may be updated.

    Returns:
        Optional[Path]: Latest run directory when applied, otherwise ``None``.
    """
    explicit = config.get("_explicit_paths", {})
    if not isinstance(explicit, Mapping):
        explicit = {}
    if any(bool(explicit.get(key, False)) for key in _EXPLICIT_PATH_KEYS):
        return None

    latest = resolve_latest_training_run(clam_results_root(config["output_dir"]))
    if latest is None:
        return None
    apply_run_artifact_paths(config, latest)
    return latest


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
    slide_batch_size = config.get("slide_evaluation_batch_size", 1)
    if (
        isinstance(slide_batch_size, bool)
        or not isinstance(slide_batch_size, int)
        or slide_batch_size <= 0
    ):
        raise ValueError(
            "Config key 'slide_evaluation_batch_size' must be a positive integer."
        )
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
        "slide_balanced_accuracy",
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

    evaluation = config.get("evaluation", {})
    if not isinstance(evaluation, Mapping):
        raise ValueError("Config key 'evaluation' must be a mapping.")
    unknown_evaluation_keys = set(evaluation) - {
        "supplementary_bag_level",
        "include_train",
    }
    if unknown_evaluation_keys:
        raise ValueError(
            "Unknown evaluation key(s): "
            + ", ".join(sorted(str(key) for key in unknown_evaluation_keys))
        )
    supplementary_level = evaluation.get("supplementary_bag_level")
    if supplementary_level not in (None, "tissue", "slide"):
        raise ValueError(
            "Config key 'evaluation.supplementary_bag_level' must be "
            "null, 'tissue', or 'slide'."
        )
    if not isinstance(evaluation.get("include_train", False), bool):
        raise ValueError("Config key 'evaluation.include_train' must be a boolean.")

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

    When ``paths.*`` entries are null, bind them to the newest dated run under
    ``{output_dir}/clam/`` when one exists; otherwise fall back to the legacy
    flat layout (``checkpoint_dir/best_model.pth`` and flat clam artifact dirs).

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
    config["clam_results_root"] = str(clam_results_root(output_dir))

    raw_paths = config.get("paths", {})
    if not isinstance(raw_paths, Mapping):
        raise ValueError("Config key 'paths' must be a mapping.")
    paths = dict(raw_paths)
    explicit_checkpoint = _optional_path(paths.get("checkpoint"), config_dir)
    explicit_evaluation = _optional_path(paths.get("evaluation_output"), config_dir)
    explicit_attention = _optional_path(paths.get("attention_output"), config_dir)
    config["_explicit_paths"] = {
        "checkpoint": explicit_checkpoint is not None,
        "evaluation_output": explicit_evaluation is not None,
        "attention_output": explicit_attention is not None,
    }

    if explicit_checkpoint is not None:
        paths["checkpoint"] = str(explicit_checkpoint)
    if explicit_evaluation is not None:
        paths["evaluation_output"] = str(explicit_evaluation)
    if explicit_attention is not None:
        paths["attention_output"] = str(explicit_attention)
    config["paths"] = paths

    auto_keys_missing = [
        key
        for key in _EXPLICIT_PATH_KEYS
        if not bool(config["_explicit_paths"][key])
    ]
    if not auto_keys_missing:
        return

    latest = resolve_latest_training_run(clam_results_root(output_dir))
    if latest is not None and len(auto_keys_missing) == len(_EXPLICIT_PATH_KEYS):
        apply_run_artifact_paths(config, latest)
        return

    if latest is not None:
        for key, relative in (
            ("checkpoint", "best_model.pth"),
            ("evaluation_output", "evaluation_results"),
            ("attention_output", "attention_heatmaps"),
        ):
            if not bool(config["_explicit_paths"][key]):
                config["paths"][key] = str(latest / relative)
        if not bool(config["_explicit_paths"]["checkpoint"]):
            config["checkpoint_dir"] = str(latest)
        return

    if not bool(config["_explicit_paths"]["checkpoint"]):
        config["paths"]["checkpoint"] = str(checkpoint_dir / "best_model.pth")
    if not bool(config["_explicit_paths"]["evaluation_output"]):
        config["paths"]["evaluation_output"] = str(
            output_dir / "clam" / "evaluation_results"
        )
    if not bool(config["_explicit_paths"]["attention_output"]):
        config["paths"]["attention_output"] = str(
            output_dir / "clam" / "attention_heatmaps"
        )


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
