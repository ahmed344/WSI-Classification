"""Load, validate, and resolve PANTHER configuration values."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence

import yaml


SPLITS = ("train", "val", "test")


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load one validated YAML file and resolve filesystem paths."""
    path = (
        Path(config_path).expanduser()
        if config_path is not None
        else Path(__file__).with_name("config.yaml")
    ).resolve()
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, MutableMapping):
        raise ValueError(f"Configuration '{path}' must contain a YAML mapping.")

    config: Dict[str, Any] = dict(loaded)
    _validate(config)
    config["input_dim"] = _mapped_value(
        config, "feature_model_input_dims", int
    )
    config["feature_file_suffix"] = _mapped_value(
        config, "feature_model_suffixes", str
    )
    if int(config["input_dim"]) <= 0:
        raise ValueError("Selected feature input dimension must be positive.")
    if not str(config["feature_file_suffix"]).endswith(".pt"):
        raise ValueError("Selected feature suffix must end with '.pt'.")
    _resolve_paths(config, path.parent)
    config["config_path"] = str(path)
    return config


def allocate_training_run(config: Dict[str, Any]) -> Path:
    """Create and bind a unique run directory unless one was explicit."""
    explicit = config.get("_explicit_paths", {})
    if bool(explicit.get("run_dir")):
        run_dir = Path(config["paths"]["run_dir"])
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        root = Path(config["panther_results_root"])
        root.mkdir(parents=True, exist_ok=True)
        base = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        run_dir = root / base
        counter = 1
        while run_dir.exists():
            run_dir = root / f"{base}_{counter:02d}"
            counter += 1
        run_dir.mkdir(parents=True)
    _bind_run_paths(config, run_dir)
    return run_dir


def resolve_evaluation_run(config: Dict[str, Any]) -> Path:
    """Resolve evaluation to an explicit run/checkpoint or newest dated run."""
    paths = config["paths"]
    explicit = config.get("_explicit_paths", {})
    if bool(explicit.get("run_dir")):
        run_dir = Path(paths["run_dir"])
    elif bool(explicit.get("checkpoint")):
        run_dir = Path(paths["checkpoint"]).parent
    else:
        root = Path(config["panther_results_root"])
        candidates = sorted(
            (
                item
                for item in root.iterdir()
                if item.is_dir() and (item / "best_model.pth").is_file()
            ),
            key=lambda item: item.name,
            reverse=True,
        ) if root.is_dir() else []
        if not candidates:
            raise FileNotFoundError(
                f"No completed PANTHER run found under {root}. Train first or "
                "set paths.run_dir/paths.checkpoint."
            )
        run_dir = candidates[0]
    if not run_dir.is_dir():
        raise FileNotFoundError(f"PANTHER run directory not found: {run_dir}")
    _bind_run_paths(config, run_dir, preserve_explicit=True)
    return run_dir


def _mapped_value(
    config: Mapping[str, Any], key: str, value_type: type
) -> Any:
    mapping = config.get(key)
    if not isinstance(mapping, Mapping):
        raise ValueError(f"Config key '{key}' must be a mapping.")
    model = str(config.get("feature_model"))
    if model not in mapping:
        raise ValueError(
            f"feature_model '{model}' is missing from '{key}'. "
            f"Available: {', '.join(sorted(str(item) for item in mapping))}."
        )
    return value_type(mapping[model])


def _validate(config: Mapping[str, Any]) -> None:
    for key in ("data_root", "output_dir", "feature_model"):
        if not isinstance(config.get(key), str) or not str(config[key]).strip():
            raise ValueError(f"Config key '{key}' must be a nonempty string.")

    ratios = [_number(config, f"{split}_ratio") for split in SPLITS]
    if any(value < 0.0 or value > 1.0 for value in ratios):
        raise ValueError("Split ratios must be between zero and one.")
    if abs(sum(ratios) - 1.0) > 1e-8 or ratios[0] <= 0.0:
        raise ValueError("Split ratios must sum to one with train_ratio > 0.")
    _nonnegative_int(config, "random_seed")

    class_folders = config.get("class_folders")
    if class_folders is not None and (
        not isinstance(class_folders, list)
        or not class_folders
        or not all(isinstance(name, str) and name for name in class_folders)
        or len(set(class_folders)) != len(class_folders)
    ):
        raise ValueError("class_folders must be null or a list of unique names.")
    _choice(config, "feature_normalization", ("none", "l2"))
    _choice(config, "tile_sampling", ("random", "uniform", "first"))
    cap = config.get("max_tiles_per_slide")
    if cap is not None and (isinstance(cap, bool) or not isinstance(cap, int) or cap <= 0):
        raise ValueError("max_tiles_per_slide must be a positive integer or null.")

    prototype = _section(config, "prototype")
    _choice(prototype, "method", ("kmeans",))
    for key in (
        "num_prototypes", "patches_per_prototype", "max_iterations",
        "num_initializations",
    ):
        _positive_int(prototype, key)
    _choice(prototype, "algorithm", ("lloyd", "elkan"))

    model = _section(config, "model")
    _positive_int(model, "em_iterations")
    for key in ("tau", "covariance_regularizer", "variance_floor"):
        if _number(model, key) <= 0.0:
            raise ValueError(f"model.{key} must be positive.")
    _choice(model, "output_type", ("allcat",))
    if not isinstance(model.get("fix_prototypes"), bool):
        raise ValueError("model.fix_prototypes must be a boolean.")
    chunk = model.get("em_chunk_size")
    if chunk is not None and (
        isinstance(chunk, bool) or not isinstance(chunk, int) or chunk <= 0
    ):
        raise ValueError("model.em_chunk_size must be a positive integer or null.")

    training = _section(config, "training")
    for key in ("epochs", "batch_size", "gradient_accumulation_steps"):
        _positive_int(training, key)
    _nonnegative_int(training, "warmup_epochs")
    for key in ("learning_rate", "weight_decay", "input_dropout"):
        value = _number(training, key)
        if value < 0.0 or (key == "learning_rate" and value == 0.0):
            raise ValueError(f"training.{key} has an invalid value.")
    if float(training["input_dropout"]) >= 1.0:
        raise ValueError("training.input_dropout must be in [0, 1).")
    _choice(training, "optimizer", ("adamw", "sgd"))
    _choice(training, "scheduler", ("cosine", "linear", "constant"))
    _choice(training, "checkpoint_selection", ("last", "best_val_loss"))
    for key in ("classifier_bias", "class_weighted_loss"):
        if not isinstance(training.get(key), bool):
            raise ValueError(f"training.{key} must be a boolean.")

    evaluation = _section(config, "evaluation")
    splits = evaluation.get("splits")
    if not isinstance(splits, list) or not splits or any(s not in SPLITS for s in splits):
        raise ValueError("evaluation.splits must be a nonempty split-name list.")
    if not isinstance(evaluation.get("include_train"), bool):
        raise ValueError("evaluation.include_train must be a boolean.")
    _positive_int(evaluation, "confusion_matrix_dpi")

    visualization = _section(config, "visualization")
    _choice(visualization, "split", SPLITS)
    maximum = visualization.get("max_slides")
    if maximum is not None and (
        isinstance(maximum, bool) or not isinstance(maximum, int) or maximum <= 0
    ):
        raise ValueError("visualization.max_slides must be a positive integer or null.")
    for key in ("tile_size", "downsample", "dpi"):
        _positive_int(visualization, key)
    alpha = _number(visualization, "alpha")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("visualization.alpha must be in [0, 1].")
    if not isinstance(visualization.get("save_mixture_proportions"), bool):
        raise ValueError("visualization.save_mixture_proportions must be a boolean.")
    if not isinstance(visualization.get("save_individual_tissues"), bool):
        raise ValueError("visualization.save_individual_tissues must be a boolean.")

    runtime = _section(config, "runtime")
    _choice(runtime, "device", ("auto", "cuda", "cpu"))
    _nonnegative_int(runtime, "num_workers")
    if not isinstance(runtime.get("pin_memory"), bool):
        raise ValueError("runtime.pin_memory must be a boolean.")
    _section(config, "paths")


def _resolve_paths(config: Dict[str, Any], config_dir: Path) -> None:
    for key in ("data_root", "output_dir"):
        path = Path(str(config[key])).expanduser()
        config[key] = str((path if path.is_absolute() else config_dir / path).resolve())
    config["panther_results_root"] = str(Path(config["output_dir"]) / "panther")
    raw = dict(config["paths"])
    explicit: Dict[str, bool] = {}
    for key in (
        "run_dir", "checkpoint", "evaluation_output", "visualization_output"
    ):
        value = raw.get(key)
        explicit[key] = value is not None
        if value is not None:
            path = Path(str(value)).expanduser()
            raw[key] = str((path if path.is_absolute() else config_dir / path).resolve())
    config["paths"] = raw
    config["_explicit_paths"] = explicit


def _bind_run_paths(
    config: Dict[str, Any], run_dir: Path, preserve_explicit: bool = False
) -> None:
    run_dir = run_dir.resolve()
    paths = config["paths"]
    explicit = config.get("_explicit_paths", {})
    paths["run_dir"] = str(run_dir)
    if not preserve_explicit or not bool(explicit.get("checkpoint")):
        paths["checkpoint"] = str(run_dir / "best_model.pth")
    if not preserve_explicit or not bool(explicit.get("evaluation_output")):
        paths["evaluation_output"] = str(run_dir / "evaluation_results")
    if not preserve_explicit or not bool(explicit.get("visualization_output")):
        paths["visualization_output"] = str(run_dir / "visualization_results")


def _section(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"Config key '{key}' must be a mapping.")
    return value


def _choice(config: Mapping[str, Any], key: str, options: Sequence[str]) -> None:
    if config.get(key) not in options:
        raise ValueError(f"Invalid {key} '{config.get(key)}'; expected {options}.")


def _number(config: Mapping[str, Any], key: str) -> float:
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Config key '{key}' must be numeric.")
    result = float(value)
    if result != result or result in (float("inf"), float("-inf")):
        raise ValueError(f"Config key '{key}' must be finite.")
    return result


def _positive_int(config: Mapping[str, Any], key: str) -> int:
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"Config key '{key}' must be a positive integer.")
    return value


def _nonnegative_int(config: Mapping[str, Any], key: str) -> int:
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"Config key '{key}' must be a nonnegative integer.")
    return value
