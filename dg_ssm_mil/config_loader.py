"""
Configuration loading utilities for the independent DG-SSM-MIL workflow.
"""
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple
import os

import yaml


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load and validate a DG-SSM-MIL YAML configuration file.

    Args:
        config_path (Optional[str]): Path to the YAML file. If None, the
            `config.yml` file next to this module is used.

    Returns:
        Dict[str, Any]: Parsed configuration with default output paths filled in.
    """
    if config_path is None:
        config_path = str(Path(__file__).parent / "config.yml")

    with open(config_path, "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, dict):
        raise ValueError("DG-SSM-MIL config must contain a YAML mapping.")

    config["paths"] = config.get("paths", {}) or {}
    checkpoint_dir = str(config.get("checkpoint_dir", "dg_ssm_mil/checkpoints"))
    output_dir = str(config.get("output_dir", "dg_ssm_mil/results"))

    if config["paths"].get("checkpoint") is None:
        config["paths"]["checkpoint"] = os.path.join(checkpoint_dir, "best_model.pth")
    if config["paths"].get("final_checkpoint") is None:
        config["paths"]["final_checkpoint"] = os.path.join(
            checkpoint_dir, "final_model.pth"
        )
    if config["paths"].get("training_history") is None:
        config["paths"]["training_history"] = os.path.join(
            output_dir, "training_history.json"
        )
    if config["paths"].get("training_report") is None:
        config["paths"]["training_report"] = os.path.join(
            output_dir, "best_model_report.json"
        )
    if config["paths"].get("training_plot") is None:
        config["paths"]["training_plot"] = os.path.join(
            output_dir, "training_history.png"
        )
    if config["paths"].get("evaluation_output") is None:
        config["paths"]["evaluation_output"] = os.path.join(
            output_dir, "evaluation_results"
        )
    if config["paths"].get("attention_output") is None:
        config["paths"]["attention_output"] = os.path.join(
            output_dir, "attention_heatmaps"
        )

    config["input_dim"] = resolve_input_dim(config)
    validate_config(config)
    return config


def resolve_feature_file_suffix(config: Mapping[str, Any]) -> str:
    """
    Resolve the feature tensor filename suffix for the selected feature model.

    Args:
        config (Mapping[str, Any]): Parsed configuration containing
            `feature_model` and `feature_model_suffixes`.

    Returns:
        str: Feature filename suffix, including the `.pt` extension.
    """
    selected_model = str(config.get("feature_model", "default"))
    suffix_map = config.get("feature_model_suffixes", {})
    if not isinstance(suffix_map, Mapping):
        raise ValueError("feature_model_suffixes must be a mapping.")
    if selected_model not in suffix_map:
        available = ", ".join(sorted(str(key) for key in suffix_map.keys()))
        raise ValueError(
            f"Unknown feature_model '{selected_model}'. Available models: {available}."
        )
    suffix = str(suffix_map[selected_model])
    if not suffix.endswith(".pt"):
        raise ValueError(
            f"Invalid feature suffix '{suffix}' for model '{selected_model}'."
        )
    return suffix


def resolve_input_dim(config: Mapping[str, Any]) -> int:
    """
    Resolve the input feature dimension for the selected feature model.

    Args:
        config (Mapping[str, Any]): Parsed configuration containing
            `feature_model` and `feature_model_input_dims`.

    Returns:
        int: Positive input feature dimension used by `DGSSMMILModel`.
    """
    selected_model = str(config.get("feature_model", "default"))
    input_dim_map = config.get("feature_model_input_dims", {})
    if not isinstance(input_dim_map, Mapping):
        raise ValueError("feature_model_input_dims must be a mapping.")
    if selected_model not in input_dim_map:
        available = ", ".join(sorted(str(key) for key in input_dim_map.keys()))
        raise ValueError(
            f"Unknown feature_model '{selected_model}'. Available input dims: {available}."
        )
    input_dim = int(input_dim_map[selected_model])
    if input_dim <= 0:
        raise ValueError(
            f"Invalid input_dim '{input_dim}' for model '{selected_model}'."
        )
    return input_dim


def resolve_device(config: Mapping[str, Any]) -> str:
    """
    Resolve the configured training device.

    Args:
        config (Mapping[str, Any]): Parsed configuration with an optional
            `device` key.

    Returns:
        str: Device name suitable for `torch.device`.
    """
    import torch

    requested_device = str(config.get("device", "auto")).lower()
    if requested_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested_device not in {"cpu", "cuda"}:
        raise ValueError("device must be one of: auto, cpu, cuda.")
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise ValueError("device is set to cuda, but CUDA is not available.")
    return requested_device


def get_coordinate_columns(config: Mapping[str, Any]) -> Tuple[str, str]:
    """
    Return the configured CSV coordinate column names.

    Args:
        config (Mapping[str, Any]): Parsed configuration with `coordinate_columns`.

    Returns:
        Tuple[str, str]: Column names for x and y coordinates.
    """
    columns = config.get("coordinate_columns", ["x", "y"])
    if not isinstance(columns, list) or len(columns) != 2:
        raise ValueError("coordinate_columns must be a two-item list, e.g. ['x', 'y'].")
    return str(columns[0]), str(columns[1])


def validate_config(config: Mapping[str, Any]) -> None:
    """
    Validate required DG-SSM-MIL configuration values.

    Args:
        config (Mapping[str, Any]): Parsed configuration dictionary.

    Returns:
        None: Raises ValueError when a required setting is invalid.
    """
    required_keys = [
        "data_root",
        "hidden_dim",
        "num_classes",
        "spatial_knn_k",
        "dynamic_graph_top_k",
        "learning_rate",
        "epochs",
        "batch_size",
    ]
    missing = [key for key in required_keys if key not in config]
    if missing:
        raise ValueError(f"Missing required config keys: {', '.join(missing)}")

    if int(config["num_classes"]) <= 1:
        raise ValueError("num_classes must be greater than 1.")
    input_dim = resolve_input_dim(config)
    if input_dim <= 0 or int(config["hidden_dim"]) <= 0:
        raise ValueError("input_dim and hidden_dim must be positive.")
    if int(config["spatial_knn_k"]) <= 0:
        raise ValueError("spatial_knn_k must be positive.")
    if int(config["dynamic_graph_top_k"]) <= 0:
        raise ValueError("dynamic_graph_top_k must be positive.")
    if int(config.get("dynamic_graph_chunk_size", 512)) <= 0:
        raise ValueError("dynamic_graph_chunk_size must be positive.")
    dynamic_lambda = float(config.get("dynamic_graph_lambda", 0.5))
    if not 0.0 <= dynamic_lambda <= 1.0:
        raise ValueError("dynamic_graph_lambda must be in [0, 1].")
    if str(config.get("dynamic_graph_activation", "silu")) not in {
        "silu",
        "relu",
        "gelu",
    }:
        raise ValueError("dynamic_graph_activation must be silu, relu, or gelu.")
    if str(config.get("attention_type", "standard")) not in {"standard", "gated"}:
        raise ValueError("attention_type must be either 'standard' or 'gated'.")
    if int(config.get("gradient_accumulation_steps", 1)) <= 0:
        raise ValueError("gradient_accumulation_steps must be positive.")
    for key in [
        "max_tiles_per_tissue_train",
        "max_tiles_per_tissue_val",
        "max_tiles_per_tissue_test",
    ]:
        value = config.get(key)
        if value is not None and int(value) <= 0:
            raise ValueError(f"{key} must be positive or null.")
    skip_threshold = config.get("skip_tissues_above_tiles")
    if skip_threshold is not None and int(skip_threshold) <= 0:
        raise ValueError("skip_tissues_above_tiles must be positive or null.")
    if str(config.get("coordinate_mismatch", "error")) != "error":
        raise ValueError("coordinate_mismatch must be 'error' for spatial alignment.")
    if bool(config.get("sort_tiles_spatially", False)):
        raise ValueError("sort_tiles_spatially must be false for the original H path.")
    if str(config.get("bag_level", "tissue")) not in {"tissue", "slide"}:
        raise ValueError("bag_level must be tissue or slide.")
    ratios = [
        float(config.get("train_ratio", 0.8)),
        float(config.get("val_ratio", 0.1)),
        float(config.get("test_ratio", 0.1)),
    ]
    if any(ratio < 0.0 for ratio in ratios) or not abs(sum(ratios) - 1.0) < 1e-8:
        raise ValueError("train_ratio, val_ratio, and test_ratio must sum to 1.")
    if int(config.get("monte_carlo_repeats", 10)) <= 0:
        raise ValueError("monte_carlo_repeats must be positive.")
    get_coordinate_columns(config)
    resolve_feature_file_suffix(config)
