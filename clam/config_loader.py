"""
Configuration loader for CLAM-MB training and evaluation.

This module provides functionality to load and process configuration files
in YAML format, with automatic path resolution for checkpoints and outputs.
"""
from typing import Dict, Any, Optional, Mapping
import yaml
import os
from pathlib import Path


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    If config_path is not provided, looks for config.yml in the same directory
    as this script. Automatically resolves default paths for checkpoints and
    output directories if not specified in the config file.
    
    Args:
        config_path (Optional[str]): Path to config.yml file. If None, looks for
            config.yml in the same directory as this script. Defaults to None.
    
    Returns:
        Dict[str, Any]: Configuration dictionary containing all settings from
            the YAML file, with resolved paths in the 'paths' section:
            - 'checkpoint' (str): Path to checkpoint file.
            - 'evaluation_output' (str): Path to evaluation results directory.
            - 'attention_output' (str): Path to attention visualization directory.
    """
    # Resolve config file path
    if config_path is None:
        # Get the directory where this script is located
        script_dir = Path(__file__).parent
        config_path = script_dir / 'config.yml'
    
    # Load YAML configuration file
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    config['input_dim'] = resolve_input_dim(config)
    
    # Set default paths if not specified in config
    if config.get('paths', {}).get('checkpoint') is None:
        checkpoint_dir = config.get('checkpoint_dir', 'checkpoints')
        config['paths'] = config.get('paths', {})
        # Default checkpoint path: checkpoint_dir/best_model.pth
        config['paths']['checkpoint'] = os.path.join(
            checkpoint_dir, 'best_model.pth'
        )
    
    if config.get('paths', {}).get('evaluation_output') is None:
        output_dir = config.get('output_dir', 'evaluation_results')
        config['paths'] = config.get('paths', {})
        # Default evaluation output: output_dir/clam/evaluation_results
        config['paths']['evaluation_output'] = os.path.join(
            output_dir, 'clam', 'evaluation_results'
        )
    
    if config.get('paths', {}).get('attention_output') is None:
        output_dir = config.get('output_dir', 'evaluation_results')
        config['paths'] = config.get('paths', {})
        # Default attention output: output_dir/clam/attention_heatmaps
        config['paths']['attention_output'] = os.path.join(
            output_dir, 'clam', 'attention_heatmaps'
        )
    
    return config


def resolve_input_dim(config: Mapping[str, Any]) -> int:
    """
    Resolve the input feature dimensionality based on selected feature model.

    Args:
        config (Mapping[str, Any]): Parsed configuration dictionary that may include
            `feature_model` and `feature_model_input_dims` keys.

    Returns:
        int: Input feature dimensionality consumed by the CLAM model.
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
    """
    Resolve the feature filename suffix based on selected feature model.

    Args:
        config (Mapping[str, Any]): Parsed configuration dictionary that may include
            `feature_model` and `feature_model_suffixes` keys.

    Returns:
        str: Feature filename suffix including extension (e.g., `_features.pt` or
            `_features_hoptimus.pt`) used by dataset discovery.
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
