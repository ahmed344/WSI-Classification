"""
Configuration loader for CLAM-MB training and evaluation.

This module provides functionality to load and process configuration files
in YAML format, with automatic path resolution for checkpoints and outputs.
"""
from typing import Dict, Any, Optional
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
        # Default evaluation output: output_dir/evaluation_results
        config['paths']['evaluation_output'] = os.path.join(
            output_dir, 'evaluation_results'
        )
    
    if config.get('paths', {}).get('attention_output') is None:
        output_dir = config.get('output_dir', 'evaluation_results')
        config['paths'] = config.get('paths', {})
        # Default attention output: output_dir/attention_heatmaps
        config['paths']['attention_output'] = os.path.join(
            output_dir, 'attention_heatmaps'
        )
    
    return config
