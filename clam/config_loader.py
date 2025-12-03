import yaml
import os
from pathlib import Path

def load_config(config_path=None):
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config.yml file. If None, looks for config.yml in the same directory.
    
    Returns:
        dict: Configuration dictionary
    """
    if config_path is None:
        # Get the directory where this script is located
        script_dir = Path(__file__).parent
        config_path = script_dir / 'config.yml'
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set default paths if not specified
    if config.get('paths', {}).get('checkpoint') is None:
        checkpoint_dir = config.get('checkpoint_dir', 'checkpoints')
        config['paths'] = config.get('paths', {})
        config['paths']['checkpoint'] = os.path.join(checkpoint_dir, 'best_model.pth')
    
    if config.get('paths', {}).get('evaluation_output') is None:
        output_dir = config.get('output_dir', 'evaluation_results')
        config['paths'] = config.get('paths', {})
        config['paths']['evaluation_output'] = os.path.join(output_dir, 'evaluation_results')
    
    if config.get('paths', {}).get('attention_output') is None:
        output_dir = config.get('output_dir', 'evaluation_results')
        config['paths'] = config.get('paths', {})
        config['paths']['attention_output'] = os.path.join(output_dir, 'attention_heatmaps')
    
    return config

