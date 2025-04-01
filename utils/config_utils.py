"""
Configuration utilities for the SNN project.

This module provides functions to load and manage configuration files.
"""

import os
import yaml
from typing import Dict, Any
from omegaconf import OmegaConf

def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """
    Load a YAML configuration file.
    
    Args:
        config_path: Path to the YAML configuration file
    
    Returns:
        Dictionary containing the configuration
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_omegaconf(config_path: str) -> OmegaConf:
    """
    Load a configuration file using OmegaConf.
    
    Args:
        config_path: Path to the configuration file
    
    Returns:
        OmegaConf configuration object
    """
    return OmegaConf.load(config_path)

def merge_configs(base_config: Dict[str, Any], 
                 override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries, with override_config taking precedence.
    
    Args:
        base_config: Base configuration
        override_config: Configuration to override base values
    
    Returns:
        Merged configuration dictionary
    """
    merged_config = base_config.copy()
    
    for key, value in override_config.items():
        if (key in merged_config and isinstance(merged_config[key], dict) 
                and isinstance(value, dict)):
            # Recursively merge nested dictionaries
            merged_config[key] = merge_configs(merged_config[key], value)
        else:
            # Override or add the key
            merged_config[key] = value
    
    return merged_config

def get_experiment_config(experiment_name: str, 
                        base_config_path: str = "configs/default.yaml",
                        experiments_dir: str = "configs/experiments") -> Dict[str, Any]:
    """
    Get configuration for a specific experiment by merging base config with experiment config.
    
    Args:
        experiment_name: Name of the experiment
        base_config_path: Path to the base configuration file
        experiments_dir: Directory containing experiment configurations
    
    Returns:
        Merged configuration dictionary
    """
    # Load base configuration
    base_config = load_yaml_config(base_config_path)
    
    # Check if experiment config exists
    experiment_config_path = os.path.join(experiments_dir, f"{experiment_name}.yaml")
    if os.path.exists(experiment_config_path):
        # Load and merge experiment config
        experiment_config = load_yaml_config(experiment_config_path)
        return merge_configs(base_config, experiment_config)
    else:
        print(f"Warning: No config found for experiment '{experiment_name}'. Using base config.")
        return base_config

def save_config(config: Dict[str, Any], output_path: str) -> None:
    """
    Save a configuration to a YAML file.
    
    Args:
        config: Configuration dictionary
        output_path: Path to save the configuration
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Configuration saved to {output_path}")

def create_experiment_directory(base_dir: str, experiment_name: str) -> str:
    """
    Create a directory for an experiment and return its path.
    
    Args:
        base_dir: Base directory for experiments
        experiment_name: Name of the experiment
    
    Returns:
        Path to the created experiment directory
    """
    experiment_dir = os.path.join(base_dir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    return experiment_dir

def generate_timestamp_id() -> str:
    """
    Generate a timestamp ID for experiment naming.
    
    Returns:
        Timestamp string in the format YYYYMMDD_HHMMSS
    """
    import time
    return time.strftime("%Y%m%d_%H%M%S")