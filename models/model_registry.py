"""
Model registry for both ANN and SNN models.

This module provides factory functions to create models of various types,
making it easy to switch between ANN and SNN implementations.
"""

import numpy as np
import torch.nn as nn
from typing import Tuple, Dict, Any

# Import model implementations
from models.snn.synthetic_snn import SyntheticSpikeModel, SpatialSpikeModel
from models.ann.synthetic_ann import SyntheticANN, SpatialANN

def create_snn_model(model_type: str, 
                    n_neurons: int, 
                    n_classes: int, 
                    length: int, 
                    config: Dict[str, Any], 
                    device: str) -> nn.Module:
    """
    Create an SNN model of the specified type.
    
    Args:
        model_type: Type of model to create ('synthetic', 'spatial', etc.)
        n_neurons: Number of input neurons
        n_classes: Number of output classes
        length: Sequence length (time steps)
        config: Configuration dictionary with hyperparameters
        device: Device to place the model on
        
    Returns:
        Instantiated SNN model
    """
    batch_size = config.hyperparameters.batch_size
    hidden_size = config.hyperparameters.hidden_size
    tau_m = config.hyperparameters.tau_m
    tau_s = config.hyperparameters.tau_s
    dropout_rate = config.hyperparameters.get('dropout_rate', 0.3)
    
    if model_type == "spatial":
        # Determine spatial dimensions (try to infer)
        height = int(np.sqrt(n_neurons))
        if height * height == n_neurons:  # Perfect square
            width = height
        else:
            # Default to a reasonable aspect ratio
            height = int(np.sqrt(n_neurons * 0.75))
            width = n_neurons // height
        
        model = SpatialSpikeModel(
            input_size=n_neurons,
            hidden_size=hidden_size,
            output_size=n_classes,
            spatial_shape=(height, width),
            length=length,
            batch_size=batch_size,
            tau_m=tau_m,
            tau_s=tau_s
        ).to(device)
    else:  # default to synthetic
        model = SyntheticSpikeModel(
            input_size=n_neurons,
            hidden_size=hidden_size,
            output_size=n_classes,
            length=length,
            batch_size=batch_size,
            tau_m=tau_m,
            tau_s=tau_s,
            dropout_rate=dropout_rate
        ).to(device)
    
    return model

def create_ann_model(model_type: str, 
                   n_neurons: int, 
                   n_classes: int, 
                   length: int, 
                   config: Dict[str, Any], 
                   device: str) -> nn.Module:
    """
    Create an ANN model of the specified type.
    
    Args:
        model_type: Type of model to create ('synthetic', 'spatial', etc.)
        n_neurons: Number of input neurons
        n_classes: Number of output classes
        length: Sequence length (time steps)
        config: Configuration dictionary with hyperparameters
        device: Device to place the model on
        
    Returns:
        Instantiated ANN model
    """
    batch_size = config.hyperparameters.batch_size
    hidden_size = config.hyperparameters.hidden_size
    dropout_rate = config.hyperparameters.get('dropout_rate', 0.3)
    
    if model_type == "spatial":
        # Determine spatial dimensions (try to infer)
        height = int(np.sqrt(n_neurons))
        if height * height == n_neurons:  # Perfect square
            width = height
        else:
            # Default to a reasonable aspect ratio
            height = int(np.sqrt(n_neurons * 0.75))
            width = n_neurons // height
        
        model = SpatialANN(
            input_size=n_neurons,
            hidden_size=hidden_size,
            output_size=n_classes,
            spatial_shape=(height, width),
            length=length,
            batch_size=batch_size,
            dropout_rate=dropout_rate
        ).to(device)
    else:  # default to synthetic
        model = SyntheticANN(
            input_size=n_neurons,
            hidden_size=hidden_size,
            output_size=n_classes,
            length=length,
            batch_size=batch_size,
            dropout_rate=dropout_rate
        ).to(device)
    
    return model

def create_model(model_arch: str, 
                model_type: str, 
                n_neurons: int, 
                n_classes: int, 
                length: int, 
                config: Dict[str, Any], 
                device: str) -> nn.Module:
    """
    Create a model of the specified architecture and type.
    
    Args:
        model_arch: Model architecture ('snn' or 'ann')
        model_type: Type of model to create ('synthetic', 'spatial', etc.)
        n_neurons: Number of input neurons
        n_classes: Number of output classes
        length: Sequence length (time steps)
        config: Configuration dictionary with hyperparameters
        device: Device to place the model on
        
    Returns:
        Instantiated model
    """
    if model_arch.lower() == 'snn':
        return create_snn_model(model_type, n_neurons, n_classes, length, config, device)
    elif model_arch.lower() == 'ann':
        return create_ann_model(model_type, n_neurons, n_classes, length, config, device)
    else:
        raise ValueError(f"Unknown model architecture: {model_arch}")