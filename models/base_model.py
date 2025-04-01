"""
Base SNN model implementations.

This module contains the base class for all SNN models, defining the common
interface and shared functionality.
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict, List, Optional, Union, Any

class BaseSNN(nn.Module):
    """
    Base class for all SNN models.
    
    This class implements common functionality for SNN architectures and
    defines the interface that specific SNN implementations should follow.
    """
    
    def __init__(self, input_size: int, output_size: int, length: int, batch_size: int):
        """
        Initialize the base SNN model.
        
        Args:
            input_size: Number of input neurons
            output_size: Number of output neurons
            length: Number of time steps in the simulation
            batch_size: Batch size for training/inference
        """
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.length = length
        self.batch_size = batch_size
        self.monitors = {}
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape [batch_size, input_size, length]
            
        Returns:
            Output tensor of shape [batch_size, output_size, length]
        """
        raise NotImplementedError("Subclasses must implement forward method")
    
    def add_monitor(self, name: str, layer) -> None:
        """
        Add a layer to be monitored during forward passes.
        
        Args:
            name: Name identifier for the monitored layer
            layer: Layer to monitor
        """
        self.monitors[name] = {"layer": layer, "activity": []}
    
    def clear_monitors(self) -> None:
        """Clear all monitoring data."""
        for monitor in self.monitors.values():
            monitor["activity"] = []
    
    def get_monitor_data(self) -> Dict[str, Any]:
        """
        Get the collected monitoring data.
        
        Returns:
            Dictionary containing monitoring data for each monitored layer
        """
        return {name: {"activity": monitor["activity"]} 
                for name, monitor in self.monitors.items()}
    
    def count_parameters(self) -> int:
        """
        Count the total number of trainable parameters in the model.
        
        Returns:
            Number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def reset_state(self) -> None:
        """
        Reset the internal state of all layers.
        Should be implemented by subclasses with stateful layers.
        """
        pass

class SyntheticSNN(BaseSNN):
    """
    Base class for SNNs designed for synthetic data experiments.
    
    This class provides additional functionality specific to synthetic data experiments.
    """
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int, 
                 length: int, batch_size: int, tau_m: float = 4.0, tau_s: float = 1.0):
        """
        Initialize the synthetic SNN model.
        
        Args:
            input_size: Number of input neurons
            hidden_size: Number of neurons in hidden layers
            output_size: Number of output neurons
            length: Number of time steps in the simulation
            batch_size: Batch size for training/inference
            tau_m: Membrane time constant
            tau_s: Synaptic time constant
        """
        super().__init__(input_size, output_size, length, batch_size)
        self.hidden_size = hidden_size
        self.tau_m = tau_m
        self.tau_s = tau_s
    
    def analyze_activations(self, input_data: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Analyze layer activations for a given input.
        
        Args:
            input_data: Input data tensor
            
        Returns:
            Dictionary containing activation statistics for each layer
        """
        self.eval()
        with torch.no_grad():
            # Forward pass with monitoring
            self.clear_monitors()
            output = self(input_data)
            
            # Analyze the activations
            activation_stats = {}
            for name, monitor in self.monitors.items():
                if len(monitor["activity"]) > 0:
                    activities = torch.stack(monitor["activity"], dim=0)
                    activation_stats[name] = {
                        "mean": activities.mean().item(),
                        "std": activities.std().item(),
                        "max": activities.max().item(),
                        "firing_rate": activities.mean(dim=(0, 1)).sum().item() / self.length
                    }
            
            return activation_stats