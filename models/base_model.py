"""
Base model implementations for SNN and ANN models.

This module contains base classes for all neural network models in the project,
defining common interfaces and shared functionality.
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict, List, Optional, Union, Any

class BaseModel(nn.Module):
    """
    Base class for all neural network models.
    
    This class implements common functionality and
    defines the interface that specific model implementations should follow.
    """
    
    def __init__(self, input_size: int, output_size: int):
        """
        Initialize the base model.
        
        Args:
            input_size: Number of input neurons/features
            output_size: Number of output neurons/classes
        """
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.monitors = {}
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
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

class BaseSNN(BaseModel):
    """
    Base class for Spiking Neural Network models.
    
    This class provides additional functionality specific to SNNs.
    """
    
    def __init__(self, input_size: int, output_size: int, length: int, batch_size: int):
        """
        Initialize the base SNN model.
        
        Args:
            input_size: Number of input neurons
            output_size: Number of output neurons/classes
            length: Number of time steps in the simulation
            batch_size: Batch size for training/inference
        """
        super().__init__(input_size, output_size)
        self.length = length
        self.batch_size = batch_size

class BaseANN(BaseModel):
    """
    Base class for Artificial Neural Network models.
    
    This class provides additional functionality specific to ANNs.
    """
    
    def __init__(self, input_size: int, output_size: int):
        """
        Initialize the base ANN model.
        
        Args:
            input_size: Number of input features
            output_size: Number of output classes
        """
        super().__init__(input_size, output_size)

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
            output_size: Number of output neurons/classes
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

class SyntheticANN(BaseANN):
    """
    Base class for ANNs designed for synthetic data experiments.
    
    This class provides additional functionality for ANNs working with
    the same synthetic data as the SNNs for fair comparison.
    """
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 length: int = 100, batch_size: int = 32):
        """
        Initialize the synthetic ANN model.
        
        Args:
            input_size: Number of input features
            hidden_size: Number of neurons in hidden layers
            output_size: Number of output classes
            length: Time steps for temporal data (for compatibility with SNN)
            batch_size: Batch size (for compatibility with SNN)
        """
        super().__init__(input_size, output_size)
        self.hidden_size = hidden_size
        self.length = length
        self.batch_size = batch_size
        
    def reshape_input_if_needed(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reshape input tensor if it has incorrect dimensions.
        
        Args:
            x: Input tensor of shape [batch_size, input_size] or [batch_size, input_size, length]
            
        Returns:
            Reshaped tensor of shape [batch_size, input_size, length]
        """
        if len(x.shape) == 2:  # [batch_size, features]
            # Reshape to [batch_size, input_size, 1] 
            x = x.view(x.size(0), self.input_size, 1)
            # Repeat to match the expected length
            x = x.repeat(1, 1, self.length)
        
        return x