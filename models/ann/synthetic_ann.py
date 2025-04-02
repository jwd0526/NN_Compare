"""
ANN model for synthetic data experiments.

This module contains a basic Artificial Neural Network (ANN) implementation
designed to be trained on the same synthetic data used by SNN models,
enabling direct comparison between ANN and SNN performance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, List, Optional, Union, Any
import sys
import os

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.base_model import SyntheticANN

class SimpleClassifier(nn.Module):
    """A simple 2-layer classifier with normalized inputs."""
    
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.5):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.bn = nn.BatchNorm1d(input_size)
    
    def forward(self, x):
        x = self.bn(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class SyntheticANN(SyntheticANN):
    """
    Ultra-simplified ANN model for synthetic spike pattern classification.
    """
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int, 
                 length: int, batch_size: int, dropout_rate: float = 0.3):
        """
        Initialize the ANN model.
        """
        super().__init__(input_size, hidden_size, output_size, length, batch_size)
        
        # Store model parameters as instance variables
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.length = length
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        
        # Create a simple classifier
        self.classifier = SimpleClassifier(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            dropout_rate=dropout_rate
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape [batch_size, features] or [batch_size, neurons, time]
            
        Returns:
            Output tensor of shape [batch_size, output_size]
        """
        # If input is 3D [batch, neurons, time], take the sum over time
        if len(x.shape) == 3:
            x = torch.sum(x, dim=2)  # [batch, neurons]
        
        # Handle input size mismatch
        batch_size = x.shape[0]
        if x.shape[1] != self.input_size:
            if x.shape[1] > self.input_size:
                # Truncate
                x = x[:, :self.input_size]
            else:
                # Pad
                padded = torch.zeros(batch_size, self.input_size, device=x.device)
                padded[:, :x.shape[1]] = x
                x = padded
                
        # Forward pass through classifier
        output = self.classifier(x)
        
        # Monitor
        if "output" in self.monitors:
            self.monitors["output"]["activity"].append(output.detach())
        
        return output
    
    def get_raw_output(self, x: torch.Tensor) -> torch.Tensor:
        """Just use the forward method directly."""
        return self.forward(x)
    
    def count_parameters(self):
        """Count the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)