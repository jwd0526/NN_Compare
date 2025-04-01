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
from models.base_model import BaseSNN

class SyntheticANN(BaseSNN):
    """
    ANN model for synthetic spike pattern classification.
    
    This model is designed to be compared with SNN models on the same datasets.
    It processes spike pattern data by aggregating spikes over time before
    feeding them through a standard feedforward neural network.
    """
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int, 
                 length: int, batch_size: int, dropout_rate: float = 0.3):
        """
        Initialize the synthetic ANN model.
        
        Args:
            input_size: Number of input neurons
            hidden_size: Number of neurons in hidden layers
            output_size: Number of output neurons
            length: Number of time steps in the simulation (for compatibility with SNN)
            batch_size: Batch size for training/inference
            dropout_rate: Dropout rate for regularization
        """
        super().__init__(input_size, output_size, length, batch_size)
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        
        # Define the network layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape [batch_size, input_size, length]
                Contains spike patterns for SNN compatibility
            
        Returns:
            Output tensor of shape [batch_size, output_size]
        """
        # Process input spikes: We'll use spike count aggregation
        # Sum over time dimension to get total number of spikes per neuron
        x_aggregate = torch.sum(x, dim=2)  # shape: [batch_size, input_size]
        
        # Alternative input processing methods (uncommented by default):
        # 1. Mean spike rate over time
        # x_aggregate = torch.mean(x, dim=2)
        
        # 2. Max pooling over time
        # x_aggregate, _ = torch.max(x, dim=2)
        
        # Pass through feedforward network
        x = F.relu(self.fc1(x_aggregate))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        # For compatibility with SNN comparison, we'll expand to add time dimension 
        # with the same prediction at each time step
        x_expanded = x.unsqueeze(-1).repeat(1, 1, self.length)
        
        # Add activity to monitors if configured
        if "output" in self.monitors:
            self.monitors["output"]["activity"].append(x_expanded.detach())
        
        return x_expanded
    
    def get_raw_output(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get the raw output without the time dimension expansion.
        
        Args:
            x: Input tensor of shape [batch_size, input_size, length]
            
        Returns:
            Output tensor of shape [batch_size, output_size]
        """
        x_aggregate = torch.sum(x, dim=2)
        
        x = F.relu(self.fc1(x_aggregate))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x

class SpatialANN(SyntheticANN):
    """
    ANN model for spatial spike pattern classification.
    
    This model is designed for processing 2D spatial patterns,
    comparable to the SpatialSpikeModel in the SNN framework.
    """
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int, 
                 spatial_shape: Tuple[int, int], length: int, batch_size: int,
                 dropout_rate: float = 0.3):
        """
        Initialize the spatial ANN model.
        
        Args:
            input_size: Number of input neurons
            hidden_size: Number of neurons in hidden layers
            output_size: Number of output neurons
            spatial_shape: Spatial dimensions (height, width) of the input
            length: Number of time steps in the simulation
            batch_size: Batch size for training/inference
            dropout_rate: Dropout rate for regularization
        """
        super().__init__(input_size, hidden_size, output_size, length, batch_size, dropout_rate)
        self.spatial_shape = spatial_shape
        height, width = spatial_shape
        
        # Replace the first fully connected layer with a convolutional layer
        # for better spatial feature extraction
        self.conv = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        
        # Calculate the size after convolution
        conv_output_size = 16 * height * width
        
        # Redefine the first linear layer to match conv output
        self.fc1 = nn.Linear(conv_output_size, hidden_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape [batch_size, height*width, length]
            
        Returns:
            Output tensor of shape [batch_size, output_size, length]
        """
        batch_size = x.shape[0]
        height, width = self.spatial_shape
        
        # Sum over time dimension to get aggregated spatial pattern
        x_aggregate = torch.sum(x, dim=2)  # [batch_size, height*width]
        
        # Reshape to 2D spatial data with channels
        x_spatial = x_aggregate.view(batch_size, 1, height, width)
        
        # Apply convolution
        x_conv = F.relu(self.conv(x_spatial))  # [batch_size, 16, height, width]
        
        # Flatten for fully connected layers
        x_flat = x_conv.view(batch_size, -1)
        
        # Pass through fully connected network
        x = F.relu(self.fc1(x_flat))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        # Expand to add time dimension for compatibility
        x_expanded = x.unsqueeze(-1).repeat(1, 1, self.length)
        
        # Add activity to monitors if configured
        if "output" in self.monitors:
            self.monitors["output"]["activity"].append(x_expanded.detach())
        
        return x_expanded