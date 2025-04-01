"""
ANN model for spatial pattern classification.

This module contains an Artificial Neural Network (ANN) implementation
designed for spatial pattern classification, using convolutional layers
to process 2D spatial inputs similar to the SNN spatial models.
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

class SpatialANN(SyntheticANN):
    """
    ANN model for spatial pattern classification.
    
    This model is designed to be compared with SNN models on the same 2D spatial datasets.
    It processes 2D spatial patterns using convolutional layers before feeding
    data through fully connected layers.
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
            length: Number of time steps in the simulation (for compatibility with SNN)
            batch_size: Batch size for training/inference
            dropout_rate: Dropout rate for regularization
        """
        super().__init__(input_size, hidden_size, output_size, length, batch_size)
        self.spatial_shape = spatial_shape
        self.dropout_rate = dropout_rate
        height, width = spatial_shape
        
        # Convolutional layers for spatial feature extraction
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, 
                              stride=1, padding=1)
        
        # Calculate output size of conv layer
        conv_h, conv_w = height, width  # Padding=1 keeps dimensions same
        conv_output_size = 16 * conv_h * conv_w
        
        # Fully connected layers
        self.fc1 = nn.Linear(conv_output_size, hidden_size)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, output_size)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape [batch_size, input_size, length]
                Contains spatial patterns for SNN compatibility
            
        Returns:
            Output tensor of shape [batch_size, output_size, length]
            Time dimension is repeated for compatibility with SNN
        """
        batch_size = x.shape[0]
        height, width = self.spatial_shape
        
        # Process input: aggregate over time dimension
        x_aggregate = torch.sum(x, dim=2)  # shape: [batch_size, height*width]
        
        # Reshape for convolutional input - use reshape for safety
        x_reshaped = x_aggregate.reshape(batch_size, 1, height, width)
        
        # Pass through convolutional layer
        x = F.relu(self.conv1(x_reshaped))
        
        # Flatten for fully connected layers
        x = x.view(batch_size, -1)
        
        # Pass through fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        
        # For compatibility with SNN comparison, expand to add time dimension
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
        batch_size = x.shape[0]
        height, width = self.spatial_shape
        
        # Process input: aggregate over time dimension
        x_aggregate = torch.sum(x, dim=2)  # shape: [batch_size, height*width]
        
        # Reshape for convolutional input - use reshape for safety
        x_reshaped = x_aggregate.reshape(batch_size, 1, height, width)
        
        # Pass through convolutional layer
        x = F.relu(self.conv1(x_reshaped))
        
        # Flatten for fully connected layers
        x = x.view(batch_size, -1)
        
        # Pass through fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        
        return x