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
        
        # Enhanced architecture with more convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, 
                              stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3,
                              stride=1, padding=1)
        
        # Calculate output size of conv layers
        conv_h, conv_w = height, width  # Padding=1 keeps dimensions same
        conv_output_size = 32 * conv_h * conv_w
        
        # Fully connected layers with more capacity
        self.fc1 = nn.Linear(conv_output_size, hidden_size*2)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size*2, hidden_size)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
    
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
        
        # OPTIMIZED FOR SPATIAL PROCESSING:
        # Instead of processing each timestep individually (which is inefficient for ANNs),
        # we'll use a more parallel approach that leverages CNN strengths.
        
        # For static or semi-static patterns, we can leverage the fact that for this dataset,
        # the spatial pattern is largely consistent across time. This is what ANNs excel at.
        
        # 1. First, sum or average across time to get a single spatial pattern per sample
        # This collapses the time dimension and creates a strong spatial representation
        x_spatial = torch.mean(x, dim=2)  # [batch, height*width]
        
        # 2. Reshape to proper spatial dimensions
        x_spatial = x_spatial.reshape(batch_size, height, width)
        
        # 3. Add channel dimension for CNN processing
        x_spatial = x_spatial.unsqueeze(1)  # [batch, 1, height, width]
        
        # 4. Use the CNN layers to extract spatial features
        conv1_out = F.relu(self.conv1(x_spatial))
        conv2_out = F.relu(self.conv2(conv1_out))
        
        # 5. Flatten for fully connected layers
        x_flat = conv2_out.view(batch_size, -1)
        
        # 6. Pass through fully connected layers
        x = F.relu(self.fc1(x_flat))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        # 7. For compatibility with SNN comparison, expand to add time dimension
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
        
        # Use the same optimized approach as in forward method
        # Collapse time dimension first - this is what ANNs excel at
        x_spatial = torch.mean(x, dim=2)
        x_spatial = x_spatial.reshape(batch_size, height, width)
        x_spatial = x_spatial.unsqueeze(1)
        
        # Extract spatial features with convolutional layers
        conv1_out = F.relu(self.conv1(x_spatial))
        conv2_out = F.relu(self.conv2(conv1_out))
        
        # Fully connected layers
        x_flat = conv2_out.view(batch_size, -1)
        x = F.relu(self.fc1(x_flat))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x