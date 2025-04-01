"""
SNN models for spatial pattern classification.

This module contains SNN model implementations designed for spatial pattern classification
data, supporting 2D convolution-based processing of spatial spike patterns.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, List, Optional, Union, Any

import sys
import os
# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from snn_lib.snn_layers import *
from models.base_model import SyntheticSNN

class SpatialSpikeModel(SyntheticSNN):
    """
    SNN model for spatial spike pattern classification.
    
    This model is designed for processing 2D spatial patterns
    (like those generated in synthetic_data_generator.py).
    """
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int, 
                 spatial_shape: Tuple[int, int], length: int, batch_size: int, 
                 tau_m: float = 4.0, tau_s: float = 1.0, dropout_rate: float = 0.3):
        """
        Initialize the spatial spike model.
        
        Args:
            input_size: Number of input neurons
            hidden_size: Number of neurons in hidden layers
            output_size: Number of output neurons
            spatial_shape: Spatial dimensions (height, width) of the input
            length: Number of time steps in the simulation
            batch_size: Batch size for training/inference
            tau_m: Membrane time constant
            tau_s: Synaptic time constant
            dropout_rate: Dropout rate for regularization
        """
        super().__init__(input_size, hidden_size, output_size, length, batch_size, tau_m, tau_s)
        self.spatial_shape = spatial_shape
        height, width = spatial_shape
        
        # Simpler approach for spatial SNN: use a standard ConvTranspose layer instead of conv2d_layer
        # This helps avoid shape mismatch issues in the original implementation
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, 
                               stride=1, padding=1, bias=True)
        
        # Calculate output size of conv layer
        conv_h, conv_w = height, width  # Padding=1 keeps dimensions same
        conv_output_size = 16 * conv_h * conv_w
        
        # Flattened processing
        self.axon1 = dual_exp_iir_layer((conv_output_size,), self.length, self.batch_size, tau_m, tau_s, True)
        self.snn1 = neuron_layer(conv_output_size, hidden_size, self.length, self.batch_size, tau_m, True, False)
        
        # Output layer
        self.axon2 = dual_exp_iir_layer((hidden_size,), self.length, self.batch_size, tau_m, tau_s, True)
        self.snn2 = neuron_layer(hidden_size, output_size, self.length, self.batch_size, tau_m, True, False)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(p=dropout_rate)
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network with proper reshaping for convolutional input.
        
        Args:
            inputs: Input tensor of shape [batch_size, height*width, length]
                
        Returns:
            Output tensor of shape [batch_size, output_size, length]
        """
        # Reshape input to [batch, channels=1, height, width, time]
        batch_size = inputs.shape[0]
        height, width = self.spatial_shape
        
        # Process each timestep through conv layer
        conv_outputs = []
        for t in range(self.length):
            # Extract this timestep - shape becomes [batch, height*width]
            x_t = inputs[:, :, t]
            
            # Reshape to [batch, height, width] - ensure proper reshaping
            x_t = x_t.reshape(batch_size, height, width)
            
            # Add channel dimension for conv input [batch, channel=1, height, width]
            x_t = x_t.unsqueeze(1)  # [batch, 1, height, width]
            
            # Pass through conv layer - Conv2D expects [batch, channel, H, W]
            conv_out = self.conv1(x_t)  # Regular Conv2d returns tensor directly
            
            # Reshape to [batch, conv_output_size]
            conv_out_flat = conv_out.view(batch_size, -1)
            conv_outputs.append(conv_out_flat)
        
        # Stack along time dimension to get [batch, conv_output_size, time]
        conv_out_sequence = torch.stack(conv_outputs, dim=2)
        
        # Continue through the network
        axon1_out, _ = self.axon1(conv_out_sequence)
        spike_l1, _ = self.snn1(axon1_out)
        spike_l1 = self.dropout(spike_l1)
        
        axon2_out, _ = self.axon2(spike_l1)
        spike_l2, _ = self.snn2(axon2_out)
        
        # Add activity to monitors if configured
        if "conv" in self.monitors:
            self.monitors["conv"]["activity"].append(conv_out_sequence.detach())
        if "hidden" in self.monitors:
            self.monitors["hidden"]["activity"].append(spike_l1.detach())
        if "output" in self.monitors:
            self.monitors["output"]["activity"].append(spike_l2.detach())
        
        return spike_l2
    
    def reset_state(self) -> None:
        """Reset the internal state of all layers."""
        # Implementation would depend on the underlying SNN layers
        pass