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
    
    This model attempts to adapt SNN architecture to handle spatial patterns,
    while working within the constraints of spike-based sequential processing.
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
        
        # ENHANCED SPATIAL FEATURE EXTRACTION FOR SNN
        # Multiple kernel sizes to detect patterns at different scales
        
        # Small receptive field for details
        self.conv1_small = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, 
                                    stride=1, padding=1, bias=True)
        
        # Medium receptive field for shapes
        self.conv1_medium = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=5, 
                                    stride=1, padding=2, bias=True)
        
        # Second layer processes combined features
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3,
                              stride=1, padding=1, bias=True)
        
        # Calculate output size after pooling
        # We'll use pooling to reduce spatial dimensions while preserving features
        pool_size = 2
        conv_h, conv_w = height // pool_size, width // pool_size  
        conv_output_size = 32 * conv_h * conv_w
        
        # Adapting hidden layer size based on input dimensions
        reduced_hidden_size = hidden_size
        
        # SNN-specific processing layers
        # First layer converts conv output to spikes
        self.axon1 = dual_exp_iir_layer((conv_output_size,), self.length, self.batch_size, tau_m, tau_s, True)
        self.snn1 = neuron_layer(conv_output_size, reduced_hidden_size, self.length, self.batch_size, tau_m, True, False)
        
        # Second spiking layer for feature integration
        self.axon2 = dual_exp_iir_layer((reduced_hidden_size,), self.length, self.batch_size, tau_m, tau_s, True)
        self.snn2 = neuron_layer(reduced_hidden_size, hidden_size//2, self.length, self.batch_size, tau_m, True, False)
        
        # Output layer
        self.axon3 = dual_exp_iir_layer((hidden_size//2,), self.length, self.batch_size, tau_m, tau_s, True)
        self.snn3 = neuron_layer(hidden_size//2, output_size, self.length, self.batch_size, tau_m, True, False)
        
        # Use dropout for regularization
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.dropout2 = nn.Dropout(p=dropout_rate)
    
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
        
        # Process each timestep through conv layer - SNNs must process timesteps
        # individually (a key limitation compared to ANNs for spatial tasks)
        conv_outputs = []
        
        # We'll sample more timesteps to try to compensate for SNN's sequential
        # processing limitation, while still keeping efficiency manageable
        # Sample at least 50% of timesteps
        use_timesteps = list(range(10, self.length-10, 2))
        
        for t in use_timesteps:
            # Extract this timestep - shape becomes [batch, height*width]
            x_t = inputs[:, :, t]
            
            # Reshape to [batch, height, width]
            x_t = x_t.reshape(batch_size, height, width)
            
            # Add channel dimension for conv input [batch, channel=1, height, width]
            x_t = x_t.unsqueeze(1)  # [batch, 1, height, width]
            
            # PARALLEL MULTI-SCALE FEATURE EXTRACTION
            # Process through two parallel pathways with different receptive fields
            
            # Small receptive field pathway
            conv1_small_out = self.conv1_small(x_t)
            
            # Medium receptive field pathway  
            conv1_medium_out = self.conv1_medium(x_t)
            
            # Concatenate features from both pathways
            conv1_combined = torch.cat([conv1_small_out, conv1_medium_out], dim=1)
            
            # Process through second conv layer for hierarchical features
            conv2_out = self.conv2(conv1_combined)
            
            # Apply pooling to reduce spatial dimensions
            pooled_out = F.avg_pool2d(conv2_out, 2)
            
            # Reshape to [batch, conv_output_size]
            conv_out_flat = pooled_out.view(batch_size, -1)
            conv_outputs.append(conv_out_flat)
        
        # Stack along time dimension to get [batch, conv_output_size, reduced_time]
        conv_out_sequence = torch.stack(conv_outputs, dim=2)
        
        # Create a full-time tensor with zeros for unused timesteps
        full_conv_out = torch.zeros((batch_size, conv_out_sequence.shape[1], self.length), 
                                   device=conv_out_sequence.device)
        
        # Place the processed timesteps into the full tensor
        for i, t in enumerate(use_timesteps):
            if i < conv_out_sequence.shape[2]:
                full_conv_out[:, :, t] = conv_out_sequence[:, :, i]
        
        # SNN PROCESSING PATH
        # Convert convolutional features to spikes and process through SNN layers
        
        # First SNN layer
        axon1_out, _ = self.axon1(full_conv_out)
        spike_l1, _ = self.snn1(axon1_out)
        spike_l1 = self.dropout1(spike_l1)
        
        # Second SNN layer for feature integration
        axon2_out, _ = self.axon2(spike_l1)
        spike_l2, _ = self.snn2(axon2_out)
        spike_l2 = self.dropout2(spike_l2)
        
        # Output layer
        axon3_out, _ = self.axon3(spike_l2)
        spike_l3, _ = self.snn3(axon3_out)
        
        # Add activity to monitors if configured
        if "conv" in self.monitors:
            self.monitors["conv"]["activity"].append(full_conv_out.detach())
        if "hidden1" in self.monitors:
            self.monitors["hidden1"]["activity"].append(spike_l1.detach())
        if "hidden2" in self.monitors:
            self.monitors["hidden2"]["activity"].append(spike_l2.detach())
        if "output" in self.monitors:
            self.monitors["output"]["activity"].append(spike_l3.detach())
        
        return spike_l3
    
    def reset_state(self) -> None:
        """Reset the internal state of all spiking neuron layers."""
        # Axons
        if hasattr(self.axon1, 'reset_state'):
            self.axon1.reset_state()
        if hasattr(self.axon2, 'reset_state'):
            self.axon2.reset_state()
        if hasattr(self.axon3, 'reset_state'):
            self.axon3.reset_state()
            
        # Neurons
        if hasattr(self.snn1, 'reset_state'):
            self.snn1.reset_state()
        if hasattr(self.snn2, 'reset_state'):
            self.snn2.reset_state()
        if hasattr(self.snn3, 'reset_state'):
            self.snn3.reset_state()