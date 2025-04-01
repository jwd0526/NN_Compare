"""
SNN models for synthetic data experiments.

This module contains SNN model implementations designed for synthetic data experiments,
based on the implementation in snn_synthetic_data_adapter.py.
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

class SyntheticSpikeModel(SyntheticSNN):
    """
    SNN model for synthetic spike pattern classification.
    
    This model is based on the implementation in snn_synthetic_data_adapter.py
    and uses dual exponential filtering for temporal encoding.
    """
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int, 
                 length: int, batch_size: int, tau_m: float = 4.0, tau_s: float = 1.0,
                 dropout_rate: float = 0.3):
        """
        Initialize the synthetic spike model.
        
        Args:
            input_size: Number of input neurons
            hidden_size: Number of neurons in hidden layers
            output_size: Number of output neurons
            length: Number of time steps in the simulation
            batch_size: Batch size for training/inference
            tau_m: Membrane time constant
            tau_s: Synaptic time constant
            dropout_rate: Dropout rate for regularization
        """
        super().__init__(input_size, hidden_size, output_size, length, batch_size, tau_m, tau_s)
        
        # Temporal encoding layer
        self.axon1 = dual_exp_iir_layer((input_size,), self.length, self.batch_size, tau_m, tau_s, True)
        self.snn1 = neuron_layer(input_size, hidden_size, self.length, self.batch_size, tau_m, True, False)
        
        # Hidden layer
        self.axon2 = dual_exp_iir_layer((hidden_size,), self.length, self.batch_size, tau_m, tau_s, True)
        self.snn2 = neuron_layer(hidden_size, hidden_size, self.length, self.batch_size, tau_m, True, False)
        
        # Output layer
        self.axon3 = dual_exp_iir_layer((hidden_size,), self.length, self.batch_size, tau_m, tau_s, True)
        self.snn3 = neuron_layer(hidden_size, output_size, self.length, self.batch_size, tau_m, True, False)
        
        # Regularization
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.dropout2 = nn.Dropout(p=dropout_rate)
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            inputs: Input tensor of shape [batch_size, input_size, length]
            
        Returns:
            Output tensor of shape [batch_size, output_size, length]
        """
        # First layer
        axon1_out, _ = self.axon1(inputs)
        spike_l1, _ = self.snn1(axon1_out)
        spike_l1 = self.dropout1(spike_l1)
        
        # Second layer
        axon2_out, _ = self.axon2(spike_l1)
        spike_l2, _ = self.snn2(axon2_out)
        spike_l2 = self.dropout2(spike_l2)
        
        # Output layer
        axon3_out, _ = self.axon3(spike_l2)
        spike_l3, _ = self.snn3(axon3_out)
        
        # Add activity to monitors if configured
        if "layer1" in self.monitors:
            self.monitors["layer1"]["activity"].append(spike_l1.detach())
        if "layer2" in self.monitors:
            self.monitors["layer2"]["activity"].append(spike_l2.detach())
        if "output" in self.monitors:
            self.monitors["output"]["activity"].append(spike_l3.detach())
        
        return spike_l3
    
    def reset_state(self) -> None:
        """Reset the internal state of all layers."""
        # For each layer, reset the internal state (usually between batches)
        # This would be implementation-specific depending on your SNN layers
        pass

class SpatialSpikeModel(SyntheticSNN):
    """
    SNN model for spatial spike pattern classification.
    
    This model is designed for processing 2D spatial patterns
    (like those generated in synthetic_data_generator.py).
    """
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int, 
                 spatial_shape: Tuple[int, int], length: int, batch_size: int, 
                 tau_m: float = 4.0, tau_s: float = 1.0):
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
        """
        super().__init__(input_size, hidden_size, output_size, length, batch_size, tau_m, tau_s)
        self.spatial_shape = spatial_shape
        height, width = spatial_shape
        
        # Convolutional layers for spatial feature extraction
        self.conv1 = conv2d_layer(height, width, 1, 16, 3, 1, 1, 1, 
                                 self.length, self.batch_size, tau_m, True, False)
        
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
        self.dropout = nn.Dropout(p=0.3)
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network with proper reshaping for convolutional input.
        
        Args:
            inputs: Input tensor of shape [batch_size, height*width, length]
                
        Returns:
            Output tensor of shape [batch_size, output_size, length]
        """
        # Debug input shape
        batch_size = inputs.shape[0]
        height, width = self.spatial_shape
        
        # Step 1: Extract each timestep and process through conv layer
        conv_outputs = []
        
        # Extract each timestep
        for t in range(self.length):
            # Get current timestep data and reshape to [batch, height, width]
            x_t = inputs[:, :, t].view(batch_size, height, width)
            
            # Add channel dimension to make [batch, channels=1, height, width]
            x_t = x_t.unsqueeze(1)
            
            # Process through conv2d layer
            conv_out, _ = self.conv1(x_t)
            
            # Flatten to [batch, features]
            conv_out_flat = conv_out.view(batch_size, -1)
            conv_outputs.append(conv_out_flat)
        
        # Stack along time dimension to get [batch, conv_features, time]
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