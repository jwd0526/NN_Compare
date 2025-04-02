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
    Enhanced SNN model for synthetic spike pattern classification.
    
    This model is optimized for temporal pattern recognition with improved
    architecture and temporal dynamics handling.
    """
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int, 
                 length: int, batch_size: int, tau_m: float = 4.0, tau_s: float = 1.0,
                 dropout_rate: float = 0.3):
        """
        Initialize the enhanced synthetic spike model.
        
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
        
        # Adjust time constants to improve temporal sensitivity
        tau_m_input = tau_m * 0.8  # Faster membrane dynamics for input processing
        tau_s_input = tau_s * 0.7  # Faster synaptic dynamics for input processing
        
        tau_m_hidden = tau_m * 1.2  # Slower membrane dynamics for hidden processing
        tau_s_hidden = tau_s * 1.0  # Standard synaptic dynamics for hidden processing
        
        # Use different time constants for different layers to capture multi-scale temporal features
        
        # Temporal encoding layer with faster dynamics for better feature extraction
        self.axon1 = dual_exp_iir_layer((input_size,), self.length, self.batch_size, tau_m_input, tau_s_input, True)
        self.snn1 = neuron_layer(input_size, hidden_size, self.length, self.batch_size, tau_m_input, True, True)  # Enable training tau
        
        # Intermediate layer with larger size for better representation
        hidden_size_larger = int(hidden_size * 1.5)
        self.axon2 = dual_exp_iir_layer((hidden_size,), self.length, self.batch_size, tau_m_hidden, tau_s_hidden, True)
        self.snn2 = neuron_layer(hidden_size, hidden_size_larger, self.length, self.batch_size, tau_m_hidden, True, True)
        
        # Additional hidden layer for deeper processing
        self.axon2b = dual_exp_iir_layer((hidden_size_larger,), self.length, self.batch_size, tau_m_hidden, tau_s_hidden, True)
        self.snn2b = neuron_layer(hidden_size_larger, hidden_size, self.length, self.batch_size, tau_m_hidden, True, True)
        
        # Output layer with standard dynamics for clean classification
        self.axon3 = dual_exp_iir_layer((hidden_size,), self.length, self.batch_size, tau_m, tau_s, True)
        self.snn3 = neuron_layer(hidden_size, output_size, self.length, self.batch_size, tau_m, True, False)
        
        # Regularization with adaptive dropout rates
        self.dropout1 = nn.Dropout(p=dropout_rate * 0.7)  # Less dropout early to preserve information
        self.dropout2 = nn.Dropout(p=dropout_rate)
        self.dropout3 = nn.Dropout(p=dropout_rate * 1.2)  # More dropout later to prevent overfitting
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the enhanced network with deeper architecture.
        
        Args:
            inputs: Input tensor of shape [batch_size, input_size, length]
            
        Returns:
            Output tensor of shape [batch_size, output_size, length]
        """
        # First layer with faster dynamics
        axon1_out, _ = self.axon1(inputs)
        spike_l1, _ = self.snn1(axon1_out)
        spike_l1 = self.dropout1(spike_l1)
        
        # First hidden layer with intermediate size
        axon2_out, _ = self.axon2(spike_l1)
        spike_l2, _ = self.snn2(axon2_out)
        spike_l2 = self.dropout2(spike_l2)
        
        # Second hidden layer for deeper processing
        axon2b_out, _ = self.axon2b(spike_l2)
        spike_l2b, _ = self.snn2b(axon2b_out)
        spike_l2b = self.dropout3(spike_l2b)
        
        # Residual connection to help with vanishing gradients
        # Add a skip connection from first to second hidden layer where dimensions match
        if spike_l1.shape[1] == spike_l2b.shape[1]:
            spike_l2b = spike_l2b + 0.2 * spike_l1  # Apply a scaling factor to the residual connection
        
        # Output layer with cleaner dynamics for classification
        axon3_out, _ = self.axon3(spike_l2b)
        spike_l3, _ = self.snn3(axon3_out)
        
        # Add activity to monitors if configured
        if "layer1" in self.monitors:
            self.monitors["layer1"]["activity"].append(spike_l1.detach())
        if "layer2" in self.monitors:
            self.monitors["layer2"]["activity"].append(spike_l2.detach())
        if "layer2b" in self.monitors:
            self.monitors["layer2b"]["activity"].append(spike_l2b.detach())
        if "output" in self.monitors:
            self.monitors["output"]["activity"].append(spike_l3.detach())
        
        return spike_l3
    
    def reset_state(self) -> None:
        """Reset the internal state of all layers."""
        # This would be implementation-specific depending on the underlying SNN layers
        pass