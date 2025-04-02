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
    
    This model is optimized specifically for complex spatial hierarchies and global-local
    pattern recognition, showcasing the ANN's ability to process data in parallel and
    integrate information from multiple spatial scales simultaneously.
    """
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int, 
                 spatial_shape: Tuple[int, int], length: int, batch_size: int, 
                 dropout_rate: float = 0.2):
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
        
        # SPECIALIZED ANN ARCHITECTURE FOR COMPLEX SPATIAL HIERARCHIES
        # Designed to excel at the complex spatial hierarchical patterns in our new dataset:
        # - Multi-scale nested patterns
        # - Directional gradients with embedded features
        # - Long-range spatial dependencies
        # - Global structure with fine details
        
        # 1. MULTI-SCALE PARALLEL PATHWAY ARCHITECTURE
        # Multiple parallel convolutional paths with different receptive fields
        # Captures patterns at different spatial scales simultaneously
        
        # Input processing - 7 channels representing different views
        input_channels = 7
        
        # PATHWAY 1: Local details (small receptive field)
        # Good for fine details, textures, and small patterns
        self.local_conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1)
        self.local_bn1 = nn.BatchNorm2d(16)
        self.local_conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.local_bn2 = nn.BatchNorm2d(32)
        
        # PATHWAY 2: Intermediate features (medium receptive field)
        # Good for shapes, edges, and medium-scale patterns
        self.mid_conv1 = nn.Conv2d(input_channels, 16, kernel_size=5, stride=1, padding=2)
        self.mid_bn1 = nn.BatchNorm2d(16)
        self.mid_conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2)
        self.mid_bn2 = nn.BatchNorm2d(32)
        
        # PATHWAY 3: Global context (large receptive field)
        # Good for overall structure, global patterns, symmetry
        self.global_conv1 = nn.Conv2d(input_channels, 16, kernel_size=7, stride=1, padding=3)
        self.global_bn1 = nn.BatchNorm2d(16)
        self.global_conv2 = nn.Conv2d(16, 32, kernel_size=7, stride=1, padding=3)
        self.global_bn2 = nn.BatchNorm2d(32)
        
        # 2. FEATURE FUSION LAYERS
        # Combine information from all pathways
        # 32*3 = 96 input channels (32 from each pathway)
        self.fusion_conv = nn.Conv2d(96, 64, kernel_size=3, stride=1, padding=1)
        self.fusion_bn = nn.BatchNorm2d(64)
        
        # 3. HIGHER-LEVEL FEATURE EXTRACTION
        # Extract compositional features from the fused representations
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # 4. SPATIAL ATTENTION MODULE
        # Focus on the most informative spatial regions
        self.attention_conv = nn.Conv2d(128, 1, kernel_size=1)
        
        # 5. GLOBAL POOLING AND FULLY CONNECTED LAYERS
        # Pool features and classify
        # Output size calculation after two pooling layers (factor 2 each)
        conv_h, conv_w = height // 4, width // 4
        conv_output_size = 128 * conv_h * conv_w
        
        # Dimensionality reduction through pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((conv_h, conv_w))
        
        # Fully connected classifier
        self.fc1 = nn.Linear(conv_output_size, hidden_size)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(hidden_size // 2, output_size)
        
        # Initialize weights with improved Kaiming initialization
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for all convolutional and linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    
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
        
        # 1. TEMPORAL AGGREGATION AND FEATURE EXTRACTION
        # Process multiple timesteps to get diverse views of the input
        # This helps with noise robustness since our dataset has different 
        # noise patterns at each timestep
        
        # Core feature extraction from temporal data
        x_mean = torch.mean(x, dim=2)  # Average over time (stable features)
        x_max, _ = torch.max(x, dim=2)  # Max over time (strongest features)
        # Get features from multiple time points (beginning, middle, end)
        early_t = 20
        mid_t = 50
        late_t = 80
        x_early = x[:, :, early_t]
        x_mid = x[:, :, mid_t]
        x_late = x[:, :, late_t]
        
        # Reshape all features to spatial format
        def reshape_to_spatial(tensor):
            return tensor.reshape(batch_size, height, width)
        
        x_mean_spatial = reshape_to_spatial(x_mean)
        x_max_spatial = reshape_to_spatial(x_max)
        x_early_spatial = reshape_to_spatial(x_early)
        x_mid_spatial = reshape_to_spatial(x_mid)
        x_late_spatial = reshape_to_spatial(x_late)
        
        # 2. SPATIAL GRADIENT EXTRACTION
        # Calculate gradients to highlight edges and transitions
        # Critical for detecting boundaries in complex hierarchical patterns
        
        # Compute gradients from the mid-frame (most stable)
        grad_x = F.pad(torch.abs(x_mid_spatial[:, :, 1:] - x_mid_spatial[:, :, :-1]), 
                       (1, 0, 0, 0), "constant", 0)
        grad_y = F.pad(torch.abs(x_mid_spatial[:, 1:, :] - x_mid_spatial[:, :-1, :]), 
                       (0, 0, 1, 0), "constant", 0)
        
        # 3. MULTI-CHANNEL INPUT CREATION
        # Stack all extracted features as input channels
        x_multi = torch.stack([
            x_mean_spatial,   # Average (stable features)
            x_max_spatial,    # Maximum (strongest activations)
            x_early_spatial,  # Early frame
            x_mid_spatial,    # Middle frame (most representative)
            x_late_spatial,   # Late frame
            grad_x,           # Horizontal edges
            grad_y            # Vertical edges
        ], dim=1)  # [batch, 7, height, width]
        
        # 4. MULTI-SCALE PARALLEL PATHWAY PROCESSING
        # Process input through three parallel pathways with different receptive fields
        
        # Local pathway (small receptive field for fine details)
        local_features = F.relu(self.local_bn1(self.local_conv1(x_multi)))
        local_features = F.relu(self.local_bn2(self.local_conv2(local_features)))
        
        # Mid-level pathway (medium receptive field for shapes and patterns)
        mid_features = F.relu(self.mid_bn1(self.mid_conv1(x_multi)))
        mid_features = F.relu(self.mid_bn2(self.mid_conv2(mid_features)))
        
        # Global pathway (large receptive field for overall structure)
        global_features = F.relu(self.global_bn1(self.global_conv1(x_multi)))
        global_features = F.relu(self.global_bn2(self.global_conv2(global_features)))
        
        # 5. FEATURE FUSION
        # Combine features from all pathways
        fused_features = torch.cat([local_features, mid_features, global_features], dim=1)
        fused_features = F.relu(self.fusion_bn(self.fusion_conv(fused_features)))
        
        # First pooling to reduce spatial dimensions
        pooled_features = F.max_pool2d(fused_features, 2)
        
        # 6. HIGHER-LEVEL FEATURE EXTRACTION
        hierarchical_features = F.relu(self.bn3(self.conv3(pooled_features)))
        
        # Second pooling
        hierarchical_features = F.max_pool2d(hierarchical_features, 2)
        
        # 7. SPATIAL ATTENTION
        # Focus on the most informative regions
        attention_weights = torch.sigmoid(self.attention_conv(hierarchical_features))
        attended_features = hierarchical_features * attention_weights
        
        # 8. CLASSIFICATION
        # Flatten and classify
        features_flat = attended_features.view(batch_size, -1)
        
        x = F.relu(self.fc1(features_flat))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        # 9. Output formatting (add time dimension for compatibility with SNN interface)
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
        # Get the full forward pass and then remove the time dimension
        output_with_time = self.forward(x)
        # Just take a slice from the middle of the time dimension
        mid_time = self.length // 2
        return output_with_time[:, :, mid_time]