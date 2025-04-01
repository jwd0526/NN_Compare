#!/usr/bin/env python3
"""
Script to generate synthetic data with multiple difficulty tiers for SNN experiments.

This enhanced version creates datasets with increasing difficulty levels for both
spatial and temporal pattern discrimination tasks.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import json
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Create a self-contained generator class so we don't rely on imports
class SyntheticDataGenerator:
    """Generate synthetic spike patterns for SNN testing and evaluation"""
    
    def __init__(self, output_dir='./data'):
        """
        Initialize the spike generator
        
        Args:
            output_dir: Directory to save generated data
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_regular_pattern(self, n_neurons=10, length=100, period=10, jitter=0, n_samples=100, phase_shift=0):
        """
        Generate regularly spaced spikes with optional jitter and phase shift
        
        Args:
            n_neurons: Number of neurons
            length: Sequence length in time steps
            period: Regular spike interval
            jitter: Random jitter to add to spike timing (std dev in time steps)
            n_samples: Number of samples to generate
            phase_shift: Phase shift for the regular pattern (in time steps)
            
        Returns:
            patterns: Array of shape (n_samples, n_neurons, length)
            labels: Array of shape (n_samples,) containing class labels (all 0)
        """
        patterns = np.zeros((n_samples, n_neurons, length), dtype=np.float32)
        
        for i in range(n_samples):
            for j in range(n_neurons):
                # Create base spike times with phase shift
                spike_times = np.arange(phase_shift, length, period)
                
                # Add jitter
                if jitter > 0:
                    spike_times = spike_times + np.random.normal(0, jitter, size=len(spike_times))
                    spike_times = np.clip(spike_times, 0, length-1).astype(int)
                
                # Set spikes at the specified times
                valid_times = spike_times[(spike_times >= 0) & (spike_times < length)]
                patterns[i, j, valid_times] = 1.0
                
        # All samples in this set are class 0
        labels = np.zeros(n_samples, dtype=np.int32)
        
        return patterns, labels
    
    def generate_burst_pattern(self, n_neurons=10, length=100, burst_start=20, 
                               burst_width=10, n_spikes=5, jitter=1, n_samples=100):
        """
        Generate spike bursts with a specified timing
        
        Args:
            n_neurons: Number of neurons
            length: Sequence length in time steps
            burst_start: Time step where burst begins
            burst_width: Width of the burst window
            n_spikes: Number of spikes in the burst
            jitter: Jitter in spike timing
            n_samples: Number of samples to generate
            
        Returns:
            patterns: Array of shape (n_samples, n_neurons, length)
            labels: Array of shape (n_samples,) containing class labels (all 1)
        """
        patterns = np.zeros((n_samples, n_neurons, length), dtype=np.float32)
        
        for i in range(n_samples):
            for j in range(n_neurons):
                # Generate random spike times within the burst window
                spike_times = burst_start + np.random.randint(0, burst_width, size=n_spikes)
                spike_times = np.clip(spike_times, 0, length-1)
                
                # Add jitter
                if jitter > 0:
                    spike_times = spike_times + np.random.normal(0, jitter, size=len(spike_times))
                    spike_times = np.clip(spike_times, 0, length-1).astype(int)
                
                patterns[i, j, spike_times] = 1.0
        
        # All samples in this set are class 1
        labels = np.ones(n_samples, dtype=np.int32)
        
        return patterns, labels
    
    def generate_sync_pattern(self, n_neurons=10, length=100, sync_times=[20, 50, 80], 
                             jitter=1, n_samples=100):
        """
        Generate patterns where neurons fire synchronously at specific times
        
        Args:
            n_neurons: Number of neurons
            length: Sequence length in time steps
            sync_times: List of time points where synchronous firing occurs
            jitter: Jitter in spike timing
            n_samples: Number of samples to generate
            
        Returns:
            patterns: Array of shape (n_samples, n_neurons, length)
            labels: Array of shape (n_samples,) containing class labels (all 2)
        """
        patterns = np.zeros((n_samples, n_neurons, length), dtype=np.float32)
        
        for i in range(n_samples):
            for j in range(n_neurons):
                # For each synchronization time
                for sync_t in sync_times:
                    # Add jitter
                    if jitter > 0:
                        actual_t = int(sync_t + np.random.normal(0, jitter))
                        actual_t = max(0, min(length-1, actual_t))
                    else:
                        actual_t = sync_t
                    
                    patterns[i, j, actual_t] = 1.0
        
        # All samples in this set are class 2
        labels = np.full(n_samples, 2, dtype=np.int32)
        
        return patterns, labels

    def generate_sequential_pattern(self, n_neurons=10, length=100, start_time=20, 
                                  interval=2, jitter=0.5, n_samples=100):
        """
        Generate patterns where neurons fire in sequence with fixed intervals
        
        Args:
            n_neurons: Number of neurons
            length: Sequence length in time steps
            start_time: Time step when the first neuron fires
            interval: Time interval between consecutive neurons
            jitter: Jitter in spike timing
            n_samples: Number of samples to generate
            
        Returns:
            patterns: Array of shape (n_samples, n_neurons, length)
            labels: Array of shape (n_samples,) containing class labels (all 3)
        """
        patterns = np.zeros((n_samples, n_neurons, length), dtype=np.float32)
        
        for i in range(n_samples):
            for j in range(n_neurons):
                # Calculate spike time for this neuron
                spike_time = start_time + j * interval
                
                # Add jitter
                if jitter > 0:
                    spike_time = int(spike_time + np.random.normal(0, jitter))
                    spike_time = max(0, min(length-1, spike_time))
                
                if spike_time < length:
                    patterns[i, j, spike_time] = 1.0
        
        # All samples in this set are class 3
        labels = np.full(n_samples, 3, dtype=np.int32)
        
        return patterns, labels
    
    def generate_spatial_patterns(self, width=10, height=10, length=100, n_patterns=4, n_samples=400):
        """
        Generate 2D spatial-temporal patterns (circles, lines, spirals, etc.)
        
        Args:
            width: Width of the 2D grid
            height: Height of the 2D grid
            length: Sequence length in time steps
            n_patterns: Number of different pattern types to generate
            n_samples: Total number of samples to generate
            
        Returns:
            patterns: Array of shape (n_samples, width*height, length)
            labels: Array of shape (n_samples,) containing class labels
        """
        samples_per_pattern = n_samples // n_patterns
        patterns = np.zeros((n_samples, width*height, length), dtype=np.float32)
        labels = np.zeros(n_samples, dtype=np.int32)
        
        sample_idx = 0
        
        # Pattern 1: Horizontal line sweeping from top to bottom
        for i in range(samples_per_pattern):
            pattern = np.zeros((height, width, length))
            
            # Time at which the line starts sweeping
            start_time = np.random.randint(10, 30)
            
            # Speed of the sweep (in time steps per row)
            speed = np.random.randint(2, 5)
            
            # Create the sweeping line
            for y in range(height):
                t = start_time + y * speed
                if t < length:
                    pattern[y, :, t] = 1.0
            
            # Reshape to required format
            patterns[sample_idx] = pattern.reshape(height*width, length)
            labels[sample_idx] = 0
            sample_idx += 1
        
        # Pattern 2: Vertical line sweeping from left to right
        for i in range(samples_per_pattern):
            pattern = np.zeros((height, width, length))
            
            # Time at which the line starts sweeping
            start_time = np.random.randint(10, 30)
            
            # Speed of the sweep (in time steps per column)
            speed = np.random.randint(2, 5)
            
            # Create the sweeping line
            for x in range(width):
                t = start_time + x * speed
                if t < length:
                    pattern[:, x, t] = 1.0
            
            # Reshape to required format
            patterns[sample_idx] = pattern.reshape(height*width, length)
            labels[sample_idx] = 1
            sample_idx += 1
        
        # Pattern 3: Expanding circle
        for i in range(samples_per_pattern):
            pattern = np.zeros((height, width, length))
            
            # Center coordinates
            center_y, center_x = height // 2, width // 2
            
            # Time at which the circle starts expanding
            start_time = np.random.randint(10, 30)
            
            # Speed of expansion (in time steps per unit radius)
            speed = np.random.randint(3, 6)
            
            # Maximum radius
            max_radius = min(height, width) // 2
            
            # Create expanding circle
            for r in range(1, max_radius + 1):
                t = start_time + r * speed
                if t < length:
                    # Draw a rough circle
                    for y in range(height):
                        for x in range(width):
                            dist = np.sqrt((y - center_y)**2 + (x - center_x)**2)
                            if abs(dist - r) < 0.5:  # Close enough to this radius
                                pattern[y, x, t] = 1.0
            
            # Reshape to required format
            patterns[sample_idx] = pattern.reshape(height*width, length)
            labels[sample_idx] = 2
            sample_idx += 1
        
        # Pattern 4: Diagonal movement
        for i in range(samples_per_pattern):
            pattern = np.zeros((height, width, length))
            
            # Starting corner (randomly choose one of four corners)
            corner = np.random.randint(0, 4)
            if corner == 0:  # Top-left to bottom-right
                start_y, start_x = 0, 0
                dir_y, dir_x = 1, 1
            elif corner == 1:  # Top-right to bottom-left
                start_y, start_x = 0, width - 1
                dir_y, dir_x = 1, -1
            elif corner == 2:  # Bottom-left to top-right
                start_y, start_x = height - 1, 0
                dir_y, dir_x = -1, 1
            else:  # Bottom-right to top-left
                start_y, start_x = height - 1, width - 1
                dir_y, dir_x = -1, -1
            
            # Time at which the movement starts
            start_time = np.random.randint(10, 30)
            
            # Speed of movement (in time steps per diagonal step)
            speed = np.random.randint(2, 4)
            
            # Maximum number of steps
            max_steps = min(height, width)
            
            # Create diagonal movement
            for step in range(max_steps):
                t = start_time + step * speed
                if t < length:
                    y = start_y + step * dir_y
                    x = start_x + step * dir_x
                    
                    # Check if coordinates are valid
                    if 0 <= y < height and 0 <= x < width:
                        pattern[y, x, t] = 1.0
            
            # Reshape to required format
            patterns[sample_idx] = pattern.reshape(height*width, length)
            labels[sample_idx] = 3
            sample_idx += 1
        
        return patterns[:sample_idx], labels[:sample_idx]
    
    def save_dataset(self, patterns, labels, filename):
        """
        Save patterns and labels to .npz file
        
        Args:
            patterns: Array of spike patterns
            labels: Array of class labels
            filename: Base filename to save to
        """
        save_path = os.path.join(self.output_dir, filename + '.npz')
        np.savez(save_path, patterns=patterns, labels=labels)
        print(f"Saved dataset to {save_path}")
    
    def visualize_patterns(self, patterns, labels, n_samples=5, filename=None):
        """
        Visualize a few examples of each pattern class
        
        Args:
            patterns: Array of spike patterns
            labels: Array of class labels
            n_samples: Number of samples per class to visualize
            filename: If provided, save the figure to this file
        """
        unique_labels = np.unique(labels)
        n_classes = len(unique_labels)
        
        fig, axes = plt.subplots(n_classes, n_samples, figsize=(n_samples*3, n_classes*3))
        
        for i, label in enumerate(unique_labels):
            # Get indices for this class
            indices = np.where(labels == label)[0]
            
            # Select a few samples
            selected_indices = indices[:n_samples]
            
            for j, idx in enumerate(selected_indices):
                if n_classes == 1 and n_samples == 1:
                    ax = axes
                elif n_classes == 1:
                    ax = axes[j]
                elif n_samples == 1:
                    ax = axes[i]
                else:
                    ax = axes[i, j]
                
                # Display the spike raster plot
                spike_data = patterns[idx]
                
                if spike_data.ndim == 2:  # Standard (neuron, time) format
                    neurons, times = np.where(spike_data > 0)
                    ax.scatter(times, neurons, s=5, marker='|', c='black')
                    ax.set_ylim(-0.5, spike_data.shape[0] - 0.5)
                    ax.invert_yaxis()
                else:
                    # Handle 3D data - flatten the spatial dimensions
                    ax.imshow(spike_data.sum(axis=2), cmap='viridis')
                
                ax.set_title(f"Class {label}")
                
                # Remove unnecessary ticks
                if j == 0:
                    ax.set_ylabel("Neuron")
                else:
                    ax.set_yticks([])
                
                if i == n_classes - 1:
                    ax.set_xlabel("Time")
                else:
                    ax.set_xticks([])
        
        plt.tight_layout()
        
        if filename:
            save_path = os.path.join(self.output_dir, filename)
            plt.savefig(save_path, dpi=150)
            print(f"Saved visualization to {save_path}")
        
        # Close the figure instead of showing it
        plt.close(fig)
def generate_tiered_temporal_patterns():
    """
    Generate temporal pattern datasets with clear properties
    - Each tier has a specific characteristic that FAVORS SNN over ANN
    - Tiers have progressive but predictable difficulty
    - Structured specifically to demonstrate SNN's temporal processing advantages
    """
    # Create the output directory
    os.makedirs('./data', exist_ok=True)
    
    # Initialize the generator
    generator = SyntheticDataGenerator(output_dir='./data')
    
    # ------------------- TIER 1: PRECISE TIMING (SHOULD FAVOR SNN) -------------------
    print("Generating Tier 1 (Precise Timing) temporal patterns dataset...")
    
    # Patterns with precise timing information that should favor SNN's temporal dynamics
    t1_classes = []
    
    # Class 0: Precise time-coded patterns
    for i in range(150):
        pattern = np.zeros((20, 100), dtype=np.float32)
        
        # Create a precise temporal pattern where timing matters
        for j in range(20):
            # Each neuron fires at a specific time with minimal jitter
            base_time = 10 + j * 3  # Precise timing between neurons
            
            # Very small jitter to make it realistic but still precise
            actual_time = int(base_time + np.random.normal(0, 0.3))
            if 0 <= actual_time < 100:
                pattern[j, actual_time] = 1.0
                
        t1_classes.append((pattern.reshape(1, 20, 100), np.zeros(1, dtype=np.int32)))
    
    # Class 1: Precisely timed bursts
    for i in range(150):
        pattern = np.zeros((20, 100), dtype=np.float32)
        
        # Create bursts at specific times
        burst_times = [20, 50, 80]
        for burst_time in burst_times:
            # Each burst has minimal jitter
            for j in range(20):
                # Only half the neurons fire in each burst
                if j % 2 == i % 2:  
                    spike_time = burst_time + np.random.normal(0, 0.5)
                    spike_time = int(max(0, min(99, spike_time)))
                    pattern[j, spike_time] = 1.0
        
        t1_classes.append((pattern.reshape(1, 20, 100), np.ones(1, dtype=np.int32)))
    
    # Class 2: Precise intervals (frequency coding)
    for i in range(150):
        pattern = np.zeros((20, 100), dtype=np.float32)
        
        # Different neurons fire at different frequencies
        for j in range(20):
            # Frequency depends on neuron index and slightly varies between samples
            base_freq = 4 + (j % 5) + (i % 3)  # Varied frequencies 
            
            # Generate pattern with precise intervals
            for t in range(10, 90, base_freq):
                actual_t = int(t + np.random.normal(0, 0.3))  # Very precise timing
                if 0 <= actual_t < 100:
                    pattern[j, actual_t] = 1.0
        
        t1_classes.append((pattern.reshape(1, 20, 100), np.full(1, 2, dtype=np.int32)))
    
    # Class 3: Phase-offset patterns
    for i in range(150):
        pattern = np.zeros((20, 100), dtype=np.float32)
        
        # Base frequency for all neurons
        base_freq = 10
        
        # Each neuron has specific phase offset that encodes information
        for j in range(20):
            # Phase shift creates a temporal code
            phase_shift = (j % 5) * 2  # 0, 2, 4, 6, 8
            
            # Generate pattern with phase encoding
            for t in range(phase_shift, 90, base_freq):
                actual_t = int(t + np.random.normal(0, 0.3))  # Very precise timing
                if 0 <= actual_t < 100:
                    pattern[j, actual_t] = 1.0
        
        t1_classes.append((pattern.reshape(1, 20, 100), np.full(1, 3, dtype=np.int32)))
    
    # Combine patterns
    tier1_patterns = np.zeros((600, 20, 100), dtype=np.float32)
    tier1_labels = np.zeros(600, dtype=np.int32)
    
    idx = 0
    for pattern_list, label in t1_classes:
        for i in range(len(pattern_list)):
            tier1_patterns[idx] = pattern_list[i]
            tier1_labels[idx] = label[0]
            idx += 1
    
    # Add minimal noise (2%) - keeping temporal patterns precise
    noise_mask = np.random.random(tier1_patterns.shape) < 0.02
    tier1_patterns[noise_mask] = 1.0
    
    # Create subdirectories if they don't exist
    os.makedirs('./data/temporal/datasets', exist_ok=True)
    os.makedirs('./data/temporal/examples', exist_ok=True)
    
    # Save to file
    save_path = os.path.join('./data/temporal/datasets', 'temporal_tier1_precise.npz')
    np.savez(save_path, patterns=tier1_patterns, labels=tier1_labels)
    print(f"Saved dataset to {save_path}")
    
    # Visualize examples
    fig, axes = plt.subplots(4, 2, figsize=(10, 12))
    
    for i in range(4):
        # Find samples for this class
        class_indices = np.where(tier1_labels == i)[0]
        
        # Select a sample
        if len(class_indices) > 0:
            idx = class_indices[0]
            
            # Display the spike raster plot
            spike_data = tier1_patterns[idx]
            neurons, times = np.where(spike_data > 0)
            axes[i, 0].scatter(times, neurons, s=5, marker='|', c='black')
            axes[i, 0].set_ylim(-0.5, spike_data.shape[0] - 0.5)
            axes[i, 0].invert_yaxis()
            axes[i, 0].set_title(f"Class {i} - Sample 1")
            
            # Select another sample for the same class
            if len(class_indices) > 1:
                idx = class_indices[1]
                spike_data = tier1_patterns[idx]
                neurons, times = np.where(spike_data > 0)
                axes[i, 1].scatter(times, neurons, s=5, marker='|', c='black')
                axes[i, 1].set_ylim(-0.5, spike_data.shape[0] - 0.5)
                axes[i, 1].invert_yaxis()
                axes[i, 1].set_title(f"Class {i} - Sample 2")
    
    plt.tight_layout()
    example_path = os.path.join('./data/temporal/examples', 'temporal_tier1_examples.png')
    plt.savefig(example_path, dpi=150)
    print(f"Saved visualization to {example_path}")
    plt.close()
    
    # ------------------- TIER 2: TEMPORAL CORRELATIONS (SHOULD FAVOR SNN) -------------------
    print("Generating Tier 2 (Temporal Correlations) temporal patterns dataset...")
    
    # Patterns with temporal correlations that SNNs should excel at
    t2_classes = []
    
    # Class 0: Temporal sequences with causal relationships
    for i in range(150):
        pattern = np.zeros((20, 100), dtype=np.float32)
        
        # Create a causal chain of activations
        start_time = 10 + i % 10
        
        # First group of neurons triggers second group and so on
        neuron_groups = [list(range(0, 5)), list(range(5, 10)), 
                         list(range(10, 15)), list(range(15, 20))]
        
        for group_idx, group in enumerate(neuron_groups):
            # Each group fires after previous group with some delay
            group_time = start_time + group_idx * 15
            
            for neuron_idx in group:
                # Add noise to individual neuron timing
                actual_time = int(group_time + np.random.normal(0, 1.0))
                if 0 <= actual_time < 100:
                    pattern[neuron_idx, actual_time] = 1.0
        
        t2_classes.append((pattern.reshape(1, 20, 100), np.zeros(1, dtype=np.int32)))
    
    # Class 1: Synchrony-then-asynchrony patterns
    for i in range(150):
        pattern = np.zeros((20, 100), dtype=np.float32)
        
        # First, synchronous firing
        sync_time = 20 + i % 5
        for j in range(10):  # First half of neurons
            actual_time = int(sync_time + np.random.normal(0, 0.5))
            if 0 <= actual_time < 100:
                pattern[j, actual_time] = 1.0
        
        # Then, asynchronous firing in sequence
        for j in range(10, 20):  # Second half of neurons
            seq_time = sync_time + 20 + (j-10) * 3
            actual_time = int(seq_time + np.random.normal(0, 0.5))
            if 0 <= actual_time < 100:
                pattern[j, actual_time] = 1.0
        
        t2_classes.append((pattern.reshape(1, 20, 100), np.ones(1, dtype=np.int32)))
    
    # Class 2: Rhythmic patterns with missing beats
    for i in range(150):
        pattern = np.zeros((20, 100), dtype=np.float32)
        
        # Create rhythmic pattern
        base_freq = 5 + i % 3
        # Determine which beats to skip (pattern-dependent)
        skip_beats = [(i % 5) + j for j in range(0, 15, 5)]
        
        for j in range(20):
            beat_count = 0
            for t in range(10, 90, base_freq):
                beat_count += 1
                if beat_count in skip_beats and j < 10:  # Skip beats for half the neurons
                    continue
                    
                actual_t = int(t + np.random.normal(0, 0.5))
                if 0 <= actual_t < 100:
                    pattern[j, actual_t] = 1.0
        
        t2_classes.append((pattern.reshape(1, 20, 100), np.full(1, 2, dtype=np.int32)))
    
    # Class 3: Delay-coded patterns
    for i in range(150):
        pattern = np.zeros((20, 100), dtype=np.float32)
        
        # Reference time
        ref_time = 20 + i % 10
        
        # Each neuron fires at a specific delay from reference
        for j in range(20):
            # Delay is neuron-specific
            delay = j * 2 + (i % 5)
            
            # Set spike with precise timing
            actual_time = int(ref_time + delay + np.random.normal(0, 0.5))
            if 0 <= actual_time < 100:
                pattern[j, actual_time] = 1.0
        
        t2_classes.append((pattern.reshape(1, 20, 100), np.full(1, 3, dtype=np.int32)))
    
    # Combine patterns
    tier2_patterns = np.zeros((600, 20, 100), dtype=np.float32)
    tier2_labels = np.zeros(600, dtype=np.int32)
    
    idx = 0
    for pattern_list, label in t2_classes:
        for i in range(len(pattern_list)):
            tier2_patterns[idx] = pattern_list[i]
            tier2_labels[idx] = label[0]
            idx += 1
    
    # Add substantial noise (10%) - Enough to challenge ANNs more than SNNs
    noise_mask = np.random.random(tier2_patterns.shape) < 0.10
    tier2_patterns[noise_mask] = 1.0
    
    # Save to file
    save_path = os.path.join('./data/temporal/datasets', 'temporal_tier2_correlation.npz')
    np.savez(save_path, patterns=tier2_patterns, labels=tier2_labels)
    print(f"Saved dataset to {save_path}")
    
    # Visualize examples
    fig, axes = plt.subplots(4, 2, figsize=(10, 12))
    
    for i in range(4):
        # Find samples for this class
        class_indices = np.where(tier2_labels == i)[0]
        
        # Select a sample
        if len(class_indices) > 0:
            idx = class_indices[0]
            
            # Display the spike raster plot
            spike_data = tier2_patterns[idx]
            neurons, times = np.where(spike_data > 0)
            axes[i, 0].scatter(times, neurons, s=5, marker='|', c='black')
            axes[i, 0].set_ylim(-0.5, spike_data.shape[0] - 0.5)
            axes[i, 0].invert_yaxis()
            axes[i, 0].set_title(f"Class {i} - Sample 1")
            
            # Select another sample for the same class
            if len(class_indices) > 1:
                idx = class_indices[1]
                spike_data = tier2_patterns[idx]
                neurons, times = np.where(spike_data > 0)
                axes[i, 1].scatter(times, neurons, s=5, marker='|', c='black')
                axes[i, 1].set_ylim(-0.5, spike_data.shape[0] - 0.5)
                axes[i, 1].invert_yaxis()
                axes[i, 1].set_title(f"Class {i} - Sample 2")
    
    plt.tight_layout()
    example_path = os.path.join('./data/temporal/examples', 'temporal_tier2_examples.png')
    plt.savefig(example_path, dpi=150)
    print(f"Saved visualization to {example_path}")
    plt.close()
    
    # ------------------- TIER 3: COMPLEX TEMPORAL PATTERNS (SHOULD STRONGLY FAVOR SNN) -------------------
    print("Generating Tier 3 (Complex Temporal) patterns dataset...")
    
    # Highly complex temporal patterns that strongly favor SNN's temporal dynamics
    t3_classes = []
    
    # Class 0: Variable interval encoding
    for i in range(150):
        pattern = np.zeros((20, 100), dtype=np.float32)
        
        # Information is encoded in the variable intervals between spikes
        for j in range(20):
            # Create a sequence of intervals that encode the class
            if j % 4 == 0:      # First pattern: increasing intervals
                intervals = [3, 6, 9, 12]
            elif j % 4 == 1:    # Second pattern: decreasing intervals
                intervals = [12, 9, 6, 3]
            elif j % 4 == 2:    # Third pattern: peak in middle
                intervals = [4, 8, 8, 4]
            else:               # Fourth pattern: valley in middle
                intervals = [8, 4, 4, 8]
                
            # Apply intervals to generate spike times
            current_time = 10 + (i % 5)
            for interval in intervals:
                current_time += interval
                if current_time < 100:
                    # Add small jitter to make it challenging
                    actual_time = int(current_time + np.random.normal(0, 0.5))
                    if 0 <= actual_time < 100:
                        pattern[j, actual_time] = 1.0
        
        t3_classes.append((pattern.reshape(1, 20, 100), np.zeros(1, dtype=np.int32)))
    
    # Class 1: Nested oscillation patterns
    for i in range(150):
        pattern = np.zeros((20, 100), dtype=np.float32)
        
        # Each neuron has a fast oscillation nested within a slow oscillation
        for j in range(20):
            # Slow oscillation parameters (varies by neuron)
            slow_freq = 25 + j % 5
            # Fast oscillation parameters (varies by sample)
            fast_freq = 4 + i % 3
            
            # Generate nested oscillation pattern
            for slow_t in range(10, 90, slow_freq):
                # Each slow cycle contains multiple fast cycles
                for fast_t in range(fast_freq):
                    t = slow_t + fast_t * (slow_freq // (fast_freq + 1))
                    if t < 100:
                        actual_t = int(t + np.random.normal(0, 0.5))
                        if 0 <= actual_t < 100:
                            pattern[j, actual_t] = 1.0
        
        t3_classes.append((pattern.reshape(1, 20, 100), np.ones(1, dtype=np.int32)))
    
    # Class 2: Spike pattern coincidence detection
    for i in range(150):
        pattern = np.zeros((20, 100), dtype=np.float32)
        
        # Groups of neurons that sometimes spike in synchrony
        neuron_groups = [list(range(0, 5)), list(range(5, 10)), 
                         list(range(10, 15)), list(range(15, 20))]
        
        # For each group, create a pattern with occasional synchrony
        for group_idx, group in enumerate(neuron_groups):
            # Base times for this group
            base_times = [15 + group_idx*5, 40 + group_idx*5, 65 + group_idx*5]
            
            # Synchronous spikes (half the time)
            if i % 2 == 0:
                # One synchronous event for all neurons in group
                sync_time = base_times[i % 3]
                for neuron_idx in group:
                    actual_time = int(sync_time + np.random.normal(0, 0.3))
                    if 0 <= actual_time < 100:
                        pattern[neuron_idx, actual_time] = 1.0
            
            # Individual spikes
            for neuron_idx in group:
                for base_time in base_times:
                    # Add neuron-specific offset
                    offset = (neuron_idx % 5) * 2
                    t = base_time + offset
                    if t < 100:
                        actual_t = int(t + np.random.normal(0, 0.5))
                        if 0 <= actual_t < 100:
                            pattern[neuron_idx, actual_t] = 1.0
        
        t3_classes.append((pattern.reshape(1, 20, 100), np.full(1, 2, dtype=np.int32)))
    
    # Class 3: Temporal XOR patterns
    for i in range(150):
        pattern = np.zeros((20, 100), dtype=np.float32)
        
        # Create temporal patterns that require XOR-like detection
        # (meaning need to detect presence of spike at t1 XOR presence at t2)
        reference_times = [20, 50, 80]
        
        for j in range(20):
            # Determine which reference times to use (in XOR pattern)
            use_t1 = j % 2 == 0
            use_t2 = j % 4 >= 2
            use_t3 = j % 3 == 0
            
            # Apply the pattern - this creates a complex temporal relationship
            if use_t1:
                t = reference_times[0] + np.random.normal(0, 0.5)
                actual_t = int(max(0, min(99, t)))
                pattern[j, actual_t] = 1.0
                
            if use_t2 and not (use_t1 and j % 8 == 0):  # XOR-like condition
                t = reference_times[1] + np.random.normal(0, 0.5)
                actual_t = int(max(0, min(99, t)))
                pattern[j, actual_t] = 1.0
                
            if use_t3 and not (use_t2 and j % 6 == 0):  # Another XOR-like condition
                t = reference_times[2] + np.random.normal(0, 0.5)
                actual_t = int(max(0, min(99, t)))
                pattern[j, actual_t] = 1.0
        
        t3_classes.append((pattern.reshape(1, 20, 100), np.full(1, 3, dtype=np.int32)))
    
    # Combine patterns
    tier3_patterns = np.zeros((600, 20, 100), dtype=np.float32)
    tier3_labels = np.zeros(600, dtype=np.int32)
    
    idx = 0
    for pattern_list, label in t3_classes:
        for i in range(len(pattern_list)):
            tier3_patterns[idx] = pattern_list[i]
            tier3_labels[idx] = label[0]
            idx += 1
    
    # Add significant noise (15%) - Making it much more challenging for ANNs but still tractable for SNNs
    noise_mask = np.random.random(tier3_patterns.shape) < 0.15
    tier3_patterns[noise_mask] = 1.0
    
    # Save to file
    save_path = os.path.join('./data/temporal/datasets', 'temporal_tier3_complex.npz')
    np.savez(save_path, patterns=tier3_patterns, labels=tier3_labels)
    print(f"Saved dataset to {save_path}")
    
    # Visualize examples
    fig, axes = plt.subplots(4, 2, figsize=(10, 12))
    
    for i in range(4):
        # Find samples for this class
        class_indices = np.where(tier3_labels == i)[0]
        
        # Select a sample
        if len(class_indices) > 0:
            idx = class_indices[0]
            
            # Display the spike raster plot
            spike_data = tier3_patterns[idx]
            neurons, times = np.where(spike_data > 0)
            axes[i, 0].scatter(times, neurons, s=5, marker='|', c='black')
            axes[i, 0].set_ylim(-0.5, spike_data.shape[0] - 0.5)
            axes[i, 0].invert_yaxis()
            axes[i, 0].set_title(f"Class {i} - Sample 1")
            
            # Select another sample for the same class
            if len(class_indices) > 1:
                idx = class_indices[1]
                spike_data = tier3_patterns[idx]
                neurons, times = np.where(spike_data > 0)
                axes[i, 1].scatter(times, neurons, s=5, marker='|', c='black')
                axes[i, 1].set_ylim(-0.5, spike_data.shape[0] - 0.5)
                axes[i, 1].invert_yaxis()
                axes[i, 1].set_title(f"Class {i} - Sample 2")
    
    plt.tight_layout()
    example_path = os.path.join('./data/temporal/examples', 'temporal_tier3_examples.png')
    plt.savefig(example_path, dpi=150)
    print(f"Saved visualization to {example_path}")
    plt.close()
    
    return {
        "tier1": (tier1_patterns, tier1_labels),
        "tier2": (tier2_patterns, tier2_labels),
        "tier3": (tier3_patterns, tier3_labels)
    }

def generate_tiered_spatial_patterns():
    """
    Generate spatial pattern datasets with clear properties
    - Each tier has a specific characteristic that FAVORS ANN over SNN
    - Datasets get progressively more challenging for SNNs while remaining ANN-friendly
    - Specifically designed to highlight ANNs' advantages in static spatial processing
    """
    # Create the output directory
    os.makedirs('./data', exist_ok=True)
    
    # Initialize the generator
    generator = SyntheticDataGenerator(output_dir='./data')
    
    # ------------------- TIER 1: STATIC CLEAN PATTERNS (SHOULD FAVOR ANN) -------------------
    print("Generating Tier 1 (Static Clean) spatial patterns dataset...")
    
    # Clean static spatial patterns - optimal for ANNs
    # Each pattern has a clear spatial structure but minimal temporal variation
    # Perfect for ANNs which excel at static pattern recognition
    width, height = 10, 10
    
    # Create distinctive geometric patterns for each class
    static_patterns = np.zeros((400, width*height, 100), dtype=np.float32)
    static_labels = np.zeros(400, dtype=np.int32)
    
    for sample_idx in range(400):
        # Determine class
        class_idx = sample_idx // 100
        
        # Create the base spatial pattern based on class
        spatial_pattern = np.zeros(width*height, dtype=np.float32)
        
        if class_idx == 0:
            # Class 0: Rectangular patterns - horizontal, vertical or square
            variant = sample_idx % 3
            
            if variant == 0:  # Horizontal rectangle
                start_x, start_y = 2, 4
                rect_width, rect_height = 6, 2
            elif variant == 1:  # Vertical rectangle
                start_x, start_y = 4, 2
                rect_width, rect_height = 2, 6
            else:  # Square
                start_x, start_y = 3, 3
                rect_width, rect_height = 4, 4
                
            # Create the rectangle
            for y in range(start_y, start_y + rect_height):
                for x in range(start_x, start_x + rect_width):
                    idx = y * width + x
                    spatial_pattern[idx] = 1.0
                    
        elif class_idx == 1:
            # Class 1: Diamond/X patterns
            variant = sample_idx % 3
            center_x, center_y = width // 2, height // 2
            
            if variant == 0:  # Diamond
                size = 3 + (sample_idx % 2)
                for y in range(height):
                    for x in range(width):
                        # Manhattan distance from center
                        dist = abs(x - center_x) + abs(y - center_y)
                        if dist == size:
                            idx = y * width + x
                            spatial_pattern[idx] = 1.0
            else:  # X pattern
                size = 3 + (sample_idx % 2)
                for offset in range(-size, size+1):
                    # Draw the two diagonals of the X
                    x1, y1 = center_x + offset, center_y + offset
                    x2, y2 = center_x + offset, center_y - offset
                    
                    if 0 <= x1 < width and 0 <= y1 < height:
                        idx = y1 * width + x1
                        spatial_pattern[idx] = 1.0
                        
                    if 0 <= x2 < width and 0 <= y2 < height:
                        idx = y2 * width + x2
                        spatial_pattern[idx] = 1.0
                        
        elif class_idx == 2:
            # Class 2: Circle/donut patterns
            variant = sample_idx % 3
            center_x, center_y = width // 2, height // 2
            
            if variant == 0:  # Solid circle
                radius = 3.0 + (sample_idx % 2)
                for y in range(height):
                    for x in range(width):
                        # Euclidean distance from center
                        dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                        if dist <= radius:
                            idx = y * width + x
                            spatial_pattern[idx] = 1.0
            else:  # Ring/donut
                radius = 3.0 + (sample_idx % 2)
                for y in range(height):
                    for x in range(width):
                        # Euclidean distance from center
                        dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                        if abs(dist - radius) <= 0.5:
                            idx = y * width + x
                            spatial_pattern[idx] = 1.0
                            
        elif class_idx == 3:
            # Class 3: Checkered/striped patterns
            variant = sample_idx % 3
            
            if variant == 0:  # Checkered
                check_size = 2 + (sample_idx % 2)
                for y in range(height):
                    for x in range(width):
                        if ((x // check_size) + (y // check_size)) % 2 == 0:
                            idx = y * width + x
                            spatial_pattern[idx] = 1.0
            else:  # Striped
                is_horizontal = sample_idx % 2 == 0
                stripe_width = 2
                
                for y in range(height):
                    for x in range(width):
                        if is_horizontal:
                            if (y // stripe_width) % 2 == 0:
                                idx = y * width + x
                                spatial_pattern[idx] = 1.0
                        else:
                            if (x // stripe_width) % 2 == 0:
                                idx = y * width + x
                                spatial_pattern[idx] = 1.0
        
        # Create a static pattern - same spatial configuration repeats
        new_pattern = np.zeros((width*height, 100), dtype=np.float32)
        
        # Create a perfectly static pattern with no variation across time
        # This heavily favors ANN's ability to extract spatial features
        for t in range(20, 80):
            # No noise added - pure spatial pattern that repeats exactly
            new_pattern[:, t] = spatial_pattern
        
        # Store the pattern
        static_patterns[sample_idx] = new_pattern
        static_labels[sample_idx] = class_idx
    
    # Create subdirectories if they don't exist
    os.makedirs('./data/spatial/datasets', exist_ok=True)
    os.makedirs('./data/spatial/examples', exist_ok=True)
    
    # Save to file
    save_path = os.path.join('./data/spatial/datasets', 'spatial_tier1_static.npz')
    np.savez(save_path, patterns=static_patterns, labels=static_labels)
    print(f"Saved dataset to {save_path}")
    
    # Visualize examples
    fig, axes = plt.subplots(4, 2, figsize=(10, 12))
    
    for i in range(4):
        # Find samples for this class
        class_indices = np.where(static_labels == i)[0]
        
        # Get two sample indices for this class
        if len(class_indices) > 0:
            idx1 = class_indices[0]
            
            # For spatial data, reshape to 2D and show the pattern at a specific timestep
            pattern = static_patterns[idx1]
            sample_t = 50  # Choose a timestep in the middle
            spatial_pattern = pattern[:, sample_t].reshape(10, 10)  # Reshape to 10x10 grid
            
            # Plot as heatmap
            axes[i, 0].imshow(spatial_pattern, cmap='viridis', interpolation='nearest')
            axes[i, 0].set_title(f"Class {i} - Sample 1")
            axes[i, 0].axis('off')
            
            # Select another sample for the same class
            if len(class_indices) > 1:
                idx2 = class_indices[1]
                pattern = static_patterns[idx2]
                spatial_pattern = pattern[:, sample_t].reshape(10, 10)
                
                axes[i, 1].imshow(spatial_pattern, cmap='viridis', interpolation='nearest')
                axes[i, 1].set_title(f"Class {i} - Sample 2")
                axes[i, 1].axis('off')
    
    plt.tight_layout()
    example_path = os.path.join('./data/spatial/examples', 'spatial_tier1_examples.png')
    plt.savefig(example_path, dpi=150)
    print(f"Saved visualization to {example_path}")
    plt.close()
    
    # ------------------- TIER 2: BATCH VARIABILITY (SHOULD FAVOR ANN) -------------------
    print("Generating Tier 2 (Batch Variability) spatial patterns dataset...")
    
    # Patterns with batch-to-batch variations but still clearly distinguishable
    # Each pattern is spatially consistent but varies somewhat between samples
    # ANNs should handle this well with their batch training approach
    
    batch_patterns = np.zeros((400, width*height, 100), dtype=np.float32)
    batch_labels = np.zeros(400, dtype=np.int32)
    
    for sample_idx in range(400):
        # Determine class
        class_idx = sample_idx // 100
        
        # Start with a base spatial pattern similar to tier 1
        spatial_pattern = np.zeros(width*height, dtype=np.float32)
        
        if class_idx == 0:
            # Class 0: Variants of line patterns
            variant = sample_idx % 4
            
            if variant == 0:  # Horizontal lines
                for y in range(2, height-2, 2):
                    for x in range(1, width-1):
                        idx = y * width + x
                        spatial_pattern[idx] = 1.0
            elif variant == 1:  # Vertical lines
                for x in range(2, width-2, 2):
                    for y in range(1, height-1):
                        idx = y * width + x
                        spatial_pattern[idx] = 1.0
            elif variant == 2:  # Diagonal lines (/)
                offset = sample_idx % 3
                for i in range(-5, 15):
                    x, y = i, height - i - offset
                    if 0 <= x < width and 0 <= y < height:
                        idx = y * width + x
                        spatial_pattern[idx] = 1.0
            else:  # Diagonal lines (\)
                offset = sample_idx % 3
                for i in range(-5, 15):
                    x, y = i, i + offset
                    if 0 <= x < width and 0 <= y < height:
                        idx = y * width + x
                        spatial_pattern[idx] = 1.0
                        
        elif class_idx == 1:
            # Class 1: Variants of corner/edge patterns
            variant = sample_idx % 4
            
            if variant == 0:  # Top-left corner
                size = 4 + (sample_idx % 3)
                for y in range(size):
                    for x in range(size - y):
                        idx = y * width + x
                        spatial_pattern[idx] = 1.0
            elif variant == 1:  # Bottom-right corner
                size = 4 + (sample_idx % 3)
                for y in range(height - size, height):
                    for x in range(width - (height - y), width):
                        idx = y * width + x
                        spatial_pattern[idx] = 1.0
            elif variant == 2:  # Top edge
                height_var = 2 + (sample_idx % 3)
                for y in range(height_var):
                    for x in range(1, width-1):
                        idx = y * width + x
                        spatial_pattern[idx] = 1.0
            else:  # Bottom edge
                height_var = 2 + (sample_idx % 3)
                for y in range(height - height_var, height):
                    for x in range(1, width-1):
                        idx = y * width + x
                        spatial_pattern[idx] = 1.0
                        
        elif class_idx == 2:
            # Class 2: Variants of grid patterns
            variant = sample_idx % 4
            grid_size = 3 + (sample_idx % 3)
            
            if variant == 0:  # Regular grid
                for y in range(0, height, grid_size):
                    for x in range(0, width):
                        idx = y * width + x
                        spatial_pattern[idx] = 1.0
                for y in range(0, height):
                    for x in range(0, width, grid_size):
                        idx = y * width + x
                        spatial_pattern[idx] = 1.0
            elif variant == 1:  # Sparse grid
                for y in range(0, height, grid_size):
                    for x in range(0, width, grid_size):
                        idx = y * width + x
                        spatial_pattern[idx] = 1.0
            elif variant == 2:  # Offset grid
                offset = sample_idx % grid_size
                for y in range(offset, height, grid_size):
                    for x in range(offset, width, grid_size):
                        idx = y * width + x
                        spatial_pattern[idx] = 1.0
            else:  # Dotted grid
                for y in range(1, height-1, grid_size):
                    for x in range(1, width-1, grid_size):
                        # Small dot at grid point
                        idx = y * width + x
                        spatial_pattern[idx] = 1.0
                        # Adjacent points
                        if x+1 < width:
                            spatial_pattern[y*width + (x+1)] = 1.0
                        if y+1 < height:
                            spatial_pattern[(y+1)*width + x] = 1.0
                        
        elif class_idx == 3:
            # Class 3: Blob patterns with position variability
            variant = sample_idx % 4
            
            # Base position of blob varies by sample
            center_x = 3 + (sample_idx % 5)
            center_y = 3 + ((sample_idx // 3) % 5)
            
            if variant == 0:  # Circular blob
                radius = 2.0 + (sample_idx % 3) * 0.5
                for y in range(height):
                    for x in range(width):
                        dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                        if dist <= radius:
                            idx = y * width + x
                            spatial_pattern[idx] = 1.0
            elif variant == 1:  # Square blob
                size = 2 + (sample_idx % 3)
                for y in range(max(0, center_y-size), min(height, center_y+size+1)):
                    for x in range(max(0, center_x-size), min(width, center_x+size+1)):
                        idx = y * width + x
                        spatial_pattern[idx] = 1.0
            elif variant == 2:  # Diamond blob
                size = 2 + (sample_idx % 3)
                for y in range(height):
                    for x in range(width):
                        dist = abs(x - center_x) + abs(y - center_y)
                        if dist <= size:
                            idx = y * width + x
                            spatial_pattern[idx] = 1.0
            else:  # Random shaped blob
                size = 2 + (sample_idx % 3)
                # Start with a square
                blob_points = set()
                for y in range(max(0, center_y-size), min(height, center_y+size+1)):
                    for x in range(max(0, center_x-size), min(width, center_x+size+1)):
                        blob_points.add((y, x))
                
                # Randomly remove some points from the periphery
                to_remove = []
                for y, x in blob_points:
                    if abs(y - center_y) + abs(x - center_x) > size and np.random.random() < 0.5:
                        to_remove.append((y, x))
                
                for pt in to_remove:
                    if pt in blob_points:
                        blob_points.remove(pt)
                
                # Set the pattern
                for y, x in blob_points:
                    idx = y * width + x
                    spatial_pattern[idx] = 1.0
        
        # Create a pattern that repeats the spatial pattern across time
        # with moderate batch-to-batch variability
        new_pattern = np.zeros((width*height, 100), dtype=np.float32)
        
        # Add the spatial pattern at multiple timesteps with moderate variation
        # This tier introduces some temporal variation but still favors ANN processing
        for t in range(20, 80):
            # Moderate variation (5%) - enough to introduce some temporal dynamics
            # but still primarily a spatial problem
            this_pattern = spatial_pattern.copy()
            noise_mask = np.random.random(this_pattern.shape) < 0.05
            this_pattern[noise_mask] = 1 - this_pattern[noise_mask]
            
            # Add the pattern at this time step
            new_pattern[:, t] = this_pattern
        
        # Store the pattern
        batch_patterns[sample_idx] = new_pattern
        batch_labels[sample_idx] = class_idx
    
    # Save to file
    save_path = os.path.join('./data/spatial/datasets', 'spatial_tier2_batch.npz')
    np.savez(save_path, patterns=batch_patterns, labels=batch_labels)
    print(f"Saved dataset to {save_path}")
    
    # Visualize examples
    fig, axes = plt.subplots(4, 2, figsize=(10, 12))
    
    for i in range(4):
        # Find samples for this class
        class_indices = np.where(batch_labels == i)[0]
        
        # Get two sample indices for this class
        if len(class_indices) > 0:
            idx1 = class_indices[0]
            
            # For spatial data, reshape to 2D and show the pattern at a specific timestep
            pattern = batch_patterns[idx1]
            sample_t = 50  # Choose a timestep in the middle
            spatial_pattern = pattern[:, sample_t].reshape(10, 10)  # Reshape to 10x10 grid
            
            # Plot as heatmap
            axes[i, 0].imshow(spatial_pattern, cmap='viridis', interpolation='nearest')
            axes[i, 0].set_title(f"Class {i} - Sample 1")
            axes[i, 0].axis('off')
            
            # Select another sample for the same class
            if len(class_indices) > 1:
                idx2 = class_indices[len(class_indices) // 2]  # Get a sample from middle of the set
                pattern = batch_patterns[idx2]
                spatial_pattern = pattern[:, sample_t].reshape(10, 10)
                
                axes[i, 1].imshow(spatial_pattern, cmap='viridis', interpolation='nearest')
                axes[i, 1].set_title(f"Class {i} - Sample 2")
                axes[i, 1].axis('off')
    
    plt.tight_layout()
    example_path = os.path.join('./data/spatial/examples', 'spatial_tier2_examples.png')
    plt.savefig(example_path, dpi=150)
    print(f"Saved visualization to {example_path}")
    plt.close()
    
    # ------------------- TIER 3: SCALED COMPLEXITY (SHOULD STILL FAVOR ANN) -------------------
    print("Generating Tier 3 (Scaled Complexity) spatial patterns dataset...")
    
    # More complex but still static patterns that benefit from ANN's hierarchical processing
    # These patterns have more detail and structure but are still recognizable across time
    # ANNs with their hierarchical processing should still perform well
    
    complex_patterns = np.zeros((400, width*height, 100), dtype=np.float32)
    complex_labels = np.zeros(400, dtype=np.int32)
    
    for sample_idx in range(400):
        # Determine class
        class_idx = sample_idx // 100
        
        # Create a more complex base pattern for this class
        spatial_pattern = np.zeros(width*height, dtype=np.float32)
        
        if class_idx == 0:
            # Class 0: Nested geometric patterns
            variant = sample_idx % 3
            
            if variant == 0:  # Nested squares
                # Outer square
                for y in range(1, height-1):
                    for x in range(1, width-1):
                        if x == 1 or x == width-2 or y == 1 or y == height-2:
                            idx = y * width + x
                            spatial_pattern[idx] = 1.0
                
                # Inner square
                inner_size = 3
                inner_start = (width - inner_size) // 2
                for y in range(inner_start, inner_start + inner_size):
                    for x in range(inner_start, inner_start + inner_size):
                        if x == inner_start or x == inner_start + inner_size - 1 or \
                           y == inner_start or y == inner_start + inner_size - 1:
                            idx = y * width + x
                            spatial_pattern[idx] = 1.0
            
            elif variant == 1:  # Nested circles
                center_x, center_y = width // 2, height // 2
                
                # Outer circle
                radius1 = 4.0
                # Inner circle
                radius2 = 2.0
                
                for y in range(height):
                    for x in range(width):
                        dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                        if abs(dist - radius1) <= 0.5 or abs(dist - radius2) <= 0.5:
                            idx = y * width + x
                            spatial_pattern[idx] = 1.0
            
            else:  # Complex cross pattern
                # Horizontal line
                for x in range(width):
                    idx = (height // 2) * width + x
                    spatial_pattern[idx] = 1.0
                
                # Vertical line
                for y in range(height):
                    idx = y * width + (width // 2)
                    spatial_pattern[idx] = 1.0
                
                # Small diamond in center
                center_x, center_y = width // 2, height // 2
                size = 2
                for y in range(height):
                    for x in range(width):
                        dist = abs(x - center_x) + abs(y - center_y)
                        if dist <= size:
                            idx = y * width + x
                            spatial_pattern[idx] = 1.0
                            
        elif class_idx == 1:
            # Class 1: Complex symmetrical patterns
            variant = sample_idx % 3
            center_x, center_y = width // 2, height // 2
            
            if variant == 0:  # Four-fold symmetry pattern
                for y in range(height):
                    for x in range(width):
                        # Distance from center
                        dx, dy = x - center_x, y - center_y
                        # Create pattern with four-fold symmetry
                        if (abs(dx) % 3 == 0 and abs(dy) < 4) or \
                           (abs(dy) % 3 == 0 and abs(dx) < 4):
                            idx = y * width + x
                            spatial_pattern[idx] = 1.0
            
            elif variant == 1:  # Spiral-like pattern
                for angle in range(0, 360, 15):
                    for r in range(1, 5):
                        # Create spiral segments
                        rad = np.radians(angle + r * 10)
                        x = int(center_x + r * np.cos(rad))
                        y = int(center_y + r * np.sin(rad))
                        if 0 <= x < width and 0 <= y < height:
                            idx = y * width + x
                            spatial_pattern[idx] = 1.0
            
            else:  # Star-like pattern
                points = 8
                for i in range(points):
                    angle = i * (2 * np.pi / points)
                    # Inner point
                    r1 = 2
                    x1 = int(center_x + r1 * np.cos(angle))
                    y1 = int(center_y + r1 * np.sin(angle))
                    # Outer point
                    r2 = 4
                    x2 = int(center_x + r2 * np.cos(angle))
                    y2 = int(center_y + r2 * np.sin(angle))
                    
                    # Draw line between inner and outer points
                    if 0 <= x1 < width and 0 <= y1 < height:
                        idx = y1 * width + x1
                        spatial_pattern[idx] = 1.0
                    if 0 <= x2 < width and 0 <= y2 < height:
                        idx = y2 * width + x2
                        spatial_pattern[idx] = 1.0
                    
                    # Connect with lines
                    steps = max(abs(x2 - x1), abs(y2 - y1))
                    for s in range(steps):
                        t = s / steps
                        x = int(x1 + t * (x2 - x1))
                        y = int(y1 + t * (y2 - y1))
                        if 0 <= x < width and 0 <= y < height:
                            idx = y * width + x
                            spatial_pattern[idx] = 1.0
                            
        elif class_idx == 2:
            # Class 2: Texture-like patterns
            variant = sample_idx % 3
            
            if variant == 0:  # Gradient-like pattern
                for y in range(height):
                    threshold = y / height
                    for x in range(width):
                        if np.random.random() < threshold:
                            idx = y * width + x
                            spatial_pattern[idx] = 1.0
            
            elif variant == 1:  # Cellular automaton inspired
                # Start with random seed points
                seed_points = []
                for _ in range(5):
                    x = np.random.randint(width)
                    y = np.random.randint(height)
                    seed_points.append((y, x))
                    idx = y * width + x
                    spatial_pattern[idx] = 1.0
                
                # Expand around seed points
                for _ in range(3):  # Iterations of growth
                    new_points = []
                    for y, x in seed_points:
                        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < height and 0 <= nx < width:
                                if np.random.random() < 0.7:  # 70% chance to grow
                                    idx = ny * width + nx
                                    if spatial_pattern[idx] == 0:  # Only if not already set
                                        spatial_pattern[idx] = 1.0
                                        new_points.append((ny, nx))
                    seed_points = new_points
            
            else:  # Regular textured pattern
                pattern_size = 2 + (sample_idx % 3)
                for y in range(height):
                    for x in range(width):
                        # Create complex textured pattern
                        if (x // pattern_size + y // pattern_size) % 2 == 0:
                            if (x % pattern_size) == (y % pattern_size):
                                idx = y * width + x
                                spatial_pattern[idx] = 1.0
                            
        elif class_idx == 3:
            # Class 3: Multi-region patterns
            variant = sample_idx % 3
            
            if variant == 0:  # Quadrant division
                for y in range(height):
                    for x in range(width):
                        if x < width//2 and y < height//2:  # Top-left: horizontal lines
                            if y % 3 == 0:
                                idx = y * width + x
                                spatial_pattern[idx] = 1.0
                        elif x >= width//2 and y < height//2:  # Top-right: vertical lines
                            if x % 3 == 0:
                                idx = y * width + x
                                spatial_pattern[idx] = 1.0
                        elif x < width//2 and y >= height//2:  # Bottom-left: dots
                            if x % 3 == 0 and y % 3 == 0:
                                idx = y * width + x
                                spatial_pattern[idx] = 1.0
                        else:  # Bottom-right: diagonal lines
                            if (x + y) % 3 == 0:
                                idx = y * width + x
                                spatial_pattern[idx] = 1.0
            
            elif variant == 1:  # Three diagonal regions
                for y in range(height):
                    for x in range(width):
                        region = (x + y) // 5  # Divide into 3-4 diagonal bands
                        if region % 3 == 0:  # First region: horizontal stripes
                            if y % 2 == 0:
                                idx = y * width + x
                                spatial_pattern[idx] = 1.0
                        elif region % 3 == 1:  # Second region: vertical stripes
                            if x % 2 == 0:
                                idx = y * width + x
                                spatial_pattern[idx] = 1.0
                        else:  # Third region: checkerboard
                            if (x + y) % 2 == 0:
                                idx = y * width + x
                                spatial_pattern[idx] = 1.0
            
            else:  # Complex segmented pattern
                segment_type = sample_idx % 5
                for y in range(height):
                    for x in range(width):
                        # Segment the space into 5 regions by distance from center
                        center_x, center_y = width // 2, height // 2
                        dist = max(abs(x - center_x), abs(y - center_y))
                        
                        if dist % 5 == segment_type:
                            idx = y * width + x
                            spatial_pattern[idx] = 1.0
        
        # Create a pattern that repeats with moderate variability
        # Still static enough for ANNs to do well
        new_pattern = np.zeros((width*height, 100), dtype=np.float32)
        
        # Add the complex spatial pattern across multiple timesteps with more temporal variation
        # This tier challenges both models but in a way that still gives ANN an edge
        for t in range(20, 80):
            # Higher noise/variability (8%) - approaching the point where temporal 
            # processing becomes important, but still maintainable as a spatial task
            this_pattern = spatial_pattern.copy()
            
            # Apply noise but ensure the core pattern remains recognizable
            noise_mask = np.random.random(this_pattern.shape) < 0.08
            this_pattern[noise_mask] = 1 - this_pattern[noise_mask]
            
            # Add the pattern at this time step
            new_pattern[:, t] = this_pattern
        
        # Store the pattern
        complex_patterns[sample_idx] = new_pattern
        complex_labels[sample_idx] = class_idx
    
    # Save to file
    save_path = os.path.join('./data/spatial/datasets', 'spatial_tier3_complex.npz')
    np.savez(save_path, patterns=complex_patterns, labels=complex_labels)
    print(f"Saved dataset to {save_path}")
    
    # Visualize examples
    fig, axes = plt.subplots(4, 2, figsize=(10, 12))
    
    for i in range(4):
        # Find samples for this class
        class_indices = np.where(complex_labels == i)[0]
        
        # Get two sample indices for this class
        if len(class_indices) > 0:
            idx1 = class_indices[0]
            
            # For spatial data, reshape to 2D and show the pattern at a specific timestep
            pattern = complex_patterns[idx1]
            sample_t = 50  # Choose a timestep in the middle
            spatial_pattern = pattern[:, sample_t].reshape(10, 10)  # Reshape to 10x10 grid
            
            # Plot as heatmap
            axes[i, 0].imshow(spatial_pattern, cmap='viridis', interpolation='nearest')
            axes[i, 0].set_title(f"Class {i} - Sample 1")
            axes[i, 0].axis('off')
            
            # Select another sample for the same class
            if len(class_indices) > 1:
                idx2 = class_indices[len(class_indices) // 2]  # Get a sample from middle of the set
                pattern = complex_patterns[idx2]
                spatial_pattern = pattern[:, sample_t].reshape(10, 10)
                
                axes[i, 1].imshow(spatial_pattern, cmap='viridis', interpolation='nearest')
                axes[i, 1].set_title(f"Class {i} - Sample 2")
                axes[i, 1].axis('off')
    
    plt.tight_layout()
    example_path = os.path.join('./data/spatial/examples', 'spatial_tier3_examples.png')
    plt.savefig(example_path, dpi=150)
    print(f"Saved visualization to {example_path}")
    plt.close()
    
    return {
        "tier1": (static_patterns, static_labels),
        "tier2": (batch_patterns, batch_labels),
        "tier3": (complex_patterns, complex_labels)
    }


def generate_datasets():
    """
    Generate all difficulty tiered datasets for experiments
    """
    # Create the output directory
    os.makedirs('./data', exist_ok=True)
    
    # Generate temporal pattern datasets
    print("\n=== Generating Temporal Pattern Datasets ===")
    temporal_datasets = generate_tiered_temporal_patterns()
    
    # Generate spatial pattern datasets
    print("\n=== Generating Spatial Pattern Datasets ===")
    spatial_datasets = generate_tiered_spatial_patterns()
    
    # Create legacy names for compatibility with existing script
    print("\n=== Creating compatible dataset names ===")
    
    # Legacy temporal dataset names
    os.makedirs('./data/synthetic', exist_ok=True)
    generator = SyntheticDataGenerator(output_dir='./data')
    
    # Map to older expected names for temporal datasets
    temporal_mappings = [
        (os.path.join('./data/temporal/datasets', 'temporal_tier1_precise.npz'), 'simple_synthetic_5class.npz'),
        (os.path.join('./data/temporal/datasets', 'temporal_tier2_correlation.npz'), 'medium_synthetic_5class.npz'),
        (os.path.join('./data/temporal/datasets', 'temporal_tier3_complex.npz'), 'complex_synthetic_5class.npz')
    ]
    
    # Create copies with legacy names
    for source, dest in temporal_mappings:
        dest_path = os.path.join('./data/synthetic', dest)
        if os.path.exists(source):
            print(f"Creating copy: {os.path.basename(source)} -> {dest}")
            with open(source, 'rb') as f_in:
                data = f_in.read()
                with open(dest_path, 'wb') as f_out:
                    f_out.write(data)
    
    # Map to older expected names for spatial datasets
    spatial_mappings = [
        (os.path.join('./data/spatial/datasets', 'spatial_tier1_static.npz'), 'small_spatial_10class.npz'),
        (os.path.join('./data/spatial/datasets', 'spatial_tier2_batch.npz'), 'medium_spatial_10class.npz'),
        (os.path.join('./data/spatial/datasets', 'spatial_tier3_complex.npz'), 'large_spatial_10class.npz')
    ]
    
    # Create copies with legacy names
    for source, dest in spatial_mappings:
        dest_path = os.path.join('./data/spatial', dest)
        if os.path.exists(source):
            print(f"Creating copy: {os.path.basename(source)} -> {dest}")
            with open(source, 'rb') as f_in:
                data = f_in.read()
                with open(dest_path, 'wb') as f_out:
                    f_out.write(data)
    
    print("\nAll datasets generated successfully!")
    print("\nGenerated Datasets:")
    print("  Temporal Patterns (Designed to favor SNN):")
    print("    - ./data/temporal/datasets/temporal_tier1_precise.npz (Precise timing patterns)")
    print("    - ./data/temporal/datasets/temporal_tier2_correlation.npz (Temporal correlation patterns)")
    print("    - ./data/temporal/datasets/temporal_tier3_complex.npz (Complex temporal patterns)")
    print("\n  Example Visualizations:")
    print("    - ./data/temporal/examples/temporal_tier1_examples.png")
    print("    - ./data/temporal/examples/temporal_tier2_examples.png")
    print("    - ./data/temporal/examples/temporal_tier3_examples.png")
    
    print("\n  Spatial Patterns (Designed to favor ANN):")
    print("    - ./data/spatial/datasets/spatial_tier1_static.npz (Static clean patterns)")
    print("    - ./data/spatial/datasets/spatial_tier2_batch.npz (Batch variability patterns)")
    print("    - ./data/spatial/datasets/spatial_tier3_complex.npz (Scaled complexity patterns)")
    print("\n  Example Visualizations:")
    print("    - ./data/spatial/examples/spatial_tier1_examples.png")
    print("    - ./data/spatial/examples/spatial_tier2_examples.png")
    print("    - ./data/spatial/examples/spatial_tier3_examples.png")
    
    print("\nCompatibility names created in subdirectories:")
    print("  ./data/synthetic/simple_synthetic_5class.npz")
    print("  ./data/synthetic/medium_synthetic_5class.npz")  
    print("  ./data/synthetic/complex_synthetic_5class.npz")
    print("  ./data/spatial/small_spatial_10class.npz")
    print("  ./data/spatial/medium_spatial_10class.npz")
    print("  ./data/spatial/large_spatial_10class.npz")

def plot_confusion_matrix(cm, class_names, title='Confusion Matrix', output_file="./results/confusion_matrix.png", show=True):
    """
    Plot a confusion matrix.
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        title: Title for the plot
        output_file: Path to save the plot
        show: Whether to display the plot
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=class_names, 
               yticklabels=class_names)
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    
    # Save the plot
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file)
    
    if show:
        plt.show()
    else:
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='Run synthetic SNN experiments')
    parser.add_argument('--generate', action='store_true', help='Generate synthetic datasets')
    parser.add_argument('--plot_results', type=str, help='Plot results from JSON file')
    parser.add_argument('--analyze_noise', action='store_true', help='Analyze impact of noise on SNN performance')
    parser.add_argument('--show_plots', action='store_true', help='Show plots interactively (otherwise just save to file)')
    parser.add_argument('--generate_matrix', action='store_true', help='Generate confusion matrix plot')
    
    args = parser.parse_args()
    
    if args.generate:
        generate_datasets()
    
    if args.generate_matrix:
        os.makedirs('./results', exist_ok=True)
        cm = confusion_matrix([0, 1, 0, 1], [1, 1, 1, 0])
        plot_confusion_matrix(cm, ['A', 'B'], output_file='./results/confusion_matrix.png', show=args.show_plots)

if __name__ == "__main__":
    main()