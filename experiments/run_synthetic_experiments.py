#!/usr/bin/env python3
"""
Script to generate synthetic data with multiple difficulty tiers for SNN experiments.

This enhanced version creates datasets with increasing difficulty levels for both
spatial and temporal pattern discrimination tasks, as well as specialized ANN-optimized
datasets that showcase ANN strengths.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import json
import time
import shutil
from sklearn.metrics import confusion_matrix, classification_report
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
    
    # Class 1: Synchrony-then-wave patterns (made more distinct from class 3)
    for i in range(150):
        pattern = np.zeros((20, 100), dtype=np.float32)
        
        # First, strong synchronous firing burst - key distinctive feature
        sync_time_1 = 15 + i % 5
        sync_time_2 = 35 + i % 5  # Second synchronous burst
        
        # Create two clear synchronous bursts
        for sync_time in [sync_time_1, sync_time_2]:
            for j in range(12):  # First 60% of neurons fire synchronously
                actual_time = int(sync_time + np.random.normal(0, 0.3))  # Less jitter for clearer pattern
                if 0 <= actual_time < 100:
                    pattern[j, actual_time] = 1.0
        
        # Then, wave pattern (not linear sequence like class 3)
        # Create a wave pattern where neurons in middle fire first, then spreads outward
        center = 15  # Center neuron index
        for offset in range(5):
            wave_time = sync_time_2 + 15 + offset * 4  # Larger time gaps between waves
            
            # Neurons fire in waves from center outward (very different from class 3's pattern)
            neurons_in_wave = [center - offset, center + offset]
            for neuron_idx in neurons_in_wave:
                if 0 <= neuron_idx < 20:
                    actual_time = int(wave_time + np.random.normal(0, 0.3))
                    if 0 <= actual_time < 100:
                        pattern[neuron_idx, actual_time] = 1.0
        
        # Add a distinctive ending burst
        end_time = 75
        for j in range(5):
            actual_time = int(end_time + np.random.normal(0, 1.0))
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
    
    # Class 3: Distinctive diagonal stripe pattern (completely different from Class 1)
    for i in range(150):
        pattern = np.zeros((20, 100), dtype=np.float32)
        
        # Create a clear diagonal firing pattern (neurons fire sequentially from bottom to top)
        start_time = 10 + i % 15  # More variable start times
        
        # DISTINCTIVE FEATURE: Clear diagonal stripe pattern with multiple stripes
        for stripe in range(3):  # Create 3 diagonal stripes
            stripe_start = start_time + stripe * 25  # Well-separated stripes
            
            # Each neuron fires at precise times creating a diagonal pattern
            for j in range(20):  # All neurons participate
                # Linear progression with precise timing (clear diagonal in visualization)
                spike_time = stripe_start + j * 2  # Consistent 2-timestep spacing
                
                if spike_time < 100:
                    # Very precise timing with minimal jitter for clear pattern
                    actual_time = int(spike_time + np.random.normal(0, 0.2))
                    if 0 <= actual_time < 100:
                        pattern[j, actual_time] = 1.0
        
        # Add a distinctive short burst near the end for certain neurons
        # This creates an easily recognizable feature unique to class 3
        if i % 2 == 0:  # For half the samples
            burst_time = 85
            for j in range(5, 10):  # Middle subset of neurons
                actual_time = int(burst_time + np.random.normal(0, 0.5))
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
    
    # Add moderate noise (8%) - Enough to challenge ANNs but still make patterns recognizable
    # Using slightly less noise to ensure the distinctive patterns we created remain clear
    noise_mask = np.random.random(tier2_patterns.shape) < 0.08
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
    - Each tier has a specific characteristic that STRONGLY FAVORS ANN over SNN
    - Datasets have virtually NO temporal structure to completely disadvantage SNNs
    - Specifically designed to highlight ANNs' advantages in static spatial processing
    """
    # Create the output directory
    os.makedirs('./data', exist_ok=True)
    
    # Initialize the generator
    generator = SyntheticDataGenerator(output_dir='./data')
    
    # ------------------- TIER 1: PURE STATIC PATTERNS (SHOULD STRONGLY FAVOR ANN) -------------------
    print("Generating Tier 1 (Pure Static) spatial patterns dataset...")
    
    # Completely static spatial patterns - optimal for ANNs and very difficult for SNNs
    # Each pattern has a clear spatial structure with ZERO temporal variation
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
        
        # Create a truly static pattern - SAME pattern at EVERY timestep
        # This strongly favors ANNs and disadvantages SNNs which rely on temporal dynamics
        new_pattern = np.zeros((width*height, 100), dtype=np.float32)
        
        # Add modest noise to spatial pattern (20%) to make it challenging
        # but apply the SAME noise pattern at all timesteps - strongly favoring ANN
        noise_mask = np.random.random(spatial_pattern.shape) < 0.20
        noisy_pattern = spatial_pattern.copy()
        noisy_pattern[noise_mask] = 1 - noisy_pattern[noise_mask]
        
        # Apply DIFFERENT noise patterns at each timestep to prevent SNN's temporal integration advantage
        # This truly forces reliance on spatial pattern recognition rather than temporal integration
        for t in range(10, 90):  # Use all time steps with unique noise per timestep
            # Apply fresh random noise to each timestep independently
            timestep_noise_mask = np.random.random(spatial_pattern.shape) < 0.20
            timestep_noisy_pattern = spatial_pattern.copy()
            timestep_noisy_pattern[timestep_noise_mask] = 1 - timestep_noisy_pattern[timestep_noise_mask]
            new_pattern[:, t] = timestep_noisy_pattern
        
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
    
    # ------------------- TIER 2: HIGH NOISE STATIC PATTERNS (SHOULD FAVOR ANN) -------------------
    print("Generating Tier 2 (High Noise Static) spatial patterns dataset...")
    
    # Static patterns with high noise but still no temporal variation
    # Each pattern is spatially consistent between samples but has significant noise
    # ANNs should handle this well while SNNs will struggle
    
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
        
        # Create a completely static pattern with high noise level
        new_pattern = np.zeros((width*height, 100), dtype=np.float32)
        
        # Apply very high noise (35%) to make it challenging
        # but keep it STATIC across time - the exact same pattern at all timesteps
        noise_mask = np.random.random(spatial_pattern.shape) < 0.35
        noisy_pattern = spatial_pattern.copy()
        noisy_pattern[noise_mask] = 1 - noisy_pattern[noise_mask]
        
        # Apply DIFFERENT noise patterns at each timestep to prevent SNN's temporal integration advantage
        for t in range(10, 90):
            # Apply fresh random noise to each timestep independently
            timestep_noise_mask = np.random.random(spatial_pattern.shape) < 0.35
            timestep_noisy_pattern = spatial_pattern.copy()
            timestep_noisy_pattern[timestep_noise_mask] = 1 - timestep_noisy_pattern[timestep_noise_mask]
            new_pattern[:, t] = timestep_noisy_pattern
        
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
    
    # ------------------- TIER 3: EXTREMELY COMPLEX PATTERNS (SHOULD STRONGLY FAVOR ANN) -------------------
    print("Generating Tier 3 (Extremely Complex) spatial patterns dataset...")
    
    # Very complex static patterns that strongly benefit from ANN's hierarchical processing
    # These patterns have intricate spatial structure but still no temporal variation
    # ANNs with their CNN architecture should excel while SNNs will struggle
    
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
        
        # Create a completely static pattern with extreme noise
        new_pattern = np.zeros((width*height, 100), dtype=np.float32)
        
        # Apply extreme noise (45%) - very challenging but still has clear spatial structure
        # Pattern remains static across time - the same pattern at all timesteps
        noise_mask = np.random.random(spatial_pattern.shape) < 0.45
        noisy_pattern = spatial_pattern.copy()
        noisy_pattern[noise_mask] = 1 - noisy_pattern[noise_mask]
        
        # Apply DIFFERENT noise patterns at each timestep to prevent SNN's temporal integration advantage
        for t in range(10, 90):
            # Apply fresh random noise to each timestep independently
            timestep_noise_mask = np.random.random(spatial_pattern.shape) < 0.45
            timestep_noisy_pattern = spatial_pattern.copy()
            timestep_noisy_pattern[timestep_noise_mask] = 1 - timestep_noisy_pattern[timestep_noise_mask]
            new_pattern[:, t] = timestep_noisy_pattern
        
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


def generate_ann_optimized_datasets():
    """
    Generate datasets that specifically showcase ANN's strengths over SNNs.
    These datasets focus on complex spatial hierarchies that benefit from
    parallel processing but are difficult to process in a sequential manner.
    
    This is the first of three ANN-optimized datasets, used to replace the spatial datasets.
    """
    print("\n=== Generating ANN-Optimized Dataset 1: Complex Spatial Hierarchies ===")
    
    # Create the output directory
    os.makedirs('./data/ann_optimized/datasets', exist_ok=True)
    os.makedirs('./data/ann_optimized/examples', exist_ok=True)
    
    width, height = 16, 16  # Larger spatial dimensions than standard datasets
    length = 100  # Time dimension for compatibility
    
    # Dataset 1: Complex Spatial Hierarchies (multi-level features)
    # ANNs excel at hierarchical feature extraction through multiple layers
    print("Generating ANN-optimized dataset with complex spatial hierarchies...")
    
    hierarchical_patterns = np.zeros((400, width*height, length), dtype=np.float32)
    hierarchical_labels = np.zeros(400, dtype=np.int32)
    
    for sample_idx in range(400):
        # Determine class (4 classes)
        class_idx = sample_idx // 100
        
        # Create pattern base
        spatial_pattern = np.zeros((height, width), dtype=np.float32)
        
        if class_idx == 0:
            # Class 0: Multi-scale nested patterns
            # Create base grid
            for y in range(0, height, 4):
                for x in range(0, width, 4):
                    # Define a 4x4 cell
                    cell_type = (x + y) % 4
                    
                    if cell_type == 0:
                        # Solid block with small hole in middle
                        for dy in range(4):
                            for dx in range(4):
                                if not (dx == 1 and dy == 1):
                                    spatial_pattern[y + dy, x + dx] = 1.0
                    elif cell_type == 1:
                        # Checkered pattern (2x2)
                        for dy in range(4):
                            for dx in range(4):
                                if (dx // 2 + dy // 2) % 2 == 0:
                                    spatial_pattern[y + dy, x + dx] = 1.0
                    elif cell_type == 2:
                        # X pattern
                        for dy in range(4):
                            for dx in range(4):
                                if dx == dy or dx == 3-dy:
                                    spatial_pattern[y + dy, x + dx] = 1.0
                    elif cell_type == 3:
                        # Border only
                        for dy in range(4):
                            for dx in range(4):
                                if dx == 0 or dy == 0 or dx == 3 or dy == 3:
                                    spatial_pattern[y + dy, x + dx] = 1.0
        
        elif class_idx == 1:
            # Class 1: Directional gradients with embedded features
            # These are hard for SNNs to detect as they need to process the whole image at once
            
            # Create base gradient (different directions)
            gradient_dir = sample_idx % 4
            
            if gradient_dir == 0:  # Top to bottom
                for y in range(height):
                    val = y / (height - 1)
                    threshold = 0.2 + val * 0.6  # Vary from 0.2 to 0.8
                    for x in range(width):
                        if np.random.random() < threshold:
                            spatial_pattern[y, x] = 1.0
                            
            elif gradient_dir == 1:  # Left to right
                for x in range(width):
                    val = x / (width - 1)
                    threshold = 0.2 + val * 0.6  # Vary from 0.2 to 0.8
                    for y in range(height):
                        if np.random.random() < threshold:
                            spatial_pattern[y, x] = 1.0
                            
            elif gradient_dir == 2:  # Center to edges (radial)
                center_y, center_x = height // 2, width // 2
                max_dist = np.sqrt((height//2)**2 + (width//2)**2)
                
                for y in range(height):
                    for x in range(width):
                        dist = np.sqrt((y - center_y)**2 + (x - center_x)**2)
                        val = dist / max_dist
                        threshold = 0.8 - val * 0.6  # Higher probability near center
                        if np.random.random() < threshold:
                            spatial_pattern[y, x] = 1.0
                            
            elif gradient_dir == 3:  # Diagonal
                for y in range(height):
                    for x in range(width):
                        val = (x + y) / (width + height - 2)
                        threshold = 0.2 + val * 0.6
                        if np.random.random() < threshold:
                            spatial_pattern[y, x] = 1.0
            
            # Add a small embedded shape that's hard to detect without global context
            shape_type = (sample_idx // 4) % 3
            
            # Position varies slightly
            pos_x = width // 4 + (sample_idx % 3)
            pos_y = height // 4 + ((sample_idx // 3) % 3)
            
            if shape_type == 0:  # Small square
                size = 3
                for dy in range(size):
                    for dx in range(size):
                        spatial_pattern[pos_y + dy, pos_x + dx] = 1.0
                        
            elif shape_type == 1:  # Small circle
                radius = 2
                for dy in range(-2, 3):
                    for dx in range(-2, 3):
                        if dx**2 + dy**2 <= radius**2:
                            y, x = pos_y + dy, pos_x + dx
                            if 0 <= y < height and 0 <= x < width:
                                spatial_pattern[y, x] = 1.0
                                
            elif shape_type == 2:  # Small cross
                for offset in range(-2, 3):
                    # Horizontal line
                    if 0 <= pos_y < height and 0 <= pos_x + offset < width:
                        spatial_pattern[pos_y, pos_x + offset] = 1.0
                    # Vertical line
                    if 0 <= pos_y + offset < height and 0 <= pos_x < width:
                        spatial_pattern[pos_y + offset, pos_x] = 1.0
        
        elif class_idx == 2:
            # Class 2: Long-range spatial dependencies
            # These are tough for SNNs because neurons that need to coordinate are far apart
            
            pattern_type = sample_idx % 5
            
            if pattern_type == 0:  # Symmetric patterns across the image
                # Create a random pattern in the left half
                for y in range(height):
                    for x in range(width // 2):
                        if np.random.random() < 0.3:
                            spatial_pattern[y, x] = 1.0
                            # Mirror to right half - creates symmetry
                            spatial_pattern[y, width - 1 - x] = 1.0
                            
            elif pattern_type == 1:  # Disconnected but related patterns
                # Create two related patterns in opposite corners
                
                # Top-left pattern: Square
                size = 4 + (sample_idx % 3)
                for y in range(size):
                    for x in range(size):
                        spatial_pattern[y, x] = 1.0
                
                # Bottom-right pattern: Same size square
                for y in range(size):
                    for x in range(size):
                        spatial_pattern[height - 1 - y, width - 1 - x] = 1.0
                
            elif pattern_type == 2:  # Interlocking patterns
                # Create interlocking grid patterns
                for y in range(height):
                    for x in range(width):
                        # Create complex grid with varying densities
                        if (x % 4 == 0 and y % 4 < 2) or (y % 4 == 0 and x % 4 < 2):
                            spatial_pattern[y, x] = 1.0
                            
            elif pattern_type == 3:  # Periodic patterns with phase relationships
                # Create horizontal and vertical stripes with phase relationship
                phase = sample_idx % 4
                
                for y in range(height):
                    if y % 4 == phase:
                        for x in range(width):
                            spatial_pattern[y, x] = 1.0
                            
                for x in range(width):
                    if x % 4 == (phase + 2) % 4:
                        for y in range(height):
                            spatial_pattern[y, x] = 1.0
                            
            elif pattern_type == 4:  # Fractal-like recursive patterns
                # Create a pseudo-fractal pattern (simple approximation)
                # Divide the image into quadrants and apply patterns recursively
                
                def apply_recursive_pattern(start_y, start_x, size):
                    if size <= 1:
                        return
                    
                    # Fill the border
                    for i in range(size):
                        spatial_pattern[start_y, start_x + i] = 1.0  # Top
                        spatial_pattern[start_y + size - 1, start_x + i] = 1.0  # Bottom
                        spatial_pattern[start_y + i, start_x] = 1.0  # Left
                        spatial_pattern[start_y + i, start_x + size - 1] = 1.0  # Right
                    
                    # Recursively apply to subquadrants
                    if size >= 4:
                        new_size = size // 2
                        apply_recursive_pattern(start_y + new_size // 2, start_x + new_size // 2, new_size)
                
                # Start with the whole image
                apply_recursive_pattern(0, 0, 16)
        
        elif class_idx == 3:
            # Class 3: Global structure with fine details
            # ANNs can capture both global structure and local details at once
            
            # Create a base global structure
            global_type = sample_idx % 4
            
            if global_type == 0:  # Concentric rings
                center_y, center_x = height // 2, width // 2
                
                for radius in range(1, 8, 2):  # Every other radius
                    for y in range(height):
                        for x in range(width):
                            dist = np.sqrt((y - center_y)**2 + (x - center_x)**2)
                            if abs(dist - radius) < 0.8:
                                spatial_pattern[y, x] = 1.0
                
                # Add small textural details in each quadrant
                quadrant_detail = (sample_idx // 4) % 4
                
                if quadrant_detail == 0:  # Top-left: dots
                    for y in range(2, height//2 - 2, 2):
                        for x in range(2, width//2 - 2, 2):
                            spatial_pattern[y, x] = 1.0
                            
                elif quadrant_detail == 1:  # Top-right: horizontal lines
                    for y in range(2, height//2 - 2, 2):
                        for x in range(width//2 + 2, width - 2):
                            spatial_pattern[y, x] = 1.0
                            
                elif quadrant_detail == 2:  # Bottom-left: vertical lines
                    for y in range(height//2 + 2, height - 2):
                        for x in range(2, width//2 - 2, 2):
                            spatial_pattern[y, x] = 1.0
                            
                elif quadrant_detail == 3:  # Bottom-right: diagonal lines
                    for offset in range(2, min(height, width)//2 - 4, 2):
                        y = height//2 + offset
                        x = width//2 + offset
                        if y < height and x < width:
                            spatial_pattern[y, x] = 1.0
                            
            elif global_type == 1:  # Grid with variable density
                # Create base grid
                for y in range(0, height, 4):
                    for x in range(0, width, 4):
                        spatial_pattern[y, x] = 1.0
                
                # Add density variation in different regions
                region_type = (sample_idx // 4) % 4
                
                if region_type == 0:  # Denser in top-left
                    for y in range(0, height//2, 2):
                        for x in range(0, width//2, 2):
                            if not (y % 4 == 0 and x % 4 == 0):  # Skip existing grid points
                                spatial_pattern[y, x] = 1.0
                                
                elif region_type == 1:  # Denser in top-right
                    for y in range(0, height//2, 2):
                        for x in range(width//2, width, 2):
                            if not (y % 4 == 0 and x % 4 == 0):
                                spatial_pattern[y, x] = 1.0
                                
                elif region_type == 2:  # Denser in bottom-left
                    for y in range(height//2, height, 2):
                        for x in range(0, width//2, 2):
                            if not (y % 4 == 0 and x % 4 == 0):
                                spatial_pattern[y, x] = 1.0
                                
                elif region_type == 3:  # Denser in bottom-right
                    for y in range(height//2, height, 2):
                        for x in range(width//2, width, 2):
                            if not (y % 4 == 0 and x % 4 == 0):
                                spatial_pattern[y, x] = 1.0
                                
            elif global_type == 2:  # Radial density gradient with pattern overlay
                center_y, center_x = height // 2, width // 2
                max_dist = np.sqrt((height//2)**2 + (width//2)**2)
                
                # Base radial density gradient
                for y in range(height):
                    for x in range(width):
                        dist = np.sqrt((y - center_y)**2 + (x - center_x)**2)
                        prob = 0.8 - (dist / max_dist) * 0.6  # Higher near center
                        if np.random.random() < prob:
                            spatial_pattern[y, x] = 1.0
                
                # Add overlay pattern
                overlay_type = (sample_idx // 4) % 3
                
                if overlay_type == 0:  # Cross overlay
                    # Horizontal line
                    for x in range(width):
                        spatial_pattern[center_y, x] = 1.0
                    # Vertical line
                    for y in range(height):
                        spatial_pattern[y, center_x] = 1.0
                        
                elif overlay_type == 1:  # Diamond overlay
                    size = 6
                    for y in range(height):
                        for x in range(width):
                            if abs(y - center_y) + abs(x - center_x) == size:
                                spatial_pattern[y, x] = 1.0
                                
                elif overlay_type == 2:  # Spiral overlay (simplified)
                    for angle in range(0, 360, 10):
                        for r in range(1, 8):
                            rad = np.radians(angle + r * 5)
                            y = int(center_y + r * np.sin(rad))
                            x = int(center_x + r * np.cos(rad))
                            if 0 <= y < height and 0 <= x < width:
                                spatial_pattern[y, x] = 1.0
                                
            elif global_type == 3:  # Multi-level hierarchical pattern
                # Create a border
                for y in range(height):
                    for x in range(width):
                        if y == 0 or y == height-1 or x == 0 or x == width-1:
                            spatial_pattern[y, x] = 1.0
                
                # Add checker pattern in center
                center_size = 8
                start_y, start_x = (height - center_size) // 2, (width - center_size) // 2
                
                for y in range(start_y, start_y + center_size):
                    for x in range(start_x, start_x + center_size):
                        if (y - start_y) % 2 == (x - start_x) % 2:
                            spatial_pattern[y, x] = 1.0
                
                # Add small details in corners
                detail_type = (sample_idx // 4) % 4
                size = 3
                
                if detail_type == 0 or detail_type == 1:  # Top corners
                    # Top-left
                    for y in range(2, 2+size):
                        for x in range(2, 2+size):
                            if detail_type == 0:
                                spatial_pattern[y, x] = 1.0  # Solid square
                            else:
                                if y == 2 or y == 2+size-1 or x == 2 or x == 2+size-1:
                                    spatial_pattern[y, x] = 1.0  # Hollow square
                    
                    # Top-right
                    for y in range(2, 2+size):
                        for x in range(width-2-size, width-2):
                            if detail_type == 0:
                                if y - 2 == x - (width-2-size):
                                    spatial_pattern[y, x] = 1.0  # Diagonal
                            else:
                                if (y - 2) % 2 == (x - (width-2-size)) % 2:
                                    spatial_pattern[y, x] = 1.0  # Checker
                                    
                elif detail_type == 2 or detail_type == 3:  # Bottom corners
                    # Bottom-left
                    for y in range(height-2-size, height-2):
                        for x in range(2, 2+size):
                            if detail_type == 2:
                                if (y - (height-2-size)) % 2 == 0:
                                    spatial_pattern[y, x] = 1.0  # Horizontal stripes
                            else:
                                if (x - 2) % 2 == 0:
                                    spatial_pattern[y, x] = 1.0  # Vertical stripes
                    
                    # Bottom-right
                    for y in range(height-2-size, height-2):
                        for x in range(width-2-size, width-2):
                            if detail_type == 2:
                                if (y - (height-2-size)) + (x - (width-2-size)) == size-1:
                                    spatial_pattern[y, x] = 1.0  # Anti-diagonal
                            else:
                                spatial_pattern[y, x] = 1.0  # Filled
        
        # Create a completely new pattern at each timestep by adding noise
        # This FORCES the network to rely on spatial pattern recognition
        # and prevents temporal integration advantages
        pattern_flat = spatial_pattern.reshape(-1)
        
        # Create a pattern for every timestep with different noise
        new_pattern = np.zeros((width*height, length), dtype=np.float32)
        
        for t in range(10, 90):
            # Apply significant but not overwhelming noise
            noise_level = 0.25  # 25% noise
            noise_mask = np.random.random(pattern_flat.shape) < noise_level
            noisy_pattern = pattern_flat.copy()
            noisy_pattern[noise_mask] = 1 - noisy_pattern[noise_mask]
            new_pattern[:, t] = noisy_pattern
        
        # Store pattern and label
        hierarchical_patterns[sample_idx] = new_pattern
        hierarchical_labels[sample_idx] = class_idx
    
    # Save this dataset
    save_path = os.path.join('./data/ann_optimized/datasets', 'ann_optimized_hierarchical.npz')
    np.savez(save_path, patterns=hierarchical_patterns, labels=hierarchical_labels)
    print(f"Saved dataset to {save_path}")
    
    # Create a visualization of examples
    fig, axes = plt.subplots(4, 2, figsize=(10, 12))
    
    for i in range(4):
        # Find samples for this class
        class_indices = np.where(hierarchical_labels == i)[0]
        
        # Get two sample indices for this class
        if len(class_indices) > 0:
            idx1 = class_indices[0]
            
            # For spatial data, reshape to 2D and show the pattern at a specific timestep
            pattern = hierarchical_patterns[idx1]
            sample_t = 50  # Choose a timestep in the middle
            spatial_pattern = pattern[:, sample_t].reshape(16, 16)  # Reshape to 16x16 grid
            
            # Plot as heatmap
            axes[i, 0].imshow(spatial_pattern, cmap='viridis', interpolation='nearest')
            axes[i, 0].set_title(f"Class {i} - Sample 1")
            axes[i, 0].axis('off')
            
            # Select another sample for the same class
            if len(class_indices) > 1:
                idx2 = class_indices[len(class_indices) // 2]  # Get a sample from middle of the set
                pattern = hierarchical_patterns[idx2]
                spatial_pattern = pattern[:, sample_t].reshape(16, 16)
                
                axes[i, 1].imshow(spatial_pattern, cmap='viridis', interpolation='nearest')
                axes[i, 1].set_title(f"Class {i} - Sample 2")
                axes[i, 1].axis('off')
    
    plt.tight_layout()
    example_path = os.path.join('./data/ann_optimized/examples', 'ann_optimized_hierarchical_examples.png')
    plt.savefig(example_path, dpi=150)
    print(f"Saved visualization to {example_path}")
    
    # Return the dataset
    return hierarchical_patterns, hierarchical_labels


def generate_ann_optimized_datasets2():
    """
    Generate datasets that specifically showcase ANN's strengths over SNNs.
    This second ANN-optimized dataset focuses on global-local integration,
    which is another area where ANNs have a significant advantage over SNNs.
    """
    print("\n=== Generating ANN-Optimized Dataset 2: Global-Local Integration ===")
    
    # Create the output directory
    os.makedirs('./data/ann_optimized/datasets', exist_ok=True)
    os.makedirs('./data/ann_optimized/examples', exist_ok=True)
    
    width, height = 16, 16  # Larger spatial dimensions
    length = 100  # Time dimension for compatibility
    
    # Dataset 2: Global-Local Integration (patterns requiring global context)
    # ANNs excel at integrating global structure with local details simultaneously
    print("Generating ANN-optimized dataset with global-local integration patterns...")
    
    global_local_patterns = np.zeros((400, width*height, length), dtype=np.float32)
    global_local_labels = np.zeros(400, dtype=np.int32)
    
    for sample_idx in range(400):
        # Determine class (4 classes)
        class_idx = sample_idx // 100
        
        # Create pattern base
        spatial_pattern = np.zeros((height, width), dtype=np.float32)
        
        if class_idx == 0:
            # Class 0: Center-surround patterns with long-range dependencies
            center_y, center_x = height // 2, width // 2
            
            # Create center pattern
            center_type = sample_idx % 4
            center_size = 4 + (sample_idx % 3)
            
            # Create surround pattern
            surround_type = (sample_idx // 4) % 4
            surround_width = 1 + (sample_idx % 2)
            
            # Fill center based on type
            for y in range(center_y - center_size//2, center_y + center_size//2):
                for x in range(center_x - center_size//2, center_x + center_size//2):
                    if 0 <= y < height and 0 <= x < width:
                        if center_type == 0:  # Solid
                            spatial_pattern[y, x] = 1.0
                        elif center_type == 1:  # Checkerboard
                            if (y + x) % 2 == 0:
                                spatial_pattern[y, x] = 1.0
                        elif center_type == 2:  # Cross
                            if y == center_y or x == center_x:
                                spatial_pattern[y, x] = 1.0
                        elif center_type == 3:  # Diamond
                            if abs(y - center_y) + abs(x - center_x) <= center_size//2:
                                spatial_pattern[y, x] = 1.0
            
            # Fill surround based on type
            for y in range(height):
                for x in range(width):
                    # Check if pixel is in the surround (outside center, inside boundary)
                    dist_to_center = max(abs(y - center_y), abs(x - center_x))
                    if center_size//2 < dist_to_center <= center_size//2 + surround_width:
                        if surround_type == 0:  # Solid surround
                            spatial_pattern[y, x] = 1.0
                        elif surround_type == 1:  # Dashed surround
                            if (y + x) % 3 == 0:
                                spatial_pattern[y, x] = 1.0
                        elif surround_type == 2:  # Gradient surround
                            prob = 0.8 - (dist_to_center - center_size//2) / surround_width * 0.6
                            if np.random.random() < prob:
                                spatial_pattern[y, x] = 1.0
                        elif surround_type == 3:  # Opposite to center
                            # If center is dense, make surround sparse and vice versa
                            if center_type in [0, 3]:  # Solid or diamond center
                                if (y + x) % 3 != 0:
                                    spatial_pattern[y, x] = 1.0
                            else:  # Sparse center
                                spatial_pattern[y, x] = 1.0
        
        elif class_idx == 1:
            # Class 1: Context-dependent features
            # The same local feature means different things depending on global context
            context_type = sample_idx % 4
            
            # Create global context
            if context_type == 0:  # Horizontal gradient
                for y in range(height):
                    for x in range(width):
                        if np.random.random() < x / width * 0.6:
                            spatial_pattern[y, x] = 1.0
            elif context_type == 1:  # Vertical gradient
                for y in range(height):
                    for x in range(width):
                        if np.random.random() < y / height * 0.6:
                            spatial_pattern[y, x] = 1.0
            elif context_type == 2:  # Radial gradient
                center_y, center_x = height // 2, width // 2
                max_dist = np.sqrt((height//2)**2 + (width//2)**2)
                for y in range(height):
                    for x in range(width):
                        dist = np.sqrt((y - center_y)**2 + (x - center_x)**2)
                        if np.random.random() < dist / max_dist * 0.7:
                            spatial_pattern[y, x] = 1.0
            elif context_type == 3:  # Quadrant gradient
                for y in range(height):
                    for x in range(width):
                        quadrant = (y < height//2) * 2 + (x < width//2)
                        if np.random.random() < quadrant / 4 * 0.8:
                            spatial_pattern[y, x] = 1.0
            
            # Add local features in different positions
            feature_count = 3 + (sample_idx % 3)
            feature_type = (sample_idx // 4) % 3
            feature_size = 2 + (sample_idx % 2)
            
            for _ in range(feature_count):
                # Random position, avoiding edges
                pos_y = np.random.randint(feature_size, height - feature_size)
                pos_x = np.random.randint(feature_size, width - feature_size)
                
                # Add feature based on type
                if feature_type == 0:  # Square feature
                    for dy in range(-feature_size//2, feature_size//2 + 1):
                        for dx in range(-feature_size//2, feature_size//2 + 1):
                            spatial_pattern[pos_y + dy, pos_x + dx] = 1.0
                
                elif feature_type == 1:  # Cross feature
                    for dy in range(-feature_size//2, feature_size//2 + 1):
                        spatial_pattern[pos_y + dy, pos_x] = 1.0
                    for dx in range(-feature_size//2, feature_size//2 + 1):
                        spatial_pattern[pos_y, pos_x + dx] = 1.0
                
                elif feature_type == 2:  # Diagonal feature
                    for offset in range(-feature_size//2, feature_size//2 + 1):
                        spatial_pattern[pos_y + offset, pos_x + offset] = 1.0
                        spatial_pattern[pos_y + offset, pos_x - offset] = 1.0
        
        elif class_idx == 2:
            # Class 2: Multi-scale feature integration
            # Patterns where both fine and coarse details matter for classification
            base_scale = sample_idx % 4
            
            # Create base coarse pattern
            if base_scale == 0:  # Coarse grid
                grid_size = 4
                for y in range(0, height, grid_size):
                    for x in range(0, width, grid_size):
                        for dy in range(min(grid_size, height - y)):
                            for dx in range(min(grid_size, width - x)):
                                if (y // grid_size + x // grid_size) % 2 == 0:
                                    spatial_pattern[y + dy, x + dx] = 1.0
            
            elif base_scale == 1:  # Coarse diagonal stripes
                stripe_width = 4
                for y in range(height):
                    for x in range(width):
                        if ((y + x) // stripe_width) % 2 == 0:
                            spatial_pattern[y, x] = 1.0
            
            elif base_scale == 2:  # Coarse concentric patterns
                center_y, center_x = height // 2, width // 2
                for y in range(height):
                    for x in range(width):
                        dist = np.sqrt((y - center_y)**2 + (x - center_x)**2)
                        if int(dist / 3) % 2 == 0:
                            spatial_pattern[y, x] = 1.0
            
            elif base_scale == 3:  # Coarse quadrants
                for y in range(height):
                    for x in range(width):
                        quadrant = (y >= height//2) * 2 + (x >= width//2)
                        if quadrant % 2 == 0:
                            spatial_pattern[y, x] = 1.0
            
            # Add fine-scale details
            fine_scale = (sample_idx // 4) % 4
            
            if fine_scale == 0:  # Fine dots overlay
                for y in range(height):
                    for x in range(width):
                        if (y % 3 == 0) and (x % 3 == 0):
                            spatial_pattern[y, x] = 1 - spatial_pattern[y, x]  # Invert
            
            elif fine_scale == 1:  # Fine checkerboard overlay in specific regions
                region_y = np.random.randint(0, height - 6)
                region_x = np.random.randint(0, width - 6)
                for y in range(region_y, region_y + 6):
                    for x in range(region_x, region_x + 6):
                        if (y + x) % 2 == 0:
                            spatial_pattern[y, x] = 1 - spatial_pattern[y, x]
            
            elif fine_scale == 2:  # Fine lines at specific angles
                angle = np.random.randint(0, 4) * 45  # 0, 45, 90, or 135 degrees
                for y in range(height):
                    for x in range(width):
                        if angle == 0 and y % 3 == 0:  # Horizontal
                            spatial_pattern[y, x] = 1 - spatial_pattern[y, x]
                        elif angle == 90 and x % 3 == 0:  # Vertical
                            spatial_pattern[y, x] = 1 - spatial_pattern[y, x]
                        elif angle == 45 and (y - x) % 3 == 0:  # Diagonal 45
                            spatial_pattern[y, x] = 1 - spatial_pattern[y, x]
                        elif angle == 135 and (y + x) % 3 == 0:  # Diagonal 135
                            spatial_pattern[y, x] = 1 - spatial_pattern[y, x]
            
            elif fine_scale == 3:  # Fine detail on borders
                for y in range(height):
                    for x in range(width):
                        edge_dist = min(min(y, height-1-y), min(x, width-1-x))
                        if edge_dist < 2 and (y + x) % 2 == 0:
                            spatial_pattern[y, x] = 1 - spatial_pattern[y, x]
        
        elif class_idx == 3:
            # Class 3: Long-range spatial correlations
            # Features that can only be detected by looking at the entire image
            correlation_type = sample_idx % 5
            
            if correlation_type == 0:  # Symmetric patterns
                # Create left half randomly
                for y in range(height):
                    for x in range(width // 2):
                        if np.random.random() < 0.3:
                            spatial_pattern[y, x] = 1.0
                            # Mirror to right half
                            spatial_pattern[y, width - 1 - x] = 1.0
            
            elif correlation_type == 1:  # Balanced proportions
                # The total number of active pixels is always ~25% of the image
                total_pixels = width * height
                target_active = total_pixels // 4
                count_active = 0
                
                while count_active < target_active:
                    y = np.random.randint(0, height)
                    x = np.random.randint(0, width)
                    if spatial_pattern[y, x] == 0:
                        spatial_pattern[y, x] = 1.0
                        count_active += 1
            
            elif correlation_type == 2:  # Cross-quadrant correlation
                # Create patterns where opposite quadrants are correlated
                pattern_a = np.random.random((height//2, width//2)) < 0.3
                pattern_b = np.random.random((height//2, width//2)) < 0.3
                
                # Top-left and bottom-right share pattern_a
                spatial_pattern[:height//2, :width//2] = pattern_a
                spatial_pattern[height//2:, width//2:] = pattern_a
                
                # Top-right and bottom-left share pattern_b
                spatial_pattern[:height//2, width//2:] = pattern_b
                spatial_pattern[height//2:, :width//2] = pattern_b
            
            elif correlation_type == 3:  # Rotational symmetry
                # Create rotationally symmetric patterns (90 degree rotations)
                for y in range(height//2):
                    for x in range(width//2):
                        if np.random.random() < 0.3:
                            # Fill all four rotational positions
                            spatial_pattern[y, x] = 1.0
                            spatial_pattern[x, height-1-y] = 1.0
                            spatial_pattern[height-1-y, width-1-x] = 1.0
                            spatial_pattern[width-1-x, y] = 1.0
            
            elif correlation_type == 4:  # Long-distance feature pairs
                # Create pairs of matching features at opposite sides of the image
                for _ in range(5):  # Create 5 feature pairs
                    feature_type = np.random.randint(0, 3)
                    
                    # Choose two opposite regions
                    region1_y = np.random.randint(0, height//3)
                    region1_x = np.random.randint(0, width//3)
                    region2_y = height - 1 - region1_y - 2
                    region2_x = width - 1 - region1_x - 2
                    
                    # Create the same feature in both regions
                    if feature_type == 0:  # Small square
                        for dy in range(3):
                            for dx in range(3):
                                spatial_pattern[region1_y + dy, region1_x + dx] = 1.0
                                spatial_pattern[region2_y + dy, region2_x + dx] = 1.0
                    
                    elif feature_type == 1:  # Small cross
                        for offset in range(3):
                            spatial_pattern[region1_y + offset, region1_x + 1] = 1.0
                            spatial_pattern[region1_y + 1, region1_x + offset] = 1.0
                            spatial_pattern[region2_y + offset, region2_x + 1] = 1.0
                            spatial_pattern[region2_y + 1, region2_x + offset] = 1.0
                    
                    elif feature_type == 2:  # Small diagonal
                        for offset in range(3):
                            spatial_pattern[region1_y + offset, region1_x + offset] = 1.0
                            spatial_pattern[region2_y + offset, region2_x + offset] = 1.0
        
        # Create a completely new pattern at each timestep by adding noise
        # This FORCES the network to rely on spatial pattern recognition
        # and prevents temporal integration advantages
        pattern_flat = spatial_pattern.reshape(-1)
        
        # Create a pattern for every timestep with different noise
        new_pattern = np.zeros((width*height, length), dtype=np.float32)
        
        for t in range(10, 90):
            # Apply significant but not overwhelming noise
            noise_level = 0.25  # 25% noise
            noise_mask = np.random.random(pattern_flat.shape) < noise_level
            noisy_pattern = pattern_flat.copy()
            noisy_pattern[noise_mask] = 1 - noisy_pattern[noise_mask]
            new_pattern[:, t] = noisy_pattern
        
        # Store pattern and label
        global_local_patterns[sample_idx] = new_pattern
        global_local_labels[sample_idx] = class_idx
    
    # Save this dataset
    save_path = os.path.join('./data/ann_optimized/datasets', 'ann_optimized_global_local.npz')
    np.savez(save_path, patterns=global_local_patterns, labels=global_local_labels)
    print(f"Saved dataset to {save_path}")
    
    # Create a visualization of examples
    fig, axes = plt.subplots(4, 2, figsize=(10, 12))
    
    for i in range(4):
        # Find samples for this class
        class_indices = np.where(global_local_labels == i)[0]
        
        # Get two sample indices for this class
        if len(class_indices) > 0:
            idx1 = class_indices[0]
            
            # For spatial data, reshape to 2D and show the pattern at a specific timestep
            pattern = global_local_patterns[idx1]
            sample_t = 50  # Choose a timestep in the middle
            spatial_pattern = pattern[:, sample_t].reshape(16, 16)  # Reshape to 16x16 grid
            
            # Plot as heatmap
            axes[i, 0].imshow(spatial_pattern, cmap='viridis', interpolation='nearest')
            axes[i, 0].set_title(f"Class {i} - Sample 1")
            axes[i, 0].axis('off')
            
            # Select another sample for the same class
            if len(class_indices) > 1:
                idx2 = class_indices[len(class_indices) // 2]  # Get a sample from middle of the set
                pattern = global_local_patterns[idx2]
                spatial_pattern = pattern[:, sample_t].reshape(16, 16)
                
                axes[i, 1].imshow(spatial_pattern, cmap='viridis', interpolation='nearest')
                axes[i, 1].set_title(f"Class {i} - Sample 2")
                axes[i, 1].axis('off')
    
    plt.tight_layout()
    example_path = os.path.join('./data/ann_optimized/examples', 'ann_optimized_global_local_examples.png')
    plt.savefig(example_path, dpi=150)
    print(f"Saved visualization to {example_path}")
    
    return global_local_patterns, global_local_labels


def generate_ann_optimized_datasets3():
    """
    Generate datasets that specifically showcase ANN's strengths over SNNs.
    This third ANN-optimized dataset focuses on multi-resolution features,
    where ANNs can effectively process information at multiple scales simultaneously.
    """
    print("\n=== Generating ANN-Optimized Dataset 3: Multi-Resolution Features ===")
    
    # Create the output directory
    os.makedirs('./data/ann_optimized/datasets', exist_ok=True)
    os.makedirs('./data/ann_optimized/examples', exist_ok=True)
    
    width, height = 16, 16  # Larger spatial dimensions
    length = 100  # Time dimension for compatibility
    
    # Dataset 3: Multi-Resolution Features
    # ANNs excel at processing information at multiple scales simultaneously
    print("Generating ANN-optimized dataset with multi-resolution features...")
    
    multiresolution_patterns = np.zeros((400, width*height, length), dtype=np.float32)
    multiresolution_labels = np.zeros(400, dtype=np.int32)
    
    for sample_idx in range(400):
        # Determine class (4 classes)
        class_idx = sample_idx // 100
        
        # Create pattern base
        spatial_pattern = np.zeros((height, width), dtype=np.float32)
        
        if class_idx == 0:
            # Class 0: Fractal-like patterns
            # Create recursive subdivision patterns where details at multiple scales matter
            scale_type = sample_idx % 4
            
            if scale_type == 0:  # Sierpinski-like pattern
                # Start with a filled square
                spatial_pattern.fill(1.0)
                
                # Recursively remove central squares
                def remove_center(start_y, start_x, size):
                    if size < 2:
                        return
                    
                    # Remove the center square
                    center_size = size // 3
                    center_y = start_y + center_size
                    center_x = start_x + center_size
                    
                    for y in range(center_y, center_y + center_size):
                        for x in range(center_x, center_x + center_size):
                            if 0 <= y < height and 0 <= x < width:
                                spatial_pattern[y, x] = 0.0
                    
                    # Recursively process the 8 surrounding squares
                    new_size = size // 3
                    for dy in range(3):
                        for dx in range(3):
                            if not (dy == 1 and dx == 1):  # Skip the center we just removed
                                new_y = start_y + dy * new_size
                                new_x = start_x + dx * new_size
                                remove_center(new_y, new_x, new_size)
                
                # Start the recursion with the whole image
                remove_center(0, 0, 16)
                
            elif scale_type == 1:  # Nested squares
                max_depth = 4
                for depth in range(max_depth):
                    size = 16 // (2**depth)
                    offset = (16 - size) // 2
                    
                    # Draw a square border at this level
                    if depth % 2 == 0:  # Alternate between filled and unfilled
                        for y in range(offset, offset + size):
                            for x in range(offset, offset + size):
                                if (y == offset or y == offset + size - 1 or 
                                    x == offset or x == offset + size - 1):
                                    spatial_pattern[y, x] = 1.0
            
            elif scale_type == 2:  # Quadtree-like subdivision
                def subdivide(start_y, start_x, size, depth):
                    if depth >= 4 or size <= 1:
                        return
                    
                    # Probability of subdivision decreases with depth
                    if np.random.random() < 0.8 - depth * 0.2:
                        # Subdivide this block into 4 quadrants
                        half_size = size // 2
                        
                        # Recursively process each quadrant
                        subdivide(start_y, start_x, half_size, depth + 1)
                        subdivide(start_y, start_x + half_size, half_size, depth + 1)
                        subdivide(start_y + half_size, start_x, half_size, depth + 1)
                        subdivide(start_y + half_size, start_x + half_size, half_size, depth + 1)
                        
                        # Draw dividing lines
                        for i in range(size):
                            # Horizontal division
                            spatial_pattern[start_y + half_size, start_x + i] = 1.0
                            # Vertical division
                            spatial_pattern[start_y + i, start_x + half_size] = 1.0
                    else:
                        # Fill this block with a pattern based on position
                        fill_type = (start_y + start_x) % 4
                        
                        for y in range(start_y, start_y + size):
                            for x in range(start_x, start_x + size):
                                if fill_type == 0 and (y - start_y + x - start_x) % 3 == 0:
                                    spatial_pattern[y, x] = 1.0
                                elif fill_type == 1 and (y - start_y) % 2 == 0:
                                    spatial_pattern[y, x] = 1.0
                                elif fill_type == 2 and (x - start_x) % 2 == 0:
                                    spatial_pattern[y, x] = 1.0
                                elif fill_type == 3 and (y - start_y) % 2 == (x - start_x) % 2:
                                    spatial_pattern[y, x] = 1.0
                
                # Start the subdivision with the whole image
                subdivide(0, 0, 16, 0)
            
            elif scale_type == 3:  # Multi-scale gradient patterns
                center_y, center_x = height // 2, width // 2
                
                for y in range(height):
                    for x in range(width):
                        # Large-scale radial gradient
                        dist = np.sqrt((y - center_y)**2 + (x - center_x)**2)
                        large_scale = np.sin(dist / 3) + 1  # Value between 0 and 2
                        
                        # Medium-scale gradient
                        medium_scale = np.sin(y / 2) * np.cos(x / 2) + 1  # Value between 0 and 2
                        
                        # Small-scale pattern
                        small_scale = 0.5 if (y % 3 == 0 or x % 3 == 0) else 0
                        
                        # Combine scales with decreasing influence
                        combined = (large_scale * 0.5 + medium_scale * 0.3 + small_scale * 0.2) / (0.5 + 0.3 + 0.2)
                        
                        if np.random.random() < combined * 0.4:  # Scale to reasonable density
                            spatial_pattern[y, x] = 1.0
        
        elif class_idx == 1:
            # Class 1: Texture-boundary patterns
            # Create patterns where texture boundaries are the key distinguishing features
            boundary_type = sample_idx % 5
            
            # First, create base textures for different regions
            texture1 = np.zeros((height, width))
            texture2 = np.zeros((height, width))
            
            # Texture 1: diagonal lines
            for y in range(height):
                for x in range(width):
                    if (y + x) % 3 == 0:
                        texture1[y, x] = 1.0
            
            # Texture 2: checker pattern
            for y in range(height):
                for x in range(width):
                    if (y % 2 == 0) != (x % 2 == 0):
                        texture2[y, x] = 1.0
            
            # Apply boundary based on type
            if boundary_type == 0:  # Vertical split
                split_point = width // 2
                for y in range(height):
                    for x in range(width):
                        if x < split_point:
                            spatial_pattern[y, x] = texture1[y, x]
                        else:
                            spatial_pattern[y, x] = texture2[y, x]
            
            elif boundary_type == 1:  # Horizontal split
                split_point = height // 2
                for y in range(height):
                    for x in range(width):
                        if y < split_point:
                            spatial_pattern[y, x] = texture1[y, x]
                        else:
                            spatial_pattern[y, x] = texture2[y, x]
            
            elif boundary_type == 2:  # Diagonal split
                for y in range(height):
                    for x in range(width):
                        if y > x:
                            spatial_pattern[y, x] = texture1[y, x]
                        else:
                            spatial_pattern[y, x] = texture2[y, x]
            
            elif boundary_type == 3:  # Circular split
                center_y, center_x = height // 2, width // 2
                radius = min(height, width) // 3
                
                for y in range(height):
                    for x in range(width):
                        dist = np.sqrt((y - center_y)**2 + (x - center_x)**2)
                        if dist < radius:
                            spatial_pattern[y, x] = texture1[y, x]
                        else:
                            spatial_pattern[y, x] = texture2[y, x]
            
            elif boundary_type == 4:  # Complex boundary
                # Create a wavy boundary
                for y in range(height):
                    # Compute a wavy x-position for the boundary
                    wave_x = width // 2 + int(np.sin(y / 3) * 3)
                    
                    for x in range(width):
                        if x < wave_x:
                            spatial_pattern[y, x] = texture1[y, x]
                        else:
                            spatial_pattern[y, x] = texture2[y, x]
        
        elif class_idx == 2:
            # Class 2: Spatial frequency patterns
            # Create patterns with different spatial frequencies in different regions
            freq_type = sample_idx % 4
            
            if freq_type == 0:  # Frequency gradient left to right
                for y in range(height):
                    for x in range(width):
                        # Frequency increases from left to right
                        local_freq = 1 + int(x / width * 5)
                        if (y % local_freq == 0) and (x % local_freq == 0):
                            spatial_pattern[y, x] = 1.0
            
            elif freq_type == 1:  # Frequency gradient top to bottom
                for y in range(height):
                    for x in range(width):
                        # Frequency increases from top to bottom
                        local_freq = 1 + int(y / height * 5)
                        if (y % local_freq == 0) and (x % local_freq == 0):
                            spatial_pattern[y, x] = 1.0
            
            elif freq_type == 2:  # Radial frequency gradient
                center_y, center_x = height // 2, width // 2
                max_dist = np.sqrt((height//2)**2 + (width//2)**2)
                
                for y in range(height):
                    for x in range(width):
                        dist = np.sqrt((y - center_y)**2 + (x - center_x)**2)
                        # Frequency increases with distance from center
                        local_freq = 1 + int(dist / max_dist * 5)
                        if (y % local_freq == 0) and (x % local_freq == 0):
                            spatial_pattern[y, x] = 1.0
            
            elif freq_type == 3:  # Quadrant-based frequencies
                for y in range(height):
                    for x in range(width):
                        quadrant = (y >= height//2) * 2 + (x >= width//2)
                        local_freq = 1 + quadrant
                        if (y % local_freq == 0) and (x % local_freq == 0):
                            spatial_pattern[y, x] = 1.0
        
        elif class_idx == 3:
            # Class 3: Composite multi-scale patterns
            # Create patterns that require both global and local feature extraction
            composite_type = sample_idx % 4
            
            if composite_type == 0:  # Global arrangement of local features
                # Create several local features with global positioning
                for region_idx in range(5):
                    # Position depends on global pattern
                    pos_y = 3 + (region_idx * 3) % (height - 6)
                    pos_x = 3 + (region_idx * 5) % (width - 6)
                    
                    # Local feature type depends on position
                    feature_type = (pos_y + pos_x) % 4
                    
                    if feature_type == 0:  # Small square
                        for dy in range(3):
                            for dx in range(3):
                                spatial_pattern[pos_y + dy, pos_x + dx] = 1.0
                    
                    elif feature_type == 1:  # Small cross
                        for offset in range(3):
                            spatial_pattern[pos_y + offset, pos_x + 1] = 1.0
                            spatial_pattern[pos_y + 1, pos_x + offset] = 1.0
                    
                    elif feature_type == 2:  # Small diamond
                        for dy in range(3):
                            for dx in range(3):
                                if abs(dy - 1) + abs(dx - 1) <= 1:
                                    spatial_pattern[pos_y + dy, pos_x + dx] = 1.0
                    
                    elif feature_type == 3:  # Small diagonal
                        for offset in range(3):
                            spatial_pattern[pos_y + offset, pos_x + offset] = 1.0
            
            elif composite_type == 1:  # Hierarchical patterns
                # Create a large-scale structure
                large_structure = sample_idx % 3
                
                if large_structure == 0:  # Large rectangle
                    start_y, start_x = height // 4, width // 4
                    size_y, size_x = height // 2, width // 2
                    
                    for y in range(start_y, start_y + size_y):
                        for x in range(start_x, start_x + size_x):
                            spatial_pattern[y, x] = 1.0
                
                elif large_structure == 1:  # Large cross
                    mid_y, mid_x = height // 2, width // 2
                    arm_width = 3
                    
                    for y in range(height):
                        for x in range(width):
                            if (abs(y - mid_y) < arm_width or abs(x - mid_x) < arm_width):
                                spatial_pattern[y, x] = 1.0
                
                elif large_structure == 2:  # Large diagonal
                    for y in range(height):
                        for x in range(width):
                            if abs(y - x) < 3 or abs(y - (width - 1 - x)) < 3:
                                spatial_pattern[y, x] = 1.0
                
                # Add medium-scale structures
                medium_count = 3 + (sample_idx % 3)
                for i in range(medium_count):
                    pos_y = 4 + (i * 3) % (height - 8)
                    pos_x = 4 + (i * 7) % (width - 8)
                    
                    # Cut out a medium-scale region
                    for dy in range(4):
                        for dx in range(4):
                            spatial_pattern[pos_y + dy, pos_x + dx] = 0.0
                
                # Add small-scale details
                for y in range(height):
                    for x in range(width):
                        if spatial_pattern[y, x] == 1.0 and (y % 3 == 0) and (x % 3 == 0):
                            # Create a small detail pattern
                            for dy in range(min(2, height - y - 1)):
                                for dx in range(min(2, width - x - 1)):
                                    spatial_pattern[y + dy, x + dx] = 0.0
            
            elif composite_type == 2:  # Multi-frequency composition
                # Combine three different spatial frequencies
                for y in range(height):
                    for x in range(width):
                        # Low frequency (large features)
                        low_freq = 8
                        low_val = 1.0 if ((y // low_freq + x // low_freq) % 2 == 0) else 0.0
                        
                        # Medium frequency
                        med_freq = 4
                        med_val = 1.0 if ((y // med_freq) % 2 == 0) != ((x // med_freq) % 2 == 0) else 0.0
                        
                        # High frequency (fine details)
                        high_freq = 2
                        high_val = 1.0 if (y % high_freq == 0) and (x % high_freq == 0) else 0.0
                        
                        # Combine frequencies with position-dependent weighting
                        rel_y, rel_x = y / height, x / width
                        weight_low = (1 - rel_y) * (1 - rel_x)
                        weight_med = rel_y * (1 - rel_x) + rel_x * (1 - rel_y)
                        weight_high = rel_y * rel_x
                        
                        total_weight = weight_low + weight_med + weight_high
                        combined_val = (low_val * weight_low + med_val * weight_med + high_val * weight_high) / total_weight
                        
                        if np.random.random() < combined_val:
                            spatial_pattern[y, x] = 1.0
            
            elif composite_type == 3:  # Spatially varying textures
                # Divide the image into regions with different textures
                for y in range(height):
                    for x in range(width):
                        region_y, region_x = y // 8, x // 8
                        local_y, local_x = y % 8, x % 8
                        
                        # Choose texture based on region
                        region_type = (region_y + region_x) % 4
                        
                        if region_type == 0:  # Fine grid
                            if (local_y % 2 == 0) != (local_x % 2 == 0):
                                spatial_pattern[y, x] = 1.0
                        
                        elif region_type == 1:  # Fine diagonal lines
                            if (local_y + local_x) % 3 == 0:
                                spatial_pattern[y, x] = 1.0
                        
                        elif region_type == 2:  # Fine dots
                            if (local_y % 3 == 0) and (local_x % 3 == 0):
                                spatial_pattern[y, x] = 1.0
                        
                        elif region_type == 3:  # Fine radial
                            local_center_y, local_center_x = 3.5, 3.5
                            dist = np.sqrt((local_y - local_center_y)**2 + (local_x - local_center_x)**2)
                            if abs(dist - 2.5) < 0.8:
                                spatial_pattern[y, x] = 1.0
        
        # Create a completely new pattern at each timestep by adding noise
        # This FORCES the network to rely on spatial pattern recognition
        # and prevents temporal integration advantages
        pattern_flat = spatial_pattern.reshape(-1)
        
        # Create a pattern for every timestep with different noise
        new_pattern = np.zeros((width*height, length), dtype=np.float32)
        
        for t in range(10, 90):
            # Apply significant but not overwhelming noise
            noise_level = 0.25  # 25% noise
            noise_mask = np.random.random(pattern_flat.shape) < noise_level
            noisy_pattern = pattern_flat.copy()
            noisy_pattern[noise_mask] = 1 - noisy_pattern[noise_mask]
            new_pattern[:, t] = noisy_pattern
        
        # Store pattern and label
        multiresolution_patterns[sample_idx] = new_pattern
        multiresolution_labels[sample_idx] = class_idx
    
    # Save this dataset
    save_path = os.path.join('./data/ann_optimized/datasets', 'ann_optimized_multiresolution.npz')
    np.savez(save_path, patterns=multiresolution_patterns, labels=multiresolution_labels)
    print(f"Saved dataset to {save_path}")
    
    # Create a visualization of examples
    fig, axes = plt.subplots(4, 2, figsize=(10, 12))
    
    for i in range(4):
        # Find samples for this class
        class_indices = np.where(multiresolution_labels == i)[0]
        
        # Get two sample indices for this class
        if len(class_indices) > 0:
            idx1 = class_indices[0]
            
            # For spatial data, reshape to 2D and show the pattern at a specific timestep
            pattern = multiresolution_patterns[idx1]
            sample_t = 50  # Choose a timestep in the middle
            spatial_pattern = pattern[:, sample_t].reshape(16, 16)  # Reshape to 16x16 grid
            
            # Plot as heatmap
            axes[i, 0].imshow(spatial_pattern, cmap='viridis', interpolation='nearest')
            axes[i, 0].set_title(f"Class {i} - Sample 1")
            axes[i, 0].axis('off')
            
            # Select another sample for the same class
            if len(class_indices) > 1:
                idx2 = class_indices[len(class_indices) // 2]  # Get a sample from middle of the set
                pattern = multiresolution_patterns[idx2]
                spatial_pattern = pattern[:, sample_t].reshape(16, 16)
                
                axes[i, 1].imshow(spatial_pattern, cmap='viridis', interpolation='nearest')
                axes[i, 1].set_title(f"Class {i} - Sample 2")
                axes[i, 1].axis('off')
    
    plt.tight_layout()
    example_path = os.path.join('./data/ann_optimized/examples', 'ann_optimized_multiresolution_examples.png')
    plt.savefig(example_path, dpi=150)
    print(f"Saved visualization to {example_path}")
    
    return multiresolution_patterns, multiresolution_labels

def generate_datasets():
    """
    Generate all difficulty tiered datasets for experiments
    
    Modified to generate 3 sets of ANN-optimized data and 3 sets of temporal data,
    replacing the original spatial datasets with ANN-optimized ones.
    """
    # Create the output directory
    os.makedirs('./data', exist_ok=True)
    
    # Generate temporal pattern datasets
    print("\n=== Generating Temporal Pattern Datasets ===")
    temporal_datasets = generate_tiered_temporal_patterns()
    
    # Generate ANN-optimized datasets (replaces all spatial datasets)
    print("\n=== Generating ANN-Optimized Pattern Datasets ===")
    # Generate first ANN-optimized dataset (hierarchical patterns - small)
    small_ann_optimized = generate_ann_optimized_datasets()
    
    # Generate two more ANN-optimized datasets using modified copies of the function
    medium_ann_optimized = generate_ann_optimized_datasets2()
    large_ann_optimized = generate_ann_optimized_datasets3()
    
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
    
    # Map ANN-optimized datasets to older expected names for spatial datasets
    spatial_mappings = [
        (os.path.join('./data/ann_optimized/datasets', 'ann_optimized_hierarchical.npz'), 'small_spatial_10class.npz'),
        (os.path.join('./data/ann_optimized/datasets', 'ann_optimized_global_local.npz'), 'medium_spatial_10class.npz'),
        (os.path.join('./data/ann_optimized/datasets', 'ann_optimized_multiresolution.npz'), 'large_spatial_10class.npz')
    ]
    
    # Create copies with legacy names
    os.makedirs('./data/spatial', exist_ok=True)
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
    
    print("\n  ANN-Optimized Patterns (Designed to favor ANN):")
    print("    - ./data/ann_optimized/datasets/ann_optimized_hierarchical.npz (Complex spatial hierarchies)")
    print("    - ./data/ann_optimized/datasets/ann_optimized_global_local.npz (Global-local integration)")
    print("    - ./data/ann_optimized/datasets/ann_optimized_multiresolution.npz (Multi-resolution features)")
    print("\n  Example Visualizations:")
    print("    - ./data/ann_optimized/examples/ann_optimized_hierarchical_examples.png")
    print("    - ./data/ann_optimized/examples/ann_optimized_global_local_examples.png")
    print("    - ./data/ann_optimized/examples/ann_optimized_multiresolution_examples.png")
    
    print("\nCompatibility names created in subdirectories:")
    print("  ./data/synthetic/simple_synthetic_5class.npz")
    print("  ./data/synthetic/medium_synthetic_5class.npz")  
    print("  ./data/synthetic/complex_synthetic_5class.npz")
    print("  ./data/spatial/small_spatial_10class.npz (replaced with ANN-optimized dataset)")
    print("  ./data/spatial/medium_spatial_10class.npz (replaced with ANN-optimized dataset)")
    print("  ./data/spatial/large_spatial_10class.npz (replaced with ANN-optimized dataset)")

def plot_confusion_matrix(cm, class_names, title='Confusion Matrix', output_file="./results/confusion_matrix.png", show=True, normalize=False):
    """
    Plot a confusion matrix.
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        title: Title for the plot
        output_file: Path to save the plot
        show: Whether to display the plot
        normalize: Whether to normalize the confusion matrix (percentages instead of counts)
    """
    plt.figure(figsize=(10, 8))
    
    if normalize:
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        cm_display = cm_norm
    else:
        fmt = 'd'
        cm_display = cm
    
    sns.heatmap(cm_display, annot=True, fmt=fmt, cmap='Blues', 
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

def analyze_model_performance(y_true, y_pred, model_name, dataset_name, output_dir="./results/analysis"):
    """
    Generate and save a confusion matrix for model performance analysis.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Name of the model (e.g., "ANN" or "SNN")
        dataset_name: Name of the dataset
        output_dir: Directory to save the analysis results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get number of classes
    n_classes = max(max(y_true), max(y_pred)) + 1
    class_names = [f"Class {i}" for i in range(n_classes)]
    
    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate accuracy, precision, recall, and F1 score
    accuracy = np.trace(cm) / np.sum(cm)
    
    # Save raw confusion matrix as numpy array for future analysis
    np.save(f"{output_dir}/{dataset_name}_{model_name}_confusion_matrix.npy", cm)
    
    # Plot and save normalized confusion matrix
    plot_title = f"{model_name} Confusion Matrix\nDataset: {dataset_name}\nAccuracy: {accuracy:.4f}"
    output_file = f"{output_dir}/{dataset_name}_{model_name}_confusion_matrix.png"
    plot_confusion_matrix(cm, class_names, title=plot_title, output_file=output_file, 
                         show=False, normalize=True)
    
    # Plot and save raw count confusion matrix
    output_file = f"{output_dir}/{dataset_name}_{model_name}_confusion_matrix_counts.png"
    plot_confusion_matrix(cm, class_names, title=f"{model_name} Confusion Matrix (Counts)\nDataset: {dataset_name}", 
                         output_file=output_file, show=False, normalize=False)
    
    # Generate analysis report
    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    
    report = {
        "model": model_name,
        "dataset": dataset_name,
        "overall_accuracy": float(accuracy),
        "class_accuracies": {f"class_{i}": float(acc) for i, acc in enumerate(class_accuracies)},
        "confusion_matrix": cm.tolist(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Save report as JSON
    with open(f"{output_dir}/{dataset_name}_{model_name}_analysis.json", 'w') as f:
        json.dump(report, f, indent=4)
    
    return report

def main():
    parser = argparse.ArgumentParser(description='Run synthetic SNN experiments')
    parser.add_argument('--generate', action='store_true', help='Generate synthetic datasets')
    parser.add_argument('--plot_results', type=str, help='Plot results from JSON file')
    parser.add_argument('--analyze_noise', action='store_true', help='Analyze impact of noise on SNN performance')
    parser.add_argument('--show_plots', action='store_true', help='Show plots interactively (otherwise just save to file)')
    parser.add_argument('--generate_matrix', action='store_true', help='Generate confusion matrix plot')
    parser.add_argument('--analyze_performance', nargs=3, metavar=('TRUE_LABELS', 'PRED_LABELS', 'MODEL_NAME'),
                        help='Analyze model performance using true and predicted labels files and model name')
    parser.add_argument('--dataset_name', type=str, default='dataset',
                        help='Dataset name for performance analysis')
    
    args = parser.parse_args()
    
    if args.generate:
        generate_datasets()
    
    if args.generate_matrix:
        os.makedirs('./results', exist_ok=True)
        # Create a more realistic example confusion matrix
        cm = np.array([
            [45, 3, 1, 2],
            [2, 42, 5, 1],
            [0, 4, 44, 2],
            [1, 2, 3, 44]
        ])
        class_names = ['Class 0', 'Class 1', 'Class 2', 'Class 3']
        plot_confusion_matrix(cm, class_names, 
                             title='Example Confusion Matrix', 
                             output_file='./results/example_confusion_matrix.png', 
                             show=args.show_plots)
        # Also create a normalized version
        plot_confusion_matrix(cm, class_names, 
                             title='Example Normalized Confusion Matrix', 
                             output_file='./results/example_normalized_confusion_matrix.png', 
                             show=args.show_plots,
                             normalize=True)
    
    if args.analyze_performance:
        true_labels_file, pred_labels_file, model_name = args.analyze_performance
        
        # Load the labels
        y_true = np.loadtxt(true_labels_file, dtype=int)
        y_pred = np.loadtxt(pred_labels_file, dtype=int)
        
        # Run analysis
        analyze_model_performance(y_true, y_pred, model_name, args.dataset_name)

if __name__ == "__main__":
    main()