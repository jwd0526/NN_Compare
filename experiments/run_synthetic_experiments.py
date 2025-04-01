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
    Generate temporal pattern datasets with proper progressive difficulty tiers
    Each tier builds on the previous with increasingly subtle distinctions
    """
    # Create the output directory
    os.makedirs('./data', exist_ok=True)
    
    # Initialize the generator
    generator = SyntheticDataGenerator(output_dir='./data')
    
    # ------------------- TIER 1: EASY DISCRIMINATION -------------------
    print("Generating Tier 1 (Easy) temporal patterns dataset...")
    
    # Four fundamentally different pattern types - very easy to distinguish
    t1_classes = []
    
    # Class 0: Regular pattern - evenly spaced spikes
    t1_pattern1, _ = generator.generate_regular_pattern(
        n_neurons=20, length=100, period=15, jitter=0.5, n_samples=125)
    t1_classes.append((t1_pattern1, np.zeros(125, dtype=np.int32)))
    
    # Class 1: Burst pattern - concentrated spikes in a short time window
    t1_pattern2, _ = generator.generate_burst_pattern(
        n_neurons=20, length=100, burst_start=40, burst_width=15, n_spikes=7, jitter=0.5, n_samples=125)
    t1_classes.append((t1_pattern2, np.ones(125, dtype=np.int32)))
    
    # Class 2: Synchronous pattern - simultaneous firing across neurons
    t1_pattern3, _ = generator.generate_sync_pattern(
        n_neurons=20, length=100, sync_times=[20, 50, 80], jitter=0.5, n_samples=125)
    t1_classes.append((t1_pattern3, np.full(125, 2, dtype=np.int32)))
    
    # Class 3: Oscillatory pattern - another distinct pattern type using regular pattern with varying parameters
    t1_pattern4, _ = generator.generate_regular_pattern(
        n_neurons=20, length=100, period=5, jitter=0.5, n_samples=125)
    t1_classes.append((t1_pattern4, np.full(125, 3, dtype=np.int32)))
    
    # Combine patterns
    tier1_patterns = np.concatenate([x[0] for x in t1_classes])
    tier1_labels = np.concatenate([x[1] for x in t1_classes])
    
    # Save to file
    generator.save_dataset(tier1_patterns, tier1_labels, 'temporal_tier1_easy')
    
    # ------------------- TIER 2: MEDIUM DISCRIMINATION -------------------
    print("Generating Tier 2 (Medium) temporal patterns dataset...")
    
    # Four patterns of the same type but with easily distinguishable parameters
    t2_classes = []
    
    # Class 0: Burst pattern early with wide spread
    t2_pattern1, _ = generator.generate_burst_pattern(
        n_neurons=20, length=100, burst_start=20, burst_width=20, n_spikes=7, jitter=1.0, n_samples=125)
    t2_classes.append((t2_pattern1, np.zeros(125, dtype=np.int32)))
    
    # Class 1: Burst pattern early with narrow spread
    t2_pattern2, _ = generator.generate_burst_pattern(
        n_neurons=20, length=100, burst_start=20, burst_width=5, n_spikes=7, jitter=1.0, n_samples=125)
    t2_classes.append((t2_pattern2, np.ones(125, dtype=np.int32)))
    
    # Class 2: Burst pattern late with wide spread
    t2_pattern3, _ = generator.generate_burst_pattern(
        n_neurons=20, length=100, burst_start=70, burst_width=20, n_spikes=7, jitter=1.0, n_samples=125)
    t2_classes.append((t2_pattern3, np.full(125, 2, dtype=np.int32)))
    
    # Class 3: Burst pattern late with narrow spread
    t2_pattern4, _ = generator.generate_burst_pattern(
        n_neurons=20, length=100, burst_start=70, burst_width=5, n_spikes=7, jitter=1.0, n_samples=125)
    t2_classes.append((t2_pattern4, np.full(125, 3, dtype=np.int32)))
    
    # Combine patterns
    tier2_patterns = np.concatenate([x[0] for x in t2_classes])
    tier2_labels = np.concatenate([x[1] for x in t2_classes])
    
    # Save to file
    generator.save_dataset(tier2_patterns, tier2_labels, 'temporal_tier2_medium')
    
    # ------------------- TIER 3: HARD DISCRIMINATION -------------------
    print("Generating Tier 3 (Hard) temporal patterns dataset...")
    
    # Four patterns with similar timing but different distribution characteristics
    t3_classes = []
    
    # All classes have events around time 50, but with different structures
    
    # Class 0: Regular pattern with higher frequency (period=10)
    t3_pattern1, _ = generator.generate_regular_pattern(
        n_neurons=20, length=100, period=10, jitter=1.5, n_samples=125)
    t3_classes.append((t3_pattern1, np.zeros(125, dtype=np.int32)))
    
    # Class 1: Burst pattern in the middle
    t3_pattern2, _ = generator.generate_burst_pattern(
        n_neurons=20, length=100, burst_start=45, burst_width=10, n_spikes=6, jitter=1.5, n_samples=125)
    t3_classes.append((t3_pattern2, np.ones(125, dtype=np.int32)))
    
    # Class 2: Synchronous pattern in the middle
    t3_pattern3, _ = generator.generate_sync_pattern(
        n_neurons=20, length=100, sync_times=[40, 50, 60], jitter=1.5, n_samples=125)
    t3_classes.append((t3_pattern3, np.full(125, 2, dtype=np.int32)))
    
    # Class 3: Irregular pattern (multiple sync points)
    t3_pattern4, _ = generator.generate_sync_pattern(
        n_neurons=20, length=100, sync_times=[35, 45, 55, 65], jitter=1.5, n_samples=125)
    t3_classes.append((t3_pattern4, np.full(125, 3, dtype=np.int32)))
    
    # Combine patterns
    tier3_patterns = np.concatenate([x[0] for x in t3_classes])
    tier3_labels = np.concatenate([x[1] for x in t3_classes])
    
    # Save to file
    generator.save_dataset(tier3_patterns, tier3_labels, 'temporal_tier3_hard')
    
    # ------------------- TIER 4: EXPERT DISCRIMINATION -------------------
    print("Generating Tier 4 (Expert) temporal patterns dataset...")
    
    # Four very similar patterns that differ in subtle statistical ways
    t4_classes = []
    
    # All patterns share the same overall structure but differ in statistical properties
    # Higher jitter makes distinction harder
    
    # Class 0: Regular pattern with period 12 (baseline)
    t4_pattern1, _ = generator.generate_regular_pattern(
        n_neurons=20, length=100, period=12, jitter=2.0, n_samples=125)
    t4_classes.append((t4_pattern1, np.zeros(125, dtype=np.int32)))
    
    # Class 1: Regular pattern with period 11.5 (very close to class 0)
    t4_pattern2, _ = generator.generate_regular_pattern(
        n_neurons=20, length=100, period=11.5, jitter=2.0, n_samples=125)
    t4_classes.append((t4_pattern2, np.ones(125, dtype=np.int32)))
    
    # Class 2: Regular pattern with period 11 (very close to class 1)
    t4_pattern3, _ = generator.generate_regular_pattern(
        n_neurons=20, length=100, period=11, jitter=2.0, n_samples=125)
    t4_classes.append((t4_pattern3, np.full(125, 2, dtype=np.int32)))
    
    # Class 3: Regular pattern with period 10.5 (very close to class 2)
    t4_pattern4, _ = generator.generate_regular_pattern(
        n_neurons=20, length=100, period=10.5, jitter=2.0, n_samples=125)
    t4_classes.append((t4_pattern4, np.full(125, 3, dtype=np.int32)))
    
    # Combine patterns
    tier4_patterns = np.concatenate([x[0] for x in t4_classes])
    tier4_labels = np.concatenate([x[1] for x in t4_classes])
    
    # Save to file
    generator.save_dataset(tier4_patterns, tier4_labels, 'temporal_tier4_expert')

    
    
    return {
        "tier1": (tier1_patterns, tier1_labels),
        "tier2": (tier2_patterns, tier2_labels),
        "tier3": (tier3_patterns, tier3_labels),
        "tier4": (tier4_patterns, tier4_labels)
    }

def generate_tiered_spatial_patterns():
    """
    Generate spatial pattern datasets with multiple difficulty tiers using a consistent approach
    """
    # Create the output directory
    os.makedirs('./data', exist_ok=True)
    
    # Initialize the generator
    generator = SyntheticDataGenerator(output_dir='./data')
    
    # Number of samples per class
    n_samples_per_class = 100  # Define this variable here
    
    # We'll use similar base patterns for all tiers but systematically increase difficulty
    
    # ------------------- TIER 1: EASY DISCRIMINATION -------------------
    print("Generating Tier 1 (Easy) spatial patterns dataset...")
    
    # Generate clearly distinct patterns with no noise
    tier1_patterns, tier1_labels = generator.generate_spatial_patterns(
        width=10, height=10, length=100, n_patterns=4, n_samples=400)
    
    # Save to file
    generator.save_dataset(tier1_patterns, tier1_labels, 'spatial_tier1_easy')
    
    # Visualize examples
    generator.visualize_patterns(tier1_patterns, tier1_labels, n_samples=2, 
                               filename='spatial_tier1_examples.png')
    
    # ------------------- TIER 2: MEDIUM DISCRIMINATION -------------------
    print("Generating Tier 2 (Medium) spatial patterns dataset...")
    
    # Start with the same base patterns but add mild noise
    tier2_patterns = tier1_patterns.copy()
    
    # Add low noise (3%)
    noise_level = 0.03
    mask = np.random.random(tier2_patterns.shape) < noise_level
    tier2_patterns[mask] = 1 - tier2_patterns[mask]  # Flip bits
    
    # Save to file
    generator.save_dataset(tier2_patterns, tier1_labels, 'spatial_tier2_medium')
    
    # Visualize examples
    generator.visualize_patterns(tier2_patterns, tier1_labels, n_samples=2, 
                               filename='spatial_tier2_examples.png')
    
    # ------------------- TIER 3: HARD DISCRIMINATION -------------------
    print("Generating Tier 3 (Hard) spatial patterns dataset...")
    
    # Start with the same base patterns but add more significant noise and perturbations
    tier3_patterns = tier1_patterns.copy()
    
    # Add moderate noise (5%)
    noise_level = 0.05
    mask = np.random.random(tier3_patterns.shape) < noise_level
    tier3_patterns[mask] = 1 - tier3_patterns[mask]  # Flip bits
    
    # Add temporal jitter (shift some activations in time)
    for i in range(len(tier3_patterns)):
        # Find where activations occur
        neuron_idx, time_idx = np.where(tier3_patterns[i] == 1)
        
        # For 20% of activations, shift them by 1-2 timesteps
        shift_mask = np.random.random(len(neuron_idx)) < 0.2
        shift_amount = np.random.randint(-2, 3, size=np.sum(shift_mask))
        
        for j, (n, t, shift) in enumerate(zip(
            neuron_idx[shift_mask], time_idx[shift_mask], shift_amount)):
            # Remove original spike
            tier3_patterns[i, n, t] = 0
            
            # Add shifted spike (if within bounds)
            new_t = t + shift
            if 0 <= new_t < tier3_patterns.shape[2]:
                tier3_patterns[i, n, new_t] = 1
    
    # Save to file
    generator.save_dataset(tier3_patterns, tier1_labels, 'spatial_tier3_hard')
    
    # Visualize examples
    generator.visualize_patterns(tier3_patterns, tier1_labels, n_samples=2, 
                               filename='spatial_tier3_examples.png')
    
    # ------------------- TIER 4: EXPERT DISCRIMINATION -------------------
    print("Generating Tier 4 (Expert) spatial patterns dataset...")
    
    # Create very similar patterns with subtle differences
    
    # Start by creating 4 new base patterns that are variations of each other
    t4_base_patterns = []
    t4_labels = []
    
    # For each of the 4 original pattern types
    for pattern_class in range(4):
        # Get examples of this class
        class_indices = np.where(tier1_labels == pattern_class)[0]
        base_pattern = tier1_patterns[class_indices[0]].copy()
        
        # Create 4 subtle variations (100 samples each)
        for variation in range(4):
            # Copy the base pattern
            var_patterns = np.repeat(base_pattern[np.newaxis, :, :], 25, axis=0)
            
            # Add class-specific alterations
            if variation == 0:
                # Original pattern with noise
                pass
            elif variation == 1:
                # Shift all activations by 3 timesteps
                shifted = np.zeros_like(var_patterns)
                shifted[:, :, 3:] = var_patterns[:, :, :-3]
                var_patterns = shifted
            elif variation == 2:
                # Add small spatial shift (1 neuron)
                shifted = np.zeros_like(var_patterns)
                shifted[:, 1:, :] = var_patterns[:, :-1, :]
                var_patterns = shifted
            elif variation == 3:
                # Slightly reduce the number of spikes
                mask = np.random.random(var_patterns.shape) < 0.10
                var_patterns[mask & (var_patterns == 1)] = 0
            
            # Add high noise level (7%)
            noise_mask = np.random.random(var_patterns.shape) < 0.03
            var_patterns[noise_mask] = 1 - var_patterns[noise_mask]
            
            t4_base_patterns.append(var_patterns)
            t4_labels.append(np.full(25, pattern_class, dtype=np.int32))
    
    # Combine all patterns
    tier4_patterns = np.concatenate(t4_base_patterns)
    tier4_labels = np.concatenate(t4_labels)
    
    # Save to file
    generator.save_dataset(tier4_patterns, tier4_labels, 'spatial_tier4_expert')
    
    # Visualize examples
    generator.visualize_patterns(tier4_patterns, tier4_labels, n_samples=2, 
                               filename='spatial_tier4_examples.png')
    
    return {
        "tier1": (tier1_patterns, tier1_labels),
        "tier2": (tier2_patterns, tier1_labels),
        "tier3": (tier3_patterns, tier1_labels),
        "tier4": (tier4_patterns, tier4_labels)
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
    
    # Generate noise variants for the hard temporal dataset
    print("\n=== Generating Noise Variants for Hard Temporal Dataset ===")
    
    # Initialize the generator
    generator = SyntheticDataGenerator(output_dir='./data')
    
    patterns, labels = temporal_datasets["tier3"]
    
    # Function to add noise to patterns
    def add_noise(patterns, noise_level):
        """Add random noise spikes to patterns"""
        noisy_patterns = patterns.copy()
        
        # Add random spikes (additive noise)
        mask = np.random.random(noisy_patterns.shape) < noise_level
        noisy_patterns[mask] = 1.0
        
        # Delete some existing spikes (subtractive noise)
        existing_spikes = noisy_patterns == 1.0
        delete_mask = np.random.random(noisy_patterns.shape) < noise_level/2
        delete_mask = delete_mask & existing_spikes
        noisy_patterns[delete_mask] = 0.0
        
        return noisy_patterns
    
    # Create datasets with different noise levels
    for noise_level in [0.01, 0.05, 0.1, 0.2]:
        noisy_patterns = add_noise(patterns, noise_level)
        
        # Save to file
        generator.save_dataset(
            noisy_patterns, labels, 
            f'temporal_tier3_noise_{int(noise_level*100)}')
        
        # Visualize examples
        generator.visualize_patterns(
            noisy_patterns, labels, n_samples=2, 
            filename=f'temporal_tier3_noise_{int(noise_level*100)}_examples.png')
    
    print("\nAll datasets generated successfully!")
    print("\nGenerated Datasets:")
    print("  Temporal Patterns:")
    print("    - temporal_tier1_easy.npz (Distinct patterns)")
    print("    - temporal_tier2_medium.npz (Moderately similar patterns)")
    print("    - temporal_tier3_hard.npz (Similar patterns)")
    print("    - temporal_tier4_expert.npz (Extremely similar patterns)")
    print("    - temporal_tier3_noise_1.npz (Hard patterns with 1% noise)")
    print("    - temporal_tier3_noise_5.npz (Hard patterns with 5% noise)")
    print("    - temporal_tier3_noise_10.npz (Hard patterns with 10% noise)")
    print("    - temporal_tier3_noise_20.npz (Hard patterns with 20% noise)")
    print("  Spatial Patterns:")
    print("    - spatial_tier1_easy.npz (Distinct patterns)")
    print("    - spatial_tier2_medium.npz (Moderately similar patterns)")
    print("    - spatial_tier3_hard.npz (Similar patterns)")
    print("    - spatial_tier4_expert.npz (Extremely similar patterns)")

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