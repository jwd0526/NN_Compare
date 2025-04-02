"""
Data utilities for SNN project.

This module contains functions for data loading, processing, and manipulation
of spike trains and related data structures.
"""

import numpy as np
import torch
import random
from scipy.ndimage import filters
from typing import Tuple
from torch.utils.data import Dataset, DataLoader
import os

def generate_rand_pattern(pattern_num: int, synapse_num: int, length: int, 
                          min_spike_num: int, max_spike_num: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create random spike patterns for testing and development.
    
    Each pattern belongs to a different class. Each pattern has multiple spike trains,
    corresponding to different synapses. 1 indicates a spike, 0 indicates no spike.
    
    Args:
        pattern_num: Number of random patterns to generate
        synapse_num: Number of spike trains for each pattern
        length: Length of patterns (time steps)
        min_spike_num: Minimum number of spikes in each spike train
        max_spike_num: Maximum number of spikes in each spike train
            
    Returns:
        Tuple containing:
            x_train: Spike patterns of shape [pattern_idx, synapse_num, time]
            y_train_onehot: One-hot encoded labels of shape [pattern_num, pattern_num]
            y_train_cat: Categorical labels of shape [pattern_number]
    """
    x_train = np.zeros([pattern_num, synapse_num, length], dtype=np.float32)
    y_train_onehot = np.zeros([pattern_num, pattern_num], dtype=np.float32)
    y_train_cat = np.zeros(pattern_num, dtype=np.float32)

    for i in range(pattern_num):
        for j in range(synapse_num):
            spike_number = random.randint(min_spike_num, max_spike_num)
            spike_time = random.sample(range(length), spike_number)
            x_train[i, j, spike_time] = 1
        y_train_onehot[i, i] = 1
        y_train_cat[i] = i

    return x_train, y_train_onehot, y_train_cat

def filter_spike(spike_train: np.ndarray, filter_type: str = 'exp', 
                tau_m: float = 10, tau_s: float = 2.5,
                normalize: bool = True) -> np.ndarray:
    """
    Generate filtered spike train using different filter types.
    
    Args:
        spike_train: 1D array representing a spike train (1=spike, 0=no spike)
        filter_type: Filter type ('exp' or 'dual_exp')
        tau_m: Time constant for membrane potential (used by dual_exp)
        tau_s: Time constant for synaptic current (used by exp and dual_exp)
        normalize: Whether to normalize the filtered output
        
    Returns:
        Filtered spike train of shape [1, len(spike_train)]
    """
    length = len(spike_train)
    eta = tau_m / tau_s
    v_0 = np.power(eta, eta / (eta - 1)) / (eta - 1)

    psp_m = 0
    psp_s = 0
    target_pattern = np.zeros([1, length], dtype=np.float32)
    
    if filter_type == 'dual_exp':
        for i in range(length):
            psp_m = psp_m * np.exp(-1 / tau_m) + spike_train[i]
            psp_s = psp_s * np.exp(-1 / tau_s) + spike_train[i]
            if normalize:
                target_pattern[0, i] = (psp_m - psp_s) * v_0
            else:
                target_pattern[0, i] = (psp_m - psp_s)
    elif filter_type == 'exp':
        for i in range(length):
            psp_s = psp_s * np.exp(-1 / tau_s) + spike_train[i]
            target_pattern[0, i] = psp_s

    return target_pattern

def filter_spike_multiple(spike_trains: np.ndarray, filter_type: str = 'exp', 
                         tau_m: float = 10, tau_s: float = 2.5,
                         normalize: bool = True) -> np.ndarray:
    """
    Create filtered spike trains for a batch of spike trains.
    
    Args:
        spike_trains: 2D array of spike trains [n_trains, time]
        filter_type: Filter type ('exp' or 'dual_exp')
        tau_m: Time constant for membrane potential (used by dual_exp)
        tau_s: Time constant for synaptic current (used by exp and dual_exp)
        normalize: Whether to normalize the filtered output
        
    Returns:
        Filtered spike trains of shape [n_trains, time]
    """
    spike_train_num, time = spike_trains.shape
    filtered_spikes = np.zeros(spike_trains.shape, dtype=np.float32)

    # For each spike train in the batch
    for i in range(spike_train_num):
        filtered_spikes[i] = filter_spike(spike_trains[i], filter_type=filter_type,
                                       tau_m=tau_m, tau_s=tau_s, normalize=normalize)

    return filtered_spikes

def mutate_spike_pattern(template_pattern: np.ndarray, mean: float, sigma: float) -> np.ndarray:
    """
    Create a new spike pattern based on a provided template with Gaussian jitter.
    
    Args:
        template_pattern: 2D array of shape [input_dimension, time]
        mean: Mean of normal distribution for spike time jitter
        sigma: Standard deviation of normal distribution for spike time jitter
        
    Returns:
        Mutated pattern of shape [input_dimension, time]
    """
    input_size, length = template_pattern.shape
    mutated_pattern = np.zeros([input_size, length], dtype=np.float32)
    
    # Find where spikes occur in the template
    input_idx, spike_time = np.where(template_pattern != 0)
    
    # Add random jitter to spike times
    delta_t = np.rint(np.random.normal(mean, sigma, spike_time.shape)).astype(int)
    mutated_spike_time = spike_time + delta_t

    # Handle edge cases: ensure spike times are within valid range
    mutated_spike_time[np.where(mutated_spike_time >= length)] = length - 1
    mutated_spike_time[np.where(mutated_spike_time < 0)] = 0

    # Set spikes in the mutated pattern
    mutated_pattern[input_idx, mutated_spike_time] = 1

    return mutated_pattern

def gaussian_filter_spike_train(spike_train: np.ndarray, sigma: float) -> np.ndarray:
    """
    Create a spike probability distribution over time using Gaussian filtering.
    
    Args:
        spike_train: 1D array representing a spike train (1=spike, 0=no spike)
        sigma: Standard deviation for Gaussian filter
        
    Returns:
        Spike probability over time as a 1D array
    """
    spike_probability = filters.gaussian_filter1d(spike_train, sigma, mode='constant', cval=0)
    return spike_probability.astype(np.float32)

def gaussian_filter_spike_train_batch(spike_train_batch: np.ndarray, sigma: float) -> np.ndarray:
    """
    Apply Gaussian filtering to a batch of spike trains.
    
    Args:
        spike_train_batch: 3D array of spike trains [batch, neuron, time]
        sigma: Standard deviation for Gaussian filter
        
    Returns:
        Filtered spike trains of shape [batch, neuron, time]
    """
    batch_size, spike_train_num, time = spike_train_batch.shape
    filtered_spike_batch = np.zeros(spike_train_batch.shape, dtype=np.float32)

    for i in range(batch_size):
        for j in range(spike_train_num):
            filtered_spike_batch[i, j] = gaussian_filter_spike_train(spike_train_batch[i, j], sigma)

    return filtered_spike_batch

def float_to_spike_train(value: float, spike_train_length: int) -> np.ndarray:
    """
    Convert a floating-point value to a spike train.
    
    Args:
        value: Floating-point value in range [0, 1.0]
        spike_train_length: Length of the resulting spike train
        
    Returns:
        Spike train as a 1D array
    """
    spike_train = np.zeros(spike_train_length)
    spike_number = int(value * spike_train_length)
    ticks = np.linspace(0, spike_train_length, num=spike_number, endpoint=False, dtype=np.int)
    spike_train[ticks] = 1

    return spike_train

def create_poisson_spike_train(rate: float, length: int) -> np.ndarray:
    """
    Generate a Poisson spike train with a given rate.
    
    Args:
        rate: Firing rate in Hz
        length: Length of spike train in time steps
        
    Returns:
        Binary spike train as a 1D array
    """
    spike_train = np.zeros(length, dtype=np.float32)
    rand_vals = np.random.rand(length)
    spike_train[rand_vals < rate] = 1.0
    return spike_train

def create_rate_coded_spike_train(values: np.ndarray, length: int) -> np.ndarray:
    """
    Create rate-coded spike trains from a set of values.
    
    Args:
        values: Array of values to encode as spike rates (should be in range [0, 1])
        length: Length of each spike train in time steps
        
    Returns:
        Spike trains of shape [len(values), length]
    """
    n_values = len(values)
    spike_trains = np.zeros((n_values, length), dtype=np.float32)
    
    for i, value in enumerate(values):
        rand_vals = np.random.rand(length)
        spike_trains[i, rand_vals < value] = 1.0
    
    return spike_trains

def encode_one_hot(labels: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Convert class labels to one-hot encoding.
    
    Args:
        labels: 1D array of class labels
        num_classes: Number of classes
        
    Returns:
        One-hot encoded labels of shape [len(labels), num_classes]
    """
    one_hot = np.zeros((len(labels), num_classes), dtype=np.float32)
    for i, label in enumerate(labels):
        one_hot[i, int(label)] = 1.0
    return one_hot

class SpikeDataset(Dataset):
    """
    Dataset for spike train data with labels.
    """
    
    def __init__(self, spike_data: np.ndarray, labels: np.ndarray, transform=None):
        """
        Initialize the dataset.
        
        Args:
            spike_data: Spike train data of shape [n_samples, n_neurons, time]
            labels: Labels for each sample
            transform: Optional transform to apply to samples
        """
        self.spike_data = spike_data.astype(np.float32)
        self.labels = labels
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.spike_data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample = self.spike_data[idx]
        label = self.labels[idx]
        
        if self.transform:
            sample = self.transform(sample)
        
        return torch.tensor(sample, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

def load_data_from_npz(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load spike data and labels from an .npz file.
    
    Args:
        file_path: Path to the .npz file
        
    Returns:
        Tuple of (spike_data, labels)
    """
    data = np.load(file_path)
    
    # Try common key names in npz files
    if 'patterns' in data and 'labels' in data:
        patterns = data['patterns']
        labels = data['labels']
    elif 'x' in data and 'y' in data:
        patterns = data['x']
        labels = data['y']
    elif 'spike_data' in data and 'labels' in data:
        patterns = data['spike_data']
        labels = data['labels']
    else:
        # If keys aren't standard, just use the first two arrays
        keys = list(data.keys())
        if len(keys) >= 2:
            patterns = data[keys[0]]
            labels = data[keys[1]]
        else:
            raise ValueError(f"Could not identify data and labels in {file_path}")
    
    # Verify shapes match and fix if needed
    if len(patterns) != len(labels):
        print(f"Warning: Shape mismatch in {file_path}")
        print(f"Patterns shape: {patterns.shape}, Labels shape: {labels.shape}")
        
        # Truncate to the minimum length
        min_len = min(len(patterns), len(labels))
        patterns = patterns[:min_len]
        labels = labels[:min_len]
        print(f"Truncated to {min_len} samples for compatibility")
    
    return patterns, labels

def save_dataset(spike_data: np.ndarray, labels: np.ndarray, file_path: str):
    """
    Save spike data and labels to an .npz file.
    
    Args:
        spike_data: Spike data array
        labels: Labels array
        file_path: Path where to save the .npz file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    
    np.savez(file_path, patterns=spike_data, labels=labels)
    print(f"Saved dataset to {file_path}")

def create_spiking_dataloader(data: np.ndarray, labels: np.ndarray, batch_size: int, 
                             shuffle: bool = True, transform=None) -> DataLoader:
    """
    Create a DataLoader for spike train data.
    
    Args:
        data: Spike data array of shape [n_samples, n_neurons, time]
        labels: Labels array
        batch_size: Batch size for DataLoader
        shuffle: Whether to shuffle the data
        transform: Optional transform to apply to samples
        
    Returns:
        PyTorch DataLoader object
    """
    dataset = SpikeDataset(data, labels, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)

def split_dataset(data: np.ndarray, labels: np.ndarray, train_ratio: float = 0.8, 
                 shuffle: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split dataset into training and test sets.
    
    Args:
        data: Data array
        labels: Labels array
        train_ratio: Ratio of data to use for training
        shuffle: Whether to shuffle before splitting
        
    Returns:
        Tuple of (train_data, train_labels, test_data, test_labels)
    """
    n_samples = len(data)
    indices = np.arange(n_samples)
    
    if shuffle:
        np.random.shuffle(indices)
    
    split_idx = int(train_ratio * n_samples)
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    
    train_data = data[train_indices]
    train_labels = labels[train_indices]
    test_data = data[test_indices]
    test_labels = labels[test_indices]
    
    return train_data, train_labels, test_data, test_labels