"""
Visualization utilities for SNN project.

This module contains functions for visualizing spike trains, network activities,
and training/evaluation metrics.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import List, Tuple, Dict, Optional, Union, Any

def plot_raster(spike_mat: np.ndarray, title: str = "Spike Raster Plot", 
                xlabel: str = "Time", ylabel: str = "Neuron ID", 
                figsize: Tuple[int, int] = (10, 6), **kwargs):
    """
    Create a raster plot of spike activities.
    
    Args:
        spike_mat: 2D array of shape [neurons, time] containing spike data (1=spike, 0=no spike)
        title: Title of the plot
        xlabel: Label for x axis
        ylabel: Label for y axis
        figsize: Figure size (width, height) in inches
        **kwargs: Additional arguments to pass to matplotlib plot function
    
    Returns:
        matplotlib Figure object
    """
    fig = plt.figure(figsize=figsize)
    neuron_idx, spike_time = np.where(spike_mat != 0)
    plt.plot(spike_time, neuron_idx, linestyle='None', marker='|', **kwargs)
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    if 'label' in kwargs:
        plt.legend(loc='upper right', fontsize='x-large')
    
    plt.tight_layout()
    return fig

def plot_raster_dot(spike_mat: np.ndarray, title: str = "Spike Raster Plot",
                   xlabel: str = "Time", ylabel: str = "Neuron ID",
                   figsize: Tuple[int, int] = (10, 6)):
    """
    Create a raster plot using dots instead of lines.
    
    Args:
        spike_mat: 2D array of shape [neurons, time] containing spike data
        title: Title of the plot
        xlabel: Label for x axis
        ylabel: Label for y axis
        figsize: Figure size (width, height) in inches
    
    Returns:
        matplotlib Figure object
    """
    h, w = spike_mat.shape
    fig = plt.figure(figsize=figsize)
    point_coordinate = np.where(spike_mat != 0)
    plt.scatter(point_coordinate[1], point_coordinate[0], s=1.5)
    plt.gca().invert_yaxis()
    plt.gca().set_xlim([0, w])
    plt.gca().set_ylim([0, h])
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    return fig

def plot_filtered_spikes(spike_train: np.ndarray, filtered_train: np.ndarray, 
                        figsize: Tuple[int, int] = (12, 6)):
    """
    Plot original spike train and its filtered version side by side.
    
    Args:
        spike_train: Original spike train array of shape [neurons, time]
        filtered_train: Filtered spike train array of same shape
        figsize: Figure size (width, height) in inches
    
    Returns:
        matplotlib Figure object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot original spikes
    neuron_idx, spike_time = np.where(spike_train != 0)
    ax1.plot(spike_time, neuron_idx, linestyle='None', marker='|')
    ax1.set_title('Original Spike Train')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Neuron ID')
    
    # Plot filtered spikes
    if filtered_train.ndim == 2:
        im = ax2.imshow(filtered_train, aspect='auto', cmap='viridis')
        plt.colorbar(im, ax=ax2, label='Activation')
    else:  # Handle 1D case
        ax2.plot(filtered_train)
    
    ax2.set_title('Filtered Spike Train')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Neuron ID')
    
    plt.tight_layout()
    return fig

def plot_neuron_activity(v: np.ndarray, spike_times: np.ndarray, threshold: float = 1.0,
                         neuron_id: int = 0, time_window: Optional[Tuple[int, int]] = None,
                         figsize: Tuple[int, int] = (10, 4)):
    """
    Plot membrane potential and spikes for a single neuron.
    
    Args:
        v: Membrane potential array of shape [time]
        spike_times: Array of time points where spikes occurred
        threshold: Spiking threshold
        neuron_id: ID of the neuron being plotted
        time_window: Optional tuple of (start_time, end_time) to zoom in on a specific period
        figsize: Figure size
    
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    time = np.arange(len(v))
    
    # Apply time window if specified
    if time_window is not None:
        start, end = time_window
        mask = (time >= start) & (time <= end)
        time = time[mask]
        v = v[mask]
        spike_times = spike_times[np.where((spike_times >= start) & (spike_times <= end))]
    
    # Plot membrane potential
    ax.plot(time, v, 'b-', label='Membrane Potential')
    
    # Plot threshold
    ax.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
    
    # Plot spikes
    if len(spike_times) > 0:
        for t in spike_times:
            if time_window is None or (start <= t <= end):
                ax.axvline(x=t, color='g', alpha=0.5)
        
        # Add a single line for the legend
        ax.axvline(x=spike_times[0], color='g', alpha=0.5, label='Spike')
    
    ax.set_title(f'Neuron {neuron_id} Activity')
    ax.set_xlabel('Time')
    ax.set_ylabel('Membrane Potential')
    ax.legend()
    
    plt.tight_layout()
    return fig

def plot_training_curves(train_acc: List[float], test_acc: List[float], 
                        train_loss: Optional[List[float]] = None,
                        test_loss: Optional[List[float]] = None,
                        figsize: Tuple[int, int] = (12, 5)):
    """
    Plot training and test accuracy/loss curves.
    
    Args:
        train_acc: List of training accuracy values per epoch
        test_acc: List of test accuracy values per epoch
        train_loss: Optional list of training loss values per epoch
        test_loss: Optional list of test loss values per epoch
        figsize: Figure size
    
    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(1, 2 if train_loss is not None else 1, figsize=figsize)
    
    if train_loss is None:
        ax = axes
    else:
        ax = axes[0]
    
    epochs = range(1, len(train_acc) + 1)
    
    # Plot accuracy
    ax.plot(epochs, train_acc, 'b-', label='Training Accuracy')
    ax.plot(epochs, test_acc, 'r-', label='Test Accuracy')
    ax.set_title('Training and Test Accuracy')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy')
    ax.legend()
    ax.grid(True)
    
    # Plot loss if provided
    if train_loss is not None:
        ax = axes[1]
        ax.plot(epochs, train_loss, 'b-', label='Training Loss')
        if test_loss is not None:
            ax.plot(epochs, test_loss, 'r-', label='Test Loss')
        ax.set_title('Training and Test Loss')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    return fig

def plot_confusion_matrix(cm: np.ndarray, class_names: Optional[List[str]] = None,
                         normalize: bool = False, title: str = 'Confusion Matrix',
                         figsize: Tuple[int, int] = (8, 6)):
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix array
        class_names: Optional list of class names
        normalize: Whether to normalize the confusion matrix
        title: Title of the plot
        figsize: Figure size
    
    Returns:
        matplotlib Figure object
    """
    import seaborn as sns
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', 
               cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    
    ax.set_title(title)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    
    plt.tight_layout()
    return fig

def visualize_network_weights(model: torch.nn.Module, layer_names: Optional[List[str]] = None,
                             figsize: Tuple[int, int] = (15, 10)):
    """
    Visualize the weights of a neural network.
    
    Args:
        model: PyTorch model
        layer_names: Optional list of layer names to visualize
        figsize: Figure size
    
    Returns:
        matplotlib Figure object
    """
    if layer_names is None:
        # Get all layers with weights
        layer_names = [name for name, param in model.named_parameters() if 'weight' in name]
    
    n_layers = len(layer_names)
    
    if n_layers == 0:
        print("No layers with weights found.")
        return None
    
    # Determine grid layout based on number of layers
    n_rows = int(np.ceil(np.sqrt(n_layers)))
    n_cols = int(np.ceil(n_layers / n_rows))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    if n_layers == 1:
        axes = np.array([axes])
    
    axes = axes.flatten()
    
    for i, name in enumerate(layer_names):
        # Extract weights
        for param_name, param in model.named_parameters():
            if name in param_name and 'weight' in param_name:
                weights = param.data.cpu().numpy()
                
                # Reshape if needed for visualization
                if len(weights.shape) > 2:
                    weights = weights.reshape(weights.shape[0], -1)
                
                # Plot as heatmap
                im = axes[i].imshow(weights, aspect='auto', cmap='viridis')
                axes[i].set_title(f"{name}")
                plt.colorbar(im, ax=axes[i])
                break
    
    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    return fig

def visualize_snn_activity(spike_data: Dict[str, np.ndarray], v_data: Optional[Dict[str, np.ndarray]] = None,
                          layer_names: Optional[List[str]] = None, time_slice: Optional[Tuple[int, int]] = None,
                          figsize: Tuple[int, int] = (15, 10)):
    """
    Visualize SNN layer activities over time.
    
    Args:
        spike_data: Dictionary mapping layer names to spike data of shape [batch, neurons, time]
        v_data: Optional dictionary mapping layer names to membrane potential data
        layer_names: Optional list of layer names to visualize
        time_slice: Optional tuple of (start_time, end_time) to zoom in on a specific period
        figsize: Figure size
    
    Returns:
        matplotlib Figure object
    """
    if layer_names is None:
        layer_names = list(spike_data.keys())
    
    n_layers = len(layer_names)
    
    if n_layers == 0:
        print("No layers found.")
        return None
    
    # Determine plot layout
    n_rows = n_layers
    n_cols = 2 if v_data is not None else 1
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    for i, layer in enumerate(layer_names):
        # Get spike data for this layer and first batch
        if layer in spike_data:
            layer_spikes = spike_data[layer][0]  # First batch
            
            # Apply time window if specified
            if time_slice is not None:
                start, end = time_slice
                layer_spikes = layer_spikes[:, start:end]
            
            # Plot spike raster
            ax = axes[i, 0]
            neuron_idx, spike_time = np.where(layer_spikes != 0)
            ax.plot(spike_time, neuron_idx, linestyle='None', marker='|', color='black')
            ax.set_title(f"{layer} Spikes")
            ax.set_xlabel("Time")
            ax.set_ylabel("Neuron ID")
            
            # Plot membrane potential if available
            if v_data is not None and layer in v_data:
                layer_v = v_data[layer][0]  # First batch
                
                # Apply time window if specified
                if time_slice is not None:
                    layer_v = layer_v[:, start:end]
                
                ax = axes[i, 1]
                im = ax.imshow(layer_v, aspect='auto', cmap='viridis')
                ax.set_title(f"{layer} Membrane Potential")
                ax.set_xlabel("Time")
                ax.set_ylabel("Neuron ID")
                plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    return fig